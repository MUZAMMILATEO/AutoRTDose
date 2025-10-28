import torch
import numpy as np
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import sys

# Add parent directory to path for imports
# This line assumes the script is run from a sub-directory, which is typical for OpenKBP projects
sys.path.append(os.path.abspath('.'))

# --- CONFIGURATION ---
INPUT_SIZE = 128    # Assuming D=H=W=128
# File naming conventions based on typical OpenKBP data structure
PRED_FILE_SUFFIX = '_predicted_dose_1ch.npy'
GT_DOSE_FILE_SUFFIX = '_dose_1ch.npy'
# Use the new 10-channel structure file for masks
STRUCT_MASK_FILE_SUFFIX = '_struct_10ch.npy'

# Define ROI labels corresponding to the 10 channels (CORRECTED)
ROI_LABELS = [
    "PTV70", "PTV63", "PTV56",
    "Brainstem", "SpinalCord", "Mandible", "Larynx",
    "RightParotid", "LeftParotid", "Esophagus"
]
NUM_ROIS = len(ROI_LABELS)

# Define clinical subsets for metric calculation (CORRECTED)
OAR_LABELS = ["Brainstem", "SpinalCord", "Mandible", "Larynx", "RightParotid", "LeftParotid", "Esophagus"]
TARGET_LABELS = ["PTV70", "PTV63", "PTV56"]

# --- UTILITY FUNCTIONS ---

def parse_args():
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="3D Conv Transformer Dose Evaluation")
    parser.add_argument('input_path', type=str, 
                        help='Path to a single patient directory (e.g., pt_1) OR a directory containing patient subfolders.')
    return parser.parse_args()


def load_patient_data_for_eval(patient_path, patient_id):
    """Loads GT dose, Predicted dose, and ROI masks for evaluation."""
    
    # 1. Load Predicted Dose
    pred_file = os.path.join(patient_path, f'{patient_id}{PRED_FILE_SUFFIX}')
    pred_dose = np.load(pred_file).astype(np.float32)

    # 2. Load Ground Truth Dose
    gt_dose_file = os.path.join(patient_path, f'{patient_id}{GT_DOSE_FILE_SUFFIX}')
    gt_dose = np.load(gt_dose_file).astype(np.float32)

    # 3. Load Structure Masks (from the 10-channel structure file)
    struct_mask_file = os.path.join(patient_path, f'{patient_id}{STRUCT_MASK_FILE_SUFFIX}')
    struct_data = np.load(struct_mask_file).astype(np.float32)
    
    # Check shape: Expected (D, H, W, 10). Axes are D, H, W, C
    if struct_data.ndim != 4 or struct_data.shape[-1] != NUM_ROIS:
        raise ValueError(f"Structure mask shape {struct_data.shape} is wrong. Expected (D, H, W, {NUM_ROIS}).")

    # struct_data now contains all 10 masks (D, H, W, C)
    return gt_dose, pred_dose, struct_data


def calculate_metrics(gt_dose, pred_dose):
    """Calculates quantitative dose comparison metrics (MAE and RMSE)."""
    
    # Flatten arrays for sklearn metrics
    gt_flat = gt_dose.flatten()
    pred_flat = pred_dose.flatten()

    mae = mean_absolute_error(gt_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))

    return {'MAE (Gy)': mae, 'RMSE (Gy)': rmse}


def calculate_dvh_constraints(dose_map, roi_mask, roi_label):
    """
    Calculates key DVH constraints using percentile doses (Dose covering X% of volume).
    :param dose_map: The 3D dose distribution.
    :param roi_mask: The 3D mask for the specific structure.
    :param roi_label: Name of the ROI (for context, currently unused).
    :returns: A dictionary of metric values (e.g., {'D_mean': 3.5, 'D_95': 54.0}).
    """
    roi_dose = dose_map[roi_mask > 0]
    metrics = {}
    
    if len(roi_dose) == 0:
        # Return zeros if the structure is empty (e.g., missing parotid gland)
        return {'D_mean': 0.0, 'D_99': 0.0, 'D_95': 0.0, 'D_1': 0.0}

    # Mean Dose (Common to all)
    metrics['D_mean'] = roi_dose.mean()
    
    # D_X% is the dose covering X percent of the volume.
    # To find D_99 (Dose covering 99%), we look for the 1st percentile dose.
    # To find D_95 (Dose covering 95%), we look for the 5th percentile dose.
    # To find D_1 (Dose covering 1%), we look for the 99th percentile dose.
    
    metrics['D_99'] = np.percentile(roi_dose, 1) # Near-minimum dose for coverage
    metrics['D_95'] = np.percentile(roi_dose, 5) # Coverage dose
    metrics['D_1'] = np.percentile(roi_dose, 99) # Near-maximum dose (proxy for max)

    return metrics


def calculate_dvh(dose_map, roi_mask):
    """
    Calculates the Dose-Volume Histogram (DVH) for a given dose map and ROI mask.
    Returns: dose bins (Gy) and cumulative volume (%)
    """
    # 1. Select only voxels within the ROI
    # We flatten the dose array and select doses where the mask is > 0
    roi_dose = dose_map[roi_mask > 0] 
    
    if len(roi_dose) == 0:
        # If ROI is empty, return a safe array for plotting (e.g., max dose 100 Gy)
        return np.array([0, 100]), np.array([0, 0])
    
    # 2. Define dose bins (e.g., from 0 up to max dose, in steps of 1 Gy)
    # Use the max of the whole dose map for consistency, but ensure it's at least 1
    max_dose_val = np.ceil(np.max(dose_map)) if np.max(dose_map) > 0 else 1
    dose_bins = np.arange(0, max_dose_val + 1, 1) # Bins from 0 to max_dose_val

    # Ensure there is at least one bin if max dose is very low
    if len(dose_bins) < 2:
        dose_bins = np.arange(0, 5, 1) # Fallback to 0, 1, 2, 3, 4

    
    # 3. Compute histogram (counts per dose bin)
    # The counts array is indexed by dose-bin - 1
    counts, bins = np.histogram(roi_dose, bins=dose_bins)
    
    # 4. Convert counts to cumulative volume (percent)
    # Start from the highest dose bin and accumulate
    cumulative_volume = np.cumsum(counts[::-1])[::-1]
    
    # 5. Normalize volume by total ROI volume
    total_roi_voxels = len(roi_dose)
    # DVH usually plots percent volume (y-axis) vs dose (x-axis)
    cumulative_volume_percent = (cumulative_volume / total_roi_voxels) * 100
    
    # We return the dose value corresponding to the start of the bin (bins[:-1])
    return bins[:-1], cumulative_volume_percent


def plot_dvh(gt_dose, pred_dose, struct_data, patient_id):
    """Computes and plots the GT and Predicted DVHs for all 10 ROIs."""
    
    # Global max dose for consistent x-axis scaling
    max_dose = max(np.max(gt_dose), np.max(pred_dose)) * 1.05
    
    # Plot 10 structures on a 2x5 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten() # Flatten 2x5 array to easily iterate over 10 axes
    fig.suptitle(f'Dose-Volume Histograms (DVH) Comparison for Patient: {patient_id}', fontsize=20)

    # Loop through all 10 channels/structures
    for i in range(NUM_ROIS):
        roi_mask = struct_data[..., i] # Select the i-th structure mask
        roi_label = ROI_LABELS[i]
        ax = axes[i]
        
        # Calculate DVHs
        dose_gt, vol_gt = calculate_dvh(gt_dose, roi_mask)
        dose_pred, vol_pred = calculate_dvh(pred_dose, roi_mask)

        # Plot GT
        ax.plot(dose_gt, vol_gt, 'r-', linewidth=2, label=f'GT Dose ({roi_label})')
        # Plot Prediction
        ax.plot(dose_pred, vol_pred, 'b--', linewidth=2, label=f'Pred Dose ({roi_label})')

        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title(roi_label, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, max_dose)
        ax.set_ylim(0, 105)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    plt.show()


def visualize_doses(gt_dose, pred_dose, patient_id):
    """
    Displays side-by-side axial, coronal, and sagittal views of GT and Predicted doses.
    Input dose_map shape: (D, H, W)
    """
    D, H, W = gt_dose.shape
    
    # Calculate middle slices
    center_D, center_H, center_W = D // 2, H // 2, W // 2
    
    # Global max dose for consistent colormap scaling
    vmax = max(np.max(gt_dose), np.max(pred_dose)) * 1.1

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dose Map Comparison for Patient: {patient_id}', fontsize=20, y=1.02)

    titles = [f"Axial (Z={center_D})", f"Coronal (Y={center_H})", f"Sagittal (X={center_W})"]
    doses = [gt_dose, pred_dose]
    labels = ["Ground Truth", "Prediction"]

    for row, dose_map in enumerate(doses):
        dose_label = labels[row]
        
        # Slices
        axial_slice = dose_map[center_D, :, :]  # H x W
        coronal_slice = dose_map[:, center_H, :] # D x W
        sagittal_slice = dose_map[:, :, center_W] # D x H
        
        slices = [axial_slice, coronal_slice, sagittal_slice]
        
        for col, slc in enumerate(slices):
            ax = axes[row, col]
            # Changed colormap from 'viridis' to 'jet' which is common for dose maps
            im = ax.imshow(slc, cmap='jet', origin='lower', vmax=vmax) 
            ax.set_title(f"{dose_label} - {titles[col]}", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add a colorbar for the prediction row
            if row == 1 and col == 2:
                cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5]) # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label='Dose Value (Gy)')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for suptitle and colorbar
    plt.show()


def evaluate_single_patient(patient_path, patient_id, detailed_visuals=False):
    """Loads data, calculates metrics, and optionally plots visualizations."""
    try:
        # load_patient_data_for_eval now returns struct_data (all masks)
        gt_dose, pred_dose, struct_data = load_patient_data_for_eval(patient_path, patient_id)
        
        # Calculate overall metrics (MAE, RMSE)
        metrics = calculate_metrics(gt_dose, pred_dose)
        metrics['PatientID'] = patient_id

        # --- Calculate DVH Constraint Differences ---
        for i, roi_label in enumerate(ROI_LABELS):
            # No 'Body' to skip in the new list, but ensure we only calculate for OARs/TARGETS
            
            roi_mask = struct_data[..., i]
            
            # Check if ROI exists (total volume > 0)
            if np.sum(roi_mask) == 0:
                # If ROI is empty, report zero difference for relevant metrics
                if roi_label in OAR_LABELS:
                    metrics[f'|GT-Pred| D_mean ({roi_label})'] = 0.0
                    metrics[f'|GT-Pred| D_1 ({roi_label})'] = 0.0
                elif roi_label in TARGET_LABELS:
                    metrics[f'|GT-Pred| D_99 ({roi_label})'] = 0.0
                    metrics[f'|GT-Pred| D_95 ({roi_label})'] = 0.0
                continue
                
            gt_constraints = calculate_dvh_constraints(gt_dose, roi_mask, roi_label)
            pred_constraints = calculate_dvh_constraints(pred_dose, roi_mask, roi_label)
            
            # Select relevant metrics for OARs vs Targets based on common clinical goals
            if roi_label in OAR_LABELS:
                # OARs (Organs at Risk): Focus on mean dose and near-maximum dose (D_1)
                metrics[f'|GT-Pred| D_mean ({roi_label})'] = abs(gt_constraints['D_mean'] - pred_constraints['D_mean'])
                metrics[f'|GT-Pred| D_1 ({roi_label})'] = abs(gt_constraints['D_1'] - pred_constraints['D_1'])
            elif roi_label in TARGET_LABELS:
                # Targets (Planning Target Volumes): Focus on coverage/near-minimum dose (D_99, D_95)
                metrics[f'|GT-Pred| D_99 ({roi_label})'] = abs(gt_constraints['D_99'] - pred_constraints['D_99'])
                metrics[f'|GT-Pred| D_95 ({roi_label})'] = abs(gt_constraints['D_95'] - pred_constraints['D_95'])

        # Display visualizations if requested (only for single patient mode)
        if detailed_visuals:
            print(f"\n--- Visualizing {patient_id} ---")
            visualize_doses(gt_dose, pred_dose, patient_id)
            # Pass all structure data to plot_dvh
            plot_dvh(gt_dose, pred_dose, struct_data, patient_id)
            
        return metrics

    except FileNotFoundError as e:
        print(f"Skipping {patient_id}: Missing file for evaluation. Expected files like '{patient_id}{PRED_FILE_SUFFIX}' in '{patient_path}'. Error: {e}")
        return None
    except ValueError as e:
        print(f"Skipping {patient_id}: Data loading error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {patient_id}: {e}")
        return None


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    all_metrics = []
    
    # --- 1. Determine Evaluation Mode ---
    
    patient_id_guess = os.path.basename(input_path)
    
    # Check for the predicted dose file name based on the directory name (the reliable method from predict.py)
    pred_file_check = os.path.join(input_path, f'{patient_id_guess}{PRED_FILE_SUFFIX}')
    
    # A single patient directory is a directory that contains the predicted dose file 
    is_single_patient = os.path.isdir(input_path) and os.path.exists(pred_file_check)
    
    
    if is_single_patient:
        # Case A: Single patient directory provided (run detailed mode)
        print(f"--- Running DETAILED evaluation for single patient: {patient_id_guess} ---")
        metrics = evaluate_single_patient(input_path, patient_id_guess, detailed_visuals=True)
        if metrics:
            all_metrics.append(metrics)
            
    elif os.path.isdir(input_path):
        # Case B: Directory containing multiple patient subfolders (run batch mode)
        print(f"--- Running BATCH evaluation for directory: {input_path} ---")
        
        # Find all patient subfolders (e.g., pt_*, patient_*)
        patient_paths = sorted([d for d in glob(os.path.join(input_path, '*')) if os.path.isdir(d)])
        
        if not patient_paths:
            print(f"Error: No patient subfolders found in {input_path}. Please check your path or ensure files like '{{patient_id}}{PRED_FILE_SUFFIX}' exist directly inside the folder for single-patient mode.")
            return
            
        print(f"Found {len(patient_paths)} patient folders to process.")
        
        for patient_path in patient_paths:
            patient_id = os.path.basename(patient_path)
            # Check if the subdirectory contains the required prediction file
            if os.path.exists(os.path.join(patient_path, f'{patient_id}{PRED_FILE_SUFFIX}')):
                metrics = evaluate_single_patient(patient_path, patient_id, detailed_visuals=False)
                if metrics:
                    all_metrics.append(metrics)
            # else: skip this folder as it doesn't contain the prediction file
                
    else:
        print(f"Error: Input path is not a valid directory or is not a recognized patient folder structure: {input_path}")
        return

    # --- 2. Print Summary Tables ---
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        
        # Round all numeric columns to 3 decimal places for cleaner display
        results_df = results_df.round(3)

        # 1. Define Column Groups for Clean Printing
        global_cols_full = ['PatientID', 'MAE (Gy)', 'RMSE (Gy)']
        global_metrics = ['MAE (Gy)', 'RMSE (Gy)']

        # OARs are not PTVs and not global
        oar_cols = [col for col in results_df.columns if col not in global_cols_full and any(oar in col for oar in OAR_LABELS) and '|GT-Pred|' in col]
        # PTVs are not global and contain 'PTV'
        ptv_cols = [col for col in results_df.columns if any(ptv in col for ptv in TARGET_LABELS) and '|GT-Pred|' in col]
        # All DVH difference columns
        dvh_diff_cols = oar_cols + ptv_cols


        print("\n" + "="*80)
        print("QUANTITATIVE EVALUATION SUMMARY (ALL PATIENTS)")
        print("="*80)

        # --- A. Global Metrics Table (Per Patient) ---
        print("\n--- A. GLOBAL DOSE ACCURACY (MAE/RMSE) - PER PATIENT ---")
        print(results_df[global_cols_full].to_string(index=False))

        # --- B. OAR Metrics Table (Per Patient) ---
        if oar_cols:
            print("\n--- B. OAR CONSTRAINTS (Absolute Difference in Gy) - PER PATIENT ---")
            
            # Create cleaner, shorter headers for display
            oar_display_df = results_df[['PatientID'] + oar_cols].copy()
            new_oar_cols = {}
            for col in oar_cols:
                # Converts "|GT-Pred| D_mean (SpinalCord)" to "SpinalCord (D_mean)"
                parts = col.split('(')
                metric_name = parts[0].strip().replace('|GT-Pred| ', '')
                roi_name = parts[1].replace(')', '')
                new_oar_cols[col] = f"{roi_name} ({metric_name})"
            
            oar_display_df = oar_display_df.rename(columns=new_oar_cols)
            print(oar_display_df.to_string(index=False))

        # --- C. PTV Metrics Table (Per Patient) ---
        if ptv_cols:
            print("\n--- C. PTV COVERAGE (Absolute Difference in Gy) - PER PATIENT ---")
            
            # Create cleaner, shorter headers for display
            ptv_display_df = results_df[['PatientID'] + ptv_cols].copy()
            new_ptv_cols = {}
            for col in ptv_cols:
                # Converts "|GT-Pred| D_99 (PTV70)" to "PTV70 (D_99)"
                parts = col.split('(')
                metric_name = parts[0].strip().replace('|GT-Pred| ', '')
                roi_name = parts[1].replace(')', '')
                new_ptv_cols[col] = f"{roi_name} ({metric_name})"
                
            ptv_display_df = ptv_display_df.rename(columns=new_ptv_cols)
            print(ptv_display_df.to_string(index=False))

        # --- 3. Print Aggregated Scores (Mean +/- STD) ---
        if len(all_metrics) > 1:
            
            # Identify all numerical metric columns
            metric_cols = [col for col in results_df.columns if col != 'PatientID']

            # Calculate mean and standard deviation for all metric columns
            summary_mean = results_df[metric_cols].mean().to_frame(name='Mean (Gy)')
            summary_std = results_df[metric_cols].std().to_frame(name='STD (Gy)')
            
            # Combine mean and std into one summary DataFrame
            summary_df = pd.concat([summary_mean, summary_std], axis=1).round(3)
            
            print("\n" + "="*80)
            print("AGGREGATED PERFORMANCE SCORES (Mean $\pm$ STD)")
            print("="*80)
            
            # D. Global Metrics Aggregated
            global_summary = summary_df.loc[global_metrics]
            print("\n--- D. AGGREGATED GLOBAL DOSE ACCURACY (MAE/RMSE) ---")
            print(global_summary.to_string())

            # E. OAR Constraint Difference Metrics Aggregated
            oar_summary = summary_df.loc[oar_cols]
            if not oar_summary.empty:
                print("\n--- E. AGGREGATED OAR CONSTRAINTS (Absolute Difference in Gy) ---")
                
                # Reformat index for cleaner display (like the detailed table)
                new_oar_index = {}
                for index_col in oar_summary.index:
                    parts = index_col.split('(')
                    metric_name = parts[0].strip().replace('|GT-Pred| ', '')
                    roi_name = parts[1].replace(')', '')
                    new_oar_index[index_col] = f"{roi_name} ({metric_name})"
                
                oar_summary = oar_summary.rename(index=new_oar_index)
                print(oar_summary.to_string())
                
            # F. PTV Constraint Difference Metrics Aggregated
            ptv_summary = summary_df.loc[ptv_cols]
            if not ptv_summary.empty:
                print("\n--- F. AGGREGATED PTV COVERAGE (Absolute Difference in Gy) ---")

                # Reformat index for cleaner display (like the detailed table)
                new_ptv_index = {}
                for index_col in ptv_summary.index:
                    parts = index_col.split('(')
                    metric_name = parts[0].strip().replace('|GT-Pred| ', '')
                    roi_name = parts[1].replace(')', '')
                    new_ptv_index[index_col] = f"{roi_name} ({metric_name})"
                    
                ptv_summary = ptv_summary.rename(index=new_ptv_index)
                print(ptv_summary.to_string())

            # G. Overall Average DVH Score (as a single number, including STD)
            if dvh_diff_cols:
                all_dvh_diffs = results_df[dvh_diff_cols] 
                # Flatten all DVH constraint differences into a single array for overall mean/std
                mean_dvh_score = all_dvh_diffs.values.flatten().mean()
                std_dvh_score = all_dvh_diffs.values.flatten().std()

                print("\n" + "-"*80)
                print(f"TOTAL AVERAGE DVH CONSTRAINT ERROR (All ROIs): {mean_dvh_score:.3f} $\\pm$ {std_dvh_score:.3f} Gy")
                print("-" * 80)
    
    else:
        print("\n" + "="*80)
        print("EVALUATION FAILED: NO PATIENTS PROCESSED SUCCESSFULLY")
        print("Please check the console output above for 'Skipping...' messages to troubleshoot file paths.")
        print("="*80)


if __name__ == '__main__':
    main()
