import numpy as np
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

# --- CONFIGURATION ---
INPUT_SIZE = 128
PRED_FILE_SUFFIX = '_predicted_dose_1ch.npy'
GT_DOSE_FILE_SUFFIX = '_dose_1ch.npy'
STRUCT_MASK_FILE_SUFFIX = '_struct_10ch.npy'

# Define ROI labels corresponding to the 10 channels
ROI_LABELS = [
    "PTV70", "PTV63", "PTV56",
    "Brainstem", "SpinalCord", "Mandible", "Larynx",
    "RightParotid", "LeftParotid", "Esophagus"
]
NUM_ROIS = len(ROI_LABELS)

OAR_LABELS = ["Brainstem", "SpinalCord", "Mandible", "Larynx", "RightParotid", "LeftParotid", "Esophagus"]
TARGET_LABELS = ["PTV70", "PTV63", "PTV56"]
MAX_DOSE_GY = 80.0 # Upper limit for DVH calculation

# --- OPENKBP OFFICIAL SCORING METRICS AND WEIGHTS ---
# This is the full set of 24 metrics and their clinical weights (W_i)
OPENKBP_CONSTRAINTS = {
    # Key: (ROI_LABEL, Metric_Type (D/V), Value)
    ("PTV70", "D", 95): {"Weight": 1.0, "Hard_Constraint": True},
    ("PTV70", "D", 99): {"Weight": 0.5, "Hard_Constraint": False},
    ("PTV63", "D", 95): {"Weight": 1.0, "Hard_Constraint": True},
    ("PTV63", "D", 99): {"Weight": 0.5, "Hard_Constraint": False},
    ("PTV56", "D", 95): {"Weight": 1.0, "Hard_Constraint": True},
    ("PTV56", "D", 99): {"Weight": 0.5, "Hard_Constraint": False},
    
    # OAR Constraints (V_Y metrics are in %)
    ("Brainstem", "D", 1):   {"Weight": 1.0, "Hard_Constraint": True},
    ("Brainstem", "D", 0.1): {"Weight": 0.5, "Hard_Constraint": False}, # D_0.1cc is approximated by D_0.1% for voxel data
    ("SpinalCord", "D", 1):  {"Weight": 1.0, "Hard_Constraint": True},
    ("SpinalCord", "D", 0.1):{"Weight": 0.5, "Hard_Constraint": False},
    
    ("Mandible", "D", "mean"): {"Weight": 0.5, "Hard_Constraint": False},
    ("Mandible", "V", 40):     {"Weight": 0.5, "Hard_Constraint": False}, # V_40Gy
    
    ("Larynx", "D", "mean"):   {"Weight": 0.5, "Hard_Constraint": False},
    ("Larynx", "V", 40):       {"Weight": 0.5, "Hard_Constraint": False}, # V_40Gy
    
    ("RightParotid", "D", "mean"): {"Weight": 1.0, "Hard_Constraint": True},
    ("RightParotid", "V", 20):     {"Weight": 0.5, "Hard_Constraint": False}, # V_20Gy
    ("RightParotid", "V", 30):     {"Weight": 0.5, "Hard_Constraint": False}, # V_30Gy
    
    ("LeftParotid", "D", "mean"):  {"Weight": 1.0, "Hard_Constraint": True},
    ("LeftParotid", "V", 20):      {"Weight": 0.5, "Hard_Constraint": False},
    ("LeftParotid", "V", 30):      {"Weight": 0.5, "Hard_Constraint": False},
    
    ("Esophagus", "D", "mean"):  {"Weight": 0.5, "Hard_Constraint": False},
    ("Esophagus", "V", 35):      {"Weight": 0.5, "Hard_Constraint": False}, # V_35Gy
    ("Esophagus", "V", 50):      {"Weight": 0.5, "Hard_Constraint": False}, # V_50Gy
}
# Total weight for normalization (Sum of all 24 weights)
TOTAL_WEIGHT = sum(item["Weight"] for item in OPENKBP_CONSTRAINTS.values())
# Total number of Hard Constraints for normalization
NUM_HARD_CONSTRAINTS = sum(1 for item in OPENKBP_CONSTRAINTS.values() if item["Hard_Constraint"])


# --- UTILITY FUNCTIONS ---

def parse_args():
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="3D Conv Transformer Dose Evaluation")
    parser.add_argument('input_path', type=str, 
                        help='Path to a single patient directory (e.g., pt_1) OR a directory containing patient subfolders.')
    return parser.parse_args()


def load_patient_data_for_eval(patient_path, patient_id):
    """Loads GT dose, Predicted dose, and ROI masks for evaluation."""
    pred_file = os.path.join(patient_path, f'{patient_id}{PRED_FILE_SUFFIX}')
    pred_dose = np.load(pred_file).astype(np.float32)
    gt_dose_file = os.path.join(patient_path, f'{patient_id}{GT_DOSE_FILE_SUFFIX}')
    gt_dose = np.load(gt_dose_file).astype(np.float32)
    struct_mask_file = os.path.join(patient_path, f'{patient_id}{STRUCT_MASK_FILE_SUFFIX}')
    struct_data = np.load(struct_mask_file).astype(np.float32)
    
    if struct_data.ndim != 4 or struct_data.shape[-1] != NUM_ROIS:
        raise ValueError(f"Structure mask shape {struct_data.shape} is wrong. Expected (D, H, W, {NUM_ROIS}).")

    return gt_dose, pred_dose, struct_data


def calculate_metrics(gt_dose, pred_dose):
    """Calculates quantitative dose comparison metrics (MAE and RMSE)."""
    gt_flat = gt_dose.flatten()
    pred_flat = pred_dose.flatten()
    mae = mean_absolute_error(gt_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
    return {'MAE (Gy)': mae, 'RMSE (Gy)': rmse}


def calculate_dvh(dose_map, roi_mask, bin_size=1.0):
    """
    Calculates the Dose-Volume Histogram (DVH) for a given dose map and ROI mask.
    Returns: dose bins (Gy) and cumulative volume (%)
    """
    roi_dose = dose_map[roi_mask > 0] 
    
    if len(roi_dose) == 0:
        # Return safe array for plotting
        return np.array([0, MAX_DOSE_GY]), np.array([100, 0])
    
    # Define dose bins from 0 up to MAX_DOSE_GY, in steps of bin_size
    dose_bins = np.arange(0, MAX_DOSE_GY + bin_size, bin_size)
    
    # Compute histogram
    counts, bins = np.histogram(roi_dose, bins=dose_bins)
    
    # Convert counts to cumulative volume (percent)
    cumulative_volume = np.cumsum(counts[::-1])[::-1]
    total_roi_voxels = len(roi_dose)
    cumulative_volume_percent = (cumulative_volume / total_roi_voxels) * 100
    
    # We return the dose value corresponding to the start of the bin (bins[:-1])
    return bins[:-1], cumulative_volume_percent


def find_volume_percent_at_dose(dose_bins, cumulative_volume_percent, dose_gy):
    """
    Interpolates the DVH curve to find the Volume (%) that receives at least 'dose_gy'.
    """
    if dose_gy <= dose_bins[0]:
        return cumulative_volume_percent[0]
    if dose_gy >= dose_bins[-1]:
        return 0.0
    
    # Find the index where dose_bins first exceeds dose_gy
    idx = np.searchsorted(dose_bins, dose_gy, side='left')
    
    # Simple linear interpolation between points idx-1 and idx
    x1, y1 = dose_bins[idx-1], cumulative_volume_percent[idx-1]
    x2, y2 = dose_bins[idx], cumulative_volume_percent[idx]
    
    if x2 == x1:
        return y1
    
    volume_at_dose = y1 + (y2 - y1) * (dose_gy - x1) / (x2 - x1)
    
    return np.clip(volume_at_dose, 0.0, 100.0)


def calculate_dvh_metrics_for_roi(dose_map, roi_mask, roi_label):
    """
    Calculates ALL necessary D_X and V_Y metrics for a single ROI based on OpenKBP needs.
    """
    roi_dose = dose_map[roi_mask > 0]
    metrics = {}
    
    if len(roi_dose) == 0:
        return metrics # Returns empty dict if ROI is zero volume

    # --- 1. Dose Metrics (D_X) ---
    metrics['D_mean'] = roi_dose.mean()
    # D_X% is the dose covering X percent of the volume (Xth percentile from the bottom)
    # D_99 is the 1st percentile. D_1 is the 99th percentile.
    metrics['D_99'] = np.percentile(roi_dose, 1)
    metrics['D_95'] = np.percentile(roi_dose, 5)
    metrics['D_1'] = np.percentile(roi_dose, 99)
    # D_0.1 (Approximation of D_0.1cc)
    metrics['D_0.1'] = np.percentile(roi_dose, 99.9)

    # --- 2. Volume Metrics (V_Y) - Requires high-res DVH ---
    
    # Calculate high-resolution DVH (0.1 Gy steps) for accurate V_Y interpolation
    dose_bins, cumulative_volume_percent = calculate_dvh(dose_map, roi_mask, bin_size=0.1)
    
    # Required V_Y metrics based on OPENKBP_CONSTRAINTS
    for _, metric_type, val in OPENKBP_CONSTRAINTS:
        if metric_type == "V":
            metric_key = f'V_{int(val)}'
            if metric_key not in metrics:
                # V_YGy: Volume receiving at least Y Gy
                metrics[metric_key] = find_volume_percent_at_dose(dose_bins, cumulative_volume_percent, val)
    
    return metrics


def calculate_openkbp_scores(gt_metrics, pred_metrics, patient_id):
    """
    Calculates the Hard Constraint Score (HCS) and Clinical Score (CS) 
    based on the official OpenKBP definition.
    :param gt_metrics: Dict of GT DVH metrics keyed by (ROI, metric_key)
    :param pred_metrics: Dict of Pred DVH metrics keyed by (ROI, metric_key)
    :returns: {'CS': clinical_score, 'HCS': hard_constraint_score, 'Constraint_Errors': dict}
    """
    total_weighted_error = 0.0
    hard_constraint_error_sum = 0.0
    constraint_errors = {}
    
    for (roi_label, metric_type, val), properties in OPENKBP_CONSTRAINTS.items():
        weight = properties["Weight"]
        is_hard = properties["Hard_Constraint"]
        
        # Format the internal metric key
        if metric_type == "D" and val != "mean":
            # D_95, D_1, D_0.1
            metric_key = f'D_{str(val).replace(".", "")}'
        elif metric_type == "D" and val == "mean":
            metric_key = 'D_mean'
        elif metric_type == "V":
            # V_40Gy, V_20Gy
            metric_key = f'V_{int(val)}'
        else:
            continue # Skip unknown metrics

        full_key = (roi_label, metric_key)
        
        # Handle cases where the ROI might not exist in the patient (metric key missing)
        gt_value = gt_metrics.get(full_key, 0.0)
        pred_value = pred_metrics.get(full_key, 0.0)

        # The error is the absolute difference in Gy (for D-metrics) or % (for V-metrics)
        # Note: OpenKBP treats V_Y as an error in % which is equivalent to Gy for planning studies.
        error = abs(gt_value - pred_value)
        
        # Store the individual error
        constraint_errors[f'|GT-Pred| {metric_key} ({roi_label})'] = error

        # Accumulate for Clinical Score (Weighted Average)
        total_weighted_error += error * weight
        
        # Accumulate for Hard Constraint Score (Simple Average)
        if is_hard:
            hard_constraint_error_sum += error

    # Normalize the scores
    clinical_score = total_weighted_error / TOTAL_WEIGHT
    hard_constraint_score = hard_constraint_error_sum / NUM_HARD_CONSTRAINTS
    
    return {
        'Clinical Score (CS)': clinical_score, 
        'Hard Constraint Score (HCS)': hard_constraint_score, 
        'Constraint_Errors': constraint_errors
    }


def plot_dvh(gt_dose, pred_dose, struct_data, patient_id):
    """Computes and plots the GT and Predicted DVHs for all 10 ROIs."""
    max_dose = max(np.max(gt_dose), np.max(pred_dose)) * 1.05
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    fig.suptitle(f'Dose-Volume Histograms (DVH) Comparison for Patient: {patient_id}', fontsize=20)

    for i in range(NUM_ROIS):
        roi_mask = struct_data[..., i] 
        roi_label = ROI_LABELS[i]
        ax = axes[i]
        
        # Calculate DVHs with 1 Gy bins for plotting clarity
        dose_gt, vol_gt = calculate_dvh(gt_dose, roi_mask, bin_size=1.0)
        dose_pred, vol_pred = calculate_dvh(pred_dose, roi_mask, bin_size=1.0)

        ax.plot(dose_gt, vol_gt, 'r-', linewidth=2, label=f'GT Dose ({roi_label})')
        ax.plot(dose_pred, vol_pred, 'b--', linewidth=2, label=f'Pred Dose ({roi_label})')

        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title(roi_label, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, max_dose)
        ax.set_ylim(0, 105)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_doses(gt_dose, pred_dose, patient_id):
    """
    Displays side-by-side axial, coronal, and sagittal views of GT and Predicted doses.
    """
    D, H, W = gt_dose.shape
    center_D, center_H, center_W = D // 2, H // 2, W // 2
    vmax = max(np.max(gt_dose), np.max(pred_dose)) * 1.1

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dose Map Comparison for Patient: {patient_id}', fontsize=20, y=1.02)

    titles = [f"Axial (Z={center_D})", f"Coronal (Y={center_H})", f"Sagittal (X={center_W})"]
    doses = [gt_dose, pred_dose]
    labels = ["Ground Truth", "Prediction"]

    for row, dose_map in enumerate(doses):
        dose_label = labels[row]
        axial_slice = dose_map[center_D, :, :]
        coronal_slice = dose_map[:, center_H, :]
        sagittal_slice = dose_map[:, :, center_W]
        
        slices = [axial_slice, coronal_slice, sagittal_slice]
        
        for col, slc in enumerate(slices):
            ax = axes[row, col]
            im = ax.imshow(slc, cmap='jet', origin='lower', vmax=vmax) 
            ax.set_title(f"{dose_label} - {titles[col]}", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if row == 1 and col == 2:
                cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5]) 
                fig.colorbar(im, cax=cbar_ax, label='Dose Value (Gy)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def evaluate_single_patient(patient_path, patient_id, detailed_visuals=False):
    """Loads data, calculates metrics, and optionally plots visualizations."""
    try:
        gt_dose, pred_dose, struct_data = load_patient_data_for_eval(patient_path, patient_id)
        
        # 1. Calculate overall MAE/RMSE (Dose Score metrics)
        metrics = calculate_metrics(gt_dose, pred_dose)
        metrics['PatientID'] = patient_id
        
        # 2. Calculate all individual DVH metrics (D_X and V_Y)
        gt_all_dvh_metrics = {}
        pred_all_dvh_metrics = {}
        
        for i, roi_label in enumerate(ROI_LABELS):
            roi_mask = struct_data[..., i]
            
            # Skip if structure is empty in this patient
            if np.sum(roi_mask) == 0:
                continue

            gt_constraints = calculate_dvh_metrics_for_roi(gt_dose, roi_mask, roi_label)
            pred_constraints = calculate_dvh_metrics_for_roi(pred_dose, roi_mask, roi_label)

            # Store metrics in a flat dictionary keyed by (ROI, metric_key)
            for key, val in gt_constraints.items():
                gt_all_dvh_metrics[(roi_label, key)] = val
            for key, val in pred_constraints.items():
                pred_all_dvh_metrics[(roi_label, key)] = val

        # 3. Calculate OpenKBP Official Scores (CS and HCS)
        openkbp_scores = calculate_openkbp_scores(gt_all_dvh_metrics, pred_all_dvh_metrics, patient_id)
        
        # Add the official scores to the main metrics dict
        metrics['Clinical Score (CS)'] = openkbp_scores['Clinical Score (CS)']
        metrics['Hard Constraint Score (HCS)'] = openkbp_scores['Hard Constraint Score (HCS)']
        
        # Add all individual constraint errors for detailed tables
        metrics.update(openkbp_scores['Constraint_Errors'])


        # Display visualizations if requested 
        if detailed_visuals:
            print(f"\n--- Visualizing {patient_id} ---")
            visualize_doses(gt_dose, pred_dose, patient_id)
            plot_dvh(gt_dose, pred_dose, struct_data, patient_id)
            
        return metrics

    except FileNotFoundError as e:
        print(f"Skipping {patient_id}: Missing file for evaluation. Error: {e}")
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
    pred_file_check = os.path.join(input_path, f'{patient_id_guess}{PRED_FILE_SUFFIX}')
    is_single_patient = os.path.isdir(input_path) and os.path.exists(pred_file_check)
    
    if is_single_patient:
        metrics = evaluate_single_patient(input_path, patient_id_guess, detailed_visuals=True)
        if metrics:
            all_metrics.append(metrics)
            
    elif os.path.isdir(input_path):
        print(f"--- Running BATCH evaluation for directory: {input_path} ---")
        patient_paths = sorted([d for d in glob(os.path.join(input_path, '*')) if os.path.isdir(d)])
        
        if not patient_paths:
            print(f"Error: No patient subfolders found in {input_path}.")
            return
            
        print(f"Found {len(patient_paths)} patient folders to process.")
        
        for patient_path in patient_paths:
            patient_id = os.path.basename(patient_path)
            if os.path.exists(os.path.join(patient_path, f'{patient_id}{PRED_FILE_SUFFIX}')):
                metrics = evaluate_single_patient(patient_path, patient_id, detailed_visuals=False)
                if metrics:
                    all_metrics.append(metrics)
                
    else:
        print(f"Error: Input path is not a valid directory or is not a recognized patient folder structure: {input_path}")
        return

    # --- 2. Print Summary Tables ---
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_df = results_df.round(3)

        # 1. Define Column Groups for Clean Printing
        global_cols_full = ['PatientID', 'MAE (Gy)', 'RMSE (Gy)']
        global_metrics = ['MAE (Gy)', 'RMSE (Gy)']
        
        openkbp_scores_cols = ['Clinical Score (CS)', 'Hard Constraint Score (HCS)']
        
        # Identify all DVH difference columns (D_X and V_Y)
        dvh_diff_cols = [col for col in results_df.columns if '|GT-Pred|' in col]

        # Function to clean column names for display
        def clean_col_name(col):
            # Converts "|GT-Pred| D_95 (PTV70)" to "PTV70 (D_95)"
            parts = col.split('(')
            metric_name = parts[0].strip().replace('|GT-Pred| ', '')
            roi_name = parts[1].replace(')', '')
            return f"{roi_name} ({metric_name})"

        print("\n" + "="*80)
        print("QUANTITATIVE EVALUATION SUMMARY (ALL PATIENTS)")
        print("="*80)

        # --- A. Global Metrics Table (Per Patient) ---
        print("\n--- A. GLOBAL DOSE ACCURACY & OPENKBP SCORES - PER PATIENT ---")
        print(results_df[global_cols_full + openkbp_scores_cols].to_string(index=False))

        # --- B. DVH Constraint Metrics Table (Per Patient) ---
        if dvh_diff_cols:
            print("\n--- B. INDIVIDUAL DVH CONSTRAINTS (Absolute Difference) - PER PATIENT ---")
            
            dvh_display_df = results_df[['PatientID'] + dvh_diff_cols].copy()
            new_dvh_cols = {col: clean_col_name(col) for col in dvh_diff_cols}
            dvh_display_df = dvh_display_df.rename(columns=new_dvh_cols)
            # Reorder columns to group by ROI
            patient_cols = ['PatientID']
            sorted_cols = sorted([col for col in dvh_display_df.columns if col not in patient_cols], key=lambda x: x.split('(')[0].strip())
            print(dvh_display_df[patient_cols + sorted_cols].to_string(index=False))


        # --- 3. Print Aggregated Scores (Mean +/- STD) ---
        if len(all_metrics) > 1:
            
            metric_cols = [col for col in results_df.columns if col != 'PatientID']
            summary_mean = results_df[metric_cols].mean().to_frame(name='Mean')
            summary_std = results_df[metric_cols].std().to_frame(name='STD')
            summary_df = pd.concat([summary_mean, summary_std], axis=1).round(3)
            
            print("\n" + "="*80)
            print("AGGREGATED PERFORMANCE SCORES (Mean $\\pm$ STD)")
            print("="*80)
            
            # C. Official Scores Aggregated
            official_summary = summary_df.loc[global_metrics + openkbp_scores_cols]
            
            print("\n--- C. OFFICIAL OPENKBP LEADERBOARD SCORES ---")
            # Note: Dose Score is MAE
            print(f"**Dose Score (DS) $\\approx$ MAE:** {official_summary.loc['MAE (Gy)']['Mean']} $\\pm$ {official_summary.loc['MAE (Gy)']['STD']} Gy")
            print(f"**Clinical Score (CS):** {official_summary.loc['Clinical Score (CS)']['Mean']} $\\pm$ {official_summary.loc['Clinical Score (CS)']['STD']} Gy")
            print(f"**Hard Constraint Score (HCS):** {official_summary.loc['Hard Constraint Score (HCS)']['Mean']} $\\pm$ {official_summary.loc['Hard Constraint Score (HCS)']['STD']} Gy")
            print("-" * 80)
            
            # D. Individual Constraint Difference Metrics Aggregated
            dvh_summary = summary_df.loc[dvh_diff_cols]
            if not dvh_summary.empty:
                print("\n--- D. AGGREGATED INDIVIDUAL DVH CONSTRAINTS (Absolute Difference) ---")

                new_dvh_index = {index_col: clean_col_name(index_col) for index_col in dvh_summary.index}
                dvh_summary = dvh_summary.rename(index=new_dvh_index)
                
                # Sort index by ROI name
                dvh_summary = dvh_summary.sort_index(key=lambda x: x.map(lambda s: s.split('(')[0].strip()))
                print(dvh_summary.to_string())


    
    else:
        print("\n" + "="*80)
        print("EVALUATION FAILED: NO PATIENTS PROCESSED SUCCESSFULLY")
        print("="*80)


if __name__ == '__main__':
    main()