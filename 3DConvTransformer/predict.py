import torch
import numpy as np
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import sys
import random

# --- IMPORT NECESSARY MODULES ---
# Add parent directory to path to allow importing modules like Models.model
# This handles the relative imports typically required in project structures.
sys.path.append(os.path.abspath('.'))

try:
    from Models.model import FCBFormer
except ImportError:
    print("Error: Could not import FCBFormer from 'Models.model'. Please ensure 'model.py' is saved in the Models directory.")
    sys.exit(1)

# Note: We do not import dataloader here, as we implement simplified single-patient loading.

# --- CONFIGURATION ---
IN_CHANNELS = 3    
OUT_CHANNELS = 1   
INPUT_SIZE = 128   # Assuming D=H=W=128
CHECKPOINT_PATH = './checkpoints/best_model.pth' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def parse_args():
    """Parse command line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="3D Conv Transformer Dose Prediction")
    parser.add_argument('input_path', type=str, 
                        help='Path to a single patient directory (e.g., pt_1) OR a directory containing patient subfolders.')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, 
                        help='Path to the model checkpoint file (.pth).')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Initializes model and loads weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = FCBFormer(IN_CHANNELS, OUT_CHANNELS, INPUT_SIZE).to(device)
    
    # Load state dict, mapping to CPU if necessary
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Successfully loaded model from {checkpoint_path}")
    return model


def load_patient_data(patient_path, patient_id, device):
    """Loads input data (CT + overlay) for a single patient."""
    input_file = os.path.join(patient_path, f'{patient_id}_ct_overlay_3ch.npy')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found for {patient_id} at {input_file}")

    input_data_np = np.load(input_file)

    # --- FIX: Use np.ascontiguousarray for robust C-order memory layout and float32 type ---
    # This is highly compatible across NumPy versions and ensures PyTorch compatibility.
    input_data_np = np.ascontiguousarray(input_data_np, dtype=np.float32)
    # ---------------------------------------------------------------------------------------

    # Ensure the data has 4 dimensions (D, H, W, C) before permuting
    if input_data_np.ndim != 4 or input_data_np.shape[-1] != IN_CHANNELS:
        print(f"Warning: Input data shape {input_data_np.shape} is unexpected for {IN_CHANNELS} channels.")

    # Convert to PyTorch tensor and set to model format (C, D, H, W)
    # The permute is C=3, D=0, H=1, W=2
    X = torch.from_numpy(input_data_np).permute(3, 0, 1, 2)
    
    # Since we used np.ascontiguousarray, X should already be contiguous, 
    # but we can keep .contiguous() for safety after permuting if NumPy didn't guarantee it.
    X = X.contiguous() 
    
    # Add batch dimension (N=1, C, D, H, W) and move to device
    X = X.unsqueeze(0).to(device)
    
    return X


def visualize_prediction(pred_dose_np, patient_id):
    """
    Displays axial, coronal, and sagittal views of the predicted dose.
    Input pred_dose_np shape: (D, H, W)
    """
    D, H, W = pred_dose_np.shape
    
    # Calculate middle slices
    center_D, center_H, center_W = D // 2, H // 2, W // 2

    # Get slices
    axial_slice = pred_dose_np[center_D, :, :]   # H x W
    coronal_slice = pred_dose_np[:, center_H, :] # D x W
    sagittal_slice = pred_dose_np[:, :, center_W] # D x H
    
    # Determine the color map maximum based on dose values (e.g., 80 or max dose)
    vmax = np.max(pred_dose_np) * 1.1 # Use a slight buffer over the max dose

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Predicted Dose Distribution for Patient: {patient_id}', fontsize=16)

    # 1. Axial View (D slice)
    im0 = axes[0].imshow(axial_slice, cmap='jet', origin='lower', vmax=vmax)
    axes[0].set_title(f"Axial View (Z={center_D})")
    axes[0].set_xlabel("Width (W)")
    axes[0].set_ylabel("Height (H)")
    
    # 2. Coronal View (H slice)
    im1 = axes[1].imshow(coronal_slice, cmap='jet', origin='lower', vmax=vmax)
    axes[1].set_title(f"Coronal View (Y={center_H})")
    axes[1].set_xlabel("Width (W)")
    axes[1].set_ylabel("Depth (D)")

    # 3. Sagittal View (W slice)
    im2 = axes[2].imshow(sagittal_slice, cmap='jet', origin='lower', vmax=vmax)
    axes[2].set_title(f"Sagittal View (X={center_W})")
    axes[2].set_xlabel("Height (H)")
    axes[2].set_ylabel("Depth (D)")
    
    # Add colorbar to the figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im0, cax=cbar_ax, label='Dose Value')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for colorbar
    plt.show()


def predict_single_patient(model, patient_path, patient_id):
    """Performs prediction for one patient, saves output, and visualizes."""
    try:
        # Load Input Data
        X = load_patient_data(patient_path, patient_id, device)
        
        print(f"Predicting dose for patient {patient_id}...")

        # Run Inference
        with torch.no_grad():
            output = model(X)

        # Process Output: (N, C, D, H, W) -> (D, H, W) NumPy
        # N=1, C=1, so we squeeze and convert to NumPy
        pred_dose_np = output.squeeze().cpu().numpy()
        
        # Check output shape
        if pred_dose_np.ndim != 3 or pred_dose_np.shape != (INPUT_SIZE, INPUT_SIZE, INPUT_SIZE):
             print(f"Warning: Output dose shape is {pred_dose_np.shape}. Expected ({INPUT_SIZE}, {INPUT_SIZE}, {INPUT_SIZE}).")

        # Save Prediction
        output_file = os.path.join(patient_path, f'{patient_id}_predicted_dose_1ch.npy')
        np.save(output_file, pred_dose_np)
        print(f"Predicted dose saved to: {output_file}")

        # Visualize Prediction
        visualize_prediction(pred_dose_np, patient_id)

    except FileNotFoundError as e:
        print(f"Error processing {patient_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {patient_id}: {e}")


def main():
    args = parse_args()

    # --- 1. Load Model ---
    try:
        model = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"Failed to initialize or load model: {e}")
        return

    input_path = os.path.abspath(args.input_path)
    
    # --- 2. Determine Prediction Mode ---
    
    # Check if the input path is a single patient folder (e.g., 'pt_1', 'patient_A')
    # A patient folder is assumed to contain the required input file directly.
    # We use a heuristic: if the input_path contains the input file, it's a patient folder.
    patient_id_guess = os.path.basename(input_path)
    test_input_file = os.path.join(input_path, f'{patient_id_guess}_ct_overlay_3ch.npy')
    
    if os.path.isdir(input_path) and os.path.exists(test_input_file):
        # Case A: Single patient directory provided
        print(f"Detected single patient directory: {patient_id_guess}")
        predict_single_patient(model, input_path, patient_id_guess)
        
    elif os.path.isdir(input_path):
        # Case B: Directory containing multiple patient subfolders
        print(f"Detected directory of patient folders: {input_path}")
        
        # Find all patient subfolders (e.g., pt_*, patient_*)
        patient_paths = sorted([d for d in glob(os.path.join(input_path, '*')) if os.path.isdir(d)])
        
        if not patient_paths:
            print(f"Error: No patient subfolders found in {input_path}.")
            return
            
        print(f"Found {len(patient_paths)} patient folders to process.")
        
        for patient_path in patient_paths:
            patient_id = os.path.basename(patient_path)
            # Only predict if the required input file exists in the subdirectory
            if os.path.exists(os.path.join(patient_path, f'{patient_id}_ct_overlay_3ch.npy')):
                predict_single_patient(model, patient_path, patient_id)
            else:
                print(f"Skipping {patient_id}: Input file not found in subfolder.")
                
    else:
        print(f"Error: Input path is not a valid directory or is not a recognized patient folder structure: {input_path}")


if __name__ == '__main__':
    main()
