import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
import traceback

# --- Constants ---
# Assuming the pre-processed volumes are 128x128x128
VOL_DIM = 128 
CT_SHAPE_3D = (VOL_DIM, VOL_DIM, VOL_DIM)

# The maximum value used in the structure encoding (4) is needed for normalization
# This value ensures the mask colors scale correctly to the 0.0-1.0 float range.
MAX_STRUCT_VALUE = 4.0 

# Transparency factor for the overlaid structures (0.0 = fully transparent, 1.0 = fully opaque)
# Adjust this value to your preference. A value between 0.3 and 0.7 usually works well.
OVERLAY_ALPHA = 0.5 

# --- Core Processing Logic ---

def create_overlay_volume(ct_vol: np.ndarray, struct_vol: np.ndarray) -> np.ndarray:
    """
    Creates a 3-channel volume where the CT data is shown in grayscale
    and is overlaid by the colored 3-channel structure mask wherever the mask is active,
    with transparency.

    The output volume is a float32 array in the range [0.0, 1.0].
    
    Args:
        ct_vol (np.ndarray): 1-channel CT volume (D, H, W) - assumed to be float32.
        struct_vol (np.ndarray): 3-channel structure volume (D, H, W, 3) - uint8 (values 0-4).

    Returns:
        np.ndarray: The overlaid 3-channel volume (D, H, W, 3) - float32.
    """
    
    # 1. Normalize CT for Grayscale Base
    ct_range = ct_vol.max() - ct_vol.min()
    if ct_range == 0:
        ct_normalized = np.zeros_like(ct_vol, dtype=np.float32)
    else:
        ct_normalized = (ct_vol - ct_vol.min()) / ct_range
        
    # Create the 3-channel grayscale base CT volume
    ct_rgb_base = np.stack([ct_normalized] * 3, axis=-1).astype(np.float32)
    
    # 2. Normalize Structure Mask Colors to [0.0, 1.0] range
    struct_normalized = struct_vol.astype(np.float32) / MAX_STRUCT_VALUE
    
    # 3. Identify all voxels that belong to ANY ROI
    # The result shape is (D, H, W), then we expand to (D, H, W, 1) for broadcasting
    is_mask = np.any(struct_vol > 0, axis=-1)[..., np.newaxis]
    
    # 4. Combine/Overlay with Transparency (Alpha Blending)
    # Where mask is active, blend structure color with CT base using OVERLAY_ALPHA.
    # Formula: `final_color = alpha * foreground_color + (1 - alpha) * background_color`
    
    overlay_vol = np.where(
        is_mask, 
        OVERLAY_ALPHA * struct_normalized + (1 - OVERLAY_ALPHA) * ct_rgb_base, # Blended color
        ct_rgb_base # CT base where no mask is present
    )
    
    return overlay_vol

def process_patient_overlay(patient_folder: Path, output_root: Path):
    """Handles file loading, overlay creation, and saving/copying for a single patient."""
    patient_id = patient_folder.name
    
    # --- A. Define Input/Output Paths ---
    output_patient_folder = output_root / patient_id
    output_patient_folder.mkdir(parents=True, exist_ok=True)
    
    # Input paths for the required files
    ct_input_path = patient_folder / f'{patient_id}_ct_1ch.npy'
    struct_input_path = patient_folder / f'{patient_id}_struct_3ch.npy'
    dose_input_path = patient_folder / f'{patient_id}_dose_1ch.npy'

    # Output paths for the new files
    overlay_output_path = output_patient_folder / f'{patient_id}_ct_overlay_3ch.npy'
    dose_output_path = output_patient_folder / f'{patient_id}_dose_1ch.npy' 

    # CHECK: Skip if ALL output files exist.
    if overlay_output_path.exists() and dose_output_path.exists():
        return

    # --- B. Load CT and Structure Volume ---
    try:
        # 1. Load CT Volume (128x128x128)
        ct_volume = np.load(ct_input_path)
        
        # 2. Load Structure Volume (128x128x128x3)
        struct_volume = np.load(struct_input_path)
        
    except FileNotFoundError as e:
        print(f"\nERROR (FileNotFound) for {patient_id}: Missing CT/Structure file. Details: {e}")
        return
    except Exception as e:
        print(f"\nFATAL ERROR loading data for {patient_id}: {e}")
        traceback.print_exc(file=sys.stdout)
        return

    # --- C. Generate Overlay Volume ---
    try:
        overlay_volume = create_overlay_volume(ct_volume, struct_volume)

    except Exception as e:
        print(f"\nFATAL ERROR creating overlay for {patient_id}: {e}")
        traceback.print_exc(file=sys.stdout)
        return
        
    # --- D. Save Overlay and Copy Dose ---
    try:
        # 1. Save the newly created overlay volume
        np.save(overlay_output_path, overlay_volume)
        
        # 2. Copy the Dose file (No processing required)
        # Note: Since the dose file is usually large, direct binary copy is safest,
        # but for simplicity and safety against corruption, we load and save it.
        # This implicitly copies the file.
        dose_volume = np.load(dose_input_path) 
        np.save(dose_output_path, dose_volume)
        
    except FileNotFoundError as e:
        # This catches dose_input_path missing, even though we loaded CT/Struct successfully
        print(f"\nERROR (FileNotFound) for {patient_id}: Missing Dose file. Details: {e}")
        return
    except Exception as e:
        print(f"\nFATAL ERROR saving files for {patient_id}: {e}")
        traceback.print_exc(file=sys.stdout)
        return

# --- 4. Main Execution Function ---

def main():
    """
    Orchestrates the overlay generation pipeline from the command line.
    
    Usage: python script_name.py <path_to_transformed_data_folder>
    """
    if len(sys.argv) != 2:
        print("Usage: python create_overlaid_volumes.py <path_to_transformed_data_folder>")
        print("Example: python create_overlaid_volumes.py ./transformed-data")
        sys.exit(1)

    input_base_dir = Path(sys.argv[1]).resolve()
    
    if not input_base_dir.is_dir():
        print(f"Error: Input path is not a valid directory: {input_base_dir}")
        sys.exit(1)

    print(f"Starting overlay creation for data in: {input_base_dir}")
    
    # Create a new output folder next to the input folder
    output_base_dir = input_base_dir.parent / 'overlaid-data'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    split_dirs = sorted([d for d in input_base_dir.glob('*') if d.is_dir()])
    
    if not split_dirs:
        print(f"Error: No split folders (e.g., 'training') found in {input_base_dir}.")
        return

    for split_dir in split_dirs:
        split_name = split_dir.name
        print(f"\n--- Processing Split: {split_name} ---")
        
        output_root = output_base_dir / split_name
        output_root.mkdir(parents=True, exist_ok=True)
        
        patient_folders = sorted([d for d in split_dir.glob('*') if d.is_dir()])

        if not patient_folders:
            print(f"No patient folders found in {split_dir}.")
            continue
        
        print(f"Found {len(patient_folders)} patients.")

        # Process each patient using the tqdm progress bar
        for patient_folder in tqdm(patient_folders, desc=f"Creating overlays for {split_name}"):
            process_patient_overlay(patient_folder, output_root)

    print("\nOverlay Creation Complete! Review any errors printed above.")
    print(f"Overlaid data saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
