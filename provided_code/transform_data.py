import numpy as np
import os
import glob
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
import traceback

# --- 1. Define Data Constants and Encoding ---

VOL_DIM = 128
CT_SHAPE_3D = (VOL_DIM, VOL_DIM, VOL_DIM)

# The list of ROIs used for 3-channel encoding
FULL_ROI_LIST = [
    "PTV70", "PTV63", "PTV56",                          
    "Brainstem", "SpinalCord", "Mandible", "Larynx",    
    "RightParotid", "LeftParotid", "Esophagus",
]

# Unique 3-Channel Encoding Scheme (Disjoint Axes)
COLOR_ENCODING = {
    # Channel 0 (R): PTVs + Brainstem
    'PTV70':        np.array([1, 0, 0], dtype=np.uint8),
    'PTV63':        np.array([2, 0, 0], dtype=np.uint8), 
    'PTV56':        np.array([3, 0, 0], dtype=np.uint8),
    'Brainstem':    np.array([4, 0, 0], dtype=np.uint8), 
    
    # Channel 1 (G): SpinalCord + Mandible + Larynx + RightParotid
    'SpinalCord':   np.array([0, 1, 0], dtype=np.uint8),
    'Mandible':     np.array([0, 2, 0], dtype=np.uint8),
    'Larynx':       np.array([0, 3, 0], dtype=np.uint8),
    'RightParotid': np.array([0, 4, 0], dtype=np.uint8),
    
    # Channel 2 (B): LeftParotid + Esophagus
    'LeftParotid':  np.array([0, 0, 1], dtype=np.uint8),
    'Esophagus':    np.array([0, 0, 2], dtype=np.uint8),
}

# --- 2. CSV Loading Logic (Robust for CT, Dose, and Mask structures) ---

def load_file_from_csv(file_path: Path):
    """
    Loads sparse CSV data, handling the header row (',data') and column count.
    Returns dict{'indices', 'data'} for scalar, or indices array for mask.
    """
    try:
        is_scalar_file = 'ct.csv' in file_path.name or 'dose.csv' in file_path.name
        is_mask_file = any(name in file_path.name for name in COLOR_ENCODING.keys())
        
        # Skip the header row (',data') for CT, Dose, and Mask files.
        skip_rows = 1 if (is_scalar_file or is_mask_file) else 0

        # Use keep_default_na=False to prevent 'NA' in the header from being interpreted as NaN
        df = pd.read_csv(file_path, header=None, skiprows=skip_rows, keep_default_na=False)
        
        # --- SCALAR VOLUME (CT/DOSE) ---
        if is_scalar_file:
            if df.shape[1] < 2:
                 raise ValueError(f"Scalar file {file_path.name} has fewer than 2 data columns.")
                 
            # Column 0 is indices, Column 1 is data/values
            return {
                "indices": df.iloc[:, 0].values,
                "data": df.iloc[:, 1].values
            }
            
        # --- MASK VOLUME (PTV/OAR) ---
        elif is_mask_file:
            # Masks have an empty second column but we only need the indices (column 0).
            # Return only the 1D array of indices.
            return df.iloc[:, 0].values
            
        # --- OTHER FILES (e.g., voxel_dimensions.csv) ---
        else:
            # Assumes single column for other files.
            return df.iloc[:, 0].values 
            
    except Exception:
        return None

def load_scalar_volume(patient_dir: Path, name: str, shape: tuple) -> np.ndarray:
    """Load a scalar volume (ct or dose) from sparse CSV -> (D,H,W) float32."""
    p = patient_dir / f"{name}.csv"
    obj = load_file_from_csv(p)
    if obj is None or "indices" not in obj:
        raise FileNotFoundError(f"Required file {p} not found or invalid.")
        
    vol = np.zeros(shape, dtype=np.float32)
    # Ensure indices are integers and values are floats before placement
    flat_idx = obj["indices"].astype(np.int64)
    vals = obj["data"].astype(np.float32)
    np.put(vol, flat_idx, vals)
    return vol

def try_load_mask(patient_dir: Path, roi_name: str, shape: tuple) -> np.ndarray | None:
    """Try to load single-ROI mask (index-only CSV). Returns (D,H,W) uint8 or None if missing."""
    p = patient_dir / f"{roi_name}.csv"
    if not p.exists():
        return None
        
    idx = load_file_from_csv(p)
    if idx is None or idx.ndim != 1:
        # If the CSV exists but the content is wrong, skip it.
        # print(f"Warning: Mask CSV {p.name} found but content is invalid. Skipping.")
        return None
        
    m = np.zeros(shape, dtype=np.uint8)
    np.put(m, idx.astype(np.int64), 1)
    return m

# --- 3. Core Processing Logic ---

def create_structure_volume(patient_folder: Path, patient_id: str) -> np.ndarray:
    """Reads individual structure masks and combines them into the 3-channel volume."""
    struct_volume = np.zeros((*CT_SHAPE_3D, 3), dtype=np.uint8)
    
    for name, color in COLOR_ENCODING.items():
        mask_3d = try_load_mask(patient_folder, name, CT_SHAPE_3D)
        
        if mask_3d is None:
            continue
            
        # Unique Overlap Resolution (Disjoint Update)
        z_idx, y_idx, x_idx = np.where(mask_3d == 1)
        active_channel_index = np.argmax(color)
        value = color[active_channel_index]
        
        struct_volume[z_idx, y_idx, x_idx, active_channel_index] = value
        
    return struct_volume

def process_patient(patient_folder: Path, output_root: Path):
    """Handles file loading, volume creation, and saving for a single patient."""
    patient_id = patient_folder.name
    
    # --- A. Define Input/Output Paths ---
    output_patient_folder = output_root / patient_id
    output_patient_folder.mkdir(parents=True, exist_ok=True)
    
    ct_output_path = output_patient_folder / f'{patient_id}_ct_1ch.npy'
    struct_output_path = output_patient_folder / f'{patient_id}_struct_3ch.npy'
    dose_output_path = output_patient_folder / f'{patient_id}_dose_1ch.npy' # NEW DOSE PATH

    # CHECK: Skip if ALL output files exist.
    if ct_output_path.exists() and struct_output_path.exists() and dose_output_path.exists():
        return

    # --- B. Load CT, Dose, and Create Structure Volume ---
    try:
        # 1. Load CT Volume (128x128x128)
        ct_volume = load_scalar_volume(patient_folder, "ct", CT_SHAPE_3D)
        
        # 2. Load Dose Volume (128x128x128) - NEW LOADING
        dose_volume = load_scalar_volume(patient_folder, "dose", CT_SHAPE_3D)
        
        # 3. Generate the 3-channel structure volume (128x128x128x3)
        struct_volume = create_structure_volume(patient_folder, patient_id)

    except FileNotFoundError as e:
        print(f"\nERROR (FileNotFound) for {patient_id}: {e}")
        return
    except Exception as e:
        print(f"\nFATAL ERROR processing {patient_id}: {e}")
        traceback.print_exc(file=sys.stdout)
        return

    # --- C. Save Results ---
    try:
        np.save(ct_output_path, ct_volume)
        np.save(struct_output_path, struct_volume)
        np.save(dose_output_path, dose_volume) # NEW DOSE SAVE
        
    except Exception as e:
        print(f"\nFATAL ERROR saving files for {patient_id}: {e}")
        traceback.print_exc(file=sys.stdout)
        return

# --- 4. Main Execution Function ---

def main():
    """
    Orchestrates the entire preprocessing pipeline from the command line.
    
    Usage: python script_name.py <path_to_provided_data_folder>
    """
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_provided_data_folder>")
        print("Example: python preprocess.py ./provided-data")
        sys.exit(1)

    input_base_dir = Path(sys.argv[1]).resolve()
    
    if not input_base_dir.is_dir():
        print(f"Error: Input path is not a valid directory: {input_base_dir}")
        sys.exit(1)

    print(f"Starting preprocessing for data in: {input_base_dir}")
    
    output_base_dir = input_base_dir.parent / 'transformed-data'
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
        for patient_folder in tqdm(patient_folders, desc=f"Processing {split_name}"):
            process_patient(patient_folder, output_root)

    print("\nPreprocessing Complete! Review any errors printed above.")
    print(f"Transformed data saved to: {output_base_dir}")


if __name__ == "__main__":
    main()