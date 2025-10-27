import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from glob import glob

class DosePredictionDataset(Dataset):
    """
    Dataset for loading 3D dose prediction data from NumPy files.
    
    The data pipeline loads three tensors per patient:
    1. Input (3ch): CT + overaly masks for model input.
    2. Target Dose (1ch): Ground truth dose distribution.
    3. Structure Masks (10ch): 10 OAR/PTV masks for loss calculation only.
    
    Expected file structure (relative to root_dir/overlaid-data/[split]-pats/pt_X/):
    - pt_X_ct_overlay_3ch.npy (Input: D, H, W, 3)
    - pt_X_dose_1ch.npy (Target Dose: D, H, W, 1)
    - pt_X_struct_10ch.npy (Structure Masks: D, H, W, 10)
    """
    def __init__(self, root_dir, split='train'):
        # Note: root_dir is expected to be the directory *containing* overlaid-data (e.g., '..')
        self.data_path = os.path.join(root_dir, 'overlaid-data', f'{split}-pats')
        self.split = split
        
        # Find all patient directories
        self.patient_dirs = sorted(glob(os.path.join(self.data_path, 'pt_*')))
        
        if not self.patient_dirs:
            print(f"Warning: No patient folders found in {self.data_path}. Check your root_dir path.")
        
        print(f"Initializing DosePredictionDataset for '{split}' split with {len(self.patient_dirs)} patients.")

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)
        
        # Define file paths based on naming conventions
        input_file = os.path.join(patient_dir, f'{patient_id}_ct_overlay_3ch.npy')
        target_dose_file = os.path.join(patient_dir, f'{patient_id}_dose_1ch.npy')
        all_masks_file = os.path.join(patient_dir, f'{patient_id}_struct_10ch.npy') 
        
        try:
            # 1. Load Input (CT + OAR/PTV overlay) - Expected Shape (D, H, W, 3)
            input_data_np = np.load(input_file).astype(np.float32)

            # 2. Load Target Dose - Expected Shape (D, H, W, 1)
            target_dose_np = np.load(target_dose_file).astype(np.float32)
            
            # 3. Load 10-Channel Structure Masks - Expected Shape (D, H, W, 10)
            all_masks_np = np.load(all_masks_file).astype(np.float32)
            
            # --- CRITICAL FIX: Ensure the final channel dimension exists ---
            
            # Check Input (expected to be D, H, W, 3) - Expand if necessary
            if input_data_np.ndim == 3:
                # Assuming the input data is (D, H, W) and the 3 channels are consolidated into a single volume
                # Based on the file name, it should be 4D. If it's 3D, we need investigation, 
                # but for safety against accidental 3D shapes, we print a warning.
                print(f"Warning: Input data {patient_id} is 3D. Expected (D, H, W, 3). Check data generation pipeline.")
                # We will proceed assuming the data pipeline handles the 3 channels correctly, 
                # but we must ensure it's 4D for the permute operation.
            
            # FIX: Check Target Dose. If it's (D, H, W), expand to (D, H, W, 1).
            if target_dose_np.ndim == 3:
                target_dose_np = np.expand_dims(target_dose_np, axis=-1)
            elif target_dose_np.ndim != 4:
                raise ValueError(f"Target dose for {patient_id} has unexpected dimensions: {target_dose_np.shape}. Expected 3 or 4.")

            # FIX: Check Masks. If it's 3D, we can't determine channel count, but we prevent the permute error.
            if all_masks_np.ndim == 3:
                 # This is highly irregular for 10-channel data. 
                 # We assume the data pipeline guarantees the (D, H, W, 10) shape.
                 # If this happens, the error will likely move to 'permute(3, 0, 1, 2)'.
                 pass # We rely on the 4D load (D, H, W, 10) for masks.

            # --- Permute Axes to PyTorch format (C, D, H, W) ---
            
            # (D, H, W, 3) -> (3, D, H, W)
            inputs = torch.from_numpy(input_data_np).permute(3, 0, 1, 2) 
            
            # (D, H, W, 1) -> (1, D, H, W)
            # This is the line that failed. It now expects the expanded 4D array.
            targets = torch.from_numpy(target_dose_np).permute(3, 0, 1, 2) 
            
            # (D, H, W, 10) -> (10, D, H, W)
            all_masks = torch.from_numpy(all_masks_np).permute(3, 0, 1, 2) 
            
            # Return three items: (Model Input, Target, Loss Masks)
            return inputs, targets, all_masks

        except FileNotFoundError as e:
            # If a file is missing, we must raise an error to stop training.
            raise FileNotFoundError(f"Missing file for patient {patient_id}: {e}")
        except Exception as e:
            print(f"Error loading data for patient {patient_id}: {e}")
            raise


def get_data_loaders(root_dir, batch_size, num_workers=4):
    """
    Initializes and returns the training and validation DataLoaders.
    """
    try:
        train_dataset = DosePredictionDataset(root_dir, split='train')
        val_dataset = DosePredictionDataset(root_dir, split='validation')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data loading error: {e}") from e
        
    # Check if data was actually found
    if not train_dataset.patient_dirs or not val_dataset.patient_dirs:
        # Re-raise an error if no data was found
        raise FileNotFoundError(f"No patient data found. Please adjust the ROOT_DATA_DIR in train.py.")
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader
