import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, Any

# --- Constants ---
VOL_DIM = 128
CENTER_SLICE_IDX = VOL_DIM // 2

def load_patient_data(folder_path: Path, patient_id: str) -> Dict[str, np.ndarray]:
    """Loads the three preprocessed NumPy arrays for a given patient."""
    print(f"Loading data for patient: {patient_id}")
    
    data = {}
    
    # 1. CT Volume (128, 128, 128) - Squeeze is done implicitly or during preprocessing save
    ct_path = folder_path / f'{patient_id}_ct_1ch.npy'
    data['CT'] = np.load(ct_path)
    
    # 2. Structure Volume (128, 128, 128, 3)
    struct_path = folder_path / f'{patient_id}_struct_3ch.npy'
    data['Structure'] = np.load(struct_path)
    
    # 3. Dose Volume (128, 128, 128) - Squeeze is done implicitly or during preprocessing save
    dose_path = folder_path / f'{patient_id}_dose_1ch.npy'
    data['Dose'] = np.load(dose_path)
    
    # Ensure CT/Dose are 3D and Structure is 4D for slicing consistency
    # (Note: Data loaded above suggests successful loading of 3D/4D data, so we proceed)
    print(f"Data loaded successfully. CT shape: {data['CT'].shape}, Structure shape: {data['Structure'].shape}")
    
    return data

def plot_slice(ax: plt.Axes, data: np.ndarray, title: str, slicing_view: str, display_label: str):
    """Plots a single slice given the view plane and sets the custom display label."""
    
    # Determine if data is multi-channel (like Structure)
    is_rgb = (data.ndim == 4 and data.shape[-1] == 3)
    
    plot_data = None
    
    # Standard 3D indexing for both 3D (CT/Dose) and 4D (Structure) volumes.
    # The structure volume is D, H, W, C (0, 1, 2, 3)
    
    if slicing_view == 'Axial':
        # Axial (Transverse): Slice perpendicular to D (index 0). Resulting shape is (H, W, C) or (H, W).
        plot_data = data[CENTER_SLICE_IDX, :, :]
        
    elif slicing_view == 'Coronal':
        # Coronal: Slice perpendicular to H (index 1). Resulting shape is (D, W, C) or (D, W).
        plot_data = data[:, CENTER_SLICE_IDX, :]
        
        # FIX: The slicing already results in (D, W, C), which is correct for imshow. 
        # The previous moveaxis call was incorrect and caused the (128, 3, 128) error.
             
    elif slicing_view == 'Sagittal':
        # Sagittal: Slice perpendicular to W (index 2). Resulting shape is (D, H, C) or (D, H).
        plot_data = data[:, :, CENTER_SLICE_IDX]
        
        # Transpose the D and H axes to get the conventional orientation (H vertical, D horizontal).
        if plot_data.ndim == 3:
            # If 4D data (D, H, C), transpose the first two spatial axes (D, H) -> (H, D)
            plot_data = np.moveaxis(plot_data, [0, 1], [1, 0])
        else:
            # If 3D data (D, H), just transpose (D, H) -> (H, D)
            plot_data = plot_data.T

    
    if plot_data is None:
        return

    # Squeeze out the single-channel dimension if present (e.g., 128x128x1)
    if plot_data.ndim == 3 and plot_data.shape[-1] == 1:
         plot_data = plot_data.squeeze(axis=-1)
    
    # Handle colormaps and interpolation
    cmap = 'gray'
    interpolation = 'nearest'
    
    # Structure data must be normalized to float for correct RGB display
    if is_rgb:
        # Structure data is uint8 (values 0-4). Normalize to 0.0-1.0 float range.
        plot_data = plot_data / plot_data.max() if plot_data.max() > 0 else plot_data
        cmap = None # Use default RGB color mapping
    elif title == 'Dose':
        # Use plasma or jet for dose visualization
        cmap = 'jet' 
        interpolation = 'bilinear'

    img = ax.imshow(plot_data, cmap=cmap, interpolation=interpolation)
    ax.set_title(f"{title} ({display_label})", fontsize=10) # Use the separate display_label
    ax.axis('off')

    # Add colorbar for scalar data
    if not is_rgb:
        # The axes for the colorbar are different if the Sagittal view was transposed
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        if title == 'CT':
             cbar.set_label('Hounsfield Units', fontsize=8)
        elif title == 'Dose':
             cbar.set_label('Dose (Gy)', fontsize=8)


def visualize_patient(folder_path: Path):
    """Main visualization function."""
    
    if not folder_path.exists():
        print(f"Error: Folder not found at {folder_path}")
        return

    patient_id = folder_path.name
    
    try:
        data_dict = load_patient_data(folder_path, patient_id)
    except FileNotFoundError as e:
        print(f"Error: Could not load data. Ensure all .npy files exist. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # Setup the 3x3 figure: Rows = Datasets (CT, Struct, Dose), Columns = Views (Axial, Coronal, Sagittal)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.suptitle(f"Patient Visualization: {patient_id}", fontsize=16)

    datasets = ['CT', 'Structure', 'Dose']
    
    # Define the physical slices being plotted in each column (Axial=Col0, Coronal=Col1, Sagittal=Col2)
    slicing_views = ['Axial', 'Coronal', 'Sagittal']
    
    # Define the labels for each column based on the new requested permutation:
    # Col 0 (Axial slice) -> 'Coronal'
    # Col 1 (Coronal slice) -> 'Sagittal'
    # Col 2 (Sagittal slice) -> 'Axial'
    display_labels = ['Coronal', 'Sagittal', 'Axial'] # Updated Permuted labels

    for i, dataset_name in enumerate(datasets):
        for j, slicing_view in enumerate(slicing_views):
            ax = axes[i, j]
            data = data_dict[dataset_name]
            
            # Use the correct display label for the column (j)
            display_label = display_labels[j] 
            
            plot_slice(ax, data, dataset_name, slicing_view, display_label) # Pass both view and label

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def main():
    """Handles command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_patient.py <path_to_patient_folder>")
        print("Example: python visualize_patient.py ./transformed-data/training/pt_001")
        sys.exit(1)

    patient_folder_path = Path(sys.argv[1]).resolve()
    visualize_patient(patient_folder_path)

if __name__ == "__main__":
    main()
