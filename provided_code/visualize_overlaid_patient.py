import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, Any

# --- Constants ---
VOL_DIM = 128
CENTER_SLICE_IDX = VOL_DIM // 2 # Center slice index (64 for 128)

def load_overlay_data(folder_path: Path, patient_id: str) -> Dict[str, np.ndarray]:
    """Loads the overlaid CT/Structure and the Dose NumPy arrays for a given patient."""
    print(f"Loading overlaid data for patient: {patient_id}")
    
    data = {}
    
    # 1. Overlaid CT/Structure Volume (128, 128, 128, 3) - float32 [0.0, 1.0]
    overlay_path = folder_path / f'{patient_id}_ct_overlay_3ch.npy'
    data['Overlay'] = np.load(overlay_path)
    
    # 2. Dose Volume (128, 128, 128) - 1-channel
    dose_path = folder_path / f'{patient_id}_dose_1ch.npy'
    data['Dose'] = np.load(dose_path)
    
    print(f"Data loaded successfully. Overlay shape: {data['Overlay'].shape}, Dose shape: {data['Dose'].shape}")
    
    return data

def plot_slice(ax: plt.Axes, data: np.ndarray, title: str, slicing_view: str, display_label: str):
    """
    Plots a single slice given the view plane.
    Handles both 3-channel (Overlay) and 1-channel (Dose) data.
    
    Args:
        slicing_view (str): The actual slice plane used for data extraction ('Axial', 'Coronal', 'Sagittal').
        display_label (str): The name to display in the plot title (e.g., 'Coronal', 'Sagittal', 'Axial').
    """
    
    plot_data = None
    
    # Standard 3D indexing based on (D, H, W, C) or (D, H, W). We use the slicing_view for logic.
    if slicing_view == 'Axial':
        # Axial (Transverse): Slice perpendicular to D (index 0). Resulting shape is (H, W, C) or (H, W).
        plot_data = data[CENTER_SLICE_IDX, :, :]
        
    elif slicing_view == 'Coronal':
        # Coronal: Slice perpendicular to H (index 1). Resulting shape is (D, W, C) or (D, W).
        plot_data = data[:, CENTER_SLICE_IDX, :]
             
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

    # Handle colormaps and interpolation
    interpolation = 'nearest'
    cbar_label = None

    if title == 'Overlay':
        # Overlaid volume is 3-channel RGB (float32, [0.0, 1.0])
        cmap = None
    elif title == 'Dose':
        # Dose volume is 1-channel scalar data
        cmap = 'jet' 
        interpolation = 'bilinear'
        cbar_label = 'Dose (Gy)'

    img = ax.imshow(plot_data, cmap=cmap, interpolation=interpolation)
    ax.set_title(f"{title} ({display_label})", fontsize=10) # Use the display_label for the title
    ax.axis('off')

    # Add colorbar for scalar Dose data
    if title == 'Dose' and cbar_label:
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=8)


def visualize_patient(folder_path: Path):
    """Main visualization function."""
    
    if not folder_path.exists():
        print(f"Error: Patient folder not found at {folder_path}")
        return

    patient_id = folder_path.name
    
    try:
        data_dict = load_overlay_data(folder_path, patient_id)
    except FileNotFoundError as e:
        print(f"Error: Could not load data. Ensure '{patient_id}_ct_overlay_3ch.npy' and '{patient_id}_dose_1ch.npy' exist in the patient folder. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # Setup the 2x3 figure: Rows = Datasets (Overlay, Dose), Columns = Views (Axial, Coronal, Sagittal)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.suptitle(f"Overlaid CT/Structure & Dose Visualization for Patient: {patient_id}", fontsize=14)

    datasets = ['Overlay', 'Dose']
    
    # 1. Define the physical views used for slicing (This is the logic, which remains the same)
    slicing_views = ['Axial', 'Coronal', 'Sagittal']
    
    # 2. Define the labels to be displayed (This is the requested naming change)
    # Col 0 (Axial slice) -> 'Coronal'
    # Col 1 (Coronal slice) -> 'Sagittal'
    # Col 2 (Sagittal slice) -> 'Axial'
    display_labels = ['Coronal', 'Sagittal', 'Axial']

    for i, dataset_name in enumerate(datasets):
        for j, slicing_view in enumerate(slicing_views):
            ax = axes[i, j]
            data = data_dict[dataset_name]
            
            # Use the correct permuted display label for the column (j)
            display_label = display_labels[j]
            
            # Pass both the physical slice name and the desired label to the plotting function
            plot_slice(ax, data, dataset_name, slicing_view, display_label)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def main():
    """Handles command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_overlaid_patient.py <path_to_patient_folder>")
        print("Example: python visualize_overlaid_patient.py ./overlaid-data/training/pt_001")
        sys.exit(1)

    patient_folder_path = Path(sys.argv[1]).resolve()
    visualize_patient(patient_folder_path)

if __name__ == "__main__":
    main()
