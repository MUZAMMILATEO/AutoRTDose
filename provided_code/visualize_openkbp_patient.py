#!/usr/bin/env python3
"""
Visualize one OpenKBP patient: CT, Dose, and multi-channel structure masks.

Shows a single window with a 3x3 grid:
  Rows:    Axial, Sagittal, Coronal
  Columns: CT, CT+Dose overlay, Label map

Usage:
  python provided_code/visualize_openkbp_patient.py \
    --patient_dir /path/to/pt_1 \
    --slice 64
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys, os

# Ensure we can import sibling package when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from provided_code.utils import load_file
from provided_code.data_shapes import DataShapes

# Same ROI list as DataLoader
ROIS = dict(
    oars=["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"],
    targets=["PTV56", "PTV63", "PTV70"],
)
FULL_ROI_LIST = [*ROIS["oars"], *ROIS["targets"]]
NUM_ROIS = len(FULL_ROI_LIST)

# -------------------- IO helpers -------------------- #

def load_scalar_volume(patient_dir: Path, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Load a scalar volume (ct or dose) from sparse CSV -> (D,H,W) float32."""
    p = patient_dir / f"{name}.csv"
    obj = load_file(p)  # dict with keys "indices" and "data"
    vol = np.zeros(shape, dtype=np.float32)
    flat_idx = obj["indices"].astype(np.int64)
    vals = obj["data"].astype(np.float32)
    np.put(vol, flat_idx, vals)
    return vol

def try_load_mask(patient_dir: Path, roi_name: str, shape: tuple[int, int, int]) -> np.ndarray | None:
    """Try to load single-ROI mask (index-only CSV). Returns (D,H,W) uint8 or None if missing."""
    p = patient_dir / f"{roi_name}.csv"
    if not p.exists():
        return None
    idx = load_file(p).astype(np.int64)
    m = np.zeros(shape, dtype=np.uint8)
    np.put(m, idx, 1)
    return m

def build_multichannel_masks(patient_dir: Path, shape: tuple[int, int, int]) -> tuple[np.ndarray, list[str]]:
    """Stack all ROI masks into (D,H,W,R). Missing ROIs become zero channels. Returns (masks, missing)."""
    masks = np.zeros((*shape, NUM_ROIS), dtype=np.uint8)
    missing = []
    for c, roi in enumerate(FULL_ROI_LIST):
        m = try_load_mask(patient_dir, roi, shape)
        if m is None:
            missing.append(roi)
        else:
            masks[..., c] = m
    return masks, missing

# -------------------- display helpers -------------------- #

def auto_window_ct(ct: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] and normalize to 0..1 for display."""
    p1, p99 = np.percentile(ct, (1, 99))
    ct_clip = np.clip(ct, p1, p99)
    ct_norm = (ct_clip - p1) / max(p99 - p1, 1e-6)
    return ct_norm

def labelmap_from_masks(masks: np.ndarray) -> np.ndarray:
    """
    Convert (D,H,W,R) multi-channel masks to a single label map 0..R (0=background) via argmax.
    """
    bg = np.zeros((*masks.shape[:3], 1), dtype=masks.dtype)
    stacked = np.concatenate([bg, masks], axis=-1)
    labels = np.argmax(stacked, axis=-1).astype(np.int32)
    return labels

def get_plane_slice(vol: np.ndarray, plane: str, index: int) -> np.ndarray:
    """
    Extract a 2D slice from a (D,H,W) volume for a given plane and apply visual rotations for clarity:
      - Axial   → rotate 90° counter-clockwise
      - Sagittal → rotate 180° clockwise
      - Coronal → rotate 90° clockwise
    Output standardized to (H,W) display.
    """
    D, H, W = vol.shape
    plane = plane.lower()
    if plane == "coronal":
        sl = vol[index, :, :]
        sl = np.rot90(sl, k=1)     # 90° clockwise
    elif plane == "axial":
        sl = vol[:, :, index]
        sl = np.rot90(sl, k=4)      # 90° counter-clockwise
    elif plane == "sagittal":
        sl = vol[:, index, :]
        sl = np.fliplr(np.rot90(sl, k=1))      # 180° rotation
    else:
        raise ValueError("plane must be one of: axial, sagittal, coronal")
    return sl


def clamp_slice_index(plane: str, index: int | None, shape: tuple[int, int, int]) -> int:
    """Clamp/choose default slice index per plane."""
    D, H, W = shape
    if plane == "coronal":
        n = D
    elif plane == "axial":
        n = W  # x index corresponds to W
    elif plane == "sagittal":
        n = H  # y index corresponds to H
    else:
        raise ValueError("plane must be one of: axial, sagittal, coronal")
    if index is None:
        return n // 2
    return int(np.clip(index, 0, n - 1))


def plane_extent(plane: str, shape_3d: tuple[int,int,int], voxel_dims: np.ndarray):
    """
    Return (xmin, xmax, ymin, ymax) in mm for the 2D slice of a given plane.
    Works with the (H, W) array you display *after* your rotations.
    """
    D, H, W = shape_3d
    dx, dy, dz = float(voxel_dims[0]), float(voxel_dims[1]), float(voxel_dims[2])

    plane = plane.lower()
    if plane == "axial":
        # H rows (y), W cols (x)
        return (0, W*dx, 0, H*dy)
    elif plane == "sagittal":
        # rows = D (z), cols = W (x)
        return (0, W*dx, 0, D*dz)
    elif plane == "coronal":
        # rows = H (y), cols = W (x)
        return (0, W*dx, 0, H*dy)
    else:
        raise ValueError("plane must be one of: axial, sagittal, coronal")

# -------------------- main -------------------- #

def main(patient_dir: Path, slice_index: int | None):
    # Shapes like DataLoader
    data_shapes = DataShapes(NUM_ROIS)
    ct_shape_3d = tuple(data_shapes.ct[:3])  # (D,H,W)

    # Load data
    voxel_dims = load_file(patient_dir / "voxel_dimensions.csv").astype(np.float32)  # [dx,dy,dz] mm
    ct = load_scalar_volume(patient_dir, "ct", ct_shape_3d)               # (D,H,W)
    dose = load_scalar_volume(patient_dir, "dose", ct_shape_3d)           # (D,H,W)
    masks, missing = build_multichannel_masks(patient_dir, ct_shape_3d)   # (D,H,W,R)

    if missing:
        print(f"[INFO] Missing ROI files (zeroed): {', '.join(missing)}")

    labels = labelmap_from_masks(masks)                                   # (D,H,W)

    # Prepare per-plane slice indices (use same provided --slice across planes, clamped separately)
    D, H, W = ct.shape
    idxs = {
        "axial":    clamp_slice_index("axial",    slice_index, (D, H, W)),
        "sagittal": clamp_slice_index("sagittal", slice_index, (D, H, W)),
        "coronal":  clamp_slice_index("coronal",  slice_index, (D, H, W)),
    }

    # Precompute display assets
    dose_max = float(dose.max()) if dose.size else 1.0
    base_cmap = matplotlib.cm.get_cmap("tab20", NUM_ROIS + 1)
    label_colors = base_cmap(np.arange(NUM_ROIS + 1))

    # Build figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True, dpi=180)
    row_names = ["Axial (x)", "Sagittal (y)", "Coronal (z)"]
    planes = ["axial", "sagittal", "coronal"]

    dose_mappables = []  # to attach a single shared colorbar

    for r, plane in enumerate(planes):
        z = idxs[plane]
        # 2D slices
        ct_sl = get_plane_slice(ct, plane, z)
        dose_sl = get_plane_slice(dose, plane, z)
        labels_sl = get_plane_slice(labels, plane, z)

        # Normalised CT & dose
        ct_disp = auto_window_ct(ct_sl)
        dose_norm_sl = dose_sl / dose_max if dose_max > 0 else dose_sl
        dose_overlay = np.ma.masked_where(dose_norm_sl <= 0, dose_norm_sl)

        extent = plane_extent(plane, (D, H, W), voxel_dims)

        # Column 1: CT
        ax = axes[r, 0]
        im_ct = ax.imshow(ct_disp, cmap="gray", origin="lower")
        ax.set_title(f"{row_names[r]} — CT (slice={z})")
        ax.axis("off")

        # Column 2: CT + Dose
        ax = axes[r, 1]
        ax.imshow(ct_disp, cmap="gray", origin="lower")
        im_dose = ax.imshow(dose_overlay, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{row_names[r]} — CT + Dose (norm by max={dose_max:.2f} Gy)")
        ax.axis("off")
        dose_mappables.append(im_dose)

        # Column 3: Labels
        ax = axes[r, 2]
        ax.imshow(ct_disp, cmap="gray", origin="lower")
        colored = label_colors[labels_sl]  # (H,W,4 RGBA)
        ax.imshow(colored, alpha=0.35, origin="lower")
        ax.set_title(f"{row_names[r]} — Structures")
        ax.axis("off")

    # One shared colorbar for the middle column (dose)
    # Attach to all axes in the middle column so it sits nicely
    cbar = fig.colorbar(dose_mappables[-1], ax=axes[:, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Dose (normalized)")

    # Put the legend only on the bottom-right panel to avoid clutter
    legend_ax = axes[2, 2]
    legend_patches = [matplotlib.patches.Patch(color=label_colors[i+1], label=FULL_ROI_LIST[i]) for i in range(NUM_ROIS)]
    if legend_patches:
        legend_ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=1, framealpha=0.6)

    # Console summary
    print(f"CT shape:      {ct.shape} (D,H,W)")
    print(f"Dose shape:    {dose.shape} (D,H,W)")
    print(f"Masks shape:   {masks.shape} (D,H,W,R) with R={NUM_ROIS}")
    print(f"Voxel spacing: {voxel_dims.tolist()} mm")
    print(f"Slices shown -> Axial(x): {idxs['axial']}, Sagittal(y): {idxs['sagittal']}, Coronal(z): {idxs['coronal']}")

    fig.suptitle(f"OpenKBP Patient Overview — voxel dims [mm]: {voxel_dims.tolist()}", fontsize=14)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_dir", type=Path, required=True, help="Path to pt_* folder")
    ap.add_argument("--slice", type=int, default=None, help="Slice index (applied per-plane with clamping)")
    args = ap.parse_args()
    main(args.patient_dir, args.slice)
