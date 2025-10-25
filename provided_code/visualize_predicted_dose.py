#!/usr/bin/env python3
# visualize_predicted_dose.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def _safe_load_file_csv(path: Path):
    txt = path.read_text().strip().splitlines()
    rows = []
    for line in txt:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            a = int(parts[0].strip()); b = float(parts[1].strip())
            rows.append((a, b))
        except ValueError:
            continue
    if rows:
        arr = np.array(rows, dtype=np.float64)
        return {"indices": arr[:, 0].astype(np.int64), "data": arr[:, 1].astype(np.float32)}
    arr = np.genfromtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return {"indices": arr[:, 0].astype(np.int64), "data": arr[:, 1].astype(np.float32)}
    raise RuntimeError(f"Could not parse CSV: {path}")

def _load_sparse_volume(csv_path: Path, shape: tuple[int,int,int]) -> np.ndarray:
    obj = _safe_load_file_csv(csv_path)
    vol = np.zeros(shape, dtype=np.float32)
    idx = obj["indices"].astype(np.int64)
    vals = obj["data"].astype(np.float32)
    flat_size = int(np.prod(shape))
    m = (idx >= 0) & (idx < flat_size)
    np.put(vol, idx[m], vals[m])
    return vol

def auto_window_ct(ct: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(ct, (1, 99))
    ct = np.clip(ct, p1, p99)
    return (ct - p1) / max(p99 - p1, 1e-6)

def get_plane_slice(vol: np.ndarray, plane: str, index: int) -> np.ndarray:
    D, H, W = vol.shape
    plane = plane.lower()
    if plane == "coronal":
        sl = vol[index, :, :];  sl = np.rot90(sl, k=1)
    elif plane == "axial":
        sl = vol[:, :, index];  sl = np.rot90(sl, k=4)
    elif plane == "sagittal":
        sl = vol[:, index, :];  sl = np.fliplr(np.rot90(sl, k=1))
    else:
        raise ValueError("plane must be one of: axial, sagittal, coronal")
    return sl

def clamp_slice_index(plane: str, index: int | None, shape: tuple[int,int,int]) -> int:
    D, H, W = shape
    n = {"coronal": D, "axial": W, "sagittal": H}[plane]
    return n // 2 if index is None else int(np.clip(index, 0, n - 1))

def per_slice_scale(img: np.ndarray, mode: str, pct_low: float, pct_high: float) -> tuple[np.ndarray, float, float]:
    x = img
    if mode == "linear":
        return x, 0.0, 1.0
    nz = x[x > 0]
    if nz.size == 0:
        return x, 0.0, 1.0
    if mode == "percentile":
        vmin = float(np.percentile(nz, pct_low))
        vmax = float(np.percentile(nz, pct_high))
        if vmax <= vmin: vmax = vmin + 1e-6
        x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
        return x, vmin, vmax
    if mode == "log":
        # log-ish emphasis of small values
        x = x / (nz.max() if nz.size else 1.0)
        x = np.log1p(20 * x) / np.log1p(20)
        return x, 0.0, 1.0
    return x, 0.0, 1.0

def main(csv: Path, shape, slice_index, overlay_csv: Path | None, contrast: str, pct_low: float, pct_high: float):
    D,H,W = shape
    dose = _load_sparse_volume(csv, shape)
    dose_max = float(dose.max()) if dose.size else 1.0
    nonzero = int((dose > 0).sum())
    print(f"[INFO] dose stats — shape={dose.shape}, min={dose.min():.6f}, max={dose.max():.6f}, "
          f"nonzero={nonzero} ({nonzero/ dose.size*100:.2f}%)")

    ct = None
    if overlay_csv:
        try:
            ct = _load_sparse_volume(overlay_csv, shape)
            print(f"[INFO] loaded CT overlay: {overlay_csv}")
        except Exception as e:
            print(f"[WARN] CT overlay failed: {e}")

    planes = ["axial", "sagittal", "coronal"]
    idxs = {p: clamp_slice_index(p, slice_index, (D,H,W)) for p in planes}

    ncols = 2 if ct is not None else 1
    fig, axes = plt.subplots(3, ncols, figsize=(12, 12), dpi=180, constrained_layout=True)
    if ncols == 1:
        axes = axes[:, None]

    for r, p in enumerate(planes):
        z = idxs[p]
        dose_sl = get_plane_slice(dose, p, z)
        # normalise by global max to 0..1 first
        dose_norm = dose_sl / (dose_max if dose_max > 0 else 1.0)
        # then apply contrast mode
        disp, vmin_used, vmax_used = per_slice_scale(dose_norm, contrast, pct_low, pct_high)

        ax = axes[r, 0]
        im = ax.imshow(disp, cmap="jet", origin="lower", vmin=0, vmax=1, interpolation="bicubic", aspect='auto')
        ax.set_title(f"{p.capitalize()} — Pred dose ({contrast}"
                     f"{'' if contrast=='linear' else f' {pct_low}/{pct_high} pct'})")
        ax.axis("off")

        if ct is not None:
            ct_sl = get_plane_slice(ct, p, z)
            ct_disp = auto_window_ct(ct_sl)
            ax2 = axes[r, 1]
            ax2.imshow(ct_disp, cmap="gray", origin="lower", interpolation="bicubic", aspect='auto')
            ax2.imshow(disp, cmap="jet", origin="lower", alpha=0.5, vmin=0, vmax=1,
                       interpolation="bicubic", aspect='auto')
            ax2.set_title(f"{p.capitalize()} — CT + Pred dose ({contrast})")
            ax2.axis("off")

        print(f"[INFO] {p:8s} slice={z:3d}  disp-range=[{disp.min():.4f}, {disp.max():.4f}]"
              f"{'' if contrast=='linear' else f' from vmin~{vmin_used:.4f}, vmax~{vmax_used:.4f}'}")

    cbar = fig.colorbar(im, ax=axes[:,0], fraction=0.046, pad=0.04)
    cbar.set_label("Display (post-contrast)")

    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Predicted dose CSV (index,value)")
    ap.add_argument("--shape", type=int, nargs=3, default=(128,128,128), metavar=("D","H","W"))
    ap.add_argument("--slice", type=int, default=None, help="Slice index to view (auto-clamped)")
    ap.add_argument("--overlay_ct", type=Path, default=None, help="Optional CT CSV to overlay (e.g., pt_205/ct.csv)")
    ap.add_argument("--contrast", type=str, choices=["linear","percentile","log"], default="percentile",
                    help="Display contrast mode")
    ap.add_argument("--pct_low", type=float, default=5.0, help="Lower percentile for 'percentile' mode")
    ap.add_argument("--pct_high", type=float, default=99.0, help="Upper percentile for 'percentile' mode")
    args = ap.parse_args()
    main(args.csv, tuple(args.shape), args.slice, args.overlay_ct, args.contrast, args.pct_low, args.pct_high)
