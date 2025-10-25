# provided_code_torch/network_functions.py
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from provided_code.data_loader import DataLoader
from provided_code.utils import get_paths, sparse_vector_function

# If you saved the UNet in provided_code_torch/network_architectures.py:
from provided_code.network_architectures import DoseFromCT3DUNet


class PredictionModel:
    """
    PyTorch version of the Keras PredictionModel.
    Exposes the same public API used by main.py:
      - train_model(...)
      - predict_dose(...)
      - .prediction_dir
    """

    def __init__(self, data_loader: DataLoader, results_patent_path: Path, model_name: str, stage: str) -> None:
        self.data_loader = data_loader
        self.full_roi_list = data_loader.full_roi_list

        # ---- Directories ----
        model_results_path = results_patent_path / model_name
        self.model_dir = model_results_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir = model_results_path / f"{stage}-predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.last_epoch = 200

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Build model ----
        num_mask_channels = len(self.full_roi_list) if self.full_roi_list is not None else 1
        self.model = DoseFromCT3DUNet(
            ct_channels=1,
            mask_channels=num_mask_channels,
            initial_filters=16,  # increase to 64+ for serious runs
            kernel=(4, 4, 4),
            stride=(2, 2, 2),
        ).to(self.device)

        # Optimizer & loss
        self.criterion = nn.L1Loss()
        self.optimizer = Adam(self.model.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ---------- utility ----------
    def _get_generator_path(self, epoch: Optional[int] = None) -> Path:
        epoch = (epoch if epoch is not None else self.current_epoch)
        return self.model_dir / f"epoch_{epoch}.pt"

    def _set_epoch_start(self) -> None:
        all_model_paths = get_paths(self.model_dir, extension="pt")
        for model_path in all_model_paths:
            if "epoch_" in model_path.stem:
                try:
                    epoch_number = int(model_path.stem.split("epoch_")[1])
                    self.current_epoch = max(self.current_epoch, epoch_number)
                except Exception:
                    pass

    def initialize_networks(self) -> None:
        """Load last checkpoint if exists, else keep fresh model."""
        last_path = self._get_generator_path(self.current_epoch) if self.current_epoch > 0 else None
        if last_path and last_path.exists():
            self.model.load_state_dict(torch.load(last_path, map_location=self.device))

    def _save_model_epoch(self, epoch: int) -> None:
        torch.save(self.model.state_dict(), self._get_generator_path(epoch))

    # ---------- training ----------
    def train_model(self, epochs: int = 200, save_frequency: int = 5, keep_model_history: int = 2) -> None:
        """
        Train for `epochs`, saving checkpoints every `save_frequency`.
        Retains up to `keep_model_history` recent checkpoints (older are removed).
        """
        self._set_epoch_start()
        self.last_epoch = epochs
        if self.current_epoch == epochs:
            print(f"The model has already been trained for {epochs} epochs; no further training.")
            return

        self.initialize_networks()
        self.model.train()
        self.data_loader.set_mode("training_model")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            print(f"Beginning epoch {self.current_epoch}")
            self.data_loader.shuffle_data()

            for idx, batch in enumerate(self.data_loader.get_batches()):
                # batch.ct: (N,D,H,W), batch.structure_masks: (N,S,D,H,W), batch.dose: (N,D,H,W)
                ct = torch.from_numpy(batch.ct).float().to(self.device)
                masks = torch.from_numpy(batch.structure_masks).float().to(self.device)
                dose = torch.from_numpy(batch.dose).float().to(self.device)

                # ---- Robust channel handling ----
                # CT: (N,D,H,W,1) → (N,1,D,H,W)
                if ct.ndim == 5 and ct.shape[-1] == 1 and ct.shape[1] != 1:
                    ct = ct.permute(0, 4, 1, 2, 3).contiguous()
                elif ct.ndim == 4:
                    ct = ct.unsqueeze(1)

                # Masks: (N,D,H,W,S) → (N,S,D,H,W)
                if masks.ndim == 5 and masks.shape[-1] <= 64 and masks.shape[1] >= 16:
                    masks = masks.permute(0, 4, 1, 2, 3).contiguous()

                # Dose: fix (N,1,D,H,W,1) or (N,D,H,W,1) → (N,1,D,H,W)
                if dose.ndim == 6 and dose.shape[-1] == 1:
                    dose = dose.squeeze(-1)
                if dose.ndim == 5 and dose.shape[-1] == 1 and dose.shape[1] != 1:
                    dose = dose.permute(0, 4, 1, 2, 3).contiguous()
                elif dose.ndim == 4:
                    dose = dose.unsqueeze(1)

                # Forward + loss
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(ct, masks)

                if pred.shape != dose.shape:
                    print(f"[WARN] pred {pred.shape}, dose {dose.shape} — reshaping dose to match")
                    dose = dose.reshape_as(pred)

                loss = self.criterion(pred, dose)
                loss.backward()
                self.optimizer.step()

                print(f"Model loss at epoch {self.current_epoch} batch {idx} is {loss.item():.3f}")

            # save / prune
            self.manage_model_storage(save_frequency, keep_model_history)

    def manage_model_storage(self, save_frequency: int = 1, keep_model_history: Optional[int] = None) -> None:
        """Save current epoch checkpoint and keep a rolling history."""
        effective_epoch_number = self.current_epoch + 1
        if (effective_epoch_number % max(save_frequency, 1) != 0) and (effective_epoch_number != self.last_epoch):
            return

        # Save current checkpoint
        path = self._get_generator_path(effective_epoch_number)
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

        # Prune old checkpoints
        if keep_model_history and keep_model_history > 0:
            all_paths = sorted(
                get_paths(self.model_dir, extension="pt"),
                key=lambda p: int(p.stem.split("epoch_")[1]),
            )
            to_keep = all_paths[-keep_model_history:]
            for p in all_paths:
                if p not in to_keep:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    # ---------- inference ----------
    def predict_dose(self, epoch: int = 1) -> None:
        """Predict dose for the specified epoch and write CSVs compatible with evaluator."""
        ckpt = self._get_generator_path(epoch)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print("Predicting dose with generator.")
        with torch.no_grad():
            for batch in self.data_loader.get_batches():
                ct = torch.from_numpy(batch.ct).float().to(self.device)
                masks = torch.from_numpy(batch.structure_masks).float().to(self.device)

                # Handle channel formats
                if ct.ndim == 5 and ct.shape[-1] == 1 and ct.shape[1] != 1:
                    ct = ct.permute(0, 4, 1, 2, 3).contiguous()
                elif ct.ndim == 4:
                    ct = ct.unsqueeze(1)

                if masks.ndim == 5 and masks.shape[-1] <= 64 and masks.shape[1] >= 16:
                    masks = masks.permute(0, 4, 1, 2, 3).contiguous()

                pred = self.model(ct, masks)  # (N,1,D,H,W)
                pred_np = pred.squeeze(1).cpu().numpy()  # (N,D,H,W)

                if batch.possible_dose_mask is not None:
                    pred_np = pred_np * batch.possible_dose_mask

                # Save each patient prediction as sparse CSV
                for i, patient_id in enumerate(batch.patient_list):
                    dose_pred_i = pred_np[i]  # (D,H,W)
                    dose_to_save = sparse_vector_function(dose_pred_i)
                    dose_df = pd.DataFrame(
                        data=dose_to_save["data"].squeeze(),
                        index=dose_to_save["indices"].squeeze(),
                        columns=["data"],
                    )
                    out_path = self.prediction_dir / f"{patient_id}.csv"
                    dose_df.to_csv(out_path)
                    print(f"Saved prediction: {out_path}")
