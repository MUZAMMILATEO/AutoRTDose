import torch
from torch import nn
import torch.nn.functional as F

class DosePredictionLoss(nn.Module):
    """
    Custom loss function for dose prediction, combining global MSE with 
    weighted MSE within critical structures (PTV and OARs).
    
    The loss accepts the 10-channel structure masks for region weighting.
    """
    def __init__(self, ptv_weight=2.0, oar_weight=1.5):
        super().__init__()
        # Primary loss function
        self.mse_loss = nn.MSELoss(reduction='none') # Use reduction='none' for weighted loss
        
        # Weights for critical structures
        self.ptv_weight = ptv_weight
        self.oar_weight = oar_weight
        
        print(f"DosePredictionLoss initialized with PTV Weight: {ptv_weight}, OAR Weight: {oar_weight}")

    def forward(self, output, target, masks):
        """
        Calculates the weighted dose prediction loss.
        
        Args:
            output (Tensor): Predicted dose (N, 1, D, H, W)
            target (Tensor): Ground truth dose (N, 1, D, H, W)
            masks (Tensor): 10 structure masks (N, 10, D, H, W)
        
        Returns:
            Tensor: Total weighted loss scalar.
        """
        
        # --- 1. Calculate Base Loss (MSE per voxel) ---
        # mse_map shape: (N, 1, D, H, W)
        mse_map = self.mse_loss(output, target)
        
        # --- 2. Extract Critical Masks ---
        # Assuming PTV is the 0th channel and a critical OAR is the 1st channel
        # We need to unsqueeze the channel dimension (1) to match the mse_map shape (N, 1, D, H, W)
        
        # PTV Mask (N, 1, D, H, W)
        ptv_mask = masks[:, 0:1, ...] 
        
        # Critical OAR Mask (N, 1, D, H, W)
        oar_mask = masks[:, 1:2, ...]
        
        # --- 3. Calculate Weighted Losses ---
        
        # L_global: Unweighted MSE over the entire volume
        L_global = torch.mean(mse_map)
        
        # L_ptv: Loss only inside the PTV, heavily weighted
        # Multiply MSE map by PTV mask to zero out voxels outside the PTV
        L_ptv_weighted = torch.sum(mse_map * ptv_mask) * self.ptv_weight
        # Normalize by the number of PTV voxels (plus a small epsilon to avoid division by zero)
        num_ptv_voxels = torch.sum(ptv_mask) + 1e-6
        L_ptv = L_ptv_weighted / num_ptv_voxels
        
        # L_oar: Loss only inside the critical OAR, moderately weighted
        # Multiply MSE map by OAR mask to zero out voxels outside the OAR
        L_oar_weighted = torch.sum(mse_map * oar_mask) * self.oar_weight
        # Normalize by the number of OAR voxels
        num_oar_voxels = torch.sum(oar_mask) + 1e-6
        L_oar = L_oar_weighted / num_oar_voxels
        
        # --- 4. Total Loss ---
        total_loss = L_global + L_ptv + L_oar
        
        return total_loss
