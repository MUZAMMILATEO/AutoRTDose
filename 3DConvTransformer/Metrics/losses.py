import torch
from torch import nn
import torch.nn.functional as F

class DosePredictionLoss(nn.Module):
    """
    Custom loss function for dose prediction, combining:
    1. Global MSE (L_global)
    2. Hierarchical Weighted MSE for PTVs and OARs (L_ptv, L_oar)
    3. DVH-based Loss (L_dvh) for volumetric constraint matching.
    
    This provides strong spatial accuracy while optimizing for clinical DVH goals.
    The DVH loss now uses a differentiable soft step approximation with vectorized bin calculation.
    """
    def __init__(self, ptv_weight=3.0, oar_weight=1.5, dvh_weight=0.5, num_bins=60, tau=1.0):
        super().__init__()
        
        # Primary loss function (used for L_global and spatial weighted loss)
        self.mse_loss = nn.MSELoss(reduction='none') 
        self.mae_loss = nn.L1Loss() # Used for the DVH loss component
        
        # Hyperparameters
        self.ptv_weight = ptv_weight
        self.oar_weight = oar_weight
        self.dvh_weight = dvh_weight # New weight for DVH component
        self.num_bins = num_bins     # Resolution of the DVH curve
        self.tau = tau               # Smoothing temperature for the soft-step DVH loss
        
        print(f"DosePredictionLoss initialized with PTV Weight: {ptv_weight}, OAR Weight: {oar_weight}, DVH Weight: {dvh_weight}, Tau: {tau}")

    def _calculate_dvh_loss(self, output, target, masks):
        """
        Calculates the Mean Absolute Error (MAE) between predicted and target DVHs 
        for all 10 structures using a vectorized, differentiable soft step function.
        """
        batch_size, _, D, H, W = output.shape
        total_dvh_loss = 0.0
        
        # Find the max dose to define the histogram range (up to 80 Gy, common max dose)
        max_dose = 80.0
        # Histogram bins (dose levels)
        dose_bins = torch.linspace(0.0, max_dose, self.num_bins, device=output.device)

        # We iterate over all 10 structure masks (channels 0 through 9)
        for i in range(10):
            # 1. Isolate the current structure mask (N, 1, D, H, W)
            structure_mask = masks[:, i:i+1, ...]
            
            # Skip if the structure is empty (no voxels)
            if torch.sum(structure_mask).item() < 1:
                continue

            # 2. Extract dose values for the structure (N, V, 1) where V is num voxels
            pred_doses = output[structure_mask.bool()].flatten()
            target_doses = target[structure_mask.bool()].flatten()
            
            # Function to calculate approximate Volume (Cumulative Histogram) - NOW VECTORIZED
            def get_approx_dvh(doses):
                num_doses = len(doses)
                if num_doses == 0:
                    return torch.zeros(self.num_bins, device=output.device)
                
                # Reshape for broadcasting: doses (V, 1), dose_bins (1, B)
                doses_v = doses.view(-1, 1)
                bins_b = dose_bins.view(1, -1)
                
                # Calculate soft step indicator for ALL voxels (V) against ALL bins (B)
                # broadcasted_diff shape: (V, B)
                broadcasted_diff = doses_v - bins_b
                
                # soft_volume_indicator shape: (V, B)
                # Uses the differentiable soft step approximation: sigmoid((doses - bin_val) / self.tau)
                soft_volume_indicator = torch.sigmoid(broadcasted_diff / self.tau)
                
                # Sum over the voxel dimension (dim=0) and normalize (Dose-Volume Histogram curve)
                # dvh_curve shape: (B)
                dvh_curve = torch.sum(soft_volume_indicator, dim=0) / num_doses
                    
                return dvh_curve

            # Calculate predicted and target DVH curves (each is a tensor of size num_bins)
            pred_dvh = get_approx_dvh(pred_doses)
            target_dvh = get_approx_dvh(target_doses)
            
            # 4. Calculate loss as MAE between the two DVH curves
            dvh_loss_structure = self.mae_loss(pred_dvh, target_dvh)
            total_dvh_loss += dvh_loss_structure

        # Average the DVH loss across all structures
        L_dvh_normalized = total_dvh_loss / 10.0 # Divide by 10 (the number of channels)

        return L_dvh_normalized

    def forward(self, output, target, masks):
        """
        Calculates the weighted dose prediction loss using separate PTV and OAR weights 
        and the DVH matching component.
        """
        
        # --- 1. Calculate Base Loss (MSE per voxel) ---
        mse_map = self.mse_loss(output, target)
        
        # --- 2. L_global: Unweighted MSE over the entire volume ---
        L_global = torch.mean(mse_map)
        
        # --- 3. Create Hierarchical Masks (for spatial weighting) ---
        
        # A. PTV Mask (Channels 0, 1, 2)
        ptv_masks = masks[:, 0:3, ...] 
        ptv_mask_union, _ = torch.max(ptv_masks, dim=1, keepdim=True)

        # B. OAR Mask (Channels 3-9)
        oar_masks = masks[:, 3:10, ...]
        oar_mask_union, _ = torch.max(oar_masks, dim=1, keepdim=True)

        # C. OAR-only Mask (OAR union excluding PTV voxels)
        oar_only_mask = oar_mask_union * (1 - ptv_mask_union)
        
        # --- 4. Calculate Normalized Weighted Spatial Losses (L_ptv, L_oar) ---
        
        # L_ptv
        L_ptv_weighted = torch.sum(mse_map * ptv_mask_union) * self.ptv_weight
        num_ptv_voxels = torch.sum(ptv_mask_union) + 1e-6
        L_ptv_normalized = L_ptv_weighted / num_ptv_voxels

        # L_oar
        L_oar_weighted = torch.sum(mse_map * oar_only_mask) * self.oar_weight
        num_oar_voxels = torch.sum(oar_only_mask) + 1e-6
        L_oar_normalized = L_oar_weighted / num_oar_voxels
        
        # --- 5. Calculate DVH-based Loss (L_dvh) ---
        
        # L_dvh is weighted by dvh_weight
        L_dvh = self._calculate_dvh_loss(output, target, masks) * self.dvh_weight

        # --- 6. Total Loss ---
        # L_global covers the background, L_ptv/L_oar prioritize critical regions spatially, 
        # and L_dvh ensures the overall dose volume statistics are met.
        total_loss = L_global + L_ptv_normalized + L_oar_normalized + L_dvh
        
        return total_loss
