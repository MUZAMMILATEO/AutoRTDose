import torch
from torch import nn
import torch.nn.functional as F

class DosePredictionLoss(nn.Module):
    """
    Custom loss function for dose prediction, combining:
    1. Global MSE (L_global)
    2. Hierarchical Weighted MSE for PTVs and OARs (L_ptv, L_oar)
    3. DVH-based Loss (L_dvh) for volumetric constraint matching (LEVEL MATCHING).
    4. NEW: DVH Gradient Loss (L_dvh_grad) for DVH curve shape matching.
    5. Gradient Matching Loss (L_gradient) for dose sharpness (SPATIAL GRADIENT).
    """
    def __init__(self, ptv_weight=0.05, oar_weight=0.2, dvh_weight=50.0, 
                 dvh_grad_weight=200.0, # <-- NEW HYPERPARAMETER for DVH gradient
                 gradient_weight=500.0, 
                 num_bins=60, tau=1.0):
        super().__init__()
        
        # Primary loss function (used for L_global and spatial weighted loss)
        self.mse_loss = nn.MSELoss(reduction='none') 
        self.mae_loss = nn.L1Loss() # Used for the DVH and Gradient loss components
        
        # Hyperparameters
        self.ptv_weight = ptv_weight
        self.oar_weight = oar_weight
        self.dvh_weight = dvh_weight 
        self.dvh_grad_weight = dvh_grad_weight # <-- NEW
        self.gradient_weight = gradient_weight 
        self.num_bins = num_bins 
        self.tau = tau 
        
        print(f"DosePredictionLoss initialized with PTV Weight: {ptv_weight}, OAR Weight: {oar_weight}, DVH Weight: {dvh_weight}, DVH Grad Weight: {dvh_grad_weight}, Gradient Weight: {gradient_weight}, Tau: {tau}")

    def _calculate_dvh_loss(self, output, target, masks):
        """
        Calculates the Mean Absolute Error (MAE) for both the DVH curve level and its gradient.
        Returns: L_dvh_normalized (level matching), L_dvh_grad_normalized (shape matching)
        """
        batch_size, _, D, H, W = output.shape
        total_dvh_loss = 0.0
        total_dvh_grad_loss = 0.0 # <-- NEW
        
        max_dose = 80.0
        dose_bins = torch.linspace(0.0, max_dose, self.num_bins, device=output.device)

        for i in range(10):
            structure_mask = masks[:, i:i+1, ...]
            
            if torch.sum(structure_mask).item() < 1:
                continue

            pred_doses = output[structure_mask.bool()].flatten()
            target_doses = target[structure_mask.bool()].flatten()
            
            def get_approx_dvh(doses):
                num_doses = len(doses)
                if num_doses == 0:
                    # Return all zeros, but ensure it has the correct number of bins
                    return torch.zeros(self.num_bins, device=output.device) 
                
                doses_v = doses.view(-1, 1)
                bins_b = dose_bins.view(1, -1)
                
                broadcasted_diff = doses_v - bins_b
                soft_volume_indicator = torch.sigmoid(broadcasted_diff / self.tau)
                dvh_curve = torch.sum(soft_volume_indicator, dim=0) / num_doses
                return dvh_curve

            pred_dvh = get_approx_dvh(pred_doses)
            target_dvh = get_approx_dvh(target_doses)
            
            # 1. DVH Level Loss (MAE)
            dvh_loss_structure = self.mae_loss(pred_dvh, target_dvh)
            total_dvh_loss += dvh_loss_structure
            
            # 2. DVH Gradient Loss (Shape Matching) <-- NEW
            # torch.diff calculates the finite difference (slope)
            # The length of the gradient is num_bins - 1
            pred_dvh_grad = torch.diff(pred_dvh) 
            target_dvh_grad = torch.diff(target_dvh)
            
            dvh_grad_loss_structure = self.mae_loss(pred_dvh_grad, target_dvh_grad)
            total_dvh_grad_loss += dvh_grad_loss_structure

        L_dvh_normalized = total_dvh_loss / 10.0
        L_dvh_grad_normalized = total_dvh_grad_loss / 10.0 # Normalize by the number of structures (10)

        return L_dvh_normalized, L_dvh_grad_normalized # <-- TWO RETURNS

    def _calculate_gradient_loss(self, output, target):
        """
        Calculates L1 loss on the first-order gradients (dose sharpness) 
        using simple 3D finite difference kernels.
        """
        # Note: Input/Output are (N, C=1, D, H, W)
        
        # 1. Define 1D finite difference kernels for D, H, W axes
        k = torch.tensor([-1., 1.], dtype=output.dtype, device=output.device).view(1, 1, 2)
        
        kernel_D = k.view(1, 1, 2, 1, 1)
        kernel_H = k.view(1, 1, 1, 2, 1)
        kernel_W = k.view(1, 1, 1, 1, 2)

        # 2. Calculate Gradients for Prediction and Target
        grad_D_pred = F.conv3d(output, kernel_D)
        grad_D_target = F.conv3d(target, kernel_D)
        
        grad_H_pred = F.conv3d(output, kernel_H)
        grad_H_target = F.conv3d(target, kernel_H)
        
        grad_W_pred = F.conv3d(output, kernel_W)
        grad_W_target = F.conv3d(target, kernel_W)

        # 3. Calculate L1 Loss on the magnitude of the gradients for each axis
        loss_D = self.mae_loss(torch.abs(grad_D_pred), torch.abs(grad_D_target))
        loss_H = self.mae_loss(torch.abs(grad_H_pred), torch.abs(grad_H_target))
        loss_W = self.mae_loss(torch.abs(grad_W_pred), torch.abs(grad_W_target))

        # 4. Total Gradient Loss (Averaged over 3 axes)
        L_gradient = (loss_D + loss_H + loss_W) / 3.0
        
        return L_gradient

    def forward(self, output, target, masks):
        """
        Calculates the full weighted dose prediction loss.
        """
        
        # --- 1. Calculate Base Loss (MSE per voxel) ---
        mse_map = self.mse_loss(output, target)
        L_global = torch.mean(mse_map)
        
        # --- 2. Calculate Hierarchical Weighted Spatial Losses (L_ptv, L_oar) ---
        ptv_masks = masks[:, 0:3, ...] 
        ptv_mask_union, _ = torch.max(ptv_masks, dim=1, keepdim=True)
        
        oar_masks = masks[:, 3:10, ...]
        oar_mask_union, _ = torch.max(oar_masks, dim=1, keepdim=True)
        oar_only_mask = oar_mask_union * (1 - ptv_mask_union)

        # L_ptv
        L_ptv_weighted = torch.sum(mse_map * ptv_mask_union) * self.ptv_weight
        num_ptv_voxels = torch.sum(ptv_mask_union) + 1e-6
        L_ptv_normalized = L_ptv_weighted / num_ptv_voxels

        # L_oar
        L_oar_weighted = torch.sum(mse_map * oar_only_mask) * self.oar_weight
        num_oar_voxels = torch.sum(oar_only_mask) + 1e-6
        L_oar_normalized = L_oar_weighted / num_oar_voxels
        
        # --- 3. Calculate DVH-based Losses (L_dvh_level, L_dvh_grad) ---
        L_dvh_base, L_dvh_grad_base = self._calculate_dvh_loss(output, target, masks) # <-- Captures TWO terms

        L_dvh_level = L_dvh_base * self.dvh_weight
        L_dvh_grad = L_dvh_grad_base * self.dvh_grad_weight # <-- NEW WEIGHTED TERM

        # --- 4. Calculate Gradient Matching Loss (L_gradient) ---
        L_gradient = self._calculate_gradient_loss(output, target) * self.gradient_weight
        
        # --- 5. Total Loss ---
        total_loss = L_global + L_ptv_normalized + L_oar_normalized + L_dvh_level + L_dvh_grad + L_gradient # <-- L_dvh_grad included
    
        # Collect all contributions
        # contributions = {
        #     "L_global": L_global,
        #     "L_ptv_normalized": L_ptv_normalized,
        #     "L_oar_normalized": L_oar_normalized,
        #     "L_dvh_level": L_dvh_level,
        #     "L_dvh_grad": L_dvh_grad,
        #     "L_gradient": L_gradient
        # }

        # total_loss_value = total_loss.item()

        # print("\n--- Loss Contributions (%) ---")
        # for name, loss_tensor in contributions.items():
        #     # Convert tensor value to float and calculate percentage
        #     percentage = (loss_tensor.item() / 1.0) * 1.0
        #     print(f"  {name}: {percentage:.2f}")
        # print("------------------------------\n")

        return total_loss