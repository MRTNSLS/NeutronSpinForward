import numpy as np
import torch

def yrot_quat(theta):
    # This is for the original numpy logic validation if needed
    import quaternion # Requires numpy-quaternion
    return quaternion.quaternion(np.cos(theta*0.5),0,np.sin(theta*0.5),0)

def yrot_matrix(theta_tensor):
    """
    Returns rotation matrix for rotation around Y axis by theta.
    theta_tensor: (N,)
    returns: (N, 3, 3)
    """
    cos_t = torch.cos(theta_tensor)
    sin_t = torch.sin(theta_tensor)
    zeros = torch.zeros_like(theta_tensor)
    ones = torch.ones_like(theta_tensor)
    
    R = torch.stack([
        torch.stack([cos_t, zeros, sin_t], dim=-1),
        torch.stack([zeros, ones,  zeros], dim=-1),
        torch.stack([-sin_t,zeros, cos_t], dim=-1)
    ], dim=-2)
    return R

def rodrigues_rotation_matrix(k, theta):
    """
    k: (..., 3) unit vectors
    theta: (...) angles
    Returns: (..., 3, 3) rotation matrices
    """
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1)
    ], dim=-2)
    
    K2 = torch.matmul(K, K)
    I = torch.eye(3, device=k.device, dtype=k.dtype).expand_as(K)
    
    sin_t = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
    
    R = I + sin_t * K + (1 - cos_t) * K2
    return R

def calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths, 
                               voxel_index, voxel_data, voxel_size, B, device='cpu'):
    """
    Optimized, fully vectorized version of calc_precession.
    Returns: (nNeutrons, nAngles, nWavelengths, 3, 3)
    """
    B_tensor = torch.tensor(B, dtype=torch.float32, device=device) # (N, 3)
    angles_tensor = torch.tensor(angles, dtype=torch.float32, device=device) # (A,)
    waves_tensor = torch.tensor(wavelengths, dtype=torch.float32, device=device) # (W,)
    
    nW = len(wavelengths)
    c = 4.632e14
    
    # 1. Rotate B fields for all angles
    # Br = B rotated by -theta
    R_y = yrot_matrix(-angles_tensor) # (A, 3, 3)
    
    # Check if B is batched: (Batch, N, 3) or (N, 3)
    if B_tensor.dim() == 3:
        # Batched B: (Batch, N, 3)
        # R_y @ B = (A, 3, 3) @ (Batch, N, 3)?? No.
        # We want Br of shape (Batch, A, N, 3)
        # Br[b, a, n, i] = sum_j R_y[a, i, j] * B[b, n, j]
        Br = torch.einsum('aij,bnj->bani', R_y, B_tensor)
        nbatch = B_tensor.size(0)
    else:
        # Single B: (N, 3)
        Br = torch.einsum('aij,nj->ani', R_y, B_tensor).unsqueeze(0) # (1, A, N, 3)
        nbatch = 1
    
    # 2. Pad voxel indices and data to create rectangular tensors
    max_len = 0
    for a in range(nAngles):
        for n in range(nNeutrons):
            max_len = max(max_len, len(voxel_index[a][n]))
            
    padded_vi = np.zeros((nAngles, nNeutrons, max_len), dtype=np.int64)
    padded_L = np.zeros((nAngles, nNeutrons, max_len), dtype=np.float32)
    
    for a in range(nAngles):
        for n in range(nNeutrons):
            v = voxel_index[a][n]
            lv = voxel_data[a][n]
            padded_vi[a, n, :len(v)] = v
            padded_L[a, n, :len(lv)] = lv
            
    padded_vi = torch.tensor(padded_vi, device=device) # (A, nN, max_len)
    padded_L = torch.tensor(padded_L, device=device) # (A, nN, max_len)
    
    # 3. Gather B fields along paths
    # Br has shape (nbatch, A, N, 3)
    # We want Bs of shape (nbatch, A, nN, max_len, 3)
    A_idx = torch.arange(nAngles, device=device).view(1, -1, 1, 1).expand(nbatch, -1, nNeutrons, max_len)
    # padded_vi has shape (A, nN, max_len). We need to expand it for nbatch.
    VI_idx = padded_vi.unsqueeze(0).expand(nbatch, -1, -1, -1)
    
    # We need a Batch index too
    B_idx = torch.arange(nbatch, device=device).view(-1, 1, 1, 1).expand(-1, nAngles, nNeutrons, max_len)
    
    Bs = Br[B_idx, A_idx, VI_idx] # (nbatch, A, nN, max_len, 3)
    
    # 4. Successive rotations calculation
    # P shape: (nbatch, A, nN, nW, 3, 3)
    P = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 1, 1, 3, 3).expand(nbatch, nAngles, nNeutrons, nW, 3, 3).contiguous()
    
    # We process each step through the grid to avoid allocating the massive R_steps tensor
    for step in range(max_len):
        # Extract data for this step
        Bs_step = Bs[:, :, :, step, :] # (nbatch, A, nN, 3)
        L_step = padded_L[:, :, step]  # (A, nN)
        
        # Avoid processing zero-length steps
        if torch.max(L_step) < 1e-12:
            continue
            
        B_norm_step = torch.norm(Bs_step, dim=-1) # (nbatch, A, nN)
        
        # Mask for non-zero B field
        mask = B_norm_step > 1e-12
        k_step = torch.zeros_like(Bs_step)
        if mask.any():
            k_step[mask] = Bs_step[mask] / B_norm_step[mask].unsqueeze(-1)
        
        # theta shape: (nbatch, A, nN, W)
        # theta = c * lambda * B * L * voxel_size
        theta_step = (c * voxel_size * waves_tensor).view(1, 1, 1, nW) * \
                     B_norm_step.unsqueeze(-1) * L_step.unsqueeze(0).unsqueeze(-1)
        
        # Expand k to match W: (nbatch, A, nN, 1, 3) -> (nbatch, A, nN, W, 3)
        k_step_exp = k_step.unsqueeze(-2).expand(-1, -1, -1, nW, -1)
        
        # Get rotation matrices for this step: (nbatch, A, nN, W, 3, 3)
        R_step = rodrigues_rotation_matrix(k_step_exp, theta_step)
        
        # Update precession matrices
        P = torch.matmul(P, R_step)
        
    # Rearrange to (nbatch, nNeutrons, nAngles, nW, 3, 3) to match original format
    out = P.permute(0, 2, 1, 3, 4, 5).transpose(-1, -2).cpu().numpy()
    
    if B_tensor.dim() == 2:
        return out[0] # Return (nN, A, nW, 3, 3) for single sample consistency
    return out

