import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# Import from local package inside gpu_optimized
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized

def gaussian_kernel_2d(sigma=1.2, kernel_size=7):
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d

def generate_smooth_b_field(im_size, num_samples=1, minmax_B=5e-3):
    """
    CPU-based reference implementation using Scipy.
    Used for verification against the GPU pipeline.
    """
    import scipy.ndimage
    grid_size = im_size - 2
    A_data = np.zeros((num_samples, im_size, im_size, 3))
    
    for i in range(num_samples):
        for c in range(3):
            grid = np.random.randn(grid_size, grid_size)
            smoothed = scipy.ndimage.gaussian_filter(grid, sigma=1.2)
            # Clip and scale
            smoothed = np.clip(smoothed * minmax_B * 2, -minmax_B, minmax_B)
            A_data[i, 1:-1, 1:-1, c] = smoothed
    return A_data

def generate_smooth_b_field_gpu(im_size, num_samples, minmax_B=5e-3, device='cuda'):
    """
    Generate B-fields that look like smooth continuous vector fields, using PyTorch for GPU acceleration.
    """
    grid_size = im_size - 2
    
    # Generate random noise using numpy so it matches CPU version identically for the same seed
    grid_np = np.random.randn(num_samples, grid_size, grid_size, 3).astype(np.float32)
    # Convert to tensor and permute to (num_samples, 3, grid_size, grid_size)
    grid = torch.from_numpy(grid_np).permute(0, 3, 1, 2).to(device)
    
    # Gaussian kernel (sigma=1.2, kernel_size=7 to approximate scipy's default truncate=4.0 * 1.2 = 4.8 ~ 5 or 7)
    kernel_size = 7
    kernel = gaussian_kernel_2d(sigma=1.2, kernel_size=kernel_size).to(device)
    
    # Reshape for depthwise conv2d (groups=3): weight shape (out_channels, in_channels/groups, kH, kW)
    kernel = kernel.unsqueeze(0).unsqueeze(0) # (1, 1, k, k)
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)
    
    # Scipy's default padding mode is 'reflect'
    pad = kernel_size // 2
    grid_padded = F.pad(grid, (pad, pad, pad, pad), mode='reflect')
    
    # Apply convolution
    grid_smoothed = F.conv2d(grid_padded, kernel, groups=3)
    
    # Clip and scale
    grid_smoothed = torch.clamp(grid_smoothed * minmax_B * 2, -minmax_B, minmax_B)
    
    # Place into final A_data tensor (num_samples, 3, im_size, im_size)
    A_data = torch.zeros((num_samples, 3, im_size, im_size), device=device)
    A_data[:, :, 1:-1, 1:-1] = grid_smoothed
    
    # Permute to match original numpy shape (num_samples, im_size, im_size, 3)
    A_data = A_data.permute(0, 2, 3, 1)
    
    return A_data

def generate_dataset(config_path='config1.json', override_seed=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    out_dir = config.get('out_dir', 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    im_size = config['im_size']
    num_samples = config['num_samples']
    nNeutrons = config['nNeutrons']
    nAngles = config['nAngles']
    scaleD = config['scaleD']
    minmax_B = config['minmax_B']
    
    # Use override seed if provided, otherwise use config seed
    seed = override_seed if override_seed is not None else config['seed']
    
    batch_size = config.get('batch_size', 1)
    
    wavelengths = np.arange(config['wavelengths_start'], 
                            config['wavelengths_end'], 
                            config['wavelengths_step']) * 1e-10
                            
    voxel_size = 1e-2 / im_size
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"1. Precomputing geometry for {im_size}x{im_size} grid (CPU)...")
    angles = np.linspace(0, np.pi*2, nAngles, endpoint=False)
    source_xs, source_zs, det_xs, det_zs = sou_det_calc(nNeutrons, im_size, angles, scaleD)
    source_xs = np.reshape(source_xs, (nAngles, nNeutrons), order='F')
    source_zs = np.reshape(source_zs, (nAngles, nNeutrons), order='F')
    det_xs = np.reshape(det_xs, (nAngles, nNeutrons), order='F')
    det_zs = np.reshape(det_zs, (nAngles, nNeutrons), order='F')
    
    voxel_index = np.empty((nAngles, nNeutrons), dtype=object)
    voxel_data = np.empty((nAngles, nNeutrons), dtype=object)
    
    for n in range(nNeutrons):
        for a in range(nAngles):
            sx = source_xs[a][n]; sz = source_zs[a][n]
            dx = det_xs[a][n]; dz = det_zs[a][n]
            vi, vd = ray_wrapper.ray_wrapper(sx, sz, dx, dz, im_size)
            voxel_index[a][n] = vi
            voxel_data[a][n] = vd

    print(f"2. Generating smooth B-fields on {device}...")
    # This runs entirely on the GPU
    A_data_gpu = generate_smooth_b_field_gpu(im_size, num_samples, minmax_B, device=device)
    # We move it to CPU for saving and for the original vectorized forward model loop
    # If the forward model expects CPU arrays, we'll convert it. The current calc_precession_vectorized 
    # internally moves B to GPU, so it's fine if A_data is numpy or torch tensor. 
    # But since it returns numpy, we keep A_data as numpy to save to disk.
    A_data = A_data_gpu.cpu().numpy()
    
    print("3. Vectorized Forward Model Calculation...")
    nW = len(wavelengths)
    shape_B = (num_samples, nW * 9, nAngles, nNeutrons)
    v = f'{im_size}x{im_size}'
    B_file = os.path.join(out_dir, f'B_data_{v}.npy')
    
    print(f"Creating optimized memory-mapped file: {B_file}")
    B_data = np.lib.format.open_memmap(B_file, mode='w+', dtype='float32', shape=shape_B)
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Simulation batches"):
        end = min(i + batch_size, num_samples)
        B_batch = A_data[i:end].reshape(end - i, im_size*im_size, 3, order='F')
        
        # calc_precession_vectorized returns (nbatch, nNeutrons, nAngles, nW, 3, 3)
        results = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                            voxel_index, voxel_data, voxel_size, B_batch, device=device)
        
        # Optimize layout for training: (B, N, A, W, 3, 3) -> (B, W, 3, 3, A, N) -> (B, W*9, A, N)
        # This removes the CPU bottleneck during the training DataLoader loop.
        results_opt = results.transpose(0, 3, 4, 5, 2, 1).reshape(end - i, nW * 9, nAngles, nNeutrons)
        
        B_data[i:end] = results_opt
        if i % 10 == 0:
            B_data.flush()
            
    print("4. Saving metadata and A_data...")
    np.save(os.path.join(out_dir, f'A_data_{v}.npy'), A_data, allow_pickle=False)
    del B_data 

    np.savez_compressed(os.path.join(out_dir, f'meta_{v}.npz'),
                        im_size=im_size, voxel_index=voxel_index, voxel_data=voxel_data,
                        nNeutrons=nNeutrons, nAngles=nAngles, angles=angles,
                        source_xs=source_xs, source_zs=source_zs, det_xs=det_xs, det_zs=det_zs,
                        seed=seed, wavelengths=wavelengths, num_samples=num_samples,
                        voxel_size=voxel_size, minmax_B=minmax_B)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1.json', help='Path to configuration JSON')
    parser.add_argument('--seed', type=int, default=None, help='Override the random seed')
    args = parser.parse_args()
    generate_dataset(config_path=args.config, override_seed=args.seed)
