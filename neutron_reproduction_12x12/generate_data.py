import os
import sys
import numpy as np
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm

# Import from local package
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized

def generate_smooth_b_field(im_size, num_samples, minmax_B=5e-3):
    """
    Generate B-fields that look like smooth continuous vector fields.
    """
    A_data = np.zeros((num_samples, im_size, im_size, 3), dtype=np.float32)
    for i in range(num_samples):
        # Generate smaller white noise
        grid = np.random.randn(im_size - 2, im_size - 2, 3)
        # Apply Gaussian blur to make it continuous/smooth
        for c in range(3):
            grid[:, :, c] = ndimage.gaussian_filter(grid[:, :, c], sigma=1.2)
        grid = np.clip(grid * minmax_B * 2, -minmax_B, minmax_B)
        A_data[i, 1:-1, 1:-1, :] = grid
    return A_data

def generate_dataset(num_samples=500, im_size=12):
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    
    nNeutrons = 251
    nAngles = 451
    wavelengths = np.arange(2.0, 8.0, 0.4) * 1e-10
    scaleD = 2.0
    voxel_size = 1e-2 / im_size
    minmax_B = 5e-3
    seed = 42
    np.random.seed(seed)
    
    print(f"1. Precomputing geometry for {im_size}x{im_size} grid...")
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
            # Assumes ray_wrapper is compiled and importable
            vi, vd = ray_wrapper.ray_wrapper(sx, sz, dx, dz, im_size)
            voxel_index[a][n] = vi
            voxel_data[a][n] = vd

    print("2. Generating smooth B-fields...")
    A_data = generate_smooth_b_field(im_size, num_samples, minmax_B)
    
    print("3. Vectorized Forward Model Calculation...")
    shape_B = (num_samples, nNeutrons, nAngles, len(wavelengths), 3, 3)
    v = f'{im_size}x{im_size}'
    B_file = os.path.join(out_dir, f'B_data_{v}.npy')
    
    # Use memmap to avoid RAM crash
    print(f"Creating memory-mapped file: {B_file}")
    B_data = np.lib.format.open_memmap(B_file, mode='w+', dtype='float32', shape=shape_B)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 1
    for i in tqdm(range(0, num_samples, batch_size), desc="Simulation batches"):
        end = min(i + batch_size, num_samples)
        B_batch = A_data[i:end].reshape(end - i, im_size*im_size, 3, order='F')
        results = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                            voxel_index, voxel_data, voxel_size, B_batch, device=device)
        B_data[i:end] = results
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
    generate_dataset()
