import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import original CPU script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_data import generate_smooth_b_field as generate_smooth_b_field_cpu

# Import the GPU version
from generate_data import generate_smooth_b_field_gpu
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized

def main():
    im_size = 12
    num_samples = 2
    minmax_B = 5e-3
    seed = 42
    
    nNeutrons = 256
    nAngles = 360
    scaleD = 2.0
    voxel_size = 1e-2 / im_size
    wavelengths = np.arange(2.0, 8.0, 0.1) * 1e-10

    print("Generating CPU data...")
    np.random.seed(seed)
    A_data_cpu = generate_smooth_b_field_cpu(im_size, num_samples, minmax_B)

    print("Generating GPU data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(seed)
    torch.manual_seed(seed)
    A_data_gpu_tensor = generate_smooth_b_field_gpu(im_size, num_samples, minmax_B, device=device)
    A_data_gpu = A_data_gpu_tensor.cpu().numpy()

    # Compare B-fields
    diff = np.abs(A_data_cpu - A_data_gpu)
    max_diff = np.max(diff)
    print(f"Max absolute difference between CPU and GPU generated B-fields: {max_diff:.8e}")

    # Forward model calculations
    print("\nPrecomputing geometry for forward physics...")
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

    print("Calculating CPU synthetic neutron data...")
    B_batch_cpu = A_data_cpu.reshape(num_samples, im_size*im_size, 3, order='F')
    B_data_cpu = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_cpu, device=device)

    print("Calculating GPU synthetic neutron data...")
    B_batch_gpu = A_data_gpu.reshape(num_samples, im_size*im_size, 3, order='F')
    B_data_gpu = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_gpu, device=device)
    
    # Compare Neutron Data
    diff_neutron = np.abs(B_data_cpu - B_data_gpu)
    max_diff_neutron = np.max(diff_neutron)
    print(f"Max absolute difference between CPU and GPU generated synthetic neutron data: {max_diff_neutron:.8e}")

    # Plot comparisons
    os.makedirs('results', exist_ok=True)
    sample_idx = 0
    row_idx = im_size // 2
    component_idx = 0 # Bx
    
    # Plot 1: 1D Slice of B-field
    print("Generating 1D slice comparison plot for B-field...")
    x_axis = np.arange(im_size)
    slice_cpu = A_data_cpu[sample_idx, row_idx, :, component_idx]
    slice_gpu = A_data_gpu[sample_idx, row_idx, :, component_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, slice_cpu, label='CPU (SciPy Gaussian Filter)', marker='o', linestyle='-', linewidth=2)
    ax.plot(x_axis, slice_gpu, label='GPU (PyTorch Conv2D)', marker='x', linestyle='--', linewidth=2)
    ax.set_title(f'1D Slice Comparison of Generated B-field (Sample {sample_idx}, Row {row_idx}, Component Bx)')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Magnetic Field (T)')
    ax.legend()
    ax.grid(True)
    
    plot_path_b = 'results/compare_1d_slice.png'
    fig.savefig(plot_path_b)
    plt.close(fig)
    print(f"B-field plot saved to {plot_path_b}")

    # Plot 2: 1D Slice of Sinogram (Synthetic Neutron Data)
    print("Generating 1D slice comparison plot for Synthetic Neutron Data...")
    # B_data shape is (num_samples, nNeutrons, nAngles, nWavelengths, 3, 3)
    # Let's plot Pxx at the middle wavelength for a specific angle
    mid_w = len(wavelengths) // 2
    angle_idx = nAngles // 2
    component_x, component_y = 0, 0 # Pxx
    
    x_axis_neutron = np.arange(nNeutrons)
    sino_slice_cpu = B_data_cpu[sample_idx, :, angle_idx, mid_w, component_x, component_y]
    sino_slice_gpu = B_data_gpu[sample_idx, :, angle_idx, mid_w, component_x, component_y]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_axis_neutron, sino_slice_cpu, label='CPU Generated Source', marker='o', linestyle='-', linewidth=2, markersize=3)
    ax2.plot(x_axis_neutron, sino_slice_gpu, label='GPU Generated Source', marker='x', linestyle='--', linewidth=2, markersize=3)
    ax2.set_title(f'1D Slice Comparison of Synthetic Neutron Data (Pxx, Angle {angle_idx})')
    ax2.set_xlabel('Neutron Ray Index')
    ax2.set_ylabel('Polarization Value')
    ax2.legend()
    ax2.grid(True)
    
    plot_path_sino = 'results/compare_neutron_data.png'
    fig2.savefig(plot_path_sino)
    plt.close(fig2)
    print(f"Neutron data plot saved to {plot_path_sino}")

if __name__ == '__main__':
    main()
