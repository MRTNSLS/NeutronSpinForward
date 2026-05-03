import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Add parent directory to path to import original CPU script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_data import generate_smooth_b_field as generate_smooth_b_field_cpu

# Import the GPU version
from generate_data import generate_smooth_b_field_gpu
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized

def main():
    im_size = 12
    num_samples = 1
    minmax_B = 5e-3
    seed = 42
    
    nNeutrons = 256
    nAngles = 360
    scaleD = 2.0
    voxel_size = 1e-2 / im_size
    wavelengths = np.arange(2.0, 8.0, 0.1) * 1e-10

    print("Generating CPU B-field...")
    np.random.seed(seed)
    A_data_cpu = generate_smooth_b_field_cpu(im_size, num_samples, minmax_B)

    print("Generating GPU B-field...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(seed)
    torch.manual_seed(seed)
    A_data_gpu_tensor = generate_smooth_b_field_gpu(im_size, num_samples, minmax_B, device=device)
    A_data_gpu = A_data_gpu_tensor.cpu().numpy()

    print("Precomputing geometry...")
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

    print("\nOpening Interactive Viewer...")
    print("Close the matplotlib window to exit the script.")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    p_labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    mid_w = len(wavelengths) // 2
    sample_idx = 0
    x_axis = np.arange(nNeutrons)
    
    lines_cpu = []
    lines_gpu = []
    
    init_angle = nAngles // 2
    
    for i in range(9):
        cx, cy = i // 3, i % 3
        ax = axes[cx, cy]
        
        slice_cpu = B_data_cpu[sample_idx, :, init_angle, mid_w, cx, cy]
        slice_gpu = B_data_gpu[sample_idx, :, init_angle, mid_w, cx, cy]
        
        l_cpu, = ax.plot(x_axis, slice_cpu, label='CPU Generated', linestyle='-', color='blue', alpha=0.7)
        l_gpu, = ax.plot(x_axis, slice_gpu, label='GPU Generated', linestyle='--', color='red', alpha=0.7)
        
        lines_cpu.append(l_cpu)
        lines_gpu.append(l_gpu)
        
        ax.set_title(f'P_{p_labels[i]}')
        if cx == 2:
            ax.set_xlabel('Neutron Ray Index')
        if cy == 0:
            ax.set_ylabel('Polarization')
        
        if i == 0:
            ax.legend(loc='best')
            
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Interactive Comparison of Synthetic Neutron Data (Angle: {init_angle})", fontsize=16)
    
    # Slider axis
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    angle_slider = Slider(
        ax=ax_slider,
        label='Angle Index',
        valmin=0,
        valmax=nAngles - 1,
        valinit=init_angle,
        valstep=1
    )
    
    def update(val):
        ang_idx = int(angle_slider.val)
        fig.suptitle(f"Interactive Comparison of Synthetic Neutron Data (Angle: {ang_idx})", fontsize=16)
        
        for i in range(9):
            cx, cy = i // 3, i % 3
            
            slice_cpu = B_data_cpu[sample_idx, :, ang_idx, mid_w, cx, cy]
            slice_gpu = B_data_gpu[sample_idx, :, ang_idx, mid_w, cx, cy]
            
            lines_cpu[i].set_ydata(slice_cpu)
            lines_gpu[i].set_ydata(slice_gpu)
            
            # Rescale the y-axis to fit the new data
            axes[cx, cy].relim()
            axes[cx, cy].autoscale_view(scalex=False, scaley=True)
            
        fig.canvas.draw_idle()
        
    angle_slider.on_changed(update)
    
    plt.show()

if __name__ == '__main__':
    main()
