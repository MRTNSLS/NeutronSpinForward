import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports from gpu_optimized structure
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized
from reproduce_neutron.model import SpinToBNet
from generate_data import generate_smooth_b_field_gpu

def get_xb_from_B(B_data):
    """
    Converts B_data shape (1, nN, nA, nW, 3, 3) to model input shape (1, nW*9, nA, nN)
    """
    # Transpose to (1, nW, 3, 3, nA, nN) -> indices (0, 3, 4, 5, 2, 1)
    b_reshaped = np.transpose(B_data, (0, 3, 4, 5, 2, 1))
    
    nW = b_reshaped.shape[1]
    nA = b_reshaped.shape[4]
    nN = b_reshaped.shape[5]
    
    xb = b_reshaped.reshape(1, nW * 9, nA, nN).astype(np.float32)
    return torch.from_numpy(xb)

def run_reconstruction(args):
    # 0. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: {args.config}")
    
    im_size = config.get('im_size', 12)
    nNeutrons = config.get('nNeutrons', 256)
    nAngles = config.get('nAngles', 360)
    scaleD = config.get('scaleD', 2.0)
    minmax_B = config.get('minmax_B', 0.005)
    seed = args.seed
    
    wavelengths = np.arange(config['wavelengths_start'], 
                            config['wavelengths_end'], 
                            config['wavelengths_step']) * 1e-10
    nW = len(wavelengths)
    voxel_size = 1e-2 / im_size
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n--- 1. Generating Ground Truth (GPU) ---")
    A_true_tensor = generate_smooth_b_field_gpu(im_size, num_samples=1, minmax_B=minmax_B, device=device)
    A_true_np = A_true_tensor.cpu().numpy()
    
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

    print("\n--- 2. Forward Physics Simulation ---")
    B_batch_true = A_true_np.reshape(1, im_size*im_size, 3, order='F')
    B_true_raw = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_true, device=device)
    
    # Format to tensor
    xb_true = get_xb_from_B(B_true_raw)
    
    print("\n--- 3. Adding Noise ---")
    xb_noisy = xb_true.clone()
    if args.noise_level > 0:
        std_dev = xb_true.std() * args.noise_level
        xb_noisy += torch.randn_like(xb_true) * std_dev
        print(f"Added noise level: {args.noise_level}")
        
    print("\n--- 4. Reconstruction with Model ---")
    print(f"Loading model: {args.model_path}")
    model = SpinToBNet(in_ch=nW*9, out_ch=3, nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, 
                       hidden_dim=config.get('hidden_dim', 64),
                       pool_res_h=config.get('pool_res_h', 18),
                       pool_res_w=config.get('pool_res_w', 6),
                       legacy=config.get('legacy', False)).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    model.eval()
    with torch.no_grad():
        pred_a = model(xb_noisy.to(device)).cpu()
        # Denormalize output
        A_pred_np = pred_a.numpy().transpose(0, 2, 3, 1) * minmax_B 
        
    # 5. Generating Detailed Plots
    print("\n--- 5. Generating Detailed Plots ---")
    os.makedirs('results', exist_ok=True)
    p_labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    file_suffix = f"_{cfg_name}_s{seed}"
    
    # Plot 1: Sinogram Comparison
    print("Plotting Sinogram Comparison...")
    # Get xb_pred by running forward model on A_pred
    B_batch_pred = A_pred_np.reshape(1, im_size*im_size, 3, order='F')
    B_pred_raw = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_pred, device=device)
    xb_pred = get_xb_from_B(B_pred_raw)
    
    mid_w = nW // 2
    fig_sino, axes_sino = plt.subplots(3, 6, figsize=(24, 12))
    for i in range(9):
        sino_idx = mid_w * 9 + i
        # True Sinogram
        ax_true = axes_sino[i // 3, (i % 3) * 2]
        im_true = ax_true.imshow(xb_true[0, sino_idx].numpy(), aspect='auto', cmap='viridis')
        ax_true.set_title(f"True P_{p_labels[i]}")
        fig_sino.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
        # Pred Sinogram
        ax_pred = axes_sino[i // 3, (i % 3) * 2 + 1]
        im_pred = ax_pred.imshow(xb_pred[0, sino_idx].numpy(), aspect='auto', cmap='viridis')
        ax_pred.set_title(f"Reconstructed P_{p_labels[i]}")
        fig_sino.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        
    fig_sino.suptitle(f"Sinogram Comparison (Config: {cfg_name}, Seed: {seed})", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_sino.savefig(f'results/e2e_sinograms{file_suffix}.png', dpi=150)
    plt.close(fig_sino)

    # Plot 2: Spectral Curves
    print("Plotting Spectral Curves...")
    mid_a = nAngles // 2
    mid_n = nNeutrons // 2
    wavelengths_A = wavelengths * 1e10
    
    fig_spec, ax_spec = plt.subplots(figsize=(12, 8))
    for i in range(9):
        signal_true = [xb_true[0, w * 9 + i, mid_a, mid_n].item() for w in range(nW)]
        signal_pred = [xb_pred[0, w * 9 + i, mid_a, mid_n].item() for w in range(nW)]
        color = plt.cm.tab10(i)
        ax_spec.plot(wavelengths_A, signal_true, label=f"True P_{p_labels[i]}", color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax_spec.plot(wavelengths_A, signal_pred, label=f"Pred P_{p_labels[i]}", color=color, linestyle='--', linewidth=2, alpha=0.8)
        
    ax_spec.set_xlabel("Wavelength (Å)")
    ax_spec.set_ylabel("Polarization Value")
    ax_spec.set_title(f"Spectral Curves (Ray: {mid_n}, Angle: {mid_a}, Config: {cfg_name}, Seed: {seed})")
    ax_spec.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax_spec.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig_spec.savefig(f'results/e2e_spectral_curves{file_suffix}.png', dpi=150)
    plt.close(fig_spec)

    # Plot 3: Field Reconstruction
    print("Plotting Field Reconstruction...")
    voxel_size_mm = voxel_size * 1e3
    extent = [0, voxel_size_mm * im_size, 0, voxel_size_mm * im_size]
    xa_mT = A_true_np[0] * 1e3
    pred_mT = A_pred_np[0] * 1e3
    diff_mT = np.abs(xa_mT - pred_mT)

    fig_recon, axes_recon = plt.subplots(3, 3, figsize=(15, 13))
    components = ['Bx', 'By', 'Bz']
    for c in range(3):
        vmax = max(np.abs(xa_mT[:, :, c]).max() * 1.1, 1e-6)
        # True
        ax_true = axes_recon[0, c]
        im_true = ax_true.imshow(xa_mT[:, :, c], cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower', extent=extent)
        ax_true.set_title(f"True {components[c]}")
        fig_recon.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04, label="B [mT]")
        # Pred
        ax_recon = axes_recon[1, c]
        im_recon = ax_recon.imshow(pred_mT[:, :, c], cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower', extent=extent)
        ax_recon.set_title(f"Reconstructed {components[c]}")
        fig_recon.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04, label="B [mT]")
        # Error
        ax_err = axes_recon[2, c]
        err_max = max(diff_mT[:, :, c].max() * 1.1, 1e-6)
        im_err = ax_err.imshow(diff_mT[:, :, c], cmap='Reds', origin='lower', vmin=0, vmax=err_max, extent=extent)
        ax_err.set_title(f"Abs Error {components[c]}")
        fig_recon.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04, label="|ΔB| [mT]")

    fig_recon.suptitle(f"Magnetic Field Reconstruction (Config: {cfg_name}, Seed: {seed})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_recon.savefig(f'results/e2e_field_reconstruction{file_suffix}.png', dpi=150)
    plt.close(fig_recon)

    print(f"\nDONE! All plots saved with suffix '{file_suffix}':")
    print(f"- results/e2e_sinograms{file_suffix}.png")
    print(f"- results/e2e_spectral_curves{file_suffix}.png")
    print(f"- results/e2e_field_reconstruction{file_suffix}.png")
    
    print("\nDONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1.json', help='Path to configuration JSON')
    parser.add_argument('--model_path', type=str, default=None, help='Explicit path to model .pth file')
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    # Smart Model Selection Logic
    if args.model_path is None:
        cfg_name = os.path.splitext(os.path.basename(args.config))[0]
        # Try to find a model that matches this config name
        model_dir = 'models'
        if os.path.exists(model_dir):
            # Check for exact match: spin2b_12x12_config1.pth
            # We don't know im_size yet without loading config, so we look for the suffix
            matching_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                             if f.endswith(f'_{cfg_name}.pth')]
            
            if matching_files:
                # Use the most recent one that matches the config name
                args.model_path = max(matching_files, key=os.path.getctime)
            else:
                # Fallback to the most recent .pth file in the folder
                all_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]
                if all_files:
                    args.model_path = max(all_files, key=os.path.getctime)
                else:
                    args.model_path = 'models/spin2b_12x12.pth' # Final fallback
    
    run_reconstruction(args)
