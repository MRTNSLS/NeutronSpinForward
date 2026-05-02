import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized
from reproduce_neutron.model import SpinToBNet
from generate_data import generate_smooth_b_field

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

def run_end_to_end(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. PARAMETERS (Matching 12x12 training data)
    im_size = 12
    nNeutrons = 256
    nAngles = 360
    wavelengths = np.arange(2.0, 8.0, 0.1) * 1e-10
    nW = len(wavelengths)
    scaleD = 2.0
    voxel_size = 1e-2 / im_size
    minmax_B = 5e-3
    seed = args.seed
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n--- 1. Generating Data ---")
    print(f"Precomputing geometry for {im_size}x{im_size} grid...")
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

    print("Generating smooth True B-field...")
    A_true_np = generate_smooth_b_field(im_size, num_samples=1, minmax_B=minmax_B)
    
    print("\n--- 2. Forward Physics (True Field) ---")
    B_batch_true = A_true_np.reshape(1, im_size*im_size, 3, order='F')
    B_true_raw = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_true, device=device)
    
    # Format to tensor
    xb_true = get_xb_from_B(B_true_raw)
    
    print("\n--- 3. Adding Statistical Noise ---")
    xb_noisy = xb_true.clone()
    if args.noise_level > 0:
        std_dev = xb_true.std() * args.noise_level
        noise = torch.randn_like(xb_true) * std_dev
        xb_noisy += noise
        print(f"Added Gaussian noise (Level: {args.noise_level}, Std: {std_dev.item():.4f})")
    else:
        print("No noise added.")
        
    print("\n--- 4. Reconstruction ---")
    print(f"Loading model from {args.model_path}...")
    model = SpinToBNet(in_ch=nW*9, out_ch=3, nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, hidden_dim=64).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Model file {args.model_path} not found.")
        print("Falling back to uninitialized model for pipeline testing.")
        
    model.eval()
    with torch.no_grad():
        inp = xb_noisy.to(device)
        # Model outputs normalized A_data, so we multiply by minmax_B
        pred_a = model(inp).cpu()
        A_pred_np = pred_a.numpy().transpose(0, 2, 3, 1) * minmax_B 
        
    print("\n--- 5. Re-Forward Physics (Reconstructed Field) ---")
    B_batch_pred = A_pred_np.reshape(1, im_size*im_size, 3, order='F')
    B_pred_raw = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_pred, device=device)
    
    xb_pred = get_xb_from_B(B_pred_raw)
    
    print("\n--- 6. Generating Plots ---")
    os.makedirs('results', exist_ok=True)
    p_labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    mid_w = nW // 2
    
    # Plot 1: Sinograms
    print("Plotting Sinogram Comparison...")
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
        
    fig_sino.suptitle(f"Sinogram Comparison (Middle Wavelength, Noise Level: {args.noise_level})", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_sino.savefig('results/e2e_sinograms.png', dpi=150)
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
        ax_spec.plot(wavelengths_A, signal_true, label=f"True P_{p_labels[i]}", 
                     color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax_spec.plot(wavelengths_A, signal_pred, label=f"Pred P_{p_labels[i]}", 
                     color=color, linestyle='--', linewidth=2, alpha=0.8)
        
    ax_spec.set_xlabel("Wavelength (Å)")
    ax_spec.set_ylabel("Polarization Value")
    ax_spec.set_title(f"Spectral Polarization Curves (Ray: {mid_n}, Angle: {mid_a})")
    
    # Shrink current axis and put legend outside
    box = ax_spec.get_position()
    ax_spec.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_spec.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    
    ax_spec.grid(True, linestyle=':', alpha=0.6)
    fig_spec.savefig('results/e2e_spectral_curves.png', dpi=150)
    plt.close(fig_spec)

    # Plot 3: Field Reconstruction
    print("Plotting Field Reconstruction...")
    voxel_size_mm = voxel_size * 1e3
    extent = [0, voxel_size_mm * im_size, 0, voxel_size_mm * im_size]
    
    # Slice the padding for visualization if present in A_data
    # A_data shape: (1, 12, 12, 3). The core field is 10x10 padded to 12x12
    xa_mT = A_true_np[0] * 1e3
    pred_mT = A_pred_np[0] * 1e3
    diff_mT = np.abs(xa_mT - pred_mT)

    fig_recon, axes_recon = plt.subplots(3, 3, figsize=(15, 13))
    components = ['Bx', 'By', 'Bz']
    for c in range(3):
        vmax_true = max(np.abs(xa_mT[:, :, c]).max() * 1.1, 1e-6)
        vmax_pred = max(np.abs(pred_mT[:, :, c]).max() * 1.1, 1e-6)
        
        # True
        ax_true = axes_recon[0, c]
        im_true = ax_true.imshow(xa_mT[:, :, c], cmap='coolwarm', vmin=-vmax_true, vmax=vmax_true, origin='lower', extent=extent)
        ax_true.set_title(f"True {components[c]}")
        fig_recon.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04, label="B [mT]")
        
        # Pred
        ax_recon = axes_recon[1, c]
        im_recon = ax_recon.imshow(pred_mT[:, :, c], cmap='coolwarm', vmin=-vmax_pred, vmax=vmax_pred, origin='lower', extent=extent)
        ax_recon.set_title(f"Reconstructed {components[c]}")
        fig_recon.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04, label="B [mT]")
        
        # Error
        ax_err = axes_recon[2, c]
        err_max = max(diff_mT[:, :, c].max() * 1.1, 1e-6)
        im_err = ax_err.imshow(diff_mT[:, :, c], cmap='Reds', origin='lower', vmin=0, vmax=err_max, extent=extent)
        ax_err.set_title(f"Abs Error {components[c]}")
        fig_recon.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04, label="|ΔB| [mT]")

    fig_recon.suptitle("Magnetic Field Reconstruction Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_recon.savefig('results/e2e_field_reconstruction.png', dpi=150)
    plt.close(fig_recon)

    print("\nDONE! All plots saved to 'results/' directory:")
    print("- results/e2e_sinograms.png")
    print("- results/e2e_spectral_curves.png")
    print("- results/e2e_field_reconstruction.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/spin2b_12x12.pth')
    parser.add_argument('--noise_level', type=float, default=0.0, help="Fraction of signal std to add as Gaussian noise")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()
    
    run_end_to_end(args)
