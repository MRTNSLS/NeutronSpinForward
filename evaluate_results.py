import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from reproduce_neutron.dataset import NSpinFullDimDataset
from reproduce_neutron.model import SpinToBNet

def evaluate(args):
    print(f"Loading data from {args.b_path}...")
    
    # Identify the test split (unseen data)
    full_a = np.load(args.a_path)
    total_available = len(full_a)
    if args.num_samples is not None:
        total_available = min(total_available, args.num_samples)
    
    start_idx = int(total_available * args.train_split)
    test_count = total_available - start_idx
    print(f"Evaluating on {test_count} unseen samples (Start index: {start_idx}).")
    
    ds = NSpinFullDimDataset(args.b_path, args.a_path, num_samples=test_count, start_idx=start_idx)

    sample_b, sample_a = ds[0]
    in_channels = sample_b.shape[0]
    nAngles = sample_b.shape[1]
    nNeutrons = sample_b.shape[2]
    im_size = sample_a.shape[1]
    
    norm_factor = ds.minmax_B
    print(f"Using normalization factor: {norm_factor}")

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = SpinToBNet(in_ch=in_channels, out_ch=sample_a.shape[0],
                       nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, hidden_dim=args.hidden_dim).to(device)
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    num_samples = len(ds)
    # Pick a few samples for visualization
    samples_indices = [num_samples // 10, num_samples // 2, (9 * num_samples) // 10]
    
    os.makedirs('results', exist_ok=True)
    
    for idx, sample_idx in enumerate(samples_indices):
        print(f"Generating detailed plots for sample {sample_idx}...")
        xb, xa = ds[sample_idx]
        with torch.no_grad():
            inp = xb.unsqueeze(0).to(device)
            pred_a = model(inp).squeeze(0).cpu() * norm_factor
        
        xa_np = xa.numpy().transpose(1, 2, 0) * norm_factor
        pred_np = pred_a.numpy().transpose(1, 2, 0)
        diff_np = np.abs(xa_np - pred_np)

        p_labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        nW = in_channels // 9

        # --- 1. SINOGRAM GALLERY (9 components for middle wavelength) ---
        mid_w = nW // 2
        fig_sino, axes_sino = plt.subplots(3, 3, figsize=(14, 12))
        for i in range(9):
            sino_idx = mid_w * 9 + i
            ax = axes_sino[i // 3, i % 3]
            im = ax.imshow(xb[sino_idx].numpy(), aspect='auto', cmap='viridis')
            ax.set_title(f"P_{p_labels[i]} (λ index {mid_w})")
            fig_sino.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig_sino.suptitle(f"Sample {sample_idx} - Full Polarization Sinograms", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        sino_path = f'results/sinograms_sample_{sample_idx}.png'
        fig_sino.savefig(sino_path, dpi=150)
        plt.close(fig_sino)

        # --- 2. 1D SPECTRAL CURVES (9 components for a central ray) ---
        mid_a = nAngles // 2
        mid_n = nNeutrons // 2
        
        # Try to load wavelengths from meta file
        meta_path = args.b_path.replace('B_data_', 'meta_').replace('.npy', '.npz')
        wavelengths_val = np.arange(nW)
        x_label = "Wavelength Index"
        if os.path.exists(meta_path):
            with np.load(meta_path, allow_pickle=True) as meta:
                if 'wavelengths' in meta:
                    wavelengths_val = meta['wavelengths'] * 1e10 # Convert to Angstrom
                    x_label = "Wavelength (Å)"

        fig_spec, ax_spec = plt.subplots(figsize=(10, 6))
        for i in range(9):
            signal = [xb[w * 9 + i, mid_a, mid_n].item() for w in range(nW)]
            ax_spec.plot(wavelengths_val, signal, label=f"P_{p_labels[i]}", marker='o', markersize=4, linewidth=1.5)
        ax_spec.set_xlabel(x_label)
        ax_spec.set_ylabel("Polarization Value")
        ax_spec.set_title(f"Sample {sample_idx} - Spectral Signature (Ray: {mid_n}, Angle: {mid_a})")
        ax_spec.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax_spec.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        spec_path = f'results/spectral_sample_{sample_idx}.png'
        fig_spec.savefig(spec_path, dpi=150)
        plt.close(fig_spec)

        # --- 3. RECONSTRUCTION COMPARISON ---
        # Get physical scale from meta if available
        voxel_size_mm = 0.833 # Default for 1cm grid at 12x12
        if os.path.exists(meta_path):
            with np.load(meta_path, allow_pickle=True) as meta:
                if 'voxel_size' in meta:
                    voxel_size_mm = float(meta['voxel_size']) * 1e3
        extent = [0, voxel_size_mm * im_size, 0, voxel_size_mm * im_size]

        # Convert to mT for plotting
        xa_mT = xa_np * 1e3
        pred_mT = pred_np * 1e3
        diff_mT = diff_np * 1e3

        fig_recon, axes_recon = plt.subplots(3, 3, figsize=(15, 13))
        components = ['Bx', 'By', 'Bz']
        for c in range(3):
            vmax_true = np.abs(xa_mT[:, :, c]).max() * 1.1
            if vmax_true < 1e-6: vmax_true = 1e-6
            vmax_pred = np.abs(pred_mT[:, :, c]).max() * 1.1
            if vmax_pred < 1e-6: vmax_pred = 1e-6
            
            # Row 0: True
            ax_true = axes_recon[0, c]
            im_true = ax_true.imshow(xa_mT[:, :, c], cmap='coolwarm', vmin=-vmax_true, vmax=vmax_true, origin='lower', extent=extent)
            ax_true.set_title(f"True {components[c]}")
            ax_true.set_xlabel("x [mm]"); ax_true.set_ylabel("y [mm]")
            fig_recon.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04, label="B [mT]")
            
            # Row 1: Reconstruction
            ax_recon = axes_recon[1, c]
            im_recon = ax_recon.imshow(pred_mT[:, :, c], cmap='coolwarm', vmin=-vmax_pred, vmax=vmax_pred, origin='lower', extent=extent)
            ax_recon.set_title(f"Reconstructed {components[c]}")
            ax_recon.set_xlabel("x [mm]"); ax_recon.set_ylabel("y [mm]")
            fig_recon.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04, label="B [mT]")
            
            # Row 2: Absolute Error
            ax_err = axes_recon[2, c]
            err_max = diff_mT[:, :, c].max() * 1.1
            if err_max < 1e-6: err_max = 1e-6
            im_err = ax_err.imshow(diff_mT[:, :, c], cmap='Reds', origin='lower', vmin=0, vmax=err_max, extent=extent)
            ax_err.set_title(f"Abs Error {components[c]}")
            ax_err.set_xlabel("x [mm]"); ax_err.set_ylabel("y [mm]")
            fig_recon.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04, label="|ΔB| [mT]")

        fig_recon.suptitle(f"Sample {sample_idx} - Field Reconstruction Comparison", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        recon_path = f'results/reconstruction_sample_{sample_idx}.png'
        fig_recon.savefig(recon_path, dpi=150)
        plt.close(fig_recon)

    print(f"Evaluation complete. Plots saved to 'results/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_path', type=str, default='data/B_data_12x12.npy')
    parser.add_argument('--a_path', type=str, default='data/A_data_12x12.npy')
    parser.add_argument('--model_path', type=str, default='models/spin2b_12x12.pth')
    parser.add_argument('--output_path', type=str, default='results/evaluation_12x12.png')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=None, help='Total samples to consider (same as used in training)')
    parser.add_argument('--train_split', type=float, default=0.95, help='Portion of data that was used for training (will be skipped)')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args)
