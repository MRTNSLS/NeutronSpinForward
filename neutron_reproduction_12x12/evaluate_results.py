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
    
    fig, axes = plt.subplots(len(samples_indices) * 3, 3, figsize=(15, 4 * len(samples_indices) * 3))
    
    for idx, sample_idx in enumerate(samples_indices):
        xb, xa = ds[sample_idx]
        with torch.no_grad():
            inp = xb.unsqueeze(0).to(device)
            pred_a = model(inp).squeeze(0).cpu() * norm_factor
        
        xa_np = xa.numpy().transpose(1, 2, 0) * norm_factor
        pred_np = pred_a.numpy().transpose(1, 2, 0)
        diff_np = np.abs(xa_np - pred_np)

        row_start = idx * 3
        components = ['Bx', 'By', 'Bz']
        
        for c in range(3):
            vmax_true = np.abs(xa_np[:, :, c]).max() * 1.1
            if vmax_true < 1e-9: vmax_true = 1e-9
            vmax_pred = np.abs(pred_np[:, :, c]).max() * 1.1
            if vmax_pred < 1e-9: vmax_pred = 1e-9
            
            # TRUE
            ax_true = axes[row_start, c]
            ax_true.imshow(xa_np[:, :, c], cmap='coolwarm', vmin=-vmax_true, vmax=vmax_true, origin='lower')
            ax_true.set_title(f"Sample {sample_idx} - True {components[c]}")
            
            # RECON
            ax_recon = axes[row_start+1, c]
            ax_recon.imshow(pred_np[:, :, c], cmap='coolwarm', vmin=-vmax_pred, vmax=vmax_pred, origin='lower')
            ax_recon.set_title(f"Reconstructed {components[c]}")
            
            # ERROR
            ax_err = axes[row_start + 2, c]
            err_max = diff_np[:, :, c].max() * 1.1
            if err_max < 1e-9: err_max = 1e-9
            ax_err.imshow(diff_np[:, :, c], cmap='Reds', origin='lower', vmin=0, vmax=err_max)
            ax_err.set_title(f"Abs Error {components[c]}")

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved evaluation plots to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_path', type=str, default='data/B_data_12x12.npy')
    parser.add_argument('--a_path', type=str, default='data/A_data_12x12.npy')
    parser.add_argument('--model_path', type=str, default='models/spin2b_12x12.pth')
    parser.add_argument('--output_path', type=str, default='results/evaluation_12x12.png')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=None, help='Total samples to consider (same as used in training)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Portion of data that was used for training (will be skipped)')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args)
