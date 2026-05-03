import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports
from reproduce_neutron.calc_tools import sou_det_calc, ray_wrapper
from reproduce_neutron.forward_model import calc_precession_vectorized
from reproduce_neutron.model import SpinToBNet
from generate_data import generate_smooth_b_field_gpu

def get_xb_from_B(B_data):
    b_reshaped = np.transpose(B_data, (0, 3, 4, 5, 2, 1))
    nW, nA, nN = b_reshaped.shape[1], b_reshaped.shape[4], b_reshaped.shape[5]
    xb = b_reshaped.reshape(1, nW * 9, nA, nN).astype(np.float32)
    return torch.from_numpy(xb)

def load_model_from_config(config_path, model_path, device):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    im_size = config.get('im_size', 12)
    nNeutrons = config.get('nNeutrons', 256)
    nAngles = config.get('nAngles', 360)
    hidden_dim = config.get('hidden_dim', 64)
    pool_res_h = config.get('pool_res_h', 18)
    pool_res_w = config.get('pool_res_w', 6)
    legacy = config.get('legacy', False)
    
    # Calculate in_channels based on config wavelengths
    w_start = config.get('wavelengths_start', 2.0)
    w_end = config.get('wavelengths_end', 8.0)
    w_step = config.get('wavelengths_step', 0.1)
    nW = len(np.arange(w_start, w_end, w_step))
    
    model = SpinToBNet(in_ch=nW*9, out_ch=3, nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, hidden_dim=hidden_dim,
                       pool_res_h=pool_res_h, pool_res_w=pool_res_w,
                       legacy=legacy).to(device)
    
    print(f"Loading {model_path} with config {config_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, config

def run_comparison(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load both configs to get simulation parameters
    with open(args.configA, 'r') as f:
        configA = json.load(f)
    
    # We use configA for the simulation geometry
    im_size = configA['im_size']
    nNeutrons = configA['nNeutrons']
    nAngles = configA['nAngles']
    scaleD = configA['scaleD']
    minmax_B = configA['minmax_B']
    wavelengths = np.arange(configA['wavelengths_start'], configA['wavelengths_end'], configA['wavelengths_step']) * 1e-10
    voxel_size = 1e-2 / im_size
    
    # 2. Generate Ground Truth (Same for both)
    print(f"\n--- 1. Generating Ground Truth (Seed: {args.seed}) ---")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    A_true_tensor = generate_smooth_b_field_gpu(im_size, num_samples=1, minmax_B=minmax_B, device=device)
    A_true_np = A_true_tensor.cpu().numpy()[0]
    
    print("Simulating Physics...")
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
            vi, vd = ray_wrapper.ray_wrapper(source_xs[a][n], source_zs[a][n], det_xs[a][n], det_zs[a][n], im_size)
            voxel_index[a][n], voxel_data[a][n] = vi, vd

    B_batch_true = A_true_np.reshape(1, im_size*im_size, 3, order='F')
    B_true_raw = calc_precession_vectorized(nNeutrons, nAngles, angles, wavelengths,
                                        voxel_index, voxel_data, voxel_size, B_batch_true, device=device)
    xb_true = get_xb_from_B(B_true_raw)
    
    # 3. Load Models
    print("\n--- 2. Loading Models ---")
    modelA, _ = load_model_from_config(args.configA, args.modelA, device)
    modelB, _ = load_model_from_config(args.configB, args.modelB, device)
    
    # 4. Inference
    print("\n--- 3. Running Inference ---")
    with torch.no_grad():
        inp = xb_true.to(device)
        predA = modelA(inp).cpu().numpy()[0].transpose(1, 2, 0) * minmax_B
        predB = modelB(inp).cpu().numpy()[0].transpose(1, 2, 0) * minmax_B
        
    # 5. Combined Plotting
    print("\n--- 4. Generating Comparison Plots ---")
    os.makedirs('results', exist_ok=True)
    voxel_size_mm = voxel_size * 1e3
    extent = [0, voxel_size_mm * im_size, 0, voxel_size_mm * im_size]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    components = ['Bx', 'By', 'Bz']
    cfgA_name = os.path.splitext(os.path.basename(args.configA))[0]
    cfgB_name = os.path.splitext(os.path.basename(args.configB))[0]
    
    for c in range(3):
        vmax = max(np.abs(A_true_np[:,:,c]).max() * 1.1, 1e-6) * 1000 # mT
        # True
        im_t = axes[0, c].imshow(A_true_np[:,:,c]*1000, cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower', extent=extent)
        axes[0, c].set_title(f"True {components[c]} (mT)")
        fig.colorbar(im_t, ax=axes[0, c], fraction=0.046, pad=0.04)
        
        # Model A
        im_a = axes[1, c].imshow(predA[:,:,c]*1000, cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower', extent=extent)
        axes[1, c].set_title(f"{cfgA_name}: {components[c]} (mT)")
        fig.colorbar(im_a, ax=axes[1, c], fraction=0.046, pad=0.04)
        
        # Model B
        im_b = axes[2, c].imshow(predB[:,:,c]*1000, cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower', extent=extent)
        axes[2, c].set_title(f"{cfgB_name}: {components[c]} (mT)")
        fig.colorbar(im_b, ax=axes[2, c], fraction=0.046, pad=0.04)

    fig.suptitle(f"Model Comparison | Seed: {args.seed}\nTop: True | Mid: {cfgA_name} | Bot: {cfgB_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = f'results/model_comparison_{cfgA_name}_vs_{cfgB_name}_s{args.seed}.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nDONE! Comparison plot saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configA', type=str, required=True)
    parser.add_argument('--modelA', type=str, required=True)
    parser.add_argument('--configB', type=str, required=True)
    parser.add_argument('--modelB', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_comparison(args)
