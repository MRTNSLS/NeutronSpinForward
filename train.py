import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Use the standalone dataset and model from within gpu_optimized
from reproduce_neutron.dataset import NSpinFullDimDataset
from reproduce_neutron.model import SpinToBNet

import argparse

def train(config_path='config1.json', override_seed=None):
    # 1. Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    im_size = config.get('im_size', 12)
    out_dir = config.get('out_dir', 'data')
    v = f'{im_size}x{im_size}'
    
    b_path = os.path.join(out_dir, f'B_data_{v}.npy')
    a_path = os.path.join(out_dir, f'A_data_{v}.npy')
    
    epochs = config.get('train_epochs', 100)
    batch_size = config.get('train_batch_size', 64)
    lr = config.get('train_lr', 1e-3)
    hidden_dim = config.get('hidden_dim', 64)
    pool_res_h = config.get('pool_res_h', 18)
    pool_res_w = config.get('pool_res_w', 6)
    train_split = config.get('train_split', 0.95)
    num_samples = config.get('num_samples', 1000)
    
    # Use override seed if provided, otherwise use config seed
    seed = override_seed if override_seed is not None else config.get('seed', 42)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Setup Device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Loading data from {b_path} and {a_path}...")
    print(f"Using device: {device_name}")
    print(f"Config: {config_path} | Seed: {seed}")
    
    # Calculate training split
    full_a = np.load(a_path)
    total_available = len(full_a)
    if num_samples is not None:
        total_available = min(total_available, num_samples)
    
    train_count = int(total_available * train_split)
    print(f"Using {train_count} samples for training (Split: {train_split}).")
    
    if train_count == 0:
        print("Error: 0 samples selected for training. Increase num_samples or train_split.")
        return

    # 3. Optimize DataLoader for GPU transfer
    use_pin_memory = (device_name == 'cuda')
    num_workers = 2 if device_name == 'cuda' else 0 
    
    ds = NSpinFullDimDataset(b_path, a_path, num_samples=train_count, start_idx=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                    num_workers=num_workers, pin_memory=use_pin_memory,
                    generator=torch.Generator().manual_seed(seed)) # Ensure reproducible shuffle

    sample_b, sample_a = ds[0]
    in_channels = sample_b.shape[0]
    nAngles = sample_b.shape[1]
    nNeutrons = sample_b.shape[2]
    
    print(f"Configuring Model: (Ch: {in_channels}, A: {nAngles}, N: {nNeutrons}) -> Image: {im_size}x{im_size}")
    
    model = SpinToBNet(in_ch=in_channels, out_ch=sample_a.shape[0],
                       nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, hidden_dim=hidden_dim,
                       pool_res_h=pool_res_h, pool_res_w=pool_res_w,
                       legacy=config.get('legacy', False)).to(device)
                       
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    use_amp = (device_name == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model.train()
    print(f"Starting optimized training on {device}...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, xa in dl:
            xb = xb.to(device, non_blocking=use_pin_memory)
            xa = xa.to(device, non_blocking=use_pin_memory)
            opt.zero_grad()
            with torch.amp.autocast(device_name, enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, xa)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(ds)
        print(f'Epoch {epoch+1:03d}/{epochs:03d} | Loss: {avg_loss:.6f}', flush=True)

    models_out = 'models'
    os.makedirs(models_out, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    save_path = os.path.join(models_out, f'spin2b_{v}_{cfg_name}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Saved trained model to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1.json', help='Path to configuration JSON')
    parser.add_argument('--seed', type=int, default=None, help='Override the random seed')
    args = parser.parse_args()
    train(config_path=args.config, override_seed=args.seed)
