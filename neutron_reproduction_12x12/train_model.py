import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reproduce_neutron.dataset import NSpinFullDimDataset
from reproduce_neutron.model import SpinToBNet

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Loading data from {args.b_path} and {args.a_path}...")
    
    ds = NSpinFullDimDataset(args.b_path, args.a_path, num_samples=args.num_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Infer sizes from the first data sample
    sample_b, sample_a = ds[0]
    in_channels = sample_b.shape[0]
    nAngles = sample_b.shape[1]
    nNeutrons = sample_b.shape[2]
    im_size = sample_a.shape[1]
    
    print(f"Configuring Model: (Ch: {in_channels}, A: {nAngles}, N: {nNeutrons}) -> Image: {im_size}x{im_size}")
    
    model = SpinToBNet(in_ch=in_channels, out_ch=sample_a.shape[0],
                       nAngles=nAngles, nNeutrons=nNeutrons,
                       im_size=im_size, hidden_dim=args.hidden_dim).to(device)
                       
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    print(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for xb, xa in dl:
            xb = xb.to(device)
            xa = xa.to(device)
            
            pred = model(xb)
            loss = loss_fn(pred, xa)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item() * xb.size(0)
            
        avg_loss = running_loss / len(ds)
        print(f'Epoch {epoch+1:03d}/{args.epochs:03d} | Loss: {avg_loss:.6f}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, 'spin2b_12x12.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Saved trained model to {save_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_path', type=str, default='data/B_data_12x12.npy')
    parser.add_argument('--a_path', type=str, default='data/A_data_12x12.npy')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='models')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()
