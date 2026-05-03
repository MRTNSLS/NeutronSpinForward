import numpy as np
import torch
from torch.utils.data import Dataset
import os

class NSpinFullDimDataset(Dataset):
    def __init__(self, b_path, a_path, num_samples=None, start_idx=0):
        """
        Loads the multidimensional dataset.
        B_raw: (N, nNeutrons, nAngles, nWavelengths, 3, 3)
        A_raw: (N, im_size, im_size, 3)
        """
        # Load A_raw first (small)
        A_raw = np.load(a_path)
        total_available = len(A_raw)
        
        # Determine slice indices
        end_idx = total_available
        if num_samples is not None:
            end_idx = min(start_idx + num_samples, total_available)
        
        A_raw = A_raw[start_idx:end_idx]
        
        # Try to load meta to get minmax_B
        meta_path = b_path.replace('B_data_', 'meta_').replace('.npy', '.npz')
        self.minmax_B = 5e-3 
        if os.path.exists(meta_path):
            with np.load(meta_path, allow_pickle=True) as meta:
                if 'minmax_B' in meta:
                    self.minmax_B = float(meta['minmax_B'])
        
        # Load B_raw using memory mapping to prevent OOM
        # Slicing the memmap is efficient
        self.B_raw = np.load(b_path, mmap_mode='r')[start_idx:end_idx]
        
        self.length = self.B_raw.shape[0]
        
        # Normalize target to ~[-1, 1]
        self.A = (np.transpose(A_raw, (0, 3, 1, 2)) / self.minmax_B).astype(np.float32)

        
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # Load a single sample directly into memory. 
        # The data is already stored in the (Channels, Angles, Neutrons) layout
        # which eliminates CPU-side transposes during training.
        return torch.from_numpy(self.B_raw[i]), torch.from_numpy(self.A[i])
