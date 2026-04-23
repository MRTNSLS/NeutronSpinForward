import numpy as np
import torch
from torch.utils.data import Dataset
import os

class NSpinFullDimDataset(Dataset):
    def __init__(self, b_path, a_path, num_samples=None):
        """
        Loads the multidimensional dataset.
        B_raw: (N, nNeutrons, nAngles, nWavelengths, 3, 3)
        A_raw: (N, im_size, im_size, 3)
        """
        # Load A_raw first (small)
        A_raw = np.load(a_path)
        if num_samples is not None:
            A_raw = A_raw[:num_samples]
        
        # Try to load meta to get minmax_B
        meta_path = b_path.replace('B_data_', 'meta_').replace('.npy', '.npz')
        self.minmax_B = 5e-3 
        if os.path.exists(meta_path):
            with np.load(meta_path, allow_pickle=True) as meta:
                if 'minmax_B' in meta:
                    self.minmax_B = float(meta['minmax_B'])
        
        # Load B_raw using memory mapping to prevent OOM
        self.B_raw = np.load(b_path, mmap_mode='r')
        if num_samples is not None:
            self.B_raw = self.B_raw[:num_samples]
        
        self.length = self.B_raw.shape[0]
        
        # Normalize target to ~[-1, 1]
        self.A = (np.transpose(A_raw, (0, 3, 1, 2)) / self.minmax_B).astype(np.float32)

        
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # Load a single sample into memory
        b_item = self.B_raw[i]  # Shape: (nN, nAngles, nWavelengths, 3, 3)
        # Transpose to (nW, 3, 3, nA, nN) -> indices (2, 3, 4, 1, 0)
        b_reshaped = np.transpose(b_item, (2, 3, 4, 1, 0))
        # Reshape to flatten wavelength and polarization dimensions
        nW = b_reshaped.shape[0]
        nA = b_reshaped.shape[3]
        nN = b_reshaped.shape[4]
        b_final = b_reshaped.reshape(nW * 9, nA, nN).astype(np.float32)

        return torch.from_numpy(b_final), torch.from_numpy(self.A[i])
