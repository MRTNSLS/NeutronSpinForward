import torch
import torch.nn as nn

class Spin2DNet(nn.Module):
    """
    CPU-Optimized version of the reconstruction network.
    Uses 2D convolutions and Adaptive Pooling to keep parameter counts manageable on CPU.
    """
    def __init__(self, in_ch=135, out_ch=3, nAngles=181, nNeutrons=81, im_size=12, hidden_dim=64):
        super().__init__()
        self.nA = nAngles
        self.nN = nNeutrons
        self.im_size = im_size
        
        # 1. 2D Convolutional Encoder
        # Extracts local features from the sinogram
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 2. Dim Reduction (Spatial Pool)
        # Crucial to reduce the whopping 128*121*41 = 635k features
        # down to something manageable for a Linear layer on CPU.
        self.pool = nn.AdaptiveAvgPool2d((18, 6))
        
        # 3. Domain Transform (Learned mapping from Sinogram Space to Image Space)
        self.domain_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch * im_size * im_size)
        )
        
        # 4. Image Refinement
        self.refiner = nn.Sequential(
            nn.Conv2d(out_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder (2D Convolutions)
        x = self.encoder(x) # (Batch, 128, nA, nN)
        
        # Pool to reduce parameter count in subsequent Linear layer
        x = self.pool(x) # (Batch, 128, 18, 6)
        
        # Transform to Image Space
        x = self.domain_transform(x) # (Batch, 3 * 4 * 4)
        x = x.view(batch_size, 3, self.im_size, self.im_size)
        
        # Refinement
        out = self.refiner(x)
        
        return out

SpinToBNet = Spin2DNet
