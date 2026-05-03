import torch
import torch.nn as nn

class Spin2DNet(nn.Module):
    """
    Configurable reconstruction network with Legacy support.
    """
    def __init__(self, in_ch=540, out_ch=3, nAngles=360, nNeutrons=256, im_size=12, 
                 hidden_dim=64, pool_res_h=18, pool_res_w=6, legacy=False):
        super().__init__()
        self.nA = nAngles
        self.nN = nNeutrons
        self.im_size = im_size
        self.legacy = legacy
        
        act = nn.ReLU(inplace=True) if legacy else nn.LeakyReLU(0.1, inplace=True)
        
        # 1. 2D Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act
        )
        
        # 2. Configurable Dim Reduction
        self.pool = nn.AdaptiveAvgPool2d((pool_res_h, pool_res_w))
        num_features = 128 * pool_res_h * pool_res_w
        
        # 3. Domain Transform
        if legacy:
            # Original simple architecture
            self.domain_transform = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_ch * im_size * im_size)
            )
        else:
            # New multi-stage GPU architecture
            self.domain_transform = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, hidden_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(hidden_dim // 2, out_ch * im_size * im_size)
            )
        
        # 4. Image Refinement
        ref_ch = 32 if legacy else 64
        self.refiner = nn.Sequential(
            nn.Conv2d(out_ch, ref_ch, kernel_size=3, padding=1),
            act,
            nn.Conv2d(ref_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.domain_transform(x)
        x = x.view(batch_size, 3, self.im_size, self.im_size)
        out = self.refiner(x)
        return out

SpinToBNet = Spin2DNet
