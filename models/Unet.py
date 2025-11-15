import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder (Contracting Path)
        # Block 1: 200x200 -> 100x100
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 100x100 -> 50x50
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3: 50x50 -> 25x25
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck: 25x25
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Expanding Path) with skip connections
        # Block 1: 25x25 -> 50x50 (concatenate with enc3 output)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 from up1 + 128 from skip
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 50x50 -> 100x100 (concatenate with enc2 output)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 from up2 + 64 from skip
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 100x100 -> 200x200 (concatenate with enc1 output)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 from up3 + 32 from skip
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path with skip connections
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.up1(x)
        x = torch.cat([x, enc3_out], dim=1)  # Skip connection
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2_out], dim=1)  # Skip connection
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc1_out], dim=1)  # Skip connection
        x = self.dec3(x)
        
        # Final output
        x = self.final(x)
        
        return x
