import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class EnhancedGarmentNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_residual_blocks=4):
        super(EnhancedGarmentNet, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = self._make_layer(base_channels, base_channels, num_residual_blocks)
        self.encoder2 = self._make_layer(base_channels, base_channels*2, num_residual_blocks)
        self.encoder3 = self._make_layer(base_channels*2, base_channels*4, num_residual_blocks)
        
        self.bridge = self._make_layer(base_channels*4, base_channels*8, num_residual_blocks)
        
        self.decoder3 = self._make_layer(base_channels*8, base_channels*4, num_residual_blocks)
        self.decoder2 = self._make_layer(base_channels*4, base_channels*2, num_residual_blocks)
        self.decoder1 = self._make_layer(base_channels*2, base_channels, num_residual_blocks)
        
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3)
        
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.downsample(e1))
        e3 = self.encoder3(self.downsample(e2))
        
        # Bridge
        b = self.bridge(self.downsample(e3))
        
        # Decoder with skip connections
        d3 = self.decoder3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Final convolution
        out = self.final(d1)
        
        return out, [e1, e2, e3, b]

class EnhancedGarmentNetWithTimestep(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_residual_blocks=4, time_emb_dim=256):
        super(EnhancedGarmentNetWithTimestep, self).__init__()
        
        self.garment_net = EnhancedGarmentNet(in_channels, base_channels, num_residual_blocks)
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Projection for text embeddings
        self.text_proj = nn.Linear(768, time_emb_dim)  # Assuming text embeddings are 768-dimensional
        
        # Combine garment features with time and text embeddings
        self.combine = nn.ModuleList([
            nn.Conv2d(base_channels + time_emb_dim, base_channels, kernel_size=1),
            nn.Conv2d(base_channels*2 + time_emb_dim, base_channels*2, kernel_size=1),
            nn.Conv2d(base_channels*4 + time_emb_dim, base_channels*4, kernel_size=1),
            nn.Conv2d(base_channels*8 + time_emb_dim, base_channels*8, kernel_size=1)
        ])

    def forward(self, x, t, text_embeds):
        # Ensure all inputs are of the same dtype
        x = x.to(dtype=self.garment_net.initial[0].weight.dtype)
        t = t.to(dtype=self.garment_net.initial[0].weight.dtype)
        text_embeds = text_embeds.to(dtype=self.garment_net.initial[0].weight.dtype)
        
        # Get garment features
        garment_out, garment_features = self.garment_net(x)
        
        # Process timestep
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        
        # Process text embeddings
        text_emb = self.text_proj(text_embeds).unsqueeze(-1).unsqueeze(-1)
        
        # Combine embeddings
        cond_emb = t_emb + text_emb
        
        # Combine garment features with conditional embedding
        combined_features = []
        for feat, comb_layer in zip(garment_features, self.combine):
            # Expand conditional embedding to match feature map size
            expanded_cond_emb = cond_emb.expand(-1, -1, feat.size(2), feat.size(3))
            combined = comb_layer(torch.cat([feat, expanded_cond_emb], dim=1))
            combined_features.append(combined)
        
        return garment_out, combined_features
