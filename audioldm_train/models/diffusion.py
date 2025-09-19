class AudioDiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.time_embedding = SinusoidalPositionEmbedding(config.time_embed_dim)
        self.text_proj = nn.Linear(config.text_embed_dim, config.hidden_dim)
        
        # U-Net architecture
        self.down_blocks = nn.ModuleList([
            ResidualBlock(config.latent_dim, 64, config.time_embed_dim),
            ResidualBlock(64, 128, config.time_embed_dim),
            ResidualBlock(128, 256, config.time_embed_dim),
            ResidualBlock(256, 512, config.time_embed_dim),
        ])
        
        self.mid_block = ResidualBlock(512, 512, config.time_embed_dim)
        
        self.up_blocks = nn.ModuleList([
            ResidualBlock(512 + 512, 256, config.time_embed_dim),
            ResidualBlock(256 + 256, 128, config.time_embed_dim),
            ResidualBlock(128 + 128, 64, config.time_embed_dim),
            ResidualBlock(64 + 64, config.latent_dim, config.time_embed_dim),
        ])
        
        self.output_conv = nn.Conv2d(config.latent_dim, config.latent_dim, 3, 1, 1)
        
    def forward(self, x, timesteps, text_embed):
        # Time embedding
        t_embed = self.time_embedding(timesteps)
        
        # Text conditioning
        text_cond = self.text_proj(text_embed)
        
        # Downsampling
        skip_connections = []
        h = x
        
        for block in self.down_blocks:
            h = block(h, t_embed, text_cond)
            skip_connections.append(h)
            h = nn.functional.max_pool2d(h, 2)
        
        # Middle
        h = self.mid_block(h, t_embed, text_cond)
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            h = nn.functional.interpolate(h, scale_factor=2, mode='nearest')
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_embed, text_cond)
        
        return self.output_conv(h)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_embed, text_embed=None):
        residual = x
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        time_proj = self.time_proj(time_embed)[:, :, None, None]
        h = h + time_proj
        
        h = nn.functional.relu(h)
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Skip connection
        residual = self.skip_connection(residual)
        
        return nn.functional.relu(h + residual)

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

