class AudioVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.latent_dim = config.latent_dim
        
        # Encoder
        self.encoder = AudioEncoder(
            in_channels=config.in_channels,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim
        )
        
        # Decoder  
        self.decoder = AudioDecoder(
            latent_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            out_channels=config.out_channels
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

class AudioEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32, 64, 128, 256, 512], latent_dim=8):
        super().__init__()
        
        modules = []
        
        # Build encoder layers
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the flattened dimension after convolutions
        # Assuming input shape [1, 1024, 64] -> after 5 stride-2 convs: [512, 32, 2]
        self.flatten_dim = hidden_dims[-1] * 32 * 2  # Adjust based on your input size
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar

class AudioDecoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dims=[512, 256, 128, 64, 32], out_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Project latent to feature map
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 32 * 2)
        
        modules = []
        
        # Build decoder layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.hidden_dims[0], 32, 2)  # Reshape to feature map
        decoded = self.decoder(h)
        return decoded