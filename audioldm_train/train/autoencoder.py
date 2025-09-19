import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from omegaconf import OmegaConf
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# =============================================================================
# VAE TRAINING MODULE
# =============================================================================

class VAETrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        
        # Initialize VAE
        self.vae = AudioVAE(config.model)
        
        # Loss weights
        self.recon_loss_weight = config.training.recon_loss_weight
        self.kl_loss_weight = config.training.kl_loss_weight
        self.perceptual_loss_weight = config.training.get('perceptual_loss_weight', 0.0)
        
        # Perceptual loss (optional)
        if self.perceptual_loss_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        
        # Discriminator for adversarial training (optional)
        if config.training.get('use_discriminator', False):
            self.discriminator = AudioDiscriminator(config.discriminator)
            self.disc_loss_weight = config.training.disc_loss_weight
            self.gen_loss_weight = config.training.gen_loss_weight
        else:
            self.discriminator = None
    
    def training_step(self, batch, batch_idx):
        mel_spec = batch  # [B, 1, H, W]
        
        if self.discriminator is not None:
            # Get optimizers
            opt_g, opt_d = self.optimizers()
            
            # Train Generator (VAE)
            self.toggle_optimizer(opt_g)
            
            # VAE forward pass
            reconstructed, mu, logvar = self.vae(mel_spec)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, mel_spec)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / (mel_spec.shape[0] * mel_spec.numel())
            
            # Perceptual loss (if enabled)
            perceptual_loss = 0.0
            if self.perceptual_loss_weight > 0:
                perceptual_loss = self.perceptual_loss(reconstructed, mel_spec)
            
            # Adversarial loss (generator)
            if self.discriminator is not None:
                fake_pred = self.discriminator(reconstructed)
                gen_loss = F.binary_cross_entropy_with_logits(
                    fake_pred, torch.ones_like(fake_pred)
                )
            else:
                gen_loss = 0.0
            
            # Total generator loss
            g_loss = (self.recon_loss_weight * recon_loss + 
                     self.kl_loss_weight * kl_loss +
                     self.perceptual_loss_weight * perceptual_loss +
                     self.gen_loss_weight * gen_loss)
            
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
            
            # Train Discriminator
            self.toggle_optimizer(opt_d)
            
            # Real samples
            real_pred = self.discriminator(mel_spec.detach())
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            
            # Fake samples
            fake_pred = self.discriminator(reconstructed.detach())
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)
            
            # Logging
            self.log_dict({
                'train/recon_loss': recon_loss,
                'train/kl_loss': kl_loss,
                'train/perceptual_loss': perceptual_loss,
                'train/gen_loss': gen_loss,
                'train/disc_loss': d_loss,
                'train/total_g_loss': g_loss
            }, prog_bar=True)
            
        else:
            # Standard VAE training without discriminator
            opt = self.optimizers()
            
            # VAE forward pass
            reconstructed, mu, logvar = self.vae(mel_spec)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, mel_spec)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / (mel_spec.shape[0] * mel_spec.numel())
            
            # Perceptual loss (if enabled)
            perceptual_loss = 0.0
            if self.perceptual_loss_weight > 0:
                perceptual_loss = self.perceptual_loss(reconstructed, mel_spec)
            
            # Total loss
            total_loss = (self.recon_loss_weight * recon_loss + 
                         self.kl_loss_weight * kl_loss +
                         self.perceptual_loss_weight * perceptual_loss)
            
            # Logging
            self.log_dict({
                'train/recon_loss': recon_loss,
                'train/kl_loss': kl_loss,
                'train/perceptual_loss': perceptual_loss,
                'train/total_loss': total_loss
            }, prog_bar=True)
            
            return total_loss
    
    def validation_step(self, batch, batch_idx):
        mel_spec = batch
        
        with torch.no_grad():
            reconstructed, mu, logvar = self.vae(mel_spec)
            
            # Losses
            recon_loss = F.mse_loss(reconstructed, mel_spec)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / (mel_spec.shape[0] * mel_spec.numel())
            
            perceptual_loss = 0.0
            if self.perceptual_loss_weight > 0:
                perceptual_loss = self.perceptual_loss(reconstructed, mel_spec)
            
            total_loss = (self.recon_loss_weight * recon_loss + 
                         self.kl_loss_weight * kl_loss +
                         self.perceptual_loss_weight * perceptual_loss)
        
        # Logging
        self.log_dict({
            'val/recon_loss': recon_loss,
            'val/kl_loss': kl_loss,
            'val/perceptual_loss': perceptual_loss,
            'val/total_loss': total_loss
        }, prog_bar=True)
        
        # Save sample reconstructions
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self.save_reconstruction_samples(mel_spec, reconstructed)
        
        return total_loss
    
    def save_reconstruction_samples(self, original, reconstructed, num_samples=4):
        """Save sample reconstructions for visual inspection"""
        import matplotlib.pyplot as plt
        
        # Create output directory
        output_dir = os.path.join(self.logger.log_dir, 'reconstructions')
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy
        orig_np = original[:num_samples].detach().cpu().numpy()
        recon_np = reconstructed[:num_samples].detach().cpu().numpy()
        
        fig, axes = plt.subplots(2 * num_samples, 1, figsize=(12, 3 * num_samples))
        
        for i in range(num_samples):
            # Original
            axes[2*i].imshow(orig_np[i, 0], aspect='auto', origin='lower')
            axes[2*i].set_title(f'Original {i+1}')
            axes[2*i].set_ylabel('Mel Bins')
            
            # Reconstructed
            axes[2*i+1].imshow(recon_np[i, 0], aspect='auto', origin='lower')
            axes[2*i+1].set_title(f'Reconstructed {i+1}')
            axes[2*i+1].set_ylabel('Mel Bins')
            axes[2*i+1].set_xlabel('Time Frames')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'epoch_{self.current_epoch:03d}.png'))
        plt.close()
    
    def configure_optimizers(self):
        if self.discriminator is not None:
            # Separate optimizers for generator and discriminator
            opt_g = torch.optim.Adam(
                self.vae.parameters(),
                lr=self.config.training.learning_rate,
                betas=(0.5, 0.999),
                weight_decay=self.config.training.weight_decay
            )
            
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.training.discriminator_lr,
                betas=(0.5, 0.999),
                weight_decay=self.config.training.weight_decay
            )
            
            return [opt_g, opt_d]
        else:
            # Single optimizer for VAE
            optimizer = torch.optim.Adam(
                self.vae.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.max_epochs
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }

class AudioDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_dims = config.hidden_dims  # [32, 64, 128, 256, 512]
        in_channels = config.in_channels
        
        modules = []
        
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.features = nn.Sequential(*modules)
        
        # Calculate flattened dimension
        self.flatten_dim = hidden_dims[-1] * 32 * 2  # Adjust based on input size
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# =============================================================================
# PERCEPTUAL LOSS
# =============================================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple perceptual loss using L1 in frequency domain
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        # Convert to frequency domain for perceptual comparison
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Use magnitude for perceptual loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return self.l1_loss(pred_mag, target_mag)

# =============================================================================
# VAE DATASET
# =============================================================================

class VAEAudioDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, duration=10.0):
        self.metadata = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            f_min=0,
            f_max=sample_rate // 2
        )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, row['audio_filename'])
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate
        if waveform.shape[1] > self.target_length:
            start = torch.randint(0, waveform.shape[1] - self.target_length + 1, (1,))
            waveform = waveform[:, start:start + self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)  # Log mel
        
        # Normalize to [-1, 1]
        mel_spec = 2 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) - 1
        
        return mel_spec

# =============================================================================
# VAE CONFIGURATION
# =============================================================================

def get_vae_config():
    config = {
        'model': {
            'in_channels': 1,
            'out_channels': 1,
            'hidden_dims': [32, 64, 128, 256, 512],
            'latent_dim': 8
        },
        
        'discriminator': {
            'in_channels': 1,
            'hidden_dims': [32, 64, 128, 256, 512]
        },
        
        'training': {
            'learning_rate': 1e-4,
            'discriminator_lr': 1e-4,
            'weight_decay': 1e-6,
            'batch_size': 8,
            'max_epochs': 200,
            'num_workers': 4,
            
            # Loss weights
            'recon_loss_weight': 1.0,
            'kl_loss_weight': 0.1,
            'perceptual_loss_weight': 0.1,
            'disc_loss_weight': 0.5,
            'gen_loss_weight': 0.1,
            
            # Adversarial training
            'use_discriminator': True
        },
        
        'data': {
            'sample_rate': 16000,
            'duration': 10.0,
            'metadata_path': 'data/dataset/metadata/train.csv',
            'audio_dir': 'data/dataset/audio/',
            'val_metadata_path': 'data/dataset/metadata/val.csv',
            'val_audio_dir': 'data/dataset/audio/'
        },
        
        'logging': {
            'log_dir': 'logs/vae/',
            'save_top_k': 3,
            'monitor': 'val/total_loss'
        }
    }
    
    return OmegaConf.create(config)

# =============================================================================
# MAIN VAE TRAINING FUNCTION
# =============================================================================

def train_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--reload_from_ckpt', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = get_vae_config()
    
    # Setup datasets
    train_dataset = VAEAudioDataset(
        config.data.metadata_path,
        config.data.audio_dir,
        config.data.sample_rate,
        config.data.duration
    )
    
    val_dataset = VAEAudioDataset(
        config.data.val_metadata_path,
        config.data.val_audio_dir,
        config.data.sample_rate,
        config.data.duration
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        drop_last=False
    )
    
    # Initialize model
    model = VAETrainingModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.logging.log_dir, 'checkpoints'),
        filename='vae-{epoch:02d}-{val/total_loss:.2f}',
        save_top_k=config.logging.save_top_k,
        monitor=config.logging.monitor,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(config.logging.log_dir, name='vae_training')
    
    # Trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        val_check_interval=0.25
    )
    
    # Load checkpoint if specified
    if args.reload_from_ckpt:
        model = VAETrainingModule.load_from_checkpoint(
            args.reload_from_ckpt,
            config=config
        )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("VAE training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    train_vae()