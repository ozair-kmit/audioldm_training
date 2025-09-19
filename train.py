from audioldm_train.train.latent_diffusion import AudioLDMTrainingModule

from audioldm_train.data.dataset import AudioCapsDataset
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitorv
from torch.utils.data import Dataset, DataLoader

def get_default_config():
    config = {
        # Model architecture
        'vae': {
            'in_channels': 1,
            'out_channels': 1,
            'hidden_dim': 256,
            'latent_dim': 8
        },
        
        'diffusion': {
            'latent_dim': 8,
            'hidden_dim': 512,
            'time_embed_dim': 128,
            'text_embed_dim': 512
        },
        
        'clap': {
            'model_name': 'microsoft/DialoGPT-medium',  # Simplified placeholder
            'hidden_size': 1024,
            'embed_dim': 512
        },
        
        # Training parameters
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'batch_size': 8,
        'max_epochs': 100,
        'num_workers': 4,
        
        # Data
        'sample_rate': 16000,
        'duration': 10.0,
        'metadata_path': 'data/audiocaps/train.csv',
        'audio_dir': 'data/audiocaps/audio/',
        'val_metadata_path': 'data/audiocaps/val.csv',
        'val_audio_dir': 'data/audiocaps/audio/',
        
        # Logging
        'log_dir': 'logs/',
        'save_top_k': 3,
        'monitor': 'val_loss'
    }
    
    return OmegaConf.create(config)

# =============================================================================
# 7. MAIN TRAINING FUNCTION
# =============================================================================

def train_audioldm():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--reload_from_ckpt', type=str, default=None)
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = get_default_config()
    
    # Setup data
    train_dataset = AudioCapsDataset(
        config.metadata_path,
        config.audio_dir,
        config.sample_rate,
        config.duration
    )
    
    val_dataset = AudioCapsDataset(
        config.val_metadata_path,
        config.val_audio_dir,
        config.sample_rate,
        config.duration
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    # Initialize model
    model = AudioLDMTrainingModule(config)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.log_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(config.log_dir, name='audioldm')
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        val_check_interval=0.25
    )
    
    # Load from checkpoint if specified
    if args.reload_from_ckpt:
        model = AudioLDMTrainingModule.load_from_checkpoint(
            args.reload_from_ckpt, 
            config=config
        )
    
    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train_audioldm()
