"""
TPS Motion Model — Training Script
Multi-scale perceptual loss, equivariance loss, warp loss, 
BG loss, and TPS dropout scheduling.

Based on: "Thin-Plate Spline Motion Model for Image Animation" (CVPR 2022)

Usage:
    python -m ml.model.train --config ml/model/config.yaml
    
    OR in Google Colab:
    from ml.model.train import Trainer
    trainer = Trainer(config, dataset)
    trainer.train()
"""

import os
import sys
import yaml
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from ml.model.model import TPSMotionModel
from ml.model.dataset import VideoDataset
from ml.model.util import TPS, make_coordinate_grid


# ─── Perceptual Loss (VGG19) ────────────────────────────────────────────────

class Vgg19(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    Extracts features at 5 different depths.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(
            pretrained=True
        ).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


import torch.nn as nn


class PerceptualLoss(nn.Module):
    """Multi-scale perceptual loss using VGG19 features."""

    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = weights or [10, 10, 10, 10, 10]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(
                x_vgg[i], y_vgg[i].detach()
            )
        return loss


# ─── Equivariance Loss ──────────────────────────────────────────────────────

def equivariance_loss(kp_detector, source_image, transform_params):
    """
    Equivariance loss: keypoints of transformed image should equal
    the transformation of keypoints.
    """
    # Random TPS transformation
    bs = source_image.shape[0]
    tps = TPS(mode='random', bs=bs, **transform_params)

    # Transform the image
    grid = tps.transform_frame(source_image)
    transformed_image = F.grid_sample(
        source_image, grid, align_corners=True
    )

    # Detect keypoints in both
    kp_original = kp_detector(source_image)
    kp_transformed = kp_detector(transformed_image)

    # Transform original keypoints
    kp_original_2d = kp_original['fg_kp']
    kp_warped = tps.warp_coordinates(kp_original_2d)

    # L1 loss between transformed KPs and KPs from transformed image
    loss = F.l1_loss(kp_warped, kp_transformed['fg_kp'])
    return loss


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training pipeline for the TPS Motion Model.
    
    Args:
        config (dict): Configuration from config.yaml
        dataset: PyTorch Dataset
        device (str): 'cuda' or 'cpu'
        checkpoint_dir (str): Where to save checkpoints
    """

    def __init__(self, config, dataset, device='cuda',
                 checkpoint_dir='checkpoints'):
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_params = config['train_params']

        # Model
        self.model = TPSMotionModel(config).to(device)
        param_counts = self.model.get_num_params()
        total = param_counts['total']
        print(f"✅ Model initialized — {total:,} parameters")
        for name, count in param_counts.items():
            if name != 'total':
                print(f"   {name}: {count:,}")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_params['lr_generator'],
            betas=(0.5, 0.999),
        )

        # LR Scheduler
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=train_params['epoch_milestones'],
            gamma=0.1,
        )

        # DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params.get('dataloader_workers', 4),
            drop_last=True,
        )

        # Losses
        self.perceptual_loss = PerceptualLoss(
            weights=train_params['loss_weights']['perceptual']
        ).to(device)

        # Training params
        self.num_epochs = train_params['num_epochs']
        self.scales = train_params['scales']
        self.loss_weights = train_params['loss_weights']
        self.transform_params = train_params['transform_params']
        self.checkpoint_freq = train_params['checkpoint_freq']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_startp = train_params['dropout_startp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.bg_start = train_params['bg_start']

    def get_dropout_p(self, epoch):
        """Calculate TPS dropout probability for current epoch."""
        if epoch < self.dropout_epoch:
            return 0
        p = self.dropout_startp + (
            (self.dropout_maxp - self.dropout_startp)
            * min(1, (epoch - self.dropout_epoch) / self.dropout_inc_epoch)
        )
        return p

    def compute_losses(self, source, driving, output, epoch):
        """Compute all training losses."""
        losses = {}

        # 1. Multi-scale perceptual loss
        prediction = output['prediction']
        perceptual = 0
        for scale in self.scales:
            if scale != 1:
                pred_scaled = F.interpolate(
                    prediction, scale_factor=scale, mode='bilinear',
                    align_corners=True
                )
                driv_scaled = F.interpolate(
                    driving, scale_factor=scale, mode='bilinear',
                    align_corners=True
                )
            else:
                pred_scaled = prediction
                driv_scaled = driving
            perceptual += self.perceptual_loss(pred_scaled, driv_scaled)
        losses['perceptual'] = perceptual

        # 2. Equivariance loss
        if self.loss_weights.get('equivariance_value', 0) > 0:
            eq_loss = equivariance_loss(
                self.model.kp_detector, driving,
                self.transform_params
            )
            losses['equivariance'] = (
                self.loss_weights['equivariance_value'] * eq_loss
            )

        # 3. Warp loss (L1 between warped encoder maps)
        if self.loss_weights.get('warp_loss', 0) > 0:
            warped_maps = output.get('warped_encoder_maps', [])
            warp_loss = 0
            for wm in warped_maps:
                warp_loss += torch.abs(wm).mean()
            if warped_maps:
                warp_loss /= len(warped_maps)
            losses['warp'] = self.loss_weights['warp_loss'] * warp_loss

        # 4. Background loss (after bg_start epoch)
        if (epoch >= self.bg_start and
                self.loss_weights.get('bg', 0) > 0 and
                'bg_param' in output):
            bg_param = output['bg_param']
            identity = torch.eye(3).unsqueeze(0).to(bg_param.device)
            identity = identity.repeat(bg_param.shape[0], 1, 1)
            bg_loss = F.l1_loss(bg_param, identity)
            losses['bg'] = self.loss_weights['bg'] * bg_loss

        return losses

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        dropout_p = self.get_dropout_p(epoch)
        dropout_flag = dropout_p > 0

        for batch_idx, batch in enumerate(self.dataloader):
            source = batch['source'].to(self.device)
            driving = batch['driving'].to(self.device)

            # Forward
            output = self.model(
                source, driving,
                dropout_flag=dropout_flag,
                dropout_p=dropout_p,
            )

            # Losses
            losses = self.compute_losses(
                source, driving, output, epoch
            )
            total_loss = sum(losses.values())

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                loss_str = ' | '.join(
                    f'{k}: {v.item():.4f}' for k, v in losses.items()
                )
                print(f"  Batch {batch_idx}/{len(self.dataloader)} — "
                      f"{loss_str}")

        # Average losses
        avg_losses = {
            k: v / num_batches for k, v in epoch_losses.items()
        }
        return avg_losses

    def save_checkpoint(self, epoch, losses):
        """Save model checkpoint."""
        path = os.path.join(
            self.checkpoint_dir, f'tps_epoch_{epoch:03d}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
        }, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {checkpoint['epoch']}")
        return start_epoch

    def train(self, start_epoch=0):
        """Full training loop."""
        print("=" * 60)
        print("🚀 TPS Motion Model Training")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Batch size: {self.config['train_params']['batch_size']}")
        print(f"   Dataset size: {len(self.dataloader.dataset)}")
        print(f"   Device: {self.device}")
        print("=" * 60)

        for epoch in range(start_epoch, self.num_epochs):
            dropout_p = self.get_dropout_p(epoch)
            print(f"\n📈 Epoch {epoch + 1}/{self.num_epochs} "
                  f"(lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"dropout_p={dropout_p:.3f})")

            avg_losses = self.train_epoch(epoch)

            loss_str = ' | '.join(
                f'{k}: {v:.4f}' for k, v in avg_losses.items()
            )
            print(f"   Avg: {loss_str}")
            total = sum(avg_losses.values())
            print(f"   Total: {total:.4f}")

            self.scheduler.step()

            if (epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch, avg_losses)

        # Final checkpoint
        self.save_checkpoint(self.num_epochs - 1, avg_losses)
        print("\n✅ Training complete!")


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train TPS Motion Model'
    )
    parser.add_argument(
        '--config', type=str,
        default='ml/model/config.yaml',
        help='Path to config YAML'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str,
        default='checkpoints/tps',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu/mps)'
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Pick device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Dataset
    dataset_params = config['dataset_params']
    dataset = VideoDataset(
        root_dir=dataset_params['root_dir'],
        frame_shape=tuple(dataset_params.get('frame_shape', [256, 256])),
        id_sampling=dataset_params.get('id_sampling', True),
        augmentation_params=dataset_params.get('augmentation_params'),
        num_repeats=config['train_params'].get('num_repeats', 75),
    )

    if len(dataset) == 0:
        print("⚠️  No training data found!")
        print(f"   Expected video folders in: {dataset_params['root_dir']}")
        print("   Each folder should contain frame images (.png/.jpg)")
        return

    print(f"📂 Dataset: {len(dataset.videos)} videos, "
          f"{len(dataset)} samples/epoch")

    # Trainer
    trainer = Trainer(
        config=config,
        dataset=dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume if checkpoint given
    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train(start_epoch=start_epoch)


if __name__ == '__main__':
    main()
