"""
TPS Motion Model — Keypoint Detector
ResNet18-based module that predicts K×5 TPS control keypoints from an image.

Based on: "Thin-Plate Spline Motion Model for Image Animation" (CVPR 2022)
"""

from torch import nn
import torch
from torchvision import models


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints per image using a ResNet18 backbone.
    
    Each of the K TPS transformations uses 5 control points.
    Output keypoints are in range [-1, 1].
    
    Args:
        num_tps (int): Number of TPS transformations (default: 10)
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) input image
            
        Returns:
            dict with 'fg_kp': (B, K*5, 2) keypoints in [-1, 1]
        """
        fg_kp = self.fg_encoder(image)
        bs = fg_kp.shape[0]
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1  # Map to [-1, 1]
        out = {'fg_kp': fg_kp.view(bs, self.num_tps * 5, -1)}
        return out
