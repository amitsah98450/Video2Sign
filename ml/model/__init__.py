"""
TPS Motion Model Package
Thin-Plate Spline Motion Model for Image Animation (CVPR 2022)

Modules:
    - KPDetector: Keypoint detection (ResNet18)
    - BGMotionPredictor: Background affine motion
    - DenseMotionNetwork: Optical flow + occlusion estimation
    - InpaintingNetwork: Frame generation/reconstruction
    - AVDNetwork: Identity-pose disentanglement
    - TPSMotionModel: Full combined model
"""

from ml.model.model import TPSMotionModel
from ml.model.keypoint_detector import KPDetector
from ml.model.bg_motion_predictor import BGMotionPredictor
from ml.model.dense_motion import DenseMotionNetwork
from ml.model.inpainting_network import InpaintingNetwork
from ml.model.avd_network import AVDNetwork
from ml.model.inference import TPSAnimator

__all__ = [
    'TPSMotionModel',
    'KPDetector',
    'BGMotionPredictor',
    'DenseMotionNetwork',
    'InpaintingNetwork',
    'AVDNetwork',
    'TPSAnimator',
]
