"""
TPS Motion Model — Full Model
Combines KPDetector, BGMotionPredictor, DenseMotionNetwork,
InpaintingNetwork, and AVDNetwork into a unified model.

Based on: "Thin-Plate Spline Motion Model for Image Animation" (CVPR 2022)
"""

import torch
from torch import nn
from ml.model.keypoint_detector import KPDetector
from ml.model.bg_motion_predictor import BGMotionPredictor
from ml.model.dense_motion import DenseMotionNetwork
from ml.model.inpainting_network import InpaintingNetwork
from ml.model.avd_network import AVDNetwork


class TPSMotionModel(nn.Module):
    """
    Full TPS Motion Model for image animation.
    
    Architecture:
        Source + Driving → KPDetector → keypoints
        Source + Driving → BGMotionPredictor → BG affine
        Keypoints + BG → DenseMotionNetwork → optical flow + occlusion
        Source + flow/occlusion → InpaintingNetwork → animated frame
    
    Args:
        config (dict): Model configuration from config.yaml
    """

    def __init__(self, config):
        super(TPSMotionModel, self).__init__()

        common = config['model_params']['common_params']
        num_tps = common['num_tps']
        num_channels = common['num_channels']
        bg = common.get('bg', True)
        multi_mask = common.get('multi_mask', True)

        # Keypoint Detector
        self.kp_detector = KPDetector(num_tps=num_tps)

        # Background Motion Predictor
        self.bg_predictor = BGMotionPredictor() if bg else None

        # Dense Motion Network
        dm_params = config['model_params']['dense_motion_params']
        self.dense_motion = DenseMotionNetwork(
            block_expansion=dm_params['block_expansion'],
            num_blocks=dm_params['num_blocks'],
            max_features=dm_params['max_features'],
            num_tps=num_tps,
            num_channels=num_channels,
            scale_factor=dm_params.get('scale_factor', 0.25),
            bg=bg,
            multi_mask=multi_mask,
        )

        # Inpainting Network (Generator)
        gen_params = config['model_params']['generator_params']
        self.inpainting = InpaintingNetwork(
            num_channels=num_channels,
            block_expansion=gen_params['block_expansion'],
            max_features=gen_params['max_features'],
            num_down_blocks=gen_params['num_down_blocks'],
            multi_mask=multi_mask,
        )

        # AVD Network (optional, for cross-identity animation)
        avd_params = config['model_params'].get('avd_network_params', None)
        if avd_params:
            self.avd_network = AVDNetwork(
                num_tps=num_tps,
                id_bottle_size=avd_params.get('id_bottle_size', 128),
                pose_bottle_size=avd_params.get('pose_bottle_size', 128),
            )
        else:
            self.avd_network = None

        self.bg = bg
        self.num_tps = num_tps
        self.train_params = config.get('train_params', {})

    def forward(self, source_image, driving_image,
                dropout_flag=False, dropout_p=0):
        """
        Full forward pass for training.
        
        Args:
            source_image: (B, 3, H, W) source/reference image
            driving_image: (B, 3, H, W) target/driving image
            dropout_flag: apply TPS dropout
            dropout_p: dropout probability
            
        Returns:
            dict with 'prediction', 'deformation', 'occlusion_map', etc.
        """
        # Detect keypoints
        kp_source = self.kp_detector(source_image)
        kp_driving = self.kp_detector(driving_image)

        # Background motion
        bg_param = None
        if self.bg_predictor is not None:
            bg_param = self.bg_predictor(source_image, driving_image)

        # Dense motion estimation
        dense_motion_out = self.dense_motion(
            source_image, kp_driving, kp_source,
            bg_param=bg_param,
            dropout_flag=dropout_flag,
            dropout_p=dropout_p,
        )

        # Inpainting (frame generation)
        out = self.inpainting(source_image, dense_motion_out)
        out['kp_source'] = kp_source
        out['kp_driving'] = kp_driving
        if bg_param is not None:
            out['bg_param'] = bg_param

        return out

    @torch.no_grad()
    def animate(self, source_image, driving_keypoints,
                source_keypoints=None):
        """
        Inference mode: animate source with driving keypoints.
        
        Args:
            source_image: (1, 3, H, W) source/reference image
            driving_keypoints: dict with 'fg_kp' (1, K*5, 2)
            source_keypoints: optional pre-computed source KPs
            
        Returns:
            (1, 3, H, W) animated frame
        """
        self.eval()

        if source_keypoints is None:
            source_keypoints = self.kp_detector(source_image)

        bg_param = None
        # For animation, we don't estimate BG from driving
        # (no driving image available, only keypoints)

        dense_motion_out = self.dense_motion(
            source_image, driving_keypoints, source_keypoints,
            bg_param=bg_param,
        )

        result = self.inpainting(source_image, dense_motion_out)
        return result['prediction']

    def get_num_params(self):
        """Get parameter counts for each sub-module."""
        counts = {}
        for name, module in [
            ('kp_detector', self.kp_detector),
            ('bg_predictor', self.bg_predictor),
            ('dense_motion', self.dense_motion),
            ('inpainting', self.inpainting),
            ('avd_network', self.avd_network),
        ]:
            if module is not None:
                counts[name] = sum(
                    p.numel() for p in module.parameters()
                )
        counts['total'] = sum(counts.values())
        return counts
