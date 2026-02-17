"""
TPS Motion Model — Inference Pipeline
Generate animated video frames from a source image and driving keypoints.

Usage:
    from ml.model.inference import TPSAnimator
    
    animator = TPSAnimator('checkpoints/tps/tps_epoch_099.pt', 
                           'ml/model/config.yaml')
    frames = animator.animate(source_image, driving_keypoints)
    animator.save_video(frames, 'output.mp4')
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from ml.model.model import TPSMotionModel


class TPSAnimator:
    """
    Inference wrapper for the TPS Motion Model.
    
    Generates animated video frames by transferring motion from
    driving keypoints onto a source image.
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        config_path (str): Path to config YAML
        device (str): 'cuda', 'mps', or 'cpu'
    """

    def __init__(self, checkpoint_path, config_path, device=None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Pick device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif (hasattr(torch.backends, 'mps') and
              torch.backends.mps.is_available()):
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Load model
        self.model = TPSMotionModel(self.config).to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', '?')
            print(f"✅ Model loaded from epoch {epoch}")
        else:
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
            print("   Using randomly initialized model")

        self.model.eval()

        # Image preprocessing
        frame_shape = self.config['dataset_params'].get(
            'frame_shape', [256, 256]
        )
        self.frame_shape = tuple(frame_shape)
        self.transform = T.Compose([
            T.Resize(self.frame_shape),
            T.ToTensor(),
        ])

    def load_source_image(self, image_path):
        """Load and preprocess a source image."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def preprocess_keypoints(self, keypoints):
        """
        Preprocess driving keypoints to model format.
        
        Args:
            keypoints: numpy array of shape:
                - (T, K, 2): sequence of keypoint frames
                - (T, K*2): flattened keypoints
                
        Returns:
            list of dicts with 'fg_kp': (1, K*5, 2) tensors
        """
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.tensor(
                keypoints, dtype=torch.float32
            )

        num_tps = self.config['model_params']['common_params']['num_tps']
        expected_kps = num_tps * 5  # K groups × 5 control points

        processed = []
        for t in range(len(keypoints)):
            frame_kps = keypoints[t]

            # Reshape if flattened
            if len(frame_kps.shape) == 1:
                frame_kps = frame_kps.view(-1, 2)

            # Pad or truncate to expected number of keypoints
            num_kps = frame_kps.shape[0]
            if num_kps < expected_kps:
                padding = torch.zeros(
                    expected_kps - num_kps, 2,
                    dtype=frame_kps.dtype
                )
                frame_kps = torch.cat([frame_kps, padding], dim=0)
            elif num_kps > expected_kps:
                frame_kps = frame_kps[:expected_kps]

            # Normalize to [-1, 1] if not already
            if frame_kps.max() > 1 or frame_kps.min() < -1:
                for dim in range(2):
                    vals = frame_kps[:, dim]
                    vmin, vmax = vals.min(), vals.max()
                    if vmax > vmin:
                        frame_kps[:, dim] = (
                            2 * (vals - vmin) / (vmax - vmin) - 1
                        )

            kp_dict = {
                'fg_kp': frame_kps.unsqueeze(0).to(self.device)
            }
            processed.append(kp_dict)

        return processed

    @torch.no_grad()
    def animate(self, source_image, driving_keypoints,
                source_keypoints=None):
        """
        Generate animated frames.
        
        Args:
            source_image: (1, 3, H, W) tensor or path string
            driving_keypoints: list of KP dicts or numpy array (T, K, 2)
            source_keypoints: optional pre-computed source KPs
            
        Returns:
            list of (H, W, 3) numpy arrays (uint8 frames)
        """
        self.model.eval()

        # Load source if needed
        if isinstance(source_image, str):
            source_image = self.load_source_image(source_image)

        # Get source keypoints
        if source_keypoints is None:
            source_keypoints = self.model.kp_detector(source_image)

        # Process driving keypoints
        if isinstance(driving_keypoints, (np.ndarray, torch.Tensor)):
            driving_keypoints = self.preprocess_keypoints(
                driving_keypoints
            )

        # Generate frames
        frames = []
        for i, kp_driving in enumerate(driving_keypoints):
            prediction = self.model.animate(
                source_image, kp_driving, source_keypoints
            )

            # Convert to numpy image
            frame = prediction.squeeze(0).permute(1, 2, 0)
            frame = (frame.cpu().numpy() * 255).clip(0, 255)
            frame = frame.astype(np.uint8)
            frames.append(frame)

            if (i + 1) % 30 == 0:
                print(f"   Generated {i + 1}/{len(driving_keypoints)} "
                      f"frames")

        print(f"✅ Generated {len(frames)} frames")
        return frames

    @torch.no_grad()
    def animate_from_video(self, source_image, driving_video_path):
        """
        Generate animation using a driving video.
        
        Args:
            source_image: path to source image
            driving_video_path: path to driving video
            
        Returns:
            list of (H, W, 3) numpy arrays
        """
        # Load source
        if isinstance(source_image, str):
            source_image = self.load_source_image(source_image)

        source_kp = self.model.kp_detector(source_image)

        # Extract frames from driving video
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")

        cap = cv2.VideoCapture(driving_video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            driving_tensor = self.transform(img).unsqueeze(0).to(
                self.device
            )

            # Get driving keypoints
            driving_kp = self.model.kp_detector(driving_tensor)

            # Animate
            prediction = self.model.animate(
                source_image, driving_kp, source_kp
            )

            out_frame = prediction.squeeze(0).permute(1, 2, 0)
            out_frame = (out_frame.cpu().numpy() * 255).clip(0, 255)
            frames.append(out_frame.astype(np.uint8))

        cap.release()
        print(f"✅ Generated {len(frames)} frames from driving video")
        return frames

    @staticmethod
    def save_video(frames, output_path, fps=25):
        """Save frames as MP4 video."""
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")

        if not frames:
            print("⚠️  No frames to save")
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        print(f"💾 Video saved: {output_path} "
              f"({len(frames)} frames, {fps} fps)")

    @staticmethod
    def save_gif(frames, output_path, fps=25):
        """Save frames as animated GIF."""
        if not frames:
            print("⚠️  No frames to save")
            return

        images = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
        print(f"💾 GIF saved: {output_path}")
