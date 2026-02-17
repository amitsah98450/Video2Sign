"""
TPS Motion Model — Video Dataset
Self-supervised dataset for training: samples random pairs of frames
from the same video for learning motion transfer.

Based on: "Thin-Plate Spline Motion Model for Image Animation" (CVPR 2022)
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class VideoDataset(Dataset):
    """
    Dataset for self-supervised motion model training.
    
    Each sample returns a pair of frames from the same video.
    The model learns to reconstruct the driving frame from the source.
    
    Expected directory structure:
        root_dir/
            video_001/
                frame_0000.png
                frame_0001.png
                ...
            video_002/
                ...
    
    Args:
        root_dir (str): Root directory containing video folders
        frame_shape (tuple): Target frame size (H, W)
        id_sampling (bool): Sample pairs from same identity
        augmentation_params (dict): Augmentation config
        num_repeats (int): How many pairs to sample per video per epoch
    """

    def __init__(self, root_dir, frame_shape=(256, 256),
                 id_sampling=True, augmentation_params=None,
                 num_repeats=75):
        self.root_dir = root_dir
        self.frame_shape = frame_shape
        self.id_sampling = id_sampling
        self.num_repeats = num_repeats

        # Discover videos
        self.videos = []
        if os.path.exists(root_dir):
            for vid_name in sorted(os.listdir(root_dir)):
                vid_path = os.path.join(root_dir, vid_name)
                if os.path.isdir(vid_path):
                    frames = self._get_frames(vid_path)
                    if len(frames) >= 2:
                        self.videos.append({
                            'name': vid_name,
                            'path': vid_path,
                            'frames': frames,
                        })

        # Image transform
        transforms = [
            T.Resize(frame_shape),
            T.ToTensor(),
        ]

        # Augmentation
        if augmentation_params:
            jitter = augmentation_params.get('jitter_param', {})
            if jitter:
                transforms.insert(1, T.ColorJitter(
                    brightness=jitter.get('brightness', 0),
                    contrast=jitter.get('contrast', 0),
                    saturation=jitter.get('saturation', 0),
                    hue=jitter.get('hue', 0),
                ))
            self.horizontal_flip = augmentation_params.get(
                'flip_param', {}
            ).get('horizontal_flip', False)
            self.time_flip = augmentation_params.get(
                'flip_param', {}
            ).get('time_flip', False)
        else:
            self.horizontal_flip = False
            self.time_flip = False

        self.transform = T.Compose(transforms)

    def _get_frames(self, vid_path):
        """Get sorted list of frame paths in a video directory."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        frames = []
        for f in sorted(os.listdir(vid_path)):
            if os.path.splitext(f)[1].lower() in extensions:
                frames.append(os.path.join(vid_path, f))
        return frames

    def __len__(self):
        return len(self.videos) * self.num_repeats

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                'source': (3, H, W) source frame tensor
                'driving': (3, H, W) driving frame tensor
        """
        vid_idx = idx % len(self.videos)
        video = self.videos[vid_idx]
        frames = video['frames']

        # Sample two random frames
        if len(frames) >= 2:
            idx_source, idx_driving = random.sample(
                range(len(frames)), 2
            )
        else:
            idx_source = idx_driving = 0

        # Time flip augmentation
        if self.time_flip and random.random() > 0.5:
            idx_source, idx_driving = idx_driving, idx_source

        source = Image.open(frames[idx_source]).convert('RGB')
        driving = Image.open(frames[idx_driving]).convert('RGB')

        # Horizontal flip (apply same flip to both)
        if self.horizontal_flip and random.random() > 0.5:
            source = source.transpose(Image.FLIP_LEFT_RIGHT)
            driving = driving.transpose(Image.FLIP_LEFT_RIGHT)

        source = self.transform(source)
        driving = self.transform(driving)

        return {
            'source': source,
            'driving': driving,
        }


class SignVideoDataset(Dataset):
    """
    Dataset specifically for sign language videos with keypoints.
    
    Uses How2Sign or similar datasets where we have:
    - Video frames (extracted as images)
    - 2D keypoints per frame
    
    Args:
        video_dir (str): Directory containing video frame folders
        keypoint_dir (str): Directory containing keypoint files
        frame_shape (tuple): Target frame size (H, W)
        num_repeats (int): Pairs per video per epoch
    """

    def __init__(self, video_dir, keypoint_dir=None,
                 frame_shape=(256, 256), num_repeats=75):
        self.video_dir = video_dir
        self.keypoint_dir = keypoint_dir
        self.frame_shape = frame_shape
        self.num_repeats = num_repeats

        self.transform = T.Compose([
            T.Resize(frame_shape),
            T.ToTensor(),
        ])

        # Discover video folders
        self.videos = []
        if os.path.exists(video_dir):
            for vid_name in sorted(os.listdir(video_dir)):
                vid_path = os.path.join(video_dir, vid_name)
                if os.path.isdir(vid_path):
                    frames = self._get_frames(vid_path)
                    if len(frames) >= 2:
                        self.videos.append({
                            'name': vid_name,
                            'path': vid_path,
                            'frames': frames,
                        })

        print(f"[SignVideoDataset] Found {len(self.videos)} videos "
              f"with ≥2 frames")

    def _get_frames(self, vid_path):
        extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        frames = []
        for f in sorted(os.listdir(vid_path)):
            if os.path.splitext(f)[1].lower() in extensions:
                frames.append(os.path.join(vid_path, f))
        return frames

    def __len__(self):
        return len(self.videos) * self.num_repeats

    def __getitem__(self, idx):
        vid_idx = idx % len(self.videos)
        video = self.videos[vid_idx]
        frames = video['frames']

        idx_source, idx_driving = random.sample(
            range(len(frames)), 2
        )

        source = Image.open(frames[idx_source]).convert('RGB')
        driving = Image.open(frames[idx_driving]).convert('RGB')

        source = self.transform(source)
        driving = self.transform(driving)

        result = {
            'source': source,
            'driving': driving,
        }

        # Load keypoints if available
        if self.keypoint_dir:
            kp_path = os.path.join(
                self.keypoint_dir, video['name'],
                f'{idx_driving:06d}.npy'
            )
            if os.path.exists(kp_path):
                kps = np.load(kp_path)
                result['driving_keypoints'] = torch.tensor(
                    kps, dtype=torch.float32
                )

        return result
