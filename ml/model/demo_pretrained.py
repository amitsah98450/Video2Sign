"""
TPS Motion Model — Demo with Pre-trained Weights
Downloads and loads official VoxCeleb checkpoint (vox.pth.tar)
to generate photorealistic animated output from sign language keypoints.

Usage:
  cd "Video to Sign"
  python -m ml.model.demo_pretrained
"""

import os
import sys
import time
import math
import yaml
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
sys.path.insert(0, PROJECT_ROOT)

# Paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, 'demo_output')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoints', 'vox.pth.tar')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.yaml')
SOURCE_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'source_face.png')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Keypoint Generator (reused from demo_tps.py) ────────────────────────────

NEUTRAL_POSE = np.array([
    [0.0, -0.65],    # 0  nose
    [0.0, -0.50],    # 1  neck
    [-0.25, -0.45],  # 2  r_shoulder
    [-0.30, -0.20],  # 3  r_elbow
    [-0.25, 0.05],   # 4  r_wrist
    [0.25, -0.45],   # 5  l_shoulder
    [0.30, -0.20],   # 6  l_elbow
    [0.25, 0.05],    # 7  l_wrist
    [-0.27, 0.12], [-0.23, 0.14], [-0.25, 0.15],
    [-0.28, 0.14], [-0.30, 0.12],
    [0.27, 0.12], [0.23, 0.14], [0.25, 0.15],
    [0.28, 0.14], [0.30, 0.12],
    [-0.06, -0.68], [0.06, -0.68],
    [-0.12, -0.65], [0.12, -0.65],
    [0.0, -0.30], [-0.15, -0.05], [0.15, -0.05],
], dtype=np.float32)

SKELETON = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 22), (22, 23), (22, 24),
    (0, 18), (0, 19), (18, 20), (19, 21),
    (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
    (7, 13), (7, 14), (7, 15), (7, 16), (7, 17),
]


def interpolate_poses(pose_a, pose_b, num_frames):
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        t = 0.5 - 0.5 * math.cos(t * math.pi)
        frames.append(pose_a * (1 - t) + pose_b * t)
    return frames


def make_hello_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.35, -0.55]; p1[4] = [-0.45, -0.70]
    p1[8:13] = [[-0.50, -0.78], [-0.47, -0.80], [-0.44, -0.80],
                [-0.41, -0.78], [-0.38, -0.76]]
    p2 = p1.copy()
    p2[4] = [-0.35, -0.70]
    p2[8:13] = [[-0.40, -0.78], [-0.37, -0.80], [-0.34, -0.80],
                [-0.31, -0.78], [-0.28, -0.76]]
    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 6)
    frames += interpolate_poses(p1, p2, 5)
    frames += interpolate_poses(p2, p1, 5)
    frames += interpolate_poses(p1, p2, 5)
    frames += interpolate_poses(p2, NEUTRAL_POSE, 4)
    return frames


def make_how_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.15, -0.25]; p1[4] = [-0.10, -0.15]
    p1[6] = [0.15, -0.25];  p1[7] = [0.10, -0.15]
    for i in range(8, 13):
        p1[i] = p1[4] + np.array([(i-10)*0.03, -0.04])
    for i in range(13, 18):
        p1[i] = p1[7] + np.array([(i-15)*0.03, -0.04])
    p2 = p1.copy()
    p2[3] = [-0.30, -0.30]; p2[4] = [-0.35, -0.20]
    p2[6] = [0.30, -0.30];  p2[7] = [0.35, -0.20]
    for i in range(8, 13):
        p2[i] = p2[4] + np.array([(i-10)*0.03, -0.04])
    for i in range(13, 18):
        p2[i] = p2[7] + np.array([(i-15)*0.03, -0.04])
    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 5)
    frames += interpolate_poses(p1, p2, 8)
    frames += interpolate_poses(p2, NEUTRAL_POSE, 5)
    return frames


def make_are_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.50]; p1[4] = [-0.05, -0.60]
    p1[8:13] = [[-0.05, -0.65], [-0.07, -0.58], [-0.08, -0.57],
                [-0.09, -0.56], [-0.10, -0.55]]
    p2 = p1.copy()
    p2[4] = [-0.05, -0.55]
    p2[8:13] = [[0.0, -0.58], [-0.07, -0.53], [-0.08, -0.52],
                [-0.09, -0.51], [-0.10, -0.50]]
    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 5)
    frames += interpolate_poses(p1, p2, 8)
    frames += interpolate_poses(p2, NEUTRAL_POSE, 5)
    return frames


def make_you_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.35]; p1[4] = [0.0, -0.30]
    p1[8:13] = [[0.05, -0.32], [-0.02, -0.28], [-0.03, -0.27],
                [-0.04, -0.26], [-0.05, -0.25]]
    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 6)
    frames += [p1] * 8
    frames += interpolate_poses(p1, NEUTRAL_POSE, 6)
    return frames


def generate_sign_sequence():
    sequence, labels = [], []
    signs = [
        ("HELLO", make_hello_poses()),
        ("HOW", make_how_poses()),
        ("ARE", make_are_poses()),
        ("YOU", make_you_poses()),
    ]
    for sign_name, frames in signs:
        for f in frames:
            sequence.append(f)
            labels.append(sign_name)
        for _ in range(3):
            sequence.append(NEUTRAL_POSE.copy())
            labels.append("—")
    return np.array(sequence), labels


# ─── Visualization ────────────────────────────────────────────────────────────

def get_joint_color(idx):
    if idx in [0, 18, 19, 20, 21]: return (255, 200, 100)
    elif idx in [1, 22, 23, 24]:   return (100, 200, 255)
    elif idx in [2, 3, 4]:         return (255, 100, 100)
    elif idx in [5, 6, 7]:         return (100, 255, 100)
    elif 8 <= idx <= 12:           return (255, 150, 150)
    elif 13 <= idx <= 17:          return (150, 255, 150)
    return (200, 200, 200)


def draw_keypoints(kps, size=256, label="", bg_color=(20, 22, 35)):
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    def to_px(x, y):
        return int((x+1)/2*(size-40)+20), int((y+1)/2*(size-40)+20)
    for a, b in SKELETON:
        ax, ay = to_px(kps[a][0], kps[a][1])
        bx, by = to_px(kps[b][0], kps[b][1])
        draw.line([(ax, ay), (bx, by)], fill=get_joint_color(a), width=2)
    for i, (x, y) in enumerate(kps):
        px, py = to_px(x, y)
        r = 4 if i < 8 or i >= 18 else 2
        draw.ellipse([px-r, py-r, px+r, py+r],
                     fill=get_joint_color(i), outline='white')
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font = ImageFont.load_default()
        draw.text((8, size-28), label, fill=(255,255,255), font=font)
    return img


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model_with_checkpoint(config_path, checkpoint_path, device):
    """
    Load TPS model and map official checkpoint keys to our model.

    Official checkpoint keys:
      - 'kp_detector'          → model.kp_detector
      - 'dense_motion_network' → model.dense_motion
      - 'inpainting_network'   → model.inpainting
      - 'bg_predictor'         → model.bg_predictor
    """
    from ml.model.model import TPSMotionModel

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = TPSMotionModel(config).to(device)

    print(f"📦 Loading checkpoint: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Map official checkpoint module names to our model's module names
    key_mapping = {
        'kp_detector':          model.kp_detector,
        'dense_motion_network': model.dense_motion,
        'inpainting_network':   model.inpainting,
        'bg_predictor':         model.bg_predictor,
    }

    loaded_modules = []
    skipped_modules = []

    for ckpt_key, module in key_mapping.items():
        if ckpt_key in checkpoint and module is not None:
            try:
                module.load_state_dict(checkpoint[ckpt_key], strict=False)
                loaded_modules.append(ckpt_key)
            except Exception as e:
                print(f"  ⚠️  Partial load for {ckpt_key}: {e}")
                # Try loading with strict=False (allows mismatched keys)
                loaded_modules.append(f"{ckpt_key} (partial)")
        else:
            skipped_modules.append(ckpt_key)

    # Also try loading AVD network if present
    if 'avd_network' in checkpoint and model.avd_network is not None:
        try:
            model.avd_network.load_state_dict(checkpoint['avd_network'], strict=False)
            loaded_modules.append('avd_network')
        except:
            skipped_modules.append('avd_network')

    print(f"  ✅ Loaded:  {', '.join(loaded_modules)}")
    if skipped_modules:
        print(f"  ⏭  Skipped: {', '.join(skipped_modules)}")

    model.eval()
    return model, config


def load_source_image(path, size=256, device='cpu'):
    """Load and preprocess source image for TPS model."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to tensor [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device), img


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, source_img_tensor, keypoint_sequence, config, device):
    """
    Run TPS model inference on each keypoint frame.

    The model's KPDetector extracts 50 TPS control points from the source.
    We use THOSE as source keypoints, then for driving we create small
    perturbations based on our sign language poses to create motion.
    """
    num_tps = config['model_params']['common_params']['num_tps']
    expected_kps = num_tps * 5  # 50

    print(f"\n🔄 Running inference on {len(keypoint_sequence)} frames...")
    print(f"   Device: {device}")
    print(f"   TPS control points: {expected_kps}")

    generated_frames = []
    times = []

    with torch.no_grad():
        # Get source keypoints from the KP detector
        source_kps = model.kp_detector(source_img_tensor)
        source_fg_kp = source_kps['fg_kp']  # (1, 50, 2)
        print(f"   Source KP shape: {source_fg_kp.shape}")

        # Use the first frame's pose as the "neutral" reference
        neutral_pose = keypoint_sequence[0]

        for i, frame_kps in enumerate(keypoint_sequence):
            # Compute the delta (movement) from neutral pose
            delta = frame_kps - neutral_pose  # (25, 2)

            # Pad to 50 keypoints by duplicating
            padded_delta = np.zeros((expected_kps, 2), dtype=np.float32)
            n = min(len(delta), expected_kps)
            padded_delta[:n] = delta[:n]
            for j in range(n, expected_kps):
                padded_delta[j] = delta[j % n]

            # Scale the delta to create subtle motion
            # (our poses are in [-1,1] but TPS expects small movements)
            scale_factor = 0.15
            delta_tensor = torch.tensor(
                padded_delta * scale_factor, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Apply delta to source keypoints → driving keypoints
            driving_kp = {'fg_kp': source_fg_kp + delta_tensor}

            # Run model
            start = time.time()
            prediction = model.animate(
                source_img_tensor, driving_kp, source_kps
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

            # Convert to numpy image
            frame = prediction.squeeze(0).permute(1, 2, 0)
            frame = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            generated_frames.append(frame)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"   Frame {i+1}/{len(keypoint_sequence)}: {elapsed*1000:.0f}ms")

    times = np.array(times)
    print(f"\n📈 Performance:")
    print(f"   Total:     {times.sum():.1f}s")
    print(f"   Avg/frame: {times.mean()*1000:.0f}ms")
    print(f"   FPS:       {1/times.mean():.1f}")

    return generated_frames, times


# ─── Save Outputs ─────────────────────────────────────────────────────────────

def save_comparison_strip(source_pil, sequence, labels, gen_frames, output_path):
    """Create side-by-side comparison: source | keypoints | generated."""
    num_samples = 8
    step = max(1, len(sequence) // num_samples)
    indices = list(range(0, len(sequence), step))[:num_samples]

    tile_size = 200
    cols = len(indices)
    rows = 3  # source row, keypoints row, generated row
    strip_w = tile_size * cols
    strip_h = tile_size * rows + 120  # room for labels

    strip = Image.new('RGB', (strip_w, strip_h), (20, 22, 35))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        font_title = font

    # Row labels
    row_labels = ["Source", "Keypoints", "TPS Output"]
    for r, rl in enumerate(row_labels):
        y = r * (tile_size + 5) + tile_size // 2 - 8
        # Will be placed to left, but let's just put at top of each row
        draw.text((5, r * (tile_size + 5) + 2), rl,
                  fill=(200, 200, 200), font=font)

    for col, idx in enumerate(indices):
        x = col * tile_size

        # Row 0: Source image (same for all)
        src_tile = source_pil.resize((tile_size, tile_size), Image.LANCZOS)
        strip.paste(src_tile, (x, 0))

        # Row 1: Keypoint skeleton
        kp_img = draw_keypoints(sequence[idx], size=tile_size)
        strip.paste(kp_img, (x, tile_size + 5))

        # Row 2: Generated frame
        gen_tile = Image.fromarray(gen_frames[idx]).resize(
            (tile_size, tile_size), Image.LANCZOS
        )
        strip.paste(gen_tile, (x, 2 * (tile_size + 5)))

        # Label
        label = labels[idx] if idx < len(labels) else ""
        draw.text((x + 5, 3 * (tile_size + 5) + 2),
                  f"F{idx}: {label}", fill=(200, 200, 200), font=font)

    strip.save(output_path)
    print(f"💾 Comparison strip: {output_path}")


def save_gif(frames, output_path, fps=12):
    pil_frames = [Image.fromarray(f) for f in frames]
    duration = int(1000 / fps)
    pil_frames[0].save(
        output_path, save_all=True,
        append_images=pil_frames[1:],
        duration=duration, loop=0,
    )
    print(f"💾 GIF saved: {output_path} ({len(frames)} frames)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤟 TPS Motion Model — Pre-trained Demo")
    print("   Phrase: 'Hello, how are you?'")
    print("=" * 60)

    # Check checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"\n📥 Please download vox.pth.tar from:")
        print(f"   https://drive.google.com/drive/folders/1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT")
        print(f"\n   Then place it in: ml/model/checkpoints/vox.pth.tar")
        return

    # Check source image exists
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"\n❌ Source image not found: {SOURCE_IMAGE_PATH}")
        print(f"   Please place a face photo as: {SOURCE_IMAGE_PATH}")
        return

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\n🖥️  Device: {device}")

    # 1. Generate sign language keypoints
    print("\n1️⃣  Generating sign language keypoints...")
    sequence, labels = generate_sign_sequence()
    print(f"   {len(sequence)} frames, {sequence.shape[1]} keypoints/frame")

    # 2. Save keypoint visualization
    print("\n2️⃣  Saving keypoint visualization...")
    kp_gif_path = os.path.join(OUTPUT_DIR, 'pretrained_keypoints.gif')
    kp_frames = []
    for i, (kps, label) in enumerate(zip(sequence, labels)):
        kp_frames.append(
            draw_keypoints(kps, size=256, label=f"{label} [{i+1}]")
        )
    kp_frames[0].save(
        kp_gif_path, save_all=True,
        append_images=kp_frames[1:], duration=80, loop=0,
    )
    print(f"   💾 {kp_gif_path}")

    # 3. Load model with pre-trained weights
    print("\n3️⃣  Loading TPS model with pre-trained weights...")
    model, config = load_model_with_checkpoint(CONFIG_PATH, CHECKPOINT_PATH, device)
    params = model.get_num_params()
    print(f"   📊 Parameters: {params['total']:,}")

    # 4. Load source image
    print("\n4️⃣  Loading source image...")
    source_tensor, source_pil = load_source_image(
        SOURCE_IMAGE_PATH, size=256, device=device
    )
    print(f"   Shape: {source_tensor.shape}")

    # 5. Run inference
    print("\n5️⃣  Running TPS inference...")
    gen_frames, times = run_inference(
        model, source_tensor, sequence, config, device
    )

    # 6. Save outputs
    print("\n6️⃣  Saving outputs...")

    # Generated video GIF
    gen_gif_path = os.path.join(OUTPUT_DIR, 'pretrained_output.gif')
    save_gif(gen_frames, gen_gif_path, fps=12)

    # Comparison strip
    comp_path = os.path.join(OUTPUT_DIR, 'pretrained_comparison.png')
    save_comparison_strip(source_pil, sequence, labels, gen_frames, comp_path)

    # Save a few individual frames
    for i in [0, 15, 30, 50, 70, len(gen_frames)-1]:
        if i < len(gen_frames):
            frame_path = os.path.join(OUTPUT_DIR, f'pretrained_frame_{i:03d}.png')
            Image.fromarray(gen_frames[i]).save(frame_path)

    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Phrase:        'Hello, how are you?'")
    print(f"   Frames:        {len(sequence)}")
    print(f"   Total time:    {times.sum():.1f}s")
    print(f"   Avg/frame:     {times.mean()*1000:.0f}ms")
    print(f"   FPS:           {1/times.mean():.1f}")
    print(f"   Device:        {device}")
    print(f"\n   Output: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('pretrained_'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"     📄 {f} ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == '__main__':
    main()
