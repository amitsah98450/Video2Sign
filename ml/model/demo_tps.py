"""
TPS Motion Model — Demo Script
Demonstrates the model pipeline with synthetic sign language keypoints
for the phrase: "hello, how are you?"

Generates:
  1. Stick-figure keypoint visualization (what the model receives)
  2. TPS model forward pass output (untrained, for pipeline validation)
  3. Performance timing statistics
"""

import os
import sys
import time
import math
import yaml
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'demo_output'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Synthetic Body Keypoint Generator ───────────────────────────────────────
# Upper-body skeleton: 25 keypoints representing head, torso, arms, hands
# Layout (normalized to [-1, 1]):
#   0: nose, 1: neck, 2: r_shoulder, 3: r_elbow, 4: r_wrist,
#   5: l_shoulder, 6: l_elbow, 7: l_wrist,
#   8-12: r_hand fingers, 13-17: l_hand fingers,
#   18: r_eye, 19: l_eye, 20: r_ear, 21: l_ear,
#   22: chest, 23: r_hip, 24: l_hip

# Base neutral pose (arms at sides)
NEUTRAL_POSE = np.array([
    [0.0, -0.65],    # 0  nose
    [0.0, -0.50],    # 1  neck
    [-0.25, -0.45],  # 2  r_shoulder
    [-0.30, -0.20],  # 3  r_elbow
    [-0.25, 0.05],   # 4  r_wrist
    [0.25, -0.45],   # 5  l_shoulder
    [0.30, -0.20],   # 6  l_elbow
    [0.25, 0.05],    # 7  l_wrist
    # r_hand fingers (5)
    [-0.27, 0.12], [-0.23, 0.14], [-0.25, 0.15],
    [-0.28, 0.14], [-0.30, 0.12],
    # l_hand fingers (5)
    [0.27, 0.12], [0.23, 0.14], [0.25, 0.15],
    [0.28, 0.14], [0.30, 0.12],
    # face
    [-0.06, -0.68],  # 18 r_eye
    [0.06, -0.68],   # 19 l_eye
    [-0.12, -0.65],  # 20 r_ear
    [0.12, -0.65],   # 21 l_ear
    # torso
    [0.0, -0.30],    # 22 chest
    [-0.15, -0.05],  # 23 r_hip
    [0.15, -0.05],   # 24 l_hip
], dtype=np.float32)

# Skeleton connections for drawing
SKELETON = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 22), (22, 23), (22, 24),
    (0, 18), (0, 19), (18, 20), (19, 21),
    # right hand
    (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
    # left hand
    (7, 13), (7, 14), (7, 15), (7, 16), (7, 17),
]

# Joint colors
JOINT_COLORS = {
    'head': (255, 200, 100),
    'body': (100, 200, 255),
    'r_arm': (255, 100, 100),
    'l_arm': (100, 255, 100),
    'r_hand': (255, 150, 150),
    'l_hand': (150, 255, 150),
}


def get_joint_color(idx):
    if idx in [0, 18, 19, 20, 21]:
        return JOINT_COLORS['head']
    elif idx in [1, 22, 23, 24]:
        return JOINT_COLORS['body']
    elif idx in [2, 3, 4]:
        return JOINT_COLORS['r_arm']
    elif idx in [5, 6, 7]:
        return JOINT_COLORS['l_arm']
    elif 8 <= idx <= 12:
        return JOINT_COLORS['r_hand']
    elif 13 <= idx <= 17:
        return JOINT_COLORS['l_hand']
    return (200, 200, 200)


def interpolate_poses(pose_a, pose_b, num_frames):
    """Smooth interpolation between two poses."""
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        # Ease in-out
        t = 0.5 - 0.5 * math.cos(t * math.pi)
        frame = pose_a * (1 - t) + pose_b * t
        frames.append(frame)
    return frames


# ─── Sign Language Pose Definitions ─────────────────────────────────────────

def make_hello_poses():
    """HELLO: Open hand wave near head (palm facing out)."""
    # Pose 1: Right hand up near head, fingers spread
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.35, -0.55]   # r_elbow up
    p1[4] = [-0.45, -0.70]   # r_wrist near head
    p1[8]  = [-0.50, -0.78]
    p1[9]  = [-0.47, -0.80]
    p1[10] = [-0.44, -0.80]
    p1[11] = [-0.41, -0.78]
    p1[12] = [-0.38, -0.76]

    # Pose 2: Hand tilted right (wave)
    p2 = p1.copy()
    p2[4] = [-0.35, -0.70]
    p2[8]  = [-0.40, -0.78]
    p2[9]  = [-0.37, -0.80]
    p2[10] = [-0.34, -0.80]
    p2[11] = [-0.31, -0.78]
    p2[12] = [-0.28, -0.76]

    # Wave back and forth
    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 6)
    frames += interpolate_poses(p1, p2, 5)
    frames += interpolate_poses(p2, p1, 5)
    frames += interpolate_poses(p1, p2, 5)
    frames += interpolate_poses(p2, NEUTRAL_POSE, 4)
    return frames


def make_how_poses():
    """HOW: Both hands palm up, fingers curled, move outward."""
    p1 = NEUTRAL_POSE.copy()
    # Both hands in front, palms up, close together
    p1[3] = [-0.15, -0.25]
    p1[4] = [-0.10, -0.15]
    p1[6] = [0.15, -0.25]
    p1[7] = [0.10, -0.15]
    for i in range(8, 13):
        p1[i] = p1[4] + np.array([(i-10)*0.03, -0.04])
    for i in range(13, 18):
        p1[i] = p1[7] + np.array([(i-15)*0.03, -0.04])

    p2 = p1.copy()
    # Hands spread outward
    p2[3] = [-0.30, -0.30]
    p2[4] = [-0.35, -0.20]
    p2[6] = [0.30, -0.30]
    p2[7] = [0.35, -0.20]
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
    """ARE: Index finger moves forward from lips."""
    p1 = NEUTRAL_POSE.copy()
    # Right hand near mouth, index pointing forward
    p1[3] = [-0.10, -0.50]
    p1[4] = [-0.05, -0.60]
    p1[8]  = [-0.05, -0.65]   # index forward
    p1[9]  = [-0.07, -0.58]
    p1[10] = [-0.08, -0.57]
    p1[11] = [-0.09, -0.56]
    p1[12] = [-0.10, -0.55]

    p2 = p1.copy()
    # Move hand forward
    p2[4] = [-0.05, -0.55]
    p2[8]  = [0.0, -0.58]
    p2[9]  = [-0.07, -0.53]
    p2[10] = [-0.08, -0.52]
    p2[11] = [-0.09, -0.51]
    p2[12] = [-0.10, -0.50]

    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 5)
    frames += interpolate_poses(p1, p2, 8)
    frames += interpolate_poses(p2, NEUTRAL_POSE, 5)
    return frames


def make_you_poses():
    """YOU: Point index finger at viewer."""
    p1 = NEUTRAL_POSE.copy()
    # Right arm extends forward, index pointing out
    p1[3] = [-0.10, -0.35]
    p1[4] = [0.0, -0.30]
    p1[8]  = [0.05, -0.32]   # index pointing forward
    p1[9]  = [-0.02, -0.28]
    p1[10] = [-0.03, -0.27]
    p1[11] = [-0.04, -0.26]
    p1[12] = [-0.05, -0.25]

    frames = []
    frames += interpolate_poses(NEUTRAL_POSE, p1, 6)
    frames += [p1] * 8  # Hold the point
    frames += interpolate_poses(p1, NEUTRAL_POSE, 6)
    return frames


def generate_sign_sequence():
    """Generate full keypoint sequence for 'hello, how are you?'"""
    sequence = []
    labels = []

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
        # Add brief pause between signs
        for _ in range(3):
            sequence.append(NEUTRAL_POSE.copy())
            labels.append("—")

    return np.array(sequence), labels


# ─── Visualization ───────────────────────────────────────────────────────────

def draw_keypoints(kps, size=512, label="", bg_color=(20, 22, 35)):
    """Draw a single keypoint frame as a stick figure."""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # Convert [-1, 1] to pixel coords
    def to_px(x, y):
        px = int((x + 1) / 2 * (size - 40) + 20)
        py = int((y + 1) / 2 * (size - 40) + 20)
        return px, py

    # Draw skeleton lines
    for (a, b) in SKELETON:
        ax, ay = to_px(kps[a][0], kps[a][1])
        bx, by = to_px(kps[b][0], kps[b][1])
        color_a = get_joint_color(a)
        draw.line([(ax, ay), (bx, by)], fill=color_a, width=2)

    # Draw joints
    for i, (x, y) in enumerate(kps):
        px, py = to_px(x, y)
        r = 5 if i < 8 or i >= 18 else 3  # Smaller for fingers
        color = get_joint_color(i)
        draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline='white')

    # Draw label
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font = ImageFont.load_default()
        draw.text((20, size - 50), label, fill=(255, 255, 255), font=font)

    return img


def create_keypoint_gif(sequence, labels, output_path, fps=15):
    """Create animated GIF of keypoint stick figures."""
    frames = []
    for i, (kps, label) in enumerate(zip(sequence, labels)):
        frame_label = f"{label}  (frame {i+1}/{len(sequence)})"
        img = draw_keypoints(kps, size=512, label=frame_label)
        frames.append(img)

    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"💾 Keypoint GIF saved: {output_path} ({len(frames)} frames)")
    return frames


def create_keypoint_strip(sequence, labels, output_path, num_samples=8):
    """Create a horizontal strip showing key frames."""
    step = max(1, len(sequence) // num_samples)
    indices = list(range(0, len(sequence), step))[:num_samples]

    tile_size = 256
    strip = Image.new('RGB', (tile_size * len(indices), tile_size + 40),
                       (20, 22, 35))
    draw = ImageDraw.Draw(strip)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for col, idx in enumerate(indices):
        tile = draw_keypoints(sequence[idx], size=tile_size, label="")
        strip.paste(tile, (col * tile_size, 0))
        # Label below
        label = labels[idx]
        draw.text(
            (col * tile_size + 10, tile_size + 8),
            f"F{idx}: {label}",
            fill=(200, 200, 200), font=font
        )

    strip.save(output_path)
    print(f"💾 Keypoint strip saved: {output_path}")
    return strip


# ─── TPS Model Performance Test ─────────────────────────────────────────────

def run_tps_performance_test(sequence):
    """Test TPS model forward pass with the keypoint sequence."""
    with open('ml/model/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    from ml.model.model import TPSMotionModel

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\n🖥️  Device: {device}")

    # Load model
    model = TPSMotionModel(config).to(device)
    model.eval()
    params = model.get_num_params()
    print(f"📊 Total parameters: {params['total']:,}")

    # Create a synthetic source image (gradient colored rectangle)
    source_img = torch.zeros(1, 3, 256, 256)
    # Create a simple body-colored region
    for c in range(3):
        source_img[0, c] = torch.linspace(0.2, 0.8, 256).unsqueeze(1).repeat(1, 256)
    source_img = source_img.to(device)

    # Get source keypoints
    with torch.no_grad():
        source_kps = model.kp_detector(source_img)

    num_tps = config['model_params']['common_params']['num_tps']
    expected_kps = num_tps * 5  # 50

    # Prepare driving keypoints from our sequence
    # Pad 25 body keypoints to 50 TPS keypoints
    driving_kp_list = []
    for frame_kps in sequence:
        padded = np.zeros((expected_kps, 2), dtype=np.float32)
        n = min(len(frame_kps), expected_kps)
        padded[:n] = frame_kps[:n]
        # Duplicate to fill remaining slots
        for i in range(n, expected_kps):
            padded[i] = frame_kps[i % n]
        driving_kp_list.append(padded)

    # Run inference and measure timing
    print(f"\n🔄 Running inference on {len(sequence)} frames...")
    generated_frames = []
    times = []

    with torch.no_grad():
        for i, kps in enumerate(driving_kp_list):
            kp_tensor = torch.tensor(kps, dtype=torch.float32).unsqueeze(0).to(device)
            kp_driving = {'fg_kp': kp_tensor}

            start = time.time()
            prediction = model.animate(source_img, kp_driving, source_kps)
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

    # Timing stats
    times = np.array(times)
    print(f"\n📈 Performance Statistics:")
    print(f"   Total frames:     {len(times)}")
    print(f"   Total time:       {times.sum():.2f}s")
    print(f"   Avg per frame:    {times.mean()*1000:.1f}ms")
    print(f"   Min per frame:    {times.min()*1000:.1f}ms")
    print(f"   Max per frame:    {times.max()*1000:.1f}ms")
    print(f"   Effective FPS:    {1/times.mean():.1f}")
    print(f"   Std deviation:    {times.std()*1000:.1f}ms")

    return generated_frames, times


def save_model_output_strip(frames, labels, output_path, num_samples=8):
    """Save a strip of model output frames."""
    step = max(1, len(frames) // num_samples)
    indices = list(range(0, len(frames), step))[:num_samples]

    tile_size = 256
    strip = Image.new('RGB', (tile_size * len(indices), tile_size + 40),
                       (20, 22, 35))
    draw = ImageDraw.Draw(strip)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for col, idx in enumerate(indices):
        tile = Image.fromarray(frames[idx]).resize((tile_size, tile_size))
        strip.paste(tile, (col * tile_size, 0))
        label = labels[idx] if idx < len(labels) else ""
        draw.text(
            (col * tile_size + 10, tile_size + 8),
            f"F{idx}: {label}",
            fill=(200, 200, 200), font=font
        )

    strip.save(output_path)
    print(f"💾 Model output strip saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤟 TPS Motion Model — Demo: 'Hello, how are you?'")
    print("=" * 60)

    # 1. Generate sign language keypoints
    print("\n1️⃣  Generating sign language keypoints...")
    sequence, labels = generate_sign_sequence()
    print(f"   Generated {len(sequence)} frames for 4 signs")
    print(f"   Signs: HELLO → HOW → ARE → YOU")
    print(f"   Keypoints per frame: {sequence.shape[1]}")

    # 2. Visualize keypoints
    print("\n2️⃣  Creating keypoint visualizations...")
    strip_path = os.path.join(OUTPUT_DIR, 'keypoint_strip.png')
    create_keypoint_strip(sequence, labels, strip_path, num_samples=10)

    gif_path = os.path.join(OUTPUT_DIR, 'keypoint_animation.gif')
    create_keypoint_gif(sequence, labels, gif_path, fps=12)

    # Save individual key frames
    key_indices = [0, 12, 30, 48, 65, 80]
    for idx in key_indices:
        if idx < len(sequence):
            img = draw_keypoints(sequence[idx], size=512,
                                  label=f"{labels[idx]} (frame {idx})")
            img.save(os.path.join(OUTPUT_DIR, f'keyframe_{idx:03d}.png'))

    # 3. Run TPS model
    print("\n3️⃣  Running TPS Motion Model...")
    generated_frames, times = run_tps_performance_test(sequence)

    # 4. Save model outputs
    print("\n4️⃣  Saving results...")
    model_strip_path = os.path.join(OUTPUT_DIR, 'model_output_strip.png')
    save_model_output_strip(generated_frames, labels, model_strip_path,
                            num_samples=10)

    # Save model output as GIF
    model_gif_path = os.path.join(OUTPUT_DIR, 'model_output.gif')
    pil_frames = [Image.fromarray(f) for f in generated_frames]
    pil_frames[0].save(
        model_gif_path, save_all=True,
        append_images=pil_frames[1:],
        duration=80, loop=0
    )
    print(f"💾 Model output GIF: {model_gif_path}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Phrase:           'Hello, how are you?'")
    print(f"   Total frames:     {len(sequence)}")
    print(f"   Total inference:  {times.sum():.2f}s")
    print(f"   Avg frame time:   {times.mean()*1000:.1f}ms")
    print(f"   Effective FPS:    {1/times.mean():.1f}")
    print(f"\n   Output directory: {OUTPUT_DIR}")
    print(f"   Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"     📄 {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("⚠️  NOTE: Model output is random (untrained weights).")
    print("   The keypoint stick figures show the actual sign gestures.")
    print("   After training on real videos, the model will produce")
    print("   photorealistic animated frames from these keypoints.")
    print("=" * 60)


if __name__ == '__main__':
    main()
