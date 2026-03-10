"""
Sign Language Avatar Renderer
Creates high-quality, stylized human figure animations from keypoints.
Renders beautiful 2D avatar with proper body proportions, gradient
coloring, smooth shapes, and professional styling.

Usage:
  cd "Video to Sign"
  python3 -m ml.model.avatar_demo
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────

SIZE = 600           # Canvas size
BG_COLOR_TOP = (15, 15, 35)
BG_COLOR_BOT = (30, 30, 60)
BODY_COLOR = (70, 130, 230)        # Primary body blue
BODY_SHADOW = (40, 80, 160)
SKIN_COLOR = (230, 195, 170)
SKIN_SHADOW = (200, 165, 140)
HEAD_OUTLINE = (50, 100, 180)
EYE_COLOR = (40, 40, 50)
MOUTH_COLOR = (200, 100, 100)
SHIRT_COLOR = (55, 115, 210)
SHIRT_SHADOW = (35, 85, 170)
HAND_COLOR = (225, 190, 165)
HAND_OUTLINE = (190, 155, 130)
GLOW_COLOR = (80, 140, 255, 30)
JOINT_GLOW = (100, 180, 255, 60)
FLOOR_COLOR = (25, 25, 50, 100)

# Body proportions (relative to canvas)
SCALE = SIZE / 600


# ─── Keypoint Definitions ────────────────────────────────────────────────────
# 25 keypoints: same layout as demo_tps.py

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


def interpolate_poses(a, b, n):
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        t = 0.5 - 0.5 * math.cos(t * math.pi)
        frames.append(a * (1 - t) + b * t)
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
    f = []
    f += interpolate_poses(NEUTRAL_POSE, p1, 8)
    f += interpolate_poses(p1, p2, 6)
    f += interpolate_poses(p2, p1, 6)
    f += interpolate_poses(p1, p2, 6)
    f += interpolate_poses(p2, NEUTRAL_POSE, 6)
    return f


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
    f = []
    f += interpolate_poses(NEUTRAL_POSE, p1, 6)
    f += interpolate_poses(p1, p2, 10)
    f += interpolate_poses(p2, NEUTRAL_POSE, 6)
    return f


def make_are_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.50]; p1[4] = [-0.05, -0.60]
    p1[8:13] = [[-0.05, -0.65], [-0.07, -0.58], [-0.08, -0.57],
                [-0.09, -0.56], [-0.10, -0.55]]
    p2 = p1.copy()
    p2[4] = [-0.05, -0.55]
    p2[8:13] = [[0.0, -0.58], [-0.07, -0.53], [-0.08, -0.52],
                [-0.09, -0.51], [-0.10, -0.50]]
    f = []
    f += interpolate_poses(NEUTRAL_POSE, p1, 6)
    f += interpolate_poses(p1, p2, 10)
    f += interpolate_poses(p2, NEUTRAL_POSE, 6)
    return f


def make_you_poses():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.35]; p1[4] = [0.0, -0.30]
    p1[8:13] = [[0.05, -0.32], [-0.02, -0.28], [-0.03, -0.27],
                [-0.04, -0.26], [-0.05, -0.25]]
    f = []
    f += interpolate_poses(NEUTRAL_POSE, p1, 8)
    f += [p1] * 10
    f += interpolate_poses(p1, NEUTRAL_POSE, 8)
    return f


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
        for _ in range(4):
            sequence.append(NEUTRAL_POSE.copy())
            labels.append("—")
    return np.array(sequence), labels


# ─── Gradient / Background Rendering ─────────────────────────────────────────

def create_gradient_background(size):
    """Create smooth dark gradient background."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / size
        r = int(BG_COLOR_TOP[0] * (1 - t) + BG_COLOR_BOT[0] * t)
        g = int(BG_COLOR_TOP[1] * (1 - t) + BG_COLOR_BOT[1] * t)
        b = int(BG_COLOR_TOP[2] * (1 - t) + BG_COLOR_BOT[2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))
    return img


def add_floor_reflection(img, size):
    """Add subtle floor line and glow."""
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    floor_y = int(size * 0.82)

    # Floor line with glow
    for offset in range(8, 0, -1):
        alpha = int(15 * (8 - offset) / 8)
        draw.line([(int(size * 0.15), floor_y + offset),
                   (int(size * 0.85), floor_y + offset)],
                  fill=(100, 150, 255, alpha), width=1)

    # Main floor line
    draw.line([(int(size * 0.15), floor_y),
               (int(size * 0.85), floor_y)],
              fill=(80, 120, 200, 60), width=2)

    return Image.alpha_composite(img, overlay)


# ─── Avatar Rendering ────────────────────────────────────────────────────────

def kp_to_px(kps, idx, size):
    """Convert keypoint [-1, 1] coords to pixel coords."""
    x, y = kps[idx]
    px = int((x + 1) / 2 * (size * 0.7) + size * 0.15)
    py = int((y + 1) / 2 * (size * 0.65) + size * 0.12)
    return px, py


def draw_thick_line(draw, p1, p2, width, color, shadow_color=None):
    """Draw a rounded thick line (limb segment)."""
    if shadow_color:
        # Shadow offset
        sp1 = (p1[0] + 2, p1[1] + 2)
        sp2 = (p2[0] + 2, p2[1] + 2)
        draw.line([sp1, sp2], fill=shadow_color, width=width + 2)
        # Two circles at endpoints for rounded caps
        r = width // 2
        draw.ellipse([sp1[0]-r, sp1[1]-r, sp1[0]+r, sp1[1]+r],
                     fill=shadow_color)
        draw.ellipse([sp2[0]-r, sp2[1]-r, sp2[0]+r, sp2[1]+r],
                     fill=shadow_color)

    draw.line([p1, p2], fill=color, width=width)
    r = width // 2
    draw.ellipse([p1[0]-r, p1[1]-r, p1[0]+r, p1[1]+r], fill=color)
    draw.ellipse([p2[0]-r, p2[1]-r, p2[0]+r, p2[1]+r], fill=color)


def draw_torso(draw, kps, size):
    """Draw filled torso shape."""
    neck = kp_to_px(kps, 1, size)
    r_sh = kp_to_px(kps, 2, size)
    l_sh = kp_to_px(kps, 5, size)
    chest = kp_to_px(kps, 22, size)
    r_hip = kp_to_px(kps, 23, size)
    l_hip = kp_to_px(kps, 24, size)

    # Shadow
    shadow_pts = [(p[0]+3, p[1]+3) for p in [r_sh, l_sh, l_hip, r_hip]]
    draw.polygon(shadow_pts, fill=SHIRT_SHADOW)

    # Main torso
    torso_pts = [r_sh, l_sh, l_hip, r_hip]
    draw.polygon(torso_pts, fill=SHIRT_COLOR)

    # Shoulder curve (neckline)
    mid_sh_x = (r_sh[0] + l_sh[0]) // 2
    mid_sh_y = (r_sh[1] + l_sh[1]) // 2
    neckline_w = int(30 * SCALE)
    neckline_h = int(15 * SCALE)
    draw.ellipse([mid_sh_x - neckline_w, mid_sh_y - neckline_h - 5,
                  mid_sh_x + neckline_w, mid_sh_y + neckline_h],
                 fill=SKIN_COLOR)

    # Collar detail
    collar_w = int(22 * SCALE)
    collar_h = int(10 * SCALE)
    draw.arc([mid_sh_x - collar_w, neck[1] - 2,
              mid_sh_x + collar_w, neck[1] + collar_h * 2],
             0, 180, fill=SHIRT_SHADOW, width=2)


def draw_head(draw, kps, size):
    """Draw stylized head with face features."""
    nose = kp_to_px(kps, 0, size)
    r_eye = kp_to_px(kps, 18, size)
    l_eye = kp_to_px(kps, 19, size)
    neck = kp_to_px(kps, 1, size)

    # Head center and radius
    cx = nose[0]
    cy = int(nose[1] - 8 * SCALE)
    head_rx = int(38 * SCALE)
    head_ry = int(44 * SCALE)

    # Neck
    neck_w = int(14 * SCALE)
    draw.rectangle([cx - neck_w, cy + head_ry - 10,
                    cx + neck_w, neck[1] + 5],
                   fill=SKIN_COLOR)
    draw.rectangle([cx - neck_w + 3, cy + head_ry - 10,
                    cx + neck_w + 3, neck[1] + 5],
                   fill=SKIN_SHADOW)

    # Head shadow
    draw.ellipse([cx - head_rx + 3, cy - head_ry + 3,
                  cx + head_rx + 3, cy + head_ry + 3],
                 fill=SKIN_SHADOW)

    # Head
    draw.ellipse([cx - head_rx, cy - head_ry,
                  cx + head_rx, cy + head_ry],
                 fill=SKIN_COLOR, outline=HEAD_OUTLINE, width=2)

    # Hair (dark cap on top of head)
    hair_y_offset = int(10 * SCALE)
    draw.chord([cx - head_rx - 2, cy - head_ry - 5,
                cx + head_rx + 2, cy + hair_y_offset],
               180, 360, fill=(40, 35, 30))
    # Hair sides
    draw.arc([cx - head_rx - 3, cy - head_ry - 6,
              cx + head_rx + 3, cy + hair_y_offset + 5],
             160, 380, fill=(30, 25, 20), width=3)

    # Eyes
    eye_size = int(5 * SCALE)
    # White of eye
    for ex, ey in [r_eye, l_eye]:
        draw.ellipse([ex - eye_size - 2, ey - eye_size,
                      ex + eye_size + 2, ey + eye_size],
                     fill=(255, 255, 255), outline=(180, 180, 190))
        # Iris
        draw.ellipse([ex - eye_size + 1, ey - eye_size + 1,
                      ex + eye_size - 1, ey + eye_size - 1],
                     fill=(80, 60, 40))
        # Pupil
        draw.ellipse([ex - 2, ey - 2, ex + 2, ey + 2],
                     fill=EYE_COLOR)
        # Light reflection
        draw.ellipse([ex - 4, ey - 3, ex - 2, ey - 1],
                     fill=(255, 255, 255))

    # Eyebrows
    brow_w = int(10 * SCALE)
    for ex, ey in [r_eye, l_eye]:
        draw.arc([ex - brow_w, ey - int(14 * SCALE),
                  ex + brow_w, ey - int(3 * SCALE)],
                 190, 350, fill=(50, 40, 35), width=2)

    # Nose (subtle)
    nose_tip = nose
    draw.arc([nose_tip[0] - int(4 * SCALE), nose_tip[1] - int(3 * SCALE),
              nose_tip[0] + int(4 * SCALE), nose_tip[1] + int(5 * SCALE)],
             30, 150, fill=(200, 170, 150), width=2)

    # Mouth (subtle smile)
    mouth_y = int(nose_tip[1] + 14 * SCALE)
    mouth_w = int(12 * SCALE)
    draw.arc([nose_tip[0] - mouth_w, mouth_y - int(4 * SCALE),
              nose_tip[0] + mouth_w, mouth_y + int(6 * SCALE)],
             10, 170, fill=MOUTH_COLOR, width=2)


def draw_arm(draw, kps, size, shoulder_idx, elbow_idx, wrist_idx, is_right):
    """Draw arm with proper segments."""
    sh = kp_to_px(kps, shoulder_idx, size)
    el = kp_to_px(kps, elbow_idx, size)
    wr = kp_to_px(kps, wrist_idx, size)

    arm_w = int(14 * SCALE)
    forearm_w = int(12 * SCALE)

    # Upper arm shadow + main
    draw_thick_line(draw, sh, el, arm_w, SHIRT_COLOR, SHIRT_SHADOW)
    # Forearm (skin)
    draw_thick_line(draw, el, wr, forearm_w, SKIN_COLOR, SKIN_SHADOW)

    # Elbow joint
    elbow_r = int(8 * SCALE)
    draw.ellipse([el[0]-elbow_r, el[1]-elbow_r,
                  el[0]+elbow_r, el[1]+elbow_r],
                 fill=SHIRT_COLOR, outline=SHIRT_SHADOW, width=1)


def draw_hand(draw, kps, size, wrist_idx, finger_start, is_right):
    """Draw hand with finger details."""
    wr = kp_to_px(kps, wrist_idx, size)

    # Palm
    palm_r = int(12 * SCALE)
    draw.ellipse([wr[0]-palm_r+2, wr[1]-palm_r+2,
                  wr[0]+palm_r+2, wr[1]+palm_r+2],
                 fill=HAND_OUTLINE)  # shadow
    draw.ellipse([wr[0]-palm_r, wr[1]-palm_r,
                  wr[0]+palm_r, wr[1]+palm_r],
                 fill=HAND_COLOR, outline=HAND_OUTLINE, width=1)

    # Fingers
    finger_w = int(4 * SCALE)
    for i in range(5):
        fi = kp_to_px(kps, finger_start + i, size)
        draw.line([wr, fi], fill=HAND_COLOR, width=finger_w)
        # Fingertip
        ft_r = int(3 * SCALE)
        draw.ellipse([fi[0]-ft_r, fi[1]-ft_r,
                      fi[0]+ft_r, fi[1]+ft_r],
                     fill=HAND_COLOR, outline=HAND_OUTLINE)


def draw_joint_glow(overlay_draw, px, py, radius=15):
    """Add subtle glow behind joints for motion emphasis."""
    for r in range(radius, 0, -2):
        alpha = int(25 * (radius - r) / radius)
        overlay_draw.ellipse([px-r, py-r, px+r, py+r],
                             fill=(100, 180, 255, alpha))


def draw_label(draw, label, frame_idx, total_frames, size):
    """Draw sign label and progress bar."""
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(28 * SCALE))
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(14 * SCALE))
    except:
        font_label = ImageFont.load_default()
        font_small = font_label

    if label != "—":
        # Label background
        text_bbox = draw.textbbox((0, 0), label, font=font_label)
        tw = text_bbox[2] - text_bbox[0]
        tx = (size - tw) // 2
        ty = int(size * 0.87)

        # Rounded label background
        pad = 12
        bg_box = [tx - pad, ty - 6, tx + tw + pad, ty + 34]
        draw.rounded_rectangle(bg_box, radius=8, fill=(30, 60, 140, 200))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font_label)

    # Frame counter
    counter = f"Frame {frame_idx + 1}/{total_frames}"
    draw.text((int(size * 0.03), int(size * 0.95)),
              counter, fill=(100, 120, 160), font=font_small)

    # Progress bar
    bar_y = int(size * 0.97)
    bar_w = int(size * 0.94)
    bar_x = int(size * 0.03)
    progress = (frame_idx + 1) / total_frames
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 3],
                   fill=(40, 50, 80))
    draw.rectangle([bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + 3],
                   fill=(80, 150, 255))


def draw_title_bar(draw, size):
    """Draw title at top."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(16 * SCALE))
    except:
        font = ImageFont.load_default()
    title = "🤟 Sign Language: \"Hello, how are you?\""
    draw.text((int(size * 0.03), int(size * 0.02)),
              title, fill=(120, 150, 200), font=font)


# ─── Main Frame Renderer ─────────────────────────────────────────────────────

# Pre-render background once
_bg_cache = {}

def render_frame(kps, label, frame_idx, total_frames, size=SIZE,
                 prev_kps=None):
    """Render a single high-quality avatar frame."""
    global _bg_cache
    if size not in _bg_cache:
        _bg_cache[size] = create_gradient_background(size)

    # Start with gradient background
    img = _bg_cache[size].copy()

    # Add floor
    img = add_floor_reflection(img, size)

    # Create overlay for glow effects
    glow_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    # Add motion glow on active joints
    if prev_kps is not None:
        for idx in [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            curr = kp_to_px(kps, idx, size)
            prev = kp_to_px(prev_kps, idx, size)
            dist = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
            if dist > 3:
                draw_joint_glow(glow_draw, curr[0], curr[1],
                               radius=int(min(dist * 1.5, 30)))

    img = Image.alpha_composite(img, glow_layer)

    # Main drawing (opaque layer)
    draw = ImageDraw.Draw(img)

    # Draw body parts (back to front)
    # 1. Torso
    draw_torso(draw, kps, size)

    # 2. Arms (back arm first based on Z-ordering)
    # Right arm (left side of screen)
    draw_arm(draw, kps, size, 2, 3, 4, is_right=True)
    draw_hand(draw, kps, size, 4, 8, is_right=True)

    # Left arm
    draw_arm(draw, kps, size, 5, 6, 7, is_right=False)
    draw_hand(draw, kps, size, 7, 13, is_right=False)

    # 3. Head (on top)
    draw_head(draw, kps, size)

    # 4. UI overlays
    draw_title_bar(draw, size)
    draw_label(draw, label, frame_idx, total_frames, size)

    return img.convert('RGB')


# ─── Output Generation ────────────────────────────────────────────────────────

def create_animation_gif(sequence, labels, output_path, fps=15):
    """Create smooth animated GIF."""
    total = len(sequence)
    frames = []
    prev_kps = None

    print(f"🎨 Rendering {total} frames...")
    for i, (kps, label) in enumerate(zip(sequence, labels)):
        img = render_frame(kps, label, i, total, prev_kps=prev_kps)
        frames.append(img)
        prev_kps = kps
        if (i + 1) % 25 == 0:
            print(f"   Frame {i+1}/{total}")

    duration = int(1000 / fps)
    frames[0].save(
        output_path, save_all=True,
        append_images=frames[1:],
        duration=duration, loop=0,
        optimize=False,
    )
    print(f"💾 Saved: {output_path} ({len(frames)} frames, {fps} FPS)")
    return frames


def create_comparison_strip(frames, labels, sequence, output_path, num=10):
    """Create horizontal strip of key frames."""
    step = max(1, len(frames) // num)
    indices = list(range(0, len(frames), step))[:num]

    tile = 256
    strip = Image.new('RGB', (tile * len(indices), tile + 40), (20, 22, 35))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    for col, idx in enumerate(indices):
        tile_img = frames[idx].resize((tile, tile), Image.LANCZOS)
        strip.paste(tile_img, (col * tile, 0))
        label = labels[idx] if idx < len(labels) else ""
        draw.text((col * tile + 8, tile + 8),
                  f"F{idx}: {label}", fill=(180, 200, 230), font=font)

    strip.save(output_path)
    print(f"💾 Strip: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤟 Sign Language Avatar — 'Hello, how are you?'")
    print("=" * 60)

    # Generate keypoints
    print("\n1️⃣  Generating sign language keypoints...")
    sequence, labels = generate_sign_sequence()
    print(f"   {len(sequence)} frames, signs: HELLO → HOW → ARE → YOU")

    # Render animation
    print(f"\n2️⃣  Rendering avatar animation ({SIZE}×{SIZE})...")
    gif_path = os.path.join(OUTPUT_DIR, 'avatar_animation.gif')
    frames = create_animation_gif(sequence, labels, gif_path, fps=15)

    # Create strip
    print(f"\n3️⃣  Creating comparison strip...")
    strip_path = os.path.join(OUTPUT_DIR, 'avatar_strip.png')
    create_comparison_strip(frames, labels, sequence, strip_path)

    # Save key frames
    print(f"\n4️⃣  Saving key frames...")
    key_idx = [0, 15, 35, 55, 70, len(frames)-1]
    for idx in key_idx:
        if idx < len(frames):
            p = os.path.join(OUTPUT_DIR, f'avatar_frame_{idx:03d}.png')
            frames[idx].save(p)
            print(f"   📄 {os.path.basename(p)}")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"✅ DONE!")
    print(f"   Output: {OUTPUT_DIR}")
    total_size = 0
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('avatar_'):
            fp = os.path.join(OUTPUT_DIR, f)
            sz = os.path.getsize(fp) / 1024
            total_size += sz
            print(f"   📄 {f} ({sz:.1f} KB)")
    print(f"   Total: {total_size:.0f} KB")
    print("=" * 60)


if __name__ == '__main__':
    main()
