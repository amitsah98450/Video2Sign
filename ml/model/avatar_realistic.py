"""
Sign Language Avatar — Realistic 3D Avatar + Animated Skeleton
Uses a high-quality AI-generated 3D avatar as the body base,
with smooth animated arm/hand skeleton overlay for sign language.

Usage:
  cd "Video to Sign"
  python3 -m ml.model.avatar_realistic
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, 'demo_output')
AVATAR_BASE = os.path.join(OUTPUT_DIR, 'avatar_base.png')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 600
FPS = 15

# ─── Sign Language Colors (professional dark theme) ──────────────────────────

COLORS = {
    'bg_top':       (12, 12, 30),
    'bg_bottom':    (20, 22, 45),
    'arm_left':     (0, 200, 150),
    'arm_right':    (255, 120, 80),
    'hand_left':    (0, 255, 180),
    'hand_right':   (255, 160, 100),
    'finger_left':  (100, 255, 200),
    'finger_right': (255, 200, 150),
    'joint':        (255, 255, 255),
    'glow_left':    (0, 200, 150, 40),
    'glow_right':   (255, 120, 80, 40),
    'label_bg':     (20, 40, 80, 200),
    'label_text':   (255, 255, 255),
    'subtitle':     (140, 160, 200),
    'progress':     (80, 150, 255),
    'progress_bg':  (40, 50, 80),
}


# ─── Keypoint System ─────────────────────────────────────────────────────────

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


def interp(a, b, n):
    """Smooth cosine interpolation between two poses."""
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        t = 0.5 - 0.5 * math.cos(t * math.pi)
        frames.append(a * (1 - t) + b * t)
    return frames


def make_hello():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.35, -0.55]; p1[4] = [-0.45, -0.70]
    p1[8:13] = [[-0.50, -0.78], [-0.47, -0.80], [-0.44, -0.80],
                [-0.41, -0.78], [-0.38, -0.76]]
    p2 = p1.copy()
    p2[4] = [-0.35, -0.70]
    p2[8:13] = [[-0.40, -0.78], [-0.37, -0.80], [-0.34, -0.80],
                [-0.31, -0.78], [-0.28, -0.76]]
    f = interp(NEUTRAL_POSE, p1, 8)
    f += interp(p1, p2, 6)
    f += interp(p2, p1, 6)
    f += interp(p1, p2, 6)
    f += interp(p2, NEUTRAL_POSE, 6)
    return f


def make_how():
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
    f = interp(NEUTRAL_POSE, p1, 6)
    f += interp(p1, p2, 10)
    f += interp(p2, NEUTRAL_POSE, 6)
    return f


def make_are():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.50]; p1[4] = [-0.05, -0.60]
    p1[8:13] = [[-0.05, -0.65], [-0.07, -0.58], [-0.08, -0.57],
                [-0.09, -0.56], [-0.10, -0.55]]
    p2 = p1.copy()
    p2[4] = [-0.05, -0.55]
    p2[8:13] = [[0.0, -0.58], [-0.07, -0.53], [-0.08, -0.52],
                [-0.09, -0.51], [-0.10, -0.50]]
    f = interp(NEUTRAL_POSE, p1, 6)
    f += interp(p1, p2, 10)
    f += interp(p2, NEUTRAL_POSE, 6)
    return f


def make_you():
    p1 = NEUTRAL_POSE.copy()
    p1[3] = [-0.10, -0.35]; p1[4] = [0.0, -0.30]
    p1[8:13] = [[0.05, -0.32], [-0.02, -0.28], [-0.03, -0.27],
                [-0.04, -0.26], [-0.05, -0.25]]
    f = interp(NEUTRAL_POSE, p1, 8)
    f += [p1] * 10
    f += interp(p1, NEUTRAL_POSE, 8)
    return f


def generate_sequence():
    seq, labels = [], []
    for name, frames in [("HELLO", make_hello()), ("HOW", make_how()),
                         ("ARE", make_are()), ("YOU", make_you())]:
        for f in frames:
            seq.append(f)
            labels.append(name)
        for _ in range(4):
            seq.append(NEUTRAL_POSE.copy())
            labels.append("—")
    return np.array(seq), labels


# ─── Rendering Engine ────────────────────────────────────────────────────────

def kp2px(kps, idx, size):
    """Convert [-1,1] keypoints to pixel coordinates."""
    x, y = kps[idx]
    px = int((x + 1) / 2 * (size * 0.7) + size * 0.15)
    py = int((y + 1) / 2 * (size * 0.65) + size * 0.12)
    return px, py


def create_bg(size):
    """Smooth dark gradient background."""
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / size
        r = int(COLORS['bg_top'][0] * (1-t) + COLORS['bg_bottom'][0] * t)
        g = int(COLORS['bg_top'][1] * (1-t) + COLORS['bg_bottom'][1] * t)
        b = int(COLORS['bg_top'][2] * (1-t) + COLORS['bg_bottom'][2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    # Subtle radial vignette
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    cx, cy = size // 2, size // 2
    for r in range(size, 0, -5):
        alpha = int(30 * (1 - r / size))
        od.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(img, overlay)


def prepare_avatar(avatar_path, size):
    """Load and prep the 3D avatar as compositable layer."""
    if not os.path.exists(avatar_path):
        return None

    img = Image.open(avatar_path).convert('RGBA')

    # Resize to fill center portion of canvas
    avatar_h = int(size * 0.7)
    avatar_w = int(avatar_h * img.width / img.height)
    img = img.resize((avatar_w, avatar_h), Image.LANCZOS)

    # Position centered horizontally, upper-center vertically
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    x = (size - avatar_w) // 2
    y = int(size * 0.06)
    canvas.paste(img, (x, y))

    return canvas


def draw_rounded_limb(draw, p1, p2, width, color, glow_color=None):
    """Draw a smooth, glowing limb segment on RGBA surface."""
    draw.line([p1, p2], fill=color, width=width)
    r = width // 2
    draw.ellipse([p1[0]-r, p1[1]-r, p1[0]+r, p1[1]+r], fill=color)
    draw.ellipse([p2[0]-r, p2[1]-r, p2[0]+r, p2[1]+r], fill=color)


def draw_joint(draw, pos, radius=6, color=(255, 255, 255)):
    """Draw a bright joint marker."""
    # Outer glow
    for i in range(3):
        r = radius + (3 - i) * 3
        alpha = 30 * (i + 1)
        gc = color + (alpha,) if len(color) == 3 else color
        draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r],
                     fill=(color[0], color[1], color[2], alpha))
    # Core
    draw.ellipse([pos[0]-radius, pos[1]-radius,
                  pos[0]+radius, pos[1]+radius],
                 fill=color + (255,) if len(color) == 3 else color,
                 outline=(255, 255, 255, 200))


def draw_skeleton_overlay(kps, size, prev_kps=None):
    """Draw colored skeleton arms, hands, and motion trails on transparent layer."""
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # === Motion trails ===
    if prev_kps is not None:
        trail_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        td = ImageDraw.Draw(trail_layer)
        for idx, color in [(4, COLORS['arm_right']), (7, COLORS['arm_left']),
                           (3, COLORS['arm_right']), (6, COLORS['arm_left'])]:
            curr = kp2px(kps, idx, size)
            prev = kp2px(prev_kps, idx, size)
            dist = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
            if dist > 3:
                # Draw trail
                for t in range(5):
                    alpha = int(60 * (5 - t) / 5)
                    frac = t / 5
                    mx = int(prev[0] * frac + curr[0] * (1 - frac))
                    my = int(prev[1] * frac + curr[1] * (1 - frac))
                    r = int(min(dist * 0.4, 20))
                    td.ellipse([mx-r, my-r, mx+r, my+r],
                              fill=(color[0], color[1], color[2], alpha))
        # Blur the trail
        trail_layer = trail_layer.filter(ImageFilter.GaussianBlur(radius=8))
        overlay = Image.alpha_composite(overlay, trail_layer)
        draw = ImageDraw.Draw(overlay)

    # === Glow layer for arm regions ===
    glow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)

    # Right arm glow (shoulder → elbow → wrist)
    for idx in [3, 4]:
        px, py = kp2px(kps, idx, size)
        r = 30
        for ring in range(r, 0, -3):
            alpha = int(20 * ring / r)
            gd.ellipse([px-ring, py-ring, px+ring, py+ring],
                      fill=(COLORS['arm_right'][0], COLORS['arm_right'][1],
                            COLORS['arm_right'][2], alpha))

    # Left arm glow
    for idx in [6, 7]:
        px, py = kp2px(kps, idx, size)
        r = 30
        for ring in range(r, 0, -3):
            alpha = int(20 * ring / r)
            gd.ellipse([px-ring, py-ring, px+ring, py+ring],
                      fill=(COLORS['arm_left'][0], COLORS['arm_left'][1],
                            COLORS['arm_left'][2], alpha))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    overlay = Image.alpha_composite(overlay, glow)
    draw = ImageDraw.Draw(overlay)

    arm_w = int(10 * SIZE / 600)
    forearm_w = int(8 * SIZE / 600)
    finger_w = int(3 * SIZE / 600)

    # === RIGHT ARM (orange tones) ===
    sh_r = kp2px(kps, 2, size)
    el_r = kp2px(kps, 3, size)
    wr_r = kp2px(kps, 4, size)

    draw_rounded_limb(draw, sh_r, el_r, arm_w, COLORS['arm_right'] + (220,))
    draw_rounded_limb(draw, el_r, wr_r, forearm_w, COLORS['arm_right'] + (220,))
    draw_joint(draw, el_r, 7, COLORS['arm_right'])
    draw_joint(draw, wr_r, 7, COLORS['hand_right'])

    # Right hand fingers
    for i in range(5):
        fi = kp2px(kps, 8 + i, size)
        draw.line([wr_r, fi], fill=COLORS['finger_right'] + (200,), width=finger_w)
        draw_joint(draw, fi, 3, COLORS['finger_right'])

    # Right hand palm
    palm_r = int(10 * SIZE / 600)
    draw.ellipse([wr_r[0]-palm_r, wr_r[1]-palm_r,
                  wr_r[0]+palm_r, wr_r[1]+palm_r],
                 fill=COLORS['hand_right'] + (180,),
                 outline=COLORS['finger_right'] + (200,))

    # === LEFT ARM (green/teal tones) ===
    sh_l = kp2px(kps, 5, size)
    el_l = kp2px(kps, 6, size)
    wr_l = kp2px(kps, 7, size)

    draw_rounded_limb(draw, sh_l, el_l, arm_w, COLORS['arm_left'] + (220,))
    draw_rounded_limb(draw, el_l, wr_l, forearm_w, COLORS['arm_left'] + (220,))
    draw_joint(draw, el_l, 7, COLORS['arm_left'])
    draw_joint(draw, wr_l, 7, COLORS['hand_left'])

    # Left hand fingers
    for i in range(5):
        fi = kp2px(kps, 13 + i, size)
        draw.line([wr_l, fi], fill=COLORS['finger_left'] + (200,), width=finger_w)
        draw_joint(draw, fi, 3, COLORS['finger_left'])

    # Left hand palm
    draw.ellipse([wr_l[0]-palm_r, wr_l[1]-palm_r,
                  wr_l[0]+palm_r, wr_l[1]+palm_r],
                 fill=COLORS['hand_left'] + (180,),
                 outline=COLORS['finger_left'] + (200,))

    return overlay


def draw_ui(draw, label, frame_idx, total, size):
    """Draw title, sign label, progress bar."""
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(16 * SIZE / 600))
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(26 * SIZE / 600))
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(13 * SIZE / 600))
    except:
        font_title = font_label = font_small = ImageFont.load_default()

    # Title
    draw.text((int(size * 0.03), int(size * 0.015)),
              '🤟 Sign Language: "Hello, how are you?"',
              fill=COLORS['subtitle'], font=font_title)

    # Sign label badge
    if label != "—":
        bbox = draw.textbbox((0, 0), label, font=font_label)
        tw = bbox[2] - bbox[0]
        tx = (size - tw) // 2
        ty = int(size * 0.88)
        pad = 16
        draw.rounded_rectangle(
            [tx - pad, ty - 8, tx + tw + pad, ty + 36],
            radius=10,
            fill=COLORS['label_bg'],
            outline=COLORS['progress'] + (100,),
        )
        draw.text((tx, ty), label, fill=COLORS['label_text'], font=font_label)

    # Frame counter
    draw.text((int(size * 0.03), int(size * 0.95)),
              f"Frame {frame_idx + 1}/{total}",
              fill=(80, 100, 140), font=font_small)

    # Progress bar
    bar_y = int(size * 0.975)
    bar_x = int(size * 0.03)
    bar_w = int(size * 0.94)
    progress = (frame_idx + 1) / total
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 4],
                           radius=2, fill=COLORS['progress_bg'])
    if progress > 0.01:
        draw.rounded_rectangle(
            [bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + 4],
            radius=2, fill=COLORS['progress'])

    # Legend
    legend_x = int(size * 0.78)
    legend_y = int(size * 0.03)
    dot_r = 4
    # Right arm
    draw.ellipse([legend_x, legend_y, legend_x + dot_r*2, legend_y + dot_r*2],
                 fill=COLORS['arm_right'])
    draw.text((legend_x + 14, legend_y - 2), "Right", fill=(150, 160, 180), font=font_small)
    # Left arm
    draw.ellipse([legend_x, legend_y + 18, legend_x + dot_r*2, legend_y + 18 + dot_r*2],
                 fill=COLORS['arm_left'])
    draw.text((legend_x + 14, legend_y + 16), "Left", fill=(150, 160, 180), font=font_small)


# ─── Frame Compositor ─────────────────────────────────────────────────────────

_cache = {}

def render_frame(kps, label, frame_idx, total, avatar_layer, size=SIZE, prev_kps=None):
    """Composite: background → avatar → skeleton → UI."""
    global _cache
    if 'bg' not in _cache:
        _cache['bg'] = create_bg(size)

    # 1. Background
    frame = _cache['bg'].copy()

    # 2. Avatar (static 3D body)
    if avatar_layer is not None:
        frame = Image.alpha_composite(frame, avatar_layer)

    # 3. Skeleton overlay (animated arms/hands with glow)
    skeleton = draw_skeleton_overlay(kps, size, prev_kps)
    frame = Image.alpha_composite(frame, skeleton)

    # 4. UI elements
    ui = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ui_draw = ImageDraw.Draw(ui)
    draw_ui(ui_draw, label, frame_idx, total, size)
    frame = Image.alpha_composite(frame, ui)

    return frame.convert('RGB')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤟 Sign Language Avatar — Realistic 3D + Skeleton")
    print("   Phrase: 'Hello, how are you?'")
    print("=" * 60)

    # Generate keypoints
    print("\n1️⃣  Generating keypoints...")
    seq, labels = generate_sequence()
    print(f"   {len(seq)} frames → HELLO → HOW → ARE → YOU")

    # Load avatar
    print(f"\n2️⃣  Loading 3D avatar...")
    avatar_layer = prepare_avatar(AVATAR_BASE, SIZE)
    if avatar_layer:
        print(f"   ✅ Avatar loaded: {os.path.basename(AVATAR_BASE)}")
    else:
        print(f"   ⚠️  No avatar found, rendering skeleton only")

    # Render frames
    print(f"\n3️⃣  Rendering {len(seq)} frames ({SIZE}×{SIZE})...")
    frames = []
    prev_kps = None
    for i, (kps, label) in enumerate(zip(seq, labels)):
        img = render_frame(kps, label, i, len(seq), avatar_layer, prev_kps=prev_kps)
        frames.append(img)
        prev_kps = kps
        if (i + 1) % 30 == 0:
            print(f"   Frame {i+1}/{len(seq)}")
    print(f"   ✅ All frames rendered")

    # Save GIF
    print(f"\n4️⃣  Saving outputs...")
    gif_path = os.path.join(OUTPUT_DIR, 'sign_language_demo.gif')
    duration = int(1000 / FPS)
    frames[0].save(
        gif_path, save_all=True,
        append_images=frames[1:],
        duration=duration, loop=0,
    )
    gif_size = os.path.getsize(gif_path) / 1024
    print(f"   💾 {os.path.basename(gif_path)} ({gif_size:.0f} KB, {FPS} FPS)")

    # Strip
    strip_path = os.path.join(OUTPUT_DIR, 'sign_language_strip.png')
    step = max(1, len(frames) // 8)
    indices = list(range(0, len(frames), step))[:8]
    tile_sz = 256
    strip = Image.new('RGB', (tile_sz * len(indices), tile_sz + 35), (15, 15, 30))
    sd = ImageDraw.Draw(strip)
    try:
        sf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        sf = ImageFont.load_default()
    for c, idx in enumerate(indices):
        strip.paste(frames[idx].resize((tile_sz, tile_sz), Image.LANCZOS),
                    (c * tile_sz, 0))
        lbl = labels[idx] if idx < len(labels) else ""
        sd.text((c * tile_sz + 6, tile_sz + 8), f"F{idx}: {lbl}",
                fill=(150, 170, 210), font=sf)
    strip.save(strip_path)
    print(f"   💾 {os.path.basename(strip_path)}")

    # Key frames
    for idx in [0, 12, 30, 50, 70, len(frames)-1]:
        if idx < len(frames):
            p = os.path.join(OUTPUT_DIR, f'sign_frame_{idx:03d}.png')
            frames[idx].save(p)

    print(f"\n" + "=" * 60)
    print(f"✅ COMPLETE")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('sign_'):
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"   📄 {f} ({sz:.1f} KB)")
    print(f"\n   🎬 Open {os.path.basename(gif_path)} to see the animation!")
    print("=" * 60)


if __name__ == '__main__':
    main()
