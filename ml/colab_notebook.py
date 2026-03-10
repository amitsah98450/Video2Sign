"""
=============================================================
VIDEO TO SIGN LANGUAGE — Complete Colab Training Notebook
=============================================================
Copy each section (# %% CELL N) into separate Colab cells.
Run them one at a time, top to bottom.

🔑 KEY FEATURE: All data is stored on Google Drive, so nothing
is lost when the Colab runtime disconnects or refreshes.

Data Format:
- OpenPose JSON format
- 137 keypoints per frame: 25 body + 70 face + 21 left hand + 21 right hand
- Each keypoint has (x, y, confidence) — we use (x, y) only → 274 values/frame
- Folder name = SENTENCE_NAME from CSV
- One JSON per frame, ~189 frames per sentence
=============================================================
"""

# %% CELL 1: Install Dependencies & Mount Drive
# ── Run this FIRST every time you open Colab ──

# Install required packages (only needed once per session)
# !pip install torch torchvision torchaudio --quiet
# !pip install pyyaml matplotlib scikit-learn nltk pandas --quiet

from google.colab import drive
drive.mount('/content/drive')

import os
print("✅ Drive mounted!")


# %% CELL 2: Path Configuration (ALL ON GOOGLE DRIVE)
# ── These paths persist across Colab reconnects ──

import os, json, re, pickle, shutil, time
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
#  ALL PERSISTENT PATHS → GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════
DRIVE_PROJECT = '/content/drive/MyDrive/VideoToSign'
DRIVE_DATA    = os.path.join(DRIVE_PROJECT, 'data')         # Raw archives + CSVs
DRIVE_EXTRACT = os.path.join(DRIVE_PROJECT, 'extracted')    # Extracted keypoints (PERSISTENT)
DRIVE_VOCAB   = os.path.join(DRIVE_PROJECT, 'vocab')        # Vocabulary files
DRIVE_CKPT    = os.path.join(DRIVE_PROJECT, 'checkpoints')  # Model checkpoints

# Local fast storage (ephemeral — lost on disconnect, but FAST I/O)
LOCAL_DATA = '/content/fast_data'

# Create all directories
for d in [DRIVE_DATA, DRIVE_EXTRACT, DRIVE_VOCAB, DRIVE_CKPT, LOCAL_DATA]:
    os.makedirs(d, exist_ok=True)

print("📂 Project paths (all on Google Drive):")
print(f"  Archives & CSVs:   {DRIVE_DATA}")
print(f"  Extracted data:    {DRIVE_EXTRACT}")
print(f"  Vocabulary:        {DRIVE_VOCAB}")
print(f"  Checkpoints:       {DRIVE_CKPT}")
print(f"  Local fast copy:   {LOCAL_DATA}")

# List what's in the data directory
print("\n📁 Files in data directory:")
if os.path.exists(DRIVE_DATA):
    for f in sorted(os.listdir(DRIVE_DATA)):
        path = os.path.join(DRIVE_DATA, f)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"  {f} — {size_mb:.1f} MB")
        else:
            print(f"  📁 {f}/")
else:
    print("  ⚠️  No data directory found! Upload your data archives here.")


# %% CELL 3: Upload CSV Files (if not already on Drive)
csv_dir = os.path.join(DRIVE_DATA, 'csv')
os.makedirs(csv_dir, exist_ok=True)

existing = [f for f in os.listdir(csv_dir) if f.endswith('.csv')] if os.path.exists(csv_dir) else []
if len(existing) < 3:
    from google.colab import files
    print("📤 Upload the 3 CSV files: how2sign_realigned_train/val/test.csv")
    uploaded = files.upload()
    for fname, content in uploaded.items():
        with open(os.path.join(csv_dir, fname), 'wb') as f:
            f.write(content)
        print(f"  ✅ Saved to Drive: {fname}")
else:
    print(f"✅ CSVs already on Drive: {existing}")


# %% CELL 4: Extract Keypoints → Google Drive (ONE-TIME, persists forever)
# ── This only runs ONCE. After extraction, data stays on Drive. ──

import tarfile, zipfile

def extract_keypoints_to_drive(archive_path, extract_to, label):
    """
    Extract keypoints to Google Drive. Skips if already extracted.
    Detects actual format by reading magic bytes (not file extension),
    since some .zip files are actually gzip/tar.gz archives.
    """
    # Check if already extracted
    json_check = os.path.join(extract_to, 'openpose_output', 'json')
    if os.path.exists(json_check) and len(os.listdir(json_check)) > 0:
        count = len(os.listdir(json_check))
        print(f"  ✅ {label} already extracted on Drive ({count} folders) — SKIPPING")
        return

    if not os.path.exists(archive_path):
        print(f"  ⚠️  Archive not found: {archive_path}")
        return

    print(f"📦 Extracting {label} → Drive (one-time operation)...")
    start = time.time()

    # Detect actual format by reading magic bytes
    with open(archive_path, 'rb') as f:
        magic = f.read(4)

    is_gzip = magic[:2] == b'\x1f\x8b'
    is_zip = magic[:2] == b'PK'

    if is_gzip:
        print(f"  Detected: gzip/tar.gz archive")
        with tarfile.open(archive_path, 'r:gz') as tar:
            json_members = [m for m in tar.getmembers() if '/json/' in m.name]
            print(f"  Extracting {len(json_members)} entries...")
            tar.extractall(extract_to, members=json_members)
    elif is_zip:
        print(f"  Detected: zip archive")
        with zipfile.ZipFile(archive_path, 'r') as zf:
            json_members = [m for m in zf.namelist() if '/json/' in m or m.startswith('json/')]
            print(f"  Extracting {len(json_members)} entries...")
            for member in json_members:
                zf.extract(member, extract_to)
    else:
        print(f"  ⚠️  Unknown archive format (magic bytes: {magic[:4]}): {archive_path}")
        return

    elapsed = time.time() - start
    count = len(os.listdir(json_check)) if os.path.exists(json_check) else 0
    print(f"  ✅ Done! {count} sentence folders extracted in {elapsed:.0f}s")


# ── Extract all splits (only runs if not already on Drive) ──
print("=" * 60)
print("📦 EXTRACTING KEYPOINTS → GOOGLE DRIVE")
print("   (Skips if already extracted)")
print("=" * 60)

# Val (smallest, ~1.16 GB)
extract_keypoints_to_drive(
    os.path.join(DRIVE_DATA, 'keypoints_val.zip'),
    os.path.join(DRIVE_EXTRACT, 'val'),
    "Val keypoints"
)

# Test (~1.58 GB)
extract_keypoints_to_drive(
    os.path.join(DRIVE_DATA, 'keypoints_test.zip'),
    os.path.join(DRIVE_EXTRACT, 'test'),
    "Test keypoints"
)

# Train (~21 GB) — takes 15-30 min FIRST TIME ONLY
extract_keypoints_to_drive(
    os.path.join(DRIVE_DATA, 'train_2D_keypoints.tar.gz'),
    os.path.join(DRIVE_EXTRACT, 'train'),
    "Train keypoints"
)


# %% CELL 5: Copy Data to Local Storage (FAST I/O for Training)
# ── Run this once per session. If runtime resets, run again. ──
# ── This copies from persistent Drive → fast local SSD. ──

def copy_to_local(drive_path, local_path, label):
    """Copy extracted data from Drive to local for fast I/O."""
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"  ✅ {label} already copied to local storage")
        return

    if not os.path.exists(drive_path):
        print(f"  ⚠️  {label} not found on Drive: {drive_path}")
        return

    print(f"  📋 Copying {label} to local (Drive → local)...")
    start = time.time()
    shutil.copytree(drive_path, local_path, dirs_exist_ok=True)
    elapsed = time.time() - start
    print(f"  ✅ {label} copied in {elapsed:.0f}s")


print("=" * 60)
print("📋 COPYING DATA → LOCAL FAST STORAGE")
print("   (Only needed once per session)")
print("=" * 60)

# Copy val
copy_to_local(
    os.path.join(DRIVE_EXTRACT, 'val'),
    os.path.join(LOCAL_DATA, 'val'),
    "Val"
)

# Copy test
copy_to_local(
    os.path.join(DRIVE_EXTRACT, 'test'),
    os.path.join(LOCAL_DATA, 'test'),
    "Test"
)

# Copy train (this takes 5-10 min for ~21GB — worth it for training speed)
copy_to_local(
    os.path.join(DRIVE_EXTRACT, 'train'),
    os.path.join(LOCAL_DATA, 'train'),
    "Train"
)

print("\n✅ All data ready on local fast storage!")


# %% CELL 6: Load & Parse OpenPose Keypoints

def load_sentence_keypoints(sentence_folder_path):
    """
    Load all frame JSONs from a sentence folder → numpy array.

    Returns: np.array of shape (T, 137, 2)
        T = number of frames
        137 = 25 body + 70 face + 21 left hand + 21 right hand
        2 = (x, y) coordinates
    """
    frame_files = sorted([
        f for f in os.listdir(sentence_folder_path)
        if f.endswith('_keypoints.json')
    ])

    if not frame_files:
        return None

    all_frames = []
    for ff in frame_files:
        with open(os.path.join(sentence_folder_path, ff)) as f:
            data = json.load(f)

        if not data['people']:
            # No person detected — use zeros
            all_frames.append(np.zeros((137, 2), dtype=np.float32))
            continue

        person = data['people'][0]

        # Extract (x, y) from each keypoint group, skip confidence
        body = np.array(person['pose_keypoints_2d']).reshape(-1, 3)[:, :2]      # (25, 2)
        face = np.array(person['face_keypoints_2d']).reshape(-1, 3)[:, :2]      # (70, 2)
        lhand = np.array(person['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2] # (21, 2)
        rhand = np.array(person['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2] # (21, 2)

        # Concatenate: 25 + 70 + 21 + 21 = 137 keypoints
        frame_kps = np.concatenate([body, face, lhand, rhand], axis=0)  # (137, 2)
        all_frames.append(frame_kps.astype(np.float32))

    return np.stack(all_frames)  # (T, 137, 2)


# Test on one sentence (use LOCAL fast storage)
json_dir = os.path.join(LOCAL_DATA, 'val', 'openpose_output', 'json')
if os.path.exists(json_dir):
    sample_folder = sorted(os.listdir(json_dir))[0]
    sample_kps = load_sentence_keypoints(os.path.join(json_dir, sample_folder))

    print(f"📊 Sample sentence: {sample_folder}")
    print(f"   Shape: {sample_kps.shape}  →  {sample_kps.shape[0]} frames × {sample_kps.shape[1]} keypoints × {sample_kps.shape[2]} coords")
    print(f"   X range: [{sample_kps[...,0].min():.1f}, {sample_kps[...,0].max():.1f}]")
    print(f"   Y range: [{sample_kps[...,1].min():.1f}, {sample_kps[...,1].max():.1f}]")
else:
    print("⚠️  Val data not yet extracted/copied. Run cells 4 and 5 first.")


# %% CELL 7: Pair CSV Sentences with Keypoint Folders
csv_dir = os.path.join(DRIVE_DATA, 'csv')
json_dir_val = os.path.join(LOCAL_DATA, 'val', 'openpose_output', 'json')

if os.path.exists(json_dir_val) and os.path.exists(csv_dir):
    # Load CSV
    val_csv = pd.read_csv(os.path.join(csv_dir, 'how2sign_realigned_val.csv'), sep='\t')
    print(f"Val CSV: {val_csv.shape[0]} sentences")

    # Available keypoint folders
    available_folders = set(os.listdir(json_dir_val))
    print(f"Available keypoint folders: {len(available_folders)}")

    # Match using SENTENCE_NAME column
    for idx, row in val_csv.head(10).iterrows():
        sname = row['SENTENCE_NAME']
        found = sname in available_folders
        status = "✅" if found else "❌"
        print(f"  {status} '{sname}'")

    # Count total matches
    total_matched = sum(1 for _, row in val_csv.iterrows() if row['SENTENCE_NAME'] in available_folders)
    print(f"\n📊 Total matched: {total_matched}/{len(val_csv)}")
else:
    print("⚠️  Data not ready. Run cells 3-5 first.")


# %% CELL 8: Build Vocabulary from Training Text
from collections import Counter

def simple_tokenize(text):
    """Simple word-level tokenizer."""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    return text.split()

# Load train CSV and build vocab
csv_dir = os.path.join(DRIVE_DATA, 'csv')
train_csv = pd.read_csv(os.path.join(csv_dir, 'how2sign_realigned_train.csv'), sep='\t')

word_counts = Counter()
for text in train_csv['SENTENCE']:
    word_counts.update(simple_tokenize(text))

print(f"Total unique words: {len(word_counts)}")
print(f"Top 20: {word_counts.most_common(20)}")

# Special tokens + vocabulary
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
MIN_FREQ = 2
vocab_words = [w for w, c in word_counts.most_common() if c >= MIN_FREQ]
vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS + vocab_words)}
idx_to_word = {v: k for k, v in vocab.items()}

PAD_IDX = vocab['<PAD>']
SOS_IDX = vocab['<SOS>']
EOS_IDX = vocab['<EOS>']
UNK_IDX = vocab['<UNK>']

print(f"\nVocab size: {len(vocab)} (min_freq={MIN_FREQ})")

# Save vocab to DRIVE (persists!)
vocab_path = os.path.join(DRIVE_VOCAB, 'vocab.pkl')
with open(vocab_path, 'wb') as f:
    pickle.dump({'vocab': vocab, 'idx_to_word': idx_to_word,
                 'PAD_IDX': PAD_IDX, 'SOS_IDX': SOS_IDX,
                 'EOS_IDX': EOS_IDX, 'UNK_IDX': UNK_IDX}, f)
print(f"✅ Vocab saved to Drive: {vocab_path}")


# %% CELL 9: Dataset & DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- Constants ----
NUM_KEYPOINTS = 137      # 25 body + 70 face + 21 left hand + 21 right hand
KP_DIM = NUM_KEYPOINTS * 2  # 274 (x, y for each)
MAX_TEXT_LEN = 50        # Max words in a sentence
MAX_KP_FRAMES = 200      # Max frames per sentence


def normalize_keypoints(kps):
    """
    Normalize keypoints to [0, 1] range per-sample.
    kps: (T, 137, 2)
    """
    valid = np.any(kps != 0, axis=-1)  # (T, 137)
    if valid.sum() == 0:
        return kps

    for dim in range(2):  # x and y separately
        vals = kps[..., dim][valid]
        vmin, vmax = vals.min(), vals.max()
        vrange = vmax - vmin if vmax != vmin else 1.0
        kps[..., dim] = (kps[..., dim] - vmin) / vrange

    kps[~valid] = 0.0
    return kps.astype(np.float32)


def encode_text(text, vocab, max_len=MAX_TEXT_LEN):
    """Text → list of token indices with SOS/EOS."""
    tokens = simple_tokenize(text)[:max_len - 2]
    return [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]


class SignLanguageDataset(Dataset):
    """Pairs English sentences with OpenPose keypoint sequences."""

    def __init__(self, csv_path, json_base_dir, vocab, max_text_len=MAX_TEXT_LEN,
                 max_kp_frames=MAX_KP_FRAMES):
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.max_kp_frames = max_kp_frames

        # Load CSV
        df = pd.read_csv(csv_path, sep='\t')

        # Filter to only sentences with available keypoints
        available = set(os.listdir(json_base_dir)) if os.path.exists(json_base_dir) else set()
        self.data = []
        for _, row in df.iterrows():
            if row['SENTENCE_NAME'] in available:
                self.data.append({
                    'text': row['SENTENCE'],
                    'kp_folder': os.path.join(json_base_dir, row['SENTENCE_NAME']),
                    'sentence_id': row['SENTENCE_ID'],
                })
        print(f"  Dataset: {len(self.data)} pairs (from {len(df)} total)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Encode text
        text_ids = encode_text(item['text'], self.vocab, self.max_text_len)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # Load keypoints
        kps = load_sentence_keypoints(item['kp_folder'])  # (T, 137, 2)
        if kps is None:
            kps = np.zeros((1, NUM_KEYPOINTS, 2), dtype=np.float32)

        # Normalize
        kps = normalize_keypoints(kps.copy())

        # Truncate
        if len(kps) > self.max_kp_frames:
            kps = kps[:self.max_kp_frames]

        # Flatten: (T, 137, 2) → (T, 274)
        T = kps.shape[0]
        kps_flat = kps.reshape(T, -1)
        kps_tensor = torch.tensor(kps_flat, dtype=torch.float32)

        return text_tensor, kps_tensor


def collate_fn(batch):
    """Pad variable-length text and keypoint sequences."""
    texts, keypoints = zip(*batch)

    text_lengths = torch.tensor([len(t) for t in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)

    kp_lengths = torch.tensor([len(k) for k in keypoints])
    keypoints_padded = pad_sequence(keypoints, batch_first=True, padding_value=0.0)

    return texts_padded, text_lengths, keypoints_padded, kp_lengths


# %% CELL 10: Create DataLoaders & Test a Batch
csv_dir = os.path.join(DRIVE_DATA, 'csv')

# Use LOCAL fast storage for data loading
json_dir_val = os.path.join(LOCAL_DATA, 'val', 'openpose_output', 'json')
json_dir_train = os.path.join(LOCAL_DATA, 'train', 'openpose_output', 'json')
json_dir_test = os.path.join(LOCAL_DATA, 'test', 'openpose_output', 'json')

print("Creating datasets...")

# Val dataset
val_dataset = SignLanguageDataset(
    csv_path=os.path.join(csv_dir, 'how2sign_realigned_val.csv'),
    json_base_dir=json_dir_val,
    vocab=vocab
)

val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False,
    collate_fn=collate_fn, num_workers=2
)

# Train dataset
train_dataset = SignLanguageDataset(
    csv_path=os.path.join(csv_dir, 'how2sign_realigned_train.csv'),
    json_base_dir=json_dir_train,
    vocab=vocab
)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True,
    collate_fn=collate_fn, num_workers=2
)

# Test batch
batch = next(iter(val_loader))
texts_padded, text_lengths, kps_padded, kp_lengths = batch

print(f"\n✅ DataLoader working!")
print(f"  Text batch:      {texts_padded.shape}  (batch × max_text_len)")
print(f"  Text lengths:    {text_lengths.tolist()}")
print(f"  Keypoint batch:  {kps_padded.shape}  (batch × max_frames × {KP_DIM})")
print(f"  KP lengths:      {kp_lengths.tolist()}")
print(f"  Feature dim:     {KP_DIM} = {NUM_KEYPOINTS} keypoints × 2 coords")
print(f"  Train samples:   {len(train_dataset)}")
print(f"  Val samples:     {len(val_dataset)}")


# %% CELL 11: Visualize Keypoints
import matplotlib.pyplot as plt

def visualize_keypoints_frame(kps_frame, title=""):
    """
    Visualize one frame. kps_frame: (137, 2) or (274,)
    """
    if len(kps_frame.shape) == 1:
        kps_frame = kps_frame.reshape(-1, 2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Body (0-24), Face (25-94), Left Hand (95-115), Right Hand (116-136)
    parts = [
        ("Body (25 pts)", kps_frame[:25]),
        ("Face (70 pts)", kps_frame[25:95]),
        ("Left Hand (21 pts)", kps_frame[95:116]),
        ("Right Hand (21 pts)", kps_frame[116:137]),
    ]

    for ax, (name, pts) in zip(axes, parts):
        valid = np.any(pts != 0, axis=-1)
        ax.scatter(pts[valid, 0], -pts[valid, 1], s=8, alpha=0.8)
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Visualize a sample
sample_text, sample_kps = val_dataset[0]
kps_np = sample_kps.numpy().reshape(-1, NUM_KEYPOINTS, 2)
mid = len(kps_np) // 2
decoded_text = ' '.join([idx_to_word.get(i.item(), '?') for i in sample_text[1:-1]])
visualize_keypoints_frame(kps_np[mid], f"Frame {mid}: \"{decoded_text[:60]}...\"")


# %% CELL 12: 🧠 Define Text-to-Keypoints Transformer Model
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Text2KeypointsModel(nn.Module):
    """
    Transformer Encoder-Decoder for Text → Keypoint Sequence generation.

    Encoder: Processes tokenized text
    Decoder: Autoregressively generates keypoint frames

    Input:  Text tokens (B, S)
    Output: Keypoint sequence (B, T, 274)
    """
    def __init__(self, vocab_size, kp_dim=274, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_kp_frames=200):
        super().__init__()

        self.d_model = d_model
        self.kp_dim = kp_dim
        self.max_kp_frames = max_kp_frames

        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Keypoint input projection (for decoder input)
        self.kp_input_proj = nn.Linear(kp_dim, d_model)
        self.kp_pos_enc = PositionalEncoding(d_model, max_len=max_kp_frames + 1, dropout=dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection: d_model → keypoint dim
        self.output_proj = nn.Linear(d_model, kp_dim)

        # Learnable start-of-sequence token for decoder
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Length predictor: predict number of output frames from encoder output
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, text_tokens, text_lengths, target_kps=None, target_lengths=None):
        """
        Training forward pass (teacher forcing).

        Args:
            text_tokens: (B, S) padded text token indices
            text_lengths: (B,) actual text lengths
            target_kps: (B, T, 274) padded target keypoint sequences
            target_lengths: (B,) actual keypoint sequence lengths

        Returns: dict with 'predicted_kps' (B, T, 274), 'predicted_lengths' (B, 1)
        """
        B = text_tokens.size(0)
        device = text_tokens.device

        # ── Encoder ──
        # Create text padding mask
        text_pad_mask = torch.arange(text_tokens.size(1), device=device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        text_emb = self.text_pos_enc(text_emb)

        memory = self.transformer.encoder(text_emb, src_key_padding_mask=text_pad_mask)

        # Predict output length
        # Use mean pooling over non-padded encoder outputs
        mask_for_pool = (~text_pad_mask).float().unsqueeze(-1)  # (B, S, 1)
        pooled = (memory * mask_for_pool).sum(dim=1) / mask_for_pool.sum(dim=1).clamp(min=1)
        predicted_lengths = self.length_predictor(pooled)  # (B, 1)

        # ── Decoder (teacher forcing) ──
        if target_kps is not None:
            T = target_kps.size(1)

            # Prepend SOS token, remove last frame (shift right)
            sos = self.sos_token.expand(B, 1, -1)
            target_proj = self.kp_input_proj(target_kps[:, :-1])  # (B, T-1, d_model)
            decoder_input = torch.cat([sos, target_proj], dim=1)  # (B, T, d_model)
            decoder_input = self.kp_pos_enc(decoder_input)

            # Causal mask
            tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(device)

            # Target padding mask
            tgt_pad_mask = torch.arange(T, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)

            # Decode
            decoded = self.transformer.decoder(
                decoder_input,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=text_pad_mask,
            )

            predicted_kps = self.output_proj(decoded)  # (B, T, 274)
        else:
            predicted_kps = None

        return {
            'predicted_kps': predicted_kps,
            'predicted_lengths': predicted_lengths,
        }

    @torch.no_grad()
    def generate(self, text_tokens, text_lengths, max_frames=200):
        """
        Autoregressive inference: generate keypoints frame by frame.

        Args:
            text_tokens: (1, S) text token indices
            text_lengths: (1,) text length

        Returns: (1, T, 274) generated keypoint sequence
        """
        self.eval()
        device = text_tokens.device

        # Encode text
        text_pad_mask = torch.arange(text_tokens.size(1), device=device).unsqueeze(0) >= text_lengths.unsqueeze(1)
        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        text_emb = self.text_pos_enc(text_emb)
        memory = self.transformer.encoder(text_emb, src_key_padding_mask=text_pad_mask)

        # Predict length
        mask_for_pool = (~text_pad_mask).float().unsqueeze(-1)
        pooled = (memory * mask_for_pool).sum(dim=1) / mask_for_pool.sum(dim=1).clamp(min=1)
        pred_len = int(self.length_predictor(pooled).item())
        pred_len = max(1, min(pred_len, max_frames))

        # Autoregressive decoding
        generated = []
        decoder_input = self.sos_token  # (1, 1, d_model)

        for t in range(pred_len):
            dec_input = self.kp_pos_enc(decoder_input)
            tgt_mask = self.transformer.generate_square_subsequent_mask(t + 1).to(device)

            decoded = self.transformer.decoder(
                dec_input, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=text_pad_mask,
            )

            # Get last frame prediction
            frame_pred = self.output_proj(decoded[:, -1:])  # (1, 1, 274)
            generated.append(frame_pred)

            # Append to decoder input for next step
            next_input = self.kp_input_proj(frame_pred)  # (1, 1, d_model)
            decoder_input = torch.cat([decoder_input, next_input], dim=1)

        return torch.cat(generated, dim=1)  # (1, T, 274)


print("✅ Text2KeypointsModel defined")
print(f"   Input:  text tokens → Output: keypoint sequence (T, {KP_DIM})")


# %% CELL 13: 🚀 Train the Model (Checkpoints saved to Google Drive!)
import torch.optim as optim

# ─── Hyperparameters ───
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
SAVE_EVERY = 5  # Save checkpoint every N epochs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {device}")

# ─── Create model ───
model = Text2KeypointsModel(
    vocab_size=len(vocab),
    kp_dim=KP_DIM,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    max_kp_frames=MAX_KP_FRAMES,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model parameters: {total_params:,}")

# ─── Optimizer & Loss ───
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
kp_criterion = nn.MSELoss(reduction='none')  # Per-element, we'll mask padding
length_criterion = nn.MSELoss()

# ─── DataLoaders ───
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

# ─── Check for existing checkpoint to resume ───
def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(ckpt_dir):
        return None, 0
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('text2kp_epoch_') and f.endswith('.pt')]
    if not ckpts:
        return None, 0
    # Sort by epoch number
    ckpts.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    latest = ckpts[-1]
    epoch = int(latest.split('_')[2].split('.')[0])
    return os.path.join(ckpt_dir, latest), epoch

latest_ckpt, start_epoch = find_latest_checkpoint(DRIVE_CKPT)
if latest_ckpt:
    print(f"📂 Found checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resumed from epoch {checkpoint['epoch']} (starting epoch {start_epoch})")
else:
    start_epoch = 0
    print("🆕 Starting fresh training")


# ─── Training Loop ───
print("=" * 60)
print("🚀 TRAINING")
print(f"   Epochs: {start_epoch} → {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Val: {len(val_dataset)} samples")
print(f"   Checkpoints → {DRIVE_CKPT}")
print("=" * 60)

best_val_loss = float('inf')

for epoch in range(start_epoch, NUM_EPOCHS):
    # ── Train ──
    model.train()
    train_loss_total = 0
    train_batches = 0

    for batch_idx, (texts_pad, text_lens, kps_pad, kp_lens) in enumerate(train_loader):
        texts_pad = texts_pad.to(device)
        text_lens = text_lens.to(device)
        kps_pad = kps_pad.to(device)
        kp_lens = kp_lens.to(device)

        # Forward
        output = model(texts_pad, text_lens, target_kps=kps_pad, target_lengths=kp_lens)

        # Keypoint reconstruction loss (masked)
        pred_kps = output['predicted_kps']
        T = pred_kps.size(1)
        mask = torch.arange(T, device=device).unsqueeze(0) < kp_lens.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1).expand_as(pred_kps)  # (B, T, 274)

        kp_loss = (kp_criterion(pred_kps, kps_pad[:, :T]) * mask).sum() / mask.sum().clamp(min=1)

        # Length prediction loss
        len_loss = length_criterion(
            output['predicted_lengths'].squeeze(-1),
            kp_lens.float()
        )

        loss = kp_loss + 0.01 * len_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_total += loss.item()
        train_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} (kp: {kp_loss.item():.4f}, len: {len_loss.item():.4f})")

    avg_train_loss = train_loss_total / max(train_batches, 1)

    # ── Validate ──
    model.eval()
    val_loss_total = 0
    val_batches = 0
    with torch.no_grad():
        for texts_pad, text_lens, kps_pad, kp_lens in val_loader:
            texts_pad = texts_pad.to(device)
            text_lens = text_lens.to(device)
            kps_pad = kps_pad.to(device)
            kp_lens = kp_lens.to(device)

            output = model(texts_pad, text_lens, target_kps=kps_pad, target_lengths=kp_lens)
            pred_kps = output['predicted_kps']
            T = pred_kps.size(1)
            mask = torch.arange(T, device=device).unsqueeze(0) < kp_lens.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(pred_kps)
            kp_loss = (kp_criterion(pred_kps, kps_pad[:, :T]) * mask).sum() / mask.sum().clamp(min=1)
            val_loss_total += kp_loss.item()
            val_batches += 1

    avg_val_loss = val_loss_total / max(val_batches, 1)
    scheduler.step(avg_val_loss)

    print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ── Save checkpoint to DRIVE ──
    if (epoch + 1) % SAVE_EVERY == 0 or avg_val_loss < best_val_loss:
        ckpt_path = os.path.join(DRIVE_CKPT, f'text2kp_epoch_{epoch+1:03d}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'vocab_size': len(vocab),
            'config': {
                'd_model': 256, 'nhead': 8,
                'num_encoder_layers': 4, 'num_decoder_layers': 4,
                'dim_feedforward': 1024, 'kp_dim': KP_DIM,
            }
        }, ckpt_path)
        print(f"💾 Checkpoint saved to Drive: {ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(DRIVE_CKPT, 'text2kp_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'vocab_size': len(vocab),
                'config': {
                    'd_model': 256, 'nhead': 8,
                    'num_encoder_layers': 4, 'num_decoder_layers': 4,
                    'dim_feedforward': 1024, 'kp_dim': KP_DIM,
                }
            }, best_path)
            print(f"⭐ Best model saved! Val loss: {best_val_loss:.4f}")

print("\n✅ Training complete!")


# %% CELL 14: 🔄 Resume Training After Disconnect
# ── If Colab disconnects, run Cells 1-5, then skip to here ──

# This cell does everything needed to resume:
# 1. Loads vocab from Drive
# 2. Recreates datasets from local data
# 3. Loads latest checkpoint from Drive
# 4. Continues training

# NOTE: Make sure you've run Cells 1-5 first to remount Drive and
#       copy data to local storage!

print("🔄 RESUMING TRAINING AFTER DISCONNECT")
print("=" * 60)

# Load vocab from Drive
vocab_path = os.path.join(DRIVE_VOCAB, 'vocab.pkl')
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
vocab = vocab_data['vocab']
idx_to_word = vocab_data['idx_to_word']
PAD_IDX = vocab_data['PAD_IDX']
SOS_IDX = vocab_data['SOS_IDX']
EOS_IDX = vocab_data['EOS_IDX']
UNK_IDX = vocab_data['UNK_IDX']
print(f"✅ Loaded vocab ({len(vocab)} words) from Drive")

# Recreate datasets
csv_dir = os.path.join(DRIVE_DATA, 'csv')
json_dir_val = os.path.join(LOCAL_DATA, 'val', 'openpose_output', 'json')
json_dir_train = os.path.join(LOCAL_DATA, 'train', 'openpose_output', 'json')

train_dataset = SignLanguageDataset(
    csv_path=os.path.join(csv_dir, 'how2sign_realigned_train.csv'),
    json_base_dir=json_dir_train, vocab=vocab
)
val_dataset = SignLanguageDataset(
    csv_path=os.path.join(csv_dir, 'how2sign_realigned_val.csv'),
    json_base_dir=json_dir_val, vocab=vocab
)

# Now run Cell 13 — it will auto-detect and load the latest checkpoint!
print("\n✅ Ready! Now run Cell 13 to continue training from last checkpoint.")


# %% CELL 15: Pipeline Summary
print("=" * 60)
print("📊 DATA PIPELINE SUMMARY")
print("=" * 60)
print(f"Vocab size:           {len(vocab)}")
print(f"Keypoints/frame:      {NUM_KEYPOINTS} (25 body + 70 face + 21+21 hands)")
print(f"Feature dim:          {KP_DIM}")
print(f"Max text length:      {MAX_TEXT_LEN}")
print(f"Max KP frames:        {MAX_KP_FRAMES}")
print(f"Train dataset size:   {len(train_dataset)}")
print(f"Val dataset size:     {len(val_dataset)}")

print(f"\n📁 Storage (Google Drive — persistent):")
print(f"   Extracted data:    {DRIVE_EXTRACT}")
print(f"   Vocabulary:        {DRIVE_VOCAB}")
print(f"   Checkpoints:       {DRIVE_CKPT}")

# Check for checkpoints
if os.path.exists(DRIVE_CKPT):
    ckpts = [f for f in os.listdir(DRIVE_CKPT) if f.endswith('.pt')]
    if ckpts:
        print(f"\n💾 Saved checkpoints ({len(ckpts)}):")
        for c in sorted(ckpts):
            size_mb = os.path.getsize(os.path.join(DRIVE_CKPT, c)) / (1024**2)
            print(f"   {c} ({size_mb:.1f} MB)")

# Get stats
frame_counts = []
text_lens = []
for i in range(min(100, len(val_dataset))):
    t, k = val_dataset[i]
    text_lens.append(len(t))
    frame_counts.append(len(k))

print(f"\nText length stats (first 100):")
print(f"  min={min(text_lens)}, max={max(text_lens)}, avg={sum(text_lens)/len(text_lens):.1f}")
print(f"Frame count stats (first 100):")
print(f"  min={min(frame_counts)}, max={max(frame_counts)}, avg={sum(frame_counts)/len(frame_counts):.1f}")

print(f"\n✅ Pipeline ready!")
print("=" * 60)
"""
