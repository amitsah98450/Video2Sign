"""
=============================================================
VIDEO TO SIGN LANGUAGE — Complete Colab Training Notebook
=============================================================
Copy each section (# %% CELL N) into separate Colab cells.
Run them one at a time, top to bottom.

Data Format Discovered:
- OpenPose JSON format
- 137 keypoints per frame: 25 body + 70 face + 21 left hand + 21 right hand
- Each keypoint has (x, y, confidence) — we use (x, y) only → 274 values/frame
- Folder name = SENTENCE_NAME from CSV
- One JSON per frame, ~189 frames per sentence
=============================================================
"""

# %% CELL 1: Setup & Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import os, json, re, pickle
import numpy as np
import pandas as pd

DATA_DIR = '/content/drive/MyDrive/VideoToSign/data'
WORK_DIR = '/content/how2sign_data'
os.makedirs(WORK_DIR, exist_ok=True)

print("📂 Files in data directory:")
for f in sorted(os.listdir(DATA_DIR)):
    path = os.path.join(DATA_DIR, f)
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"  {f} — {size_mb:.1f} MB")
    else:
        print(f"  📁 {f}/")


# %% CELL 2: Upload English Translation CSVs (if not already uploaded)
csv_dir = os.path.join(DATA_DIR, 'csv')
os.makedirs(csv_dir, exist_ok=True)

existing = [f for f in os.listdir(csv_dir) if f.endswith('.csv')] if os.path.exists(csv_dir) else []
if len(existing) < 3:
    from google.colab import files
    print("📤 Upload the 3 CSV files: how2sign_realigned_train/val/test.csv")
    uploaded = files.upload()
    for fname, content in uploaded.items():
        with open(os.path.join(csv_dir, fname), 'wb') as f:
            f.write(content)
        print(f"  ✅ {fname}")
else:
    print(f"✅ CSVs already exist: {existing}")


# %% CELL 3: Extract Keypoints (JSON only, skip videos)
import tarfile

def extract_json_keypoints(archive_path, work_dir, label):
    """Extract only the json/ folder from the archive (skip video/)."""
    json_check = os.path.join(work_dir, 'openpose_output', 'json')
    if os.path.exists(json_check) and len(os.listdir(json_check)) > 0:
        count = len(os.listdir(json_check))
        print(f"  ✅ {label} already extracted ({count} folders)")
        return

    print(f"📦 Extracting {label}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        json_members = [m for m in tar.getmembers() if '/json/' in m.name]
        print(f"  Extracting {len(json_members)} entries...")
        tar.extractall(work_dir, members=json_members)
    
    count = len(os.listdir(json_check))
    print(f"  ✅ Done! {count} sentence folders extracted")

# Extract val (smallest, ~1.16 GB)
extract_json_keypoints(
    os.path.join(DATA_DIR, 'keypoints_val.zip'),
    os.path.join(WORK_DIR, 'val'),
    "Val keypoints"
)

# Extract test (~1.58 GB)
extract_json_keypoints(
    os.path.join(DATA_DIR, 'keypoints_test.zip'),
    os.path.join(WORK_DIR, 'test'),
    "Test keypoints"
)

# Extract train (~21 GB) — takes 10-15 min!
extract_json_keypoints(
    os.path.join(DATA_DIR, 'train_2D_keypoints.tar.gz'),
    os.path.join(WORK_DIR, 'train'),
    "Train keypoints"
)


# %% CELL 4: Load & Parse OpenPose Keypoints
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


# Test on one sentence
json_dir = os.path.join(WORK_DIR, 'val', 'openpose_output', 'json')
sample_folder = sorted(os.listdir(json_dir))[0]
sample_kps = load_sentence_keypoints(os.path.join(json_dir, sample_folder))

print(f"📊 Sample sentence: {sample_folder}")
print(f"   Shape: {sample_kps.shape}  →  {sample_kps.shape[0]} frames × {sample_kps.shape[1]} keypoints × {sample_kps.shape[2]} coords")
print(f"   X range: [{sample_kps[...,0].min():.1f}, {sample_kps[...,0].max():.1f}]")
print(f"   Y range: [{sample_kps[...,1].min():.1f}, {sample_kps[...,1].max():.1f}]")


# %% CELL 5: Pair CSV Sentences with Keypoint Folders
csv_dir = os.path.join(DATA_DIR, 'csv')
json_dir_val = os.path.join(WORK_DIR, 'val', 'openpose_output', 'json')

# Load CSV
val_csv = pd.read_csv(os.path.join(csv_dir, 'how2sign_realigned_val.csv'), sep='\t')
print(f"Val CSV: {val_csv.shape[0]} sentences")

# Available keypoint folders
available_folders = set(os.listdir(json_dir_val))
print(f"Available keypoint folders: {len(available_folders)}")

# Match using SENTENCE_NAME column
matched = 0
missing = 0
sample_matches = []

for idx, row in val_csv.head(10).iterrows():
    sname = row['SENTENCE_NAME']
    found = sname in available_folders
    status = "✅" if found else "❌"
    print(f"  {status} '{sname}'")
    if found:
        matched += 1
    else:
        missing += 1

# Count total matches
total_matched = sum(1 for _, row in val_csv.iterrows() if row['SENTENCE_NAME'] in available_folders)
print(f"\n📊 Total matched: {total_matched}/{len(val_csv)}")


# %% CELL 6: Build Vocabulary from Training Text
from collections import Counter

def simple_tokenize(text):
    """Simple word-level tokenizer."""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    return text.split()

# Load train CSV and build vocab
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

# Save vocab
vocab_path = os.path.join(DATA_DIR, 'vocab.pkl')
with open(vocab_path, 'wb') as f:
    pickle.dump({'vocab': vocab, 'idx_to_word': idx_to_word,
                 'PAD_IDX': PAD_IDX, 'SOS_IDX': SOS_IDX,
                 'EOS_IDX': EOS_IDX, 'UNK_IDX': UNK_IDX}, f)
print(f"✅ Vocab saved to {vocab_path}")


# %% CELL 7: Complete Data Pipeline — Dataset & DataLoader
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
        available = set(os.listdir(json_base_dir))
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


# %% CELL 8: Create DataLoaders & Test a Batch
json_dir_val = os.path.join(WORK_DIR, 'val', 'openpose_output', 'json')

print("Creating val dataset...")
val_dataset = SignLanguageDataset(
    csv_path=os.path.join(csv_dir, 'how2sign_realigned_val.csv'),
    json_base_dir=json_dir_val,
    vocab=vocab
)

val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False,
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
print(f"\n  Feature dim: {KP_DIM} = {NUM_KEYPOINTS} keypoints × 2 coords")


# %% CELL 9: Visualize Keypoints
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


# %% CELL 10: Pipeline Summary
print("=" * 60)
print("📊 DATA PIPELINE SUMMARY")
print("=" * 60)
print(f"Vocab size:        {len(vocab)}")
print(f"Keypoints/frame:   {NUM_KEYPOINTS} (25 body + 70 face + 21+21 hands)")
print(f"Feature dim:       {KP_DIM}")
print(f"Max text length:   {MAX_TEXT_LEN}")
print(f"Max KP frames:     {MAX_KP_FRAMES}")
print(f"Val dataset size:  {len(val_dataset)}")

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

print(f"\n✅ Pipeline ready! Next → Build Transformer model")
print("=" * 60)
