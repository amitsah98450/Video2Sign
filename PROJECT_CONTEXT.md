# 🤟 Video to Sign Language — Full Project Context

> **Purpose**: Feed this document to Claude (or any AI assistant) on a new machine so it has complete context about this project.

---

## 1. Project Overview

**B.Tech Final Year Project** — An AI-powered web application that converts spoken language in videos into animated sign language, bridging the communication gap for the deaf and hard-of-hearing community.

**GitHub Repo**: https://github.com/amitsah98450/Video2Sign.git
**Developer**: Amit Kumar Sah

### What It Does (End-to-End Pipeline)

1. User uploads a video/audio file via the React frontend
2. The Flask backend transcribes the speech using **OpenAI Whisper**
3. Transcribed text is converted to sign language tokens
4. Sign language is displayed via animated GIF/SVG assets (word-level signs + letter-by-letter fingerspelling fallback)
5. **(ML Research Module)**: A Transformer generates body keypoint sequences from text, then a TPS Motion Model produces photorealistic sign language video

### Two Rendering Modes

1. **GIF/SVG-based** (working) — Maps words to pre-built sign language GIF assets; unknown words are finger-spelled letter-by-letter using SVG hand assets
2. **AI-generated photorealistic** (research/training stage) — Transformer encoder-decoder generates 137-point body/hand/face keypoints from text → TPS Motion Model renders realistic human video frames

---

## 2. Tech Stack

| Layer           | Technology                       | Purpose                                                             |
| --------------- | -------------------------------- | ------------------------------------------------------------------- |
| Frontend        | React 19, Vite 7, Vanilla CSS    | Component-based UI with glassmorphism dark theme                    |
| Backend         | Flask 3.1, Flask-CORS, Gunicorn  | REST API for upload, transcription, sign conversion                 |
| ASR             | OpenAI Whisper (base model)      | Speech-to-text with auto-translation                                |
| ML Framework    | PyTorch                          | Deep learning model training and inference                          |
| Text→Keypoints  | Transformer (Encoder-Decoder)    | Generates 137-keypoint sequences from English text                  |
| Keypoints→Video | TPS Motion Model (CVPR 2022)     | Photorealistic frame generation                                     |
| Dataset         | How2Sign                         | 35K+ sentence-level English ↔ ASL pairs with CMU OpenPose keypoints |
| Training Infra  | Google Colab (GPU) / Windows GPU | Cloud or local GPU training                                         |

---

## 3. Project Structure

```
Video to Sign/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx              # Main app: 3 views (upload → transcription → sign)
│   │   ├── api.js               # API client (uploadVideo, textToSign, healthCheck)
│   │   ├── components/
│   │   │   ├── Header.jsx       # Nav header with back/reset controls
│   │   │   ├── VideoUpload.jsx  # Drag-and-drop upload with progress
│   │   │   ├── TranscriptionView.jsx  # Timestamped transcription display
│   │   │   └── SignLanguagePlayer.jsx # Animated sign playback with speed control
│   │   ├── index.css            # Global styles (glassmorphism, dark theme)
│   │   └── main.jsx             # React entry point
│   ├── package.json
│   └── vite.config.js
│
├── backend/                     # Flask REST API
│   ├── app.py                   # Main Flask server (port 5001)
│   ├── sign_dictionary.py       # Word → sign asset mapping + fingerspelling fallback
│   ├── signs/                   # Sign language GIF assets (word-level)
│   │   └── letters/             # A-Z SVG fingerspelling assets
│   ├── requirements.txt         # flask, flask-cors, openai-whisper, torch, gunicorn
│   └── uploads/                 # Temporary upload directory (auto-cleaned)
│
├── ml/                          # Machine Learning module
│   ├── colab_notebook.py        # Complete Colab training notebook (~1000 lines)
│   ├── data/                    # How2Sign CSV metadata
│   │   ├── how2sign_realigned_train.csv  (~5.6 MB, ~26K sentences)
│   │   ├── how2sign_realigned_val.csv    (~311 KB, ~2K sentences)
│   │   └── how2sign_realigned_test.csv   (~424 KB, ~3K sentences)
│   ├── transcription/
│   │   └── transcribe.py        # Whisper transcription utility
│   ├── model/
│   │   ├── model.py             # TPSMotionModel (full model combining all sub-networks)
│   │   ├── train.py             # Training script (perceptual, equivariance, warp losses)
│   │   ├── inference.py         # TPSAnimator — generate video from keypoints
│   │   ├── dataset.py           # PyTorch dataset for TPS training
│   │   ├── dense_motion.py      # Dense motion estimation network
│   │   ├── inpainting_network.py # Inpainting generator network
│   │   ├── keypoint_detector.py # Keypoint detection network
│   │   ├── bg_motion_predictor.py # Background motion predictor
│   │   ├── avd_network.py       # AVD (Animate Via Disentangling) network
│   │   ├── config.yaml          # TPS model hyperparameters (VoxCeleb-256 based)
│   │   ├── util.py              # TPS transformations, coordinate grids
│   │   ├── checkpoints/         # Model weights (vox.pth.tar — NOT in git, 335MB)
│   │   ├── demo_tps.py          # TPS demo with pretrained weights
│   │   ├── demo_pretrained.py   # Pretrained model demo
│   │   ├── avatar_demo.py       # 3D avatar demo
│   │   └── demo_output/         # Generated demo images/GIFs
│   └── requirements.txt         # openai-whisper, torch, torchvision, numpy, etc.
│
├── .gitignore
├── .env                         # API keys (NOT in git)
└── README.md
```

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER (Browser)                                │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐   │
│  │ VideoUpload  │──▶│ TranscriptionView│──▶│ SignLanguagePlayer   │   │
│  │  (Upload)    │   │  (Review Text)   │   │ (Animated Playback)  │   │
│  └──────────────┘   └──────────────────┘   └──────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ REST API (HTTP)
┌──────────────────────────────▼──────────────────────────────────────────┐
│                       FLASK BACKEND (Port 5001)                        │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ /api/upload    │  │ /api/text-to-sign│  │ /api/signs/<filename>  │ │
│  │ Whisper ASR    │  │ Sign Dictionary  │  │ Serve GIF/SVG Assets   │ │
│  └────────────────┘  └──────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                     ML MODULE (Research / Training)                     │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │ Text-to-Keypoints           │  │ TPS Motion Model                 │ │
│  │ Transformer (Enc-Dec)       │  │ (CVPR 2022 — Image Animation)    │ │
│  │ Input: English text tokens  │  │ Input: source image + keypoints  │ │
│  │ Output: (T, 137, 2) kps    │  │ Output: photorealistic video     │ │
│  └─────────────────────────────┘  └──────────────────────────────────┘ │
│  Dataset: How2Sign (CMU OpenPose keypoints)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. API Endpoints

| Method | Endpoint                | Description                                                                   |
| ------ | ----------------------- | ----------------------------------------------------------------------------- |
| `GET`  | `/api/health`           | Health check → `{ status: "ok" }`                                             |
| `POST` | `/api/upload`           | Upload video/audio → Whisper transcription (full text + timestamped segments) |
| `POST` | `/api/text-to-sign`     | Convert text → sign language tokens (word signs or fingerspelling)            |
| `GET`  | `/api/signs/<filename>` | Serve sign language GIF/SVG assets                                            |

---

## 6. ML Pipeline (Deep Dive)

### Stage 1: Text → Keypoint Generation (Transformer)

The **Text2KeypointsModel** is a Transformer encoder-decoder:

- **Input**: Tokenized English sentence `(B, S)` with SOS/EOS tokens
- **Output**: Keypoint sequence `(B, T, 274)` — 137 keypoints × 2 (x, y)
- **Architecture**:
  - Embedding layer (vocab → d_model=256) + sinusoidal positional encoding
  - 4-layer Transformer encoder (8 heads, FFN=1024)
  - 4-layer Transformer decoder (8 heads, FFN=1024, teacher forcing during training)
  - Linear projection head → 274-dim keypoint output
  - Length predictor head → predicted sequence length
- **Training Loss**: MSE (keypoint reconstruction) + L1 (sequence length prediction)
- **Dataset**: How2Sign — 35K+ sentence-level English ↔ ASL pairs with CMU OpenPose keypoints
  - Train: ~26K sentences, Val: ~2K, Test: ~3K
- **Keypoint Format**: 137 keypoints per frame = 25 body + 70 face + 21 left hand + 21 right hand
- **Training Config**: batch_size=16, lr=1e-4, Adam optimizer, ReduceLROnPlateau, 100 epochs, max_kp_frames=200, max_text_len=50

### Stage 2: Keypoint → Video Generation (TPS Motion Model)

The **TPSMotionModel** (based on CVPR 2022 paper "Thin-Plate Spline Motion Model for Image Animation"):

- **Input**: Source reference image `(1, 3, 256, 256)` + driving keypoints
- **Output**: Animated video frame `(1, 3, 256, 256)`
- **Sub-networks**:
  - `KPDetector` — detects TPS control point keypoints from images (10 TPS transformations)
  - `BGMotionPredictor` — estimates background affine transformation
  - `DenseMotionNetwork` — computes dense optical flow + occlusion maps
  - `InpaintingNetwork` — generates final frame via warping + inpainting
  - `AVDNetwork` — Animate-via-Disentangling for cross-identity transfer
- **Training Losses**: Multi-scale VGG19 perceptual loss at 4 scales, equivariance value loss, warp loss, background loss
- **TPS Dropout Schedule**: starts at epoch 35, increases from 0.1 to 0.3
- **Pretrained Weights**: `vox.pth.tar` (335MB) — pretrained on VoxCeleb dataset

### TPS Model Config (config.yaml)

```yaml
model_params:
  common_params:
    num_tps: 10
    num_channels: 3
    bg: true
    multi_mask: true
  generator_params:
    block_expansion: 64, max_features: 512, num_down_blocks: 3
  dense_motion_params:
    block_expansion: 64, max_features: 1024, num_blocks: 5, scale_factor: 0.25

train_params:
  num_epochs: 100, lr_generator: 2.0e-4, batch_size: 16
  scales: [1, 0.5, 0.25, 0.125]
  dropout_epoch: 35, dropout_maxp: 0.3
  loss_weights:
    perceptual: [10, 10, 10, 10, 10]
    equivariance_value: 10, warp_loss: 10, bg: 10
```

---

## 7. How2Sign Dataset Details

- **Source**: https://how2sign.github.io/
- **Format**: CSV files with columns: SENTENCE_NAME, SENTENCE_ID, SENTENCE, START, END, etc.
- **Keypoints**: CMU OpenPose JSON files — one JSON per frame per sentence
  - Each JSON has `pose_keypoints_2d` (25 body), `face_keypoints_2d` (70 face), `hand_left_keypoints_2d` (21 left), `hand_right_keypoints_2d` (21 right)
  - Total: 137 keypoints × 2 (x, y) = 274 values per frame
- **Splits**:
  - Train: `how2sign_realigned_train.csv` (~26K sentences, ~5.6MB)
  - Val: `how2sign_realigned_val.csv` (~2K sentences, ~311KB)
  - Test: `how2sign_realigned_test.csv` (~424KB)
- **Keypoint archives**: Downloaded separately (multiple GB), extracted to Google Drive for Colab training

---

## 8. Colab Training Notebook (ml/colab_notebook.py)

The training notebook is designed for Google Colab with Google Drive persistence:

### Cell Structure

1. **Cell 1**: Mount Google Drive
2. **Cell 2**: Path configuration (all paths on Drive for persistence)
3. **Cell 3**: Copy CSVs to Drive
4. **Cell 4**: Extract keypoint archives → Drive (one-time)
5. **Cell 5**: Copy data to local SSD (fast I/O, once per session)
6. **Cell 6**: Load & parse OpenPose keypoints (`load_sentence_keypoints()`)
7. **Cell 7**: Build vocabulary from training text
8. **Cell 8**: Save vocabulary to Drive
9. **Cell 9**: Dataset & DataLoader (`SignLanguageDataset`, `collate_fn`)
10. **Cell 10+**: Training loop with Text2KeypointsModel

### Key Functions

- `load_sentence_keypoints(folder)` → numpy array `(T, 137, 2)`
- `normalize_keypoints(kps)` → normalized to [0, 1] per-sample
- `encode_text(text, vocab)` → token indices with SOS/EOS
- `SignLanguageDataset` → pairs English sentences with keypoint sequences
- `collate_fn` → pads variable-length text and keypoint sequences

### Resuming After Colab Disconnect

Run Cells 1-5 (mount Drive, copy data), then Cell 14 auto-loads vocab, finds latest checkpoint, resumes training.

---

## 9. Frontend Details

### Components

- **App.jsx** — State machine with 3 views: `upload` → `transcription` → `sign`
- **VideoUpload.jsx** — Drag-and-drop upload supporting MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, OGG
- **TranscriptionView.jsx** — Shows Whisper transcription with per-segment timestamps
- **SignLanguagePlayer.jsx** — Animated playback with play/pause, prev/next, speed control, word queue

### Styling

- Dark theme with glassmorphism effects
- Animated background glows (`bg-glow-1`, `bg-glow-2`, `bg-glow-3`)
- Responsive layout

### Running

```bash
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

---

## 10. Backend Details

### Flask Server (app.py)

- Runs on port **5001**
- CORS enabled for `localhost:5173` and `5174`
- Whisper model loaded on-demand (`whisper.load_model("base")`)
- `task="translate"` — auto-translates non-English audio to English
- Uploaded files are cleaned up after transcription
- Max upload size: 500MB

### Sign Dictionary (sign_dictionary.py)

- Maps words to GIF assets in `signs/` folder
- Stop words (`a`, `an`, `the`, `of`, `to`, etc.) are filtered out
- Unknown words → fingerspelled letter-by-letter using SVG assets in `signs/letters/`

### Running

```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py              # → http://localhost:5001
```

---

## 11. Important Files NOT in Git (Transfer Separately)

These files are in `.gitignore` and must be transferred via USB/Google Drive:

| File                               | Size        | Purpose                                          |
| ---------------------------------- | ----------- | ------------------------------------------------ |
| `ml/model/checkpoints/vox.pth.tar` | 335 MB      | Pretrained TPS model weights (VoxCeleb)          |
| `.env`                             | small       | API keys and configuration                       |
| How2Sign keypoint archives         | multiple GB | Training data (OpenPose JSON files)              |
| `ml/venv/`                         | large       | Python virtual environment (recreate on Windows) |
| `frontend/node_modules/`           | large       | Node dependencies (run `npm install`)            |

---

## 12. Development History Summary

### Key Milestones (chronological)

1. **Project Setup** — Initial React + Flask app with video upload and Whisper transcription
2. **Sign Language Rendering** — Built GIF/SVG-based sign language player with word-level signs and fingerspelling fallback
3. **How2Sign Dataset** — Downloaded dataset CSVs and keypoint archives to Google Drive
4. **Colab Training Pipeline** — Created comprehensive training notebook with Drive persistence, data extraction, vocabulary building, dataset loaders
5. **TPS Motion Model** — Implemented full TPS model architecture (KPDetector, DenseMotion, InpaintingNetwork, AVDNetwork) based on CVPR 2022 paper
6. **Pretrained Demo** — Downloaded `vox.pth.tar` pretrained weights and created demo scripts showing the TPS model can generate video
7. **3D Avatar Exploration** — Explored Three.js + GLB models (Soldier.glb, human.glb, Thanh.glb) for alternative sign language visualization
8. **Report & Presentation** — Generated B.Tech project report and presentation slides
9. **Transfer to Windows GPU** — Current stage: moving project to Windows laptop with GPU for local training

### Current Status & Next Steps

- ✅ Frontend + Backend fully working (GIF-based sign language)
- ✅ ML architecture fully implemented
- ✅ Colab training notebook ready
- ✅ How2Sign dataset CSVs in repo
- 🔄 **NEXT**: Train the Text2Keypoints Transformer model on Windows GPU
- 🔄 **NEXT**: Fine-tune TPS model on sign language data
- 🔄 **NEXT**: Integrate trained models into the web app for AI-generated sign language output

---

## 13. Setup on Windows (Quick Start)

```bash
# 1. Clone the repo
git clone https://github.com/amitsah98450/Video2Sign.git
cd Video2Sign

# 2. Backend setup
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

# 3. Frontend setup (new terminal)
cd frontend
npm install
npm run dev

# 4. ML setup
cd ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 6. Copy vox.pth.tar → ml/model/checkpoints/ (via USB/Google Drive)
# 7. Create .env file with your API keys
```

---

## 14. Key Dependencies

### Backend (backend/requirements.txt)

```
flask==3.1.0
flask-cors==5.0.1
openai-whisper
torch
torchaudio
ffmpeg-python
gunicorn==23.0.0
```

### ML (ml/requirements.txt)

```
openai-whisper
torch
torchvision
torchaudio
numpy
matplotlib
scikit-learn
nltk
ffmpeg-python
pandas
```

### Frontend (package.json)

- React 19
- Vite 7
- ESLint

### System Requirements

- Python 3.9+
- Node.js 18+
- FFmpeg (required by Whisper)
- NVIDIA CUDA Toolkit + cuDNN (for GPU training)

---

## 15. References

- **OpenAI Whisper**: https://github.com/openai/whisper
- **How2Sign Dataset**: https://how2sign.github.io/
- **TPS Motion Model (CVPR 2022)**: "Thin-Plate Spline Motion Model for Image Animation" — https://arxiv.org/abs/2203.14367
- **CMU OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
