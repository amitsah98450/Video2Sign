# 🤟 Video to Sign Language

> **B.Tech Final Year Project** — An AI-powered web application that converts spoken language in videos into animated sign language, bridging the communication gap for the deaf and hard-of-hearing community.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
- [Training on Google Colab](#training-on-google-colab)
- [Future Scope](#future-scope)

---

## Overview

This project provides an end-to-end pipeline that takes a video (or audio) file as input, transcribes the spoken content using **OpenAI Whisper**, and then translates the transcribed text into an animated **sign language** sequence. The system supports two rendering modes:

1. **GIF/SVG-based rendering** — Maps words to pre-built sign language GIF assets; unknown words are finger-spelled letter-by-letter using SVG hand assets.
2. **AI-generated photorealistic rendering** _(research module)_ — Uses a **Transformer encoder-decoder** to generate body/hand/face keypoint sequences from text, which are then fed into a **Thin-Plate Spline (TPS) Motion Model** to produce realistic animated human video frames.

---

## Key Features

| Feature                         | Description                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| 🎥 **Video/Audio Upload**       | Drag-and-drop upload supporting MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, OGG                      |
| 🗣️ **AI Transcription**         | Automatic speech-to-text using OpenAI Whisper (supports multiple languages with auto-translation) |
| 📝 **Timestamped Segments**     | Transcription output includes per-segment timestamps (start/end times)                            |
| 🤟 **Sign Language Conversion** | Text → sign language tokens with word-level signs and letter-level fingerspelling fallback        |
| ▶️ **Animated Playback**        | Interactive sign language player with play/pause, prev/next, speed control, and visual word queue |
| 🧠 **Deep Learning Pipeline**   | Transformer model to generate OpenPose keypoint sequences from English text                       |
| 🎬 **TPS Motion Model**         | Photorealistic video generation from keypoint sequences using Thin-Plate Spline image animation   |

---

## System Architecture

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

## Tech Stack

### Frontend

| Technology      | Purpose                                                  |
| --------------- | -------------------------------------------------------- |
| **React 19**    | Component-based UI framework                             |
| **Vite 7**      | Fast build tool and dev server                           |
| **Vanilla CSS** | Custom styling with glassmorphism, gradients, animations |

### Backend

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| **Flask 3.1**      | Lightweight Python REST API framework |
| **Flask-CORS**     | Cross-Origin Resource Sharing         |
| **OpenAI Whisper** | Speech-to-text transcription model    |
| **Gunicorn**       | Production WSGI server                |

### Machine Learning

| Technology                        | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| **PyTorch**                       | Deep learning framework                        |
| **Transformer (Encoder-Decoder)** | Text → keypoint sequence generation            |
| **TPS Motion Model**              | Photorealistic frame generation from keypoints |
| **OpenPose (How2Sign)**           | 137-keypoint body/face/hand pose format        |
| **VGG19**                         | Perceptual loss for training the TPS model     |
| **Google Colab**                  | Cloud GPU training environment                 |

---

## Project Structure

```
Video to Sign/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx              # Main app with view routing (upload → transcription → sign)
│   │   ├── api.js               # API client (uploadVideo, textToSign, healthCheck)
│   │   ├── components/
│   │   │   ├── Header.jsx       # Navigation header with back/reset controls
│   │   │   ├── VideoUpload.jsx  # Drag-and-drop video upload with progress tracking
│   │   │   ├── TranscriptionView.jsx  # Timestamped transcription display
│   │   │   └── SignLanguagePlayer.jsx # Animated sign playback with speed control
│   │   ├── index.css            # Global styles (glassmorphism, dark theme, animations)
│   │   └── main.jsx             # React entry point
│   ├── package.json
│   └── vite.config.js
│
├── backend/                     # Flask REST API
│   ├── app.py                   # Main Flask server (upload, transcribe, text-to-sign)
│   ├── sign_dictionary.py       # Word → sign asset mapping + fingerspelling fallback
│   ├── signs/                   # Sign language GIF/SVG assets
│   │   └── letters/             # A-Z SVG fingerspelling assets
│   ├── requirements.txt
│   └── uploads/                 # Temporary upload directory (auto-cleaned)
│
├── ml/                          # Machine Learning module
│   ├── colab_notebook.py        # Complete Colab training notebook (1000+ lines)
│   ├── data/                    # How2Sign CSV metadata (train/val/test splits)
│   ├── transcription/
│   │   └── transcribe.py        # Whisper transcription utility
│   ├── model/
│   │   ├── model.py             # TPSMotionModel (KPDetector + DenseMotion + Inpainting)
│   │   ├── train.py             # Training script (perceptual, equivariance, warp losses)
│   │   ├── inference.py         # TPSAnimator — generate video from keypoints
│   │   ├── dataset.py           # PyTorch dataset for TPS training
│   │   ├── dense_motion.py      # Dense motion estimation network
│   │   ├── inpainting_network.py # Inpainting generator network
│   │   ├── keypoint_detector.py # Keypoint detection network
│   │   ├── bg_motion_predictor.py # Background motion predictor
│   │   ├── avd_network.py       # AVD (Animate Via Disentangling) network
│   │   ├── config.yaml          # Model hyperparameters
│   │   └── util.py              # TPS transformations, coordinate grids, etc.
│   ├── evaluation/              # Evaluation metrics and results
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## ML Pipeline

### Stage 1: Text → Keypoint Generation (Transformer)

The **Text2KeypointsModel** is a Transformer encoder-decoder that converts tokenized English text into sequences of body keypoints:

- **Input**: Tokenized English sentence `(B, S)` with SOS/EOS tokens
- **Output**: Keypoint sequence `(B, T, 274)` — 137 keypoints × 2 (x, y)
- **Architecture**:
  - Embedding layer (vocab → d_model=256) + sinusoidal positional encoding
  - 4-layer Transformer encoder (8 heads, FFN=1024)
  - 4-layer Transformer decoder (8 heads, FFN=1024, teacher forcing during training)
  - Linear projection head → 274-dim keypoint output
  - Length predictor head → predicted sequence length
- **Training Loss**: MSE (keypoint reconstruction) + L1 (sequence length prediction)
- **Dataset**: [How2Sign](https://how2sign.github.io/) — 35K+ sentence-level English ↔ ASL pairs with CMU OpenPose keypoints (train: ~26K, val: ~2K, test: ~3K)
- **Keypoint Format**: 137 keypoints per frame = 25 body + 70 face + 21 left hand + 21 right hand

### Stage 2: Keypoint → Video Generation (TPS Motion Model)

The **TPSMotionModel** (based on [CVPR 2022](https://arxiv.org/abs/2203.14367)) transforms keypoint sequences into photorealistic video frames:

- **Input**: Source reference image `(1, 3, H, W)` + driving keypoints
- **Output**: Animated video frame `(1, 3, H, W)`
- **Sub-networks**:
  - **KPDetector**: Detects TPS control point keypoints from images
  - **BGMotionPredictor**: Estimates background affine transformation
  - **DenseMotionNetwork**: Computes dense optical flow + occlusion maps
  - **InpaintingNetwork**: Generates the final frame via warping + inpainting
  - **AVDNetwork**: Animate-via-Disentangling for cross-identity transfer
- **Training Losses**: Multi-scale VGG19 perceptual loss, equivariance loss, warp loss, background loss, TPS dropout scheduling

---

## API Endpoints

| Method | Endpoint                | Description                                                                           |
| ------ | ----------------------- | ------------------------------------------------------------------------------------- |
| `GET`  | `/api/health`           | Health check — returns `{ status: "ok" }`                                             |
| `POST` | `/api/upload`           | Upload video/audio → Whisper transcription (returns full text + timestamped segments) |
| `POST` | `/api/text-to-sign`     | Convert text → sign language tokens (word signs or fingerspelling)                    |
| `GET`  | `/api/signs/<filename>` | Serve sign language GIF/SVG assets                                                    |

### Example: Text to Sign

**Request:**

```json
POST /api/text-to-sign
{ "text": "hello how are you" }
```

**Response:**

```json
{
  "success": true,
  "original_text": "hello how are you",
  "signs": [
    {
      "type": "word",
      "text": "hello",
      "assets": ["hello.gif"],
      "has_asset": true
    },
    { "type": "word", "text": "how", "assets": ["how.gif"], "has_asset": true },
    { "type": "word", "text": "are", "assets": ["are.gif"], "has_asset": true },
    { "type": "word", "text": "you", "assets": ["you.gif"], "has_asset": true }
  ],
  "total_signs": 4
}
```

> **Note:** Common stop words (`a`, `an`, `the`, `of`, `to`, etc.) are automatically filtered out, as sign language omits these for conciseness.

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** and npm
- **FFmpeg** (required by Whisper for audio processing)

### 1. Clone the Repository

```bash
git clone https://github.com/amitsah98450/Video2Sign.git
cd Video2Sign
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The backend API will start on **http://localhost:5001**.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start on **http://localhost:5173**.

### 4. Open the Application

Navigate to **http://localhost:5173** in your browser. Upload a video, review the transcription, and watch it converted to sign language!

---

## Training on Google Colab

The ML training pipeline is designed for **Google Colab** with GPU support and Google Drive persistence:

### Quick Start

1. Upload `ml/colab_notebook.py` to Google Colab
2. Copy each `# %% CELL N` section into separate Colab cells
3. Mount Google Drive (Cell 1) — all data and checkpoints persist here
4. Download the [How2Sign dataset](https://how2sign.github.io/) to Drive
5. Run cells sequentially: Extract → Build Vocab → Create DataLoaders → Train

### Training Configuration

| Parameter              | Value                                               |
| ---------------------- | --------------------------------------------------- |
| Batch Size             | 16                                                  |
| Learning Rate          | 1e-4 (with ReduceLROnPlateau scheduler)             |
| Optimizer              | Adam                                                |
| Epochs                 | 100                                                 |
| Max Keypoint Frames    | 200                                                 |
| Max Text Length        | 50 tokens                                           |
| Model Dimension        | 256                                                 |
| Attention Heads        | 8                                                   |
| Encoder/Decoder Layers | 4 each                                              |
| Checkpointing          | Every 5 epochs + best model (saved to Google Drive) |

### Resuming After Disconnect

If Colab disconnects, run Cells 1–5 (mount Drive, copy data to local), then jump to **Cell 14** which automatically:

- Loads vocabulary from Drive
- Finds the latest checkpoint
- Resumes training from where it stopped

---

## Future Scope

- **3D Avatar Rendering** — Integrate a rigged 3D human model (e.g., Three.js + GLB) for more expressive sign language visualization
- **Real-time Processing** — Stream video/audio for live sign language translation
- **Indian Sign Language (ISL)** — Expand the sign dictionary and training data to support ISL
- **Bidirectional Translation** — Sign language video → text (sign recognition)
- **Mobile Application** — Cross-platform mobile app for on-the-go accessibility
- **Fine-tuned Whisper** — Domain-specific fine-tuning for better accuracy on educational/conversational content

---

## References

- **OpenAI Whisper**: [github.com/openai/whisper](https://github.com/openai/whisper)
- **How2Sign Dataset**: [how2sign.github.io](https://how2sign.github.io/)
- **TPS Motion Model (CVPR 2022)**: Zhao, J., Zhang, H. _"Thin-Plate Spline Motion Model for Image Animation"_ — [arxiv.org/abs/2203.14367](https://arxiv.org/abs/2203.14367)
- **CMU OpenPose**: [github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

---

## License

This project is developed as part of a **B.Tech Final Year Project** for academic purposes.

---

<p align="center">
  Built with ❤️ for accessibility
</p>
