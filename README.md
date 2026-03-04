# 👻 Polyglot Ghost

**Real-time Voice Deepfake Detection** — A deep learning system that classifies audio as **REAL** or **FAKE** using wav2vec2 neural networks, Bayesian acoustic forensics, and Groq-powered LLM reasoning.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blueviolet)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb)
![AI](https://img.shields.io/badge/AI-wav2vec2%20%2B%20FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama3-ff6b35)
![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)
![Deploy](https://img.shields.io/badge/Deploy-Vercel%20%2B%20HF%20Spaces-black)

---

## 🏗️ Architecture

```
┌──────────────────┐          ┌──────────────────────────────┐
│    Frontend       │  HTTPS   │     AI Service (Cloud)       │
│  React + Vite     │─────────▶│  FastAPI + wav2vec2 + Groq   │
│  Vercel (CDN)     │◀─────────│  Hugging Face Spaces (24/7)  │
│  Port 5173 (dev)  │          │  Port 7860 (prod) / 8000 (dev)│
└──────────────────┘          └──────────────────────────────┘
```

**Simplified 2-tier architecture** — Frontend communicates directly with the AI service. No Node.js middleman needed.

---

## ✨ Features

### Core Detection
- 🧠 **wav2vec2 Deep Learning** — HuggingFace `Hemgg/Deepfake-audio-detection` model (400MB)
- 📊 **Bayesian Confidence Scoring** — Posterior probability with Gaussian smoothing
- 🔬 **Acoustic Forensics** — Pitch std, spectral centroid, dynamic range, HF/LF ratio, silence noise
- 💬 **Explainable AI** — Groq LLM (Llama 3 8B) generates human-readable forensic reasoning
- 🛡️ **Self-contained Fallback** — Smart local explanation engine works offline without any API

### User Interface
- 🎙️ **Live recording** via browser microphone
- 📁 **File upload** — drag & drop .wav/.mp3/.webm/.ogg/.flac
- 📊 **Confidence metrics** — circular gauge + feature breakdown
- 🔐 **Liveness verification** — semantic check via Groq AI
- 📄 **PDF Report Generation** — downloadable forensic evidence report
- 👻 **Cinematic Intro** — GSAP-powered 5-phase animation with circuit-board ghost SVG
- 🎨 **Premium Dark UI** — glassmorphism, CRT scan lines, micro-animations, responsive

### Engineering
- 🐳 **Docker-ready** — Dockerfile for cloud deployment
- ☁️ **Cloud Deployable** — Hugging Face Spaces (AI) + Vercel (Frontend)
- 🧪 **34 Unit Tests** — Full test coverage across all modules
- 🔒 **Privacy-by-Design** — Zero data retention, stateless processing

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (for frontend dev only)
- **pip** (Python package manager)

### 1. AI Service

```bash
cd ai-service
pip install -r requirements.txt
python app.py                  # Starts on port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                    # Starts on port 5173
```

Open **http://localhost:5173** in your browser.

> **Note:** The Node.js backend (`backend/`) is no longer required. The frontend communicates directly with the FastAPI AI service.

---

## 🐳 Docker

### Local Docker
```bash
docker-compose up --build
```

### Cloud Deployment (Hugging Face Spaces)
The `hf-space-deploy/` directory contains a ready-to-deploy Docker Space:

```bash
cd hf-space-deploy
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/polyglot-ghost
git push space main
```

Then set `VITE_API_URL` in Vercel to your Space URL.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload audio → get REAL/FAKE verdict with forensic reasoning |
| `GET` | `/health` | Service health check |
| `GET` | `/` | Service info and status |

### POST `/analyze`

**Request:** `multipart/form-data` with `audio` field

**Response:**
```json
{
    "verdict": "FAKE",
    "confidence": 0.87,
    "reasoning": "The wav2vec2 neural network detected synthetic speech patterns with 87% confidence. Key acoustic anomalies include unnaturally flat pitch variation (12.3 Hz std), compressed dynamic range (5.2x)...",
    "features_analyzed": 7,
    "feature_breakdown": {
        "pitch_std_hz": 12.3,
        "spectral_centroid_std": 450.2,
        "rms_dynamic_range": 5.2,
        "spectral_flatness_mean": 0.008,
        "silence_noise_level": 0.001,
        "hf_to_lf_ratio": 0.07,
        "zero_crossing_std": 0.12
    },
    "timestamp": "2026-03-03T18:30:00Z"
}
```

---

## 🧬 ML Pipeline

```
Audio Input (any format)
    │
    ▼
┌──────────────────────┐
│ 1. Audio Loading      │  librosa + soundfile
│    Resample → 16kHz   │  Mono channel
│    Normalize [-1, 1]  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. wav2vec2 Inference │  HuggingFace Transformers
│    Feature Extraction │  Wav2Vec2FeatureExtractor
│    Classification     │  Wav2Vec2ForSequenceClassification
│    → AIVoice/Human    │  2-class softmax
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Bayesian Scoring   │  Gaussian smoothing
│    Posterior prob.     │  Confidence calibration
│    → 0.50–0.99 range  │  Label mapping
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Acoustic Forensics │  Pitch, spectral, dynamic
│    7 forensic metrics  │  range, silence noise, etc.
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. LLM Explanation    │  Groq API (Llama 3 8B)
│    OR Local Fallback  │  Measurement-aware engine
│    → Forensic text    │  Works offline, forever
└──────────┬───────────┘
           ▼
       JSON Response
```

---

## 🎬 Cinematic Intro

A full-screen GSAP-powered intro sequence plays on first load:

| Phase | Duration | Effect |
|-------|----------|--------|
| 1. Ghost SVG | 1.5s | Circuit-board pixelated ghost fades in |
| 2. Ghost Out | 1.0s | Ghost fades to black |
| 3. Title In | 2.0s | "POLYGLOT GHOST" blur-to-focus + subtitle |
| 4. Title Out | 1.5s | Text fades out |
| 5. Reveal | 1.2s | Overlay dissolves → main site |

**Tech:** Inline SVG with gradient + glitch SVG filter, CRT scan lines, GSAP timeline()

---

## 📁 Project Structure

```
polyglot_ghost/
├── ai-service/                  # Python AI Engine
│   ├── app.py                   # FastAPI server (/analyze, /health)
│   ├── inference.py             # wav2vec2 model + Groq LLM + local fallback
│   ├── audio_forensics.py       # Acoustic feature extraction
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Docker container for cloud deploy
│   ├── test_ai_service.py       # 34 unit tests
│   └── .env                     # GROQ_API_KEY (gitignored)
│
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── App.jsx              # Root app with CinematicIntro
│   │   ├── components/
│   │   │   ├── CinematicIntro.jsx   # GSAP 5-phase intro animation
│   │   │   ├── AudioRecorder.jsx    # Browser mic recording
│   │   │   ├── FileUploader.jsx     # Drag & drop upload
│   │   │   ├── VerdictPanel.jsx     # REAL/FAKE result display
│   │   │   ├── ConfidenceMetrics.jsx # Circular gauge + breakdown
│   │   │   ├── LivenessPrompt.jsx   # Semantic liveness check
│   │   │   ├── Navbar.jsx           # Navigation bar
│   │   │   └── RhythmBackground.jsx # Animated background
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Home page
│   │   │   ├── RecordPage.jsx   # Live recording page
│   │   │   └── UploadPage.jsx   # File upload page
│   │   ├── services/
│   │   │   ├── api.js           # API client (configurable URL)
│   │   │   └── reportGenerator.js # PDF report export
│   │   └── index.css            # Design system + dark theme
│   └── package.json
│
├── backend/                     # Node.js Gateway (legacy, optional)
│   ├── server.js
│   ├── controllers/
│   └── routes/
│
├── hf-space-deploy/             # Ready-to-deploy HF Space
│   ├── README.md                # HF Space YAML frontmatter
│   ├── Dockerfile
│   ├── app.py
│   ├── inference.py
│   ├── audio_forensics.py
│   └── requirements.txt
│
├── SYSTEM_DESIGN.md             # Full system architecture docs
├── TEST_REPORT.md               # Test results (34/34 passing)
├── docker-compose.yml           # Multi-service Docker setup
├── vercel.json                  # Vercel frontend deployment config
└── .gitignore
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key for LLM reasoning (optional) |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1/chat/completions` | Groq API endpoint |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model name |
| `VITE_API_URL` | `http://localhost:8000` | Frontend → AI service URL |

> **Note:** If `GROQ_API_KEY` is not set, the system uses a built-in forensic explanation engine that generates detailed, measurement-aware reasoning locally — no API required.

---

## 🧪 Testing

```bash
cd ai-service
pytest test_ai_service.py -v
```

**34/34 tests** covering:
- Audio loading & preprocessing (5 tests)
- Forensic measurements (4 tests)
- Model loading & validation (5 tests)
- Inference pipeline (8 tests)
- Explanation engine (3 tests)
- FastAPI endpoints (5 tests)
- Configuration (4 tests)

---

## 🛡️ Privacy & Security

- **Zero data retention** — Audio is processed in-memory and never stored
- **Stateless architecture** — No database, no user tracking, no cookies
- **API key protection** — Groq key stored in `.env` (gitignored), never committed
- **CORS enabled** — Configurable origin restrictions

---

## 📚 Research Foundation

This system builds on the following research:

1. Baevski et al. (2020) — *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* (NeurIPS)
2. Yi et al. (2022) — *Audio Deepfake Detection Using wav2vec 2.0* (ASVspoof Challenge)
3. Tak et al. (2021) — *End-to-End Anti-Spoofing with RawNet2* (ICASSP)

---

## 🚢 Deployment

| Component | Platform | Cost | Status |
|-----------|----------|------|--------|
| Frontend | Vercel | Free | ✅ Live |
| AI Service | Hugging Face Spaces | Free (CPU) | 🔧 Deploy via `hf-space-deploy/` |

---

## License

MIT
