# 🔮 Polyglot Ghost

**Real-time Voice Deepfake Detection** — An acoustic AI system that classifies audio as **REAL** or **FAKE** using machine learning.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blueviolet)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb)
![Backend](https://img.shields.io/badge/Backend-Node.js%20%2B%20Express-339933)
![AI](https://img.shields.io/badge/AI-Python%20%2B%20FastAPI-009688)

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Frontend   │────▶│   Backend    │────▶│  AI Inference    │
│  React/Vite  │◀────│  Express.js  │◀────│  FastAPI + SVM   │
│  Port 5173   │     │  Port 3001   │     │  Port 8000       │
└─────────────┘     └──────────────┘     └──────────────────┘
```

**Audio flow:** Frontend → Backend API → AI Inference Service → Backend → Frontend

---

## Quick Start

### Prerequisites

- **Node.js** 18+
- **Python** 3.11+
- **pip** (Python package manager)

### 1. AI Service

```bash
cd ai-service
pip install -r requirements.txt
python train_dummy_model.py    # Generate demo model
python app.py                  # Starts on port 8000
```

### 2. Backend

```bash
cd backend
npm install
npm start                      # Starts on port 3001
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                    # Starts on port 5173
```

Open **http://localhost:5173** in your browser.

---

## Docker

```bash
docker-compose up --build
```

All three services start automatically. Frontend at `http://localhost:5173`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload audio → get REAL/FAKE verdict |
| `POST` | `/liveness` | Semantic liveness verification |
| `GET` | `/health` | Backend health check |
| `POST` | `/predict` | Direct AI service prediction (port 8000) |

---

## Features

- 🎙 **Live recording** via browser microphone
- 📁 **File upload** — drag & drop .wav/.mp3
- 🧠 **SVM-based acoustic analysis** — MFCC, spectral centroid, ZCR, bandwidth
- 📊 **Confidence metrics** — circular gauge + feature breakdown
- 🔐 **Liveness verification** — optional semantic check via Groq AI
- 🎨 **Premium dark UI** — glassmorphism, animations, responsive

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3001` | Backend port |
| `AI_SERVICE_URL` | `http://localhost:8000` | AI service URL |
| `GROQ_API_URL` | — | Groq API URL (auto-configured) |
| `GROQ_API_KEY` | — | Groq API key for LLM reasoning |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model name |
| `VITE_API_URL` | — | Frontend → backend URL (for production) |

---

## ML Pipeline

1. Load audio at 16 kHz mono
2. Trim silence (top_db=20)
3. Enforce fixed duration (3 seconds)
4. Extract 30 acoustic features (MFCC mean/std, spectral centroid, bandwidth, ZCR, RMS)
5. Scale with StandardScaler
6. Classify with SVM (RBF kernel, probability=True)

---

## License

MIT
