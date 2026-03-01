# Polyglot Ghost: System Design & Architecture

This document outlines the complete, end-to-end system architecture of the Polyglot Ghost deepfake detection platform. The system is built using a modern decoupled, 3-tier hybrid architecture.

## 1. System Visualization (Architecture Diagram)

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef aiengine fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff;
    classDef external fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff;
    classDef client fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;

    %% Client Layer
    User((End User)):::client

    %% Frontend Layer (Vercel)
    subgraph Frontend [Presentation Layer (Vercel Edge CDN)]
        ReactUI[React.js Web App]:::frontend
        AudioCapture[WebAudio Microphone + File Drop]:::frontend
        ReportGen[jsPDF Report Generator]:::frontend
    end

    %% Backend Layer (Local / Cloud API)
    subgraph Gateway [API Gateway Layer (Node.js/Express)]
        ExpressAPI[Express Server :3001]:::backend
        Multer[Multer Memory Storage]:::backend
        RateLimiter[Rate Limiter & Security]:::backend
    end

    %% Inference Layer (Local / Cloud GPU)
    subgraph Inference [Inference Engine (Python/FastAPI)]
        FastAPI[FastAPI Server :8000]:::aiengine
        Librosa[Audio Decoder & Resampler]:::aiengine
        PyTorch[PyTorch wav2vec 2.0 Model]:::aiengine
        Bayesian[Bayesian Statistical Classifier]:::aiengine
    end

    %% External APIs
    FeatherlessAPI[Featherless AI LLM (Llama 3.1 8B)]:::external

    %% Data Flow
    User -->|Accesses globally| ReactUI
    User -->|Uploads MP3/WAV or Speaks| AudioCapture
    
    AudioCapture -->|POST /analyze (FormData)| ExpressAPI
    ExpressAPI -->|Validates & Buffers| Multer
    Multer -->|Proxies raw audio buffer| FastAPI
    
    FastAPI --> Librosa
    Librosa -->|16kHz Mono sine waves| PyTorch
    PyTorch -->|Extracts Tensors| Bayesian
    Bayesian -->|Sends Acoustic Anomalies| FeatherlessAPI
    
    FeatherlessAPI -->|Returns Forensic Reasoning| Bayesian
    Bayesian -->|Returns JSON Results| ExpressAPI
    ExpressAPI -->|Returns Verdict| ReactUI
    ReactUI -->|Generates Evidence| ReportGen
    ReportGen -->|Downloads PDF| User

```

---

## 2. Component Breakdown

The architecture is divided into three distinct layers to ensure scalability, security, and performance.

### A. The Presentation Layer (Frontend)
- **Technology:** React.js, Vite, TailwindCSS.
- **Hosting:** Deployed globally on the **Vercel CDN**.
- **Role:** Handles all user interactions, UI animations (glassmorphism), capturing microphone audio via WebAudio APIs, parsing dragged-and-dropped audio files, and generating downloadable PDF forensic reports on the client side using `jspdf`.
- **Why?** Deploying a static, compiled React app to Vercel ensures that users globally experience zero-latency loading times. The frontend does no heavy math; it only handles UI/UX.

### B. The Gateway Layer (Backend)
- **Technology:** Node.js, Express.js, Multer.
- **Hosting:** Local or lightweight cloud server (e.g., Render, AWS EC2 / Port 3001).
- **Role:** Acts as the API Gateway and traffic controller. It receives massive `FormData` audio uploads, validates file types, handles timeouts, and proxies the raw memory buffer directly to the Python inference engine.
- **Why?** A Node.js gateway handles hundreds of concurrent web requests far better than Python. It prevents the heavy PyTorch model from being overwhelmed by managing queues, rate limiting, and security before the AI ever touches the file.

### C. The Inference Layer (AI Engine)
- **Technology:** Python, FastAPI, PyTorch, HuggingFace (`wav2vec2`), `librosa`.
- **Hosting:** Local or dedicated GPU Instance (Port 8000).
- **Role:** This is the heavy computation engine. 
  1. **Preprocessing:** Uses `librosa` to universally decode any audio format (M4A, MP3, WAV) and resample it down to a standardized 16kHz mono waveform.
  2. **Feature Extraction:** Passes the raw waveform through the PyTorch `wav2vec 2.0` tensors to extract microscopic acoustic features (pitch std, spectral centroid, RMS dynamics).
  3. **Classification:** Uses Bayesian Gaussian probability distributions to score the likelihood of the audio being real vs. simulated.

### D. Explainable AI (External LLM)
- **Technology:** Featherless AI API (Llama 3.1 8B Instruct).
- **Role:** Once the Python engine generates the mathematical probabilities and anomaly metrics, it POSTs them to Featherless AI. The LLM translates the complex acoustic math into a human-readable forensic paragraph explaining *why* the fake was detected.

## 3. Privacy-by-Design
This architecture implements **Stateless Execution**. 
There is **no database** (PostgreSQL, MongoDB, etc.) attached to the core analysis pipeline. Audio files are streamed, analyzed purely in RAM, and immediately garbage-collected upon completion. This ensures zero data retention for highly sensitive biometric voice data.
