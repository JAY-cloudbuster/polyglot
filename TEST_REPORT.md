# Polyglot Ghost — Unit Test Report

> **Date:** February 28, 2026  
> **Result: ✅ 34/34 PASSED (100%)**  
> **Runtime:** 68.02 seconds  
> **Framework:** pytest 8.x + FastAPI TestClient

---

## Summary

| Metric | Value |
|---|---|
| Total Tests | 34 |
| Passed | 34 |
| Failed | 0 |
| Pass Rate | **100%** |
| Duration | 68.02s |

---

## Test Categories

### 1. Audio Loading (5/5 ✅)

| ID | Test | Status |
|---|---|---|
| T01 | Load valid WAV file | ✅ |
| T02 | Verify audio sample count at 16kHz | ✅ |
| T03 | Auto-resample from 44.1kHz → 16kHz | ✅ |
| T04 | Reject invalid byte data | ✅ |
| T05 | Output values normalized [-1, 1] | ✅ |

### 2. Forensic Measurements (4/4 ✅)

| ID | Test | Status |
|---|---|---|
| T06 | Returns dictionary | ✅ |
| T07 | Contains forensic feature keys | ✅ |
| T08 | All values are numeric | ✅ |
| T09 | No NaN values | ✅ |

### 3. Model Loading (5/5 ✅)

| ID | Test | Status |
|---|---|---|
| T10 | wav2vec2 model loads successfully | ✅ |
| T11 | Feature extractor loaded alongside model | ✅ |
| T12 | Model is in evaluation mode (not training) | ✅ |
| T13 | Model has exactly 2 output labels | ✅ |
| T14 | Labels are AIVoice and HumanVoice | ✅ |

### 4. Inference Pipeline (8/8 ✅)

| ID | Test | Status |
|---|---|---|
| T15 | predict() returns dict | ✅ |
| T16 | Result has label, probability, reasoning | ✅ |
| T17 | Label is REAL or FAKE | ✅ |
| T18 | Probability in [0.5, 0.99] range | ✅ |
| T19 | Result includes model info and measurements | ✅ |
| T20 | AI + Human voice probs sum to ≈ 1.0 | ✅ |
| T21 | Reasoning is meaningful string (>10 chars) | ✅ |
| T22 | Synthetic sine wave detected as FAKE | ✅ |

### 5. Explanation Engine (3/3 ✅)

| ID | Test | Status |
|---|---|---|
| T23 | Explanation generated for REAL verdict | ✅ |
| T24 | Explanation generated for FAKE verdict | ✅ |
| T25 | Fallback explanation without Featherless API | ✅ |

### 6. FastAPI Endpoints (5/5 ✅)

| ID | Test | Status |
|---|---|---|
| T26 | GET / returns service health info | ✅ |
| T27 | POST /predict with valid WAV returns prediction | ✅ |
| T28 | Response probability in valid range | ✅ |
| T29 | Empty file upload → 400 Bad Request | ✅ |
| T30 | Oversized file (>10MB) → 413 Too Large | ✅ |

### 7. Configuration (4/4 ✅)

| ID | Test | Status |
|---|---|---|
| T31 | HuggingFace model name is configured | ✅ |
| T32 | Featherless API URL is configured | ✅ |
| T33 | Target sample rate = 16kHz | ✅ |
| T34 | Model sample rate is valid | ✅ |

---

## Architecture Coverage

```
Audio Input ──► load_audio_universal() ──► wav2vec2 Model ──► Label + Probability
                     │                                              │
                     ▼                                              ▼
              Forensic Measurements ──────────────────► Featherless LLM
                                                              │
                                                              ▼
                                                       Final Response
```

| Layer | Tests | What's Covered |
|---|---|---|
| Audio I/O | 5 | Loading, resampling, validation, normalization |
| Feature Extraction | 4 | Forensic measurements, types, completeness |
| Deep Learning Model | 5 | Loading, config, labels, eval mode |
| Inference Pipeline | 8 | End-to-end prediction, output validation |
| Explanation (LLM) | 3 | Generation, fallback, both verdicts |
| API Endpoints | 5 | Health, prediction, error handling |
| Configuration | 4 | Model params, API config, sample rates |

---

## How to Run

```bash
cd ai-service
python -m pytest test_ai_service.py -v
```

---

> Polyglot Ghost — Voice Deepfake Detection Platform
