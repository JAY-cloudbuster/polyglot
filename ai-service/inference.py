"""
Polyglot Ghost — Deep Learning Inference Engine.

Uses Hemgg/Deepfake-audio-detection (wav2vec2-based) for accurate
deepfake detection. This model was trained on actual real/fake audio
and processes raw waveforms — no manual feature engineering needed.

Architecture:
  1. wav2vec2 deep learning model → REAL/FAKE (99% of the work)
  2. Featherless LLM → human-readable explanation
"""

import os
import logging
import requests
import numpy as np
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from audio_forensics import load_audio_universal, extract_forensic_measurements

logger = logging.getLogger(__name__)

# Model config
HF_MODEL_NAME = "Hemgg/Deepfake-audio-detection"

# Globals
_model = None
_extractor = None
_model_sr = 16000

# Featherless (explanation only)
FEATHERLESS_API_URL = os.environ.get(
    "FEATHERLESS_API_URL",
    "https://api.featherless.ai/v1/chat/completions"
)
FEATHERLESS_API_KEY = os.environ.get(
    "FEATHERLESS_API_KEY",
    "rc_de0493447d51123bf5b5f761807dbdc9edca1c2fdc2cd4bcc0a4301083c70d86"
)
FEATHERLESS_MODEL = os.environ.get(
    "FEATHERLESS_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)


def load_model():
    """Download and load the wav2vec2 deepfake detection model."""
    global _model, _extractor, _model_sr

    logger.info(f"Loading deep learning model: {HF_MODEL_NAME}")
    logger.info("This downloads ~360MB on first run, cached after that.")

    _model = AutoModelForAudioClassification.from_pretrained(HF_MODEL_NAME)
    _extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_NAME)
    _model.eval()

    _model_sr = _extractor.sampling_rate

    logger.info("=== Deep Learning Engine Ready ===")
    logger.info(f"  Model: {HF_MODEL_NAME}")
    logger.info(f"  Labels: {_model.config.id2label}")
    logger.info(f"  Sampling rate: {_model_sr}")
    logger.info(f"  Featherless: explanation only")


def predict(audio_bytes: bytes) -> dict:
    """
    Deep learning inference pipeline.

    1. Load and preprocess audio (any format)
    2. Run wav2vec2 model → AIVoice / HumanVoice probability
    3. Map to REAL/FAKE label
    4. Get Featherless explanation
    """
    global _model, _extractor, _model_sr

    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    logger.info("=== DEEP LEARNING ANALYSIS ===")

    # Step 1: Load audio
    logger.info("Step 1: Loading audio...")
    y = load_audio_universal(audio_bytes)
    logger.info(f"  Audio: {len(y)} samples, {len(y)/16000:.1f}s")

    # Resample to model's expected rate if needed
    if _model_sr != 16000:
        y = librosa.resample(y, orig_sr=16000, target_sr=_model_sr)
        logger.info(f"  Resampled to {_model_sr} Hz")

    # Step 2: Run wav2vec2 model
    logger.info("Step 2: Running wav2vec2 inference...")
    inputs = _extractor(y, sampling_rate=_model_sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_class = torch.argmax(probs).item()
    pred_label = _model.config.id2label[pred_class]

    # Log all class probabilities
    for i, p in enumerate(probs.tolist()):
        logger.info(f"  {_model.config.id2label[i]}: {p:.4f}")

    # Map to our label system
    # Model labels: {0: 'AIVoice', 1: 'HumanVoice'}
    ai_prob = float(probs[0])    # probability of AIVoice
    human_prob = float(probs[1]) # probability of HumanVoice

    if human_prob > ai_prob:
        label = "REAL"
        confidence = human_prob
    else:
        label = "FAKE"
        confidence = ai_prob

    confidence = max(0.50, min(0.99, confidence))

    logger.info(f"  Verdict: {label} ({confidence:.2%})")

    # Step 3: Get forensic measurements for explanation context
    logger.info("Step 3: Getting explanation...")
    measurements = {}
    try:
        measurements = extract_forensic_measurements(audio_bytes)
    except Exception:
        pass

    # Step 4: Get Featherless explanation
    reasoning = get_explanation(label, confidence, ai_prob, human_prob, measurements)

    result = {
        "label": label,
        "probability": round(confidence, 4),
        "reasoning": reasoning,
        "ai_voice_prob": round(ai_prob, 4),
        "human_voice_prob": round(human_prob, 4),
        "model_used": HF_MODEL_NAME,
        "measurements": measurements,
        "total_features": len(measurements),
    }

    logger.info(f"=== FINAL: {label} ({confidence:.2%}) ===")
    return result


def get_explanation(label, confidence, ai_prob, human_prob, measurements):
    """Ask Featherless to explain the verdict."""
    if not FEATHERLESS_API_KEY:
        if label == "REAL":
            return f"Deep learning analysis indicates genuine human speech with {confidence:.0%} confidence. The wav2vec2 model detected natural vocal patterns."
        else:
            return f"Deep learning analysis detected AI-generated speech artifacts with {confidence:.0%} confidence. The wav2vec2 model found synthetic voice patterns."

    m = measurements
    prompt = f"""A state-of-the-art wav2vec2 deep learning model analyzed an audio sample for deepfake detection.

Result: {label} ({confidence:.0%} confidence)
AI voice probability: {ai_prob:.2%}
Human voice probability: {human_prob:.2%}

Audio characteristics: pitch_std={m.get('pitch_std_hz', 'N/A')}Hz, spectral_centroid_std={m.get('spectral_centroid_std', 'N/A')}, dynamic_range={m.get('rms_dynamic_range', 'N/A')}

Write 2 concise sentences explaining why this audio was classified as {label}. Reference specific audio characteristics. No labels or JSON — just the explanation."""

    try:
        r = requests.post(
            FEATHERLESS_API_URL,
            headers={
                "Authorization": f"Bearer {FEATHERLESS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": FEATHERLESS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.2,
            },
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().strip('"')
    except Exception as e:
        logger.warning(f"Explanation failed: {e}")

    if label == "REAL":
        return f"Deep learning analysis indicates genuine human speech with {confidence:.0%} confidence."
    else:
        return f"Deep learning analysis detected AI-generated speech patterns with {confidence:.0%} confidence."
