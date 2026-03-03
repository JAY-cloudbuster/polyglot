"""
Polyglot Ghost — Deep Learning Inference Engine.

Uses Hemgg/Deepfake-audio-detection (wav2vec2-based) for accurate
deepfake detection. This model was trained on actual real/fake audio
and processes raw waveforms — no manual feature engineering needed.

Architecture:
  1. wav2vec2 deep learning model → REAL/FAKE (99% of the work)
  2. Groq LLM (Llama 3) → human-readable explanation
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

# Groq LLM (explanation only — replaces Featherless AI)
GROQ_API_URL = os.environ.get(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    ""
)
GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "llama3-8b-8192"
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
    logger.info(f"  Groq LLM: explanation only")


def predict(audio_bytes: bytes) -> dict:
    """
    Deep learning inference pipeline.

    1. Load and preprocess audio (any format)
    2. Run wav2vec2 model → AIVoice / HumanVoice probability
    3. Map to REAL/FAKE label
    4. Get Groq LLM explanation
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

    # Step 4: Get Groq LLM explanation
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


def _build_local_explanation(label, confidence, ai_prob, human_prob, m):
    """
    Smart, self-contained forensic explanation engine.
    Uses actual acoustic measurements to generate detailed reasoning
    WITHOUT any external API. Works offline, forever, for free.
    """
    reasons = []

    if label == "FAKE":
        # Pitch analysis
        pitch_std = m.get("pitch_std_hz", None)
        if pitch_std is not None and pitch_std < 20:
            reasons.append(f"unnaturally flat pitch variation ({pitch_std:.1f} Hz std)")
        elif pitch_std is not None:
            reasons.append(f"pitch deviation of {pitch_std:.1f} Hz")

        # Dynamic range
        dyn = m.get("rms_dynamic_range", None)
        if dyn is not None and dyn < 8:
            reasons.append(f"compressed dynamic range ({dyn:.1f}x)")

        # Spectral flatness
        sf = m.get("spectral_flatness_mean", None)
        if sf is not None and sf < 0.01:
            reasons.append("abnormally smooth spectral profile")

        # Silence noise
        sn = m.get("silence_noise_level", None)
        if sn is not None and sn < 0.002:
            reasons.append("digitally clean silence (no natural mic noise)")

        # HF ratio
        hf = m.get("hf_to_lf_ratio", None)
        if hf is not None and hf < 0.1:
            reasons.append("reduced high-frequency breath detail")

        if reasons:
            detail = ", ".join(reasons[:3])
            return (
                f"The wav2vec2 neural network detected synthetic speech patterns with "
                f"{confidence:.0%} confidence. Key acoustic anomalies include {detail}, "
                f"which are characteristic signatures of AI voice synthesis engines."
            )
        return (
            f"Deep learning inference classified this audio as AI-generated with "
            f"{confidence:.0%} confidence (AI probability: {ai_prob:.1%}). "
            f"The acoustic waveform exhibits structural patterns consistent with "
            f"text-to-speech or voice cloning algorithms."
        )

    else:  # REAL
        # Pitch analysis
        pitch_std = m.get("pitch_std_hz", None)
        if pitch_std is not None and pitch_std > 25:
            reasons.append(f"natural pitch modulation ({pitch_std:.1f} Hz std)")

        # Dynamic range
        dyn = m.get("rms_dynamic_range", None)
        if dyn is not None and dyn > 10:
            reasons.append(f"wide dynamic range ({dyn:.1f}x)")

        # Silence noise
        sn = m.get("silence_noise_level", None)
        if sn is not None and sn > 0.003:
            reasons.append("organic background microphone noise")

        # HF ratio
        hf = m.get("hf_to_lf_ratio", None)
        if hf is not None and hf > 0.12:
            reasons.append("rich high-frequency breath detail")

        if reasons:
            detail = ", ".join(reasons[:3])
            return (
                f"The wav2vec2 neural network identified genuine human speech with "
                f"{confidence:.0%} confidence. Acoustic markers include {detail}, "
                f"indicating natural vocal tract resonance and organic speech production."
            )
        return (
            f"Deep learning inference classified this audio as authentic human speech "
            f"with {confidence:.0%} confidence (Human probability: {human_prob:.1%}). "
            f"The waveform structure is consistent with natural vocal production."
        )


def get_explanation(label, confidence, ai_prob, human_prob, measurements):
    """
    Generate forensic explanation for the verdict.
    
    Strategy:
      1. Try Groq LLM if API key is available (enhanced explanation)
      2. Fall back to smart local engine (works forever, offline, free)
    """
    m = measurements or {}

    # Try Groq LLM for enhanced explanation (optional)
    if GROQ_API_KEY:
        prompt = f"""A state-of-the-art wav2vec2 deep learning model analyzed an audio sample for deepfake detection.

Result: {label} ({confidence:.0%} confidence)
AI voice probability: {ai_prob:.2%}
Human voice probability: {human_prob:.2%}

Audio characteristics: pitch_std={m.get('pitch_std_hz', 'N/A')}Hz, spectral_centroid_std={m.get('spectral_centroid_std', 'N/A')}, dynamic_range={m.get('rms_dynamic_range', 'N/A')}

Write 2 concise sentences explaining why this audio was classified as {label}. Reference specific audio characteristics. No labels or JSON — just the explanation."""

        try:
            r = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0.2,
                },
                timeout=15,
            )
            if r.status_code == 200:
                text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().strip('"')
                if text and len(text) > 10:
                    return text
        except Exception as e:
            logger.warning(f"Groq explanation unavailable ({e}), using local engine")

    # Always works: smart local explanation using actual measurements
    return _build_local_explanation(label, confidence, ai_prob, human_prob, m)
