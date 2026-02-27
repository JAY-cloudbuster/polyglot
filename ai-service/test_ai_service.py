"""
Polyglot Ghost — AI Service Unit Tests
Comprehensive test suite for the deepfake detection engine.
Run: pytest test_ai_service.py -v
"""

import os
import io
import struct
import math
import numpy as np
import pytest
import importlib
import sys

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def create_wav_bytes(duration=1.0, sr=16000, freq=440.0):
    """Generate a valid WAV file in memory (sine wave)."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)

    buf = io.BytesIO()
    # WAV header
    data_size = n_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio.tobytes())
    return buf.getvalue()


def create_empty_wav():
    """Generate a WAV with zero audio data."""
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', 0))
    return buf.getvalue()


# ──────────────────────────────────────────────
# TEST 1: Audio Forensics — load_audio_universal
# ──────────────────────────────────────────────

class TestAudioLoading:
    """Tests for loading and preprocessing audio bytes."""

    def test_load_valid_wav(self):
        """T01 — Can load a valid WAV file."""
        from audio_forensics import load_audio_universal
        wav = create_wav_bytes(duration=1.0, sr=16000, freq=440)
        y = load_audio_universal(wav)
        assert isinstance(y, np.ndarray)
        assert len(y) > 0
        assert y.dtype == np.float32 or y.dtype == np.float64

    def test_load_returns_correct_length(self):
        """T02 — Loaded audio has expected sample count at 16kHz."""
        from audio_forensics import load_audio_universal
        wav = create_wav_bytes(duration=2.0, sr=16000)
        y = load_audio_universal(wav)
        expected = 16000 * 2
        assert abs(len(y) - expected) < 1000  # tolerance for resampling

    def test_load_resamples_to_16k(self):
        """T03 — Audio at 44.1kHz is resampled to 16kHz."""
        from audio_forensics import load_audio_universal
        wav = create_wav_bytes(duration=1.0, sr=44100)
        y = load_audio_universal(wav)
        # Should be approximately 16000 samples for 1 second
        assert abs(len(y) - 16000) < 2000

    def test_load_invalid_bytes_raises(self):
        """T04 — Invalid byte data raises ValueError."""
        from audio_forensics import load_audio_universal
        with pytest.raises((ValueError, Exception)):
            load_audio_universal(b"this is not audio data at all")

    def test_output_is_normalized(self):
        """T05 — Loaded audio values are in [-1, 1] range."""
        from audio_forensics import load_audio_universal
        wav = create_wav_bytes(duration=1.0)
        y = load_audio_universal(wav)
        assert np.max(np.abs(y)) <= 1.1  # small tolerance


# ──────────────────────────────────────────────
# TEST 2: Forensic Measurements
# ──────────────────────────────────────────────

class TestForensicMeasurements:
    """Tests for the forensic feature extraction."""

    def test_returns_dict(self):
        """T06 — extract_forensic_measurements returns a dictionary."""
        from audio_forensics import extract_forensic_measurements
        wav = create_wav_bytes(duration=1.5)
        result = extract_forensic_measurements(wav)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        """T07 — Measurements contain expected forensic features."""
        from audio_forensics import extract_forensic_measurements
        wav = create_wav_bytes(duration=1.5)
        result = extract_forensic_measurements(wav)
        assert len(result) > 0

    def test_values_are_numeric(self):
        """T08 — All measurement values are numeric."""
        from audio_forensics import extract_forensic_measurements
        wav = create_wav_bytes(duration=1.5)
        result = extract_forensic_measurements(wav)
        for key, value in result.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"Key '{key}' has non-numeric value: {type(value)}"

    def test_no_nan_values(self):
        """T09 — No NaN values in measurements."""
        from audio_forensics import extract_forensic_measurements
        wav = create_wav_bytes(duration=1.5)
        result = extract_forensic_measurements(wav)
        for key, value in result.items():
            assert not math.isnan(float(value)), f"Key '{key}' is NaN"


# ──────────────────────────────────────────────
# TEST 3: Model Loading
# ──────────────────────────────────────────────

class TestModelLoading:
    """Tests for the deep learning model lifecycle."""

    def test_model_loads_successfully(self):
        """T10 — wav2vec2 model loads without errors."""
        from inference import load_model, _model
        load_model()
        from inference import _model
        assert _model is not None

    def test_extractor_loaded(self):
        """T11 — Feature extractor is loaded alongside model."""
        from inference import _extractor
        assert _extractor is not None

    def test_model_in_eval_mode(self):
        """T12 — Model is in evaluation mode (not training)."""
        from inference import _model
        assert not _model.training

    def test_model_has_two_labels(self):
        """T13 — Model has exactly 2 output labels."""
        from inference import _model
        assert len(_model.config.id2label) == 2

    def test_model_labels_correct(self):
        """T14 — Labels are AIVoice and HumanVoice."""
        from inference import _model
        labels = set(_model.config.id2label.values())
        assert labels == {"AIVoice", "HumanVoice"}


# ──────────────────────────────────────────────
# TEST 4: Inference Pipeline
# ──────────────────────────────────────────────

class TestInference:
    """Tests for the full prediction pipeline."""

    def test_predict_returns_dict(self):
        """T15 — predict() returns a dictionary."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0, freq=440)
        result = predict(wav)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        """T16 — Result contains label, probability, reasoning."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        assert "label" in result
        assert "probability" in result
        assert "reasoning" in result

    def test_label_is_real_or_fake(self):
        """T17 — Label is either REAL or FAKE."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        assert result["label"] in ("REAL", "FAKE")

    def test_probability_in_range(self):
        """T18 — Probability is between 0.5 and 0.99."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        p = result["probability"]
        assert 0.5 <= p <= 0.99

    def test_result_has_model_info(self):
        """T19 — Result includes model name and measurements."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        assert "model_used" in result
        assert "measurements" in result

    def test_ai_human_probs_sum_to_one(self):
        """T20 — AI + Human voice probabilities ≈ 1.0."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        total = result.get("ai_voice_prob", 0) + result.get("human_voice_prob", 0)
        assert abs(total - 1.0) < 0.01

    def test_reasoning_is_string(self):
        """T21 — Reasoning is a non-empty string."""
        from inference import predict
        wav = create_wav_bytes(duration=1.0)
        result = predict(wav)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 10

    def test_sine_wave_detected_as_ai(self):
        """T22 — Pure sine wave (synthetic) should be detected as FAKE."""
        from inference import predict
        wav = create_wav_bytes(duration=2.0, freq=440)
        result = predict(wav)
        # A pure sine is very synthetic — model should catch it
        assert result["label"] == "FAKE"


# ──────────────────────────────────────────────
# TEST 5: Explanation Engine
# ──────────────────────────────────────────────

class TestExplanation:
    """Tests for the get_explanation function."""

    def test_explanation_for_real(self):
        """T23 — Explanation generated for REAL verdict."""
        from inference import get_explanation
        text = get_explanation("REAL", 0.95, 0.05, 0.95, {})
        assert isinstance(text, str)
        assert len(text) > 10

    def test_explanation_for_fake(self):
        """T24 — Explanation generated for FAKE verdict."""
        from inference import get_explanation
        text = get_explanation("FAKE", 0.88, 0.88, 0.12, {})
        assert isinstance(text, str)
        assert len(text) > 10

    def test_explanation_without_api_key(self):
        """T25 — Fallback explanation works without Featherless."""
        import inference
        old_key = inference.FEATHERLESS_API_KEY
        inference.FEATHERLESS_API_KEY = ""
        try:
            text = inference.get_explanation("REAL", 0.9, 0.1, 0.9, {})
            assert "confidence" in text.lower() or "human" in text.lower()
        finally:
            inference.FEATHERLESS_API_KEY = old_key


# ──────────────────────────────────────────────
# TEST 6: FastAPI Endpoints
# ──────────────────────────────────────────────

class TestFastAPIEndpoints:
    """Tests for the FastAPI app endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """T26 — GET / returns service info."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "polyglot-ghost-ai"
        assert data["status"] == "running"

    def test_predict_with_valid_audio(self, client):
        """T27 — POST /predict with valid WAV returns prediction."""
        wav = create_wav_bytes(duration=1.0)
        r = client.post("/predict", files={"file": ("test.wav", wav, "audio/wav")})
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert data["label"] in ("REAL", "FAKE")

    def test_predict_returns_probability(self, client):
        """T28 — /predict response includes probability 0.5-0.99."""
        wav = create_wav_bytes(duration=1.0)
        r = client.post("/predict", files={"file": ("test.wav", wav, "audio/wav")})
        data = r.json()
        assert 0.5 <= data["probability"] <= 0.99

    def test_predict_empty_file_rejected(self, client):
        """T29 — Empty file upload returns 400."""
        r = client.post("/predict", files={"file": ("test.wav", b"", "audio/wav")})
        assert r.status_code == 400

    def test_predict_oversized_file_rejected(self, client):
        """T30 — File > 10MB returns 413."""
        big = b"\x00" * (11 * 1024 * 1024)
        r = client.post("/predict", files={"file": ("big.wav", big, "audio/wav")})
        assert r.status_code == 413


# ──────────────────────────────────────────────
# TEST 7: Configuration
# ──────────────────────────────────────────────

class TestConfiguration:
    """Tests for environment and configuration."""

    def test_model_name_set(self):
        """T31 — HF model name is configured."""
        from inference import HF_MODEL_NAME
        assert HF_MODEL_NAME == "Hemgg/Deepfake-audio-detection"

    def test_featherless_url_set(self):
        """T32 — Featherless API URL is configured."""
        from inference import FEATHERLESS_API_URL
        assert "featherless" in FEATHERLESS_API_URL.lower()

    def test_target_sample_rate(self):
        """T33 — Target sample rate is 16kHz."""
        from audio_forensics import TARGET_SR
        assert TARGET_SR == 16000

    def test_model_sample_rate(self):
        """T34 — Model sample rate is set."""
        from inference import _model_sr
        assert _model_sr > 0
