"""
Feature extraction for Polyglot Ghost deepfake detection.

EXACT COPY of the Kaggle training notebook feature extraction.
DO NOT MODIFY — must match training-time features byte-for-byte.

Feature breakdown (123 total):
  - MFCC (n_mfcc=20): mean(20) + std(20) = 40
  - Delta MFCC: mean(20) + std(20) = 40
  - Delta-Delta MFCC: mean(20) + std(20) = 40
  - Spectral Centroid: mean = 1
  - Spectral Bandwidth: mean = 1
  - Zero Crossing Rate: mean = 1
  Total = 40 + 40 + 40 + 3 = 123
"""

import io
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

TARGET_SR = 16000
TARGET_DURATION = 3
TARGET_LENGTH = TARGET_SR * TARGET_DURATION


def extract_features_from_array(y, sr=16000):
    """
    Extract 123 features from audio array.
    EXACT replica of Kaggle notebook logic.
    """
    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad or truncate to fixed length
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    else:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta MFCC
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Mean and std
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)

    # Spectral features
    spectral = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        delta2_mean,
        delta2_std,
        np.mean(spectral),
        np.mean(bandwidth),
        np.mean(zcr)
    ])

    return features


def load_audio(audio_bytes: bytes) -> np.ndarray:
    """Load audio from bytes, resample to 16 kHz mono.

    Strategy:
      1. Try librosa directly (handles WAV, MP3, FLAC natively)
      2. Fallback: call ffmpeg directly via subprocess to convert webm/ogg → WAV
         (bypasses pydub which requires ffprobe — not bundled with imageio_ffmpeg)
    """
    # Strategy 1: Try librosa directly (works for WAV, MP3, FLAC without ffmpeg)
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=TARGET_SR, mono=True)
        if len(y) > 0:
            logger.info(f"Loaded audio via librosa: {len(y)} samples at {sr} Hz")
            return y
    except Exception as e:
        logger.warning(f"Librosa direct load failed: {e}")

    # Strategy 2: Use ffmpeg directly via subprocess (no ffprobe needed)
    try:
        import imageio_ffmpeg
        import subprocess
        import tempfile
        import os

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"Using ffmpeg at: {ffmpeg_path}")

        # Write input bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        # Output WAV temp file
        tmp_out_path = tmp_in_path.replace('.webm', '.wav')

        try:
            # Convert to 16kHz mono WAV using ffmpeg directly
            cmd = [
                ffmpeg_path,
                '-y',                  # overwrite output
                '-i', tmp_in_path,     # input file
                '-ar', str(TARGET_SR), # sample rate 16000
                '-ac', '1',            # mono
                '-sample_fmt', 's16',  # 16-bit
                '-f', 'wav',           # output format
                tmp_out_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')[-500:]}")
                raise RuntimeError("ffmpeg conversion failed")

            # Load the converted WAV
            y, sr = librosa.load(tmp_out_path, sr=TARGET_SR, mono=True)
            logger.info(f"Loaded audio via ffmpeg conversion: {len(y)} samples at {sr} Hz")
            return y
        finally:
            # Clean up temp files
            for path in [tmp_in_path, tmp_out_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"ffmpeg conversion also failed: {e}")

    raise ValueError("Could not decode audio file with any method")


def process_audio(audio_bytes: bytes) -> np.ndarray:
    """Full pipeline: load audio bytes → extract 123 features."""
    y = load_audio(audio_bytes)
    features = extract_features_from_array(y, sr=TARGET_SR)

    # Shape contract enforcement
    assert features.shape == (123,), f"Feature shape mismatch: {features.shape}, expected (123,)"

    logger.info(f"Extracted {len(features)} features successfully")
    return features


def get_feature_names() -> list:
    """Return human-readable feature names for all 123 features."""
    names = []
    for i in range(20):
        names.append(f"mfcc_{i+1}_mean")
    for i in range(20):
        names.append(f"mfcc_{i+1}_std")
    for i in range(20):
        names.append(f"delta_mfcc_{i+1}_mean")
    for i in range(20):
        names.append(f"delta_mfcc_{i+1}_std")
    for i in range(20):
        names.append(f"delta2_mfcc_{i+1}_mean")
    for i in range(20):
        names.append(f"delta2_mfcc_{i+1}_std")
    names.extend(["spectral_centroid_mean", "spectral_bandwidth_mean", "zcr_mean"])
    return names
