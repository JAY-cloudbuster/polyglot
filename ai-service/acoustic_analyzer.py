"""
Acoustic Heuristic Analyzer v3 for Deepfake Voice Detection.

KEY INSIGHT: AI-generated speech is "too clean" compared to real recordings.

Real voice recordings contain:
  - Ambient room noise / noise floor
  - Breathing sounds between words
  - Microphone coloration and self-noise
  - Natural imperfections in spectral smoothness

AI-generated deepfakes (ElevenLabs, XTTS, TortoiseTTS, etc.) produce:
  - Unnaturally clean silence between words
  - Very high Signal-to-Noise Ratio (SNR)
  - Over-smooth spectral envelopes
  - Missing breathing artifacts
  - Periodic vocoder artifacts in the spectrogram

This analyzer detects these differences to classify REAL vs FAKE.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


def analyze_noise_floor(y: np.ndarray, sr: int) -> float:
    """
    Measure the noise floor level in the quietest portions of the audio.

    Real recordings: have ambient noise even in silent parts (SNR floor > -60dB).
    AI speech: has near-zero signal in silent parts (unnaturally clean).

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        # Compute RMS energy in short frames
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]

        if len(rms) < 10:
            return 0.5

        # Sort frames by energy to find the quietest 20%
        sorted_rms = np.sort(rms)
        quiet_portion = sorted_rms[:max(1, len(sorted_rms) // 5)]
        loud_portion = sorted_rms[-max(1, len(sorted_rms) // 5):]

        avg_quiet = np.mean(quiet_portion)
        avg_loud = np.mean(loud_portion)

        # Dynamic range ratio
        if avg_quiet < 1e-8:
            # Near-zero silence → very likely AI generated
            return 0.10

        dynamic_range_ratio = avg_loud / (avg_quiet + 1e-8)

        # Real recordings: ratio typically 10-500 (noise floor is audible)
        # AI speech: ratio > 1000 (silence is near-zero, speech is loud)
        if dynamic_range_ratio > 2000:
            return 0.08  # Extremely clean → almost certainly fake
        elif dynamic_range_ratio > 1000:
            return 0.15
        elif dynamic_range_ratio > 500:
            return 0.30
        elif dynamic_range_ratio > 200:
            return 0.55  # Borderline
        elif dynamic_range_ratio > 50:
            return 0.75  # Natural dynamic range
        else:
            return 0.85  # Lots of ambient noise → very likely real recording

    except Exception as e:
        logger.warning(f"Noise floor analysis failed: {e}")
        return 0.5


def analyze_silence_cleanliness(y: np.ndarray, sr: int) -> float:
    """
    Detect how "clean" the silence regions are.

    Real recordings: silence contains ambient noise, hum, breathing.
    AI speech: silence is digitally perfect (near-zero amplitude).

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        # Find silence regions using energy threshold
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        threshold = np.mean(rms) * 0.1  # 10% of mean energy

        silent_frames = rms[rms < threshold]

        if len(silent_frames) < 3:
            return 0.6  # Not enough silence to analyze

        # Measure the energy level in silent regions
        silence_energy = np.mean(silent_frames)
        silence_std = np.std(silent_frames)

        # Real silence has noise (energy > 1e-5) and variation (std > 0)
        # AI silence has near-zero energy and zero variation
        if silence_energy < 1e-6:
            return 0.10  # Digitally perfect silence → fake
        elif silence_energy < 1e-5:
            return 0.25
        elif silence_energy < 5e-5:
            return 0.45

        # Check variation in silence (real noise varies, digital silence doesn't)
        if silence_std < 1e-7:
            return 0.20  # No variation in silence → fake
        elif silence_std / (silence_energy + 1e-8) > 0.3:
            return 0.80  # Variable silence → real noise
        else:
            return 0.60

    except Exception as e:
        logger.warning(f"Silence analysis failed: {e}")
        return 0.5


def analyze_spectral_smoothness(y: np.ndarray, sr: int) -> float:
    """
    Measure how smooth the spectral envelope is.

    Real speech through a microphone: has micro-roughness from room acoustics.
    AI vocoder output: produces over-smooth spectral envelopes.

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        S = np.abs(librosa.stft(y, n_fft=2048))

        # Compute spectral difference between adjacent frequency bins
        # (roughness of the spectral envelope)
        spectral_roughness = np.mean(np.abs(np.diff(S, axis=0)))
        spectral_mean = np.mean(S) + 1e-8

        roughness_ratio = spectral_roughness / spectral_mean

        # Real speech: roughness_ratio typically 0.3 to 1.2
        # AI speech: rougher ratio often < 0.25 (over-smooth) or specific patterns
        if roughness_ratio > 0.5:
            return 0.80  # Good spectral roughness → likely real
        elif roughness_ratio > 0.35:
            return 0.65
        elif roughness_ratio > 0.25:
            return 0.45
        elif roughness_ratio > 0.15:
            return 0.25  # Too smooth → likely fake
        else:
            return 0.15  # Very smooth → almost certainly fake

    except Exception as e:
        logger.warning(f"Spectral smoothness analysis failed: {e}")
        return 0.5


def analyze_breathing_artifacts(y: np.ndarray, sr: int) -> float:
    """
    Detect breathing sounds between speech segments.

    Real speech: contains audible breaths (broad-spectrum noise bursts).
    AI speech: lacks breathing unless explicitly synthesized.

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        # Get spectral flatness over time
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        if len(flatness) < 5:
            return 0.5

        mean_rms = np.mean(rms)

        # Breathing sounds: moderate energy (10-50% of mean) + high spectral flatness
        breath_candidates = 0
        total_quiet_frames = 0

        for i in range(len(flatness)):
            if rms[i] < mean_rms * 0.5 and rms[i] > mean_rms * 0.05:
                total_quiet_frames += 1
                if flatness[i] > 0.1:  # Noise-like (flat spectrum = breathing)
                    breath_candidates += 1

        if total_quiet_frames < 2:
            return 0.5  # Not enough quiet frames

        breath_ratio = breath_candidates / total_quiet_frames

        # Real speech: 20-80% of quiet frames contain breathing-like noise
        # AI speech: < 10% (silence is clean, not breathy)
        if breath_ratio > 0.5:
            return 0.85  # Many breathing-like sounds → real
        elif breath_ratio > 0.3:
            return 0.75
        elif breath_ratio > 0.15:
            return 0.60
        elif breath_ratio > 0.05:
            return 0.35
        else:
            return 0.15  # No breathing detected → likely fake

    except Exception as e:
        logger.warning(f"Breathing analysis failed: {e}")
        return 0.5


def analyze_high_frequency_detail(y: np.ndarray, sr: int) -> float:
    """
    Analyze high-frequency content detail (>4kHz).

    Real speech: natural sibilants (s, sh, f) produce detailed HF content.
    AI vocoders: often produce smeared or missing HF detail.

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        hf_mask = freqs > 4000
        lf_mask = (freqs >= 200) & (freqs <= 2000)

        if not np.any(hf_mask) or not np.any(lf_mask):
            return 0.5

        hf_energy = np.mean(S[hf_mask, :] ** 2)
        lf_energy = np.mean(S[lf_mask, :] ** 2) + 1e-8

        hf_ratio = hf_energy / lf_energy

        # Also check HF temporal variation
        hf_frames = np.mean(S[hf_mask, :] ** 2, axis=0)
        hf_cv = np.std(hf_frames) / (np.mean(hf_frames) + 1e-8) if len(hf_frames) > 2 else 0

        score = 0.5

        # Natural HF ratio: 0.01 to 0.20
        if 0.005 <= hf_ratio <= 0.25:
            score += 0.15
        elif hf_ratio < 0.002:
            score -= 0.25  # Almost no HF → vocoder artifact

        # Natural HF variation (sibilants come and go)
        if hf_cv > 1.0:
            score += 0.20  # Good variation → real sibilants
        elif hf_cv > 0.5:
            score += 0.10
        elif hf_cv < 0.2:
            score -= 0.20  # Constant HF → artificial

        return max(0.05, min(0.95, score))

    except Exception as e:
        logger.warning(f"HF detail analysis failed: {e}")
        return 0.5


def analyze_temporal_micro_variation(y: np.ndarray, sr: int) -> float:
    """
    Detect micro-level temporal variations in the signal.

    Real speech: has natural micro-tremor and irregularity.
    AI speech: is unnaturally consistent at micro-timescales.

    Returns: realness score 0.0 (fake) to 1.0 (real)
    """
    try:
        # Compute MFCC delta-delta (acceleration of spectral change)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta2 = librosa.feature.delta(mfccs, order=2)

        # Mean absolute acceleration
        mean_accel = np.mean(np.abs(delta2))

        # Variation in acceleration
        accel_cv = np.std(np.abs(delta2)) / (mean_accel + 1e-8)

        # Real speech: higher, more varied acceleration
        # AI speech: smoother, more consistent acceleration
        score = 0.5

        if mean_accel > 1.5:
            score += 0.20  # Active micro-variation → real
        elif mean_accel > 0.8:
            score += 0.10
        elif mean_accel < 0.3:
            score -= 0.20  # Too smooth → fake

        if accel_cv > 1.5:
            score += 0.15  # Varied acceleration → real
        elif accel_cv > 1.0:
            score += 0.05
        elif accel_cv < 0.5:
            score -= 0.15  # Too consistent → fake

        return max(0.05, min(0.95, score))

    except Exception as e:
        logger.warning(f"Micro-variation analysis failed: {e}")
        return 0.5


def compute_realness_score(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Compute "realness" score using the AI-is-too-clean principle.

    Core idea: Real recordings are messy (noise, breathing, room acoustics).
    AI speech is unnaturally pristine.

    Returns:
        dict with per-indicator scores and final weighted score
    """
    indicators = {}

    # Primary indicators (strongest differentiation)
    indicators["noise_floor"] = round(analyze_noise_floor(y, sr), 4)
    indicators["silence_quality"] = round(analyze_silence_cleanliness(y, sr), 4)
    indicators["breathing_detect"] = round(analyze_breathing_artifacts(y, sr), 4)

    # Secondary indicators (supporting evidence)
    indicators["spectral_smoothness"] = round(analyze_spectral_smoothness(y, sr), 4)
    indicators["hf_detail"] = round(analyze_high_frequency_detail(y, sr), 4)
    indicators["micro_variation"] = round(analyze_temporal_micro_variation(y, sr), 4)

    # Weighted combination
    # Primary: noise floor, silence cleanliness, breathing (strongest signals)
    # Secondary: spectral smoothness, HF detail, micro-variation
    weights = {
        "noise_floor": 0.25,          # Strongest indicator
        "silence_quality": 0.20,      # AI silence is too clean
        "breathing_detect": 0.20,     # Breathing = human
        "spectral_smoothness": 0.15,  # Vocoders are too smooth
        "hf_detail": 0.10,            # HF differences
        "micro_variation": 0.10,      # Naturalness of articulation
    }

    weighted_score = sum(indicators[k] * weights[k] for k in indicators)
    indicators["overall_realness"] = round(weighted_score, 4)

    logger.info(f"Acoustic v3: {indicators}")
    return indicators
