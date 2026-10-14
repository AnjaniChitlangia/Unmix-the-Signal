"""
audio_utils.py — loading, resampling, aligning, mixing and exporting audio.

Everything here is plain NumPy/SciPy. Audio is handled internally as float64 in
[-1, 1], shaped (channels, samples) whenever several signals are stacked.
"""
from __future__ import annotations

import io
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

DEFAULT_SR = 16_000
ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "data" / "sources"
LIBRARY_DIR = ROOT / "data" / "library"
SEPARATED_DIR = ROOT / "outputs" / "separated"

SILENCE_RMS = 1e-5  # below this a clip is treated as silent


class AudioError(ValueError):
    """Raised with a human-readable message when an audio file is unusable."""


# ---------------------------------------------------------------------------
# Loading & conversion
# ---------------------------------------------------------------------------


def _to_float(data: np.ndarray) -> np.ndarray:
    """Convert any WAV sample type to float64 in [-1, 1]."""
    if data.dtype == np.uint8:                      # 8-bit WAV is unsigned
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64) / float(np.iinfo(data.dtype).max + 1)
    return data.astype(np.float64)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average channels. Accepts (T,), (T, ch) as scipy returns, or (ch, T)."""
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        axis = 1 if data.shape[0] > data.shape[1] else 0
        return data.mean(axis=axis)
    raise AudioError(f"Unsupported audio array shape {data.shape}.")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased) by the exact rational factor sr_out/sr_in."""
    if sr_in == sr_out:
        return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)


def load_audio(source, target_sr: int = DEFAULT_SR, name: str | None = None) -> np.ndarray:
    """
    Load a WAV file (path, bytes or file-like) -> mono float64 at target_sr.
    Raises AudioError with an informative message on bad/silent input.
    """
    label = name or (os.path.basename(source) if isinstance(source, (str, Path)) else "uploaded file")
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        sr, data = wavfile.read(source)
    except Exception as exc:  # scipy raises several types for malformed files
        raise AudioError(f"Could not read '{label}' as a WAV file ({exc}). Please upload PCM/float WAV audio.") from exc
    if data.size == 0:
        raise AudioError(f"'{label}' contains no samples.")
    x = to_mono(_to_float(data))
    x = resample(x, int(sr), target_sr)
    x = np.nan_to_num(x)
    if rms(x) < SILENCE_RMS:
        raise AudioError(f"'{label}' is silent (RMS ≈ 0). ICA needs signals with actual content.")
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def align_lengths(signals: list[np.ndarray], mode: str = "trim") -> np.ndarray:
    """Stack 1-D signals of different lengths: 'trim' to the shortest or 'pad' with zeros."""
    if not signals:
        raise AudioError("No signals given.")
    lengths = [len(s) for s in signals]
    if mode == "trim":
        n = min(lengths)
        return np.vstack([s[:n] for s in signals])
    n = max(lengths)
    return np.vstack([np.pad(s, (0, n - len(s))) for s in signals])


def standardize(S: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance rows (safe for silent rows)."""
    S = S - S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return S / np.where(sd < 1e-12, 1.0, sd)


def normalize_peak(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale so max |x| == peak; returns zeros (not NaN) for silent input."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 1e-12 else np.zeros_like(x)


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


def random_mixing_matrix(n_mics: int, n_sources: int, seed: int | None = None,
                         max_cond: float = 50.0) -> np.ndarray:
    """
    Random positive-ish mixing matrix (like microphone gains), redrawn until its
    condition number is below max_cond so the demo is not accidentally near-singular.
    """
    if n_mics < n_sources:
        raise ValueError("Need at least as many microphones (observations) as sources.")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        A = rng.uniform(0.2, 1.0, size=(n_mics, n_sources))
        # Give each mic a dominant source so the heat-map tells a physical story.
        for i in range(n_mics):
            A[i, i % n_sources] += rng.uniform(0.3, 0.8)
        if np.linalg.cond(A) < max_cond:
            return np.round(A, 3)
    return np.round(A, 3)


def mix(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Instantaneous linear mixture X = A S."""
    A = np.asarray(A, dtype=float)
    if A.shape[1] != S.shape[0]:
        raise ValueError(f"Mixing matrix has {A.shape[1]} columns but there are {S.shape[0]} sources.")
    return A @ S


def add_noise(X: np.ndarray, snr_db: float | None, seed: int | None = None) -> np.ndarray:
    """Add white Gaussian sensor noise at the given per-channel SNR (None/inf = clean)."""
    if snr_db is None or np.isinf(snr_db):
        return X
    rng = np.random.default_rng(seed)
    p_signal = np.mean(X ** 2, axis=1, keepdims=True)
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    return X + rng.standard_normal(X.shape) * np.sqrt(p_noise)


def align_by_xcorr(X: np.ndarray, sr: int, max_lag_s: float = 0.5) -> tuple[np.ndarray, list[int]]:
    """
    Real-world mode helper: recordings from different devices start at different
    times. Shift every channel to best match channel 0 (FFT cross-correlation).
    Returns (aligned & trimmed X, lags in samples).
    """
    ref = X[0] - X[0].mean()
    n = len(ref)
    max_lag = int(max_lag_s * sr)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    R = np.fft.rfft(ref, nfft)
    lags = [0]
    for ch in X[1:]:
        c = np.fft.irfft(np.fft.rfft(ch - ch.mean(), nfft) * np.conj(R), nfft)
        c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])  # lags -max..+max
        lags.append(int(np.argmax(np.abs(c)) - max_lag))
    # c[k] = Σ ch[t + k] ref[t]  -> channel is delayed by k samples, so use ch[t + k].
    # Keep only the window of t where every shifted channel has valid samples.
    lo, hi = max(0, -min(lags)), min(n, n - max(lags))
    out = np.vstack([ch[lo + lag: hi + lag] for ch, lag in zip(X, lags)])
    return out, lags


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def to_int16(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (normalize_peak(np.asarray(x, dtype=float), peak) * 32767).astype(np.int16)


def wav_bytes(x: np.ndarray, sr: int = DEFAULT_SR, peak: float = 0.9) -> bytes:
    """Peak-normalised 16-bit PCM WAV as bytes (for in-browser players/downloads)."""
    buf = io.BytesIO()
    wavfile.write(buf, sr, to_int16(x, peak))
    return buf.getvalue()


def save_wav(path, x: np.ndarray, sr: int = DEFAULT_SR, peak: float = 0.9) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, to_int16(x, peak))
    return path


def list_sources() -> dict[str, Path]:
    """Bundled clean sources, keyed by a friendly name."""
    return {p.stem.replace("_", " ").title(): p for p in sorted(SOURCES_DIR.glob("*.wav"))}


def load_sources(paths, sr: int = DEFAULT_SR, duration_s: float | None = None) -> np.ndarray:
    """Load several files -> stacked (n, T) array trimmed to a common length."""
    sigs = [load_audio(p, sr) for p in paths]
    S = align_lengths(sigs, "trim")
    if duration_s:
        S = S[:, : int(duration_s * sr)]
    return S
