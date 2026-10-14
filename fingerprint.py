"""
fingerprint.py — OPTIONAL EXTENSION: a mini "Shazam" (landmark audio fingerprinting).

This is *not* part of ICA. It shows a practical use of separation: a song that is
buried under speech and rain is hard to identify, but after ICA has pulled the
music component out, the fingerprint matcher recognises it.

Algorithm (Wang, 2003 — "An Industrial-Strength Audio Search Algorithm"):
1. Spectrogram (8 kHz, 1024-pt STFT, hop 256) in dB.
2. Constellation map: keep local maxima of the spectrogram (2-D maximum filter)
   that are well above the median level.
3. Landmarks: pair each anchor peak with a few later peaks in a target zone;
   hash = (f_anchor, f_target, Δt). Hashes are robust to noise and gain.
4. Matching: for every query hash found in the database, vote for
   (song, t_song - t_query). A true match produces many votes for ONE consistent
   time offset; random coincidences spread out. Score = height of the tallest bin.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft

from audio_utils import LIBRARY_DIR, load_audio

FP_SR = 8000
NPERSEG, HOP = 1024, 256
FAN_OUT, DT_MAX = 6, 48


def constellation(x: np.ndarray, sr: int = FP_SR):
    """Return spectrogram (dB) and peak coordinates (freq_bins, time_frames)."""
    _, _, Zxx = stft(x, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    S = 20 * np.log10(np.abs(Zxx) + 1e-9)
    local_max = maximum_filter(S, size=(21, 11)) == S
    thresh = np.median(S) + 15
    f_idx, t_idx = np.nonzero(local_max & (S > thresh))
    order = np.argsort(t_idx, kind="stable")
    return S, f_idx[order], t_idx[order]


def hashes(f_idx, t_idx):
    """Yield (hash, anchor_time) landmarks."""
    n = len(t_idx)
    for i in range(n):
        paired = 0
        for j in range(i + 1, n):
            dt = t_idx[j] - t_idx[i]
            if dt == 0:
                continue
            if dt > DT_MAX or paired >= FAN_OUT:
                break
            yield (int(f_idx[i]) << 20) | (int(f_idx[j]) << 8) | int(dt), int(t_idx[i])
            paired += 1


class FingerprintDB:
    def __init__(self):
        self.index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.titles: list[str] = []

    def add(self, title: str, x: np.ndarray):
        sid = len(self.titles)
        self.titles.append(title)
        _, f, t = constellation(x)
        for h, t0 in hashes(f, t):
            self.index[h].append((sid, t0))

    @classmethod
    def from_library(cls, folder: Path = LIBRARY_DIR) -> "FingerprintDB":
        db = cls()
        for p in sorted(Path(folder).glob("*.wav")):
            title = p.stem.split("_", 1)[1].replace("_", " ").title()
            db.add(title, load_audio(p, FP_SR))
        if not db.titles:
            raise FileNotFoundError(f"No .wav files in {folder}. Run `python generate_sources.py`.")
        return db

    def identify(self, x: np.ndarray, sr: int) -> list[dict]:
        """Rank library songs for query x. Returns list of {title, score, offset_s}."""
        from audio_utils import resample
        q = resample(np.asarray(x, float), sr, FP_SR)
        _, f, t = constellation(q)
        votes: dict[int, Counter] = defaultdict(Counter)
        n_hashes = 0
        for h, tq in hashes(f, t):
            n_hashes += 1
            for sid, ts in self.index.get(h, ()):
                votes[sid][ts - tq] += 1
        ranking = []
        for sid, title in enumerate(self.titles):
            c = votes.get(sid)
            score, off = (0, 0) if not c else max((v, k) for k, v in c.items())
            ranking.append({"title": title, "score": int(score),
                            "offset_s": off * HOP / FP_SR, "query_hashes": n_hashes})
        ranking.sort(key=lambda r: -r["score"])
        return ranking


def confidence(ranking: list[dict]) -> float:
    """Ratio-based confidence in [0, 1]: how much the winner beats the runner-up."""
    if not ranking or ranking[0]["score"] == 0:
        return 0.0
    a = ranking[0]["score"]
    b = ranking[1]["score"] if len(ranking) > 1 else 0
    return float(1 - (b + 1) / (a + 1))
