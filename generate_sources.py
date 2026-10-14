"""
generate_sources.py — builds every bundled audio file from scratch, license-free.

    python generate_sources.py

Outputs
-------
data/sources/   clean sources for ICA (deliberately stored at DIFFERENT sample rates
                and channel counts so the loader's resampling / mono conversion is exercised)
    speech_female.wav   22.05 kHz mono   Windows SAPI voice "Zira" (rendered from data/raw)
    speech_male.wav     22.05 kHz mono   Windows SAPI voice "David"
    music_neon_drift.wav 44.1 kHz stereo procedural synth-pop (Karplus–Strong plucks, bass, drums)
    ambient_rain.wav    48 kHz  stereo   procedural rain: filtered noise, droplets, wind, thunder
    ambient_birds.wav   16 kHz  mono     procedural bird chirps (FM sweeps)
data/library/   6 procedural "songs" (16 kHz mono, 16 s) for the mini-Shazam extension.
                Song #1 is the same composition as music_neon_drift.wav.

The speech clips were rendered with the text-to-speech voices that ship with Windows
(System.Speech, see README). If data/raw/ is missing (e.g. on Linux/macOS) the
committed data/sources/speech_*.wav files are simply kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt

from audio_utils import load_audio, normalize_peak

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "sources"
LIB = ROOT / "data" / "library"
RAW = ROOT / "data" / "raw"
DUR = 10.0


def write(path: Path, x: np.ndarray, sr: int):
    """x is (T,) or (T, 2). Peak-normalised 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (normalize_peak(x, 0.89) * 32767).astype(np.int16))
    print(f"  wrote {path.relative_to(ROOT)}  ({sr} Hz, {x.shape})")


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12)


def pluck(freq, dur, sr, decay=0.996, rng=None):
    """Karplus–Strong plucked string: y[n] = x[n] + d/2 (y[n-N] + y[n-N-1])."""
    rng = rng or np.random.default_rng(0)
    n = int(dur * sr)
    N = max(2, int(round(sr / freq)))
    x = np.zeros(n)
    x[:N] = rng.uniform(-1, 1, min(N, n))
    a = np.zeros(N + 2)
    a[0], a[N], a[N + 1] = 1.0, -0.5 * decay, -0.5 * decay
    return lfilter([1.0], a, x)


def tone(freq, dur, sr, kind="saw", attack=0.01, release=0.15):
    t = np.arange(int(dur * sr)) / sr
    if kind == "sine":
        y = np.sin(2 * np.pi * freq * t)
    elif kind == "square":
        y = sum(np.sin(2 * np.pi * freq * k * t) / k for k in (1, 3, 5, 7) if freq * k < sr / 2)
    else:  # band-limited saw
        y = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 9) if freq * k < sr / 2)
    env = np.minimum(1, t / attack) * np.minimum(1, np.maximum(0, (dur - t) / release))
    return y * env


def kick(sr):
    t = np.arange(int(0.25 * sr)) / sr
    f = 50 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 14)


def hat(sr, rng):
    n = int(0.05 * sr)
    sos = butter(2, 6000 / (sr / 2) if 6000 < sr / 2 else 0.9, "high", output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.012 * sr))


SONGS = [
    dict(name="Neon Drift", root=57, scale="minor", prog=[0, 5, 2, 6], bpm=112, lead="pluck", seed=1),
    dict(name="Copper Sky", root=60, scale="major", prog=[0, 4, 5, 3], bpm=96, lead="square", seed=2),
    dict(name="Midnight Arcade", root=62, scale="minor", prog=[0, 3, 4, 4], bpm=128, lead="saw", seed=3),
    dict(name="Paper Lanterns", root=55, scale="major", prog=[0, 5, 3, 4], bpm=84, lead="sine", seed=4),
    dict(name="Static Bloom", root=64, scale="minor", prog=[0, 6, 5, 4], bpm=140, lead="pluck", seed=5),
    dict(name="Glass Harbor", root=59, scale="major", prog=[3, 4, 0, 0], bpm=104, lead="square", seed=6),
]
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def render_song(spec: dict, sr: int, dur: float, stereo: bool = False) -> np.ndarray:
    rng = np.random.default_rng(spec["seed"])
    n = int(dur * sr)
    L, R = np.zeros(n + sr), np.zeros(n + sr)
    scale = SCALES[spec["scale"]]
    step = 60.0 / spec["bpm"] / 4  # sixteenth note
    deg = lambda d, octave=0: spec["root"] + 12 * (d // 7 + octave) + scale[d % 7]

    def add(sig, start, pan=0.5, gain=1.0):
        i = int(start * sr)
        j = min(len(L), i + len(sig))
        L[i:j] += sig[: j - i] * gain * (1 - pan)
        R[i:j] += sig[: j - i] * gain * pan

    melody_deg = 7
    t, bar = 0.0, 0
    while t < dur:
        chord = spec["prog"][bar % len(spec["prog"])]
        for s16 in range(16):
            ts = t + s16 * step
            if ts >= dur:
                break
            # drums
            if s16 % 4 == 0:
                add(kick(sr), ts, 0.5, 0.9)
            if s16 % 2 == 1:
                add(hat(sr, rng), ts, 0.65, 0.25)
            # bass on 8ths
            if s16 % 2 == 0:
                add(tone(midi_hz(deg(chord, -2)), step * 1.8, sr, "sine"), ts, 0.5, 0.55)
            # arpeggio
            arp = [chord, chord + 2, chord + 4, chord + 7][s16 % 4]
            add(pluck(midi_hz(deg(arp)), step * 3, sr, 0.994, rng), ts, 0.3, 0.35)
            # lead melody: random walk on the scale, mostly on chord tones
            if s16 % 4 == 0 or rng.random() < 0.25:
                melody_deg += int(rng.choice([-2, -1, 0, 1, 2]))
                melody_deg = int(np.clip(melody_deg, 3, 13))
                f = midi_hz(deg(melody_deg, 1))
                note_len = step * rng.choice([1, 2, 3])
                sig = pluck(f, note_len + 0.3, sr, 0.997, rng) if spec["lead"] == "pluck" \
                    else tone(f, note_len, sr, spec["lead"]) * 0.6
                add(sig, ts, 0.7, 0.45)
        t += 16 * step
        bar += 1
    out = np.stack([L[:n], R[:n]], axis=1)
    return out if stereo else out.mean(axis=1)


# ---------------------------------------------------------------------------
# Ambient
# ---------------------------------------------------------------------------

def rain(sr: int, dur: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    # Bed: band-passed noise whose loudness swells with a slow "wind" LFO.
    sos = butter(2, [400 / (sr / 2), 5000 / (sr / 2)], "band", output="sos")
    bed = sosfilt(sos, rng.standard_normal((2, n)), axis=1)
    wind = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * t + 1.0)) ** 2
    bed *= wind * 0.25
    # Droplets: Poisson-timed, heavy-tailed amplitudes -> strongly non-Gaussian.
    drops = np.zeros((2, n))
    k = rng.poisson(45 * dur)
    pos = rng.integers(0, n - 2000, k)
    amp = rng.pareto(2.5, k) * 0.15
    dl = int(0.006 * sr)
    shape = np.sin(2 * np.pi * rng.uniform(2500, 6000) * np.arange(dl) / sr) * np.exp(-np.arange(dl) / (0.0015 * sr))
    for p, a in zip(pos, amp):
        ch = rng.integers(0, 2)
        drops[ch, p:p + dl] += a * shape * rng.choice([-1, 1])
    # Distant thunder roll at ~5.5 s.
    sos_lp = butter(2, 120 / (sr / 2), "low", output="sos")
    rumble = sosfilt(sos_lp, rng.standard_normal(n)) * 6
    env = np.exp(-((t - 6.2) / 1.1) ** 2) * (t > 5.4)
    thunder = rumble * env
    x = bed + drops + thunder
    return x.T


def birds(sr: int, dur: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = np.zeros(n + sr)
    t0 = 0.2
    while t0 < dur:
        f0, f1 = rng.uniform(2200, 3800), rng.uniform(3000, 6500)
        for _ in range(rng.integers(2, 6)):  # a phrase of chirps
            d = rng.uniform(0.04, 0.12)
            tt = np.arange(int(d * sr)) / sr
            f = f0 + (f1 - f0) * (tt / d) ** rng.uniform(0.5, 2)
            chirp = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.sin(np.pi * tt / d) ** 2
            i = int(t0 * sr)
            x[i:i + len(chirp)] += chirp * rng.uniform(0.4, 1)
            t0 += d + rng.uniform(0.03, 0.08)
        t0 += rng.uniform(0.25, 1.1)
    return x[:n]


def main():
    print("Generating UNMIX audio …")
    for raw, out in [("speech_female_raw.wav", "speech_female.wav"), ("speech_male_raw.wav", "speech_male.wav")]:
        if (RAW / raw).exists():
            x = load_audio(RAW / raw, 22050)[: int(DUR * 22050)]
            write(SRC / out, x, 22050)
        else:
            print(f"  (skipping {out}: {RAW / raw} not found, keeping committed file)")
    write(SRC / "music_neon_drift.wav", render_song(SONGS[0], 44100, DUR, stereo=True), 44100)
    write(SRC / "ambient_rain.wav", rain(48000, DUR), 48000)
    write(SRC / "ambient_birds.wav", birds(16000, DUR), 16000)
    for i, spec in enumerate(SONGS, 1):
        slug = spec["name"].lower().replace(" ", "_")
        write(LIB / f"{i:02d}_{slug}.wav", render_song(spec, 16000, 16.0), 16000)


if __name__ == "__main__":
    main()
