"""Unit tests for the mathematical pipeline.  Run:  python -m pytest -q"""
import io

import numpy as np
import pytest
from scipy.io import wavfile

import audio_utils as au
import evaluation as ev
from ica import excess_kurtosis, fastica, run_ica, run_pca
from whitening import center, pca_whiten, whiteness_error

T = 20_000


def synthetic_sources(n=3, seed=0):
    """Known independent, non-Gaussian sources: sine, square-ish sawtooth, Laplacian noise, uniform noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    bank = [np.sin(2 * np.pi * 13 * t), np.sign(np.sin(2 * np.pi * 7 * t + 0.3)),
            rng.laplace(size=T), rng.uniform(-1, 1, T), ((t * 23) % 1) - 0.5]
    return au.standardize(np.vstack(bank[:n]))


# --------------------------------------------------------------------------- whitening

def test_center_removes_mean():
    X = np.random.default_rng(1).normal(5, 2, (3, 1000))
    Xc, mean = center(X)
    assert np.allclose(Xc.mean(axis=1), 0)
    assert mean.shape == (3, 1)


def test_whitening_gives_identity_covariance():
    S = synthetic_sources(4)
    A = np.random.default_rng(2).normal(size=(4, 4))
    wr = pca_whiten(A @ S + 3.0)
    assert whiteness_error(wr.Z) < 1e-10
    assert np.allclose(wr.K @ wr.K_inv, np.eye(4), atol=1e-10)


def test_whitening_reduces_overdetermined_mixture():
    S = synthetic_sources(3)
    A = np.random.default_rng(3).normal(size=(6, 3))  # 6 mics, 3 sources
    wr = pca_whiten(A @ S, n_components=3)
    assert wr.Z.shape == (3, T)
    assert whiteness_error(wr.Z) < 1e-10
    assert wr.eigvals[3:].max() < 1e-8 * wr.eigvals[0]


def test_whitening_rejects_rank_deficient_request():
    S = synthetic_sources(2)
    X = np.vstack([S, S[0] + S[1]])  # third mic is a linear combination -> rank 2
    with pytest.raises(ValueError, match="rank"):
        pca_whiten(X, n_components=3)


def test_whitening_rejects_silence():
    with pytest.raises(ValueError, match="silent"):
        pca_whiten(np.zeros((2, 100)))


# --------------------------------------------------------------------------- ICA

@pytest.mark.parametrize("contrast", ["logcosh", "exp", "cube"])
@pytest.mark.parametrize("algorithm", ["symmetric", "deflation"])
def test_ica_recovers_known_sources(contrast, algorithm):
    S = synthetic_sources(3)
    A = np.array([[1.0, 0.5, 0.3], [0.4, 1.0, 0.6], [0.7, 0.2, 1.0]])
    r = run_ica(A @ S, contrast=contrast, algorithm=algorithm, seed=1)
    assert r.converged
    res = ev.evaluate(S, r.sources, W_total=r.unmixing, A=A)
    assert min(x["abs_corr"] for x in res["rows"]) > 0.99
    assert res["summary"]["amari_index"] < 0.02


def test_unmixing_is_orthogonal_in_whitened_space():
    S = synthetic_sources(4)
    A = np.random.default_rng(4).normal(size=(4, 4))
    r = run_ica(A @ S)
    assert np.allclose(r.W @ r.W.T, np.eye(4), atol=1e-8)
    # global system W_total A must be a scaled permutation
    P = np.abs(r.unmixing @ A)
    assert np.all(P.max(axis=1) / P.sum(axis=1) > 0.97)


def test_estimated_mixing_reconstructs_observations():
    S = synthetic_sources(3)
    A = np.random.default_rng(5).normal(size=(3, 3))
    X = A @ S
    r = run_ica(X)
    Xc, _ = center(X)
    assert np.allclose(r.mixing @ r.sources, Xc, atol=1e-8)


def test_ica_is_deterministic_with_seed():
    S = synthetic_sources(3)
    X = np.random.default_rng(6).normal(size=(3, 3)) @ S
    assert np.allclose(run_ica(X, seed=7).sources, run_ica(X, seed=7).sources)


def test_fastica_rejects_bad_arguments():
    Z = pca_whiten(synthetic_sources(2)).Z
    with pytest.raises(ValueError):
        fastica(Z, contrast="nope")
    with pytest.raises(ValueError):
        fastica(Z, algorithm="nope")


def test_pca_alone_does_not_separate():
    S = synthetic_sources(3)
    A = np.array([[1.0, 0.8, 0.6], [0.7, 1.0, 0.8], [0.6, 0.7, 1.0]])
    res = ev.evaluate(S, run_pca(A @ S))
    assert res["summary"]["mean_abs_corr"] < 0.95  # uncorrelated ≠ independent


def test_kurtosis_of_gaussian_is_near_zero():
    g = np.random.default_rng(0).standard_normal((1, 200_000))
    assert abs(excess_kurtosis(g)[0]) < 0.05
    assert excess_kurtosis(np.random.default_rng(0).laplace(size=(1, 200_000)))[0] > 2.5


# --------------------------------------------------------------------------- evaluation

def test_matching_handles_permutation_sign_and_scale():
    S = synthetic_sources(4)
    perm = [2, 0, 3, 1]
    gains = np.array([-3.0, 0.5, 2.0, -0.1])
    Y = gains[:, None] * S[perm]
    p, signs, _ = ev.match_components(S, Y)
    for i in range(4):
        assert perm[p[i]] == i
    res = ev.evaluate(S, Y)
    assert all(x["si_sdr"] > 100 for x in res["rows"])
    assert np.allclose(res["aligned"], S, atol=1e-8)


def test_si_sdr_is_scale_and_sign_invariant():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    y = s + 0.1 * rng.normal(size=5000)
    assert ev.si_sdr(s, y) == pytest.approx(ev.si_sdr(s, -7.5 * y), abs=1e-9)
    assert ev.si_sdr(s, y) == pytest.approx(20.0, abs=0.5)  # noise 0.1 -> ≈ 20 dB


def test_amari_index_bounds():
    A = np.random.default_rng(0).normal(size=(3, 3))
    assert ev.amari_index(np.linalg.inv(A), A) < 1e-10
    perm_scaled = np.diag([2, -1, 0.5]) @ np.eye(3)[[1, 2, 0]] @ np.linalg.inv(A)
    assert ev.amari_index(perm_scaled, A) < 1e-10
    assert ev.amari_index(np.ones((3, 3)) @ np.linalg.inv(A) + 0, A) > 0.9


# --------------------------------------------------------------------------- audio utils

def _wav(data, sr):
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def test_load_audio_converts_int16_stereo_and_resamples():
    sr = 44_100
    t = np.arange(sr) / sr
    stereo = (np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 16000).astype(np.int16)
    x = au.load_audio(_wav(stereo, sr), 16_000)
    assert x.ndim == 1 and abs(len(x) - 16_000) <= 1
    assert 0.4 < np.max(np.abs(x)) < 0.55


def test_load_audio_rejects_silence_and_garbage():
    with pytest.raises(au.AudioError, match="silent"):
        au.load_audio(_wav(np.zeros(8000, np.int16), 8000))
    with pytest.raises(au.AudioError, match="WAV"):
        au.load_audio(b"definitely not a wav file")


def test_align_lengths_trim_and_pad():
    sigs = [np.ones(10), np.ones(7), np.ones(12)]
    assert au.align_lengths(sigs, "trim").shape == (3, 7)
    assert au.align_lengths(sigs, "pad").shape == (3, 12)


def test_normalize_peak_is_safe_on_silence():
    assert np.all(au.normalize_peak(np.zeros(100)) == 0)
    assert np.max(np.abs(au.normalize_peak(np.array([0.1, -0.2])))) == pytest.approx(0.9)


def test_exported_wav_is_valid(tmp_path):
    x = np.random.default_rng(0).normal(size=16_000) * 5  # out of range on purpose
    path = au.save_wav(tmp_path / "out.wav", x, 16_000)
    sr, data = wavfile.read(path)
    assert sr == 16_000 and data.dtype == np.int16 and len(data) == 16_000
    assert np.max(np.abs(data)) <= 32767
    sr2, data2 = wavfile.read(io.BytesIO(au.wav_bytes(np.zeros(100))))  # silent export must not crash
    assert np.all(data2 == 0)


def test_add_noise_hits_requested_snr():
    X = np.random.default_rng(0).normal(size=(2, 200_000))
    Xn = au.add_noise(X, 10.0, seed=1)
    snr = 10 * np.log10(np.mean(X ** 2, axis=1) / np.mean((Xn - X) ** 2, axis=1))
    assert np.allclose(snr, 10.0, atol=0.1)
    assert au.add_noise(X, None) is X


def test_random_mixing_matrix_is_well_conditioned():
    A = au.random_mixing_matrix(4, 3, seed=0)
    assert A.shape == (4, 3) and np.linalg.cond(A) < 50
    with pytest.raises(ValueError):
        au.random_mixing_matrix(2, 3)


def test_xcorr_alignment_recovers_device_offset():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20_000)
    X = np.vstack([base[1000:15000], base[1000 - 237:15000 - 237]])  # ch1 lags by 237 samples
    aligned, lags = au.align_by_xcorr(X, 16_000)
    assert lags == [0, 237]
    assert np.corrcoef(aligned)[0, 1] > 0.999


# --------------------------------------------------------------------------- end to end on bundled audio

def test_end_to_end_on_bundled_sources():
    files = [au.SOURCES_DIR / f for f in ("speech_female.wav", "music_neon_drift.wav", "ambient_rain.wav")]
    S = au.standardize(au.load_sources(files, 16_000, duration_s=4))
    A = np.array([[1.0, 0.6, 0.45], [0.55, 1.0, 0.5], [0.4, 0.65, 1.0]])
    r = run_ica(A @ S)
    res = ev.evaluate(S, r.sources, X=A @ S)
    assert res["summary"]["mean_si_sdr"] > 25
    assert res["summary"]["mean_si_sdr_improvement"] > 20


def test_fingerprint_identifies_clean_snippet():
    import fingerprint as fp
    db = fp.FingerprintDB.from_library()
    x = au.load_audio(au.LIBRARY_DIR / "03_midnight_arcade.wav", 16_000)[16_000 * 4: 16_000 * 9]
    ranking = db.identify(x, 16_000)
    assert ranking[0]["title"] == "Midnight Arcade"
    assert ranking[0]["offset_s"] == pytest.approx(4.0, abs=0.1)
