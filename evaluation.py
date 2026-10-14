"""
evaluation.py — matching recovered components to the true sources and scoring them.

ICA returns components in an arbitrary ORDER, with arbitrary SIGN and SCALE.
A fair evaluation therefore has to:

1. Permutation  — find which estimate belongs to which source. We build the
   |Pearson correlation| matrix between every (source, estimate) pair and solve
   the linear assignment problem (Hungarian algorithm, scipy's
   linear_sum_assignment) to maximise the total |correlation|. Greedy
   "take the best pair first" can double-assign; the assignment solver cannot.
2. Sign & scale — every metric below is invariant to a scalar gain (including -1):
   correlation by construction, and SDR because the target is the projection of
   the estimate onto the source (the optimal gain is solved for, not assumed).

Metrics
-------
|r|      : |Pearson correlation| — intuitive, 1 = perfect, but saturates near 1.
SI-SDR   : scale-invariant signal-to-distortion ratio (Le Roux et al., 2019), in dB.
           Decompose the estimate y = α s + e with α = <y,s>/<s,s>;
           SI-SDR = 10 log10(‖α s‖² / ‖e‖²). Every 10 dB = 10x less distortion energy.
SIR/SAR  : BSS-Eval-style split of the error into *interference* from the other
           known sources and *artefacts* (noise, nonlinear leftovers), with
           time-invariant (0-tap) gains — appropriate because our mixing is
           instantaneous.
Amari    : normalised Amari index of the global system P = W_total A (needs the true
           A). 0 = P is a scaled permutation (perfect separation), 1 = worst.

Limitations: all of these need the clean ground-truth sources (or A). For real
recordings there is no ground truth — we can then only report independence
proxies such as negentropy/kurtosis, plus listening tests.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _zm(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def corr_matrix(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of S (sources) and every row of Y (estimates)."""
    S, Y = _zm(np.atleast_2d(S)), _zm(np.atleast_2d(Y))
    ns = np.linalg.norm(S, axis=1, keepdims=True)
    ny = np.linalg.norm(Y, axis=1, keepdims=True)
    return (S @ Y.T) / np.maximum(ns * ny.T, 1e-12)


def match_components(S: np.ndarray, Y: np.ndarray):
    """
    Hungarian matching. Returns (perm, signs, C) where estimate perm[i] is the match
    for source i, signs[i] = sign of their correlation, C = full correlation matrix.
    """
    C = corr_matrix(S, Y)
    rows, cols = linear_sum_assignment(-np.abs(C))
    perm = np.full(S.shape[0], -1, dtype=int)
    perm[rows] = cols
    signs = np.sign(C[np.arange(len(perm)), perm])
    signs[signs == 0] = 1
    return perm, signs, C


def si_sdr(s: np.ndarray, y: np.ndarray) -> float:
    """Scale-invariant SDR in dB of estimate y against reference s."""
    s, y = _zm(s), _zm(y)
    alpha = (y @ s) / max(s @ s, 1e-12)
    target = alpha * s
    err = y - target
    return float(10 * np.log10(max(target @ target, 1e-20) / max(err @ err, 1e-20)))


def bss_eval_instantaneous(S: np.ndarray, y: np.ndarray, j: int) -> dict:
    """SDR/SIR/SAR of estimate y for source j given all true sources S (0-tap BSS Eval)."""
    S, y = _zm(S), _zm(y)
    sj = S[j]
    target = (y @ sj) / max(sj @ sj, 1e-12) * sj
    # Least-squares projection of y on the span of ALL sources.
    coef, *_ = np.linalg.lstsq(S.T, y, rcond=None)
    p_all = S.T @ coef
    e_interf = p_all - target
    e_artif = y - p_all

    def db(num, den):
        return float(10 * np.log10(max(num, 1e-20) / max(den, 1e-20)))

    e = e_interf + e_artif
    return {
        "sdr": db(target @ target, e @ e),
        "sir": db(target @ target, e_interf @ e_interf),
        # With a noise-free instantaneous mixture there are no artefacts at all; the ratio is then
        # just floating-point noise (~250 dB), so report None ("no artefacts") instead of a fake number.
        "sar": None if e_artif @ e_artif < 1e-12 * (target @ target) else
        db((target + e_interf) @ (target + e_interf), e_artif @ e_artif),
    }


def amari_index(W_total: np.ndarray, A: np.ndarray) -> float:
    """Normalised Amari performance index of P = W_total A, in [0, 1]."""
    P = np.abs(W_total @ A)
    n = P.shape[0]
    if P.shape[0] != P.shape[1] or n < 2:
        return float("nan")
    r = (P / P.max(axis=1, keepdims=True)).sum(axis=1) - 1
    c = (P / P.max(axis=0, keepdims=True)).sum(axis=0) - 1
    return float((r.sum() + c.sum()) / (2 * n * (n - 1)))


def align_estimates(S: np.ndarray, Y: np.ndarray, perm, signs=None) -> np.ndarray:
    """Reorder estimates to source order and apply the least-squares gain (fixes sign & scale)."""
    out = np.zeros_like(S, dtype=float)
    for i, k in enumerate(perm):
        y = _zm(Y[k])
        s = _zm(S[i])
        out[i] = (y @ s) / max(y @ y, 1e-12) * y
    return out


def evaluate(S: np.ndarray, Y: np.ndarray, X: np.ndarray | None = None,
             W_total: np.ndarray | None = None, A: np.ndarray | None = None,
             names: list[str] | None = None) -> dict:
    """
    Full report comparing estimates Y (k x T) against true sources S (n x T).
    If the observed mixtures X are given, an input baseline is reported: the SI-SDR
    of the *best single microphone* for each source, so improvement = output - input.
    """
    n = S.shape[0]
    names = names or [f"Source {i + 1}" for i in range(n)]
    perm, signs, C = match_components(S, Y)
    rows = []
    for i in range(n):
        y = Y[perm[i]]
        b = bss_eval_instantaneous(S, y, i)
        row = {
            "source": names[i],
            "matched_component": int(perm[i]) + 1,
            "sign": int(signs[i]),
            "abs_corr": float(abs(C[i, perm[i]])),
            "si_sdr": si_sdr(S[i], y),
            "sir": b["sir"],
            "sar": b["sar"],
        }
        if X is not None:
            row["input_si_sdr"] = max(si_sdr(S[i], x) for x in X)
            row["si_sdr_improvement"] = row["si_sdr"] - row["input_si_sdr"]
        rows.append(row)
    summary = {
        "mean_abs_corr": float(np.mean([r["abs_corr"] for r in rows])),
        "mean_si_sdr": float(np.mean([r["si_sdr"] for r in rows])),
        "mean_sir": float(np.mean([r["sir"] for r in rows])),
    }
    if X is not None:
        summary["mean_input_si_sdr"] = float(np.mean([r["input_si_sdr"] for r in rows]))
        summary["mean_si_sdr_improvement"] = float(np.mean([r["si_sdr_improvement"] for r in rows]))
    if W_total is not None and A is not None:
        summary["amari_index"] = amari_index(W_total, A)
    return {"rows": rows, "summary": summary, "perm": perm.tolist(), "signs": signs.tolist(),
            "corr_matrix": C.tolist(), "aligned": align_estimates(S, Y, perm)}


def quality_label(si_sdr_db: float) -> tuple[str, str]:
    """Plain-language label for an SI-SDR value (rule of thumb, not a standard)."""
    if si_sdr_db >= 20:
        return "Excellent", "near-transparent; residual crosstalk is barely audible"
    if si_sdr_db >= 10:
        return "Good", "clearly separated; faint leakage audible in quiet passages"
    if si_sdr_db >= 3:
        return "Fair", "source dominates but other sources are clearly audible"
    return "Poor", "separation failed or is heavily corrupted"
