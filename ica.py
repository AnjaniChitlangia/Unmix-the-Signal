"""
ica.py — FastICA implemented from scratch (NumPy only, no sklearn).

Idea
----
After whitening (see whitening.py) we look for an orthogonal matrix W such that the
rows of  Y = W Z  are as *non-Gaussian* as possible. By the Central Limit Theorem a
sum of independent non-Gaussian variables is "more Gaussian" than its parts, so the
directions of maximal non-Gaussianity are exactly the directions that pick out a
single source.

Non-Gaussianity is measured by an approximation of negentropy

    J(y) ∝ ( E[G(y)] - E[G(ν)] )^2 ,   ν ~ N(0, 1)

with a smooth contrast G. Maximising it under ||w|| = 1 with a Newton step gives
the famous fixed-point iteration (Hyvärinen & Oja, 1997/2000):

    w  <-  E[ z g(wᵀz) ] - E[ g'(wᵀz) ] w ,      then  w <- w / ||w||

where g = G'. Contrasts provided:

    'logcosh'  G(u) = log cosh(a u)/a     g(u) = tanh(a u)        (robust default)
    'exp'      G(u) = -exp(-u²/2)         g(u) = u exp(-u²/2)     (super-Gaussian, robust)
    'cube'     G(u) = u⁴/4                g(u) = u³               (kurtosis; fast but outlier-sensitive)

Two strategies to extract several components:

    'symmetric' (parallel) — update all rows at once, then re-orthogonalise with
                 W <- (W Wᵀ)^{-1/2} W  so no component is favoured.
    'deflation' — extract one component at a time and Gram–Schmidt it against
                 the ones already found.

Ambiguities that no ICA method can resolve: the *order* of the recovered
components (permutation) and their *scale/sign*. evaluation.py handles both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whitening import WhiteningResult, pca_whiten

# ---------------------------------------------------------------------------
# Contrast functions: each returns (g(u), mean over samples of g'(u))
# ---------------------------------------------------------------------------


def _logcosh(u: np.ndarray, alpha: float = 1.0):
    t = np.tanh(alpha * u)
    return t, (alpha * (1.0 - t * t)).mean(axis=-1)


def _exp(u: np.ndarray, alpha: float = 1.0):
    e = np.exp(-0.5 * u * u)
    return u * e, ((1.0 - u * u) * e).mean(axis=-1)


def _cube(u: np.ndarray, alpha: float = 1.0):
    return u ** 3, (3.0 * u * u).mean(axis=-1)


CONTRASTS = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelate(W: np.ndarray) -> np.ndarray:
    """W <- (W Wᵀ)^{-1/2} W  — the closest orthogonal matrix to W (symmetric)."""
    s, U = np.linalg.eigh(W @ W.T)
    s = np.clip(s, 1e-15, None)  # guard against a degenerate start
    return (U / np.sqrt(s)) @ U.T @ W


@dataclass
class ICAResult:
    sources: np.ndarray        # recovered components, (k, T), unit variance
    W: np.ndarray              # rotation in whitened space, (k, k)
    unmixing: np.ndarray       # full unmixing matrix, (k, m):  S_hat = unmixing (X - mean)
    mixing: np.ndarray         # estimated mixing matrix, (m, k): X - mean ≈ mixing S_hat
    whitening: WhiteningResult
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)  # convergence measure per iteration


def fastica(
    Z: np.ndarray,
    contrast: str = "logcosh",
    algorithm: str = "symmetric",
    max_iter: int = 400,
    tol: float = 1e-6,
    alpha: float = 1.0,
    seed: int | None = 0,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool, list[float]]:
    """
    Run FastICA on *already whitened* data Z (k x T).
    Returns (W, n_iter, converged, history) with rows of W Z = independent components.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"Unknown contrast '{contrast}'. Choose from {list(CONTRASTS)}.")
    if algorithm not in ("symmetric", "deflation"):
        raise ValueError("algorithm must be 'symmetric' or 'deflation'.")
    g = CONTRASTS[contrast]
    k, T = Z.shape
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((k, k)) if w_init is None else np.array(w_init, dtype=float)
    history: list[float] = []

    if algorithm == "symmetric":
        W = _sym_decorrelate(W0)
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            gwz, g_prime_mean = g(W @ Z, alpha)
            # Fixed-point update for every row simultaneously:
            #   W+ = E[g(WZ) Zᵀ] - diag(E[g'(WZ)]) W
            W_new = (gwz @ Z.T) / T - g_prime_mean[:, None] * W
            W_new = _sym_decorrelate(W_new)
            # Converged when every new row points along its old row (up to sign).
            delta = float(np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1.0)))
            history.append(delta)
            W = W_new
            if delta < tol:
                converged = True
                break
        return W, it, converged, history

    # Deflation: one component at a time.
    W = np.zeros((k, k))
    total_iter, all_converged = 0, True
    for p in range(k):
        w = W0[p] / np.linalg.norm(W0[p])
        ok = False
        for it in range(1, max_iter + 1):
            gwz, g_prime_mean = g(w @ Z, alpha)
            w_new = (Z * gwz).mean(axis=1) - g_prime_mean * w
            # Gram–Schmidt: remove projections on components already found,
            # otherwise every unit would converge to the same strongest source.
            w_new -= W[:p].T @ (W[:p] @ w_new)
            w_new /= np.linalg.norm(w_new)
            delta = float(abs(abs(w_new @ w) - 1.0))
            history.append(delta)
            w = w_new
            if delta < tol:
                ok = True
                break
        total_iter += it
        all_converged &= ok
        W[p] = w
    return W, total_iter, all_converged, history


def run_ica(
    X: np.ndarray,
    n_components: int | None = None,
    contrast: str = "logcosh",
    algorithm: str = "symmetric",
    max_iter: int = 400,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> ICAResult:
    """Full pipeline: center -> PCA-whiten -> FastICA. X is (m mics x T samples)."""
    wr = pca_whiten(X, n_components=n_components)
    W, n_iter, converged, history = fastica(
        wr.Z, contrast=contrast, algorithm=algorithm, max_iter=max_iter, tol=tol, seed=seed
    )
    unmixing = W @ wr.K                 # S_hat = W K (X - mean)
    mixing = wr.K_inv @ W.T             # W orthogonal -> its inverse is Wᵀ
    sources = W @ wr.Z
    return ICAResult(sources=sources, W=W, unmixing=unmixing, mixing=mixing,
                     whitening=wr, n_iter=n_iter, converged=converged, history=history)


def run_pca(X: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """PCA-only baseline: the whitened principal components (uncorrelated, not independent)."""
    return pca_whiten(X, n_components=n_components).Z


# ---------------------------------------------------------------------------
# Statistics used for explanation / diagnostics
# ---------------------------------------------------------------------------

_GAUSS_LOGCOSH = None


def negentropy(y: np.ndarray) -> np.ndarray:
    """Negentropy approximation J(y) ∝ (E[log cosh y] - E[log cosh ν])², per row (standardised)."""
    global _GAUSS_LOGCOSH
    if _GAUSS_LOGCOSH is None:
        nu = np.random.default_rng(12345).standard_normal(2_000_000)
        _GAUSS_LOGCOSH = float(np.mean(np.log(np.cosh(nu))))
    y = np.atleast_2d(y)
    y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-12)
    return (np.mean(np.log(np.cosh(y)), axis=1) - _GAUSS_LOGCOSH) ** 2


def excess_kurtosis(y: np.ndarray) -> np.ndarray:
    """E[y⁴]/E[y²]² - 3 per row: 0 for a Gaussian, > 0 for spiky (speech-like) signals."""
    y = np.atleast_2d(y)
    y = y - y.mean(axis=1, keepdims=True)
    v = np.mean(y * y, axis=1)
    return np.mean(y ** 4, axis=1) / np.maximum(v * v, 1e-24) - 3.0
