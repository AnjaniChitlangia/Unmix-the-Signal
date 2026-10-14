"""
whitening.py — centering and PCA whitening, written by hand.

The ICA model is   X = A S   where
    S : (n_sources, n_samples)   independent sources
    A : (n_mics,    n_sources)   unknown mixing matrix
    X : (n_mics,    n_samples)   what the microphones record

Before running ICA we *whiten* the observations: find a linear map K such that
Z = K (X - mean) has identity covariance, E[Z Z^T] = I.

Why this helps
--------------
If Z = K A S and both S and Z have identity covariance (sources are assumed unit
variance — the scale ambiguity of ICA lets us assume that for free), then

    I = E[Z Z^T] = (K A) E[S S^T] (K A)^T = (K A)(K A)^T

so the combined matrix K A is *orthogonal*. ICA no longer has to search over all
n x n matrices (n^2 unknowns) — only over rotations (n(n-1)/2 unknowns). That is
what makes the fixed-point FastICA updates simple and stable.

How it is done (PCA whitening)
------------------------------
1. Center:          Xc = X - mean(X)
2. Covariance:      C  = Xc Xc^T / (T - 1)
3. Eigendecompose:  C  = E D E^T        (C is symmetric PSD -> eigh)
4. Whitening map:   K  = D^{-1/2} E^T   (rotate onto principal axes, then
                                         rescale every axis to unit variance)
5. Whitened data:   Z  = K Xc           -> Cov(Z) = D^{-1/2} E^T C E D^{-1/2} = I

When there are more microphones than sources (m > n), only n eigenvalues carry
signal; the rest are (near) zero or noise. Keeping the top-n eigenvectors here is
classic PCA dimensionality reduction and makes the problem square again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WhiteningResult:
    Z: np.ndarray            # whitened data, (k, T)
    K: np.ndarray            # whitening matrix, (k, m):   Z = K (X - mean)
    K_inv: np.ndarray        # de-whitening matrix, (m, k): X - mean ≈ K_inv Z
    mean: np.ndarray         # per-channel mean that was removed, (m, 1)
    eigvals: np.ndarray      # all eigenvalues of Cov(X), descending
    eigvecs: np.ndarray      # matching eigenvectors as columns
    n_components: int        # k, number of kept components

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        total = np.sum(np.clip(self.eigvals, 0, None))
        return np.clip(self.eigvals, 0, None) / total if total > 0 else np.zeros_like(self.eigvals)


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each row's mean. Returns (centered X, mean column vector)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D (channels x samples) array, got shape {X.shape}.")
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def covariance(Xc: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of already-centered rows: Xc Xc^T / (T - 1)."""
    T = Xc.shape[1]
    if T < 2:
        raise ValueError("Need at least two samples to estimate a covariance.")
    return (Xc @ Xc.T) / (T - 1)


def pca_whiten(
    X: np.ndarray,
    n_components: int | None = None,
    eps: float = 1e-10,
) -> WhiteningResult:
    """
    Center and PCA-whiten observations X (m channels x T samples).

    n_components : number of principal components to keep (default: all
                   numerically non-zero ones). Use n_sources when m > n_sources.
    eps          : relative threshold below which an eigenvalue is treated as
                   zero (rank-deficient mixtures, duplicated microphones, silence).
    """
    Xc, mean = center(X)
    C = covariance(Xc)

    # eigh is the right tool: C is symmetric, so eigenvalues are real and the
    # eigenvectors orthonormal. It returns them in *ascending* order -> flip.
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    if eigvals[0] <= 0:
        raise ValueError("All observed channels are silent (zero variance); nothing to whiten.")

    usable = int(np.sum(eigvals > eps * eigvals[0]))
    if n_components is None:
        n_components = usable
    if n_components > usable:
        raise ValueError(
            f"Requested {n_components} components but the observations only have rank ≈ {usable}. "
            "The mixing matrix may be singular, or some channels are copies/silent — "
            "ICA cannot recover more sources than the data's rank."
        )

    D = eigvals[:n_components]
    E = eigvecs[:, :n_components]
    # K = D^{-1/2} E^T, written as row-scaling to avoid building diag matrices.
    K = (E / np.sqrt(D)).T
    K_inv = E * np.sqrt(D)
    Z = K @ Xc
    return WhiteningResult(Z=Z, K=K, K_inv=K_inv, mean=mean, eigvals=eigvals,
                           eigvecs=eigvecs, n_components=n_components)


def whiteness_error(Z: np.ndarray) -> float:
    """Max absolute deviation of Cov(Z) from the identity — 0 for perfect whitening."""
    Zc, _ = center(Z)
    return float(np.max(np.abs(covariance(Zc) - np.eye(Z.shape[0]))))
