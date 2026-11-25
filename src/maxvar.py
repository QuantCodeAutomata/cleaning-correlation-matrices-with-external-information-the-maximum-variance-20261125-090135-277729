"""Maximum Variance (MaxVar) correlation-matrix estimator.

Custom implementation of the MaxVar estimator described in the paper
"Cleaning correlation matrices with external information: the Maximum
Variance estimator" (Section 2). skfolio's covariance-denoising estimators
only provide Marchenko-Pastur eigenvalue clipping / detoning, not a
group-constrained maximum-variance completion, so the four steps of
Section 2 are implemented directly from the paper's equations.

Steps (paper, Section 2):
  (a) for each group G_k, p_k is the leading eigenvector of the restriction
      of the correlation matrix C to G_k, completed with zeros outside G_k;
  (b) G = P^T P, H = P^T C P; solve the generalized eigenvalue problem
      H a = mu G a by diagonalizing G^{-1/2} H G^{-1/2} = Omega M Omega^T,
      giving constrained eigenvalues mu_1 >= ... >= mu_K and constrained
      eigenvectors Z = P G^{-1/2} Omega;
  (c) complete the matrix with F = Z M Z^T + delta (I_n - Z Z^T),
      delta = (n - sum_k mu_k) / (n - K);
  (d) normalize to a correlation matrix: hat F_ij = F_ij / sqrt(F_ii F_jj).
"""

from __future__ import annotations

import numpy as np


def _leading_eigenvector(block: np.ndarray) -> np.ndarray:
    """Leading (largest-eigenvalue) eigenvector of a symmetric block, unit-norm."""
    eigvals, eigvecs = np.linalg.eigh(block)
    leading = eigvecs[:, np.argmax(eigvals)]
    # Fix the sign so the portfolio has a non-negative mean loading (paper's
    # "maximum variance portfolio" is defined up to a sign; the convention
    # below only affects the sign of p_k, not F, since F is built from P P^T
    # products that are sign-invariant).
    if leading.sum() < 0:
        leading = -leading
    return leading


def _group_loading_matrix(correlation: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Matrix P (n x K): column k is the leading eigenvector of group k, zero elsewhere."""
    n = correlation.shape[0]
    k = len(groups)
    p = np.zeros((n, k))
    for col, idx in enumerate(groups):
        block = correlation[np.ix_(idx, idx)]
        p[idx, col] = _leading_eigenvector(block)
    return p


def maxvar_correlation(correlation: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Clean a sample correlation matrix with the MaxVar estimator.

    Parameters
    ----------
    correlation : (n, n) array
        Sample (empirical) correlation matrix.
    groups : list of int arrays
        Partition of ``range(n)`` into K groups (e.g. asset classes); the
        external information used by the estimator.

    Returns
    -------
    (n, n) array
        The cleaned, normalized correlation matrix hat F (paper, Eq. in
        Section 2, step (d)).
    """
    n = correlation.shape[0]
    k = len(groups)
    assert sum(len(g) for g in groups) == n, "groups must partition all n assets"
    assert np.allclose(correlation, correlation.T, atol=1e-8), "correlation must be symmetric"

    p = _group_loading_matrix(correlation, groups)
    g = p.T @ p
    h = p.T @ correlation @ p

    # Diagonalize G^{-1/2} H G^{-1/2} = Omega M Omega^T (paper, step (b)).
    g_eigvals, g_eigvecs = np.linalg.eigh(g)
    g_eigvals = np.clip(g_eigvals, 1e-12, None)  # groups are linearly independent in practice
    g_inv_sqrt = g_eigvecs @ np.diag(g_eigvals**-0.5) @ g_eigvecs.T

    m_matrix = g_inv_sqrt @ h @ g_inv_sqrt
    mu, omega = np.linalg.eigh(m_matrix)
    order = np.argsort(mu)[::-1]  # mu_1 >= ... >= mu_K
    mu = mu[order]
    omega = omega[:, order]

    z = p @ g_inv_sqrt @ omega  # constrained eigenvectors (paper, step (b))

    delta = (n - mu.sum()) / (n - k)  # paper, step (c)
    f = z @ np.diag(mu) @ z.T + delta * (np.eye(n) - z @ z.T)

    diag = np.diag(f)
    assert np.all(diag > 0), "MaxVar diagonal must stay positive for normalization"
    scale = 1.0 / np.sqrt(diag)
    f_hat = f * scale[:, None] * scale[None, :]  # paper, step (d)
    return f_hat


def groups_from_sizes(sizes: list[int]) -> list[np.ndarray]:
    """Convenience: build index groups from consecutive block sizes."""
    groups = []
    start = 0
    for size in sizes:
        groups.append(np.arange(start, start + size))
        start += size
    return groups
