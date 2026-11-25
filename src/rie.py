"""Rotationally Invariant Estimator (RIE) of Bun, Bouchaud & Potters, used in
the paper as the main noise-cleaning benchmark against the MaxVar estimator
(paper, Section 3: "the RIE of Bun et al. (2016, 2017)").

Context7 was queried for a library implementation (skfolio's covariance
estimators: ``DenoiseCovariance``/``DetoneCovariance`` implement Laloux et
al. eigenvalue *clipping* with a constant residual eigenvalue, not the
optimal nonlinear-shrinkage formula of Bun et al.; no installed package
implements the Bun-Bouchaud-Potters RIE), so the estimator is implemented
directly from Bun, Bouchaud & Potters (2017), "Cleaning large correlation
matrices: tools from random matrix theory", Physics Reports 666, Section 4
(the finite-N regularized Stieltjes-transform shrinkage formula, their
Eqs. 4.14-4.18): for each sample eigenvalue lambda_k of the empirical
correlation matrix E, with q = N/T and a finite-N regularized Stieltjes
transform g(z) = (1/N) sum_j 1/(z - lambda_j) evaluated just below the real
axis,

    xi_k = lambda_k / |1 - q + q * lambda_k * g(lambda_k - i*eta)|^2,

eta = 1/sqrt(N) (their recommended finite-N regularization). The cleaned
matrix keeps the sample eigenvectors and replaces the eigenvalues by xi_k,
rescaled to preserve the trace (= N) and finally renormalized to a
correlation matrix.
"""

from __future__ import annotations

import numpy as np


def rie_correlation(correlation: np.ndarray, n_observations: int) -> np.ndarray:
    """Clean a sample correlation matrix with the Bun-Bouchaud-Potters RIE.

    Parameters
    ----------
    correlation : (n, n) array
        Sample (empirical) correlation matrix of N assets.
    n_observations : int
        Number of observations T used to estimate ``correlation``.

    Returns
    -------
    (n, n) array
        The cleaned, trace-preserving, unit-diagonal correlation matrix.
    """
    n = correlation.shape[0]
    t = n_observations
    assert np.allclose(correlation, correlation.T, atol=1e-8), "correlation must be symmetric"
    assert t > 0, "n_observations must be positive"

    q = n / t
    eigvals, eigvecs = np.linalg.eigh(correlation)  # ascending order
    eigvals = np.clip(eigvals, 0.0, None)  # correlation matrix is PSD up to float noise

    eta = 1.0 / np.sqrt(n)  # finite-N regularization (Bun, Bouchaud & Potters 2017)
    z = eigvals - 1j * eta
    # g(z) = (1/N) sum_j 1/(z - lambda_j), vectorized over all evaluation points z.
    g = np.mean(1.0 / (z[:, None] - eigvals[None, :]), axis=1)

    denom = np.abs(1 - q + q * eigvals * g) ** 2
    denom = np.where(denom < 1e-12, 1e-12, denom)
    xi = eigvals / denom
    xi = np.clip(xi, 1e-10, None)

    xi *= n / xi.sum()  # preserve Tr(hat C) = N (paper's normalization convention)

    cleaned = (eigvecs * xi) @ eigvecs.T
    diag = np.diag(cleaned)
    assert np.all(diag > 0), "RIE diagonal must stay positive for normalization"
    scale = 1.0 / np.sqrt(diag)
    return cleaned * scale[:, None] * scale[None, :]
