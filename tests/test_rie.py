"""Unit tests for the Rotationally Invariant Estimator (src/rie.py),
Bun, Bouchaud & Potters (2017), as cited in the paper's Section 3."""

from __future__ import annotations

import numpy as np

from src.rie import rie_correlation


def _random_correlation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n + 5))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_rie_output_is_valid_correlation_matrix():
    c = _random_correlation(20, seed=10)
    cleaned = rie_correlation(c, n_observations=200)
    assert np.allclose(np.diag(cleaned), 1.0, atol=1e-8)
    assert np.allclose(cleaned, cleaned.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(cleaned)
    assert np.all(eigvals > -1e-8), "cleaned correlation matrix must stay PSD"


def test_rie_identity_input_stays_near_identity():
    """A pure-noise sample correlation matrix (identity, i.e. zero true
    correlation) should be cleaned towards eigenvalues near 1, not amplify
    spurious structure."""
    n = 15
    cleaned = rie_correlation(np.eye(n), n_observations=500)
    assert np.allclose(cleaned, np.eye(n), atol=1e-6)


def test_rie_shrinks_towards_identity_relative_to_noisy_sample():
    """With q = N/T not small, the sample eigenvalue spread is inflated by
    noise; the RIE-cleaned eigenvalue spread should be smaller than the
    sample's (Bun, Bouchaud & Potters 2017, Section 4: eigenvalue cleaning
    reduces dispersion introduced by estimation noise)."""
    rng = np.random.default_rng(11)
    n, t = 40, 60
    true_corr = np.eye(n)
    sample = rng.multivariate_normal(np.zeros(n), true_corr, size=t)
    empirical = np.corrcoef(sample, rowvar=False)
    cleaned = rie_correlation(empirical, n_observations=t)

    sample_eigvals = np.linalg.eigvalsh(empirical)
    cleaned_eigvals = np.linalg.eigvalsh(cleaned)
    assert np.std(cleaned_eigvals) < np.std(sample_eigvals)


def test_rie_trace_preserved():
    c = _random_correlation(10, seed=12)
    cleaned = rie_correlation(c, n_observations=100)
    # The pre-normalization eigenvalue rescaling preserves Tr = N; the final
    # unit-diagonal normalization also implies Tr(cleaned) = N exactly.
    assert np.isclose(np.trace(cleaned), cleaned.shape[0], atol=1e-8)
