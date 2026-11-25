"""Unit tests for the MaxVar estimator (src/maxvar.py), paper Section 2."""

from __future__ import annotations

import numpy as np
import pytest

from src.maxvar import groups_from_sizes, maxvar_correlation


def _random_correlation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n + 5))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_maxvar_output_is_valid_correlation_matrix():
    groups = groups_from_sizes([4, 3, 2])
    c = _random_correlation(9, seed=1)
    f = maxvar_correlation(c, groups)
    assert np.allclose(np.diag(f), 1.0, atol=1e-8)
    assert np.allclose(f, f.T, atol=1e-8)


def test_maxvar_reproduces_exact_block_structure():
    """If C already has the MaxVar factor structure (Section 2, step (c))
    built from equal-size groups sharing one common within-group leading
    eigenvalue mu and uniform (equal-magnitude) leading eigenvectors,
    cleaning it should return the same matrix: the estimator has zero bias
    on its own model. Equal group sizes and a common mu are needed for an
    *exact* recovery check: the row/column normalization (step (d),
    hat F_ij = F_ij / sqrt(F_ii F_jj)) is a diagonal congruence, and a
    block's post-normalization diagonal F_ii = delta + (mu_k - delta)/n_k
    must match *across* groups (not only within one group) for the
    generalized eigenvalue problem solved on the normalized matrix to
    recover the same relative eigenvalues and hence the same fixed point."""
    groups = groups_from_sizes([4, 4, 4])
    n, k = 12, 3

    p = np.zeros((n, k))
    for col, idx in enumerate(groups):
        p[idx, col] = 1.0 / np.sqrt(len(idx))  # uniform unit vector per group
    mu = np.array([2.5, 2.5, 2.5])  # common mu => F_ii identical across groups
    z = p
    delta = (n - mu.sum()) / (n - k)
    f = z @ np.diag(mu) @ z.T + delta * (np.eye(n) - z @ z.T)
    diag = np.diag(f)
    scale = 1.0 / np.sqrt(diag)
    c0 = f * scale[:, None] * scale[None, :]

    cleaned = maxvar_correlation(c0, groups)
    assert np.allclose(cleaned, c0, atol=1e-6)


def test_maxvar_groups_must_partition_all_assets():
    c = _random_correlation(5, seed=3)
    with pytest.raises(AssertionError):
        maxvar_correlation(c, [np.array([0, 1]), np.array([2, 3])])  # missing index 4


def test_maxvar_invertible_even_if_input_is_singular():
    """Proposition 1(iii) of the paper: the cleaned matrix stays invertible
    even when the empirical matrix is singular (e.g. more assets than
    observations would make the sample correlation singular)."""
    groups = groups_from_sizes([3, 3, 3])
    c = _random_correlation(9, seed=4)
    # Force near-singularity by collapsing two rows/columns to be identical.
    c[1, :] = c[0, :]
    c[:, 1] = c[:, 0]
    c[1, 1] = 1.0
    f = maxvar_correlation(c, groups)
    sign, logdet = np.linalg.slogdet(f)
    assert sign > 0 and np.isfinite(logdet), "cleaned matrix must be invertible"


def test_groups_from_sizes():
    groups = groups_from_sizes([2, 3])
    assert np.array_equal(groups[0], np.array([0, 1]))
    assert np.array_equal(groups[1], np.array([2, 3, 4]))
