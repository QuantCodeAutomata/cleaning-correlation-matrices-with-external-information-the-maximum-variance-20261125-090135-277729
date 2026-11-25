"""Unit tests for src/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import annualized_sharpe_ratio, frobenius_distance


def test_frobenius_distance_zero_for_identical_matrices():
    c = np.eye(5)
    assert frobenius_distance(c, c) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_distance_known_value():
    a = np.zeros((2, 2))
    b = np.eye(2)
    # diff = -I, Tr(diff^T diff) = Tr(I) = 2, sqrt(2).
    assert frobenius_distance(a, b) == pytest.approx(np.sqrt(2), abs=1e-12)


def test_frobenius_distance_symmetric():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    b = rng.standard_normal((4, 4))
    assert frobenius_distance(a, b) == pytest.approx(frobenius_distance(b, a))


def test_annualized_sharpe_ratio_sign_matches_mean_return():
    positive = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
    negative = -positive
    assert annualized_sharpe_ratio(positive, 52) > 0
    assert annualized_sharpe_ratio(negative, 52) < 0


def test_annualized_sharpe_ratio_scales_with_sqrt_periods():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal(100) * 0.01 + 0.001
    sr_weekly = annualized_sharpe_ratio(returns, 52)
    sr_daily_equiv = annualized_sharpe_ratio(returns, 252)
    assert sr_daily_equiv / sr_weekly == pytest.approx(np.sqrt(252 / 52), rel=1e-6)


def test_annualized_sharpe_ratio_rejects_degenerate_series():
    with pytest.raises(AssertionError):
        annualized_sharpe_ratio(np.array([0.01]), 52)
    with pytest.raises(AssertionError):
        annualized_sharpe_ratio(np.zeros(10), 52)
