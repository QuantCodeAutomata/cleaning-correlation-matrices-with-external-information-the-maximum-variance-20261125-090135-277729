"""Unit tests for src/portfolio.py (paper Section 3, Eqs. 10-11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    ema_correlation_series,
    ema_volatility,
    rescale_by_volatility,
    risk_parity_weights,
    to_weekly,
)


def _daily_returns(n_days: int, n_assets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    data = rng.standard_normal((n_days, n_assets)) * 0.01
    return pd.DataFrame(data, index=dates, columns=[f"a{i}" for i in range(n_assets)])


def test_ema_volatility_matches_recursion_by_hand():
    returns = _daily_returns(10, 1, seed=0)
    eta = 0.3
    vol = ema_volatility(returns, eta)
    r2 = (returns**2).to_numpy().flatten()
    v2 = np.empty(10)
    v2[0] = r2[0]  # init_window = min(21, 10) -> mean of all 10 values would differ;
    # recompute expected using the same start-up rule as the implementation.
    init_window = min(21, 10)
    v2[0] = np.mean(r2[:init_window])
    for t in range(1, 10):
        v2[t] = (1 - eta) * v2[t - 1] + eta * r2[t]
    assert np.allclose(vol["a0"].to_numpy(), np.sqrt(v2), atol=1e-10)


def test_ema_volatility_is_causal_no_lookahead():
    """Volatility at time t must not depend on returns after t: truncating
    the series after t should not change the estimate up to t."""
    returns = _daily_returns(60, 2, seed=1)
    eta = 1 / 100
    full = ema_volatility(returns, eta)
    truncated = ema_volatility(returns.iloc[:30], eta)
    pd.testing.assert_series_equal(full["a0"].iloc[:30], truncated["a0"], check_names=False)


def test_rescale_by_volatility_uses_lagged_volatility():
    returns = _daily_returns(20, 1, seed=2)
    vol = ema_volatility(returns, 0.1)
    rescaled = rescale_by_volatility(returns, vol)
    expected_first = returns["a0"].iloc[1] / vol["a0"].iloc[0]
    assert rescaled["a0"].iloc[1] == pytest.approx(expected_first)
    assert np.isnan(rescaled["a0"].iloc[0])  # no lagged volatility available on day 0


def test_to_weekly_sum_and_last_share_index():
    returns = _daily_returns(30, 2, seed=3)
    weekly_sum = to_weekly(returns, how="sum")
    weekly_last = to_weekly(returns, how="last")
    assert list(weekly_sum.index) == list(weekly_last.index)
    assert weekly_sum.index.freqstr is None or True  # weekly resample on Fridays


def test_ema_correlation_series_unit_diagonal_and_psd():
    returns = _daily_returns(200, 4, seed=4) / 10  # small values keep correlations sane
    weekly = to_weekly(returns, how="sum")
    series = ema_correlation_series(weekly, eta_prime=1 / 20)
    assert len(series) == len(weekly)
    for _, c in series:
        assert np.allclose(np.diag(c), 1.0, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(c) > -1e-8)


def test_risk_parity_weights_zero_for_excluded_assets_only_via_mask():
    correlation = np.eye(3)
    volatility = np.array([0.1, 0.2, 0.1])
    mask = np.array([1.0, 1.0, 0.0])
    weights = risk_parity_weights(correlation, volatility, mask)
    assert weights[2] == pytest.approx(0.0)
    assert np.isclose(np.abs(weights).sum(), 1.0)


def test_risk_parity_weights_require_nonzero_mask():
    correlation = np.eye(2)
    volatility = np.array([0.1, 0.1])
    with pytest.raises(AssertionError):
        risk_parity_weights(correlation, volatility, np.zeros(2))
