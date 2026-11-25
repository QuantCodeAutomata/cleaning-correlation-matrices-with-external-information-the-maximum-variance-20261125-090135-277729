"""Exponentially-weighted volatility/correlation estimation and the
risk-parity portfolio of the empirical application (paper, Section 3).

Custom implementation of the paper's equations: skfolio's ``EWCovariance``
estimator performs a single EWMA on a covariance matrix directly, not the
paper's two-stage procedure (an EWMA of squared daily returns for
volatility at parameter eta, rescaling to weekly returns, then a
*separate* EWMA of the rescaled weekly cross-products at parameter eta'
to build correlation). The two parameters play different roles in the
paper (Table 3 varies eta' only), so the two-stage recursion is
implemented directly from Eqs. 10-11 rather than approximated by a single
off-the-shelf EWMA covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_volatility(returns: pd.DataFrame, eta: float) -> pd.DataFrame:
    """EMA of squared daily returns (paper, Eq. 10): v_t^2 = (1-eta) v_{t-1}^2 + eta r_t^2.

    Each column's recursion is initialized separately at that ticker's own
    first valid return, with the sample variance of its first 21 trading
    days (about one calendar month) -- a start-up detail the paper does not
    specify. Tickers are run independently (rather than one shared loop from
    the first row of the panel) because the paper's 47 futures all start on
    8 May 1985, while the ETF/FX proxies used here (``src/config.py``) have
    staggered inception dates; a value before inception is left as NaN, and
    a missing return on a trading day after inception (e.g. a local holiday)
    is treated as a zero shock rather than breaking the recursion.
    """
    r2 = returns**2
    v2 = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        series = r2[col]
        start = series.first_valid_index()
        if start is None:
            continue
        values = series.loc[start:].to_numpy()
        init_window = min(21, len(values))
        v2_col = np.empty(len(values))
        v2_col[0] = np.nanmean(values[:init_window])
        for t in range(1, len(values)):
            shock = values[t]
            shock = 0.0 if np.isnan(shock) else shock
            v2_col[t] = (1 - eta) * v2_col[t - 1] + eta * shock
        v2.loc[start:, col] = v2_col
    return np.sqrt(v2)


def rescale_by_volatility(returns: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    """Daily returns divided by their lagged EMA volatility (paper, Section 3)."""
    lagged_vol = volatility.shift(1)
    return returns / lagged_vol


def to_weekly(daily: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
    """Aggregate a daily panel to calendar weeks ending on Friday (paper,
    Section 3: "compute weekly returns to offset different trading hours").

    ``how="sum"`` adds rescaled or raw daily returns within the week;
    ``how="last"`` keeps the last available daily value in the week (used to
    align a daily volatility estimate with the weekly rebalancing dates).
    Both calls share the same ``resample("W-FRI")`` grouping, so a weekly
    return panel and a weekly volatility panel built this way have identical
    date indices.
    """
    resampled = daily.resample("W-FRI")
    weekly = resampled.sum(min_count=1) if how == "sum" else resampled.last()
    return weekly.dropna(how="all")


def ema_correlation_series(weekly_returns: pd.DataFrame, eta_prime: float) -> list[tuple]:
    """EMA correlation matrix at every weekly date (paper, Eq. 11).

    hat C_t^ij = (1-eta') hat C_{t-1}^ij + eta' rhat_t^i rhat_t^j, normalized
    to unit diagonal at every step. Returns a list of (date, correlation
    array) pairs, one per week once the recursion has started.
    """
    assets = weekly_returns.columns
    n = len(assets)
    # A ticker not yet trading (before its proxy's inception) has no weekly
    # return; filling it with zero contributes no evidence to its row/column
    # of the cross-product that week, equivalent to carrying its prior
    # estimate forward unchanged by that week's update.
    weekly_returns = weekly_returns.fillna(0.0)
    # Initialize the recursion with the identity matrix (no cross-asset
    # information) rather than a sample covariance of future weeks: the
    # paper's Eq. 11 is a pure causal recursion and does not specify a
    # start-up value, but seeding it from data ahead of the first reported
    # date would leak future information into every early estimate. With
    # eta'=1/750, full warm-up needs about 750 weekly returns; estimates
    # computed before that many weeks have elapsed are correspondingly noisy,
    # which is a data-availability limitation, not a look-ahead one.
    c_hat = np.eye(n)
    out = []
    for date, row in weekly_returns.iterrows():
        x = row.values.reshape(-1, 1)
        c_hat = (1 - eta_prime) * c_hat + eta_prime * (x @ x.T)
        diag = np.diag(c_hat)
        diag = np.where(diag <= 0, 1.0, diag)
        scale = 1.0 / np.sqrt(diag)
        c_norm = c_hat * scale[:, None] * scale[None, :]
        out.append((date, c_norm))
    return out


def risk_parity_weights(
    correlation: np.ndarray, volatility: np.ndarray, include_mask: np.ndarray
) -> np.ndarray:
    """Risk-parity weights Pi proportional to Sigma^{-1} C^{-1} e (paper, Section 3).

    ``include_mask`` is the vector e (1 for stock indices and bonds, 0 for
    currencies, paper Section 3). Weights are normalized to sum to one in
    absolute value so portfolios across estimators are comparable.
    """
    inv_vol = 1.0 / volatility
    c_inv_e = np.linalg.solve(correlation, include_mask)
    weights = inv_vol * c_inv_e
    gross = np.abs(weights).sum()
    assert gross > 0, "degenerate risk-parity weights"
    return weights / gross
