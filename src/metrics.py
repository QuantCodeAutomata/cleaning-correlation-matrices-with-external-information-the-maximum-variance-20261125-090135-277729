"""Evaluation metrics shared by both experiments."""

from __future__ import annotations

import numpy as np


def frobenius_distance(estimate: np.ndarray, reference: np.ndarray) -> float:
    """d(hat C, C0) = sqrt(Tr[(hat C - C0)^T (hat C - C0)]) (paper, Section 3)."""
    diff = estimate - reference
    return float(np.sqrt(np.trace(diff.T @ diff)))


def annualized_sharpe_ratio(
    returns: np.ndarray, periods_per_year: float
) -> float:
    """Annualized Sharpe ratio of a return series (zero risk-free rate).

    mean(returns) / std(returns) * sqrt(periods_per_year), the standard
    annualization of a per-period Sharpe ratio (paper, Section 3: the
    out-of-sample Sharpe ratio of the risk-parity portfolio).
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    assert returns.size > 1, "need at least two returns to estimate a Sharpe ratio"
    std = returns.std(ddof=1)
    assert std > 0, "return series has zero variance"
    return float(returns.mean() / std * np.sqrt(periods_per_year))
