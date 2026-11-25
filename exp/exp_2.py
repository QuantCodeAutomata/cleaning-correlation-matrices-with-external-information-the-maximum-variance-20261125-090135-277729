"""Experiment 2: out-of-sample Sharpe ratio of the risk-parity portfolio
(paper, Section 3, Table 3).

For each EMA parameter eta' (slow, eta'=1/750; fast, eta'=1/100), the
correlation matrix is estimated at every weekly date from 8 May 1985 and
cleaned with the empirical (raw), RIE and MaxVar estimators. A risk-parity
portfolio Pi_t ~ Sigma_t^{-1} C_t^{-1} e (Sigma_t the diagonal volatility
matrix, e = 1 for stock indices and bonds, 0 for currencies) is formed at
each weekly date using only information available up to that date, and
applied to the following week's realized returns. The annualized Sharpe
ratio of the resulting weekly portfolio-return series is computed over the
paper's out-of-sample window, 1 January 1993 to 31 December 2018.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import load_price_panel
from src import config
from src.maxvar import maxvar_correlation
from src.metrics import annualized_sharpe_ratio
from src.portfolio import (
    ema_correlation_series,
    ema_volatility,
    rescale_by_volatility,
    risk_parity_weights,
    to_weekly,
)
from src.rie import rie_correlation
from src.universe import group_indices, include_mask, ordered_tickers


def load_returns(use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    tickers = ordered_tickers()
    prices, meta = load_price_panel(
        tickers, config.SAMPLE_START_PAPER, config.SAMPLE_END_PAPER,
        invert=config.CURRENCY_INVERT, use_cache=use_cache,
    )
    prices = prices.reindex(columns=tickers)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    return returns, meta


def portfolio_returns_for_estimator(
    weekly_vol: pd.DataFrame,
    weekly_raw_returns: pd.DataFrame,
    corr_series: list[tuple],
    clean_fn,
    groups: list[np.ndarray],
    mask: np.ndarray,
) -> pd.Series:
    """Weekly out-of-sample portfolio returns for one correlation-cleaning
    method, rebalancing at date t with (C_t, v_t) and applying the resulting
    weights to the next week's realized return r_{t+1} (no look-ahead)."""
    dates = [d for d, _ in corr_series]
    returns_out = {}
    for i in range(len(dates) - 1):
        date_t, next_date = dates[i], dates[i + 1]
        c_t = dict(corr_series)[date_t]
        if date_t not in weekly_vol.index or next_date not in weekly_raw_returns.index:
            continue
        vol_t = weekly_vol.loc[date_t].to_numpy()
        if not np.all(np.isfinite(vol_t)) or np.any(vol_t <= 0):
            continue
        try:
            cleaned = clean_fn(c_t)
        except np.linalg.LinAlgError:
            continue
        try:
            weights = risk_parity_weights(cleaned, vol_t, mask)
        except np.linalg.LinAlgError:
            continue
        next_returns = weekly_raw_returns.loc[next_date].to_numpy()
        if not np.all(np.isfinite(next_returns)):
            continue
        returns_out[next_date] = float(weights @ next_returns)
    return pd.Series(returns_out).sort_index()


def run_for_eta(
    eta_prime: float,
    returns: pd.DataFrame,
    groups: list[np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    volatility = ema_volatility(returns, config.ETA_VOL)
    rescaled = rescale_by_volatility(returns, volatility)
    weekly_rescaled = to_weekly(rescaled, how="sum")
    weekly_vol = to_weekly(volatility, how="last")
    weekly_raw_returns = to_weekly(returns, how="sum")
    corr_series = ema_correlation_series(weekly_rescaled, eta_prime)

    oos_start, oos_end = pd.Timestamp(config.OOS_START_PAPER), pd.Timestamp(config.OOS_END_PAPER)

    estimators = {
        "empirical": lambda c: c,
        "rie": lambda c: rie_correlation(c, n_observations=max(int(round(1 / eta_prime)), 2)),
        "maxvar": lambda c: maxvar_correlation(c, groups),
    }
    sharpe = {}
    coverage = {}
    for name, clean_fn in estimators.items():
        port_returns = portfolio_returns_for_estimator(
            weekly_vol, weekly_raw_returns, corr_series, clean_fn, groups, mask
        )
        oos_returns = port_returns.loc[(port_returns.index >= oos_start) & (port_returns.index <= oos_end)]
        if len(oos_returns) < 2 or oos_returns.std(ddof=1) == 0:
            sharpe[name] = None
        else:
            sharpe[name] = annualized_sharpe_ratio(oos_returns.to_numpy(), config.WEEKS_PER_YEAR)
        coverage[name] = {
            "n_weeks_used": int(len(oos_returns)),
            "first_week_used": str(oos_returns.index.min()) if len(oos_returns) else None,
            "last_week_used": str(oos_returns.index.max()) if len(oos_returns) else None,
        }
    return sharpe, coverage


def main(use_cache: bool = True) -> pd.DataFrame:
    returns, meta = load_returns(use_cache=use_cache)
    groups, _ = group_indices()
    mask = include_mask(config.RISK_PARITY_INCLUDE_GROUPS)

    rows = []
    coverage_by_label = {}
    for label, eta_prime, expected in (
        ("slow (eta'=1/750)", config.ETA_CORR_SLOW, config.EXPECTED_SHARPE_ETA_SLOW),
        ("fast (eta'=1/100)", config.ETA_CORR_FAST, config.EXPECTED_SHARPE_ETA_FAST),
    ):
        sharpe, coverage = run_for_eta(eta_prime, returns, groups, mask)
        coverage_by_label[label] = coverage
        for method in ("empirical", "rie", "maxvar"):
            rows.append({
                "eta_prime_label": label,
                "method": method,
                "sharpe_ratio": sharpe[method],
                "paper_table3_value": expected[method],
                "n_weeks_used": coverage[method]["n_weeks_used"],
                "first_week_used": coverage[method]["first_week_used"],
                "last_week_used": coverage[method]["last_week_used"],
            })

    results = pd.DataFrame(rows)
    results.to_csv("results/exp2_sharpe_ratios.csv", index=False)

    # Diagnostic: how many of the 42 proxy tickers have a simultaneously
    # finite weekly return/volatility in each calendar year of the paper's
    # nominal out-of-sample window (paper Section 3: 1 Jan 1993-31 Dec 2018).
    # The risk-parity weights and the MaxVar/RIE cleaning at date t require
    # every ticker's weekly volatility to be finite, so a week is usable only
    # once all 42 proxies have started trading.
    volatility = ema_volatility(returns, config.ETA_VOL)
    weekly_vol = to_weekly(volatility, how="last")
    active_counts = weekly_vol.notna().sum(axis=1)
    oos_mask = (weekly_vol.index >= pd.Timestamp(config.OOS_START_PAPER)) & (
        weekly_vol.index <= pd.Timestamp(config.OOS_END_PAPER)
    )
    active_in_oos = active_counts.loc[oos_mask]
    first_full_universe = active_counts[active_counts == weekly_vol.shape[1]].index.min()

    with open("results/exp2_meta.json", "w") as f:
        json.dump({
            "n_tickers": len(ordered_tickers()),
            "oos_start_paper": config.OOS_START_PAPER,
            "oos_end_paper": config.OOS_END_PAPER,
            "non_massive_source": meta.get("non_massive_source", {}),
            "failed_tickers": meta.get("failed", {}),
            "realized_weekly_return_coverage": coverage_by_label,
            "active_ticker_count_in_oos_window": {
                "min": int(active_in_oos.min()),
                "median": float(active_in_oos.median()),
                "max": int(active_in_oos.max()),
                "total_tickers": int(weekly_vol.shape[1]),
            },
            "first_date_all_tickers_active": str(first_full_universe),
        }, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["empirical", "rie", "maxvar"]
    labels = ["slow (eta'=1/750)", "fast (eta'=1/100)"]
    width = 0.12
    x = np.arange(len(methods))
    for i, label in enumerate(labels):
        subset = results[results["eta_prime_label"] == label].set_index("method")
        vals = [subset.loc[m, "sharpe_ratio"] if subset.loc[m, "sharpe_ratio"] is not None else np.nan for m in methods]
        exp_vals = [subset.loc[m, "paper_table3_value"] for m in methods]
        ax.bar(x + (2 * i - 1) * width, vals, width=width, label=f"This replication, {label}")
        ax.bar(x + (2 * i - 1) * width, exp_vals, width=width, alpha=0.0)
        ax.scatter(x + (2 * i - 1) * width, exp_vals, color="black", marker="_", s=200,
                   label=f"Paper, Table 3, {label}" if i < 2 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(["Empirical", "RIE", "MaxVar"])
    ax.set_ylabel("Annualized out-of-sample Sharpe ratio")
    ax.set_title("Risk-parity portfolio Sharpe ratio, 1993-2018 (paper, Table 3)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/exp2_sharpe_ratios.png", dpi=150)
    plt.close(fig)

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
