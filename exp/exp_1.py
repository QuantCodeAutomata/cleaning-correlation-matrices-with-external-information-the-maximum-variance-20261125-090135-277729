"""Experiment 1: Monte Carlo noise-reduction test (paper, Section 3, Table 1).

A reference correlation matrix C0 is built by cleaning the empirical
correlation of the universe (``src/config.py``) on the reference date with
the MaxVar estimator (K=3 asset-class groups), so C0 exactly satisfies the
MaxVar block structure. ``MC_N_SAMPLES`` independent samples of
``MC_T_PAPER`` i.i.d. Gaussian vectors with covariance C0 are drawn; each
sample's empirical correlation matrix is cleaned with MaxVar and with the
RIE, and the Frobenius distance of each estimate (and of the raw empirical
matrix) to C0 is recorded and averaged over the samples (paper, Section 3).
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
from src.metrics import frobenius_distance
from src.portfolio import ema_correlation_series, ema_volatility, rescale_by_volatility, to_weekly
from src.rie import rie_correlation
from src.universe import group_indices, ordered_tickers


def build_reference_matrix(use_cache: bool = True) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """C0: the MaxVar-cleaned correlation matrix on ``config.REFERENCE_DATE``
    (paper, Section 3), using the full history up to the reference date so
    the slow EMA (eta'=1/750, about 150 weekly returns) has warmed up.
    """
    tickers = ordered_tickers()
    prices, meta = load_price_panel(
        tickers, config.SAMPLE_START_PAPER, config.REFERENCE_DATE,
        invert=config.CURRENCY_INVERT, use_cache=use_cache,
    )
    prices = prices.reindex(columns=tickers)
    returns = prices.pct_change().iloc[1:]

    volatility = ema_volatility(returns, config.ETA_VOL)
    rescaled = rescale_by_volatility(returns, volatility)
    weekly = to_weekly(rescaled, how="sum")
    corr_series = ema_correlation_series(weekly, config.ETA_CORR_SLOW)

    dates = pd.DatetimeIndex([d for d, _ in corr_series])
    reference_date = pd.Timestamp(config.REFERENCE_DATE)
    as_of = dates[dates <= reference_date]
    assert len(as_of) > 0, "no weekly correlation estimate at or before the reference date"
    empirical_ref = dict(corr_series)[as_of[-1]]

    groups, _ = group_indices()
    c0 = maxvar_correlation(empirical_ref, groups)
    return c0, tickers, groups, meta


def run_monte_carlo(c0: np.ndarray, groups: list[np.ndarray]) -> pd.DataFrame:
    """Average Frobenius distance to C0 for the empirical, RIE and MaxVar
    estimators over ``config.MC_N_SAMPLES`` samples of length
    ``config.MC_T_PAPER`` drawn from N(0, C0) (paper, Section 3)."""
    rng = np.random.default_rng(config.MC_SEED)
    n = c0.shape[0]
    distances = {"empirical": [], "rie": [], "maxvar": []}
    for _ in range(config.MC_N_SAMPLES):
        sample = rng.multivariate_normal(mean=np.zeros(n), cov=c0, size=config.MC_T_PAPER)
        empirical = np.corrcoef(sample, rowvar=False)
        rie = rie_correlation(empirical, n_observations=config.MC_T_PAPER)
        maxvar = maxvar_correlation(empirical, groups)
        distances["empirical"].append(frobenius_distance(empirical, c0))
        distances["rie"].append(frobenius_distance(rie, c0))
        distances["maxvar"].append(frobenius_distance(maxvar, c0))

    expected = {
        "empirical": config.EXPECTED_FROBENIUS_EMPIRICAL,
        "rie": config.EXPECTED_FROBENIUS_RIE,
        "maxvar": config.EXPECTED_FROBENIUS_MAXVAR,
    }
    rows = []
    for method, values in distances.items():
        rows.append({
            "method": method,
            "mean_frobenius_distance": float(np.mean(values)),
            "std_frobenius_distance": float(np.std(values, ddof=1)),
            "n_samples": config.MC_N_SAMPLES,
            "paper_table1_value": expected[method],
        })
    return pd.DataFrame(rows)


def main(use_cache: bool = True) -> pd.DataFrame:
    c0, tickers, groups, meta = build_reference_matrix(use_cache=use_cache)
    results = run_monte_carlo(c0, groups)

    results.to_csv("results/exp1_frobenius_distances.csv", index=False)
    with open("results/exp1_meta.json", "w") as f:
        json.dump({
            "n_tickers": len(tickers),
            "reference_date": config.REFERENCE_DATE,
            "mc_n_samples": config.MC_N_SAMPLES,
            "mc_t": config.MC_T_PAPER,
            "non_massive_source": meta.get("non_massive_source", {}),
            "failed_tickers": meta.get("failed", {}),
        }, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["empirical", "rie", "maxvar"]
    values = [results.set_index("method").loc[m, "mean_frobenius_distance"] for m in order]
    expected = [results.set_index("method").loc[m, "paper_table1_value"] for m in order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, values, width=0.4, label="This replication")
    ax.bar(x + 0.2, expected, width=0.4, label="Paper, Table 1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Empirical", "RIE", "MaxVar"])
    ax.set_ylabel("Mean Frobenius distance to C0")
    ax.set_title("Monte Carlo noise reduction (paper, Section 3, Table 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_frobenius_distances.png", dpi=150)
    plt.close(fig)

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
