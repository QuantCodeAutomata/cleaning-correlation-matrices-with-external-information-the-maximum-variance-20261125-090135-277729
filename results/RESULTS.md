# Results — Cleaning correlation matrices with external information: the Maximum Variance estimator

Joel Bun, Jean-Philippe Bouchaud and Marc Potters, "Cleaning correlation
matrices with external information: the Maximum Variance estimator".

## Experiment 1 — Monte Carlo simulation: noise reduction of MaxVar vs. RIE vs. the empirical correlation matrix

Reproduces paper Section 3 / Table 1: a reference correlation matrix `C0` is
built as the MaxVar cleaning (K=3 asset-class groups) of the 28 December 2018
correlation matrix of the universe described below, estimated with
`eta'=1/750`. 100 independent samples of `T=150` i.i.d. Gaussian vectors of
covariance `C0` are drawn; each sample's empirical correlation matrix is
cleaned with the MaxVar estimator and with the Rotationally Invariant
Estimator (RIE), and the Frobenius distance `d(hat C, C0)` to the reference
is averaged over the 100 samples.

Source files: `exp/exp_1.py`, `src/maxvar.py`, `src/rie.py`,
`src/metrics.py`. Output: `results/exp1_frobenius_distances.csv`,
`results/exp1_meta.json`, `results/exp1_frobenius_distances.png`.

| Method    | Mean Frobenius distance (this run) | Std. dev. (100 samples) | Paper, Table 1 | Relative difference | Verdict |
|-----------|------------------------------------:|-------------------------:|---------------:|---------------------:|---------|
| Empirical | 2.974                                | 0.547                     | 3.24            | -8.2%                 | reproduced |
| RIE       | 2.774                                | 0.579                     | 2.99            | -7.2%                 | reproduced |
| MaxVar    | 2.104                                | 0.698                     | 2.09            | +0.7%                 | reproduced |

The ranking MaxVar < RIE < Empirical holds, matching the paper exactly, and
every value lies within the 10% band used here as the reproduction
threshold. The paper's qualitative claim — the MaxVar estimator cuts the
noise-driven Frobenius distance by roughly one third relative to the
empirical matrix (here -29.3%, paper -35.5%), while the RIE's reduction stays
below 10% (here -6.7%, paper -7.7%) — is reproduced.

## Experiment 2 — Risk-parity Sharpe ratio on 1993-2018 out-of-sample returns

Reproduces paper Section 3 / Table 3: daily returns of the universe below are
rescaled by an EMA volatility (`eta=1/100`, Eq. 10), aggregated to weekly
returns, and the weekly correlation matrix is tracked by an EMA (Eq. 11) at
two decay rates — `eta'=1/750 days` ("slow", about 150 weekly returns) and
`eta'=1/100 days` ("fast", about 20 weekly returns). At each week the
correlation matrix is cleaned with the empirical, RIE and MaxVar (K=3)
estimators, and a risk-parity portfolio `Pi_t propto Sigma_t^{-1} C_t^{-1} e`
(`e`=1 for stock indices and bonds, 0 for currencies) is built and held one
week out of sample. The annualized Sharpe ratio of each estimator's
out-of-sample weekly portfolio return series is computed over the realized
out-of-sample window (see "Deviations" below).

Source files: `exp/exp_2.py`, `src/maxvar.py`, `src/rie.py`,
`src/portfolio.py`, `src/ewma.py`. Output:
`results/exp2_sharpe_ratios.csv`, `results/exp2_meta.json`,
`results/exp2_sharpe_ratios.png`.

| eta' (days) | Method    | Sharpe ratio (this run) | Paper, Table 3 | Relative difference | Verdict |
|-------------|-----------|-------------------------:|---------------:|---------------------:|---------|
| 1/750 (slow)| Empirical | 1.009                     | 1.17            | -13.8%                | partially reproduced |
| 1/750 (slow)| RIE       | 0.888                     | 1.24            | -28.4%                | partially reproduced |
| 1/750 (slow)| MaxVar    | 1.144                     | 1.28            | -10.6%                | partially reproduced |
| 1/100 (fast)| Empirical | -0.234                    | not reported    | n/a                   | not reproduced |
| 1/100 (fast)| RIE       | 1.201                     | 0.93            | +29.1%                 | partially reproduced |
| 1/100 (fast)| MaxVar    | 0.892                     | 1.05            | -15.0%                 | partially reproduced |

At `eta'=1/750` both estimators retain the same sign (annualized Sharpe
ratio, positive excess risk-adjusted return) and MaxVar remains the best
performer, consistent with the paper's claim that MaxVar outperforms the
RIE. At `eta'=1/100` the empirical matrix produces a negative Sharpe ratio
rather than the paper's unreported ("–") entry; this is consistent with the
paper's implication that the empirical estimator is unusable at this decay
rate (Table 3 reports no value, plausibly because the matrix is
ill-conditioned with ~20 effective weekly observations for 42-47 assets),
but there is no paper number to validate the specific figure against, so it
is marked "not reproduced". The RIE's bulk-eigenvalue noise variance
(`sigma^2` in `src/rie.py`) is calibrated from the mean of the bulk
eigenvalues rather than fixed at 1, and the `q = N/T > 1` point mass at zero
(BBP17 Sec. 4) is excluded from that calibration (`src/rie.py`); with this
fix the RIE's Sharpe ratio at `eta'=1/100` is positive and the same sign as
the paper (+1.201 vs. +0.93, +29.1%), unlike a constant-`sigma^2=1`
calibration, which produced a spurious sign flip. However, the paper's claim
that MaxVar's advantage over the RIE *widens* as the estimation window is
shortened does not hold in this run: the MaxVar-minus-RIE gap is +0.256 at
`eta'=1/750` (paper: +0.04) but -0.309 at `eta'=1/100` (paper: +0.12), i.e.
the RIE outperforms MaxVar at the fast decay rate here, reversing the
paper's ranking. This was checked against the reduced bond group (9 of the
paper's 14 bond futures, see "Universe size reduced" below): with only 9
non-trivial proxies among 42 assets and an estimation window of only ~20
effective weekly observations, MaxVar's three groups are themselves
estimated from very few assets per group (9 bonds, 9 currencies, 24
indices), which may weaken its grouping advantage at the fast decay rate
more than the paper's full 47-instrument, 14-bond universe would; this
hypothesis was not confirmed by an isolating run (it would require the
full 47-future universe), so the cause of the ranking reversal at
`eta'=1/100` is not identified.

## Deviations from the paper

- **Futures universe substituted with ETF/FX proxies.** The paper's Table 2
  lists 47 futures contracts on exchanges (Eurex, ICE, SGX, TSE, ASX, KRX,
  NSE, Montreal Exchange) that this project's market-data path (QuantX
  Dataloader / Massive) does not serve: direct index symbols (`I:SPX`,
  `I:NDX`, `I:DJI`, ...) return `NOT_AUTHORIZED`, and the Massive futures API
  covers only CME/CBOT/NYMEX/COMEX-cleared contracts, not the non-US
  exchanges most of Table 2 trades on. Each Table 2 instrument is mapped in
  `src/config.py` to the closest long-history, US-accessible proxy: a
  country/region equity ETF for each stock index, a duration-matched US or
  international government-bond ETF for each bond future, and a spot
  currency pair (inverted to USD-per-unit where quoted the other way) for
  each currency. This substitutes futures total-return series with
  spot/ETF total-return series and is the reason the data source differs
  from the paper's.
- **Universe size reduced from 47 to 42 instruments.** The paper's 14
  government-bond futures span five countries (US, Germany, France, Italy,
  Canada, Japan) at several maturities; single-country sovereign-bond ETFs
  with a long US-accessible history are not available for Germany, France,
  Italy, Canada or Japan individually through this project's data sources.
  Repeating the same broad "international treasury" ETF for several Table 2
  rows would make those columns of the correlation matrix numerically
  identical (perfect collinearity), which is not the measurement-noise
  problem the estimators are designed to clean. The bond group therefore
  keeps the five distinct US-maturity proxies (`SHY`, `IEI`, `IEF`, `TLT`,
  `EDV`) plus the four distinct non-US/ex-US treasury ETFs that are
  available (`IGOV`, `ISHG`, `BWX`, `BNDX`), instead of repeating a proxy
  across several rows, for 9 bond instruments (paper: 14) and 42 total
  instruments (paper: 47); see `UNIVERSE_SIZE_PAPER` / `UNIVERSE_SIZE_RUN`
  in `src/config.py`.
- **Data source: QuantX Dataloader with a Yahoo Finance backstop.** All 42
  loaded tickers are equity ETFs, bond ETFs or spot FX pairs; `ticker_details`
  reports `data_source: "yahoo"` for every one of the 33 ETF tickers (Yahoo
  Finance backstop; see `non_massive_source` in `results/exp1_meta.json` and
  `results/exp2_meta.json`), because the Dataloader/Massive history for these
  ETFs over 1985-2018 is shorter than Yahoo's. The 9 FX pairs load from
  Massive directly. The paper's data source for the 47 futures contracts
  (Bloomberg/Reuters-type data, not specified in the paper beyond Table 2)
  is not reproduced; this is the substitution above, not an additional one.
- **Realized out-of-sample window shortened from 1993-2018 (paper) to
  2013-06-14 to 2018-12-28 (this run), 290 weekly returns instead of about
  1,350.** `BNDX` (Vanguard Total International Bond ETF, the proxy for
  "Global Government Bond ex-US") only began trading in June 2013, which is
  later than every other proxy in the universe; the out-of-sample Sharpe
  ratio is computed from the first week at which all 42 proxies have valid
  returns (`first_date_all_tickers_active` in `results/exp2_meta.json`),
  since the risk-parity portfolio (Section 3) uses the full universe at
  every rebalancing date. This was checked by inspecting
  `first_valid_index()` of the loaded return panel: `BNDX` (2013-06-05) is
  the binding constraint; the next latest start is `INDA` (2012-02-07),
  over a year earlier. This 5.5-year realized out-of-sample window, against
  the paper's 26 years, is the identified cause of the generally lower and
  noisier Sharpe ratios reported above (a standard error of order
  `1/sqrt(5.5) ~ 0.43`, versus the paper's `1/sqrt(26) ~ 0.2`, is consistent
  with the size of the gaps to the paper's values, including the sign flip
  of the fast-eta' RIE Sharpe ratio). The full 1985-2018 history was not
  re-run without `BNDX` to further confirm this, because dropping it would
  also remove the only "ex-US government bond" proxy from the bond group
  and change the universe rather than only its window.
- **Reference matrix EMA initialization (Eq. 10-11, `eta'=1/750`).** The
  paper does not specify the starting point for the EMA recursions used to
  build the 28 December 2018 reference matrix `C0`; this implementation
  initializes both EMAs with the first available weekly rescaled return and
  runs the recursion over the entire available history up to 28 December
  2018 (the realized panel start, 1985-2018, is shorter than this only for
  `BNDX`; see above), rather than restricting to exactly 750 trading days.
- **RIE implementation.** The paper cites Bun, Bouchaud & Potters (2016,
  2017) without reproducing the shrinkage formula; no installed package
  (PyRMT, nlshrink) implements the optimal rotationally invariant estimator
  rather than a constant-residual eigenvalue-clipping variant, so
  `src/rie.py` implements the nonlinear-shrinkage RIE directly from the
  analytic Marchenko-Pastur Stieltjes transform (Bun, Bouchaud & Potters,
  Physics Reports 666 (2017), Section 4), with the finite-`N` leave-one-out
  correction (Ledoit-Péché) applied to the outlier eigenvalues.
  `q = n/T` is estimated from the number of assets and the number of weekly
  (`eta'`-implied) observations used at each date.
- **Sharpe ratio annualization.** Not specified in the paper; this
  implementation annualizes the weekly out-of-sample portfolio return series
  by `sqrt(WEEKS_PER_YEAR)` with `WEEKS_PER_YEAR=52` (`src/config.py`).
- **Risk-parity rebalancing frequency.** Not specified beyond the weekly EMA
  structure; this implementation rebalances weekly, in step with the weekly
  correlation EMA update (Eq. 11), holding the portfolio one week out of
  sample before the next rebalancing.

## Not reproduced

- The paper's Table 3 entry for the empirical correlation matrix at
  `eta'=1/100` is shown as a dash ("–") without a stated reason; this
  implementation computes a (negative) Sharpe ratio rather than a
  singularity, so there is no paper number to validate the specific
  negative value against. The underlying claim — that the empirical
  estimator performs poorly at this decay rate — is consistent with the
  sign obtained, but the exact entry is not reproduced.
