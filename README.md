# Cleaning correlation matrices with external information: the Maximum Variance estimator

Replication of selected experiments from Joel Bun, Jean-Philippe Bouchaud and
Marc Potters, "Cleaning correlation matrices with external information: the
Maximum Variance estimator".

This repository reproduces:

1. **Monte Carlo noise-reduction test** (paper, Section 3, Table 1): a known
   reference correlation matrix built with the MaxVar block structure is
   corrupted by sampling noise, and the Frobenius distance to the reference
   of the empirical, RIE-cleaned and MaxVar-cleaned correlation matrices is
   compared.
2. **Risk-parity Sharpe ratio** (paper, Section 3, Table 3): an out-of-sample
   risk-parity portfolio built on 47 futures contracts (Table 2: 24 stock
   indices, 14 government bonds, 9 currencies) is evaluated with the
   empirical, RIE and MaxVar correlation estimators, at two EMA parameters
   (eta'=1/750 and eta'=1/100), over 1993-2018.

## Layout

```
src/        MaxVar estimator, RIE, EWMA volatility/correlation, risk-parity
            portfolio, metrics, universe/group definitions, config
exp/        exp_1.py (Monte Carlo test), exp_2.py (risk-parity Sharpe ratio)
data/       data_loader.py: OHLCV bars from the QuantX Dataloader REST API
tests/      unit tests (synthetic fixtures) and market-data tests
results/    RESULTS.md, aggregate metric tables (.csv/.json), static plots (.png)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests only, no API key needed
python -m pytest                        # full suite, requires API keys
```

## Experiments

```bash
python -m exp.exp_1   # Monte Carlo noise-reduction test (Table 1)
python -m exp.exp_2   # risk-parity Sharpe ratio (Table 3)
```

Both experiments load the universe defined in `src/config.py` through
`data/data_loader.py` and write their outputs to `results/`. Re-running an
experiment overwrites its result files in `results/`; the comparison numbers
and verdicts in `results/RESULTS.md` are not updated automatically and must
be checked against the new files by hand.

## Data

Price history is loaded through the QuantX Dataloader REST API
(`POST /fetch-ohlcvs`), backed by Massive market data, with a Yahoo Finance
backstop for stock/ETF daily bars. Bars are dividend-adjusted. The paper's 47
futures contracts (Table 2) are not directly available through this data
path (most trade on exchanges the provider does not cover); `src/config.py`
documents the proxy instruments used instead and `results/RESULTS.md`
records the resulting deviations.

## Reproduce everything

```bash
bash reproduce.sh
```
