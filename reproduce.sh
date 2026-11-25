#!/usr/bin/env bash
# Reproduce the full replication: unit tests, experiments, and the market-data
# tests (the latter require DATALOADER_API_KEY and MASSIVE_API_KEY).
set -uo pipefail

cd "$(dirname "$0")"

echo "=== Unit tests (no API key required) ==="
python -m pytest -m "not market_data"

echo "=== Experiment 1: Monte Carlo noise-reduction test (Table 1) ==="
python -m exp.exp_1

echo "=== Experiment 2: risk-parity Sharpe ratio (Table 3) ==="
python -m exp.exp_2

echo "=== Market-data tests ==="
python -m pytest -m "market_data"
