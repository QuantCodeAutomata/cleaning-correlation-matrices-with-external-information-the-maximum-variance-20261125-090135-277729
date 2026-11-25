"""Tests for data/data_loader.py against the live QuantX Dataloader REST API.

Marked ``market_data`` and skipped when credentials are absent, per the
project's data policy (a reduced slice, cached outside the repository).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.market_data


def test_small_price_panel_has_expected_shape_and_tickers(small_price_panel):
    prices, tickers = small_price_panel
    assert isinstance(prices, pd.DataFrame)
    assert not prices.empty
    assert set(tickers).issubset(set(prices.columns))
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_small_price_panel_values_positive(small_price_panel):
    prices, tickers = small_price_panel
    values = prices[tickers].to_numpy()
    finite = values[~np.isnan(values)]
    assert finite.size > 0
    assert np.all(finite > 0), "prices must be strictly positive"


def test_small_price_panel_dates_within_requested_window(small_price_panel):
    prices, _ = small_price_panel
    assert prices.index.min() >= pd.Timestamp("2016-01-01")
    assert prices.index.max() <= pd.Timestamp("2017-12-31")
