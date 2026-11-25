"""Shared pytest fixtures.

Market-data tests load a reduced slice of the universe through
``data/data_loader.py`` (the project's single data-loading path) and are
skipped when ``DATALOADER_API_KEY``/``MASSIVE_API_KEY`` are not set.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that call the QuantX Dataloader REST API"
    )


@pytest.fixture(scope="session")
def market_data_available() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def small_price_panel(request: pytest.FixtureRequest, market_data_available: bool):
    """A few tickers of the universe, two years of daily closes.

    ``data/data_loader.py`` already caches the downloaded bars under
    ``/tmp/maxvar_data_cache`` (outside the repository), so repeated test
    sessions do not re-download them; ``request.config.cache`` only records
    that the slice was loaded in this session.
    """
    if not market_data_available:
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")

    from data.data_loader import load_price_panel

    tickers = ["SPY", "TLT", "C:EURUSD"]
    start, end = "2016-01-01", "2017-12-31"
    prices, meta = load_price_panel(tickers, start, end, invert=set(), use_cache=True)
    request.config.cache.set("small_price_panel/loaded", True)
    return prices, tickers
