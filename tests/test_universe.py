"""Unit tests for src/universe.py."""

from __future__ import annotations

import numpy as np

from src.config import GROUPS
from src.universe import group_indices, include_mask, ordered_tickers


def test_ordered_tickers_deduplicated_and_covers_all_groups():
    tickers = ordered_tickers()
    assert len(tickers) == len(set(tickers)), "ticker list must be deduplicated"
    all_proxies = {t for universe in GROUPS.values() for t in universe.values()}
    assert set(tickers) == all_proxies


def test_group_indices_partition_the_universe():
    tickers = ordered_tickers()
    groups, names = group_indices()
    assert names == ["stock_indices", "bonds", "currencies"]
    covered = np.concatenate(groups)
    assert sorted(covered.tolist()) == list(range(len(tickers)))
    assert len(set(covered.tolist())) == len(tickers), "no ticker in two groups"


def test_include_mask_excludes_currencies():
    tickers = ordered_tickers()
    groups, names = group_indices()
    mask = include_mask(("stock_indices", "bonds"))
    currency_idx = groups[names.index("currencies")]
    assert np.all(mask[currency_idx] == 0.0)
    non_currency_idx = np.setdiff1d(np.arange(len(tickers)), currency_idx)
    assert np.all(mask[non_currency_idx] == 1.0)
