"""Single source of truth for the asset universe and its K=3 group partition
(paper, Section 3: stock indices, government bonds, currencies), shared by
both experiments so the ticker order and group indices used to clean a
correlation matrix are never redefined in an experiment script.
"""

from __future__ import annotations

import numpy as np

from src.config import GROUPS


def ordered_tickers() -> list[str]:
    """Deduplicated ticker list in group order (stock indices, bonds,
    currencies), the fixed column order used for every correlation matrix."""
    tickers: list[str] = []
    seen: set[str] = set()
    for universe in GROUPS.values():
        for ticker in universe.values():
            if ticker not in seen:
                tickers.append(ticker)
                seen.add(ticker)
    return tickers


def group_indices() -> tuple[list[np.ndarray], list[str]]:
    """Index arrays into ``ordered_tickers()`` for each of the K groups.

    Returns ``(groups, group_names)`` where ``groups[k]`` holds the positions
    of group ``k``'s tickers in the shared ticker order. Each ticker belongs
    to exactly one group, since ``GROUPS`` partitions the universe by asset
    class (no ticker is reused across stock indices, bonds and currencies).
    """
    tickers = ordered_tickers()
    position = {ticker: i for i, ticker in enumerate(tickers)}
    groups, names = [], []
    for name, universe in GROUPS.items():
        idx = sorted({position[t] for t in universe.values()})
        groups.append(np.array(idx, dtype=int))
        names.append(name)
    covered = sum(len(g) for g in groups)
    assert covered == len(tickers), "GROUPS must partition the shared ticker list"
    return groups, names


def include_mask(include_group_names: tuple[str, ...]) -> np.ndarray:
    """Vector e: 1 for tickers in ``include_group_names``, 0 otherwise
    (paper, Section 3: e includes stock indices and bonds, excludes currencies).
    """
    tickers = ordered_tickers()
    groups, names = group_indices()
    mask = np.zeros(len(tickers))
    for name, idx in zip(names, groups):
        if name in include_group_names:
            mask[idx] = 1.0
    return mask
