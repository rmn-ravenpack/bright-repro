"""Binary nDCG@10 for a submitted passage ranking.

Gains are 1 for a passage in the gold set and 0 otherwise. The score uses the
ranking recorded after retrieval stops.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _as_set(ids: Iterable[str] | None) -> set[str]:
    return {str(item) for item in (ids or []) if str(item)}


def _dedupe(ranked: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in ranked if str(item)))


def ndcg_at(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float | None:
    rel = _as_set(relevant)
    if not rel:
        return None
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, docid in enumerate(_dedupe(ranked)[:k], start=1)
        if docid in rel
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg


def recall_at(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float | None:
    rel = _as_set(relevant)
    if not rel:
        return None
    return len(set(_dedupe(ranked)[:k]) & rel) / len(rel)


def mean(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)
