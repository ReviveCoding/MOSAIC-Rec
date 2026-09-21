"""Event-time FIT negative sampling shared by BPR and graph baselines."""

from __future__ import annotations

import bisect
import random
from collections import defaultdict


def time_safe_negatives(
    fit: list[tuple[int, int, int]], seed: int
) -> tuple[list[int], list[int], list[int], dict[int, set[int]], set[int]]:
    first_fit: dict[int, int] = {}
    for stamp, _, item in fit:
        first_fit[item] = min(stamp, first_fit.get(item, stamp))
    warm = set(first_fit)
    available = sorted(warm, key=lambda item: (first_fit[item], item))
    available_times = [first_fit[item] for item in available]
    seen: dict[int, set[int]] = defaultdict(set)
    rng = random.Random(seed)
    training_users, training_items, negatives = [], [], []
    for stamp, user, item in fit:
        limit = bisect.bisect_right(available_times, stamp)
        excluded = seen[user] | {item}
        if len(excluded) < limit:
            negative = available[rng.randrange(limit)]
            while negative in excluded:
                negative = available[rng.randrange(limit)]
            training_users.append(user)
            training_items.append(item)
            negatives.append(negative)
        seen[user].add(item)
    if not training_items:
        raise ValueError("No time-eligible FIT negative samples")
    return training_users, training_items, negatives, seen, warm
