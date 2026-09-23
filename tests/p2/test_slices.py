from __future__ import annotations

from mosaic_rec.p2.slices import (
    age_bin,
    aggregate_slices,
    slice_memberships,
    support_bin,
    validate_joined_output,
)


def _row() -> dict[str, object]:
    return {
        "user_id": "u1",
        "query_timestamp": 10,
        "target_item_id": "i1",
        "target_rank": 1,
        "fit_interactions": 0,
        "item_age_days": 4.0,
        "interaction_quintile": 1,
        "interaction_decile": 1,
        "unique_user_quintile": 1,
        "popularity_tier": "tail",
        "history_length_quartile": 2,
        "user_recency_days": 45.0,
        "tail_share_quartile": 3,
        "niche_quartile": 1,
        "diversity_quartile": 4,
        "is_controlled_zero": True,
        "is_natural_zero": False,
        "is_warm": False,
        "ndcg_at_20": 1.0,
        "recall_at_20": 1.0,
        "mrr_at_20": 1.0,
        "hit_at_20": 1.0,
        "precision_at_20": 0.05,
    }


def test_frozen_support_and_age_boundaries() -> None:
    assert [support_bin(value) for value in (0, 1, 5, 6, 20, 21, 50, 51, 100, 101)] == [
        "0",
        "1-5",
        "1-5",
        "6-20",
        "6-20",
        "21-50",
        "21-50",
        "51-100",
        "51-100",
        "101+",
    ]
    assert [age_bin(value) for value in (7, 8, 30, 31, 90, 91, 365, 366)] == [
        "0-7d",
        "8-30d",
        "8-30d",
        "31-90d",
        "31-90d",
        "91-365d",
        "91-365d",
        ">365d",
    ]


def test_membership_and_aggregation_preserve_small_slice_flag() -> None:
    row = _row()
    assert ("support_x_item_age", "0|0-7d") in slice_memberships(row)
    table = aggregate_slices([row])
    records = table.to_pylist()
    controlled = next(
        value
        for value in records
        if value["slice_family"] == "primary_population"
        and value["slice"] == "controlled_zero"
    )
    assert controlled["n"] == 1
    assert controlled["minimum_n"] == 50
    assert not controlled["inferentially_eligible"]
    validate_joined_output([row])
