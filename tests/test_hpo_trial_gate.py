"""HPO orchestration must reject invalid model evidence before the next GPU job."""

from scripts.run_hpo_p1 import valid_overall_metric


def test_overall_only_and_slice_metrics() -> None:
    assert valid_overall_metric({"ndcg_at_20": 0.2, "validation_users": 41})
    assert valid_overall_metric(
        {"validation_slices": {"all": {"ndcg_at_20": 0.2, "n": 41}}}
    )
    assert not valid_overall_metric(
        {"ndcg_at_20": float("nan"), "validation_users": 41}
    )
    assert not valid_overall_metric({"validation_slices": None, "ndcg_at_20": 0.2})
