from typing import Any

import pandas as pd

from ml import gate


def _summary(
    model_top: float = 5.0, base_top: float = 4.0, ep: tuple[float, float] | None = (5.0, 4.0)
) -> dict[str, Any]:
    s: dict[str, Any] = {
        "gameweeks": [3, 4, 5, 6, 7, 8],
        "rows": 600,
        "model": {"top30": model_top, "spearman_regular": 0.3, "mae_regular": 2.0},
        "baseline_l5": {"top30": base_top, "spearman_regular": 0.2, "mae_regular": 2.5},
    }
    if ep is not None:
        s["vs_ep_next"] = {
            "gameweeks": [3, 4, 5, 6],
            "model_top30": ep[0],
            "ep_next_top30": ep[1],
        }
    return s


def test_a_model_that_beats_everything_is_promoted_and_every_check_is_recorded() -> None:
    outcome = gate.evaluate_gate(_summary(), "OK", champion_delta=0.2)

    assert outcome.promoted
    assert all(r.startswith("pass") for r in outcome.reasons)
    assert len(outcome.reasons) == 5  # leak, top-k, spearman, ep_next, champion


def test_a_leak_warning_blocks_promotion() -> None:
    outcome = gate.evaluate_gate(_summary(), "LEAK WARNING", None)

    assert not outcome.promoted and any("leak audit" in r for r in outcome.reasons)


def test_losing_to_the_rolling_form_baseline_blocks_promotion() -> None:
    outcome = gate.evaluate_gate(_summary(model_top=3.5, base_top=4.0), "OK", None)

    assert not outcome.promoted and any("below rolling-form baseline" in r for r in outcome.reasons)


def test_losing_to_fpls_expected_points_blocks_promotion() -> None:
    outcome = gate.evaluate_gate(_summary(ep=(3.9, 4.2)), "OK", None)

    assert not outcome.promoted and any("below FPL ep_next" in r for r in outcome.reasons)


def test_ep_next_comparison_is_skipped_not_failed_when_there_is_little_overlap() -> None:
    summary = _summary()
    summary["vs_ep_next"]["gameweeks"] = [3, 4]
    outcome = gate.evaluate_gate(summary, "OK", None)

    assert outcome.promoted and any(
        r.startswith("skipped") and "ep_next" in r for r in outcome.reasons
    )


def test_a_clearly_worse_model_than_the_serving_one_is_not_promoted() -> None:
    assert not gate.evaluate_gate(_summary(), "OK", champion_delta=-0.5).promoted
    assert gate.evaluate_gate(_summary(), "OK", champion_delta=-0.05).promoted  # within tolerance
    assert gate.evaluate_gate(_summary(), "OK", champion_delta=None).promoted  # nothing to compare


def test_too_little_holdout_evidence_never_promotes() -> None:
    summary = _summary()
    summary["gameweeks"] = [3, 4]
    outcome = gate.evaluate_gate(summary, "OK", None)

    assert not outcome.promoted and "holdout gameweeks" in outcome.reasons[0]


def test_holdout_summary_computes_the_decision_metrics() -> None:
    rng_rows = []
    for gw in (3, 4, 5):
        for pid in range(1, 41):
            rng_rows.append(
                {
                    "season": "2025-26", "gameweek": gw, "id": pid, "min_l5": 90.0,
                    "target": float(pid % 7), "model": float(pid % 7),  # a perfect model
                    "baseline_l5": float(pid % 3), "baseline_season": float(pid % 3),
                    "ep_next": float(pid % 5) if gw >= 4 else None,
                }
            )  # fmt: skip
    summary = gate.summarise_holdout(pd.DataFrame(rng_rows))

    assert summary["gameweeks"] == [3, 4, 5]
    assert summary["model"]["top30"] > summary["baseline_l5"]["top30"]
    assert summary["model"]["spearman_regular"] > 0.99
    assert summary["vs_ep_next"]["gameweeks"] == [4, 5]  # ep_next only where it exists
    assert summary["vs_ep_next"]["model_top30"] > summary["vs_ep_next"]["ep_next_top30"]
