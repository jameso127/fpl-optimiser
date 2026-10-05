"""Promotion gate: decide whether a freshly trained model may replace the serving one.

A challenger is judged on a recent holdout: gameweeks it was never trained on, scored
walk-forward (each gameweek by a model trained only on earlier data). It must beat the
simple baselines and FPL's own `ep_next`, must not be worse than the current model on the
gameweeks that are out-of-sample for both, and the leak audit must be clean. Every check, pass
or fail, is recorded in the model card so the decision can be audited later.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.backtest import REGULAR_MINUTES, TOP_K, top_k_points
from ml.registry import GateOutcome


@dataclass(frozen=True)
class GateConfig:
    min_holdout_gameweeks: int = 3  # too little evidence below this: do not promote
    min_ep_next_gameweeks: int = 3  # compare with ep_next only with this much overlap
    champion_tolerance: float = 0.10  # points per pick the challenger may trail the champion


def summarise_holdout(preds: pd.DataFrame) -> dict[str, Any]:
    """Decision metrics for each method on the holdout (predictions need `target`, `min_l5`)."""
    out: dict[str, Any] = {
        "gameweeks": sorted(int(g) for g in preds["gameweek"].unique()),
        "rows": len(preds),
    }
    regular = preds[preds["min_l5"] >= REGULAR_MINUTES]
    methods = [m for m in ("model", "baseline_l5", "baseline_season") if m in preds.columns]
    for m in methods:
        out[m] = {
            f"top{TOP_K}": top_k_points(preds, m),
            "spearman_regular": float(regular[m].corr(regular["target"], method="spearman")),
            "mae_regular": float((regular[m] - regular["target"]).abs().mean()),
        }
    if "ep_next" in preds.columns and preds["ep_next"].notna().any():
        has = preds[preds["ep_next"].notna()]
        out["vs_ep_next"] = {
            "gameweeks": sorted(int(g) for g in has["gameweek"].unique()),
            f"model_top{TOP_K}": top_k_points(has, "model"),
            f"ep_next_top{TOP_K}": top_k_points(has, "ep_next"),
        }
    return out


def evaluate_gate(
    summary: dict[str, Any],
    leak_status: str,
    champion_delta: float | None,
    config: GateConfig | None = None,
) -> GateOutcome:
    """`champion_delta` = challenger minus champion top-K points on gameweeks after the
    champion's training cutoff (None when there is no champion or no such gameweek)."""
    config = config or GateConfig()
    reasons: list[str] = []
    ok = True
    key = f"top{TOP_K}"

    if len(summary["gameweeks"]) < config.min_holdout_gameweeks:
        return GateOutcome(
            promoted=False,
            reasons=[f"FAIL: only {len(summary['gameweeks'])} holdout gameweeks, need "
                     f"{config.min_holdout_gameweeks}"],
        )  # fmt: skip

    if leak_status == "LEAK WARNING":
        ok = False
        reasons.append("FAIL: leak audit raised a warning, so ep_next-based evidence is invalid")
    else:
        reasons.append(f"pass: leak audit {leak_status}")

    model, base = summary["model"], summary["baseline_l5"]
    if model[key] < base[key]:
        ok = False
        reasons.append(f"FAIL: {key} {model[key]:.3f} below rolling-form baseline {base[key]:.3f}")
    else:
        reasons.append(f"pass: {key} {model[key]:.3f} >= rolling-form baseline {base[key]:.3f}")
    if model["spearman_regular"] < base["spearman_regular"]:
        ok = False
        reasons.append(
            f"FAIL: regular-player Spearman {model['spearman_regular']:.3f} below baseline "
            f"{base['spearman_regular']:.3f}"
        )
    else:
        reasons.append("pass: regular-player Spearman at or above the baseline")

    ep = summary.get("vs_ep_next")
    if ep is not None and len(ep["gameweeks"]) >= config.min_ep_next_gameweeks:
        if ep[f"model_{key}"] < ep[f"ep_next_{key}"]:
            ok = False
            reasons.append(
                f"FAIL: {key} {ep[f'model_{key}']:.3f} below FPL ep_next {ep[f'ep_next_{key}']:.3f}"
            )
        else:
            reasons.append(
                f"pass: {key} {ep[f'model_{key}']:.3f} >= FPL ep_next {ep[f'ep_next_{key}']:.3f}"
            )
    else:
        reasons.append("skipped: too few holdout gameweeks with a clean ep_next to compare")

    if champion_delta is None:
        reasons.append("skipped: no champion, or no gameweek out-of-sample for both models")
    elif champion_delta < -config.champion_tolerance:
        ok = False
        reasons.append(f"FAIL: {champion_delta:+.3f} {key} vs the serving model (tolerance "
                       f"{config.champion_tolerance})")  # fmt: skip
    else:
        reasons.append(f"pass: {champion_delta:+.3f} {key} vs the serving model")

    return GateOutcome(promoted=ok, reasons=reasons)
