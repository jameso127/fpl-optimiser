"""Live monitoring: how did the predictions we actually served perform once results were in?

For every gameweek that has both a stored `predictions.parquet` and finished `live` stats,
compare the served expected points (`xpts`) with FPL's `ep_next` and with what happened. This
is the honest, out-of-sample record of the model in production (backtests are only a proxy)
and the trigger for noticing drift. Written to
`monitoring/season=<s>/live_performance.parquet` by the train job, which also warns if the
served model has lately been doing worse than FPL's own figure.
"""

import logging

import numpy as np
import pandas as pd

from common.config import Settings
from common.storage import available_gameweeks, gw_path, read_parquet, write_parquet

log = logging.getLogger(__name__)

TOP_K = 30
RECENT = 4  # gameweeks in the "lately" window


def _top_k(df: pd.DataFrame, col: str) -> float:
    return float(df.nlargest(TOP_K, col)["points"].mean())


def live_performance(settings: Settings, season: str) -> pd.DataFrame:
    """One row per gameweek with served predictions and results."""
    served = set(available_gameweeks(settings, season, "predictions"))
    finished = set(available_gameweeks(settings, season, "live"))
    rows = []
    for gw in sorted(served & finished):
        pred = read_parquet(settings, gw_path(season, gw, "predictions"))
        live = read_parquet(settings, gw_path(season, gw, "live"))[["id", "total_points"]]
        df = pred.merge(live.rename(columns={"total_points": "points"}), on="id", how="inner")
        df = df[df["fix_n"] >= 1]
        if df.empty:
            continue
        versions = df["model_version"].mode() if "model_version" in df else pd.Series(["?"])
        rows.append(
            {
                "season": season,
                "gameweek": gw,
                "model_version": str(versions.iloc[0]),
                "players": len(df),
                f"top{TOP_K}_xpts": _top_k(df, "xpts"),
                f"top{TOP_K}_ep_next": _top_k(df.fillna({"ep_next": 0.0}), "ep_next"),
                "spearman_xpts": float(df["xpts"].corr(df["points"], method="spearman")),
                "spearman_ep_next": float(df["ep_next"].corr(df["points"], method="spearman")),
                "mae_xpts": float((df["xpts"] - df["points"]).abs().mean()),
                "mae_ep_next": float((df["ep_next"] - df["points"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def check_recent(perf: pd.DataFrame) -> str | None:
    """A warning message if the served model has lately trailed FPL's own expected points."""
    if len(perf) < RECENT:
        return None
    last = perf.sort_values("gameweek").tail(RECENT)
    ours, theirs = last[f"top{TOP_K}_xpts"].mean(), last[f"top{TOP_K}_ep_next"].mean()
    if ours < theirs:
        return (
            f"served model trails FPL ep_next over the last {RECENT} gameweeks: "
            f"top-{TOP_K} points {ours:.2f} vs {theirs:.2f}"
        )
    return None


def write_live_performance(settings: Settings, season: str) -> pd.DataFrame:
    perf = live_performance(settings, season)
    if perf.empty:
        log.info("no served predictions with results yet", extra={"season": season})
        return perf
    write_parquet(settings, perf, f"monitoring/season={season}/live_performance.parquet")
    warning = check_recent(perf)
    if warning:
        log.warning(warning, extra={"season": season})
    last = perf.sort_values("gameweek").iloc[-1]
    log.info(
        "live performance",
        extra={
            "season": season,
            "gameweeks": len(perf),
            "latest_gameweek": int(last["gameweek"]),
            "mean_top_k_model": float(np.nan_to_num(perf[f"top{TOP_K}_xpts"].mean())),
            "mean_top_k_ep_next": float(np.nan_to_num(perf[f"top{TOP_K}_ep_next"].mean())),
        },
    )
    return perf
