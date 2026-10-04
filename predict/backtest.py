"""Walk-forward backtest: for each gameweek g of a test season, train on every earlier season
plus that season's gameweeks < g, and score g.

Compared against naive baselines and FPL's expected points (`ep_next`). For past seasons
`ep_next` is the community archive's `xP` summed per gameweek (timing of capture is not
documented). For the current season it is only known where ingest stored a pre-deadline
snapshot, so it stays empty until enough gameweeks have accumulated. The report says which.

    uv run python -m predict.backtest              # last two seasons -> docs/backtest.md
    uv run python -m predict.backtest 2025-26
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.storage import available_gameweeks, available_seasons, gw_path, read_parquet
from predict import features, model
from predict.data import season_features

FIRST_TEST_GW = 3  # gameweeks 1-2 have almost no in-season form to use
REGULAR_MINUTES = 60  # "regular" = averaged >= 60 minutes over the previous 5 gameweeks
METHODS = ["model", "baseline_l5", "baseline_season", "ep_next"]


def walk_forward(feats: pd.DataFrame, test_seasons: list[str]) -> pd.DataFrame:
    labelled = features.usable(feats).dropna(subset=["target"])
    parts = []
    for ts in test_seasons:
        earlier = labelled[labelled["season"] < ts]
        in_season = labelled[labelled["season"] == ts]
        for g in sorted(in_season["gameweek"].unique()):
            if g < FIRST_TEST_GW:
                continue
            train = pd.concat([earlier, in_season[in_season["gameweek"] < g]])
            test = in_season[in_season["gameweek"] == g]
            if train.empty or test.empty:
                continue
            booster = model.train(train, train["target"])
            parts.append(
                test.assign(
                    model=model.predict(booster, test),
                    baseline_l5=test["pts_l5"].fillna(0.0),
                    baseline_season=test["pts_season"].fillna(0.0),
                )[
                    ["season", "id", "gameweek", "target", "min_l5"]
                    + ["model", "baseline_l5", "baseline_season"]
                ]
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_ep_next(preds: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add `ep_next` where a players snapshot exists for that (season, gameweek)."""
    preds = preds.assign(ep_next=np.nan)
    for season in sorted(set(preds["season"])):
        stored = set(available_gameweeks(settings, season, "players"))
        for g in sorted(set(preds.loc[preds["season"] == season, "gameweek"]) & stored):
            ep = read_parquet(settings, gw_path(season, g, "players")).set_index("id")["ep_next"]
            mask = (preds["season"] == season) & (preds["gameweek"] == g)
            preds.loc[mask, "ep_next"] = preds.loc[mask, "id"].map(ep)
    return preds


def _metrics(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        err = df[m] - df["target"]
        rows.append(
            {
                "method": m,
                "MAE": err.abs().mean(),
                "RMSE": float(np.sqrt((err**2).mean())),
                "Spearman": df[m].corr(df["target"], method="spearman"),
                "rows": len(df),
            }
        )
    return pd.DataFrame(rows)


def summarise(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for season in sorted(set(preds["season"])):
        p = preds[preds["season"] == season]
        core = [m for m in METHODS if m != "ep_next"]
        regular_label = f"regular players (>= {REGULAR_MINUTES} min/game over last 5)"
        is_regular = p["min_l5"] >= REGULAR_MINUTES
        tables[f"{season}: all players"] = _metrics(p, core)
        tables[f"{season}: {regular_label}"] = _metrics(p[is_regular], core)
        # ep_next is only scored on rows that have it, so every method in these tables is
        # evaluated on exactly the same rows.
        if "ep_next" in p.columns and p["ep_next"].notna().any():
            has_ep = p["ep_next"].notna()
            tables[f"{season}: head-to-head with ep_next, all players"] = _metrics(
                p[has_ep], METHODS
            )
            tables[f"{season}: head-to-head with ep_next, {regular_label}"] = _metrics(
                p[has_ep & is_regular], METHODS
            )
    return tables


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(preds: pd.DataFrame) -> str:
    if preds.empty:
        return "# Backtest\n\nNot enough finished gameweeks to run a walk-forward backtest yet.\n"

    tables = summarise(preds)
    scope = ", ".join(
        f"{s} (gameweeks {int(g.min())}-{int(g.max())})"
        for s, g in preds.groupby("season")["gameweek"]
    )
    out = [
        "# Backtest results",
        "",
        f"Generated {datetime.date.today()}. Walk-forward: each gameweek is scored by a model "
        "trained only on earlier seasons plus earlier gameweeks of the same season. "
        f"Tested: {scope}; {len(preds)} player-rows. Lower MAE/RMSE and higher Spearman "
        "are better.",
        "",
    ]
    coverage = []
    for season, p in preds.groupby("season"):
        total = p["gameweek"].nunique()
        with_ep = p.loc[p["ep_next"].notna(), "gameweek"].nunique() if "ep_next" in p else 0
        coverage.append(f"{season}: {with_ep} of {total} tested gameweeks")
    out += [f"`ep_next` is available for: {'; '.join(coverage)}.", ""]
    for title, table in tables.items():
        out += [f"## {title}", "", _md_table(table), ""]

    out += ["## Verdict", ""]
    for title, table in tables.items():
        baseline = "ep_next" if "head-to-head" in title else None
        out += [_verdict(table, title, baseline)]
    if not (preds["ep_next"].notna().any() if "ep_next" in preds.columns else False):
        out += [
            "- FPL's `ep_next` benchmark is **not available** for the tested gameweeks. "
            "The model has therefore not been shown to beat `ep_next`.",
        ]
    out += [
        "",
        "## Caveats",
        "",
        "- For past seasons `ep_next` is the community archive's `xP`; when it was captured "
        "relative to the deadline is not documented, so it may not be a strict pre-deadline "
        "benchmark. If it was captured late (e.g. after team news), the head-to-head gap "
        "overstates FPL's real advantage. Only snapshots stored by our own ingest before a "
        "deadline give a trustworthy comparison.",
        "- The archive records some gameweeks' `xP` as zero for everyone; those gameweeks are "
        "treated as missing, not scored as zero.",
        "- Fixture difficulty and team strengths use end-of-season values for past gameweeks.",
        "- Position and team come from the latest snapshot of each season.",
        "- FPL scoring rules change between seasons (e.g. defensive-contribution points from "
        "2025-26), so older seasons' targets are not exactly comparable.",
        "- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.",
        "",
    ]
    return "\n".join(out)


def _verdict(table: pd.DataFrame, label: str, baseline: str | None = None) -> str:
    methods = [str(m) for m in table["method"]]
    mae = dict(zip(methods, (float(v) for v in table["MAE"]), strict=True))
    rho = dict(zip(methods, (float(v) for v in table["Spearman"]), strict=True))
    others = [m for m in methods if m != "model"] if baseline is None else [baseline]
    best_mae = min(others, key=lambda m: mae[m])
    best_rho = max(others, key=lambda m: rho[m])
    mae_verb = "beats" if mae["model"] < mae[best_mae] else "does NOT beat"
    rho_verb = "beats" if rho["model"] > rho[best_rho] else "does NOT beat"
    return (
        f"- **{label}**: on MAE the model {mae_verb} the best baseline (`{best_mae}`), "
        f"{mae['model']:.3f} vs {mae[best_mae]:.3f}; on Spearman it {rho_verb} "
        f"`{best_rho}`, {rho['model']:.3f} vs {rho[best_rho]:.3f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("test_seasons", nargs="*", help="default: the last two seasons stored")
    args = parser.parse_args()

    settings = get_settings()
    with_live = [s for s in available_seasons(settings) if available_gameweeks(settings, s, "live")]
    if not with_live:
        raise SystemExit("no data; run ingest first")
    test_seasons = args.test_seasons or with_live[-2:]
    feats = pd.concat([season_features(settings, s) for s in with_live], ignore_index=True)
    preds = attach_ep_next(walk_forward(feats, test_seasons), settings)
    report = render_report(preds)
    path = Path("docs/backtest.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
