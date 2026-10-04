"""Walk-forward backtest: for each gameweek g of a test season, train on every earlier season
plus that season's gameweeks < g, and score g.

Compared against naive baselines and FPL's expected points (`ep_next`). For past seasons
`ep_next` is the community archive's `xP`, re-aligned to the gameweek it was known before (see
ingest/history.py; the raw column leaks the outcome). For the current season it is only known
where ingest stored a pre-deadline snapshot. The report says which gameweeks have it.

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
CORE_METHODS = ["model", "baseline_l5", "baseline_season"]
# model_xp = the model with FPL's ep_next as an extra feature. It is only compared on rows
# that have ep_next, so it appears in the head-to-head tables alongside ep_next itself.
METHODS = ["model", "model_xp", "baseline_l5", "baseline_season", "ep_next"]


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
            plain = model.train(train, train["target"], features.FEATURES)
            with_xp = model.train(train, train["target"], features.XP_FEATURES)
            parts.append(
                test.assign(
                    model=model.predict(plain, test),
                    model_xp=model.predict(with_xp, test),
                    baseline_l5=test["pts_l5"].fillna(0.0),
                    baseline_season=test["pts_season"].fillna(0.0),
                )[
                    ["season", "id", "gameweek", "target", "min_l5"]
                    + ["model", "model_xp", "baseline_l5", "baseline_season"]
                ]
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


LEAK_THRESHOLD = 0.1


def leak_audit(feats: pd.DataFrame) -> pd.DataFrame:
    """Check that each season's stored `ep_next` does not contain its own gameweek's outcome.

    A genuine pre-deadline forecast changes from g-1 to g in response to points scored in g-1,
    so its change should be uncorrelated with points scored in g. A value captured after the
    matches does the opposite. Run on regular players with a value in consecutive gameweeks.
    """
    rows = []
    for season, f in feats.groupby("season"):
        f = f.sort_values(["id", "gameweek"])
        g = f.groupby("id")
        d = f.assign(d_ep=f["ep_next"] - g["ep_next"].shift(1), prev=g["target"].shift(1))
        d = d[d["min_l5"] >= REGULAR_MINUTES].dropna(subset=["d_ep", "prev", "target"])
        if len(d) < 200:
            continue
        same = float(d["d_ep"].corr(d["target"]))
        rows.append(
            {
                "season": season,
                "rows": len(d),
                "corr_with_points_in_g": same,
                "corr_with_points_in_g-1": float(d["d_ep"].corr(d["prev"])),
                "status": "OK" if abs(same) < LEAK_THRESHOLD else "LEAK WARNING",
            }
        )
    return pd.DataFrame(rows)


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
        core = CORE_METHODS
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


def render_report(preds: pd.DataFrame, audit: pd.DataFrame | None = None) -> str:
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

    if audit is not None and not audit.empty:
        out += [
            "## Leak audit of stored `ep_next`",
            "",
            "Correlation of the change in `ep_next` (g-1 to g) with points scored in g. A "
            "pre-deadline value should show about 0 here and a clear positive correlation "
            f"with points in g-1. `LEAK WARNING` means |corr| >= {LEAK_THRESHOLD}: the value "
            "contains the outcome and every `ep_next` result in this report is invalid.",
            "",
            _md_table(audit),
            "",
        ]

    out += ["## Verdict", ""]
    for title, table in tables.items():
        if "head-to-head" in title:
            out += [
                _verdict(table, f"{title}: model_xp vs ep_next", "model_xp", ["ep_next"]),
                _verdict(
                    table,
                    f"{title}: model_xp vs the model without ep_next",
                    "model_xp",
                    ["model"],
                ),
                _verdict(table, f"{title}: model vs ep_next", "model", ["ep_next"]),
            ]
        else:
            out += [_verdict(table, title, "model", ["baseline_l5", "baseline_season"])]
    if not (preds["ep_next"].notna().any() if "ep_next" in preds.columns else False):
        out += [
            "- FPL's `ep_next` benchmark is **not available** for the tested gameweeks. "
            "The model has therefore not been shown to beat `ep_next`.",
        ]
    out += [
        "",
        "## Caveats",
        "",
        "- For past seasons `ep_next` comes from the community archive, not from snapshots "
        "we stored before deadlines, and its exact capture time is still undocumented beyond "
        "what the leak audit shows. Snapshots stored by our own ingest are the gold standard "
        "and will replace it as they accumulate.",
        "- The archive records some gameweeks' `xP` as zero for everyone; those gameweeks are "
        "treated as missing, not scored as zero. Single-player values above 20 are also "
        "treated as missing (data errors).",
        "- The archive's `xP` on row g was captured after gameweek g (it leaked: its change "
        "tracked points scored in g). It is therefore stored against gameweek g+1, as the value "
        "known before that deadline; the leak audit above re-checks this on every run. An "
        "earlier version of this report used the same-gameweek value and wrongly showed "
        "`ep_next` as a very strong predictor.",
        "- `model_xp` uses `ep_next` as a feature. Live, `ep_next` comes from our own "
        "snapshot, taken at ingest time while that gameweek is the next one, so it is a "
        "genuine pre-deadline value.",
        "- Fixture difficulty and team strengths use end-of-season values for past gameweeks.",
        "- Position and team come from the latest snapshot of each season.",
        "- FPL scoring rules change between seasons (e.g. defensive-contribution points from "
        "2025-26), so older seasons' targets are not exactly comparable.",
        "- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.",
        "",
    ]
    return "\n".join(out)


def _verdict(table: pd.DataFrame, label: str, subject: str, rivals: list[str]) -> str:
    """One line: does `subject` beat the best of `rivals` on MAE and on Spearman?"""
    methods = [str(m) for m in table["method"]]
    mae = dict(zip(methods, (float(v) for v in table["MAE"]), strict=True))
    rho = dict(zip(methods, (float(v) for v in table["Spearman"]), strict=True))
    best_mae = min(rivals, key=lambda m: mae[m])
    best_rho = max(rivals, key=lambda m: rho[m])
    mae_verb = "beats" if mae[subject] < mae[best_mae] else "does NOT beat"
    rho_verb = "beats" if rho[subject] > rho[best_rho] else "does NOT beat"
    return (
        f"- **{label}**: on MAE `{subject}` {mae_verb} `{best_mae}`, "
        f"{mae[subject]:.3f} vs {mae[best_mae]:.3f}; on Spearman it {rho_verb} "
        f"`{best_rho}`, {rho[subject]:.3f} vs {rho[best_rho]:.3f}."
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
    report = render_report(preds, leak_audit(feats))
    path = Path("docs/backtest.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
