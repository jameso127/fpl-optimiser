"""Walk-forward backtest: for each gameweek g, train on gameweeks < g and score g.

Compared against naive baselines and, where available, FPL's own `ep_next`. `ep_next` is only
known for gameweeks where ingest ran before the deadline and stored a players snapshot, so
the comparison stays empty until enough gameweeks have accumulated. The report states this
rather than hiding it.

    uv run python -m predict.backtest      # writes docs/backtest.md from ./data
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.storage import available_gameweeks, gw_path, read_parquet
from predict import features, model
from predict.main import load_live

FIRST_TEST_GW = 3  # gameweeks 1-2 leave almost nothing to train on
REGULAR_MINUTES = 60  # "regular" = averaged >= 60 minutes over the previous 5 gameweeks
METHODS = ["model", "baseline_l5", "baseline_season", "ep_next"]


def walk_forward(feats: pd.DataFrame, last_gw: int) -> pd.DataFrame:
    parts = []
    for g in range(FIRST_TEST_GW, last_gw + 1):
        train = features.usable(feats[feats["gameweek"] < g]).dropna(subset=["target"])
        test = features.usable(feats[feats["gameweek"] == g]).dropna(subset=["target"])
        if train.empty or test.empty:
            continue
        booster = model.train(train, train["target"])
        parts.append(
            test.assign(
                model=model.predict(booster, test),
                baseline_l5=test["pts_l5"].fillna(0.0),
                baseline_season=test["pts_season"].fillna(0.0),
            )[["id", "gameweek", "target", "min_l5", "model", "baseline_l5", "baseline_season"]]
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_ep_next(preds: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add FPL's ep_next for gameweeks that have a stored pre-deadline snapshot."""
    preds = preds.assign(ep_next=np.nan)
    for g in sorted(set(preds["gameweek"]) & set(available_gameweeks(settings, "players"))):
        ep = read_parquet(settings, gw_path(g, "players")).set_index("id")["ep_next"]
        mask = preds["gameweek"] == g
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
    methods = [m for m in METHODS if m in preds.columns and preds[m].notna().any()]
    tables = {
        "All players": _metrics(preds, methods),
        f"Regular players (>= {REGULAR_MINUTES} min/game over last 5)": _metrics(
            preds[preds["min_l5"] >= REGULAR_MINUTES], methods
        ),
    }
    if "ep_next" in preds.columns and preds["ep_next"].notna().any():
        both = preds[preds["ep_next"].notna()]
        tables["Head-to-head with ep_next (gameweeks with a stored snapshot)"] = _metrics(
            both, methods
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
    gws = sorted(set(preds["gameweek"]))
    out = [
        "# Backtest results",
        "",
        f"Generated {datetime.date.today()}. Walk-forward: each gameweek is scored by a model "
        f"trained only on earlier gameweeks. Tested gameweeks: {gws[0]}-{gws[-1]} "
        f"({len(gws)} gameweeks, {len(preds)} player-rows). Lower MAE/RMSE and higher Spearman "
        "are better.",
        "",
    ]
    for title, table in tables.items():
        out += [f"## {title}", "", _md_table(table), ""]

    out += ["## Verdict", ""]
    for title, table in tables.items():
        baseline = "ep_next" if title.startswith("Head-to-head") else None
        out += [_verdict(table, title, baseline)]
    if not (preds["ep_next"].notna().any() if "ep_next" in preds.columns else False):
        out += [
            "- FPL's `ep_next` benchmark is **not yet available**: it can only be captured "
            "before a deadline and no tested gameweek has a stored snapshot. The model has "
            "therefore not been shown to beat `ep_next`.",
        ]
    out += [
        "",
        "## Caveats",
        "",
        "- Tiny sample (current season only, few gameweeks), so differences are noisy.",
        "- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.",
        "- Fixture difficulty and team strengths use today's values for past gameweeks.",
        "- Position and team come from the latest snapshot.",
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
    settings = get_settings()
    snapshots = available_gameweeks(settings, "players")
    live_gws = available_gameweeks(settings, "live")
    if not snapshots or not live_gws:
        raise SystemExit("no data; run ingest first")
    latest = snapshots[-1]
    last_gw = max(live_gws)
    players = read_parquet(settings, gw_path(latest, "players"))
    teams = read_parquet(settings, gw_path(latest, "teams"))
    fixtures = read_parquet(settings, gw_path(latest, "fixtures"))
    live = load_live(settings, before=last_gw + 1)
    feats = features.build_features(live, players, teams, fixtures, list(range(1, last_gw + 1)))
    preds = attach_ep_next(walk_forward(feats, last_gw), settings)
    report = render_report(preds)
    path = Path("docs/backtest.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
