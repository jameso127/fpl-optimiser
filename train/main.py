"""Train job: build the training set, evaluate on a recent holdout, fit, register, maybe promote.

Run on its own schedule (not inside the per-gameweek predict step). It:

0. records how the predictions served so far performed against results and `ep_next`;
1. builds features for every stored season and runs the leak audit on stored `ep_next`;
2. scores the last N finished gameweeks walk-forward (each by a model trained before it) and
   compares with baselines, FPL's `ep_next` and the model currently serving;
3. fits the final model on all labelled data;
4. writes it as a new immutable version with a model card (data hash, features, parameters,
   library versions, git SHA, evidence);
5. promotes it to `latest` only if the promotion gate passes. Otherwise the serving model is
   unchanged and the reasons are in the card.

    uv run python -m train.main
"""

import logging

import pandas as pd

from common.config import Settings, get_settings
from common.logging import configure_logging
from ml import backtest, features, gate, model, monitor, registry
from ml.data import all_season_features
from ml.registry import ModelCard

log = logging.getLogger(__name__)


def _holdout_gameweeks(labelled: pd.DataFrame, n: int) -> tuple[str, set[int]]:
    season = str(labelled["season"].max())
    gws = sorted(labelled.loc[labelled["season"] == season, "gameweek"].unique())
    usable = [int(g) for g in gws if g >= backtest.FIRST_TEST_GW]
    return season, set(usable[-n:])


def _leak_status(audit: pd.DataFrame) -> str:
    if audit.empty:
        return "n/a"
    return "LEAK WARNING" if (audit["status"] == "LEAK WARNING").any() else "OK"


def _champion_delta(settings: Settings, feats: pd.DataFrame, preds: pd.DataFrame) -> float | None:
    """Challenger minus serving model, in top-K points, on gameweeks after the serving model's
    training cutoff (the only ones that are out-of-sample for both)."""
    if preds.empty:
        return None
    try:
        champion, card = registry.load(settings, settings.model_name)
    except FileNotFoundError:
        return None
    after = preds[
        (preds["season"] > card.trained_through_season)
        | (
            (preds["season"] == card.trained_through_season)
            & (preds["gameweek"] > card.trained_through_gameweek)
        )
    ]
    if after.empty:
        return None
    rows = after.merge(
        feats[["season", "id", "gameweek", *card.features]],
        on=["season", "id", "gameweek"],
        how="left",
        suffixes=("", "_feat"),
    )
    try:
        registry.check_schema(card, rows)
    except ValueError as exc:
        log.warning("serving model incompatible with current features: %s", exc)
        return None
    rows["champion"] = model.expected_points(champion, rows)
    return backtest.top_k_points(rows, "model") - backtest.top_k_points(rows, "champion")


def run(settings: Settings) -> ModelCard:
    feats = all_season_features(settings)
    labelled = features.usable(feats).dropna(subset=["target"])
    season, hold = _holdout_gameweeks(labelled, settings.holdout_gameweeks)
    log.info(
        "train start",
        extra={"season": season, "holdout": sorted(hold), "rows": len(labelled)},
    )
    # How did the predictions we actually served do? (Written for dashboards and drift checks.)
    monitor.write_live_performance(settings, season)

    audit = backtest.leak_audit(feats)
    preds = backtest.walk_forward(feats, [season], only_gameweeks=hold, with_v1=False)
    preds = backtest.attach_ep_next(preds, settings) if not preds.empty else preds
    summary = gate.summarise_holdout(preds) if not preds.empty else {"gameweeks": [], "rows": 0}

    final = model.train_v2(labelled)
    version = registry.make_version(settings.git_sha or "unknown")
    card = registry.build_card(
        settings,
        version,
        labelled,
        final,
        metrics={"holdout": summary, "leak_audit": audit.to_dict("records")},
    )
    registry.save(settings, final, card, preds if not preds.empty else None)

    outcome = gate.evaluate_gate(
        summary, _leak_status(audit), _champion_delta(settings, feats, preds)
    )
    card.gate = outcome
    registry.update_card(settings, card)
    if outcome.promoted:
        registry.promote(settings, card.name, card.version)
    for reason in outcome.reasons:
        log.info("gate", extra={"version": version, "reason": reason})
    log.info(
        "train done",
        extra={"version": version, "promoted": outcome.promoted, "train_rows": card.train_rows},
    )
    return card


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    run(settings)


if __name__ == "__main__":
    main()
