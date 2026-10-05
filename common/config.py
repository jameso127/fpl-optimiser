from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_bucket: str | None = None
    data_dir: str = "./data"
    gameweek: int | None = None
    season: str | None = None  # e.g. "2026-27"; default: derived from the data

    # Model lifecycle: train once (job `train`), serve many times (job `predict`).
    model_name: str = "xpts-hurdle"
    holdout_gameweeks: int = 8  # recent finished gameweeks used to evaluate a new model
    git_sha: str | None = None  # stamped on model cards; set by CI/the image build

    # Where users (chat id, FPL team id, settings) live. "memory" is single-user mode: the one
    # user is the owner below. "firestore" holds everyone who registered with the bot.
    users_backend: Literal["firestore", "memory"] = "memory"
    gcp_project_id: str | None = None
    firestore_database: str = "(default)"

    telegram_bot_token: SecretStr | None = None  # a secret: from Secret Manager in production
    telegram_chat_id: int | None = None  # the owner's chat, used in single-user mode

    # Optimiser: whose team, how many transfers to consider, and how cautious to be.
    fpl_team_id: int | None = None
    max_transfers: int = 4  # evaluate 0..this many transfers
    bench_weight: float = 0.1  # how much the bench counts towards the objective (tie-break)
    min_gain_per_transfer: float = 0.5  # expected points a transfer must earn to be recommended
    max_free_transfers: int = 5  # free transfers can be banked up to this many
    free_transfers_override: int | None = None  # use when the estimate is wrong

    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    fpl_user_agent: str = "fpl-optimiser-demo/0.1"
    fpl_min_interval_seconds: float = 0.5

    # Community archive of past seasons' per-gameweek FPL data (vaastav/Fantasy-Premier-League).
    history_base_url: str = (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    )

    log_level: str = "INFO"

    @property
    def data_root(self) -> str:
        return f"gs://{self.data_bucket}" if self.data_bucket else self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
