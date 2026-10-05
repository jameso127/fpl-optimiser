from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_bucket: str | None = None
    data_dir: str = "./data"
    dry_run: bool = False
    gameweek: int | None = None
    season: str | None = None  # e.g. "2026-27"; default: derived from the data

    # Model lifecycle: train once (job `train`), serve many times (job `predict`).
    model_name: str = "xpts-hurdle"
    holdout_gameweeks: int = 8  # recent finished gameweeks used to evaluate a new model
    git_sha: str | None = None  # stamped on model cards; set by CI/the image build

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
        # Dry runs never touch the real bucket or real local data: fixture data would overwrite it.
        if self.dry_run:
            return f"{self.data_dir.rstrip('/')}/dry_run"
        return f"gs://{self.data_bucket}" if self.data_bucket else self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
