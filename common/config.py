from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_bucket: str | None = None
    data_dir: str = "./data"
    dry_run: bool = False
    gameweek: int | None = None

    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    fpl_user_agent: str = "fpl-optimiser-demo/0.1"
    fpl_min_interval_seconds: float = 0.5

    log_level: str = "INFO"

    @property
    def data_root(self) -> str:
        return f"gs://{self.data_bucket}" if self.data_bucket else self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
