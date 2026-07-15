from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_FILENAME = "model.joblib"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "qoldauflow"
    postgres_password: str = "change_me"
    postgres_db: str = "qoldauflow"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    confidence_threshold: float = 0.5

    artifacts_dir: Path = BASE_DIR / "artifacts"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
