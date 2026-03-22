"""Application configuration loaded from environment and YAML."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_yaml_config() -> dict[str, Any]:
    config_dir = PROJECT_ROOT / "config"
    base = config_dir / "settings.yaml"
    local = config_dir / "settings.local.yaml"

    config: dict[str, Any] = {}
    if base.exists():
        config = yaml.safe_load(base.read_text()) or {}
    if local.exists():
        local_config = yaml.safe_load(local.read_text()) or {}
        config = _deep_merge(config, local_config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_yaml = _load_yaml_config()


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8",
        populate_by_name=True, extra="ignore",
    )
    name: str = _yaml.get("exchange", {}).get("name", "binance")
    testnet: bool = Field(
        default=_yaml.get("exchange", {}).get("testnet", True),
        alias="BINANCE_TESTNET",
    )
    api_key: str = Field(default="", alias="BINANCE_API_KEY")
    api_secret: str = Field(default="", alias="BINANCE_API_SECRET")


class TradingConfig(BaseSettings):
    base_currency: str = _yaml.get("trading", {}).get("base_currency", "USDT")
    symbols: list[str] = _yaml.get("trading", {}).get("symbols", ["BTCUSDT"])
    interval: str = _yaml.get("trading", {}).get("interval", "1h")
    max_open_positions: int = _yaml.get("trading", {}).get("max_open_positions", 3)
    position_size_pct: float = _yaml.get("trading", {}).get("position_size_pct", 0.1)


class RiskConfig(BaseSettings):
    stop_loss_pct: float = _yaml.get("risk", {}).get("stop_loss_pct", 0.03)
    take_profit_pct: float = _yaml.get("risk", {}).get("take_profit_pct", 0.06)
    max_daily_loss_pct: float = _yaml.get("risk", {}).get("max_daily_loss_pct", 0.05)
    trailing_stop: bool = _yaml.get("risk", {}).get("trailing_stop", False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8",
        populate_by_name=True, extra="ignore",
    )
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/trades.db", alias="DATABASE_URL"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()
