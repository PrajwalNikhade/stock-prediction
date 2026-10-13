"""Application settings loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# --- API Keys ---
APIFY_API_TOKEN = _get("APIFY_API_TOKEN")
APIFY_ACTOR_ID = _get("APIFY_ACTOR_ID", "apidojo/tweet-scraper")
APIFY_MAX_ITEMS = int(_get("APIFY_MAX_ITEMS", "1000"))
APIFY_TIMEOUT_SECONDS = int(_get("APIFY_TIMEOUT_SECONDS", "180"))

TAVILY_API_KEY = _get("TAVILY_API_KEY")
EXA_API_KEY = _get("EXA_API_KEY")

# --- Database ---
DATABASE_PATH = PROJECT_ROOT / _get("DATABASE_PATH", "data/crowdwisdom.db")

# --- Defaults ---
DEFAULT_ASSET = _get("DEFAULT_ASSET", "BTCUSDT")
DEFAULT_TIMEFRAME = _get("DEFAULT_TIMEFRAME", "1d")
HISTORY_DAYS = int(_get("HISTORY_DAYS", "180"))

# --- Sentiment ---
SENTIMENT_MODEL = _get("SENTIMENT_MODEL", "hybrid")
ENABLE_TRANSFORMER_SENTIMENT = _get("ENABLE_TRANSFORMER_SENTIMENT", "true").lower() == "true"

# --- ML ---
RANDOM_STATE = int(_get("RANDOM_STATE", "42"))

# --- Trading ---
TRANSACTION_COST_BPS = int(_get("TRANSACTION_COST_BPS", "10"))


def load_asset_config() -> dict:
    """Load asset configuration from YAML."""
    config_path = Path(__file__).parent / "assets.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_assets() -> list[dict]:
    """Return flat list of all asset configurations."""
    config = load_asset_config()
    assets = []
    for asset_class, items in config.get("asset_classes", {}).items():
        for item in items:
            item["asset_class"] = asset_class
            assets.append(item)
    return assets


def get_asset_by_ticker(ticker: str) -> dict | None:
    """Find asset config by ticker."""
    for asset in get_all_assets():
        if asset["ticker"] == ticker:
            return asset
    return None


def api_status() -> dict:
    """Return availability status of each API (no secrets revealed)."""
    return {
        "apify": bool(APIFY_API_TOKEN),
        "tavily": bool(TAVILY_API_KEY),
        "exa": bool(EXA_API_KEY),
    }
