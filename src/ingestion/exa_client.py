"""Exa news search client for CrowdWisdomTrading."""

import hashlib
import requests
import pandas as pd
from config.settings import EXA_API_KEY
from src.utils.logging_config import logger
from src.utils.time_utils import format_utc, utc_now, parse_timestamp

EXA_SEARCH_URL = "https://api.exa.ai/search"


def is_available() -> bool:
    """Check if Exa API key is configured."""
    return bool(EXA_API_KEY and EXA_API_KEY.strip())


def test_exa_connection(query: str = "Bitcoin", max_results: int = 2) -> dict:
    """Perform a small search request and verify actual results, titles, URLs, and dates."""
    if not is_available():
        return {
            "success": False,
            "configured": False,
            "articles_retrieved": 0,
            "sample_article": None,
            "error": "EXA_API_KEY is not configured in .env",
        }

    try:
        headers = {
            "x-api-key": EXA_API_KEY.strip(),
            "Content-Type": "application/json",
        }
        payload = {
            "query": f"{query} news",
            "numResults": max_results,
            "useAutoprompt": True,
            "type": "neural",
            "contents": {
                "text": {"maxCharacters": 500}
            },
        }
        resp = requests.post(EXA_SEARCH_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 401 or resp.status_code == 403:
            return {
                "success": False,
                "configured": True,
                "articles_retrieved": 0,
                "sample_article": None,
                "error": f"Invalid Exa API key (HTTP {resp.status_code})",
            }
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        if not results:
            return {
                "success": False,
                "configured": True,
                "articles_retrieved": 0,
                "sample_article": None,
                "error": f"Exa returned 0 results for query '{query}'",
            }

        first = results[0]
        text_content = first.get("text") or first.get("snippet") or ""
        sample = {
            "title": first.get("title", ""),
            "url": first.get("url", ""),
            "published_date": first.get("publishedDate") or first.get("published_date") or "N/A",
            "content_length": len(text_content),
        }

        return {
            "success": True,
            "configured": True,
            "articles_retrieved": len(results),
            "sample_article": sample,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "configured": True,
            "articles_retrieved": 0,
            "sample_article": None,
            "error": f"Exa request failed: {type(e).__name__}: {str(e)}",
        }


def search_news(asset_ticker: str, search_terms: list[str],
                max_results: int = 10) -> pd.DataFrame:
    """Search recent news using Exa API."""
    if not is_available():
        logger.warning("Exa API key not configured. Set EXA_API_KEY in .env")
        return pd.DataFrame()

    all_articles = []
    query = f"{search_terms[0]} latest news" if search_terms else f"{asset_ticker} latest news"

    try:
        headers = {
            "x-api-key": EXA_API_KEY.strip(),
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "numResults": max_results,
            "useAutoprompt": True,
            "type": "neural",
            "contents": {
                "text": {"maxCharacters": 500}
            },
        }
        resp = requests.post(EXA_SEARCH_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        retrieved_at = format_utc(utc_now())
        for item in data.get("results", []):
            url = item.get("url", "")
            news_id = hashlib.md5(url.encode()).hexdigest() if url else hashlib.md5(
                (item.get("title", "") + retrieved_at).encode()
            ).hexdigest()

            published = item.get("publishedDate") or item.get("published_date")
            pub_ts = parse_timestamp(published) if published else None

            all_articles.append({
                "news_id": news_id,
                "asset": asset_ticker,
                "title": str(item.get("title", "")).strip(),
                "snippet": str(item.get("text") or item.get("snippet") or "")[:500].strip(),
                "source_name": str(item.get("source", "")).strip(),
                "url": url,
                "published_at": format_utc(pub_ts) if pub_ts else "",
                "retrieved_at": retrieved_at,
                "provider": "exa",
            })
        logger.info("Exa: retrieved %d articles for %s", len(all_articles), asset_ticker)
    except Exception as e:
        logger.error("Exa search error: %s", e)

    return pd.DataFrame(all_articles) if all_articles else pd.DataFrame()
