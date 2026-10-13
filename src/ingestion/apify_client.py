"""Apify X (Twitter) post collection client for CrowdWisdomTrading."""

import json
import hashlib
import time
from datetime import datetime, timezone
import requests
import pandas as pd
from config.settings import (
    APIFY_API_TOKEN, APIFY_ACTOR_ID, APIFY_MAX_ITEMS, APIFY_TIMEOUT_SECONDS,
)
from src.utils.logging_config import logger
from src.utils.time_utils import parse_timestamp, format_utc, utc_now


APIFY_RUN_URL = "https://api.apify.com/v2/acts/{actor_id}/runs"
APIFY_DATASET_URL = "https://api.apify.com/v2/datasets/{dataset_id}/items"


def is_available() -> bool:
    """Check if Apify credentials are configured."""
    return bool(APIFY_API_TOKEN and APIFY_API_TOKEN.strip())


def format_actor_id(actor_id: str) -> str:
    """Apify REST API requires username/actor-name slash to be replaced with tilde (~)."""
    if not actor_id:
        return "apidojo~tweet-scraper"
    return actor_id.strip().replace("/", "~")


def get_actor_input(actor_id: str, search_query: str, max_items: int = 100) -> dict:
    """Construct actor input payload tailored to the specific Apify Actor.
    
    Different scrapers use distinct input schemas:
    - apidojo/tweet-scraper: searchTerms, maxItems, sort, tweetLanguage
    - quacker/twitter-scraper: queries, maxTweets
    - curious_coder/twitter-scraper: searchQueries, tweetsDesired
    - generic/custom: union of common search & limit keys
    """
    aid = (actor_id or "").lower()
    if "apidojo" in aid:
        return {
            "searchTerms": [search_query],
            "maxItems": max_items,
            "sort": "Latest",
            "tweetLanguage": "en",
        }
    elif "quacker" in aid:
        return {
            "searchTerms": [search_query],
            "queries": [search_query],
            "maxTweets": max_items,
        }
    elif "curious_coder" in aid:
        return {
            "searchTerms": [search_query],
            "searchQueries": [search_query],
            "tweetsDesired": max_items,
        }
    else:
        # Generic input providing common keys understood across scrapers
        return {
            "searchTerms": [search_query],
            "queries": [search_query],
            "searchQueries": [search_query],
            "maxItems": max_items,
            "maxTweets": max_items,
            "tweetsDesired": max_items,
            "sort": "Latest",
        }


def verify_credentials() -> dict:
    """Verify Apify token validity against the Apify API without exposing the token."""
    if not is_available():
        return {
            "configured": False,
            "valid": False,
            "username": None,
            "error": "APIFY_API_TOKEN is not configured in .env",
        }
    try:
        headers = {"Authorization": f"Bearer {APIFY_API_TOKEN.strip()}"}
        resp = requests.get("https://api.apify.com/v2/users/me", headers=headers, timeout=10)
        if resp.status_code == 200:
            user_data = resp.json().get("data", {})
            username = user_data.get("username", "verified_user")
            return {
                "configured": True,
                "valid": True,
                "username": username,
                "error": None,
            }
        elif resp.status_code == 401:
            return {
                "configured": True,
                "valid": False,
                "username": None,
                "error": "Invalid API token (HTTP 401 Unauthorized)",
            }
        else:
            return {
                "configured": True,
                "valid": False,
                "username": None,
                "error": f"Apify returned HTTP {resp.status_code}",
            }
    except Exception as e:
        return {
            "configured": True,
            "valid": False,
            "username": None,
            "error": f"Connection check failed: {type(e).__name__}",
        }


def test_apify_connection(actor_id: str = None, query: str = "BTC", max_items: int = 2) -> dict:
    """Small, configurable live test to verify Apify execution before full collection."""
    actor_id = actor_id or APIFY_ACTOR_ID
    cred = verify_credentials()
    if not cred["valid"]:
        return {
            "success": False,
            "configured": cred["configured"],
            "actor_id": actor_id,
            "items_retrieved": 0,
            "error": cred["error"],
        }

    try:
        raw_items = _run_actor(search_query=query, max_items=max_items, actor_id=actor_id)
        if not raw_items:
            return {
                "success": False,
                "configured": True,
                "actor_id": actor_id,
                "items_retrieved": 0,
                "error": f"Actor {actor_id} completed but returned 0 items for '{query}'",
            }

        if any(isinstance(x, dict) and x.get("noResults") for x in raw_items):
            return {
                "success": False,
                "configured": True,
                "actor_id": actor_id,
                "items_retrieved": 0,
                "error": f"Actor {actor_id} returned noResults (actor monthly limit exceeded on Apify plan)",
            }

        df = _normalize_posts(raw_items, asset_ticker=query)
        if df.empty:
            return {
                "success": False,
                "configured": True,
                "actor_id": actor_id,
                "items_retrieved": 0,
                "error": f"Actor {actor_id} returned no valid post text for '{query}'",
            }

        sample = df.iloc[0].to_dict()
        sample.pop("raw_json", None)

        return {
            "success": True,
            "configured": True,
            "actor_id": actor_id,
            "items_retrieved": len(df),
            "sample_post": sample,
            "earliest_ts": str(df["timestamp_utc"].min()),
            "latest_ts": str(df["timestamp_utc"].max()),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "configured": True,
            "actor_id": actor_id,
            "items_retrieved": 0,
            "error": f"Actor execution failed: {type(e).__name__}: {str(e)}",
        }


def collect_posts(asset_ticker: str, search_terms: list[str],
                  max_items: int = None, days_back: int = 180,
                  actor_id: str = None) -> pd.DataFrame:
    """Collect X posts using the configured Apify actor.

    Returns normalized DataFrame with standard post columns.
    """
    if not is_available():
        logger.warning("Apify API token not configured. Set APIFY_API_TOKEN in .env")
        return pd.DataFrame()

    actor_id = actor_id or APIFY_ACTOR_ID
    max_items = max_items or APIFY_MAX_ITEMS
    all_posts = []

    queries = search_terms[:5] if search_terms else [asset_ticker]
    per_query_items = max(5, max_items // len(queries))

    for query in queries:
        try:
            posts = _run_actor(query, max_items=per_query_items, actor_id=actor_id)
            for post in posts:
                post["search_query"] = query
                post["asset"] = asset_ticker
            all_posts.extend(posts)
        except Exception as e:
            logger.error("Apify collection error for query '%s': %s", query, e)
            continue

    if not all_posts:
        logger.warning("No posts collected for %s", asset_ticker)
        return pd.DataFrame()

    df = _normalize_posts(all_posts, asset_ticker)
    logger.info(
        "Apify: collected %d posts for %s (earliest: %s, latest: %s)",
        len(df),
        asset_ticker,
        df["timestamp_utc"].min() if len(df) > 0 else "N/A",
        df["timestamp_utc"].max() if len(df) > 0 else "N/A",
    )
    return df


def _run_actor(search_query: str, max_items: int = 200, actor_id: str = None) -> list[dict]:
    """Run the configured Apify actor and return raw results."""
    actor_id = actor_id or APIFY_ACTOR_ID
    formatted_actor_id = format_actor_id(actor_id)
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN.strip()}"}

    actor_input = get_actor_input(actor_id, search_query, max_items)
    run_url = APIFY_RUN_URL.format(actor_id=formatted_actor_id)
    logger.info("Starting Apify actor %s for query: %s", actor_id, search_query)

    resp = requests.post(
        run_url,
        headers=headers,
        json=actor_input,
        params={"timeout": APIFY_TIMEOUT_SECONDS, "waitForFinish": APIFY_TIMEOUT_SECONDS},
        timeout=APIFY_TIMEOUT_SECONDS + 30,
    )
    resp.raise_for_status()
    run_data = resp.json().get("data", {})

    status = run_data.get("status")
    dataset_id = run_data.get("defaultDatasetId")
    run_id = run_data.get("id")

    # If still RUNNING after waitForFinish parameter, poll briefly
    if status == "RUNNING" and run_id:
        poll_start = time.time()
        while time.time() - poll_start < 45:
            time.sleep(5)
            try:
                status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
                s_resp = requests.get(status_url, headers=headers, timeout=10)
                if s_resp.status_code == 200:
                    status = s_resp.json().get("data", {}).get("status", status)
                    if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):
                        break
            except Exception:
                break

    if status != "SUCCEEDED" or not dataset_id:
        logger.error("Apify actor %s ended with status: %s", actor_id, status)
        if status in ("FAILED", "TIMED-OUT", "ABORTED"):
            return []

    # Fetch dataset items
    items_url = APIFY_DATASET_URL.format(dataset_id=dataset_id)
    items_resp = requests.get(
        items_url,
        headers=headers,
        params={"limit": max_items},
        timeout=60,
    )
    items_resp.raise_for_status()
    data = items_resp.json()
    return data if isinstance(data, list) else []


def _normalize_posts(raw_posts: list[dict], asset_ticker: str) -> pd.DataFrame:
    """Normalize raw Apify results across different actor output schemas."""
    collection_ts = format_utc(utc_now())
    records = []

    for post in raw_posts:
        if not isinstance(post, dict) or post.get("noResults"):
            continue

        # Post ID
        post_id = (
            post.get("id")
            or post.get("tweet_id")
            or post.get("tweetId")
            or post.get("id_str")
            or post.get("conversation_id")
            or hashlib.md5(json.dumps(post, sort_keys=True, default=str).encode()).hexdigest()
        )

        # Timestamp
        ts_raw = (
            post.get("createdAt")
            or post.get("created_at")
            or post.get("timestamp")
            or post.get("date")
            or post.get("time")
            or post.get("posted_at")
        )
        ts = parse_timestamp(ts_raw)

        # Text
        text = (
            post.get("full_text")
            or post.get("text")
            or post.get("tweet_text")
            or post.get("content")
            or post.get("body")
            or post.get("rawContent")
            or ""
        )

        # Author / User object
        author = post.get("author") if isinstance(post.get("author"), dict) else {}
        user = post.get("user") if isinstance(post.get("user"), dict) else {}

        author_id = (
            author.get("id")
            or author.get("id_str")
            or user.get("id_str")
            or user.get("id")
            or post.get("userId")
            or post.get("user_id")
            or ""
        )

        author_username = (
            author.get("userName")
            or author.get("username")
            or user.get("screen_name")
            or user.get("username")
            or post.get("username")
            or post.get("screen_name")
            or ""
        )

        author_followers = (
            author.get("followers")
            or author.get("followersCount")
            or user.get("followers_count")
            or post.get("author_followers")
            or post.get("followersCount")
        )

        # Metrics (likes, reposts, replies, views)
        likes = (
            post.get("likeCount")
            or post.get("favorite_count")
            or post.get("favoriteCount")
            or post.get("likes")
            or post.get("favourites_count")
        )

        reposts = (
            post.get("retweetCount")
            or post.get("retweet_count")
            or post.get("reposts")
            or post.get("retweets")
        )

        replies = (
            post.get("replyCount")
            or post.get("reply_count")
            or post.get("replies")
        )

        views = (
            post.get("viewCount")
            or post.get("views")
            or post.get("viewsCount")
        )

        post_url = (
            post.get("url")
            or post.get("tweet_url")
            or post.get("twitterUrl")
            or (f"https://x.com/{author_username}/status/{post_id}" if author_username and post_id else "")
        )

        records.append({
            "post_id": str(post_id),
            "asset": asset_ticker,
            "timestamp_utc": format_utc(ts) if ts else None,
            "post_text": str(text).strip(),
            "author_id": str(author_id) if author_id else None,
            "author_username": str(author_username) if author_username else None,
            "author_followers": int(author_followers) if author_followers is not None and str(author_followers).isdigit() else None,
            "likes": int(likes) if likes is not None and str(likes).isdigit() else None,
            "reposts": int(reposts) if reposts is not None and str(reposts).isdigit() else None,
            "replies": int(replies) if replies is not None and str(replies).isdigit() else None,
            "views": int(views) if views is not None and str(views).isdigit() else None,
            "post_url": str(post_url),
            "search_query": str(post.get("search_query", "")),
            "collection_timestamp_utc": collection_ts,
            "source": "apify",
            "raw_json": json.dumps(post, default=str)[:2000],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Deduplicate by post_id
    df = df.drop_duplicates(subset=["post_id"])
    # Filter empty text
    df = df[df["post_text"].str.strip().str.len() > 0]
    return df.reset_index(drop=True)
