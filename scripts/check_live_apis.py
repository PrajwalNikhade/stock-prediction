"""Diagnostic script to verify live API connections without exposing secret tokens.

Tests:
1. Binance public market price provider (Live)
2. Yahoo Finance market price provider (Live / rate-limit check)
3. Apify X (Twitter) scraper connection & small query execution (if configured)
4. Tavily financial news search connection & result retrieval (if configured)
5. Exa neural web search connection & result retrieval (if configured)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    APIFY_API_TOKEN, APIFY_ACTOR_ID, TAVILY_API_KEY, EXA_API_KEY,
)
from src.ingestion.price_providers import test_price_connection
from src.ingestion.apify_client import verify_credentials as verify_apify, test_apify_connection
from src.ingestion.tavily_client import test_tavily_connection
from src.ingestion.exa_client import test_exa_connection


def mask_credential(val: str, prefix_len: int = 4) -> str:
    """Return safe masked string showing presence without leaking secrets."""
    if not val or not val.strip():
        return "[NOT SET]"
    v = val.strip()
    if len(v) <= prefix_len:
        return "[SET: ***]"
    return f"[SET: {v[:prefix_len]}...*** ({len(v)} chars)]"


def run_checks(asset: str = "BTCUSDT", query: str = "Bitcoin", max_items: int = 2, actor_id: str = None):
    print("=" * 80)
    print(" CrowdWisdomTrading — Live API & Provider Health Check")
    print("=" * 80)
    print(f"Target Asset: {asset} | Search Query: '{query}' | Max Sample Items: {max_items}")
    print("-" * 80)

    results = []

    # 1. Binance Public API
    print("\n[1/5] Testing Binance Public API...")
    binance_res = test_price_connection(asset_ticker=asset, timeframe="1d", days=5)
    if binance_res["success"]:
        print(f"  -> SUCCESS: Retrieved {binance_res['count']} bars ({binance_res['date_range']})")
        print(f"     Latest close price: {binance_res['latest_close']:,.2f}")
    else:
        print(f"  -> FAILED: {binance_res['error']}")
    results.append({
        "provider": "Binance (Crypto Prices)",
        "type": "Market Data",
        "auth_status": "PUBLIC (No Token Needed)",
        "test_status": "PASS" if binance_res["success"] else "FAIL",
        "records": binance_res["count"],
        "details": binance_res["date_range"],
        "error": binance_res["error"],
    })

    # 2. Yahoo Finance
    print("\n[2/5] Testing Yahoo Finance...")
    yf_res = test_price_connection(asset_ticker="AAPL", timeframe="1d", days=5)
    if yf_res["success"]:
        print(f"  -> SUCCESS: Retrieved {yf_res['count']} bars ({yf_res['date_range']})")
    else:
        print(f"  -> NOTICE / FALLBACK: {yf_res['error']}")
    results.append({
        "provider": "Yahoo Finance (Equities)",
        "type": "Market Data",
        "auth_status": "PUBLIC (No Token Needed)",
        "test_status": "PASS" if yf_res["success"] else "RATE-LIMITED / FALLBACK",
        "records": yf_res["count"],
        "details": yf_res["date_range"],
        "error": yf_res["error"],
    })

    # 3. Apify X (Twitter) Scraper
    print("\n[3/5] Testing Apify X Post Collection...")
    print(f"  Credential: {mask_credential(APIFY_API_TOKEN)}")
    selected_actor = actor_id or APIFY_ACTOR_ID
    print(f"  Actor: {selected_actor}")
    apify_cred = verify_apify()
    if not apify_cred["configured"]:
        print("  -> SKIPPED: APIFY_API_TOKEN is not configured in .env")
        results.append({
            "provider": "Apify (X / Twitter)",
            "type": "Social Sentiment",
            "auth_status": "NOT CONFIGURED",
            "test_status": "SKIPPED",
            "records": 0,
            "details": f"Actor: {selected_actor}",
            "error": "Missing APIFY_API_TOKEN",
        })
    elif not apify_cred["valid"]:
        print(f"  -> AUTH FAILED: {apify_cred['error']}")
        results.append({
            "provider": "Apify (X / Twitter)",
            "type": "Social Sentiment",
            "auth_status": "INVALID TOKEN",
            "test_status": "AUTH_FAILED",
            "records": 0,
            "details": f"Actor: {selected_actor}",
            "error": apify_cred["error"],
        })
    else:
        print(f"  Auth verified for Apify user: {apify_cred.get('username')}")
        print(f"  Executing small live test with {max_items} items...")
        apify_test = test_apify_connection(actor_id=selected_actor, query=query, max_items=max_items)
        if apify_test["success"]:
            print(f"  -> SUCCESS: Retrieved {apify_test['items_retrieved']} posts!")
            sample = apify_test.get("sample_post", {})
            if sample:
                print(f"     Sample post ID: {sample.get('post_id')} by @{sample.get('author_username')}")
                print(f"     Sample timestamp: {sample.get('timestamp_utc')}")
                print(f"     Text snippet: {sample.get('post_text', '')[:80]}...")
            results.append({
                "provider": "Apify (X / Twitter)",
                "type": "Social Sentiment",
                "auth_status": "CONFIGURED",
                "test_status": "PASS",
                "records": apify_test["items_retrieved"],
                "details": f"{apify_test.get('earliest_ts')} to {apify_test.get('latest_ts')}",
                "error": None,
            })
        else:
            print(f"  -> FAILED: {apify_test['error']}")
            results.append({
                "provider": "Apify (X / Twitter)",
                "type": "Social Sentiment",
                "auth_status": "CONFIGURED",
                "test_status": "FAIL",
                "records": 0,
                "details": f"Actor: {selected_actor}",
                "error": apify_test["error"],
            })

    # 4. Tavily News Search
    print("\n[4/5] Testing Tavily Financial News Search...")
    print(f"  Credential: {mask_credential(TAVILY_API_KEY)}")
    tavily_res = test_tavily_connection(query=query, max_results=max_items)
    if not tavily_res["configured"]:
        print("  -> SKIPPED: TAVILY_API_KEY is not configured in .env")
        results.append({
            "provider": "Tavily News Search",
            "type": "News & Web",
            "auth_status": "NOT CONFIGURED",
            "test_status": "SKIPPED",
            "records": 0,
            "details": "N/A",
            "error": "Missing TAVILY_API_KEY",
        })
    elif tavily_res["success"]:
        print(f"  -> SUCCESS: Retrieved {tavily_res['articles_retrieved']} articles")
        sample = tavily_res.get("sample_article", {})
        if sample:
            print(f"     Title: {sample.get('title')}")
            print(f"     URL: {sample.get('url')}")
            print(f"     Date: {sample.get('published_date')}")
        results.append({
            "provider": "Tavily News Search",
            "type": "News & Web",
            "auth_status": "CONFIGURED",
            "test_status": "PASS",
            "records": tavily_res["articles_retrieved"],
            "details": sample.get("title", "")[:50],
            "error": None,
        })
    else:
        print(f"  -> FAILED: {tavily_res['error']}")
        results.append({
            "provider": "Tavily News Search",
            "type": "News & Web",
            "auth_status": "CONFIGURED",
            "test_status": "FAIL",
            "records": 0,
            "details": "N/A",
            "error": tavily_res["error"],
        })

    # 5. Exa Neural Search
    print("\n[5/5] Testing Exa Neural Web Search...")
    print(f"  Credential: {mask_credential(EXA_API_KEY)}")
    exa_res = test_exa_connection(query=query, max_results=max_items)
    if not exa_res["configured"]:
        print("  -> SKIPPED: EXA_API_KEY is not configured in .env")
        results.append({
            "provider": "Exa Neural Search",
            "type": "News & Web",
            "auth_status": "NOT CONFIGURED",
            "test_status": "SKIPPED",
            "records": 0,
            "details": "N/A",
            "error": "Missing EXA_API_KEY",
        })
    elif exa_res["success"]:
        print(f"  -> SUCCESS: Retrieved {exa_res['articles_retrieved']} articles")
        sample = exa_res.get("sample_article", {})
        if sample:
            print(f"     Title: {sample.get('title')}")
            print(f"     URL: {sample.get('url')}")
            print(f"     Date: {sample.get('published_date')}")
        results.append({
            "provider": "Exa Neural Search",
            "type": "News & Web",
            "auth_status": "CONFIGURED",
            "test_status": "PASS",
            "records": exa_res["articles_retrieved"],
            "details": sample.get("title", "")[:50],
            "error": None,
        })
    else:
        print(f"  -> FAILED: {exa_res['error']}")
        results.append({
            "provider": "Exa Neural Search",
            "type": "News & Web",
            "auth_status": "CONFIGURED",
            "test_status": "FAIL",
            "records": 0,
            "details": "N/A",
            "error": exa_res["error"],
        })

    # ---------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(" SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Provider':<26} | {'Auth Status':<18} | {'Test Status':<12} | {'Records':<7} | {'Notes / Error'}")
    print("-" * 80)
    for r in results:
        err_msg = r["error"] or r["details"] or ""
        if len(err_msg) > 35:
            err_msg = err_msg[:32] + "..."
        print(f"{r['provider']:<26} | {r['auth_status']:<18} | {r['test_status']:<12} | {r['records']:<7} | {err_msg}")
    print("=" * 80)

    # Next steps & instructions
    configured_apis = [r["provider"] for r in results if r["test_status"] == "PASS"]
    missing_creds = [r["provider"] for r in results if "NOT CONFIGURED" in r["auth_status"]]
    failing_creds = [r["provider"] for r in results if r["test_status"] in ("AUTH_FAILED", "FAIL") and r["auth_status"] != "PUBLIC"]

    print("\n[STATUS BREAKDOWN]")
    print(f"* Live APIs Verified: {', '.join(configured_apis) if configured_apis else 'None'}")
    print(f"* Credentials Missing: {', '.join(missing_creds) if missing_creds else 'None'}")
    print(f"* Invalid / Auth-Failing Credentials: {', '.join(failing_creds) if failing_creds else 'None'}")
    print("\n[NEXT STEPS]")
    if missing_creds or failing_creds:
        print("1. Update your valid credentials in D:\\stock\\.env:")
        if "Apify (X / Twitter)" in failing_creds or "Apify (X / Twitter)" in missing_creds:
            print("   - APIFY_API_TOKEN=<your_valid_apify_token> (Token must be valid on apify.com)")
        if "Tavily News Search" in missing_creds or "Tavily News Search" in failing_creds:
            print("   - TAVILY_API_KEY=<your_tavily_api_key>")
        if "Exa Neural Search" in missing_creds or "Exa Neural Search" in failing_creds:
            print("   - EXA_API_KEY=<your_exa_api_key>")
        print("2. Re-run this check: python scripts/check_live_apis.py")
        print("3. When Apify is authenticated, run full live ingestion in the dashboard or via:")
        print("   python scripts/generate_reports.py")
    else:
        print("All API credentials are configured and verified! You can run live ingestion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check live provider and API connections.")
    parser.add_argument("--asset", default="BTCUSDT", help="Asset ticker to test (default: BTCUSDT)")
    parser.add_argument("--query", default="Bitcoin", help="Search query for live test (default: Bitcoin)")
    parser.add_argument("--max-items", type=int, default=2, help="Max items for small test (default: 2)")
    parser.add_argument("--apify-actor", default=None, help="Apify Actor ID override")
    args = parser.parse_args()

    run_checks(asset=args.asset, query=args.query, max_items=args.max_items, actor_id=args.apify_actor)
