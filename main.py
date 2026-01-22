# main.py
"""
ADUmedia Telegram Publisher Service

Intelligent publishing service that:
1. Determines edition type based on day of week
2. Fetches candidates from R2 (single day or multiple days)
3. Uses AI to select top 7 articles
4. Tracks publications in Supabase for deduplication
5. Sends digest to Telegram channel
6. Archives sent articles

Schedule:
    Monday    - Weekly Edition (covers full week)
    Tuesday   - Weekend Catch-Up Edition (covers Sat, Sun, Mon)
    Wednesday - Daily Edition
    Thursday  - Daily Edition
    Friday    - Daily Edition
    Saturday  - No publication
    Sunday    - No publication

Usage:
    python main.py                    # Auto-detect edition type
    python main.py --edition daily    # Force daily edition
    python main.py --edition weekly   # Force weekly edition
    python main.py --test             # Test connections only
    python main.py --dry-run          # Fetch and select but don't send

Environment Variables:
    TELEGRAM_BOT_TOKEN      - Telegram bot token
    TELEGRAM_CHANNEL_ID     - Telegram channel ID
    R2_ACCOUNT_ID           - Cloudflare R2 account ID
    R2_ACCESS_KEY_ID        - R2 access key
    R2_SECRET_ACCESS_KEY    - R2 secret key
    R2_BUCKET_NAME          - R2 bucket name
    R2_PUBLIC_URL           - R2 public URL for images
    SUPABASE_URL            - Supabase project URL
    SUPABASE_KEY            - Supabase API key
    OPENAI_API_KEY          - OpenAI API key for GPT-4o
    LANGCHAIN_TRACING_V2    - Set to "true" for LangSmith tracing
    LANGCHAIN_API_KEY       - LangSmith API key
    LANGCHAIN_PROJECT       - LangSmith project name
"""

import asyncio
import argparse
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

# Import modules
from storage.r2 import R2Storage
from telegram_bot import TelegramBot
from editor.selector import (
    ArticleSelector, 
    EditionType, 
    determine_edition_type,
    get_edition_display_name
)
from editor.deduplication import DeduplicationChecker
from database.connection import test_connection as test_db_connection


# =============================================================================
# Configuration
# =============================================================================

ARTICLES_PER_EDITION = 7


# =============================================================================
# Command Line Arguments
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ADUmedia Telegram Publisher")

    parser.add_argument(
        "--test", action="store_true",
        help="Test connections only (R2, Telegram, Supabase, OpenAI)"
    )
    parser.add_argument(
        "--edition", type=str, choices=["daily", "weekend", "weekly"], default=None,
        help="Force specific edition type (default: auto-detect from day)"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date to publish (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and select articles but don't send to Telegram"
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Send to Telegram but don't archive articles"
    )
    parser.add_argument(
        "--skip-selection", action="store_true",
        help="Skip AI selection, use all candidates (for testing)"
    )

    return parser.parse_args()


# =============================================================================
# Date Range Helpers
# =============================================================================

def get_dates_for_edition(edition_type: EditionType, target_date: date) -> List[date]:
    """Get the dates to fetch candidates for based on edition type."""
    if edition_type == EditionType.DAILY:
        return [target_date]
    
    elif edition_type == EditionType.WEEKEND:
        # Weekend (Tuesday): Saturday, Sunday, Monday
        monday = target_date - timedelta(days=1)
        sunday = target_date - timedelta(days=2)
        saturday = target_date - timedelta(days=3)
        return [saturday, sunday, monday]
    
    elif edition_type == EditionType.WEEKLY:
        # Weekly (Monday): previous 7 days
        dates = []
        for i in range(7, 0, -1):
            dates.append(target_date - timedelta(days=i))
        return dates
    
    return [target_date]


# =============================================================================
# Candidate Fetching
# =============================================================================

def fetch_candidates_for_dates(r2: R2Storage, dates: List[date]) -> List[Dict[str, Any]]:
    """Fetch all candidates from R2 for the given dates."""
    all_candidates = []
    
    for target_date in dates:
        print(f"   Fetching {target_date.isoformat()}...")
        candidates = r2.get_all_candidates(target_date)
        
        for candidate in candidates:
            candidate["_fetch_date"] = target_date.isoformat()
        
        all_candidates.extend(candidates)
        print(f"      Found {len(candidates)} candidates")
    
    print(f"   [OK] Total candidates: {len(all_candidates)}")
    return all_candidates


def prepare_articles_for_telegram(articles: List[Dict[str, Any]], r2: R2Storage) -> List[Dict[str, Any]]:
    """Prepare selected articles for Telegram sending."""
    prepared = []

    for article in articles:
        telegram_article = {
            "id": article.get("id", ""),
            "title": article.get("title", ""),
            "link": article.get("link", ""),
            "ai_summary": article.get("ai_summary", ""),
            "tags": article.get("tags", []),
            "source_id": article.get("source_id", ""),
            "source_name": article.get("source_name", article.get("source_id", "Unknown")),
            "published": article.get("published"),
            "_fetch_date": article.get("_fetch_date"),
        }

        image_info = article.get("image", {})
        if image_info.get("has_image") and image_info.get("r2_path"):
            r2_path = image_info["r2_path"]
            if r2.public_url:
                image_url = f"{r2.public_url.rstrip('/')}/{r2_path}"
            else:
                image_url = image_info.get("original_url")

            telegram_article["hero_image"] = {
                "url": image_info.get("original_url"),
                "r2_url": image_url,
                "r2_path": r2_path,
            }
        else:
            telegram_article["hero_image"] = None

        if telegram_article["ai_summary"]:
            prepared.append(telegram_article)
        else:
            print(f"   [WARN] Skipping article without summary: {telegram_article['title'][:40]}...")

    return prepared


# =============================================================================
# Selection Pipeline
# =============================================================================

async def select_articles(
    edition_type: EditionType,
    candidates: List[Dict[str, Any]],
    dedup: DeduplicationChecker,
    target_date: date
) -> List[Dict[str, Any]]:
    """Use AI to select articles for the edition."""
    print(f"\n[SELECT] AI selection for {edition_type.value} edition...")
    
    selector = ArticleSelector()
    
    # Get previously published URLs for deduplication
    published_urls = dedup.get_published_urls(since_date=target_date - timedelta(days=90))
    print(f"   Previously published: {len(published_urls)} articles")
    
    # Edition-specific data
    weekly_article_urls = []
    daily_published_this_week = []
    recent_weekly_urls = []
    
    if edition_type == EditionType.WEEKEND:
        monday = target_date - timedelta(days=1)
        weekly_article_urls = dedup.get_weekly_edition_urls(monday)
        print(f"   Weekly to exclude: {len(weekly_article_urls)} articles")
    
    elif edition_type == EditionType.WEEKLY:
        week_end = target_date - timedelta(days=1)
        daily_published_this_week = dedup.get_daily_editions_this_week(week_end)
        print(f"   In daily editions: {len(daily_published_this_week)} articles")
        recent_weekly_urls = dedup.get_recent_weekly_urls(days=30)
        print(f"   Recent weekly: {len(recent_weekly_urls)} articles")
    
    # Run AI selection
    selection = await selector.select(
        edition_type=edition_type,
        candidates=candidates,
        published_urls=published_urls,
        weekly_article_urls=weekly_article_urls,
        daily_published_this_week=daily_published_this_week,
        recent_weekly_urls=recent_weekly_urls,
        target_date=target_date
    )
    
    # Map selected IDs back to full candidate data
    candidate_map = {c.get("id"): c for c in candidates}
    selected_articles = []
    
    for item in selection.selected:
        article_id = item.id
        if article_id in candidate_map:
            article = candidate_map[article_id].copy()
            article["_selection_reason"] = item.reason
            article["_selection_category"] = item.category
            article["_weekly_candidate"] = item.weekly_candidate
            article["_is_repeat"] = item.is_repeat
            selected_articles.append(article)
        else:
            print(f"   [WARN] Selected article not found: {article_id}")
    
    print(f"\n[SELECT] Edition summary: {selection.edition_summary}")
    print(f"[SELECT] Selected {len(selected_articles)} articles")
    
    return selected_articles


# =============================================================================
# Publication Recording
# =============================================================================

def record_publications(
    dedup: DeduplicationChecker,
    articles: List[Dict[str, Any]],
    edition_type: EditionType,
    edition_date: date,
    total_candidates: int
) -> List[str]:
    """Record publications in database."""
    print(f"\n[RECORD] Recording {len(articles)} publications...")
    
    article_ids = []
    articles_new = 0
    articles_repeated = 0
    
    for article in articles:
        is_repeat = article.get("_is_repeat", False)
        
        article_id = dedup.record_publication(
            article=article,
            edition_type=edition_type.value,
            edition_date=edition_date,
            r2_path=article.get("image", {}).get("r2_path")
        )
        
        if article_id:
            article_ids.append(article_id)
            
            if is_repeat:
                articles_repeated += 1
            else:
                articles_new += 1
            
            if edition_type == EditionType.DAILY and article.get("_weekly_candidate"):
                week_start = edition_date - timedelta(days=edition_date.weekday())
                dedup.flag_weekly_candidate(
                    article_id=article_id,
                    week_start=week_start,
                    category=article.get("_selection_category"),
                    notes=article.get("_selection_reason")
                )
    
    dedup.record_edition(
        edition_type=edition_type.value,
        edition_date=edition_date,
        article_ids=article_ids,
        total_candidates=total_candidates,
        articles_new=articles_new,
        articles_repeated=articles_repeated
    )
    
    print(f"   [OK] Recorded: {articles_new} new, {articles_repeated} repeated")
    return article_ids


# =============================================================================
# Archive Logic
# =============================================================================

def archive_sent_articles(r2: R2Storage, articles: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Archive articles that were successfully sent."""
    print(f"\n[ARCHIVE] Archiving {len(articles)} articles...")
    
    results = {}
    
    for article in articles:
        article_id = article.get("id")
        fetch_date_str = article.get("_fetch_date")
        
        if not article_id or not fetch_date_str:
            continue
        
        try:
            fetch_date = date.fromisoformat(fetch_date_str)
        except ValueError:
            print(f"   [WARN] Invalid date for {article_id}: {fetch_date_str}")
            continue
        
        success = r2.archive_article(article_id, fetch_date)
        results[article_id] = success
    
    archived = sum(1 for v in results.values() if v)
    failed = len(results) - archived
    
    print(f"   [OK] Archived: {archived}, Failed: {failed}")
    return results


# =============================================================================
# Connection Testing
# =============================================================================

async def test_connections() -> bool:
    """Test all service connections."""
    print("\n[TEST] Testing Connections...")
    print("=" * 50)

    all_ok = True

    print("\n1. Testing R2 Storage...")
    try:
        r2 = R2Storage()
        r2.test_connection()
        print("   [OK] R2 connection OK")
    except Exception as e:
        print(f"   [ERROR] R2 connection failed: {e}")
        all_ok = False

    print("\n2. Testing Telegram Bot...")
    try:
        bot = TelegramBot()
        if await bot.test_connection():
            print("   [OK] Telegram connection OK")
        else:
            print("   [ERROR] Telegram connection failed")
            all_ok = False
    except Exception as e:
        print(f"   [ERROR] Telegram connection failed: {e}")
        all_ok = False

    print("\n3. Testing Supabase Database...")
    try:
        if test_db_connection():
            print("   [OK] Supabase connection OK")
        else:
            print("   [ERROR] Supabase connection failed")
            all_ok = False
    except Exception as e:
        print(f"   [ERROR] Supabase connection failed: {e}")
        all_ok = False

    print("\n4. Testing OpenAI API...")
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", max_tokens=10)
        response = await llm.ainvoke([("human", "Say OK")])
        print(f"   [OK] OpenAI connection OK")
    except Exception as e:
        print(f"   [ERROR] OpenAI connection failed: {e}")
        all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("[OK] All connections successful!")
    else:
        print("[ERROR] Some connections failed. Check credentials.")

    return all_ok


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_publisher(
    edition_type: Optional[EditionType] = None,
    target_date: Optional[date] = None,
    dry_run: bool = False,
    no_archive: bool = False,
    skip_selection: bool = False,
):
    """Main publishing pipeline."""
    if target_date is None:
        target_date = date.today()
    
    if edition_type is None:
        edition_type = determine_edition_type(target_date)
        
        if edition_type is None:
            print(f"\n[INFO] No publication scheduled for {target_date.strftime('%A')}")
            print("[INFO] Use --edition flag to force a specific edition type")
            return
    
    edition_name = get_edition_display_name(edition_type)
    
    print(f"\n{'=' * 60}")
    print(f"[START] ADUmedia Telegram Publisher")
    print(f"{'=' * 60}")
    print(f"[INFO] {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    print(f"[INFO] Edition: {edition_name}")
    print(f"[INFO] Date: {target_date.isoformat()} ({target_date.strftime('%A')})")
    if dry_run:
        print("[WARN] DRY RUN - No messages will be sent")
    if no_archive:
        print("[WARN] NO ARCHIVE - Articles will not be archived")
    if skip_selection:
        print("[WARN] SKIP SELECTION - Using all candidates")
    print(f"{'=' * 60}")

    try:
        print("\n[INIT] Connecting to services...")
        r2 = R2Storage()
        r2.test_connection()
        
        dedup = DeduplicationChecker()
        
        dates_to_fetch = get_dates_for_edition(edition_type, target_date)
        print(f"\n[FETCH] Fetching candidates for {len(dates_to_fetch)} day(s)...")
        
        candidates = fetch_candidates_for_dates(r2, dates_to_fetch)
        
        if not candidates:
            print("\n[EMPTY] No candidates found. Exiting.")
            return
        
        unpublished, already_published = dedup.filter_unpublished(candidates)
        print(f"[DEDUP] After filtering: {len(unpublished)} unpublished, {len(already_published)} already published")
        
        if not unpublished:
            print("\n[EMPTY] All candidates already published. Exiting.")
            return
        
        if skip_selection:
            selected = unpublished[:ARTICLES_PER_EDITION]
            print(f"\n[SELECT] Using first {len(selected)} candidates (skipped AI)")
        else:
            selected = await select_articles(
                edition_type=edition_type,
                candidates=unpublished,
                dedup=dedup,
                target_date=target_date
            )
        
        if not selected:
            print("\n[EMPTY] No articles selected. Exiting.")
            return
        
        articles = prepare_articles_for_telegram(selected, r2)
        
        print(f"\n[READY] {len(articles)} articles ready to publish")
        
        print("\n[PREVIEW] Articles to publish:")
        for i, article in enumerate(articles, 1):
            has_image = "[IMG]" if article.get("hero_image") else "[TXT]"
            source = article.get("source_name", "Unknown")[:15]
            title = article.get("title", "No title")[:40]
            print(f"   {i}. {has_image} [{source}] {title}...")
        
        if dry_run:
            print("\n[DRY RUN] Complete. No messages sent.")
            return
        
        print("\n[SEND] Sending to Telegram...")
        bot = TelegramBot()
        
        results = await bot.send_digest(articles, include_header=True)
        
        print(f"\n{'=' * 60}")
        print("[TELEGRAM RESULTS]")
        print(f"{'=' * 60}")
        print(f"   [OK] Sent: {results['sent']}")
        print(f"   [ERROR] Failed: {results['failed']}")
        print(f"   [TIME] Duration: {results['total_time']/60:.1f} minutes")
        
        if results['sent'] > 1:
            article_ids = record_publications(
                dedup=dedup,
                articles=articles[:results['sent'] - 1],
                edition_type=edition_type,
                edition_date=target_date,
                total_candidates=len(candidates)
            )
        
        if not no_archive and results['sent'] > 1:
            archive_sent_articles(r2, articles[:results['sent'] - 1])
        elif no_archive:
            print("\n[SKIP] Archiving skipped (--no-archive flag)")
        
        print(f"\n{'=' * 60}")
        print("[DONE] Pipeline complete!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n[FATAL] Pipeline error: {e}")
        raise


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    args = parse_args()

    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHANNEL_ID",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "OPENAI_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Please set these in Railway dashboard.")
        return

    if args.test:
        await test_connections()
        return

    target_date = None
    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}")
            print("Use YYYY-MM-DD format (e.g., 2026-01-20)")
            return

    edition_type = None
    if args.edition:
        edition_map = {
            "daily": EditionType.DAILY,
            "weekend": EditionType.WEEKEND,
            "weekly": EditionType.WEEKLY,
        }
        edition_type = edition_map.get(args.edition)

    await run_publisher(
        edition_type=edition_type,
        target_date=target_date,
        dry_run=args.dry_run,
        no_archive=args.no_archive,
        skip_selection=args.skip_selection,
    )


if __name__ == "__main__":
    asyncio.run(main())
