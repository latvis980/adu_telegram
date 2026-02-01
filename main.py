# main.py
"""
ADUmedia Telegram Publisher Service

Intelligent publishing service with AI-BASED PROJECT DEDUPLICATION:
1. Fetches candidates from R2
2. Extracts project info from summaries using AI (NO DB writes)
3. Filters duplicates against PUBLISHED projects only
4. Uses AI to select top 7 articles
5. Sends digest to Telegram
6. Records published articles and creates projects in Supabase

IMPORTANT: Projects are ONLY created in the database when articles are
actually published, not during extraction/filtering.

The AI matching handles:
- Name variations ("The Spiral" vs "Spiral Tower")
- Capitalization differences
- Architect name variations ("BIG" vs "Bjarke Ingels Group")
- Location variations ("NYC" vs "New York")

Schedule:
    Monday    - Weekly Edition (covers full week, Mon-Sun)
    Tuesday   - Weekend Catch-Up Edition (covers Sat, Sun, Mon - no daily editions)
    Wednesday - Daily Edition
    Thursday  - Daily Edition
    Friday    - Daily Edition
    Saturday  - No publication
    Sunday    - No publication

Usage:
    python main.py                    # Auto-detect edition type
    python main.py --edition daily    # Force daily edition
    python main.py --test             # Test connections only
    python main.py --dry-run          # Fetch and select but don't send

Environment Variables:
    TELEGRAM_BOT_TOKEN      - Telegram bot token
    TELEGRAM_CHANNEL_ID     - Telegram channel ID
    R2_*                    - Cloudflare R2 credentials
    SUPABASE_URL/KEY        - Supabase credentials
    OPENAI_API_KEY          - OpenAI API key
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
from editor.project_extractor import ProjectExtractor
from database.connection import test_connection as test_db_connection


# =============================================================================
# Configuration
# =============================================================================

ARTICLES_PER_EDITION = 7
PROJECT_COOLDOWN_MONTHS = 3


# =============================================================================
# Command Line Arguments
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ADUmedia Telegram Publisher")

    parser.add_argument(
        "--test", action="store_true",
        help="Test connections only"
    )
    parser.add_argument(
        "--edition", type=str, choices=["daily", "weekend", "weekly"], default=None,
        help="Force specific edition type"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't send to Telegram"
    )
    parser.add_argument(
        "--skip-selection", action="store_true",
        help="Skip AI selection"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip project extraction"
    )

    return parser.parse_args()


# =============================================================================
# Date Range Helpers
# =============================================================================

def get_dates_for_edition(edition_type: EditionType, target_date: date) -> List[date]:
    """
    Get the dates to fetch candidates for based on edition type.

    Schedule:
    - Daily (Wed/Thu/Fri): Same day only
    - Weekend (Tuesday): Sat, Sun, Mon (3 days with no daily editions)
    - Weekly: Previous 7 days (Mon-Sun)
      * TEMPORARY: Testing on Sunday (see selector.py determine_edition_type)
      * NORMAL: Runs on Monday (see selector.py determine_edition_type)

    Note: This function works the same for both Sunday and Monday.
    The day selection is controlled in selector.py's determine_edition_type().
    """
    if edition_type == EditionType.DAILY:
        # Same day
        return [target_date]
    elif edition_type == EditionType.WEEKEND:
        # Tuesday covers: Saturday (-3), Sunday (-2), Monday (-1)
        return [target_date - timedelta(days=i) for i in range(3, 0, -1)]
    elif edition_type == EditionType.WEEKLY:
        # Weekly covers: Previous 7 days (Mon-Sun)
        # Works for both Sunday testing and Monday normal operation
        return [target_date - timedelta(days=i) for i in range(7, 0, -1)]
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


# =============================================================================
# Project Extraction Pipeline (NO DB writes)
# =============================================================================

async def extract_project_info(
    candidates: List[Dict[str, Any]],
    skip_extraction: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract project info from article summaries using AI.

    IMPORTANT: This step ONLY extracts information, it does NOT write to the database.
    Projects are only created when articles are actually published.

    Args:
        candidates: List of candidate articles
        skip_extraction: Skip AI extraction (for testing)

    Returns:
        Candidates with _extracted_info attached (no _project_id yet)
    """
    if skip_extraction:
        print("[EXTRACT] Skipping project extraction (--skip-extraction)")
        for candidate in candidates:
            candidate["_extracted_info"] = {"is_project": False}
        return candidates

    print(f"\n[EXTRACT] Extracting project info from {len(candidates)} articles...")
    print("          (No database writes - just extracting metadata)")

    extractor = ProjectExtractor()

    # Extract project info in batches
    extractions = await extractor.extract_batch(candidates, batch_size=10)

    # Attach extraction results to candidates (NO DB writes)
    projects_found = 0
    non_projects = 0

    for candidate, extraction in zip(candidates, extractions):
        candidate["_extracted_info"] = extraction.model_dump()
        # NOTE: We do NOT set _project_id here - that happens during filtering/publishing

        if extraction.is_project and extraction.project_name:
            projects_found += 1
        else:
            non_projects += 1

    print(f"   [OK] Extracted: {projects_found} project articles, {non_projects} non-project articles")

    return candidates


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

    # Get recent weekly URLs for exclusion (weekly edition only)
    recent_weekly_urls = []
    if edition_type == EditionType.WEEKLY:
        # Get URLs from weekly editions in the last 30 days
        try:
            result = dedup.client.table("all_articles")\
                .select("article_url")\
                .contains("selected_for_editions", ["weekly"])\
                .gte("fetch_date", (target_date - timedelta(days=30)).isoformat())\
                .execute()

            recent_weekly_urls = [row["article_url"] for row in result.data if row.get("article_url")]
            print(f"   Recent weekly URLs (exclude): {len(recent_weekly_urls)}")
        except Exception as e:
            print(f"   [WARN] Could not fetch recent weekly URLs: {e}")

    # Run AI selection
    # Note: We pass empty published_urls since we're using project-based dedup now
    published_urls = []

    selection = await selector.select(
        edition_type=edition_type,
        candidates=candidates,
        published_urls=published_urls,
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


def prepare_articles_for_telegram(articles: List[Dict[str, Any]], r2: R2Storage) -> List[Dict[str, Any]]:
    """Prepare selected articles for Telegram sending."""
    prepared = []

    for article in articles:
        telegram_article = {
            "id": article.get("id", ""),
            "title": article.get("title", ""),  # Keep for logging
            "headline": article.get("headline", ""),  # CRITICAL: This is what Telegram uses
            "tag": article.get("tag", ""),  # CRITICAL: Single tag for Telegram
            "link": article.get("link", ""),
            "ai_summary": article.get("ai_summary", ""),
            "tags": article.get("tags", []),  # Keep for metadata
            "source_id": article.get("source_id", ""),
            "source_name": article.get("source_name", article.get("source_id", "Unknown")),
            "published": article.get("published"),
            "_fetch_date": article.get("_fetch_date"),
            "_extracted_info": article.get("_extracted_info"),  # For recording
            "_selection_reason": article.get("_selection_reason"),
            "_selection_category": article.get("_selection_category"),
        }

        image_info = article.get("image", {})
        if image_info and image_info.get("has_image") and image_info.get("r2_path"):
            r2_path = image_info["r2_path"]

            # CRITICAL FIX: Check if R2 public URL is configured
            if not r2.public_url:
                print(f"   [ERROR] R2_PUBLIC_URL not configured! Cannot generate image URLs.")
                telegram_article["hero_image"] = None
            else:
                # Clean the path - remove leading slash if present
                clean_path = r2_path.lstrip('/')

                # Construct full public URL
                image_url = f"{r2.public_url.rstrip('/')}/{clean_path}"

                # VALIDATE URL before adding
                if not image_url.startswith(('http://', 'https://')):
                    print(f"   [ERROR] Invalid URL (no protocol): {image_url}")
                    telegram_article["hero_image"] = None
                elif '//' in image_url.replace('://', ''):  # Double slash check
                    print(f"   [ERROR] Malformed URL (double slash): {image_url}")
                    telegram_article["hero_image"] = None
                else:
                    print(f"   [DEBUG] Image URL: {image_url}")
                    telegram_article["hero_image"] = {
                        "url": image_info.get("original_url"),
                        "r2_url": image_url,  # This is the full public URL
                        "r2_path": r2_path,
                    }
        else:
            telegram_article["hero_image"] = None

        # Validate required fields before adding
        if not telegram_article["ai_summary"]:
            print(f"   [WARN] Skipping article without summary: {telegram_article['title'][:40]}...")
            continue

        if not telegram_article["headline"]:
            print(f"   [WARN] Skipping article without headline: {telegram_article['title'][:40]}...")
            continue

        prepared.append(telegram_article)

    return prepared


# =============================================================================
# Publication Recording (Projects created HERE)
# =============================================================================

async def record_publications(
    dedup: DeduplicationChecker,
    articles: List[Dict[str, Any]],
    edition_type: EditionType,
    edition_date: date,
    total_candidates: int,
    edition_summary: Optional[str] = None
) -> List[str]:
    """
    Record publications in database.

    IMPORTANT: This is where projects are created - only for PUBLISHED articles.
    """
    print(f"\n[RECORD] Recording {len(articles)} publications...")

    article_ids = []
    articles_new = 0
    articles_repeated = 0

    for article in articles:
        extracted_info = article.get("_extracted_info", {})

        # Determine if this is a project article
        is_project = extracted_info.get("is_project", False)
        project_name = extracted_info.get("project_name")
        project_id = None

        if is_project and project_name:
            # Find or create project - projects are ONLY created here, on publish
            architect = extracted_info.get("architect")
            location = extracted_info.get("location", {})
            location_city = location.get("city") if isinstance(location, dict) else None
            location_country = location.get("country") if isinstance(location, dict) else None

            project_id, is_new_project = await dedup.find_or_create_project_on_publish(
                project_name=project_name,
                architect=architect,
                location_city=location_city,
                location_country=location_country,
                project_type=extracted_info.get("project_type"),
                project_status=extracted_info.get("project_status"),
                summary_excerpt=article.get("ai_summary", "")[:200],
                publish_date=edition_date
            )

            if is_new_project:
                articles_new += 1
            else:
                articles_repeated += 1
        else:
            # Non-project article (news, interview, etc.)
            articles_new += 1

        # Record article in all_articles
        article_db_id = dedup.record_article(
            article=article,
            project_id=project_id,
            extracted_info=extracted_info,
            status="published"
        )

        if article_db_id:
            article_ids.append(article_db_id)

            # Mark as published with edition info
            dedup.mark_article_published(
                article_id=article_db_id,
                edition_type=edition_type.value,
                edition_date=edition_date
            )

    # Record the edition
    dedup.record_edition(
        edition_type=edition_type.value,
        edition_date=edition_date,
        article_ids=article_ids,
        total_candidates=total_candidates,
        articles_new=articles_new,
        articles_repeated=articles_repeated,
        edition_summary=edition_summary
    )

    print(f"   [OK] Recorded: {articles_new} new, {articles_repeated} updates")
    return article_ids


# =============================================================================
# Connection Testing
# =============================================================================

async def test_connections() -> bool:
    """Test all service connections."""
    print("\n[TEST] Testing Connections...")
    print("=" * 50)

    all_ok = True

    print("\n1. R2 Storage...")
    try:
        r2 = R2Storage()
        r2.test_connection()
        print("   [OK]")
    except Exception as e:
        print(f"   [ERROR] {e}")
        all_ok = False

    print("\n2. Telegram Bot...")
    try:
        bot = TelegramBot()
        if await bot.test_connection():
            print("   [OK]")
        else:
            all_ok = False
    except Exception as e:
        print(f"   [ERROR] {e}")
        all_ok = False

    print("\n3. Supabase...")
    try:
        if test_db_connection():
            print("   [OK]")
        else:
            all_ok = False
    except Exception as e:
        print(f"   [ERROR] {e}")
        all_ok = False

    print("\n4. OpenAI API...")
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=10)
        await llm.ainvoke([("human", "Say OK")])
        print("   [OK]")
    except Exception as e:
        print(f"   [ERROR] {e}")
        all_ok = False

    print("\n" + "=" * 50)
    print("[OK] All connections successful!" if all_ok else "[ERROR] Some connections failed")
    return all_ok


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_publisher(
    edition_type: Optional[EditionType] = None,
    target_date: Optional[date] = None,
    dry_run: bool = False,
    skip_selection: bool = False,
    skip_extraction: bool = False,
):
    """
    Main publishing pipeline with AI-based project deduplication.

    Pipeline:
    1. Fetch candidates from R2
    2. Extract project info (AI) - NO DB writes
    3. Filter against PUBLISHED projects only
    4. AI selection
    5. Send to Telegram
    6. Record publications and create projects (DB writes happen HERE)
    """
    if target_date is None:
        target_date = date.today()

    if edition_type is None:
        edition_type = determine_edition_type(target_date)
        if edition_type is None:
            print(f"\n[INFO] No publication scheduled for {target_date.strftime('%A')}")
            return

    edition_name = get_edition_display_name(edition_type)

    print(f"\n{'=' * 60}")
    print(f"ADUmedia Telegram Publisher")
    print(f"{'=' * 60}")
    print(f"Edition: {edition_name}")
    print(f"Date: {target_date.isoformat()} ({target_date.strftime('%A')})")
    if dry_run: print("[MODE] DRY RUN - will not send to Telegram or record")
    if skip_extraction: print("[MODE] SKIP EXTRACTION")
    print(f"{'=' * 60}")

    try:
        # Initialize services
        print("\n[INIT] Connecting...")
        r2 = R2Storage()
        r2.test_connection()
        dedup = DeduplicationChecker()

        # Step 1: Fetch candidates from R2
        dates_to_fetch = get_dates_for_edition(edition_type, target_date)
        print(f"\n[FETCH] {len(dates_to_fetch)} day(s)...")
        candidates = fetch_candidates_for_dates(r2, dates_to_fetch)

        if not candidates:
            print("\n[EMPTY] No candidates. Exiting.")
            return

        # Step 2: Extract project info (AI) - NO DB WRITES
        candidates = await extract_project_info(candidates, skip_extraction)

        # Step 3: Filter duplicates against PUBLISHED projects
        if not skip_extraction:
            eligible, duplicates, updates = await dedup.filter_candidates(
                candidates, PROJECT_COOLDOWN_MONTHS
            )

            if duplicates:
                print(f"\n[DEDUP] Filtered {len(duplicates)} duplicates:")
                for dup in duplicates[:5]:
                    print(f"   - {dup.get('title', '')[:50]}...")
                    print(f"     Reason: {dup.get('_duplicate_reason', 'Unknown')}")

            candidates = eligible

        if not candidates:
            print("\n[EMPTY] All filtered as duplicates. Exiting.")
            return

        # Step 4: AI selection
        if skip_selection:
            selected = candidates[:ARTICLES_PER_EDITION]
            edition_summary = "Selection skipped"
        else:
            selected = await select_articles(edition_type, candidates, dedup, target_date)
            edition_summary = f"Top {len(selected)} architecture news"

        if not selected:
            print("\n[EMPTY] Nothing selected. Exiting.")
            return

        # Step 5: Prepare for Telegram
        articles = prepare_articles_for_telegram(selected, r2)

        print(f"\n[READY] {len(articles)} articles:")
        for i, a in enumerate(articles, 1):
            img = "[IMG]" if a.get("hero_image") else "[TXT]"
            print(f"   {i}. {img} [{a.get('source_name', '')[:12]}] {a.get('title', '')[:40]}...")

        if dry_run:
            print("\n[DRY RUN] Complete. No Telegram send, no DB records.")
            return

        # Step 6: Send to Telegram
        print("\n[SEND] Sending to Telegram...")
        bot = TelegramBot()
        results = await bot.send_digest(articles, include_header=True)

        print(f"\n[RESULT] Sent: {results['sent']}, Failed: {results['failed']}")

        # Step 7: Record publications (PROJECTS CREATED HERE)
        # Only record articles that were actually sent (minus the header)
        if results['sent'] > 1:
            sent_articles = articles[:results['sent'] - 1]  # -1 for header
            await record_publications(
                dedup, sent_articles,
                edition_type, target_date, len(candidates), edition_summary
            )

        # Print final statistics
        from utils.rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()

        print(f"\n{'=' * 60}")
        print("[DONE] Publishing Complete")
        print(f"{'=' * 60}")
        print(f"Edition: {edition_name}")
        print(f"Articles sent: {results['sent']}")
        print(f"Total candidates processed: {len(candidates)}")
        limiter.print_stats()
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    args = parse_args()

    required_vars = [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID",
        "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
        "SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"[ERROR] Missing: {', '.join(missing)}")
        return

    if args.test:
        await test_connections()
        return

    target_date = date.fromisoformat(args.date) if args.date else None

    edition_type = None
    if args.edition:
        edition_type = {"daily": EditionType.DAILY, "weekend": EditionType.WEEKEND, "weekly": EditionType.WEEKLY}.get(args.edition)

    await run_publisher(
        edition_type=edition_type,
        target_date=target_date,
        dry_run=args.dry_run,
        skip_selection=args.skip_selection,
        skip_extraction=args.skip_extraction,
    )


if __name__ == "__main__":
    asyncio.run(main())