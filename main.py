# main.py
"""
ADUmedia Telegram Publisher Service

Intelligent publishing service with AI-BASED PROJECT DEDUPLICATION:
1. Fetches candidates from R2
2. Extracts project info from summaries using AI
3. Uses AI to match articles to existing projects (fuzzy matching)
4. Filters duplicates (same project < 3 months = skip)
5. Uses AI to select top 7 articles
6. Records everything in Supabase
7. Sends digest to Telegram

The AI matching handles:
- Name variations ("The Spiral" vs "Spiral Tower")
- Capitalization differences
- Architect name variations ("BIG" vs "Bjarke Ingels Group")
- Location variations ("NYC" vs "New York")

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
        "--no-archive", action="store_true",
        help="Don't archive articles"
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
    """Get the dates to fetch candidates for based on edition type."""
    if edition_type == EditionType.DAILY:
        return [target_date]
    elif edition_type == EditionType.WEEKEND:
        return [target_date - timedelta(days=i) for i in range(3, 0, -1)]
    elif edition_type == EditionType.WEEKLY:
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
# Project Extraction & Matching Pipeline
# =============================================================================

async def extract_and_link_projects(
    candidates: List[Dict[str, Any]],
    dedup: DeduplicationChecker,
    skip_extraction: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract project info and link articles to projects using AI.

    This is the key step for project-based deduplication:
    1. Extract project name, architect, location from summary (AI)
    2. Find existing project using AI fuzzy matching
    3. Or create new project if no match
    4. Attach project_id to candidate for dedup checking

    Args:
        candidates: List of candidate articles
        dedup: DeduplicationChecker instance (with AI matching)
        skip_extraction: Skip AI extraction (for testing)

    Returns:
        Candidates with _project_id and _extracted_info attached
    """
    if skip_extraction:
        print("[EXTRACT] Skipping project extraction (--skip-extraction)")
        return candidates

    print(f"\n[EXTRACT] Extracting project info from {len(candidates)} articles...")

    extractor = ProjectExtractor()

    # Extract project info in batches
    extractions = await extractor.extract_batch(candidates, batch_size=10)

    # Link to projects using AI matching
    projects_created = 0
    projects_matched = 0
    non_projects = 0

    for candidate, extraction in zip(candidates, extractions):
        candidate["_extracted_info"] = extraction.model_dump()

        if not extraction.is_project or not extraction.project_name:
            # Not a project article (news, interview, etc.)
            non_projects += 1
            candidate["_project_id"] = None
            continue

        # Format location
        location_city = None
        location_country = None
        if extraction.location:
            location_city = extraction.location.city
            location_country = extraction.location.country

        # Find or create project using AI matching
        project_id, is_new = await dedup.find_or_create_project(
            project_name=extraction.project_name,
            architect=extraction.architect,
            location_city=location_city,
            location_country=location_country,
            project_type=extraction.project_type,
            project_status=extraction.project_status,
            summary_excerpt=candidate.get("ai_summary", "")[:200],
        )

        candidate["_project_id"] = project_id

        if is_new:
            projects_created += 1
        else:
            projects_matched += 1

    print(f"   [OK] Projects: {projects_created} new, {projects_matched} matched, {non_projects} non-project articles")

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

    # Get previously published project IDs for context
    published_projects = dedup.get_published_project_ids(
        since_date=target_date - timedelta(days=90)
    )
    print(f"   Recently published projects: {len(published_projects)}")

    # Mark candidates that are about recently-published projects
    for candidate in candidates:
        project_id = candidate.get("_project_id")
        if project_id and project_id in published_projects:
            candidate["_recently_published"] = True

    # Run AI selection (using existing selector)
    published_urls = []  # We use project-based dedup now

    selection = await selector.select(
        edition_type=edition_type,
        candidates=candidates,
        published_urls=published_urls,
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
            "title": article.get("title", ""),
            "link": article.get("link", ""),
            "ai_summary": article.get("ai_summary", ""),
            "tags": article.get("tags", []),
            "source_id": article.get("source_id", ""),
            "source_name": article.get("source_name", article.get("source_id", "Unknown")),
            "published": article.get("published"),
            "_fetch_date": article.get("_fetch_date"),
            "_project_id": article.get("_project_id"),
            "_selection_reason": article.get("_selection_reason"),
            "_selection_category": article.get("_selection_category"),
        }

        image_info = article.get("image", {})
        if image_info and image_info.get("has_image") and image_info.get("r2_path"):
            r2_path = image_info["r2_path"]
            if r2.public_url:
                image_url = f"{r2.public_url.rstrip('/')}/{r2_path}"
                telegram_article["hero_image"] = {
                    "url": image_info.get("original_url"),
                    "r2_url": image_url,
                    "r2_path": r2_path,
                }
            else:
                # No public R2 URL configured - don't include image to avoid external URLs
                print(f"   [WARN] No R2 public URL configured, skipping image for: {telegram_article['title'][:40]}...")
                telegram_article["hero_image"] = None
        else:
            telegram_article["hero_image"] = None

        if telegram_article["ai_summary"]:
            prepared.append(telegram_article)
        else:
            print(f"   [WARN] Skipping article without summary: {telegram_article['title'][:40]}...")

    return prepared


# =============================================================================
# Publication Recording
# =============================================================================

def record_publications(
    dedup: DeduplicationChecker,
    articles: List[Dict[str, Any]],
    edition_type: EditionType,
    edition_date: date,
    total_candidates: int,
    edition_summary: Optional[str] = None
) -> List[str]:
    """Record publications in database."""
    print(f"\n[RECORD] Recording {len(articles)} publications...")

    article_ids = []
    articles_new = 0
    articles_repeated = 0

    for article in articles:
        project_id = article.get("_project_id")

        # Record article in all_articles
        article_id = dedup.record_article(
            article=article,
            project_id=project_id,
            extracted_info=article.get("_extracted_info"),
            status="published"
        )

        if article_id:
            article_ids.append(article_id)

            # Mark as published
            dedup.mark_article_published(
                article_id=article_id,
                edition_type=edition_type.value,
                edition_date=edition_date
            )

            # Update project's last_published_date
            if project_id:
                is_dup, last_published = dedup.check_project_duplicate(project_id)
                if last_published:
                    articles_repeated += 1
                else:
                    articles_new += 1

                dedup.update_project_published(project_id, edition_date)
            else:
                articles_new += 1

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

    print(f"   [OK] Recorded: {articles_new} new projects, {articles_repeated} updates")
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
            continue

        success = r2.archive_article(article_id, fetch_date)
        results[article_id] = success

    archived = sum(1 for v in results.values() if v)
    print(f"   [OK] Archived: {archived}/{len(results)}")
    return results


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
    no_archive: bool = False,
    skip_selection: bool = False,
    skip_extraction: bool = False,
):
    """Main publishing pipeline with AI-based project deduplication."""
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
    if dry_run: print("[MODE] DRY RUN")
    if skip_extraction: print("[MODE] SKIP EXTRACTION")
    print(f"{'=' * 60}")

    try:
        # Initialize services
        print("\n[INIT] Connecting...")
        r2 = R2Storage()
        r2.test_connection()
        dedup = DeduplicationChecker()

        # Fetch candidates
        dates_to_fetch = get_dates_for_edition(edition_type, target_date)
        print(f"\n[FETCH] {len(dates_to_fetch)} day(s)...")
        candidates = fetch_candidates_for_dates(r2, dates_to_fetch)

        if not candidates:
            print("\n[EMPTY] No candidates. Exiting.")
            return

        # Extract and link projects (AI-based)
        candidates = await extract_and_link_projects(candidates, dedup, skip_extraction)

        # Filter duplicates
        if not skip_extraction:
            eligible, duplicates, updates = await dedup.filter_candidates(candidates, PROJECT_COOLDOWN_MONTHS)

            if duplicates:
                print(f"\n[DEDUP] Filtered {len(duplicates)} duplicates:")
                for dup in duplicates[:3]:
                    print(f"   - {dup.get('title', '')[:50]}... ({dup.get('_duplicate_reason', '')})")

            candidates = eligible

        if not candidates:
            print("\n[EMPTY] All filtered. Exiting.")
            return

        # AI selection
        if skip_selection:
            selected = candidates[:ARTICLES_PER_EDITION]
            edition_summary = "Selection skipped"
        else:
            selected = await select_articles(edition_type, candidates, dedup, target_date)
            edition_summary = f"Top {len(selected)} architecture news"

        if not selected:
            print("\n[EMPTY] Nothing selected. Exiting.")
            return

        # Prepare for Telegram
        articles = prepare_articles_for_telegram(selected, r2)

        print(f"\n[READY] {len(articles)} articles:")
        for i, a in enumerate(articles, 1):
            img = "[IMG]" if a.get("hero_image") else "[TXT]"
            print(f"   {i}. {img} [{a.get('source_name', '')[:12]}] {a.get('title', '')[:40]}...")

        if dry_run:
            print("\n[DRY RUN] Complete.")
            return

        # Send to Telegram
        print("\n[SEND] Sending...")
        bot = TelegramBot()
        results = await bot.send_digest(articles, include_header=True)

        print(f"\n[RESULT] Sent: {results['sent']}, Failed: {results['failed']}")

        # Record publications
        if results['sent'] > 1:
            record_publications(
                dedup, articles[:results['sent'] - 1],
                edition_type, target_date, len(candidates), edition_summary
            )

        # Archive
        if not no_archive and results['sent'] > 1:
            archive_sent_articles(r2, articles[:results['sent'] - 1])

        print(f"\n{'=' * 60}")
        print("[DONE]")
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
        no_archive=args.no_archive,
        skip_selection=args.skip_selection,
        skip_extraction=args.skip_extraction,
    )


if __name__ == "__main__":
    asyncio.run(main())