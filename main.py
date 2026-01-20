# run_telegram.py
"""
ADUmedia Telegram Publisher Service

Standalone service that fetches today's articles from R2 storage
and publishes them to the Telegram channel.

Schedule: After RSS and Custom Scrapers complete (e.g., 19:00 Lisbon time)

Pipeline:
    1. Connect to R2 storage
    2. Fetch today's candidates (or selected digest if available)
    3. Format and send to Telegram channel

Usage:
    python run_telegram.py              # Send today's articles
    python run_telegram.py --test       # Test connections only
    python run_telegram.py --date 2026-01-20  # Send specific date

Environment Variables (set in Railway):
    TELEGRAM_BOT_TOKEN      - Telegram bot token from BotFather
    TELEGRAM_CHANNEL_ID     - Telegram channel ID (@channel or -100xxx)
    R2_ACCOUNT_ID           - Cloudflare R2 account ID
    R2_ACCESS_KEY_ID        - R2 access key
    R2_SECRET_ACCESS_KEY    - R2 secret key
    R2_BUCKET_NAME          - R2 bucket name (adumedia)
    R2_PUBLIC_URL           - R2 public URL (optional, for image URLs)
"""

import asyncio
import argparse
import os
from datetime import datetime, date
from typing import Optional, List

# Import modules
from storage.r2 import R2Storage
from telegram_bot import TelegramBot


# =============================================================================
# Configuration
# =============================================================================

# Default: use today's date
DEFAULT_DATE = None  # None means today


# =============================================================================
# Command Line Arguments
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ADUmedia Telegram Publisher"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connections only (R2 and Telegram)"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to publish (YYYY-MM-DD format)"
    )

    parser.add_argument(
        "--use-selected",
        action="store_true",
        default=False,
        help="Use selected digest instead of all candidates"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch articles but don't send to Telegram"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to send"
    )

    return parser.parse_args()


# =============================================================================
# R2 Article Fetching
# =============================================================================

def fetch_articles_from_r2(
    r2: R2Storage,
    target_date: Optional[date] = None,
    use_selected: bool = False
) -> List[dict]:
    """
    Fetch articles from R2 storage for the given date.

    Args:
        r2: R2Storage instance
        target_date: Date to fetch (defaults to today)
        use_selected: If True, use selected digest; otherwise use all candidates

    Returns:
        List of article dicts ready for Telegram
    """
    if target_date is None:
        target_date = date.today()

    print(f"\n📥 Fetching articles for {target_date.isoformat()}...")

    if use_selected:
        # Try to get curated selection first
        digest = r2.get_selected_digest(target_date)
        if digest:
            articles = digest.get("articles", [])
            print(f"   ✅ Found selected digest: {len(articles)} articles")
            return _prepare_articles_for_telegram(articles, r2)
        else:
            print("   ⚠️ No selected digest found, falling back to candidates")

    # Get all candidates
    candidates = r2.get_all_candidates(target_date)

    if not candidates:
        # Fallback: try to get manifest and load manually
        manifest = r2.get_manifest(target_date)
        if manifest:
            print(f"   📋 Found manifest with {manifest.get('total_candidates', 0)} candidates")
            for entry in manifest.get("candidates", []):
                article_id = entry.get("id")
                if article_id:
                    candidate = r2.get_candidate(article_id, target_date)
                    if candidate:
                        candidates.append(candidate)

    print(f"   ✅ Found {len(candidates)} candidates")

    return _prepare_articles_for_telegram(candidates, r2)


def _prepare_articles_for_telegram(articles: List[dict], r2: R2Storage) -> List[dict]:
    """
    Prepare articles for Telegram sending.

    Ensures all required fields are present and formats hero_image URLs.

    Args:
        articles: Raw article dicts from R2
        r2: R2Storage instance (for public URL)

    Returns:
        List of articles formatted for TelegramBot
    """
    prepared = []

    for article in articles:
        # Build telegram-ready article dict
        telegram_article = {
            "title": article.get("title", ""),
            "link": article.get("link", ""),
            "ai_summary": article.get("ai_summary", ""),
            "tags": article.get("tags", []),
            "source_id": article.get("source_id", ""),
            "source_name": article.get("source_name", article.get("source_id", "Unknown")),
            "published": article.get("published"),
        }

        # Handle hero image
        image_info = article.get("image", {})
        if image_info.get("has_image") and image_info.get("r2_path"):
            # Build public URL for the image
            r2_path = image_info["r2_path"]
            if r2.public_url:
                image_url = f"{r2.public_url.rstrip('/')}/{r2_path}"
            else:
                # Fallback to original URL if no public URL configured
                image_url = image_info.get("original_url")

            telegram_article["hero_image"] = {
                "url": image_info.get("original_url"),
                "r2_url": image_url,
                "r2_path": r2_path,
            }
        else:
            telegram_article["hero_image"] = None

        # Only include articles with summaries
        if telegram_article["ai_summary"]:
            prepared.append(telegram_article)
        else:
            print(f"   ⚠️ Skipping article without summary: {telegram_article['title'][:40]}...")

    return prepared


# =============================================================================
# Connection Testing
# =============================================================================

async def test_connections() -> bool:
    """
    Test R2 and Telegram connections.

    Returns:
        True if all connections successful
    """
    print("\n🧪 Testing Connections...")
    print("=" * 50)

    all_ok = True

    # Test R2
    print("\n1. Testing R2 Storage...")
    try:
        r2 = R2Storage()
        r2.test_connection()
        print("   ✅ R2 connection OK")
    except Exception as e:
        print(f"   ❌ R2 connection failed: {e}")
        all_ok = False

    # Test Telegram
    print("\n2. Testing Telegram Bot...")
    try:
        bot = TelegramBot()
        if await bot.test_connection():
            print("   ✅ Telegram connection OK")
        else:
            print("   ❌ Telegram connection failed")
            all_ok = False
    except Exception as e:
        print(f"   ❌ Telegram connection failed: {e}")
        all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("✅ All connections successful!")
    else:
        print("❌ Some connections failed. Check credentials.")

    return all_ok


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_telegram_publisher(
    target_date: Optional[date] = None,
    use_selected: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
):
    """
    Main Telegram publishing pipeline.

    Args:
        target_date: Date to publish (defaults to today)
        use_selected: Use curated selection instead of all candidates
        dry_run: Fetch but don't send
        limit: Maximum number of articles to send
    """
    if target_date is None:
        target_date = date.today()

    # Log pipeline start
    print(f"\n{'=' * 60}")
    print("📱 ADUmedia Telegram Publisher")
    print(f"{'=' * 60}")
    print(f"📅 {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    print(f"📆 Publishing date: {target_date.isoformat()}")
    print(f"🎯 Mode: {'Selected digest' if use_selected else 'All candidates'}")
    if dry_run:
        print("⚠️  DRY RUN - No messages will be sent")
    print(f"{'=' * 60}")

    try:
        # Initialize R2
        print("\n📦 Connecting to R2 storage...")
        r2 = R2Storage()
        r2.test_connection()

        # Fetch articles
        articles = fetch_articles_from_r2(r2, target_date, use_selected)

        if not articles:
            print("\n📭 No articles found for this date. Exiting.")
            return

        # Apply limit if specified
        if limit and len(articles) > limit:
            print(f"\n⚠️ Limiting to {limit} articles (found {len(articles)})")
            articles = articles[:limit]

        print(f"\n📰 Ready to publish {len(articles)} articles")

        # Preview articles
        print("\n📋 Articles to publish:")
        for i, article in enumerate(articles, 1):
            has_image = "🖼️" if article.get("hero_image") else "📄"
            source = article.get("source_name", "Unknown")[:15]
            title = article.get("title", "No title")[:40]
            print(f"   {i}. {has_image} [{source}] {title}...")

        if dry_run:
            print("\n⚠️ DRY RUN complete. No messages sent.")
            return

        # Send to Telegram
        print("\n📤 Sending to Telegram...")
        bot = TelegramBot()

        results = await bot.send_digest(articles, include_header=True)

        # Print results
        print(f"\n{'=' * 60}")
        print("📊 Results")
        print(f"{'=' * 60}")
        print(f"   ✅ Sent: {results['sent']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ⏱️  Time: {results['total_time']/60:.1f} minutes")
        if results.get('flood_retries', 0) > 0:
            print(f"   🔄 Flood retries: {results['flood_retries']}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        raise


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    args = parse_args()

    # Validate environment
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHANNEL_ID",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in Railway dashboard.")
        return

    # Test mode
    if args.test:
        await test_connections()
        return

    # Parse date if provided
    target_date = None
    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("Use YYYY-MM-DD format (e.g., 2026-01-20)")
            return

    # Run publisher
    await run_telegram_publisher(
        target_date=target_date,
        use_selected=args.use_selected,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    asyncio.run(main())
