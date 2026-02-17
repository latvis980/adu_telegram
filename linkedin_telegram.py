# linkedin_telegram.py
"""
LinkedIn Post via Telegram (Temporary Solution)

Picks one article from today's published digest, formats it
in the new LinkedIn post format, and sends it to the admin
via Telegram DM (not to the channel).

Runs AFTER the daily digest is published.

Format:
    Riyadh Cultural District / OMA

    Masterplan / Saudi Arabia.

    Part of a wider wave of state-backed cultural infrastructure across the Gulf.

    Dezeen

    a/d/u -- curated for professionals
    daily selection:
    adu.media

Usage:
    python linkedin_telegram.py                # Normal run
    python linkedin_telegram.py --dry-run      # Pick + format, don't send
    python linkedin_telegram.py --date 2026-02-15

Environment Variables:
    TELEGRAM_BOT_TOKEN   - Telegram bot token (same as digest bot)
    SUPABASE_URL         - Supabase project URL
    SUPABASE_KEY         - Supabase API key
    R2_PUBLIC_URL        - R2 public URL for images
    OPENAI_API_KEY       - OpenAI API key (for context sentence)
"""

import asyncio
import argparse
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from supabase import create_client, Client
from telegram import Bot
from telegram.constants import ParseMode

from langchain_openai import ChatOpenAI


# =============================================================================
# Configuration
# =============================================================================

# Telegram admin user ID (Ksenia)
ADMIN_USER_ID = 176556234

# Edition type labels for the post footer
EDITION_LABELS = {
    "daily": "daily",
    "weekend": "weekend",
    "weekly": "weekly",
}


# =============================================================================
# Command Line Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LinkedIn Post via Telegram")

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pick and format but don't send to Telegram"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date (YYYY-MM-DD)"
    )

    return parser.parse_args()


# =============================================================================
# Pipeline Article Normalizer
# =============================================================================

def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize pipeline article fields to match Supabase field names.

    Pipeline articles use different field names (e.g. _selection_category,
    _extracted_info dict, hero_image dict, link). This function creates
    a unified view so downstream functions work with both sources.

    Does NOT modify the original article -- returns a new dict.
    """
    # If article already has Supabase-style fields, return as-is
    if article.get("selection_category") and article.get("extracted_location"):
        return article

    normalized = dict(article)  # shallow copy

    # Category: pipeline uses _selection_category
    if not normalized.get("selection_category"):
        normalized["selection_category"] = article.get("_selection_category", "")

    # Extracted info: pipeline stores in _extracted_info dict
    extracted = article.get("_extracted_info", {})
    if not normalized.get("extracted_project_name"):
        normalized["extracted_project_name"] = extracted.get("project_name", "")
    if not normalized.get("extracted_architect"):
        normalized["extracted_architect"] = extracted.get("architect", "")
    if not normalized.get("extracted_location"):
        normalized["extracted_location"] = extracted.get("location", "")

    # Article URL: pipeline uses "link"
    if not normalized.get("article_url"):
        normalized["article_url"] = article.get("link", "")

    # Original title: pipeline uses "title"
    if not normalized.get("original_title"):
        normalized["original_title"] = article.get("title", "")

    # Source name: pipeline may store it the same way, but double-check
    if not normalized.get("source_name"):
        normalized["source_name"] = article.get("source_name", "")

    return normalized


# =============================================================================
# Article Picker (simplified, from all_articles)
# =============================================================================

def pick_article(db: Client, target_date: date, articles_from_pipeline: list = None) -> Optional[Dict[str, Any]]:
    """
    Pick the article flagged by AI for LinkedIn.

    Priority:
    1. Article with _linkedin_pick = True (set by AI editor)
    2. Fallback: highest scored article from today's published articles

    Args:
        db: Supabase client
        target_date: Date to pick from
        articles_from_pipeline: Articles passed directly from the digest pipeline
            (with _linkedin_pick flag). If None, queries Supabase.

    Returns:
        Article dict or None
    """
    print(f"\n[PICKER] Looking for LinkedIn article for {target_date}...")

    # Option A: Articles passed from pipeline (has _linkedin_pick flag)
    if articles_from_pipeline:
        # Check for AI-flagged article
        for article in articles_from_pipeline:
            if article.get("_linkedin_pick"):
                normalized = normalize_article(article)
                headline = normalized.get("headline") or normalized.get("original_title", "")
                print(f"   [PICKER] AI-flagged article: {headline[:60]}")
                return normalized

        print(f"   [PICKER] No AI flag found in pipeline articles, using score fallback")
        # Fallback: score pipeline articles
        scored = [(score_article_from_pipeline(a), a) for a in articles_from_pipeline]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_article = scored[0]
        normalized = normalize_article(best_article)
        headline = normalized.get("headline") or normalized.get("original_title", "")
        print(f"   [PICKER] Score fallback: {headline[:60]} (score: {best_score})")
        return normalized

    # Option B: Query Supabase (standalone run)
    try:
        result = db.table("all_articles") \
            .select(
                "id, article_url, original_title, ai_summary, tags, "
                "source_name, source_id, r2_image_path, "
                "headline, adu_media_url, "
                "extracted_project_name, extracted_architect, extracted_location, "
                "selection_reason, selection_category, "
                "selected_for_editions, fetch_date"
            ) \
            .eq("status", "published") \
            .eq("fetch_date", target_date.isoformat()) \
            .execute()

        articles = result.data or []
    except Exception as e:
        print(f"   [PICKER] Error fetching articles: {e}")
        return None

    if not articles:
        print(f"   [PICKER] No published articles found for {target_date}")
        return None

    print(f"   [PICKER] Found {len(articles)} published articles")

    # Get already-posted LinkedIn article IDs
    posted_ids = set()
    try:
        result = db.table("linkedin_posts") \
            .select("article_id") \
            .execute()
        posted_ids = {row["article_id"] for row in result.data if row.get("article_id")}
    except Exception as e:
        print(f"   [PICKER] Note (linkedin_posts): {e}")

    # Filter out already posted
    candidates = [a for a in articles if a["id"] not in posted_ids]

    if not candidates:
        print(f"   [PICKER] All articles already posted to LinkedIn")
        return None

    print(f"   [PICKER] Eligible candidates: {len(candidates)}")

    # Score and rank
    scored = [(score_article(a), a) for a in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_article = scored[0]
    headline = best_article.get("headline") or best_article.get("original_title", "")
    print(f"   [PICKER] Selected: {headline[:60]} (score: {best_score})")

    return best_article


def score_article_from_pipeline(article: Dict[str, Any]) -> int:
    """Score a pipeline article (has image dict instead of r2_image_path)."""
    score = 0

    # Has image -- check both pipeline formats
    hero_image = article.get("hero_image")
    image_info = article.get("image", {})
    if hero_image and hero_image.get("r2_url"):
        score += 3
    elif image_info and image_info.get("has_image"):
        score += 3

    # Category
    category = (article.get("_selection_category") or "").lower()
    if category == "project":
        score += 3
    elif category in ("award", "competition"):
        score += 2
    elif category == "news":
        score += 1

    # Has headline
    if article.get("headline"):
        score += 1

    # Has architect (from extracted info)
    extracted = article.get("_extracted_info", {})
    if extracted.get("architect"):
        score += 1

    return score


def score_article(article: Dict[str, Any]) -> int:
    """Score an article for LinkedIn suitability. Higher = better."""
    score = 0

    # Has image (+3)
    if article.get("r2_image_path"):
        score += 3

    # Category scoring
    category = (article.get("selection_category") or "").lower()
    if category == "project":
        score += 3
    elif category in ("award", "competition"):
        score += 2
    elif category == "news":
        score += 1

    # Edition type scoring
    editions = article.get("selected_for_editions") or []
    if "weekly" in editions:
        score += 2
    elif "daily" in editions:
        score += 1

    # Has headline (+1)
    if article.get("headline"):
        score += 1

    # Has architect name (+1)
    if article.get("extracted_architect"):
        score += 1

    return score


# =============================================================================
# Context Sentence Generator (AI)
# =============================================================================

async def generate_context_sentence(article: Dict[str, Any]) -> str:
    """
    Generate a one-sentence contextual line using AI.

    Falls back to first sentence of ai_summary if AI fails.
    """
    # Load prompt
    prompt_template = _load_context_prompt()

    headline = article.get("headline") or article.get("original_title", "")
    project_name = article.get("extracted_project_name", "")
    architect = article.get("extracted_architect", "")
    location = article.get("extracted_location", "")
    category = article.get("selection_category", "")
    summary = article.get("ai_summary", "")
    source_name = article.get("source_name", "")

    prompt = prompt_template.format(
        headline=headline,
        project_name=project_name or "N/A",
        architect=architect or "N/A",
        location=location or "N/A",
        category=category or "N/A",
        ai_summary=summary,
        source_name=source_name,
    )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        print(f"   [CONTEXT] Generating context sentence...")
        response = await llm.ainvoke([("human", prompt)])
        sentence = response.content.strip().strip('"').strip("'")

        # Validate: should be one sentence, reasonable length
        # Allow up to 300 chars -- good context sentences can be 200-280 chars
        if len(sentence) < 20 or len(sentence) > 300:
            print(f"   [CONTEXT] AI output bad length ({len(sentence)}), using fallback")
            return _fallback_context(summary)

        # Ensure it ends with a period
        if not sentence.endswith("."):
            sentence += "."

        print(f"   [CONTEXT] Generated: {sentence[:80]}...")
        return sentence

    except Exception as e:
        print(f"   [CONTEXT] AI error: {e}. Using fallback.")
        return _fallback_context(summary)


def _load_context_prompt() -> str:
    """Load the context sentence prompt template."""
    possible_paths = [
        Path(__file__).parent / "prompts" / "linkedin_context.txt",
        Path("prompts/linkedin_context.txt"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"   [CONTEXT] Loaded prompt from {path}")
            return path.read_text(encoding="utf-8")

    print("[CONTEXT] Warning: prompt file not found, using inline fallback")
    return _get_fallback_context_prompt()


def _get_fallback_context_prompt() -> str:
    return """Write a single contextual sentence (15-25 words) about this architecture news.
Do NOT use any emoji. Do NOT start with the project name or architect name.
Provide broader context about why this matters to the profession.

Headline: {headline}
Summary: {ai_summary}
Architect: {architect}
Location: {location}

Write ONLY the sentence. Nothing else."""


def _fallback_context(summary: str) -> str:
    """Extract first sentence from ai_summary as fallback."""
    if not summary:
        return ""

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    if sentences:
        first = sentences[0].strip()
        # Truncate if too long (keep it concise for LinkedIn format)
        if len(first) > 250:
            first = first[:247] + "..."
        if not first.endswith((".", "...", "!", "?")):
            first += "."
        return first
    return summary.strip()


# =============================================================================
# Post Formatting
# =============================================================================

def get_edition_type_for_date(target_date: date) -> str:
    """Determine edition type from day of week."""
    weekday = target_date.weekday()  # 0=Mon
    if weekday == 0:
        return "weekly"
    elif weekday == 1:
        return "weekend"
    elif weekday in (2, 3, 4):
        return "daily"
    return "daily"


def format_category_line(article: Dict[str, Any]) -> str:
    """
    Build the category / location line.

    Examples:
        Masterplan / Saudi Arabia.
        Cultural / Stockholm.
        Residential / New York.
        Award / International.
    """
    # Category: use selection_category (already normalized)
    category = (article.get("selection_category") or "").strip()
    if category:
        category = category.capitalize()
    else:
        category = "Architecture"

    # Location (already normalized)
    location = (article.get("extracted_location") or "").strip()

    if category and location:
        return f"{category} / {location}."
    elif category:
        return f"{category}."
    elif location:
        return f"{location}."
    return ""


def format_linkedin_post(
    article: Dict[str, Any],
    context_sentence: str,
    edition_type: str,
) -> str:
    """
    Format the LinkedIn post in the new structured template.

    Format:
        {headline}

        {category} / {location}.

        {context_sentence}

        {source_name}

        a/d/u -- curated for professionals
        {edition_type} selection:
        adu.media
    """
    headline = article.get("headline") or article.get("original_title", "")
    source_name = article.get("source_name", "")
    category_line = format_category_line(article)
    edition_label = EDITION_LABELS.get(edition_type, "daily")

    parts = []

    # 1. Headline
    parts.append(headline)
    parts.append("")

    # 2. Category / Location
    if category_line:
        parts.append(category_line)
        parts.append("")

    # 3. Context sentence
    if context_sentence:
        parts.append(context_sentence)
        parts.append("")

    # 4. Source name
    if source_name:
        parts.append(source_name)
        parts.append("")

    # 5. Branding footer
    parts.append("a/d/u -- curated for professionals")
    parts.append(f"{edition_label} selection:")
    parts.append("adu.media")

    return "\n".join(parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML mode."""
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_telegram_message(
    article: Dict[str, Any],
    context_sentence: str,
    edition_type: str,
) -> str:
    """
    Format the Telegram message with HTML bold headline.

    Same structure as the LinkedIn post, but with <b> tags on the headline
    so it stands out visually in the Telegram DM.
    """
    headline = article.get("headline") or article.get("original_title", "")
    source_name = article.get("source_name", "")
    category_line = format_category_line(article)
    edition_label = EDITION_LABELS.get(edition_type, "daily")

    parts = []

    # 1. Headline (bold in Telegram)
    parts.append(f"<b>{_escape_html(headline)}</b>")
    parts.append("")

    # 2. Category / Location
    if category_line:
        parts.append(_escape_html(category_line))
        parts.append("")

    # 3. Context sentence
    if context_sentence:
        parts.append(_escape_html(context_sentence))
        parts.append("")

    # 4. Source name
    if source_name:
        parts.append(_escape_html(source_name))
        parts.append("")

    # 5. Branding footer
    parts.append("a/d/u -- curated for professionals")
    parts.append(f"{edition_label} selection:")
    parts.append("adu.media")

    return "\n".join(parts)


# =============================================================================
# Image URL Helper
# =============================================================================

def get_image_url(article: Dict[str, Any]) -> Optional[str]:
    """
    Get the public image URL for the article.

    Handles both pipeline articles (hero_image dict) and
    Supabase articles (r2_image_path string).
    """
    # Option A: Pipeline article with hero_image dict (already a full URL)
    hero_image = article.get("hero_image")
    if hero_image:
        r2_url = hero_image.get("r2_url")
        if r2_url and r2_url.startswith("http"):
            return r2_url
        # Fallback to original URL
        original_url = hero_image.get("url")
        if original_url and original_url.startswith("http"):
            return original_url

    # Option B: Pipeline article with image dict
    image_info = article.get("image", {})
    if image_info:
        r2_path = image_info.get("r2_path")
        if r2_path:
            public_url = os.getenv("R2_PUBLIC_URL")
            if public_url:
                clean_path = r2_path.lstrip("/")
                return f"{public_url.rstrip('/')}/{clean_path}"

    # Option C: Supabase article with r2_image_path string
    r2_image_path = article.get("r2_image_path")
    if not r2_image_path:
        return None

    public_url = os.getenv("R2_PUBLIC_URL")
    if not public_url:
        print("   [WARN] R2_PUBLIC_URL not set, cannot generate image URL")
        return None

    clean_path = r2_image_path.lstrip("/")
    return f"{public_url.rstrip('/')}/{clean_path}"


# =============================================================================
# Telegram Sender (to admin DM, not channel)
# =============================================================================

async def send_to_admin(
    bot_token: str,
    telegram_text: str,
    image_url: Optional[str] = None,
) -> bool:
    """
    Send the formatted LinkedIn post to the admin via Telegram DM.

    Sends image first (if available), then the text with HTML formatting
    (bold headline) as a separate message so admin can see the structure.
    """
    bot = Bot(token=bot_token)

    try:
        # Send image first if available
        if image_url:
            print(f"   [SEND] Sending image to admin...")
            try:
                await bot.send_photo(
                    chat_id=ADMIN_USER_ID,
                    photo=image_url,
                )
            except Exception as img_err:
                print(f"   [SEND] Image send failed: {img_err}")
                # Continue -- still send the text
            # Small delay between messages
            await asyncio.sleep(1)

        # Send text with HTML formatting (bold headline)
        print(f"   [SEND] Sending post text to admin...")
        await bot.send_message(
            chat_id=ADMIN_USER_ID,
            text=telegram_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

        print(f"   [SEND] Sent to admin (user {ADMIN_USER_ID})")
        return True

    except Exception as e:
        print(f"   [SEND] Error sending to admin: {e}")
        return False


# =============================================================================
# Record in linkedin_posts Table
# =============================================================================

def _is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def record_linkedin_post(
    db: Client,
    article: Dict[str, Any],
    post_text: str,
    status: str = "sent_to_admin",
) -> None:
    """Record the LinkedIn post in Supabase for tracking."""
    try:
        # Get article_id -- only use if it's a valid UUID
        # Pipeline articles have non-UUID IDs like "metalocus_001"
        article_id = article.get("id")
        if article_id and not _is_valid_uuid(article_id):
            print(f"   [RECORD] Article ID '{article_id}' is not a UUID, setting to null")
            article_id = None

        data = {
            "article_id": article_id,
            "article_url": article.get("article_url") or article.get("link", ""),
            "adu_media_url": article.get("adu_media_url"),
            "linkedin_post_urn": None,  # Not posted to LinkedIn yet
            "post_text": post_text[:2000],
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "edition_type": _get_edition_type(article),
            "status": status,
            "error_message": None,
        }

        db.table("linkedin_posts").insert(data).execute()
        print(f"   [RECORD] Saved to linkedin_posts table")

    except Exception as e:
        print(f"   [RECORD] Error saving: {e}")


def _get_edition_type(article: Dict[str, Any]) -> str:
    """Extract edition type from article data."""
    editions = article.get("selected_for_editions") or []
    if "weekly" in editions:
        return "weekly"
    if "weekend" in editions:
        return "weekend"
    return "daily"


# =============================================================================
# Main Pipeline
# =============================================================================

async def run(target_date: Optional[date] = None, dry_run: bool = False, articles_from_pipeline: list = None):
    """
    Main pipeline:
    1. Pick best article from today's digest
    2. Generate context sentence (AI)
    3. Format LinkedIn post (plain) + Telegram message (HTML bold headline)
    4. Send image + text to admin via Telegram
    5. Record in linkedin_posts table
    """
    if target_date is None:
        target_date = date.today()

    edition_type = get_edition_type_for_date(target_date)

    print(f"\n{'=' * 60}")
    print(f"LinkedIn Post via Telegram")
    print(f"{'=' * 60}")
    print(f"Date: {target_date.isoformat()} ({target_date.strftime('%A')})")
    print(f"Edition: {edition_type}")
    if dry_run:
        print("[MODE] DRY RUN -- will not send or record")
    print(f"{'=' * 60}")

    # Init Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY required")
        return

    db = create_client(supabase_url, supabase_key)

    # Step 1: Pick article (uses AI flag if available from pipeline)
    article = pick_article(db, target_date, articles_from_pipeline)
    if not article:
        print("\n[EMPTY] No suitable article found. Exiting.")
        return

    # Step 2: Generate context sentence
    context_sentence = await generate_context_sentence(article)

    # Step 3a: Format plain-text LinkedIn post (for recording / copy-paste)
    post_text = format_linkedin_post(article, context_sentence, edition_type)

    # Step 3b: Format Telegram message with HTML bold headline
    telegram_text = format_telegram_message(article, context_sentence, edition_type)

    # Step 4: Get image URL (handles both pipeline and Supabase articles)
    image_url = get_image_url(article)

    # Print preview
    print(f"\n{'=' * 60}")
    print("[PREVIEW] LinkedIn Post:")
    print(f"{'=' * 60}")
    print(post_text)
    print(f"{'=' * 60}")
    if image_url:
        print(f"[IMAGE] {image_url[:80]}...")
    else:
        print("[IMAGE] No image available")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[DRY RUN] Complete. Not sending.")
        return

    # Step 5: Send to admin via Telegram
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("[ERROR] TELEGRAM_BOT_TOKEN required")
        return

    success = await send_to_admin(bot_token, telegram_text, image_url)

    if success:
        # Step 6: Record in linkedin_posts (plain text version, not HTML)
        record_linkedin_post(db, article, post_text, status="sent_to_admin")
        print("\n[DONE] LinkedIn post sent to admin and recorded.")
    else:
        record_linkedin_post(db, article, post_text, status="send_failed")
        print("\n[ERROR] Failed to send to admin.")


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    args = parse_args()

    required_vars = ["TELEGRAM_BOT_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        return

    target_date = date.fromisoformat(args.date) if args.date else None

    await run(target_date=target_date, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
