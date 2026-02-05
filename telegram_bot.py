# telegram_bot.py
"""
Telegram Bot Module
Handles all communication between backend and Telegram interface.

Flood Control:
    - Telegram limits: 20 messages/minute to groups/channels
    - Safe rate: 1 message every 3-4 seconds
    - Handles RetryAfter errors with exponential backoff

Usage:
    from telegram_bot import TelegramBot

    bot = TelegramBot()
    await bot.send_digest(articles)
"""

import os
import re
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter, TimedOut, NetworkError

# Import source registry
from config.sources import get_source_name


# =============================================================================
# Flood Control Constants
# =============================================================================

# Telegram limits for groups/channels: 20 messages per minute
# Safe interval: 60 / 20 = 3 seconds minimum
# We use 3.5 seconds to have some buffer
CHANNEL_MESSAGE_INTERVAL: float = 3.5  # seconds between messages

# Maximum retries for rate limit errors
MAX_RETRIES: int = 5

# Base delay for exponential backoff (seconds)
BASE_RETRY_DELAY: float = 5.0

# Maximum delay between retries (seconds)
MAX_RETRY_DELAY: float = 120.0


# =============================================================================
# Edition Type Labels (for header formatting)
# =============================================================================

EDITION_LABELS = {
    "daily": "Daily Edition",
    "weekend": "Weekend Catch-Up",
    "weekly": "Weekly Edition",
}


class TelegramBot:
    """
    Handles all Telegram bot operations with flood control.

    Respects Telegram's rate limits:
    - 20 messages/minute for groups/channels
    - Automatic retry with exponential backoff on RetryAfter errors
    """

    def __init__(self, token: str | None = None, channel_id: str | None = None):
        """
        Initialize Telegram bot.

        Args:
            token: Bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            channel_id: Channel ID (defaults to TELEGRAM_CHANNEL_ID env var)
        """
        self.token: str = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.channel_id: str = channel_id or os.getenv("TELEGRAM_CHANNEL_ID", "")

        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        if not self.channel_id:
            raise ValueError("TELEGRAM_CHANNEL_ID not set")

        self.bot = Bot(token=self.token)

        # Flood control tracking
        self._last_message_time: float = 0.0
        self._message_count: int = 0
        self._retry_count: int = 0
        self._flood_wait_total: float = 0.0

    # =========================================================================
    # Flood Control
    # =========================================================================

    async def _wait_for_rate_limit(self) -> None:
        """
        Wait if needed to respect Telegram's rate limits.

        Ensures at least CHANNEL_MESSAGE_INTERVAL seconds between messages.
        """
        now = time.time()
        elapsed = now - self._last_message_time

        if elapsed < CHANNEL_MESSAGE_INTERVAL:
            wait_time = CHANNEL_MESSAGE_INTERVAL - elapsed
            await asyncio.sleep(wait_time)

        self._last_message_time = time.time()

    async def _send_with_retry(
        self, 
        send_func,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ) -> Optional[object]:
        """
        Send a message with automatic retry on rate limit errors.

        Args:
            send_func: Async function to call (e.g., self.bot.send_message)
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for send_func

        Returns:
            The Message object if sent successfully, None otherwise
        """
        # Wait for rate limit before attempting
        await self._wait_for_rate_limit()

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                result = await send_func(**kwargs)
                self._message_count += 1
                return result

            except RetryAfter as e:
                # Telegram is telling us to wait
                # e.retry_after can be int or timedelta
                retry_after = e.retry_after
                if isinstance(retry_after, timedelta):
                    wait_seconds: float = retry_after.total_seconds()
                else:
                    wait_seconds = float(retry_after)

                self._retry_count += 1
                self._flood_wait_total += wait_seconds

                print(f"   [FLOOD] Rate limited. Waiting {wait_seconds}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait_seconds + 1.0)  # Add 1 second buffer

                # Update last message time after waiting
                self._last_message_time = time.time()
                last_error = e

            except TimedOut as e:
                # Network timeout - retry with backoff
                if attempt < max_retries:
                    delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"   [TIMEOUT] Request timed out. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    print(f"   [ERROR] Timeout after {max_retries + 1} attempts")
                    return None

            except NetworkError as e:
                # Network error - retry with backoff
                if attempt < max_retries:
                    delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"   [NETWORK] Network error. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    print(f"   [ERROR] Network error after {max_retries + 1} attempts: {e}")
                    return None

            except TelegramError as e:
                # Other Telegram errors
                print(f"   [ERROR] Telegram error: {e}")
                return None

        # If we get here, all retries failed
        print(f"   [ERROR] Failed after {max_retries + 1} attempts: {last_error}")
        return None

    def print_flood_stats(self) -> None:
        """Print flood control statistics."""
        print(f"   Flood control stats:")
        print(f"   - Messages sent: {self._message_count}")
        print(f"   - Flood retries: {self._retry_count}")
        print(f"   - Total flood wait: {self._flood_wait_total:.1f}s")

    # =========================================================================
    # Markdown Escaping
    # =========================================================================

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """
        Escape special characters in Markdown text.

        Telegram MarkdownV2 requires escaping: _*[]()~`>#+-=|{}.!

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for Telegram Markdown
        """
        if not text:
            return ""

        # Characters that need escaping in MarkdownV2
        special_chars = '_*[]()~`>#+-=|{}.!'

        result = text
        for char in special_chars:
            result = result.replace(char, f'\\{char}')

        return result

    @staticmethod
    def _escape_url(url: str) -> str:
        """
        Escape special characters in URLs for Markdown links.

        Parentheses in URLs break Markdown link syntax.

        Args:
            url: URL to escape

        Returns:
            URL with escaped parentheses
        """
        if not url:
            return ""

        # Escape parentheses which break Markdown link syntax
        return url.replace('(', '%28').replace(')', '%29')

    # =========================================================================
    # Message Sending
    # =========================================================================

    async def send_message(
        self, 
        text: str, 
        parse_mode: str = ParseMode.HTML,  # Changed from MARKDOWN
        disable_preview: bool = False
    ) -> Optional[int]:
        """
        Send a single message to the channel with flood control.

        Args:
            text: Message text
            parse_mode: Telegram parse mode (Markdown/HTML)
            disable_preview: Disable link preview

        Returns:
            message_id if sent successfully, None otherwise
        """
        # Try with formatting first
        result = await self._send_with_retry(
            self.bot.send_message,
            chat_id=self.channel_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_preview
        )

        if result:
            return result.message_id

        # Fallback: try without formatting
        print("   [INFO] Retrying without Markdown formatting...")
        result = await self._send_with_retry(
            self.bot.send_message,
            chat_id=self.channel_id,
            text=text,
            parse_mode=None,
            disable_web_page_preview=disable_preview
        )

        return result.message_id if result else None

    async def send_photo(
        self,
        photo_url: str,
        caption: str | None = None,
        parse_mode: str = ParseMode.HTML  # Changed from MARKDOWN to HTML
    ) -> bool:
        """
        Send a photo with optional caption to the channel with flood control.

        Args:
            photo_url: URL of the image to send
            caption: Optional caption text
            parse_mode: Telegram parse mode (HTML)

        Returns:
            True if sent successfully
        """
        # Try with formatting first
        result = await self._send_with_retry(
            self.bot.send_photo,
            chat_id=self.channel_id,
            photo=photo_url,
            caption=caption,
            parse_mode=parse_mode
        )

        if result:
            return True

        # Fallback: try without formatting
        print("   [INFO] Retrying photo without HTML formatting...")
        result = await self._send_with_retry(
            self.bot.send_photo,
            chat_id=self.channel_id,
            photo=photo_url,
            caption=caption,
            parse_mode=None
        )

        return result is not None


    # =========================================================================
    # Digest Sending
    # =========================================================================

    async def send_digest(
        self, 
        articles: list[dict],
        include_header: bool = True,
        edition_type: str = "daily"
    ) -> dict:
        """
        Send a news digest to the channel with flood control.

        Respects Telegram's 20 messages/minute limit for channels.
        Estimated time: ~3.5 seconds per article.

        Args:
            articles: List of article dicts with keys:
                - link: Article URL
                - headline: Article headline (PROJECT NAME / ARCHITECT)
                - ai_summary: AI-generated summary
                - tag: Single tag (optional)
                - hero_image: Dict with 'url' or 'r2_url' (optional)
            include_header: Whether to send daily header message
            edition_type: Edition type string ("daily", "weekend", "weekly")

        Returns:
            Dict with sent/failed counts, timing info, and header_message_id
        """
        results: dict = {
            "sent": 0, 
            "failed": 0,
            "total_time": 0.0,
            "flood_retries": 0,
            "header_message_id": None,
        }

        if not articles:
            print("[INFO] No articles to send")
            return results

        # Estimate time
        total_messages = len(articles) + (1 if include_header else 0)
        estimated_time = total_messages * CHANNEL_MESSAGE_INTERVAL
        print(f"[INFO] Sending {total_messages} messages (estimated time: {estimated_time/60:.1f} minutes)")

        start_time = time.time()

        # Send header (optional)
        if include_header:
            header = self._format_header(len(articles), edition_type)
            header_message_id = await self.send_message(header, disable_preview=False)
            if header_message_id:
                results["sent"] += 1
                results["header_message_id"] = header_message_id
            else:
                results["failed"] += 1

        # Send each article
        for i, article in enumerate(articles, 1):
            source_name = article.get("source_name", "Unknown")
            title_preview = article.get("title", "")[:30]
            print(f"   [{i}/{len(articles)}] [{source_name}] {title_preview}...")

            success = await self._send_article(article)

            if success:
                results["sent"] += 1
            else:
                results["failed"] += 1

        # Calculate timing
        results["total_time"] = time.time() - start_time
        results["flood_retries"] = self._retry_count

        # Print summary
        print(f"[OK] Digest sent: {results['sent']} messages, {results['failed']} failed")
        print(f"     Total time: {results['total_time']/60:.1f} minutes")
        self.print_flood_stats()

        return results

    async def _send_article(self, article: dict) -> bool:
        """
        Send a single article - with image if available.

        Args:
            article: Article dict with:
                - hero_image: Dict with 'r2_url' (preferred) or 'url' or 'r2_path'
                - headline: Article headline (not 'title')
                - ai_summary: Summary text
                - tag: Single tag
                - link: Article URL

        Returns:
            True if sent successfully
        """
        # Get image URL - prefer r2_url (already constructed full URL)
        image_url = None
        hero_image = article.get("hero_image") or article.get("image")

        if hero_image:
            # Priority 1: r2_url (full URL already constructed)
            if hero_image.get("r2_url"):
                image_url = hero_image["r2_url"]
            # Priority 2: original URL
            elif hero_image.get("url"):
                image_url = hero_image["url"]
            # Priority 3: r2_path (shouldn't happen if prepare_articles_for_telegram ran)
            elif hero_image.get("r2_path"):
                print(f"   [WARN] Using r2_path without base URL - this may fail")
                image_url = hero_image["r2_path"]

        # Format caption
        caption = self._format_article(article)

        # Send with image if available
        if image_url:
            print(f"   [DEBUG] Sending image: {image_url[:80]}...")
            return await self.send_photo(image_url, caption)
        else:
            # Send as text message if no image
            msg_id = await self.send_message(caption)
            return msg_id is not None

    async def send_status(self, status: str) -> bool:
        """
        Send a status message to the channel.

        Args:
            status: Status message text

        Returns:
            True if sent successfully
        """
        # No escaping needed for HTML
        msg_id = await self.send_message(status, disable_preview=True)
        return msg_id is not None

    # =========================================================================
    # Message Formatting
    # =========================================================================

    def _format_header(self, article_count: int, edition_type: str = "daily") -> str:
        """
        Format digest header message in HTML.

        Format:
            <b>04 February 2026</b>
            Our editorial selection for today -- Weekly Edition

            Telegram | adu.media

        Args:
            article_count: Number of articles in this edition
            edition_type: Edition type string ("daily", "weekend", "weekly")

        Returns:
            Formatted HTML header string
        """
        today = datetime.now().strftime("%d %B %Y")
        edition_label = EDITION_LABELS.get(edition_type, "Daily Edition")

        return (
            f"<b>{today}</b>\n"
            f"Our editorial selection for today -- {edition_label}\n"
            f"\n"
            f'<a href="https://t.me/a_d_u_media">Telegram</a>'
            f" | "
            f'<a href="https://adu.media/">adu.media</a>'
        )

    def _format_article(self, article: dict) -> str:
        """
        Format single article message in HTML.

        Format:
        <b>PROJECT NAME / ARCHITECT</b>

        Two-sentence summary

        #tag

        Source Name (as link)
        """
        url = article.get("link", "")
        headline = article.get("headline", "")
        summary = article.get("ai_summary", "No summary available.")
        tag = article.get("tag", "")

        source_name = get_source_name(url)

        message_parts = []

        # 1. Bold headline (PROJECT NAME / ARCHITECT)
        if headline:
            message_parts.append(f"<b>{headline}</b>")
            message_parts.append("")  # Empty line after headline

        # 2. Summary (plain text)
        message_parts.append(summary)
        message_parts.append("")  # Empty line after summary

        # 3. Tag (exactly one)
        if tag:
            # Clean tag: lowercase, replace spaces with underscores, remove special chars
            clean_tag = tag.strip().lower().replace(" ", "_")
            clean_tag = re.sub(r'[^a-z0-9_]', '', clean_tag)
            if clean_tag:
                message_parts.append(f"#{clean_tag}")
                message_parts.append("")  # Empty line after tag

        # 4. Source link
        if url:
            # HTML link format: <a href="URL">Source Name</a>
            message_parts.append(f'<a href="{url}">{source_name}</a>')

        # Join with single newlines (empty strings create the blank lines)
        return "\n".join(message_parts)

    # =========================================================================
    # Connection Testing
    # =========================================================================

    async def test_connection(self) -> bool:
        """
        Test bot connection and permissions.

        Returns:
            True if bot can send to channel
        """
        try:
            bot_info = await self.bot.get_me()
            print(f"[OK] Bot connected: @{bot_info.username}")

            # Try to get chat info
            chat = await self.bot.get_chat(self.channel_id)
            print(f"[OK] Channel accessible: {chat.title}")

            return True
        except TelegramError as e:
            print(f"[ERROR] Connection test failed: {e}")
            return False

# =============================================================================
# Convenience Functions
# =============================================================================

async def send_to_telegram(
    articles: list[dict],
    include_header: bool = True,
    edition_type: str = "daily"
) -> dict:
    """
    Quick function to send articles to Telegram.

    Args:
        articles: List of article dicts
        include_header: Whether to include daily header
        edition_type: Edition type string ("daily", "weekend", "weekly")

    Returns:
        Dict with sent/failed counts
    """
    bot = TelegramBot()
    return await bot.send_digest(articles, include_header, edition_type)