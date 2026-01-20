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
    ) -> bool:
        """
        Send a message with automatic retry on rate limit errors.

        Args:
            send_func: Async function to call (e.g., self.bot.send_message)
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for send_func

        Returns:
            True if message sent successfully, False otherwise
        """
        # Wait for rate limit before attempting
        await self._wait_for_rate_limit()

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                await send_func(**kwargs)
                self._message_count += 1
                return True

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
                    return False

            except NetworkError as e:
                # Network error - retry with backoff
                if attempt < max_retries:
                    delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"   [NETWORK] Network error. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    print(f"   [ERROR] Network error after {max_retries + 1} attempts: {e}")
                    return False

            except TelegramError as e:
                # Other Telegram errors - don't retry
                print(f"   [ERROR] Telegram error: {e}")
                return False

            except Exception as e:
                # Unexpected error
                print(f"   [ERROR] Unexpected error: {e}")
                return False

        # All retries exhausted
        print(f"   [ERROR] Failed after {max_retries + 1} attempts. Last error: {last_error}")
        return False

    def get_flood_stats(self) -> dict:
        """Get flood control statistics."""
        return {
            "messages_sent": self._message_count,
            "retries": self._retry_count,
            "total_flood_wait": self._flood_wait_total,
        }

    def print_flood_stats(self) -> None:
        """Print flood control statistics."""
        stats = self.get_flood_stats()
        if stats["retries"] > 0:
            print(f"   [FLOOD STATS] Messages: {stats['messages_sent']}, "
                  f"Retries: {stats['retries']}, "
                  f"Total wait: {stats['total_flood_wait']:.1f}s")

    # =========================================================================
    # Message Escaping
    # =========================================================================

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """
        Escape special Markdown characters for Telegram.

        Telegram's legacy Markdown mode uses these special characters:
        - _ for italic
        - * for bold
        - ` for code
        - [ for links

        Args:
            text: Raw text to escape

        Returns:
            Text with escaped Markdown characters
        """
        if not text:
            return ""

        # Characters that need escaping in Telegram Markdown
        escape_chars = ['_', '*', '[', ']', '`']

        result = text
        for char in escape_chars:
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
        parse_mode: str = ParseMode.MARKDOWN,
        disable_preview: bool = False
    ) -> bool:
        """
        Send a single message to the channel with flood control.

        Args:
            text: Message text
            parse_mode: Telegram parse mode (Markdown/HTML)
            disable_preview: Disable link preview

        Returns:
            True if sent successfully
        """
        # Try with formatting first
        success = await self._send_with_retry(
            self.bot.send_message,
            chat_id=self.channel_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_preview
        )

        if success:
            return True

        # Fallback: try without formatting
        print("   [INFO] Retrying without Markdown formatting...")
        return await self._send_with_retry(
            self.bot.send_message,
            chat_id=self.channel_id,
            text=text,
            parse_mode=None,
            disable_web_page_preview=disable_preview
        )

    async def send_photo(
        self,
        photo_url: str,
        caption: str | None = None,
        parse_mode: str = ParseMode.MARKDOWN
    ) -> bool:
        """
        Send a photo with optional caption to the channel with flood control.

        Args:
            photo_url: URL of the image to send
            caption: Optional caption text
            parse_mode: Telegram parse mode

        Returns:
            True if sent successfully
        """
        # Try with formatting first
        success = await self._send_with_retry(
            self.bot.send_photo,
            chat_id=self.channel_id,
            photo=photo_url,
            caption=caption,
            parse_mode=parse_mode
        )

        if success:
            return True

        # Fallback: try without formatting
        print("   [INFO] Retrying photo without Markdown formatting...")
        return await self._send_with_retry(
            self.bot.send_photo,
            chat_id=self.channel_id,
            photo=photo_url,
            caption=caption,
            parse_mode=None
        )

    # =========================================================================
    # Digest Sending
    # =========================================================================

    async def send_digest(
        self, 
        articles: list[dict],
        include_header: bool = True
    ) -> dict:
        """
        Send a news digest to the channel with flood control.

        Respects Telegram's 20 messages/minute limit for channels.
        Estimated time: ~3.5 seconds per article.

        Args:
            articles: List of article dicts with keys:
                - link: Article URL
                - ai_summary: AI-generated summary
                - tags: List of tags (optional)
                - hero_image: Dict with 'url' or 'r2_url' (optional)
            include_header: Whether to send daily header message

        Returns:
            Dict with sent/failed counts and timing info
        """
        results: dict = {
            "sent": 0, 
            "failed": 0,
            "total_time": 0.0,
            "flood_retries": 0,
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
            header = self._format_header(len(articles))
            if await self.send_message(header, disable_preview=True):
                results["sent"] += 1
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
            article: Article dict

        Returns:
            True if sent successfully
        """
        # Get hero image URL (prefer R2, fallback to original)
        hero_image = article.get("hero_image")
        image_url: str | None = None

        if hero_image:
            image_url = hero_image.get("r2_url") or hero_image.get("url")

        # Format the caption/message
        caption = self._format_article(article)

        # Send with image if available
        if image_url:
            success = await self.send_photo(image_url, caption)
            if success:
                return True
            # Fallback to text-only if image fails
            print("   [WARN] Image failed, sending text only")

        # Send as text message
        return await self.send_message(caption, disable_preview=False)

    async def send_single_article(self, article: dict) -> bool:
        """
        Send a single article notification.

        Args:
            article: Article dict with link, ai_summary, tags

        Returns:
            True if sent successfully
        """
        return await self._send_article(article)

    async def send_error_notification(self, error_message: str) -> bool:
        """
        Send an error notification to the channel (for monitoring).

        Args:
            error_message: Error description

        Returns:
            True if sent successfully
        """
        # Escape the error message to prevent Markdown issues
        escaped_message = self._escape_markdown(error_message)
        text = f"*System Alert*\n\n{escaped_message}"
        return await self.send_message(text, disable_preview=True)

    async def send_status_update(self, status: str) -> bool:
        """
        Send a status update (e.g., "Monitoring started").

        Args:
            status: Status message

        Returns:
            True if sent successfully
        """
        # Escape status message
        escaped_status = self._escape_markdown(status)
        return await self.send_message(escaped_status, disable_preview=True)

    # =========================================================================
    # Message Formatting
    # =========================================================================

    def _format_header(self, article_count: int) -> str:
        """Format digest header message."""
        today = datetime.now().strftime("%d %B %Y")
        return (
            f"{today}\n"
            f"Our editorial selection for today."
        )

    def _format_article(self, article: dict) -> str:
        """
        Format single article message with proper Markdown escaping.

        Format:
            Summary text here.

            #tag1 #tag2

            SourceName (linked)
        """
        url = article.get("link", "")
        summary = article.get("ai_summary", "No summary available.")
        tags = article.get("tags", [])

        # Get source display name
        source_name = get_source_name(url)

        # Escape special Markdown characters in summary
        escaped_summary = self._escape_markdown(summary)

        # Build message - start with escaped summary
        message = escaped_summary

        # Add tags if present
        if tags:
            if isinstance(tags, list):
                # Clean tags: lowercase, replace spaces with underscores
                cleaned_tags = []
                for tag in tags:
                    if tag:
                        # Remove any special characters from tags
                        clean_tag = tag.strip().lower()
                        # Replace spaces with underscores
                        clean_tag = clean_tag.replace(" ", "_")
                        # Remove any remaining Markdown special chars from tag
                        clean_tag = re.sub(r'[_*\[\]`]', '', clean_tag)
                        if clean_tag:
                            cleaned_tags.append(f"#{clean_tag}")
                tags_str = " ".join(cleaned_tags)
            else:
                tags_str = str(tags)

            if tags_str:
                message += f"\n\n{tags_str}"

        # Add source link with escaped source name and URL
        if url:
            escaped_source = self._escape_markdown(source_name)
            escaped_url = self._escape_url(url)
            message += f"\n\n[{escaped_source}]({escaped_url})"

        return message

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

async def send_to_telegram(articles: list[dict], include_header: bool = True) -> dict:
    """
    Quick function to send articles to Telegram.

    Args:
        articles: List of article dicts
        include_header: Whether to include daily header

    Returns:
        Dict with sent/failed counts
    """
    bot = TelegramBot()
    return await bot.send_digest(articles, include_header)