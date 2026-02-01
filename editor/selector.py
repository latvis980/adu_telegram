# editor/selector.py
"""
AI Article Selector for ADUmedia

Uses LangChain with GPT-4o to select articles for different edition types.
Includes LangSmith tracing for monitoring and debugging.

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key
    LANGCHAIN_TRACING_V2    - Set to "true" to enable LangSmith
    LANGCHAIN_API_KEY       - LangSmith API key
    LANGCHAIN_PROJECT       - LangSmith project name (default: "adumedia-editor")
"""

import os
import json
from enum import Enum
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import rate limiter
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.rate_limiter import get_rate_limiter, retry_on_rate_limit


# =============================================================================
# Edition Types
# =============================================================================

class EditionType(Enum):
    """Types of editorial editions."""
    DAILY = "daily"           # Wednesday, Thursday, Friday
    WEEKEND = "weekend"       # Tuesday (covers Sat, Sun, Mon)
    WEEKLY = "weekly"         # Monday (covers full week)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class SelectedArticle(BaseModel):
    """A single selected article."""
    id: str = Field(description="Article ID (e.g., 'archdaily_001')")
    title: str = Field(description="Article title for reference")
    reason: str = Field(description="Brief 10-15 word selection rationale")
    category: str = Field(description="Article category")
    weekly_candidate: Optional[bool] = Field(default=False, description="Flag for weekly consideration")
    weekly_reason: Optional[str] = Field(default=None, description="Why this deserves weekly")
    is_repeat: Optional[bool] = Field(default=False, description="Was in daily edition (weekly only)")
    repeat_from_date: Optional[str] = Field(default=None, description="Date of daily publication")
    significance_score: Optional[int] = Field(default=None, description="1-10 significance rating")
    publication_day: Optional[str] = Field(default=None, description="Day of publication (weekend)")


class SelectionStats(BaseModel):
    """Statistics about the selection process."""
    total_candidates: int
    excluded_duplicates: Optional[int] = 0
    excluded_from_weekly: Optional[int] = None
    geographic_spread: List[str]
    source_spread: List[str]


class WeeklyStats(BaseModel):
    """Weekly edition specific statistics."""
    repeated_count: int
    new_count: int
    repeat_justification: Optional[str] = None


class EditorSelection(BaseModel):
    """Complete editor selection output."""
    selected: List[SelectedArticle] = Field(description="Exactly 7 selected articles")
    edition_summary: str = Field(description="One sentence edition summary")
    selection_stats: SelectionStats
    weekly_stats: Optional[WeeklyStats] = None
    coverage_breakdown: Optional[Dict[str, int]] = None


# =============================================================================
# Prompt Loading
# =============================================================================

def load_prompt(edition_type: EditionType) -> str:
    """
    Load prompt template for the given edition type.

    Args:
        edition_type: Type of edition to load prompt for

    Returns:
        Prompt template string
    """
    # Get the directory where this file is located
    current_dir = Path(__file__).parent.parent
    prompts_dir = current_dir / "prompts"

    prompt_files = {
        EditionType.DAILY: "editor_daily.txt",
        EditionType.WEEKEND: "editor_weekend.txt",
        EditionType.WEEKLY: "editor_weekly.txt",
    }

    prompt_file = prompts_dir / prompt_files[edition_type]

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    return prompt_file.read_text(encoding="utf-8")


# =============================================================================
# Article Selector Class
# =============================================================================

class ArticleSelector:
    """
    AI-powered article selector for ADUmedia editions.

    Uses GPT-4o for intelligent article selection with LangSmith tracing
    for monitoring and debugging.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        langsmith_project: Optional[str] = None
    ):
        """
        Initialize the article selector.

        Args:
            model: OpenAI model to use
            temperature: Sampling temperature (lower = more deterministic)
            langsmith_project: LangSmith project name for tracing
        """
        # Configure LangSmith tracing
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            project = langsmith_project or os.getenv("LANGCHAIN_PROJECT", "adumedia-editor")
            os.environ["LANGCHAIN_PROJECT"] = project
            print(f"[LANGSMITH] Tracing enabled for project: {project}")

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # Get global rate limiter
        self.rate_limiter = get_rate_limiter()

        # JSON output parser
        self.parser = JsonOutputParser(pydantic_object=EditorSelection)

        print(f"[EDITOR] Initialized with model: {model} and rate limiting")

    def _format_candidates_for_prompt(
        self,
        candidates: List[Dict[str, Any]],
        max_chars: int = 40000  # Reduced from 50000 to save tokens
    ) -> str:
        """
        Format candidate articles for inclusion in prompt.

        OPTIMIZED: Aggressively truncates to reduce token usage for weekly editions.

        Args:
            candidates: List of candidate article dicts
            max_chars: Maximum characters for candidates section

        Returns:
            JSON string of candidates
        """
        # Simplify candidate data for prompt (reduce token usage)
        simplified = []
        for article in candidates:
            simplified.append({
                "id": article.get("id", ""),
                "title": article.get("title", "")[:120],  # Reduced from 150
                "source_id": article.get("source_id", ""),
                "source_name": article.get("source_name", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "ai_summary": article.get("ai_summary", "")[:200],  # Reduced from 300
                "tags": article.get("tags", [])[:3],  # Reduced from 5
                "has_image": article.get("image", {}).get("has_image", False),
            })

        # Use compact JSON (no indentation)
        candidates_json = json.dumps(simplified, indent=None, ensure_ascii=False)

        # Truncate if too long
        if len(candidates_json) > max_chars:
            print(f"   ⚠️  [WARN] Truncating candidates from {len(candidates_json)} to {max_chars} chars")
            candidates_json = candidates_json[:max_chars] + "\n... (truncated)"

        return candidates_json

    def _format_urls_list(self, urls: List[str]) -> str:
        """Format a list of URLs for prompt inclusion."""
        if not urls:
            return "(none)"
        return "\n".join(f"- {url}" for url in urls)

    def _build_daily_prompt(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: date
    ) -> str:
        """Build the complete daily edition prompt."""
        template = load_prompt(EditionType.DAILY)

        return template.format(
            current_date=target_date.strftime("%A, %B %d, %Y"),
            day_name=target_date.strftime("%A"),
            candidates_json=self._format_candidates_for_prompt(candidates),
            published_urls=self._format_urls_list(published_urls),
        )

    def _build_weekend_prompt(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        weekly_article_urls: List[str],
        target_date: date
    ) -> str:
        """Build the complete weekend edition prompt."""
        template = load_prompt(EditionType.WEEKEND)

        # Calculate coverage dates (Sat, Sun, Mon before Tuesday)
        monday = target_date - timedelta(days=1)
        sunday = target_date - timedelta(days=2)
        saturday = target_date - timedelta(days=3)
        coverage = f"{saturday.strftime('%B %d')} - {monday.strftime('%B %d, %Y')}"

        return template.format(
            current_date=target_date.strftime("%A, %B %d, %Y"),
            coverage_dates=coverage,
            candidates_json=self._format_candidates_for_prompt(candidates),
            weekly_article_urls=self._format_urls_list(weekly_article_urls),
            published_urls=self._format_urls_list(published_urls),
        )

    def _build_weekly_prompt(
        self,
        candidates: List[Dict[str, Any]],
        daily_published_this_week: List[Dict[str, str]],
        recent_weekly_urls: List[str],
        target_date: date
    ) -> str:
        """Build the complete weekly edition prompt."""
        template = load_prompt(EditionType.WEEKLY)

        # Calculate week range
        week_end = target_date - timedelta(days=1)  # Sunday
        week_start = week_end - timedelta(days=6)   # Previous Monday

        # Format daily published as readable list
        daily_list = []
        for item in daily_published_this_week:
            daily_list.append(f"- [{item.get('date', 'unknown')}] {item.get('title', 'No title')[:80]}")
            daily_list.append(f"  URL: {item.get('url', '')}")
        daily_formatted = "\n".join(daily_list) if daily_list else "(no daily editions this week)"

        return template.format(
            current_date=target_date.strftime("%A, %B %d, %Y"),
            week_start=week_start.strftime("%B %d"),
            week_end=week_end.strftime("%B %d, %Y"),
            candidates_json=self._format_candidates_for_prompt(candidates),
            daily_published_this_week=daily_formatted,
            recent_weekly_urls=self._format_urls_list(recent_weekly_urls),
        )

    async def select_daily(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: Optional[date] = None,
        run_name: Optional[str] = None
    ) -> EditorSelection:
        """
        Select articles for daily edition (Wed/Thu/Fri).

        Args:
            candidates: Available candidate articles for today
            published_urls: URLs of previously published articles
            target_date: Target date (defaults to today)
            run_name: Optional name for LangSmith trace

        Returns:
            EditorSelection with 7 selected articles
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n📰 [EDITOR] Selecting DAILY edition for {target_date}")
        print(f"   Candidates: {len(candidates)}")
        print(f"   Previously published: {len(published_urls)}")

        prompt = self._build_daily_prompt(candidates, published_urls, target_date)

        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 2000  # ~2000 for response

        # Configure run metadata for LangSmith
        config = RunnableConfig(
            run_name=run_name or f"daily-{target_date.isoformat()}",
            tags=["daily", "editor", target_date.strftime("%Y-%m-%d")],
            metadata={
                "edition_type": "daily",
                "target_date": target_date.isoformat(),
                "candidate_count": len(candidates),
            }
        )

        try:
            # Use rate limiter
            async with self.rate_limiter.acquire(estimated_tokens):
                # Run the selection with retry on rate limit
                messages = [("human", prompt)]

                async def _call_llm():
                    return await self.llm.ainvoke(messages, config=config)

                response = await retry_on_rate_limit(_call_llm)

                # Record actual token usage if available
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    total_tokens = usage.get('total_tokens', estimated_tokens)
                    self.rate_limiter.record_usage(total_tokens)
                else:
                    self.rate_limiter.record_usage(estimated_tokens)

            # Parse response
            result = json.loads(response.content)
            selection = EditorSelection(**result)

            print(f"   ✅ [OK] Selected {len(selection.selected)} articles")
            print(f"   Summary: {selection.edition_summary}")

            return selection

        except Exception as e:
            print(f"   ❌ [ERROR] Failed to parse selection: {e}")
            print(f"   Raw response: {response.content[:500] if 'response' in locals() else 'N/A'}...")
            raise

    async def select_weekend(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        weekly_article_urls: List[str],
        target_date: Optional[date] = None,
        run_name: Optional[str] = None
    ) -> EditorSelection:
        """
        Select articles for weekend catch-up edition (Tuesday).

        Args:
            candidates: Candidates from Saturday, Sunday, Monday
            published_urls: All previously published URLs
            weekly_article_urls: URLs from Monday's Weekly Edition (must exclude)
            target_date: Target date (defaults to today, should be Tuesday)
            run_name: Optional name for LangSmith trace

        Returns:
            EditorSelection with 7 selected articles
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n📰 [EDITOR] Selecting WEEKEND edition for {target_date}")
        print(f"   Candidates (3 days): {len(candidates)}")
        print(f"   Weekly to exclude: {len(weekly_article_urls)}")
        print(f"   Previously published: {len(published_urls)}")

        prompt = self._build_weekend_prompt(
            candidates, published_urls, weekly_article_urls, target_date
        )

        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 2000

        config = RunnableConfig(
            run_name=run_name or f"weekend-{target_date.isoformat()}",
            tags=["weekend", "editor", target_date.strftime("%Y-%m-%d")],
            metadata={
                "edition_type": "weekend",
                "target_date": target_date.isoformat(),
                "candidate_count": len(candidates),
                "weekly_exclusions": len(weekly_article_urls),
            }
        )

        try:
            # Use rate limiter
            async with self.rate_limiter.acquire(estimated_tokens):
                messages = [("human", prompt)]

                async def _call_llm():
                    return await self.llm.ainvoke(messages, config=config)

                response = await retry_on_rate_limit(_call_llm)

                # Record token usage
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    total_tokens = usage.get('total_tokens', estimated_tokens)
                    self.rate_limiter.record_usage(total_tokens)
                else:
                    self.rate_limiter.record_usage(estimated_tokens)

            result = json.loads(response.content)
            selection = EditorSelection(**result)

            print(f"   ✅ [OK] Selected {len(selection.selected)} articles")
            print(f"   Summary: {selection.edition_summary}")

            return selection

        except Exception as e:
            print(f"   ❌ [ERROR] Failed to parse selection: {e}")
            raise

    async def select_weekly(
        self,
        candidates: List[Dict[str, Any]],
        daily_published_this_week: List[Dict[str, str]],
        recent_weekly_urls: List[str],
        target_date: Optional[date] = None,
        run_name: Optional[str] = None
    ) -> EditorSelection:
        """
        Select articles for weekly flagship edition (Monday).

        OPTIMIZED: Handles large candidate sets (300+) with aggressive token reduction.

        Args:
            candidates: All candidates from the past 7 days
            daily_published_this_week: Articles published in daily editions
                Format: [{"url": "...", "title": "...", "date": "YYYY-MM-DD"}, ...]
            recent_weekly_urls: URLs from recent weekly editions (last 30 days)
            target_date: Target date (defaults to today, should be Monday)
            run_name: Optional name for LangSmith trace

        Returns:
            EditorSelection with 7 selected articles (mix of repeats and new)
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n🏆 [EDITOR] Selecting WEEKLY edition for {target_date}")
        print(f"   Candidates (7 days): {len(candidates)}")
        print(f"   In daily editions: {len(daily_published_this_week)}")
        print(f"   Recent weekly (exclude): {len(recent_weekly_urls)}")

        prompt = self._build_weekly_prompt(
            candidates, daily_published_this_week, recent_weekly_urls, target_date
        )

        # Estimate tokens - weekly prompts are LARGE
        estimated_tokens = len(prompt) // 4 + 2500  # Larger response for weekly

        print(f"   📊 Estimated tokens: {estimated_tokens:,}")

        config = RunnableConfig(
            run_name=run_name or f"weekly-{target_date.isoformat()}",
            tags=["weekly", "editor", "flagship", target_date.strftime("%Y-%m-%d")],
            metadata={
                "edition_type": "weekly",
                "target_date": target_date.isoformat(),
                "candidate_count": len(candidates),
                "daily_published_count": len(daily_published_this_week),
            }
        )

        try:
            # Use rate limiter with large token estimate
            async with self.rate_limiter.acquire(estimated_tokens):
                messages = [("human", prompt)]

                async def _call_llm():
                    return await self.llm.ainvoke(messages, config=config)

                print(f"   ⏳ Calling AI editor (this may take a moment for {len(candidates)} candidates)...")
                response = await retry_on_rate_limit(_call_llm, max_retries=7)  # More retries for weekly

                # Record actual token usage
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    total_tokens = usage.get('total_tokens', estimated_tokens)
                    self.rate_limiter.record_usage(total_tokens)
                    print(f"   📊 Actual tokens used: {total_tokens:,}")
                else:
                    self.rate_limiter.record_usage(estimated_tokens)

            result = json.loads(response.content)
            selection = EditorSelection(**result)

            print(f"   ✅ [OK] Selected {len(selection.selected)} articles")
            if selection.weekly_stats:
                print(f"   Repeats: {selection.weekly_stats.repeated_count}, New: {selection.weekly_stats.new_count}")
            print(f"   Summary: {selection.edition_summary}")

            # Print rate limiter stats after weekly (important to see)
            self.rate_limiter.print_stats()

            return selection

        except Exception as e:
            print(f"   ❌ [ERROR] Failed to parse selection: {e}")
            raise

    async def select(
        self,
        edition_type: EditionType,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        weekly_article_urls: Optional[List[str]] = None,
        daily_published_this_week: Optional[List[Dict[str, str]]] = None,
        recent_weekly_urls: Optional[List[str]] = None,
        target_date: Optional[date] = None
    ) -> EditorSelection:
        """
        Unified selection method that routes to appropriate edition selector.

        Args:
            edition_type: Type of edition to create
            candidates: Available candidate articles
            published_urls: Previously published URLs (for deduplication)
            weekly_article_urls: URLs from Weekly Edition (for weekend exclusion)
            daily_published_this_week: Daily edition articles (for weekly)
            recent_weekly_urls: Recent weekly URLs (for weekly exclusion)
            target_date: Target publication date

        Returns:
            EditorSelection with 7 selected articles
        """
        if edition_type == EditionType.DAILY:
            return await self.select_daily(
                candidates=candidates,
                published_urls=published_urls,
                target_date=target_date
            )

        elif edition_type == EditionType.WEEKEND:
            return await self.select_weekend(
                candidates=candidates,
                published_urls=published_urls,
                weekly_article_urls=weekly_article_urls or [],
                target_date=target_date
            )

        elif edition_type == EditionType.WEEKLY:
            return await self.select_weekly(
                candidates=candidates,
                daily_published_this_week=daily_published_this_week or [],
                recent_weekly_urls=recent_weekly_urls or [],
                target_date=target_date
            )

        else:
            raise ValueError(f"Unknown edition type: {edition_type}")


# =============================================================================
# Convenience Functions
# =============================================================================

def determine_edition_type(target_date: Optional[date] = None) -> Optional[EditionType]:
    """
    Determine what edition type to publish based on day of week.

    Args:
        target_date: Date to check (defaults to today)

    Returns:
        EditionType or None if no publication today
    """
    if target_date is None:
        target_date = date.today()

    weekday = target_date.weekday()  # Monday = 0, Sunday = 6

    # ==========================================================================
    # TEMPORARY: Testing weekly edition on Sunday instead of Monday
    # TO RESTORE: Comment out the Sunday block, uncomment the Monday block
    # ==========================================================================

    # TEMPORARY - Sunday weekly edition (for testing)
    if weekday == 6:      # Sunday
        return EditionType.WEEKLY

    # NORMAL - Monday weekly edition (restore this when testing is done)
    # if weekday == 0:      # Monday
    #     return EditionType.WEEKLY

    # ==========================================================================

    if weekday == 1:    # Tuesday
        return EditionType.WEEKEND
    elif weekday in [2, 3, 4]:  # Wed, Thu, Fri
        return EditionType.DAILY
    else:                 # Saturday, Sunday, Monday (during Sunday testing)
        return None


def get_edition_display_name(edition_type: EditionType) -> str:
    """Get human-readable name for edition type."""
    names = {
        EditionType.DAILY: "Daily Edition",
        EditionType.WEEKEND: "Weekend Catch-Up Edition",
        EditionType.WEEKLY: "Weekly Edition - The Week in Architecture",
    }
    return names.get(edition_type, "Edition")