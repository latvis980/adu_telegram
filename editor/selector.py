# editor/selector.py
"""
AI Article Selector for ADUmedia

Uses LangChain with GPT-4o to select articles for different edition types.
Includes LangSmith tracing for monitoring and debugging.

Schedule:
    Monday    - Weekend Catch-Up Edition (covers Fri, Sat, Sun, Mon - 4 days)
    Tuesday   - Daily Edition (covers 2 days)
    Wednesday - Daily Edition (covers 2 days)
    Thursday  - Daily Edition (covers 2 days)
    Friday    - Daily Edition (covers 2 days)
    Saturday  - No publication
    Sunday    - No publication

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
    DAILY = "daily"           # Tuesday, Wednesday, Thursday, Friday
    WEEKEND = "weekend"       # Monday (covers Fri, Sat, Sun, Mon)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class SelectedArticle(BaseModel):
    """A single selected article."""
    id: str = Field(description="Article ID (e.g., 'archdaily_001')")
    title: str = Field(description="Article title for reference")
    reason: str = Field(description="Brief 10-15 word selection rationale")
    category: str = Field(description="Article category")
    publication_day: Optional[str] = Field(default=None, description="Day of publication (weekend)")


class SelectionStats(BaseModel):
    """Statistics about the selection process."""
    total_candidates: int
    excluded_duplicates: Optional[int] = 0
    geographic_spread: List[str]
    source_spread: List[str]


class EditorSelection(BaseModel):
    """Complete editor selection output."""
    selected: List[SelectedArticle] = Field(description="Exactly 7 selected articles")
    edition_summary: str = Field(description="One sentence edition summary")
    selection_stats: SelectionStats
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
        max_chars: int = 40000
    ) -> str:
        """
        Format candidate articles for inclusion in prompt.

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
                "title": article.get("title", "")[:120],
                "source_id": article.get("source_id", ""),
                "source_name": article.get("source_name", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "ai_summary": article.get("ai_summary", "")[:200],
                "tags": article.get("tags", [])[:3],
                "has_image": article.get("image", {}).get("has_image", False),
            })

        # Use compact JSON (no indentation)
        candidates_json = json.dumps(simplified, indent=None, ensure_ascii=False)

        # Truncate if too long
        if len(candidates_json) > max_chars:
            print(f"   [WARN] Truncating candidates from {len(candidates_json)} to {max_chars} chars")
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

        # Calculate coverage dates (today + yesterday)
        yesterday = target_date - timedelta(days=1)
        coverage = f"{yesterday.strftime('%B %d')} - {target_date.strftime('%B %d, %Y')}"

        return template.format(
            current_date=target_date.strftime("%A, %B %d, %Y"),
            day_name=target_date.strftime("%A"),
            coverage_dates=coverage,
            candidates_json=self._format_candidates_for_prompt(candidates),
            published_urls=self._format_urls_list(published_urls),
        )

    def _build_weekend_prompt(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: date
    ) -> str:
        """Build the complete weekend catch-up edition prompt."""
        template = load_prompt(EditionType.WEEKEND)

        # Calculate coverage dates (Fri, Sat, Sun, Mon = today)
        friday = target_date - timedelta(days=3)
        coverage = f"{friday.strftime('%B %d')} - {target_date.strftime('%B %d, %Y')}"

        return template.format(
            current_date=target_date.strftime("%A, %B %d, %Y"),
            coverage_dates=coverage,
            candidates_json=self._format_candidates_for_prompt(candidates),
            published_urls=self._format_urls_list(published_urls),
        )

    async def select_daily(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: Optional[date] = None,
        run_name: Optional[str] = None
    ) -> EditorSelection:
        """
        Select articles for daily edition (Tue/Wed/Thu/Fri).

        Args:
            candidates: Available candidate articles (2 days)
            published_urls: URLs of previously published articles
            target_date: Target date (defaults to today)
            run_name: Optional name for LangSmith trace

        Returns:
            EditorSelection with 7 selected articles
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n[EDITOR] Selecting DAILY edition for {target_date}")
        print(f"   Candidates (2 days): {len(candidates)}")
        print(f"   Previously published: {len(published_urls)}")

        prompt = self._build_daily_prompt(candidates, published_urls, target_date)

        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 2000

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

            print(f"   [OK] Selected {len(selection.selected)} articles")
            print(f"   Summary: {selection.edition_summary}")

            return selection

        except Exception as e:
            print(f"   [ERROR] Failed to parse selection: {e}")
            print(f"   Raw response: {response.content[:500] if 'response' in locals() else 'N/A'}...")
            raise

    async def select_weekend(
        self,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: Optional[date] = None,
        run_name: Optional[str] = None
    ) -> EditorSelection:
        """
        Select articles for weekend catch-up edition (Monday).

        Args:
            candidates: Candidates from Friday, Saturday, Sunday, Monday (4 days)
            published_urls: All previously published URLs
            target_date: Target date (defaults to today, should be Monday)
            run_name: Optional name for LangSmith trace

        Returns:
            EditorSelection with 7 selected articles
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n[EDITOR] Selecting WEEKEND edition for {target_date}")
        print(f"   Candidates (4 days): {len(candidates)}")
        print(f"   Previously published: {len(published_urls)}")

        prompt = self._build_weekend_prompt(
            candidates, published_urls, target_date
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

            print(f"   [OK] Selected {len(selection.selected)} articles")
            print(f"   Summary: {selection.edition_summary}")

            return selection

        except Exception as e:
            print(f"   [ERROR] Failed to parse selection: {e}")
            raise

    async def select(
        self,
        edition_type: EditionType,
        candidates: List[Dict[str, Any]],
        published_urls: List[str],
        target_date: Optional[date] = None
    ) -> EditorSelection:
        """
        Unified selection method that routes to appropriate edition selector.

        Args:
            edition_type: Type of edition to create
            candidates: Available candidate articles
            published_urls: Previously published URLs (for deduplication)
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

    Schedule:
        Monday (0)    - Weekend Catch-Up Edition (covers Fri, Sat, Sun, Mon)
        Tuesday (1)   - Daily Edition (covers 2 days)
        Wednesday (2) - Daily Edition (covers 2 days)
        Thursday (3)  - Daily Edition (covers 2 days)
        Friday (4)    - Daily Edition (covers 2 days)
        Saturday (5)  - No publication
        Sunday (6)    - No publication

    Args:
        target_date: Date to check (defaults to today)

    Returns:
        EditionType or None if no publication today
    """
    if target_date is None:
        target_date = date.today()

    weekday = target_date.weekday()  # Monday = 0, Sunday = 6

    if weekday == 0:                  # Monday
        return EditionType.WEEKEND
    elif weekday in [1, 2, 3, 4]:     # Tue, Wed, Thu, Fri
        return EditionType.DAILY
    else:                              # Saturday, Sunday
        return None


def get_edition_display_name(edition_type: EditionType) -> str:
    """Get human-readable name for edition type."""
    names = {
        EditionType.DAILY: "Daily Edition",
        EditionType.WEEKEND: "Weekend Catch-Up Edition",
    }
    return names.get(edition_type, "Edition")
