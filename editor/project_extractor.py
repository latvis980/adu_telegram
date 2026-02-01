# editor/project_extractor.py
"""
Topic/Project Information Extractor for ADUmedia

Uses GPT-4o-mini to extract topic details from article summaries.
This information is used for project-level deduplication.

IMPORTANT: ALL articles get tracked as "topics" for deduplication.
Topics can be: building projects, awards, studio profiles, events, etc.

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key
    LANGCHAIN_TRACING_V2    - Set to "true" to enable LangSmith
    LANGCHAIN_API_KEY       - LangSmith API key
    LANGCHAIN_PROJECT       - LangSmith project name
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import date
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import rate limiter
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.rate_limiter import get_rate_limiter, retry_on_rate_limit


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class ProjectLocation(BaseModel):
    """Location information."""
    city: Optional[str] = None
    country: Optional[str] = None


class ExtractedProject(BaseModel):
    """Result of topic/project extraction."""
    # New field for topic categorization
    topic_type: str = Field(
        default="other",
        description="Type of topic: building, award, studio, exhibition, competition, masterplan, other"
    )

    # Keep is_project for backward compatibility (True if topic_type == "building")
    is_project: bool = Field(
        default=True,  # Now defaults to True - we always track
        description="Whether this is a trackable topic (always True for dedup)"
    )

    project_name: Optional[str] = Field(None, description="Name of the topic for deduplication")
    architect: Optional[str] = Field(None, description="Architect/firm/person name")
    location: ProjectLocation = Field(default_factory=ProjectLocation)
    project_type: Optional[str] = Field(None, description="Type classification")
    project_status: Optional[str] = Field(None, description="Status of topic")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# =============================================================================
# Project Extractor Class
# =============================================================================

class ProjectExtractor:
    """
    Extracts topic information from article summaries using AI.

    Used for:
    1. Linking articles to unique topics (buildings, awards, events, etc.)
    2. Detecting when same topic is covered by multiple sources
    3. Detecting topic updates (same topic, 3+ months later)

    ALL articles get tracked as topics for deduplication.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        langsmith_project: Optional[str] = None
    ):
        """
        Initialize the project extractor.

        Args:
            model: OpenAI model to use (gpt-4o-mini is cost-effective)
            temperature: Low temperature for consistent extraction
            langsmith_project: LangSmith project name for tracing
        """
        # Configure LangSmith tracing
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            project = langsmith_project or os.getenv("LANGCHAIN_PROJECT", "adumedia-extractor")
            os.environ["LANGCHAIN_PROJECT"] = project
            print(f"[EXTRACTOR] LangSmith tracing: {project}")

        # Initialize LLM with JSON mode
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # Get global rate limiter
        self.rate_limiter = get_rate_limiter()

        # Load prompt template
        self.prompt_template = self._load_prompt()

        print(f"[EXTRACTOR] Initialized with {model} and rate limiting")

    def _load_prompt(self) -> str:
        """Load the extraction prompt template."""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "prompts" / "extract_project_info.txt",
            Path("prompts/extract_project_info.txt"),
            Path("/home/claude/prompts/extract_project_info.txt"),
        ]

        for path in possible_paths:
            if path.exists():
                return path.read_text(encoding="utf-8")

        # Fallback: inline prompt
        return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Return default prompt if file not found."""
        return """You are an architecture news analyst. Extract the KEY TOPIC from this article for deduplication.

RULES:
- Do NOT use any emoji
- Every article has SOME trackable topic - find it
- Topics can be: buildings, awards, studios, exhibitions, competitions, masterplans

ARTICLE SUMMARY:
{summary}

METADATA:
Title: {title}
Source: {source_name}

TOPIC CATEGORIES:
1. building - Specific architectural project/building
2. award - Award/honor given to person or firm
3. studio - Feature about an architecture firm  
4. exhibition - Architecture exhibition, biennale, event
5. competition - Architecture competition
6. masterplan - Urban plan/large-scale development
7. other - Industry news, policy, general topics

Return ONLY valid JSON:
{{
    "topic_type": "building|award|studio|exhibition|competition|masterplan|other",
    "project_name": "Descriptive name for deduplication",
    "architect": "Firm or person name (null if not applicable)",
    "location": {{"city": "City or null", "country": "Country or null"}},
    "project_type": "residential|commercial|cultural|educational|healthcare|hospitality|mixed_use|infrastructure|landscape|interior|masterplan|installation|award|exhibition|competition|other|null",
    "project_status": "completed|under_construction|announced|competition|renovation|awarded|ongoing|null",
    "confidence": 0.0-1.0
}}

EXAMPLES:
Award: {{"topic_type": "award", "project_name": "Niall McLaughlin - Royal Gold Medal 2026", "architect": "Niall McLaughlin", ...}}
Studio: {{"topic_type": "studio", "project_name": "NAAW - Next Practices Award 2025", "architect": "NAAW", ...}}
Building: {{"topic_type": "building", "project_name": "The Spiral", "architect": "BIG", ...}}"""

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Normalize text for matching."""
        if not text:
            return ""

        import re
        # Lowercase, remove special chars, collapse whitespace
        normalized = text.lower().strip()
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    @staticmethod
    def generate_project_hash(name: Optional[str], architect: Optional[str]) -> str:
        """Generate hash for topic matching."""
        normalized_name = ProjectExtractor.normalize_text(name)
        normalized_architect = ProjectExtractor.normalize_text(architect)
        combined = f"{normalized_name}|{normalized_architect}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def extract(
        self,
        summary: str,
        title: str,
        source_name: str,
        url: str = "",
        run_name: Optional[str] = None
    ) -> ExtractedProject:
        """
        Extract topic information from an article.

        Args:
            summary: AI-generated summary of the article
            title: Original article title
            source_name: Source publication name
            url: Article URL (for context)
            run_name: Optional name for LangSmith trace

        Returns:
            ExtractedProject with extracted information
        """
        # Build prompt
        prompt = self.prompt_template.format(
            summary=summary,
            title=title,
            source_name=source_name,
            url=url
        )

        # Estimate tokens (rough: prompt + response)
        estimated_tokens = len(prompt) // 4 + 500  # ~500 tokens for response

        # Configure run metadata
        config = RunnableConfig(
            run_name=run_name or f"extract-{source_name[:10]}",
            tags=["extraction", "topic-info", source_name],
            metadata={
                "source": source_name,
                "title_preview": title[:50] if title else "",
            }
        )

        try:
            # Use rate limiter
            async with self.rate_limiter.acquire(estimated_tokens):
                # Call LLM with retry on rate limit
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
                    # Fallback to estimate
                    self.rate_limiter.record_usage(estimated_tokens)

            # Parse response
            result = json.loads(response.content)

            # Ensure we always have a trackable topic
            extracted = ExtractedProject(**result)

            # Set is_project based on topic_type for backward compatibility
            extracted.is_project = True  # Always trackable now

            # If no project_name was extracted, create one from title
            if not extracted.project_name and title:
                # Use the title as a fallback topic name
                extracted.project_name = self._create_fallback_topic_name(title, source_name)
                extracted.confidence = max(0.5, extracted.confidence)  # Lower confidence for fallback

            return extracted

        except Exception as e:
            print(f"[EXTRACTOR] Error extracting topic info: {e}")
            # Return fallback extraction - still trackable
            return ExtractedProject(
                topic_type="other",
                is_project=True,  # Still trackable for dedup
                project_name=self._create_fallback_topic_name(title, source_name) if title else None,
                confidence=0.3
            )

    def _create_fallback_topic_name(self, title: str, source_name: str) -> str:
        """Create a fallback topic name from the article title."""
        # Truncate long titles
        if len(title) > 80:
            title = title[:77] + "..."
        return f"{title} ({source_name})"

    async def extract_batch(
        self,
        articles: List[Dict[str, Any]],
        batch_size: int = 5  # Reduced from 10 to 5 for better rate limit handling
    ) -> List[ExtractedProject]:
        """
        Extract topic info from multiple articles.

        Args:
            articles: List of article dicts with summary, title, source_name
            batch_size: Number of concurrent extractions (reduced to 5 for rate limits)

        Returns:
            List of ExtractedProject results (same order as input)
        """
        results = []
        total_batches = (len(articles) + batch_size - 1) // batch_size

        print(f"[EXTRACTOR] Processing {len(articles)} articles in {total_batches} batches of {batch_size}")

        for batch_num, i in enumerate(range(0, len(articles), batch_size), 1):
            batch = articles[i:i + batch_size]

            print(f"   📦 Batch {batch_num}/{total_batches} ({len(batch)} articles)...")

            # Create tasks for batch
            tasks = [
                self.extract(
                    summary=article.get("ai_summary", "")[:300],  # Truncate summaries to save tokens
                    title=article.get("title", ""),
                    source_name=article.get("source_name", "Unknown"),
                    url=article.get("link", "")
                )
                for article in batch
            ]

            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"      ❌ Batch item {i+j} failed: {result}")
                    # Still create a trackable fallback
                    article = batch[j]
                    results.append(ExtractedProject(
                        topic_type="other",
                        is_project=True,
                        project_name=self._create_fallback_topic_name(
                            article.get("title", "Unknown"),
                            article.get("source_name", "Unknown")
                        ),
                        confidence=0.3
                    ))
                else:
                    results.append(result)

            print(f"      ✅ Processed {min(i + batch_size, len(articles))}/{len(articles)} articles")

            # Add a small delay between batches to prevent rate limit spikes
            if batch_num < total_batches:
                await asyncio.sleep(0.5)

        # Print rate limiter stats
        print(f"\n[EXTRACTOR] Completed extraction:")
        self.rate_limiter.print_stats()

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_project_info(
    summary: str,
    title: str,
    source_name: str
) -> ExtractedProject:
    """
    Quick function to extract topic info from a single article.

    Args:
        summary: Article summary
        title: Article title
        source_name: Source name

    Returns:
        ExtractedProject
    """
    extractor = ProjectExtractor()
    return await extractor.extract(summary, title, source_name)