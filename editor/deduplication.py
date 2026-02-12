# editor/deduplication.py
"""
Topic-Based Deduplication Checker for ADUmedia

Uses AI for fuzzy topic matching - handles variations in naming,
capitalization, and description differences.

KEY CHANGE: ALL articles are tracked as "topics" for deduplication,
not just building projects. Topics include: buildings, awards, studios,
exhibitions, competitions, masterplans, etc.

Key Logic:
- Same topic published < 3 months ago = DUPLICATE (skip)
- Same topic published > 3 months ago = UPDATE (allow with flag)
- AI determines if two articles are about the same topic

IMPORTANT: Projects/topics are only created in the database when articles 
are PUBLISHED, not during the extraction/filtering phase.
"""

import os
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import rate limiter
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.rate_limiter import get_rate_limiter, retry_on_rate_limit


# =============================================================================
# Configuration
# =============================================================================

# How many months before a topic can be published again
PROJECT_COOLDOWN_MONTHS = 3

# Minimum confidence for AI match
MATCH_CONFIDENCE_THRESHOLD = 0.75

# How many recent topics to check against (increased for monthly news cycle)
MAX_PROJECTS_TO_CHECK = 1000  # Increased from 100 to 1000 for monthly coverage


# =============================================================================
# Pydantic Models
# =============================================================================

class ProjectMatch(BaseModel):
    """Result of AI topic matching."""
    match_found: bool
    matched_project_id: Optional[str] = None
    confidence: float = 0.0
    match_reason: str = ""


# =============================================================================
# Deduplication Checker
# =============================================================================

class DeduplicationChecker:
    """
    Handles topic-based deduplication using AI for fuzzy matching.

    Key methods:
    - find_matching_project_ai(): AI-based fuzzy topic matching against PUBLISHED topics
    - check_project_duplicate(): Is this topic published recently?
    - record_article(): Track article in all_articles table
    - create_project_on_publish(): Create topic record ONLY when publishing

    IMPORTANT: Topics are only added to the database when articles are published,
    not during candidate processing.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the deduplication checker.

        Args:
            supabase_url: Supabase project URL (or SUPABASE_URL env var)
            supabase_key: Supabase API key (or SUPABASE_KEY env var)
            model: OpenAI model for matching (gpt-4o-mini is cost-effective)
        """
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

        self.client: Client = create_client(url, key)

        # Initialize LLM for matching
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Low temperature for consistent matching
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # Get global rate limiter
        self.rate_limiter = get_rate_limiter()

        # Load prompt template
        self.match_prompt = self._load_match_prompt()

        print("[DEDUP] Connected to Supabase with AI matching and rate limiting")
        print(f"[DEDUP] Checking against {MAX_PROJECTS_TO_CHECK} recent projects")

    def _load_match_prompt(self) -> str:
        """Load the topic matching prompt template."""
        possible_paths = [
            Path(__file__).parent.parent / "prompts" / "match_project.txt",
            Path("prompts/match_project.txt"),
        ]

        for path in possible_paths:
            if path.exists():
                return path.read_text(encoding="utf-8")

        # Fallback inline prompt
        return self._get_default_match_prompt()

    def _get_default_match_prompt(self) -> str:
        """Return default matching prompt if file not found."""
        return """Determine if the NEW article is about the SAME topic as any EXISTING topic.

Do NOT use any emoji in your response.

NEW ARTICLE:
Topic Type: {new_topic_type}
Topic Name: {new_project_name}
Architect/Person: {new_architect}
Location: {new_location}
Summary: {new_summary_excerpt}

EXISTING TOPICS (these have been published before):
{existing_projects_json}

MATCHING RULES:
- Building: Same building + same architect + same/similar location
- Award: Same award + same year + same recipient
- Studio: Same studio + same context/award
- Exhibition: Same event + same year
- Competition: Same competition + same year

Return JSON:
{{"match_found": true/false, "matched_project_id": "uuid or null", "confidence": 0.0-1.0, "match_reason": "explanation"}}

Match if: confidence >= 0.75"""

    # =========================================================================
    # AI-Based Topic Matching (against PUBLISHED topics only)
    # =========================================================================

    async def find_matching_project_ai(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location: Optional[str] = None,
        summary_excerpt: Optional[str] = None,
        topic_type: str = "building",
        run_name: Optional[str] = None
    ) -> Tuple[Optional[str], float, str]:
        """
        Use AI to find matching topic in database.

        IMPORTANT: Only matches against PUBLISHED topics (those with last_published_date set).

        This handles fuzzy matching for:
        - Name variations ("The Spiral" vs "Spiral Tower")
        - Capitalization differences
        - Architect name variations ("BIG" vs "Bjarke Ingels Group")
        - Location variations ("NYC" vs "New York")
        - Award variations ("Royal Gold Medal" vs "RIBA Gold Medal")

        Args:
            project_name: Name of the topic from new article
            architect: Architect/person name (if extracted)
            location: Location (city, country)
            summary_excerpt: First 200 chars of summary for context
            topic_type: Type of topic (building, award, studio, etc.)
            run_name: Optional name for LangSmith trace

        Returns:
            Tuple of (project_id or None, confidence, reason)
        """
        # Get only PUBLISHED topics from database
        existing_projects = self._get_published_projects(limit=MAX_PROJECTS_TO_CHECK)

        if not existing_projects:
            return None, 0.0, "No published topics in database"

        # Format existing topics for prompt - MINIMIZE TOKEN USAGE
        projects_for_prompt = []
        for p in existing_projects:
            # Only include essential fields, truncate long values
            projects_for_prompt.append({
                "id": p["id"],
                "project_name": (p.get("project_name", "") or "")[:100],  # Truncate long names
                "architect": (p.get("architect", "") or "")[:80],  # Truncate
                "location": f"{p.get('location_city', '') or ''}, {p.get('location_country', '') or ''}".strip(", ")[:80],
                "last_pub": (p.get("last_published_date", "") or "")[:10],  # Just date, no time
                "type": (p.get("topic_type", "") or "building")[:20],
            })

        # Truncate summary_excerpt to save tokens
        summary_short = (summary_excerpt or "")[:150]

        # Build prompt
        prompt = self.match_prompt.format(
            new_topic_type=topic_type,
            new_project_name=project_name or "Unknown",
            new_architect=architect or "Unknown",
            new_location=location or "Unknown",
            new_summary_excerpt=summary_short,
            existing_projects_json=json.dumps(projects_for_prompt, indent=None)  # No indentation to save tokens
        )

        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 300  # ~300 tokens for response

        # Configure run metadata
        config = RunnableConfig(
            run_name=run_name or f"match-{(project_name or 'unknown')[:20]}",
            tags=["dedup", "topic-match", topic_type],
            metadata={
                "project_name": project_name,
                "architect": architect,
                "topic_type": topic_type,
                "candidates_count": len(existing_projects),
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
                    self.rate_limiter.record_usage(estimated_tokens)

            # Parse response
            result = json.loads(response.content)
            match = ProjectMatch(**result)

            if match.match_found and match.confidence >= MATCH_CONFIDENCE_THRESHOLD:
                print(f"   ✅ [DEDUP] AI Match: {project_name} -> {(match.matched_project_id or '')[:8]}... "
                      f"(conf: {match.confidence:.2f})")
                return match.matched_project_id, match.confidence, match.match_reason
            else:
                return None, match.confidence, match.match_reason

        except Exception as e:
            print(f"   ❌ [DEDUP] AI matching error: {e}")
            return None, 0.0, f"Error: {e}"

    def _get_published_projects(self, limit: int = MAX_PROJECTS_TO_CHECK) -> List[Dict[str, Any]]:
        """
        Get topics from database for matching.

        Returns ALL topics - the existence of a topic in the database
        means it was published (or at least processed). We use first_seen_date
        as fallback for legacy data without last_published_date.
        """
        try:
            result = self.client.table("projects")\
                .select("id, project_name, architect, location_city, location_country, last_published_date, first_seen_date, topic_type")\
                .order("first_seen_date", desc=True)\
                .limit(limit)\
                .execute()

            return result.data
        except Exception as e:
            print(f"[DEDUP] Failed to get topics: {e}")
            return []

    def _get_recent_projects(self, limit: int = MAX_PROJECTS_TO_CHECK) -> List[Dict[str, Any]]:
        """
        Get recent projects from database for matching.

        DEPRECATED: Use _get_published_projects instead.
        Keeping for backward compatibility.
        """
        return self._get_published_projects(limit)

    # =========================================================================
    # Topic/Project Operations
    # =========================================================================

    def create_project(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
        project_type: Optional[str] = None,
        project_status: Optional[str] = None,
        topic_type: str = "building",
        first_article_id: Optional[str] = None,
        publish_date: Optional[date] = None
    ) -> Optional[str]:
        """
        Create a new topic entry.

        NOTE: This should ONLY be called when actually publishing an article.

        Args:
            project_name: Name of the topic
            architect: Architect/person name
            location_city: City
            location_country: Country
            project_type: Type classification
            project_status: Status (completed, announced, awarded, etc.)
            topic_type: Category (building, award, studio, exhibition, etc.)
            first_article_id: UUID of the first article about this topic
            publish_date: Date of publication (sets last_published_date)

        Returns:
            UUID of created topic, or None if failed
        """
        if publish_date is None:
            publish_date = date.today()

        data = {
            "project_name": project_name,
            "architect": architect,
            "location_city": location_city,
            "location_country": location_country,
            "first_seen_date": publish_date.isoformat(),
            "first_article_id": first_article_id,
            "project_type": project_type,
            "project_status": project_status,
            "topic_type": topic_type,
            "times_published": 1,
            "last_published_date": publish_date.isoformat(),
        }

        try:
            result = self.client.table("projects")\
                .insert(data)\
                .execute()

            if result.data:
                project_id = result.data[0]["id"]
                print(f"[DEDUP] Created topic ({topic_type}): {project_name} ({project_id[:8]}...)")
                return project_id
        except Exception as e:
            print(f"[DEDUP] Failed to create topic: {e}")

        return None

    async def find_existing_project(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
        summary_excerpt: Optional[str] = None,
        topic_type: str = "building",
    ) -> Tuple[Optional[str], bool, Optional[date]]:
        """
        Check if a matching topic already exists (was published before).

        This is used during filtering to detect duplicates WITHOUT creating new topics.

        Args:
            project_name: Name of the topic
            architect: Architect/person name
            location_city: City
            location_country: Country
            summary_excerpt: Summary for matching context
            topic_type: Type of topic for matching context

        Returns:
            Tuple of (project_id or None, is_duplicate, last_published_date)
            - project_id: UUID if match found, None otherwise
            - is_duplicate: True if published within cooldown period
            - last_published_date: When it was last published
        """
        # Format location for matching
        location = None
        if location_city or location_country:
            parts = [p for p in [location_city, location_country] if p]
            location = ", ".join(parts)

        # Try AI matching against published topics
        matched_id, confidence, reason = await self.find_matching_project_ai(
            project_name=project_name,
            architect=architect,
            location=location,
            summary_excerpt=summary_excerpt,
            topic_type=topic_type
        )

        if not matched_id:
            # No match - this is a new topic (not a duplicate)
            return None, False, None

        # Check when this topic was last published
        is_duplicate, last_published = self.check_project_duplicate(matched_id)

        return matched_id, is_duplicate, last_published

    async def find_or_create_project_on_publish(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
        project_type: Optional[str] = None,
        project_status: Optional[str] = None,
        topic_type: str = "building",
        summary_excerpt: Optional[str] = None,
        article_id: Optional[str] = None,
        publish_date: Optional[date] = None
    ) -> Tuple[Optional[str], bool]:
        """
        Find existing topic or create new one - ONLY call when publishing.

        This is the method to use when recording a published article.
        It will either:
        - Find existing topic and update its last_published_date
        - Create new topic with last_published_date set

        Args:
            project_name: Name of the topic
            architect: Architect/person name
            location_city: City
            location_country: Country
            project_type: Type classification
            project_status: Status of topic
            topic_type: Category (building, award, studio, etc.)
            summary_excerpt: Summary for matching context
            article_id: UUID of the article (for first_article_id)
            publish_date: Date of publication

        Returns:
            Tuple of (project_id, is_new)
        """
        if publish_date is None:
            publish_date = date.today()

        # Format location for matching
        location = None
        if location_city or location_country:
            parts = [p for p in [location_city, location_country] if p]
            location = ", ".join(parts)

        # Try AI matching first
        matched_id, confidence, reason = await self.find_matching_project_ai(
            project_name=project_name,
            architect=architect,
            location=location,
            summary_excerpt=summary_excerpt,
            topic_type=topic_type
        )

        if matched_id:
            print(f"[DEDUP] Matched to existing topic: {reason}")
            # Update the existing topic's publish date
            self.update_project_published(matched_id, publish_date)
            return matched_id, False

        # No match found - create new topic with publish_date set
        project_id = self.create_project(
            project_name=project_name,
            architect=architect,
            location_city=location_city,
            location_country=location_country,
            project_type=project_type,
            project_status=project_status,
            topic_type=topic_type,
            first_article_id=article_id,
            publish_date=publish_date
        )

        return project_id, True

    # DEPRECATED: Old method that creates projects too early
    async def find_or_create_project(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
        project_type: Optional[str] = None,
        project_status: Optional[str] = None,
        summary_excerpt: Optional[str] = None,
        article_id: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """
        DEPRECATED: Use find_existing_project() for filtering and 
        find_or_create_project_on_publish() when publishing.

        This method is kept for backward compatibility but should not be used
        in new code as it creates projects before they're published.
        """
        print("[DEDUP] WARNING: Using deprecated find_or_create_project()")
        return await self.find_or_create_project_on_publish(
            project_name=project_name,
            architect=architect,
            location_city=location_city,
            location_country=location_country,
            project_type=project_type,
            project_status=project_status,
            summary_excerpt=summary_excerpt,
            article_id=article_id
        )

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    def check_project_duplicate(
        self,
        project_id: str,
        cooldown_months: int = PROJECT_COOLDOWN_MONTHS
    ) -> Tuple[bool, Optional[date]]:
        """
        Check if a topic was published recently.

        Args:
            project_id: UUID of the topic
            cooldown_months: Months before topic can be republished

        Returns:
            Tuple of (is_duplicate, last_published_date)
            - is_duplicate: True if published within cooldown period
            - last_published_date: When it was last published
        """
        try:
            result = self.client.table("projects")\
                .select("last_published_date, times_published, first_seen_date")\
                .eq("id", project_id)\
                .limit(1)\
                .execute()

            if not result.data:
                return False, None

            project = result.data[0]
            last_published = project.get("last_published_date")

            # If last_published_date is NULL but topic exists, use first_seen_date
            if not last_published:
                first_seen = project.get("first_seen_date")
                if first_seen:
                    print(f"[DEDUP] Topic has no last_published_date, using first_seen_date: {first_seen}")
                    last_published = first_seen
                else:
                    print("[DEDUP] Topic has no dates, treating as published today")
                    last_published = date.today().isoformat()

            # Parse date
            if isinstance(last_published, str):
                last_published = date.fromisoformat(last_published)

            # Check cooldown
            cooldown_date = date.today() - timedelta(days=cooldown_months * 30)
            is_duplicate = last_published > cooldown_date

            return is_duplicate, last_published

        except Exception as e:
            print(f"[DEDUP] Error checking duplicate: {e}")
            return False, None

    def update_project_published(
        self,
        project_id: str,
        publish_date: Optional[date] = None
    ) -> bool:
        """Update topic when an article about it is published."""
        if publish_date is None:
            publish_date = date.today()

        try:
            # Get current times_published
            result = self.client.table("projects")\
                .select("times_published")\
                .eq("id", project_id)\
                .limit(1)\
                .execute()

            times_published = 0
            if result.data:
                times_published = result.data[0].get("times_published", 0) or 0

            # Update
            self.client.table("projects")\
                .update({
                    "last_published_date": publish_date.isoformat(),
                    "times_published": times_published + 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })\
                .eq("id", project_id)\
                .execute()

            return True
        except Exception as e:
            print(f"[DEDUP] Failed to update topic: {e}")
            return False

    # =========================================================================
    # Article Operations
    # =========================================================================

    def is_url_recorded(self, url: str) -> bool:
        """Check if URL already exists in all_articles with status=published."""
        normalized = url.lower().strip().rstrip("/")

        try:
            result = self.client.table("all_articles")\
                .select("id, status")\
                .eq("article_url", normalized)\
                .limit(1)\
                .execute()

            if not result.data:
                return False

            # Only consider it "recorded" if it was actually published
            return result.data[0].get("status") == "published"
        except Exception as e:
            print(f"[DEDUP] Error checking URL: {e}")
            return False

    def is_url_published(self, url: str) -> bool:
        """Check if URL was actually published (sent to Telegram)."""
        return self.is_url_recorded(url)

    def _slugify(self, text: str) -> str:
        """
        Convert headline to URL slug matching adu.media format.

        "Nobel Center / David Chipperfield" -> "nobel-center-david-chipperfield"
        """
        if not text:
            return ""
        slug = text.lower()
        # Replace slashes and common separators with spaces
        slug = slug.replace("/", " ").replace("\\", " ")
        # Remove all punctuation except hyphens and spaces
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        # Collapse whitespace and convert to hyphens
        slug = re.sub(r'[\s]+', '-', slug.strip())
        # Remove duplicate hyphens
        slug = re.sub(r'-+', '-', slug)
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        return slug


    def _build_adu_media_url(self, headline: str, fetch_date: str) -> str:
        """
        Build adu.media article URL from headline and date.

        Args:
            headline: Formatted headline ("PROJECT / ARCHITECT")
            fetch_date: ISO date string "YYYY-MM-DD"

        Returns:
            Full URL like https://www.adu.media/article/2026-02-11/nobel-center-david-chipperfield
        """
        slug = self._slugify(headline)
        if not slug:
            return ""
        # Ensure date is just YYYY-MM-DD (no time component)
        date_part = fetch_date[:10]
        return f"https://www.adu.media/article/{date_part}/{slug}"

    def record_article(
        self,
        article: Dict[str, Any],
        project_id: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        status: str = "published"
    ) -> Optional[str]:
        """
        Record an article in all_articles table.

        NOTE: Default status changed to 'published' - only call this when publishing.

        Args:
            article: Article dict from R2
            project_id: UUID of linked topic (if any)
            extracted_info: AI-extracted topic info
            status: Status (default: 'published')

        Returns:
            UUID of created record, or None if failed/duplicate
        """
        url = article.get("link", "").lower().strip().rstrip("/")

        if not url:
            print("[DEDUP] Cannot record article without URL")
            return None

        # Check if already exists
        try:
            existing = self.client.table("all_articles")\
                .select("id")\
                .eq("article_url", url)\
                .limit(1)\
                .execute()

            if existing.data:
                print(f"[DEDUP] Article already recorded: {url[:50]}...")
                return existing.data[0]["id"]
        except Exception:
            pass

        # Build adu.media URL from headline
        headline = article.get("headline", "")
        adu_media_url = None
        if headline:
             adu_media_url = self._build_adu_media_url(
                 headline,
                 article.get("_fetch_date", date.today().isoformat())
             )

        data = {
            "article_url": url,
            "source_id": article.get("source_id", "unknown"),
            "source_name": article.get("source_name", ""),
            "original_title": article.get("title", ""),
            "original_publish_date": article.get("published"),
            "ai_summary": article.get("ai_summary", ""),
            "tags": article.get("tags", []),
            "r2_path": article.get("_r2_path"),
            "r2_image_path": article.get("image", {}).get("r2_path") if article.get("image") else None,
            "fetch_date": article.get("_fetch_date", date.today().isoformat()),
            "status": status,
            "project_id": project_id,
            "headline": headline or None,
            "adu_media_url": adu_media_url,
         }

        # Add extracted info if provided
        if extracted_info:
            data["extracted_project_name"] = extracted_info.get("project_name")
            data["extracted_architect"] = extracted_info.get("architect")
            data["extracted_topic_type"] = extracted_info.get("topic_type", "building")
            location = extracted_info.get("location", {})
            if location and isinstance(location, dict):
                city = location.get("city", "")
                country = location.get("country", "")
                data["extracted_location"] = f"{city}, {country}".strip(", ")

        try:
            result = self.client.table("all_articles")\
                .insert(data)\
                .execute()

            if result.data:
                return result.data[0]["id"]
        except Exception as e:
            print(f"[DEDUP] Failed to record article: {e}")

        return None

    def update_article_status(
        self,
        article_id: str,
        status: str,
        selection_reason: Optional[str] = None,
        selection_category: Optional[str] = None,
        weekly_candidate: Optional[bool] = None
    ) -> bool:
        """Update article status and selection metadata."""
        data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if selection_reason:
            data["selection_reason"] = selection_reason
        if selection_category:
            data["selection_category"] = selection_category
        if weekly_candidate is not None:
            data["weekly_candidate"] = weekly_candidate

        try:
            self.client.table("all_articles")\
                .update(data)\
                .eq("id", article_id)\
                .execute()
            return True
        except Exception as e:
            print(f"[DEDUP] Failed to update article status: {e}")
            return False

    def mark_article_published(
        self,
        article_id: str,
        edition_type: str,
        edition_date: date
    ) -> bool:
        """Mark article as published and update tracking."""
        try:
            # Get current data
            result = self.client.table("all_articles")\
                .select("selected_for_editions, edition_dates, first_published_at")\
                .eq("id", article_id)\
                .limit(1)\
                .execute()

            if not result.data:
                return False

            current = result.data[0]
            editions = current.get("selected_for_editions") or []
            dates = current.get("edition_dates") or []

            if edition_type not in editions:
                editions.append(edition_type)
            dates.append(edition_date.isoformat())

            update_data = {
                "status": "published",
                "selected_for_editions": editions,
                "edition_dates": dates,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Set first_published_at if not set
            if not current.get("first_published_at"):
                update_data["first_published_at"] = datetime.now(timezone.utc).isoformat()

            self.client.table("all_articles")\
                .update(update_data)\
                .eq("id", article_id)\
                .execute()

            return True
        except Exception as e:
            print(f"[DEDUP] Failed to mark article published: {e}")
            return False

    # =========================================================================
    # Candidate Filtering (UPDATED to track ALL articles)
    # =========================================================================

    async def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        cooldown_months: int = PROJECT_COOLDOWN_MONTHS
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter candidates based on topic deduplication.

        UPDATED: Now tracks ALL articles, not just building projects.
        Every article gets checked against existing topics.

        This method:
        1. Checks each candidate against PUBLISHED topics
        2. Marks duplicates (same topic published < 3 months)
        3. Marks updates (same topic published > 3 months)
        4. Does NOT create any new topics (that happens on publish)

        Args:
            candidates: List of candidate article dicts with _extracted_info
            cooldown_months: Months before topic can be republished

        Returns:
            Tuple of (eligible, duplicates, updates)
            - eligible: Articles that can be published
            - duplicates: Articles about recently-published topics
            - updates: Articles about older topics (allow with flag)
        """
        eligible = []
        duplicates = []
        updates = []

        print(f"\n📊 [DEDUP] Filtering {len(candidates)} candidates against published topics...")

        # Process in smaller batches to show progress
        batch_size = 50
        total_batches = (len(candidates) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(candidates), batch_size), 1):
            batch = candidates[i:i + batch_size]
            print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} articles)...")

            for j, candidate in enumerate(batch, 1):
                # First check: URL already published?
                url = candidate.get("link", "")
                if url and self.is_url_published(url):
                    candidate["_duplicate_reason"] = "URL already published"
                    duplicates.append(candidate)
                    continue

                # Get extracted topic info
                extracted = candidate.get("_extracted_info", {})

                # UPDATED: Now we always try to match, not just for is_project=True
                # The extractor now always provides a project_name for tracking
                project_name = extracted.get("project_name")

                if not project_name:
                    # No topic name extracted - still eligible but can't dedupe
                    # This should rarely happen with the updated extractor
                    candidate["_project_id"] = None
                    eligible.append(candidate)
                    continue

                # Get topic details
                architect = extracted.get("architect")
                location = extracted.get("location", {})
                location_city = location.get("city") if isinstance(location, dict) else None
                location_country = location.get("country") if isinstance(location, dict) else None
                topic_type = extracted.get("topic_type", "building")

                # Check if this topic was published before
                project_id, is_duplicate, last_published = await self.find_existing_project(
                    project_name=project_name,
                    architect=architect,
                    location_city=location_city,
                    location_country=location_country,
                    summary_excerpt=candidate.get("ai_summary", "")[:150],  # Truncated
                    topic_type=topic_type
                )

                candidate["_project_id"] = project_id  # Will be None for new topics
                candidate["_topic_type"] = topic_type  # Store for later use

                if is_duplicate:
                    days_ago = (date.today() - last_published).days if last_published else 0
                    candidate["_duplicate_reason"] = f"Topic published {days_ago} days ago on {last_published}"
                    duplicates.append(candidate)
                elif project_id and last_published:
                    # Published before, but outside cooldown = update
                    candidate["_is_update"] = True
                    candidate["_last_published"] = last_published
                    updates.append(candidate)
                    eligible.append(candidate)  # Updates are eligible
                else:
                    # Never published = new topic = eligible
                    eligible.append(candidate)

            print(f"      ✅ Batch {batch_num}/{total_batches} complete")

        print("\n📊 [DEDUP] Filtering complete:")
        print(f"   ✅ Eligible: {len(eligible)}")
        print(f"   ❌ Duplicates: {len(duplicates)}")
        print(f"   🔄 Updates: {len(updates)}")

        # Print rate limiter stats
        self.rate_limiter.print_stats()

        return eligible, duplicates, updates

    # =========================================================================
    # Edition Tracking
    # =========================================================================

    def get_published_project_ids(
        self,
        since_date: Optional[date] = None,
    ) -> set:
        """Get IDs of projects published since a date."""
        if since_date is None:
            since_date = date.today() - timedelta(days=90)

        try:
            result = self.client.table("projects")\
                .select("id")\
                .gte("last_published_date", since_date.isoformat())\
                .execute()

            return {row["id"] for row in result.data}
        except Exception as e:
            print(f"[DEDUP] Error getting published project IDs: {e}")
            return set()

    def get_daily_published_for_weekly(
        self,
        week_start: date,
        week_end: date
    ) -> List[Dict[str, str]]:
        """
        Get articles published in daily editions for the weekly edition.

        This helps the Weekly AI know which articles were already shown
        in Wed/Thu/Fri daily editions, so it can mark them as "repeats"
        vs "new" discoveries.

        Args:
            week_start: Start of the week (Monday)
            week_end: End of the week (Sunday)

        Returns:
            List of dicts with: url, title, date, edition_type
        """
        try:
            # Query all_articles that were published in daily editions during this week
            result = self.client.table("all_articles")\
                .select("article_url, original_title, fetch_date, selected_for_editions")\
                .gte("fetch_date", week_start.isoformat())\
                .lte("fetch_date", week_end.isoformat())\
                .eq("status", "published")\
                .execute()

            daily_articles = []
            for row in result.data:
                editions = row.get("selected_for_editions") or []
                # Only include if it was in a daily edition
                if "daily" in editions:
                    daily_articles.append({
                        "url": row.get("article_url", ""),
                        "title": row.get("original_title", ""),
                        "date": row.get("fetch_date", ""),
                    })

            print(f"[DEDUP] Found {len(daily_articles)} daily-published articles for weekly")
            return daily_articles

        except Exception as e:
            print(f"[DEDUP] Error getting daily published for weekly: {e}")
            return []

    def save_weekly_candidates(
        self,
        article_ids: List[str],
        week_start_date: date,
        categories: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Save articles flagged as weekly candidates.

        Called after daily selection when AI marks articles with weekly_candidate=true.

        Args:
            article_ids: List of article UUIDs to flag
            week_start_date: Monday of the week
            categories: Optional dict mapping article_id to category

        Returns:
            Number of candidates saved
        """
        saved = 0
        categories = categories or {}

        for article_id in article_ids:
            try:
                data = {
                    "article_id": article_id,
                    "week_start_date": week_start_date.isoformat(),
                    "category": categories.get(article_id),
                    "is_selected": False,
                }

                self.client.table("weekly_candidates")\
                    .insert(data)\
                    .execute()

                saved += 1
            except Exception as e:
                # Might be duplicate, ignore
                print(f"[DEDUP] Failed to save weekly candidate {article_id}: {e}")

        print(f"[DEDUP] Saved {saved} weekly candidates for week of {week_start_date}")
        return saved

    def get_weekly_candidates(
        self,
        week_start_date: date
    ) -> List[str]:
        """
        Get article IDs flagged as weekly candidates for a given week.

        Args:
            week_start_date: Monday of the week

        Returns:
            List of article UUIDs
        """
        try:
            result = self.client.table("weekly_candidates")\
                .select("article_id")\
                .eq("week_start_date", week_start_date.isoformat())\
                .eq("is_selected", False)\
                .execute()

            return [row["article_id"] for row in result.data]
        except Exception as e:
            print(f"[DEDUP] Error getting weekly candidates: {e}")
            return []

    def record_edition(
        self,
        edition_type: str,
        edition_date: date,
        article_ids: List[str],
        total_candidates: int,
        articles_new: int,
        articles_repeated: int = 0,
        edition_summary: Optional[str] = None,
        header_message_id: Optional[int] = None
    ) -> Optional[str]:
        """Record an edition in the database."""
        data = {
            "edition_type": edition_type,
            "edition_date": edition_date.isoformat(),
            "published_at": datetime.now(timezone.utc).isoformat(),
            "total_candidates": total_candidates,
            "articles_selected": len(article_ids),
            "articles_new": articles_new,
            "articles_repeated": articles_repeated,
            "article_ids": article_ids,
            "edition_summary": edition_summary,
            "header_message_id": header_message_id,
            "telegram_status": "sent",
        }

        try:
            result = self.client.table("editions")\
                .insert(data)\
                .execute()

            if result.data:
                edition_id = result.data[0]["id"]
                print(f"[DEDUP] Recorded edition: {edition_type} - {edition_id[:8]}...")
                return edition_id
        except Exception as e:
            print(f"[DEDUP] Failed to record edition: {e}")

        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self, since_date: Optional[date] = None) -> Dict[str, Any]:
        """Get publication statistics."""
        if since_date is None:
            since_date = date.today() - timedelta(days=30)

        try:
            # Count articles by status
            articles = self.client.table("all_articles")\
                .select("status, source_id")\
                .gte("fetch_date", since_date.isoformat())\
                .execute()

            status_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}

            for article in articles.data:
                status = article.get("status", "unknown")
                source = article.get("source_id", "unknown")

                status_counts[status] = status_counts.get(status, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1

            # Count editions
            editions = self.client.table("editions")\
                .select("edition_type, articles_new, articles_repeated")\
                .gte("edition_date", since_date.isoformat())\
                .execute()

            edition_counts: Dict[str, int] = {}
            total_new = 0
            total_repeated = 0

            for edition in editions.data:
                etype = edition.get("edition_type", "unknown")
                edition_counts[etype] = edition_counts.get(etype, 0) + 1
                total_new += edition.get("articles_new", 0) or 0
                total_repeated += edition.get("articles_repeated", 0) or 0

            # Count unique PUBLISHED projects
            projects = self.client.table("projects")\
                .select("id")\
                .not_.is_("last_published_date", "null")\
                .gte("first_seen_date", since_date.isoformat())\
                .execute()

            return {
                "period_start": since_date.isoformat(),
                "total_articles": len(articles.data),
                "status_distribution": status_counts,
                "source_distribution": source_counts,
                "editions": edition_counts,
                "unique_projects": len(projects.data),
                "total_new_publications": total_new,
                "total_repeated_publications": total_repeated,
            }
        except Exception as e:
            print(f"[DEDUP] Error getting stats: {e}")
            return {}