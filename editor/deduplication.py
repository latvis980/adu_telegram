# editor/deduplication.py
"""
Project-Based Deduplication Checker for ADUmedia

Uses AI for fuzzy project matching - handles variations in naming,
capitalization, and description differences.

Key Logic:
- Same project published < 3 months ago = DUPLICATE (skip)
- Same project published > 3 months ago = UPDATE (allow with flag)
- AI determines if two articles are about the same building/project
"""

import os
import json
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

# How many months before a project can be published again
PROJECT_COOLDOWN_MONTHS = 3

# Minimum confidence for AI match
MATCH_CONFIDENCE_THRESHOLD = 0.75

# How many recent projects to check against
MAX_PROJECTS_TO_CHECK = 100


# =============================================================================
# Pydantic Models
# =============================================================================

class ProjectMatch(BaseModel):
    """Result of AI project matching."""
    match_found: bool
    matched_project_id: Optional[str] = None
    confidence: float = 0.0
    match_reason: str = ""


# =============================================================================
# Deduplication Checker
# =============================================================================

class DeduplicationChecker:
    """
    Handles project-based deduplication using AI for fuzzy matching.
    
    Key methods:
    - find_matching_project_ai(): AI-based fuzzy project matching
    - check_project_duplicate(): Is this project published recently?
    - record_article(): Track article in all_articles table
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
        
        # Load prompt template
        self.match_prompt = self._load_match_prompt()
        
        print("[DEDUP] Connected to Supabase with AI matching")
    
    def _load_match_prompt(self) -> str:
        """Load the project matching prompt template."""
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
        return """Determine if the NEW article is about the SAME project as any EXISTING project.

NEW ARTICLE:
Project: {new_project_name}
Architect: {new_architect}
Location: {new_location}

EXISTING PROJECTS:
{existing_projects_json}

Return JSON:
{{"match_found": true/false, "matched_project_id": "uuid or null", "confidence": 0.0-1.0, "match_reason": "explanation"}}

Match if: same building (allowing name variations), same architect, same/similar location.
Don't match if: different buildings by same architect, or same name but different architect/location."""
    
    # =========================================================================
    # AI-Based Project Matching
    # =========================================================================
    
    async def find_matching_project_ai(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location: Optional[str] = None,
        summary_excerpt: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Tuple[Optional[str], float, str]:
        """
        Use AI to find matching project in database.
        
        This handles fuzzy matching for:
        - Name variations ("The Spiral" vs "Spiral Tower")
        - Capitalization differences
        - Architect name variations ("BIG" vs "Bjarke Ingels Group")
        - Location variations ("NYC" vs "New York")
        
        Args:
            project_name: Name of the project from new article
            architect: Architect name (if extracted)
            location: Location (city, country)
            summary_excerpt: First 200 chars of summary for context
            run_name: Optional name for LangSmith trace
            
        Returns:
            Tuple of (project_id or None, confidence, reason)
        """
        # Get recent projects from database
        existing_projects = self._get_recent_projects(limit=MAX_PROJECTS_TO_CHECK)
        
        if not existing_projects:
            return None, 0.0, "No existing projects in database"
        
        # Format existing projects for prompt
        projects_for_prompt = []
        for p in existing_projects:
            projects_for_prompt.append({
                "id": p["id"],
                "project_name": p.get("project_name", ""),
                "architect": p.get("architect", ""),
                "location_city": p.get("location_city", ""),
                "location_country": p.get("location_country", ""),
                "last_published_date": p.get("last_published_date", ""),
            })
        
        # Build prompt
        prompt = self.match_prompt.format(
            new_project_name=project_name or "Unknown",
            new_architect=architect or "Unknown",
            new_location=location or "Unknown",
            new_summary_excerpt=(summary_excerpt or "")[:200],
            existing_projects_json=json.dumps(projects_for_prompt, indent=2)
        )
        
        # Configure run metadata
        config = RunnableConfig(
            run_name=run_name or f"match-{(project_name or 'unknown')[:20]}",
            tags=["dedup", "project-match"],
            metadata={
                "project_name": project_name,
                "architect": architect,
                "candidates_count": len(existing_projects),
            }
        )
        
        try:
            # Call LLM
            messages = [("human", prompt)]
            response = await self.llm.ainvoke(messages, config=config)
            
            # Parse response
            result = json.loads(response.content)
            match = ProjectMatch(**result)
            
            if match.match_found and match.confidence >= MATCH_CONFIDENCE_THRESHOLD:
                print(f"[DEDUP] AI Match: {project_name} -> {match.matched_project_id} "
                      f"(confidence: {match.confidence:.2f})")
                return match.matched_project_id, match.confidence, match.match_reason
            else:
                print(f"[DEDUP] No AI match for: {project_name} (confidence: {match.confidence:.2f})")
                return None, match.confidence, match.match_reason
                
        except Exception as e:
            print(f"[DEDUP] AI matching error: {e}")
            return None, 0.0, f"Error: {e}"
    
    def _get_recent_projects(self, limit: int = MAX_PROJECTS_TO_CHECK) -> List[Dict[str, Any]]:
        """Get recent projects from database for matching."""
        try:
            result = self.client.table("projects")\
                .select("id, project_name, architect, location_city, location_country, last_published_date, first_seen_date")\
                .order("first_seen_date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"[DEDUP] Failed to get recent projects: {e}")
            return []
    
    # =========================================================================
    # Project Operations
    # =========================================================================
    
    def create_project(
        self,
        project_name: str,
        architect: Optional[str] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
        project_type: Optional[str] = None,
        project_status: Optional[str] = None,
        first_article_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new project entry.
        
        Returns:
            UUID of created project, or None if failed
        """
        data = {
            "project_name": project_name,
            "architect": architect,
            "location_city": location_city,
            "location_country": location_country,
            "first_seen_date": date.today().isoformat(),
            "first_article_id": first_article_id,
            "project_type": project_type,
            "project_status": project_status,
            "times_published": 0,
        }
        
        try:
            result = self.client.table("projects")\
                .insert(data)\
                .execute()
            
            if result.data:
                project_id = result.data[0]["id"]
                print(f"[DEDUP] Created project: {project_name} ({project_id[:8]}...)")
                return project_id
        except Exception as e:
            print(f"[DEDUP] Failed to create project: {e}")
        
        return None
    
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
        Find existing project using AI matching, or create new one.
        
        Args:
            project_name: Name of the project
            architect: Architect firm name
            location_city: City
            location_country: Country
            project_type: Type of project
            project_status: Status of project
            summary_excerpt: Summary for matching context
            article_id: UUID of the article (for first_article_id)
            
        Returns:
            Tuple of (project_id, is_new)
        """
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
            summary_excerpt=summary_excerpt
        )
        
        if matched_id:
            print(f"[DEDUP] Matched to existing project: {reason}")
            return matched_id, False
        
        # No match found - create new project
        project_id = self.create_project(
            project_name=project_name,
            architect=architect,
            location_city=location_city,
            location_country=location_country,
            project_type=project_type,
            project_status=project_status,
            first_article_id=article_id
        )
        
        return project_id, True
    
    # =========================================================================
    # Duplicate Detection
    # =========================================================================
    
    def check_project_duplicate(
        self,
        project_id: str,
        cooldown_months: int = PROJECT_COOLDOWN_MONTHS
    ) -> Tuple[bool, Optional[date]]:
        """
        Check if a project was published recently.
        
        Args:
            project_id: UUID of the project
            cooldown_months: Months before project can be republished
            
        Returns:
            Tuple of (is_duplicate, last_published_date)
            - is_duplicate: True if published within cooldown period
            - last_published_date: When it was last published (or None)
        """
        try:
            result = self.client.table("projects")\
                .select("last_published_date, times_published")\
                .eq("id", project_id)\
                .limit(1)\
                .execute()
            
            if not result.data:
                return False, None
            
            project = result.data[0]
            last_published = project.get("last_published_date")
            
            if not last_published:
                return False, None
            
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
        """Update project when an article about it is published."""
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
                    "updated_at": datetime.utcnow().isoformat(),
                })\
                .eq("id", project_id)\
                .execute()
            
            return True
        except Exception as e:
            print(f"[DEDUP] Failed to update project: {e}")
            return False
    
    # =========================================================================
    # Article Operations
    # =========================================================================
    
    def is_url_recorded(self, url: str) -> bool:
        """Check if URL already exists in all_articles."""
        normalized = url.lower().strip().rstrip("/")
        
        try:
            result = self.client.table("all_articles")\
                .select("id")\
                .eq("article_url", normalized)\
                .limit(1)\
                .execute()
            
            return len(result.data) > 0
        except Exception as e:
            print(f"[DEDUP] Error checking URL: {e}")
            return False
    
    def record_article(
        self,
        article: Dict[str, Any],
        project_id: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        status: str = "candidate"
    ) -> Optional[str]:
        """
        Record an article in all_articles table.
        
        Args:
            article: Article dict from R2
            project_id: UUID of linked project (if any)
            extracted_info: AI-extracted project info
            status: Initial status (default: 'candidate')
            
        Returns:
            UUID of created record, or None if failed/duplicate
        """
        url = article.get("link", "").lower().strip().rstrip("/")
        
        if not url:
            print("[DEDUP] Cannot record article without URL")
            return None
        
        # Check if already exists
        if self.is_url_recorded(url):
            print(f"[DEDUP] Article already recorded: {url[:50]}...")
            return None
        
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
        }
        
        # Add extracted info if provided
        if extracted_info:
            data["extracted_project_name"] = extracted_info.get("project_name")
            data["extracted_architect"] = extracted_info.get("architect")
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
            "updated_at": datetime.utcnow().isoformat(),
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
                update_data["first_published_at"] = datetime.utcnow().isoformat()
            
            self.client.table("all_articles")\
                .update(update_data)\
                .eq("id", article_id)\
                .execute()
            
            return True
        except Exception as e:
            print(f"[DEDUP] Failed to mark article published: {e}")
            return False
    
    # =========================================================================
    # Candidate Filtering
    # =========================================================================
    
    async def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        cooldown_months: int = PROJECT_COOLDOWN_MONTHS
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter candidates based on project deduplication.
        
        Args:
            candidates: List of candidate article dicts
            cooldown_months: Months before project can be republished
            
        Returns:
            Tuple of (eligible, duplicates, updates)
            - eligible: Articles that can be published
            - duplicates: Articles about recently-published projects
            - updates: Articles about older projects (allow with flag)
        """
        eligible = []
        duplicates = []
        updates = []
        
        for candidate in candidates:
            project_id = candidate.get("_project_id")
            
            if not project_id:
                # No project linked = always eligible (news, interviews, etc.)
                eligible.append(candidate)
                continue
            
            is_duplicate, last_published = self.check_project_duplicate(
                project_id, cooldown_months
            )
            
            if is_duplicate:
                days_ago = (date.today() - last_published).days if last_published else 0
                candidate["_duplicate_reason"] = f"Published {days_ago} days ago on {last_published}"
                duplicates.append(candidate)
            elif last_published:
                # Published before, but outside cooldown = update
                candidate["_is_update"] = True
                candidate["_last_published"] = last_published
                updates.append(candidate)
                eligible.append(candidate)  # Updates are eligible
            else:
                # Never published
                eligible.append(candidate)
        
        print(f"[DEDUP] Filtered: {len(eligible)} eligible, {len(duplicates)} duplicates, {len(updates)} updates")
        
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
            "published_at": datetime.utcnow().isoformat(),
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
            
            # Count unique projects
            projects = self.client.table("projects")\
                .select("id")\
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
