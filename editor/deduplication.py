# editor/deduplication.py
"""
Deduplication Checker for ADUmedia

Checks articles against Supabase database to prevent duplicate publications
and tracks publication history across editions.

Uses URL-based primary deduplication with optional title similarity checking.
"""

import os
import hashlib
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Set, Tuple
from difflib import SequenceMatcher

from supabase import create_client, Client


# =============================================================================
# Configuration
# =============================================================================

# Similarity threshold for near-duplicate detection (0.0 to 1.0)
TITLE_SIMILARITY_THRESHOLD = 0.85


# =============================================================================
# Deduplication Checker
# =============================================================================

class DeduplicationChecker:
    """
    Handles deduplication logic using Supabase database.
    
    Provides methods to:
    - Check if URL has been published
    - Find near-duplicates by title similarity
    - Get articles published in specific editions
    - Track new publications
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        """
        Initialize the deduplication checker.
        
        Args:
            supabase_url: Supabase project URL (or SUPABASE_URL env var)
            supabase_key: Supabase API key (or SUPABASE_KEY env var)
        """
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(url, key)
        print("[DEDUP] Connected to Supabase")
    
    # =========================================================================
    # URL-based Deduplication
    # =========================================================================
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL for consistent matching.
        
        Removes trailing slashes, lowercases, removes common tracking params.
        """
        if not url:
            return ""
        
        normalized = url.lower().strip()
        
        # Remove trailing slash
        normalized = normalized.rstrip("/")
        
        # Remove common tracking parameters
        for param in ["?utm_", "&utm_", "?ref=", "&ref=", "?source=", "&source="]:
            if param in normalized:
                normalized = normalized.split(param)[0]
        
        return normalized
    
    @staticmethod
    def hash_url(url: str) -> str:
        """Create MD5 hash of normalized URL."""
        normalized = DeduplicationChecker.normalize_url(url)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_url_published(self, url: str) -> bool:
        """
        Check if a URL has already been published.
        
        Args:
            url: Article URL to check
            
        Returns:
            True if URL exists in published_articles table
        """
        normalized = self.normalize_url(url)
        
        result = self.client.table("published_articles")\
            .select("id")\
            .eq("article_url", normalized)\
            .limit(1)\
            .execute()
        
        return len(result.data) > 0
    
    def get_published_urls(
        self,
        since_date: Optional[date] = None,
        edition_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of all published URLs.
        
        Args:
            since_date: Only get URLs published since this date
            edition_types: Filter by edition types (e.g., ["daily", "weekly"])
            
        Returns:
            List of published article URLs
        """
        query = self.client.table("published_articles").select("article_url")
        
        if since_date:
            query = query.gte("first_published_at", since_date.isoformat())
        
        # Note: edition_types filtering would require array contains query
        # For now, we fetch all and filter
        
        result = query.execute()
        
        urls = [row["article_url"] for row in result.data]
        
        if edition_types:
            # Filter by edition type (requires separate query or post-filter)
            # This is a simplification - full implementation would use array_contains
            pass
        
        return urls
    
    def filter_unpublished(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter candidates to only include unpublished articles.
        
        Args:
            candidates: List of candidate article dicts
            
        Returns:
            Tuple of (unpublished_candidates, already_published)
        """
        # Get all URLs from candidates
        candidate_urls = {
            self.normalize_url(c.get("link", "")): c 
            for c in candidates 
            if c.get("link")
        }
        
        if not candidate_urls:
            return [], []
        
        # Batch check against database
        url_list = list(candidate_urls.keys())
        
        # Supabase doesn't support IN queries well, so we check in batches
        published_urls: Set[str] = set()
        batch_size = 50
        
        for i in range(0, len(url_list), batch_size):
            batch = url_list[i:i + batch_size]
            result = self.client.table("published_articles")\
                .select("article_url")\
                .in_("article_url", batch)\
                .execute()
            
            for row in result.data:
                published_urls.add(row["article_url"])
        
        # Split into published and unpublished
        unpublished = []
        already_published = []
        
        for url, candidate in candidate_urls.items():
            if url in published_urls:
                already_published.append(candidate)
            else:
                unpublished.append(candidate)
        
        print(f"[DEDUP] Filtered: {len(unpublished)} unpublished, {len(already_published)} already published")
        
        return unpublished, already_published
    
    # =========================================================================
    # Title Similarity (Near-Duplicate Detection)
    # =========================================================================
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Lowercase and strip
        normalized = title.lower().strip()
        
        # Remove common suffixes like "| ArchDaily"
        for suffix in [" | archdaily", " | dezeen", " - dezeen", " | designboom"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        return normalized
    
    @staticmethod
    def title_similarity(title1: str, title2: str) -> float:
        """
        Calculate similarity ratio between two titles.
        
        Returns value between 0.0 (completely different) and 1.0 (identical).
        """
        norm1 = DeduplicationChecker.normalize_title(title1)
        norm2 = DeduplicationChecker.normalize_title(title2)
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def find_similar_titles(
        self,
        title: str,
        threshold: float = TITLE_SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Find published articles with similar titles.
        
        Args:
            title: Title to check
            threshold: Similarity threshold (default 0.85)
            
        Returns:
            List of similar articles with similarity scores
        """
        # Get recent articles to compare against
        recent = self.client.table("published_articles")\
            .select("id, article_url, original_title, first_published_at")\
            .order("first_published_at", desc=True)\
            .limit(500)\
            .execute()
        
        similar = []
        
        for row in recent.data:
            similarity = self.title_similarity(title, row.get("original_title", ""))
            
            if similarity >= threshold:
                similar.append({
                    "id": row["id"],
                    "url": row["article_url"],
                    "title": row["original_title"],
                    "similarity": similarity,
                    "published_at": row["first_published_at"],
                })
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar
    
    # =========================================================================
    # Edition-Specific Queries
    # =========================================================================
    
    def get_weekly_edition_urls(
        self,
        edition_date: date
    ) -> List[str]:
        """
        Get URLs from a specific weekly edition.
        
        Args:
            edition_date: Date of the weekly edition (Monday)
            
        Returns:
            List of article URLs from that edition
        """
        result = self.client.table("editions")\
            .select("article_ids")\
            .eq("edition_type", "weekly")\
            .eq("edition_date", edition_date.isoformat())\
            .limit(1)\
            .execute()
        
        if not result.data:
            return []
        
        article_ids = result.data[0].get("article_ids", [])
        
        if not article_ids:
            return []
        
        # Get URLs for these article IDs
        articles = self.client.table("published_articles")\
            .select("article_url")\
            .in_("id", article_ids)\
            .execute()
        
        return [row["article_url"] for row in articles.data]
    
    def get_daily_editions_this_week(
        self,
        week_end_date: date
    ) -> List[Dict[str, str]]:
        """
        Get all articles published in daily editions for the current week.
        
        Args:
            week_end_date: End of week (Sunday)
            
        Returns:
            List of dicts with url, title, date for each article
        """
        week_start = week_end_date - timedelta(days=6)
        
        # Get daily editions for this week
        editions = self.client.table("editions")\
            .select("article_ids, edition_date")\
            .eq("edition_type", "daily")\
            .gte("edition_date", week_start.isoformat())\
            .lte("edition_date", week_end_date.isoformat())\
            .execute()
        
        if not editions.data:
            return []
        
        # Collect all article IDs with their dates
        article_date_map = {}
        for edition in editions.data:
            edition_date = edition["edition_date"]
            for article_id in edition.get("article_ids", []):
                article_date_map[article_id] = edition_date
        
        if not article_date_map:
            return []
        
        # Get article details
        articles = self.client.table("published_articles")\
            .select("id, article_url, original_title")\
            .in_("id", list(article_date_map.keys()))\
            .execute()
        
        result = []
        for article in articles.data:
            result.append({
                "url": article["article_url"],
                "title": article["original_title"],
                "date": article_date_map.get(article["id"], ""),
            })
        
        return result
    
    def get_recent_weekly_urls(
        self,
        days: int = 30
    ) -> List[str]:
        """
        Get URLs from weekly editions in the last N days.
        
        Args:
            days: Number of days to look back (default 30)
            
        Returns:
            List of article URLs from recent weekly editions
        """
        since_date = date.today() - timedelta(days=days)
        
        editions = self.client.table("editions")\
            .select("article_ids")\
            .eq("edition_type", "weekly")\
            .gte("edition_date", since_date.isoformat())\
            .execute()
        
        all_article_ids = []
        for edition in editions.data:
            all_article_ids.extend(edition.get("article_ids", []))
        
        if not all_article_ids:
            return []
        
        # Get URLs
        articles = self.client.table("published_articles")\
            .select("article_url")\
            .in_("id", all_article_ids)\
            .execute()
        
        return [row["article_url"] for row in articles.data]
    
    # =========================================================================
    # Publication Recording
    # =========================================================================
    
    def record_publication(
        self,
        article: Dict[str, Any],
        edition_type: str,
        edition_date: date,
        r2_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Record a newly published article.
        
        Args:
            article: Article dict with url, title, source_id, etc.
            edition_type: Type of edition ("daily", "weekend", "weekly")
            edition_date: Date of the edition
            r2_path: Optional R2 path to archived JSON
            
        Returns:
            UUID of the created record, or None if failed
        """
        url = self.normalize_url(article.get("link", ""))
        
        if not url:
            print("[DEDUP] Cannot record article without URL")
            return None
        
        # Check if already exists
        existing = self.client.table("published_articles")\
            .select("id, published_in_editions, edition_dates")\
            .eq("article_url", url)\
            .limit(1)\
            .execute()
        
        if existing.data:
            # Update existing record with new edition
            record = existing.data[0]
            editions = record.get("published_in_editions", [])
            dates = record.get("edition_dates", [])
            
            if edition_type not in editions:
                editions.append(edition_type)
            dates.append(edition_date.isoformat())
            
            self.client.table("published_articles")\
                .update({
                    "published_in_editions": editions,
                    "edition_dates": dates,
                    "updated_at": datetime.utcnow().isoformat(),
                })\
                .eq("id", record["id"])\
                .execute()
            
            print(f"[DEDUP] Updated existing record: {record['id']}")
            return record["id"]
        
        # Create new record
        data = {
            "article_url": url,
            "normalized_title": self.normalize_title(article.get("title", "")),
            "original_title": article.get("title", ""),
            "source_id": article.get("source_id", ""),
            "source_name": article.get("source_name", ""),
            "original_publish_date": article.get("published"),
            "r2_path": r2_path,
            "first_published_at": datetime.utcnow().isoformat(),
            "published_in_editions": [edition_type],
            "edition_dates": [edition_date.isoformat()],
        }
        
        # Generate content hash from summary
        if article.get("ai_summary"):
            data["content_hash"] = hashlib.md5(
                article["ai_summary"].encode()
            ).hexdigest()
        
        result = self.client.table("published_articles")\
            .insert(data)\
            .execute()
        
        if result.data:
            record_id = result.data[0]["id"]
            print(f"[DEDUP] Created new record: {record_id}")
            return record_id
        
        return None
    
    def record_edition(
        self,
        edition_type: str,
        edition_date: date,
        article_ids: List[str],
        total_candidates: int,
        articles_new: int,
        articles_repeated: int = 0,
        header_message_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Record an edition publication.
        
        Args:
            edition_type: Type of edition
            edition_date: Date of edition
            article_ids: UUIDs of published_articles records
            total_candidates: How many candidates were available
            articles_new: How many were first-time publications
            articles_repeated: How many were repeats (weekly only)
            header_message_id: Telegram message ID of header
            
        Returns:
            UUID of the edition record
        """
        data = {
            "edition_type": edition_type,
            "edition_date": edition_date.isoformat(),
            "published_at": datetime.utcnow().isoformat(),
            "total_candidates": total_candidates,
            "articles_selected": len(article_ids),
            "articles_new": articles_new,
            "articles_repeated": articles_repeated,
            "article_ids": article_ids,
            "header_message_id": header_message_id,
            "telegram_status": "sent",
        }
        
        result = self.client.table("editions")\
            .insert(data)\
            .execute()
        
        if result.data:
            edition_id = result.data[0]["id"]
            print(f"[DEDUP] Recorded edition: {edition_type} - {edition_id}")
            return edition_id
        
        return None
    
    # =========================================================================
    # Weekly Candidate Tracking
    # =========================================================================
    
    def flag_weekly_candidate(
        self,
        article_id: str,
        week_start: date,
        relevance_score: Optional[float] = None,
        category: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Flag an article as a candidate for weekly edition.
        
        Args:
            article_id: UUID of the published_articles record
            week_start: Monday of the target week
            relevance_score: AI-assigned importance score
            category: Article category
            notes: Optional editor notes
            
        Returns:
            True if flagged successfully
        """
        data = {
            "article_id": article_id,
            "week_start_date": week_start.isoformat(),
            "relevance_score": relevance_score,
            "category": category,
            "editor_notes": notes,
        }
        
        try:
            self.client.table("weekly_candidates")\
                .upsert(data, on_conflict="article_id,week_start_date")\
                .execute()
            return True
        except Exception as e:
            print(f"[DEDUP] Failed to flag weekly candidate: {e}")
            return False
    
    def get_weekly_candidates(
        self,
        week_start: date
    ) -> List[Dict[str, Any]]:
        """
        Get all flagged candidates for a week.
        
        Args:
            week_start: Monday of the week
            
        Returns:
            List of candidate records with article details
        """
        result = self.client.table("weekly_candidates")\
            .select("*, published_articles(*)")\
            .eq("week_start_date", week_start.isoformat())\
            .eq("is_selected", False)\
            .order("relevance_score", desc=True)\
            .execute()
        
        return result.data
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_publication_stats(
        self,
        since_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get publication statistics.
        
        Args:
            since_date: Start date for stats (default: last 30 days)
            
        Returns:
            Dict with various statistics
        """
        if since_date is None:
            since_date = date.today() - timedelta(days=30)
        
        # Count articles
        articles = self.client.table("published_articles")\
            .select("id, source_id, published_in_editions")\
            .gte("first_published_at", since_date.isoformat())\
            .execute()
        
        # Count editions
        editions = self.client.table("editions")\
            .select("edition_type, articles_selected, articles_new, articles_repeated")\
            .gte("edition_date", since_date.isoformat())\
            .execute()
        
        # Calculate stats
        total_articles = len(articles.data)
        
        # Source distribution
        source_counts: Dict[str, int] = {}
        for article in articles.data:
            source = article.get("source_id", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Edition counts
        edition_counts = {"daily": 0, "weekend": 0, "weekly": 0}
        total_new = 0
        total_repeated = 0
        
        for edition in editions.data:
            edition_type = edition.get("edition_type", "")
            if edition_type in edition_counts:
                edition_counts[edition_type] += 1
            total_new += edition.get("articles_new", 0)
            total_repeated += edition.get("articles_repeated", 0)
        
        return {
            "period_start": since_date.isoformat(),
            "total_articles": total_articles,
            "source_distribution": source_counts,
            "editions": edition_counts,
            "total_new_publications": total_new,
            "total_repeated_publications": total_repeated,
            "repeat_ratio": total_repeated / max(total_new + total_repeated, 1),
        }
