# database/models.py
"""
Pydantic Models for ADUmedia Database v2

Data models matching the new Supabase schema with project-based deduplication.
"""

from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class ArticleStatus(str, Enum):
    """Status values for article lifecycle."""
    FETCHED = "fetched"                    # Just fetched, not yet processed
    FILTERED_OUT = "filtered_out"          # AI filtered (not architecture-relevant)
    DUPLICATE_PROJECT = "duplicate_project" # Same project published recently
    CANDIDATE = "candidate"                # Available for selection
    SELECTED_DAILY = "selected_daily"      # Selected for daily edition
    SELECTED_WEEKEND = "selected_weekend"  # Selected for weekend edition
    SELECTED_WEEKLY = "selected_weekly"    # Selected for weekly edition
    PUBLISHED = "published"                # Sent to Telegram
    ARCHIVED = "archived"                  # Moved to archive


class ProjectType(str, Enum):
    """Types of architecture projects."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    CULTURAL = "cultural"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    HOSPITALITY = "hospitality"
    MIXED_USE = "mixed_use"
    INFRASTRUCTURE = "infrastructure"
    LANDSCAPE = "landscape"
    INTERIOR = "interior"
    MASTERPLAN = "masterplan"
    INSTALLATION = "installation"
    OTHER = "other"


class ProjectStatus(str, Enum):
    """Status of architecture projects."""
    COMPLETED = "completed"
    UNDER_CONSTRUCTION = "under_construction"
    ANNOUNCED = "announced"
    COMPETITION = "competition"
    RENOVATION = "renovation"
    DEMOLISHED = "demolished"


class EditionType(str, Enum):
    """Types of editorial editions."""
    DAILY = "daily"
    WEEKEND = "weekend"
    WEEKLY = "weekly"


# =============================================================================
# Project Models
# =============================================================================

class ProjectLocation(BaseModel):
    """Location information for a project."""
    city: Optional[str] = None
    country: Optional[str] = None


class ProjectExtraction(BaseModel):
    """Result of AI project extraction from article summary."""
    is_project: bool = Field(description="Whether this article is about a specific project")
    project_name: Optional[str] = Field(None, description="Name of the building/project")
    architect: Optional[str] = Field(None, description="Architect firm name")
    location: ProjectLocation = Field(default_factory=ProjectLocation)
    project_type: Optional[ProjectType] = None
    project_status: Optional[ProjectStatus] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Project(BaseModel):
    """
    Model for projects table.
    
    Tracks unique architecture projects for deduplication.
    """
    id: Optional[UUID] = None
    
    # Project identification
    project_name: str
    project_name_normalized: Optional[str] = None
    architect: Optional[str] = None
    architect_normalized: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    
    # Matching hash
    name_hash: Optional[str] = None
    
    # Tracking
    first_seen_date: date
    first_article_id: Optional[UUID] = None
    last_published_date: Optional[date] = None
    times_published: int = 0
    
    # Metadata
    project_type: Optional[ProjectType] = None
    project_status: Optional[ProjectStatus] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# =============================================================================
# Article Models
# =============================================================================

class AllArticle(BaseModel):
    """
    Model for all_articles table.
    
    Tracks every article from fetch through publication.
    """
    id: Optional[UUID] = None
    
    # Article identification
    article_url: str
    article_url_hash: Optional[str] = None
    
    # Source info
    source_id: str
    source_name: Optional[str] = None
    
    # Original metadata
    original_title: str
    original_publish_date: Optional[date] = None
    
    # Project extraction
    project_id: Optional[UUID] = None
    extracted_project_name: Optional[str] = None
    extracted_architect: Optional[str] = None
    extracted_location: Optional[str] = None
    
    # AI content
    ai_summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # R2 storage
    r2_path: Optional[str] = None
    r2_image_path: Optional[str] = None
    fetch_date: date
    
    # Status tracking
    status: ArticleStatus = ArticleStatus.FETCHED
    
    # Selection metadata
    selected_for_editions: List[str] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    selection_category: Optional[str] = None
    weekly_candidate: bool = False
    
    # Publication tracking
    first_published_at: Optional[datetime] = None
    edition_dates: List[date] = Field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# =============================================================================
# Edition Models
# =============================================================================

class Edition(BaseModel):
    """
    Model for editions table.
    
    Tracks each digest sent to Telegram.
    """
    id: Optional[UUID] = None
    
    edition_type: EditionType
    edition_date: date
    published_at: datetime
    
    # Statistics
    total_candidates: int = 0
    articles_selected: int = 0
    articles_new: int = 0
    articles_repeated: int = 0
    
    # References
    article_ids: List[UUID] = Field(default_factory=list)
    
    # Telegram
    header_message_id: Optional[int] = None
    telegram_status: str = "pending"
    
    # Summary
    edition_summary: Optional[str] = None
    
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class WeeklyCandidate(BaseModel):
    """
    Model for weekly_candidates table.
    
    Tracks articles flagged for potential weekly edition inclusion.
    """
    id: Optional[UUID] = None
    
    article_id: UUID
    week_start_date: date
    
    relevance_score: Optional[float] = None
    engagement_potential: Optional[str] = None
    category: Optional[str] = None
    
    is_selected: bool = False
    selected_at: Optional[datetime] = None
    
    editor_notes: Optional[str] = None
    
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# =============================================================================
# Helper Models
# =============================================================================

class ArticleWithProject(BaseModel):
    """Article joined with its project data."""
    article: AllArticle
    project: Optional[Project] = None


class CandidateForSelection(BaseModel):
    """Simplified article data for AI selection."""
    id: str
    title: str
    source_id: str
    source_name: str
    link: str
    published: Optional[str] = None
    ai_summary: str
    tags: List[str] = Field(default_factory=list)
    has_image: bool = False
    
    # Project info for dedup context
    project_name: Optional[str] = None
    architect: Optional[str] = None
    location: Optional[str] = None
    previously_published: bool = False
    days_since_published: Optional[int] = None
