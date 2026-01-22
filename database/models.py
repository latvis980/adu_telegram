# database/models.py
"""
Pydantic Models for ADUmedia Database

Data models matching the Supabase schema for type safety and validation.
"""

from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class PublishedArticle(BaseModel):
    """
    Model for published_articles table.
    
    Tracks every article that has been published in any edition.
    """
    id: Optional[UUID] = None
    
    # Article identification
    article_url: str = Field(..., description="Normalized article URL (unique)")
    article_url_hash: Optional[str] = Field(None, description="MD5 hash of URL (auto-generated)")
    normalized_title: Optional[str] = Field(None, description="Lowercase, cleaned title")
    
    # Article metadata
    original_title: str = Field(..., description="Original article title")
    source_id: str = Field(..., description="Source identifier (e.g., 'archdaily')")
    source_name: Optional[str] = Field(None, description="Human-readable source name")
    original_publish_date: Optional[date] = Field(None, description="When source published it")
    
    # R2 reference
    r2_path: Optional[str] = Field(None, description="Path to archived JSON in R2")
    
    # Publication tracking
    first_published_at: datetime = Field(..., description="When we first published it")
    published_in_editions: List[str] = Field(default_factory=list, description="Edition types")
    edition_dates: List[date] = Field(default_factory=list, description="Publication dates")
    
    # Content fingerprint
    content_hash: Optional[str] = Field(None, description="MD5 hash of summary")
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class Edition(BaseModel):
    """
    Model for editions table.
    
    Tracks each digest sent to Telegram.
    """
    id: Optional[UUID] = None
    
    # Edition identification
    edition_type: str = Field(..., description="daily, weekend, or weekly")
    edition_date: date = Field(..., description="Date this edition covers")
    published_at: datetime = Field(..., description="When we sent to Telegram")
    
    # Statistics
    total_candidates: int = Field(0, description="How many were available")
    articles_selected: int = Field(0, description="How many were chosen")
    articles_new: int = Field(0, description="First-time publications")
    articles_repeated: int = Field(0, description="Repeats from daily to weekly")
    
    # Article references
    article_ids: List[UUID] = Field(default_factory=list, description="References to published_articles")
    
    # Telegram metadata
    header_message_id: Optional[int] = Field(None, description="Telegram message ID of header")
    telegram_status: str = Field("pending", description="pending, sent, or failed")
    
    # Timestamps
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class WeeklyCandidate(BaseModel):
    """
    Model for weekly_candidates table.
    
    Tracks articles flagged for potential weekly edition inclusion.
    """
    id: Optional[UUID] = None
    
    # References
    article_id: UUID = Field(..., description="Reference to published_articles")
    week_start_date: date = Field(..., description="Monday of the target week")
    
    # Scoring
    relevance_score: Optional[float] = Field(None, description="AI-assigned importance 0-10")
    engagement_potential: Optional[str] = Field(None, description="high, medium, low")
    category: Optional[str] = Field(None, description="project, news, etc.")
    
    # Selection status
    is_selected: bool = Field(False, description="Whether selected for weekly")
    selected_at: Optional[datetime] = None
    
    # Notes
    editor_notes: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# =============================================================================
# Helper Models for API Responses
# =============================================================================

class ArticleSummary(BaseModel):
    """Lightweight article summary for lists."""
    id: UUID
    title: str
    source_name: str
    url: str
    published_at: datetime


class EditionSummary(BaseModel):
    """Lightweight edition summary."""
    id: UUID
    edition_type: str
    edition_date: date
    articles_count: int
    telegram_status: str


class PublicationStats(BaseModel):
    """Publication statistics."""
    period_start: date
    total_articles: int
    source_distribution: dict
    editions: dict
    total_new_publications: int
    total_repeated_publications: int
    repeat_ratio: float
