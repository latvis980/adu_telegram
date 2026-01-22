# editions/formatters.py
"""
Edition Formatters for ADUmedia

Provides edition-specific headers and formatting for Telegram messages.
No emoji - clean, professional formatting only.
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Optional

from editor.selector import EditionType


# =============================================================================
# Base Formatter
# =============================================================================

class EditionFormatter(ABC):
    """Abstract base class for edition formatters."""
    
    @abstractmethod
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        """
        Format the edition header message.
        
        Args:
            target_date: Publication date
            article_count: Number of articles in this edition
            
        Returns:
            Formatted header string
        """
        pass
    
    @abstractmethod
    def get_edition_name(self) -> str:
        """Get the display name for this edition type."""
        pass


# =============================================================================
# Daily Edition Formatter
# =============================================================================

class DailyFormatter(EditionFormatter):
    """Formatter for daily editions (Wed/Thu/Fri)."""
    
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        date_str = target_date.strftime("%d %B %Y")
        return (
            f"{date_str}\n"
            f"Our editorial selection for today."
        )
    
    def get_edition_name(self) -> str:
        return "Daily Edition"


# =============================================================================
# Weekend Edition Formatter
# =============================================================================

class WeekendFormatter(EditionFormatter):
    """Formatter for weekend catch-up edition (Tuesday)."""
    
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        date_str = target_date.strftime("%d %B %Y")
        
        # Calculate coverage period
        monday = target_date - timedelta(days=1)
        saturday = target_date - timedelta(days=3)
        
        # Format as "17-20 January" or "29 Jan - 1 Feb" if crossing months
        if saturday.month == monday.month:
            coverage = f"{saturday.day}-{monday.day} {monday.strftime('%B')}"
        else:
            coverage = f"{saturday.strftime('%d %b')} - {monday.strftime('%d %b')}"
        
        return (
            f"{date_str}\n"
            f"Weekend Catch-Up: Top stories from {coverage}."
        )
    
    def get_edition_name(self) -> str:
        return "Weekend Catch-Up Edition"


# =============================================================================
# Weekly Edition Formatter
# =============================================================================

class WeeklyFormatter(EditionFormatter):
    """Formatter for weekly flagship edition (Monday)."""
    
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        date_str = target_date.strftime("%d %B %Y")
        
        # Calculate week range
        week_end = target_date - timedelta(days=1)  # Sunday
        week_start = week_end - timedelta(days=6)   # Previous Monday
        
        # Get ISO week number
        week_num = target_date.isocalendar()[1] - 1  # Previous week
        
        # Format week range
        if week_start.month == week_end.month:
            week_range = f"{week_start.day}-{week_end.day} {week_end.strftime('%B')}"
        else:
            week_range = f"{week_start.strftime('%d %b')} - {week_end.strftime('%d %b')}"
        
        return (
            f"{date_str}\n"
            f"The Week in Architecture: Best of Week {week_num} ({week_range})."
        )
    
    def get_edition_name(self) -> str:
        return "Weekly Edition"


# =============================================================================
# Factory Function
# =============================================================================

def get_formatter(edition_type: EditionType) -> EditionFormatter:
    """
    Get the appropriate formatter for an edition type.
    
    Args:
        edition_type: Type of edition
        
    Returns:
        EditionFormatter instance
    """
    formatters = {
        EditionType.DAILY: DailyFormatter,
        EditionType.WEEKEND: WeekendFormatter,
        EditionType.WEEKLY: WeeklyFormatter,
    }
    
    formatter_class = formatters.get(edition_type)
    if formatter_class is None:
        raise ValueError(f"Unknown edition type: {edition_type}")
    
    return formatter_class()


def format_edition_header(
    edition_type: EditionType,
    target_date: Optional[date] = None,
    article_count: int = 7
) -> str:
    """
    Format header for any edition type.
    
    Args:
        edition_type: Type of edition
        target_date: Publication date (defaults to today)
        article_count: Number of articles
        
    Returns:
        Formatted header string
    """
    if target_date is None:
        target_date = date.today()
    
    formatter = get_formatter(edition_type)
    return formatter.format_header(target_date, article_count)
