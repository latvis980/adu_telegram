# editions/formatters.py
"""
Edition Formatters for ADUmedia

Provides edition-specific headers and formatting for Telegram messages.
No emoji - clean, professional formatting only.

Schedule:
    Monday    - Weekend Catch-Up Edition (Fri, Sat, Sun, Mon)
    Tue-Fri   - Daily Edition (2 days)
    Sat/Sun   - No publication
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
        pass
    
    @abstractmethod
    def get_edition_name(self) -> str:
        """Get the display name for this edition type."""
        pass


# =============================================================================
# Daily Edition Formatter
# =============================================================================

class DailyFormatter(EditionFormatter):
    """Formatter for daily editions (Tue/Wed/Thu/Fri)."""
    
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        date_str = target_date.strftime("%d %B %Y")
        return (
            f"{date_str}\n"
            f"Our editorial selection for today -- Daily Edition"
        )
    
    def get_edition_name(self) -> str:
        return "Daily Edition"


# =============================================================================
# Weekend Edition Formatter
# =============================================================================

class WeekendFormatter(EditionFormatter):
    """Formatter for weekend catch-up edition (Monday)."""
    
    def format_header(self, target_date: date, article_count: int = 7) -> str:
        date_str = target_date.strftime("%d %B %Y")
        
        # Calculate coverage period (Fri through Mon)
        friday = target_date - timedelta(days=3)
        
        # Format as "14-17 February" or "29 Jan - 1 Feb" if crossing months
        if friday.month == target_date.month:
            coverage = f"{friday.day}-{target_date.day} {target_date.strftime('%B')}"
        else:
            coverage = f"{friday.strftime('%d %b')} - {target_date.strftime('%d %b')}"
        
        return (
            f"{date_str}\n"
            f"Our editorial selection for today -- Weekend Catch-Up\n"
            f"Top stories from {coverage}."
        )
    
    def get_edition_name(self) -> str:
        return "Weekend Catch-Up Edition"


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
