# editor/__init__.py
"""
AI Editor Module for ADUmedia

Handles article selection using LangChain with LangSmith tracing.
"""

from .selector import ArticleSelector, EditionType
from .deduplication import DeduplicationChecker

__all__ = ["ArticleSelector", "EditionType", "DeduplicationChecker"]
