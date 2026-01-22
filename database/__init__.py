# database/__init__.py
"""
Database Module for ADUmedia

Handles Supabase connection and data models.
"""

from .connection import get_supabase_client, SupabaseConnection
from .models import PublishedArticle, Edition, WeeklyCandidate

__all__ = [
    "get_supabase_client",
    "SupabaseConnection", 
    "PublishedArticle",
    "Edition",
    "WeeklyCandidate",
]
