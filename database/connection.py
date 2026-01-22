# database/connection.py
"""
Supabase Connection Handler for ADUmedia

Provides singleton connection to Supabase database.

Environment Variables:
    SUPABASE_URL    - Supabase project URL
    SUPABASE_KEY    - Supabase API key (anon or service role)
"""

import os
from typing import Optional
from supabase import create_client, Client


# Global client instance
_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase client instance.
    
    Returns:
        Supabase client
        
    Raises:
        ValueError: If credentials are not configured
    """
    global _client
    
    if _client is not None:
        return _client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is not set")
    if not key:
        raise ValueError("SUPABASE_KEY environment variable is not set")
    
    _client = create_client(url, key)
    print(f"[DB] Connected to Supabase")
    
    return _client


class SupabaseConnection:
    """
    Context manager for Supabase operations.
    
    Usage:
        with SupabaseConnection() as db:
            result = db.table("articles").select("*").execute()
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
    
    def __enter__(self) -> Client:
        self.client = get_supabase_client()
        return self.client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Supabase client doesn't need explicit cleanup
        pass


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        client = get_supabase_client()
        
        # Try a simple query
        result = client.table("published_articles")\
            .select("id")\
            .limit(1)\
            .execute()
        
        print("[DB] Connection test successful")
        return True
        
    except Exception as e:
        print(f"[DB] Connection test failed: {e}")
        return False
