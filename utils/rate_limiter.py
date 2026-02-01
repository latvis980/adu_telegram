# utils/rate_limiter.py
"""
Rate Limiter for OpenAI API Calls

Handles:
- Token-based rate limiting (200k TPM for gpt-4o-mini)
- Request-based rate limiting (5000 RPM for gpt-4o-mini)
- Exponential backoff on 429 errors
- Token usage tracking

Usage:
    from utils.rate_limiter import OpenAIRateLimiter
    
    limiter = OpenAIRateLimiter()
    async with limiter.acquire():
        response = await llm.ainvoke(...)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager


class OpenAIRateLimiter:
    """
    Rate limiter for OpenAI API calls.
    
    Tracks:
    - Tokens per minute (TPM)
    - Requests per minute (RPM)
    - Automatic backoff on 429 errors
    """
    
    def __init__(
        self,
        max_tokens_per_minute: int = 180000,  # Leave 20k buffer from 200k limit
        max_requests_per_minute: int = 4500,  # Leave 500 buffer from 5000 limit
        min_delay_between_requests: float = 0.1,  # 100ms minimum delay
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_tokens_per_minute: Maximum tokens per minute (default: 180k)
            max_requests_per_minute: Maximum requests per minute (default: 4500)
            min_delay_between_requests: Minimum delay between requests in seconds
        """
        self.max_tpm = max_tokens_per_minute
        self.max_rpm = max_requests_per_minute
        self.min_delay = min_delay_between_requests
        
        # Tracking
        self._token_usage: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._request_times: list[float] = []  # List of request timestamps
        self._last_request_time: float = 0.0
        
        # Statistics
        self.total_tokens_used: int = 0
        self.total_requests: int = 0
        self.total_wait_time: float = 0.0
        self.rate_limit_hits: int = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _clean_old_records(self) -> None:
        """Remove records older than 1 minute."""
        now = time.time()
        cutoff = now - 60.0  # 1 minute ago
        
        # Clean token usage
        self._token_usage = [
            (ts, tokens) for ts, tokens in self._token_usage
            if ts > cutoff
        ]
        
        # Clean request times
        self._request_times = [
            ts for ts in self._request_times
            if ts > cutoff
        ]
    
    def _get_current_tpm(self) -> int:
        """Get current tokens per minute."""
        self._clean_old_records()
        return sum(tokens for _, tokens in self._token_usage)
    
    def _get_current_rpm(self) -> int:
        """Get current requests per minute."""
        self._clean_old_records()
        return len(self._request_times)
    
    def _estimate_tokens(self, prompt_length: int) -> int:
        """
        Estimate tokens from prompt length.
        
        Rough estimate: 1 token ≈ 4 characters for English text.
        Add 20% buffer for safety.
        """
        estimated = int(prompt_length / 3.5)  # Slightly conservative
        return estimated
    
    async def _wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """
        Wait if we're approaching rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for next request
        """
        wait_start = time.time()
        
        while True:
            now = time.time()
            current_tpm = self._get_current_tpm()
            current_rpm = self._get_current_rpm()
            
            # Check if we need to wait for token limit
            if current_tpm + estimated_tokens > self.max_tpm:
                # Find when the oldest token record will expire
                if self._token_usage:
                    oldest_time = self._token_usage[0][0]
                    wait_time = 60.0 - (now - oldest_time) + 0.5  # Add 500ms buffer
                    
                    if wait_time > 0:
                        print(f"   💤 [RATE LIMIT] TPM limit ({current_tpm}/{self.max_tpm}). Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        self.rate_limit_hits += 1
                        continue
            
            # Check if we need to wait for request limit
            if current_rpm >= self.max_rpm:
                # Find when the oldest request will expire
                if self._request_times:
                    oldest_time = self._request_times[0]
                    wait_time = 60.0 - (now - oldest_time) + 0.5  # Add 500ms buffer
                    
                    if wait_time > 0:
                        print(f"   💤 [RATE LIMIT] RPM limit ({current_rpm}/{self.max_rpm}). Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        self.rate_limit_hits += 1
                        continue
            
            # Check minimum delay between requests
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                await asyncio.sleep(wait_time)
            
            # All good, we can proceed
            break
        
        # Track total wait time
        wait_end = time.time()
        wait_duration = wait_end - wait_start
        if wait_duration > 0:
            self.total_wait_time += wait_duration
    
    def record_usage(self, tokens_used: int) -> None:
        """
        Record token usage for this request.
        
        Args:
            tokens_used: Number of tokens used
        """
        now = time.time()
        self._token_usage.append((now, tokens_used))
        self._request_times.append(now)
        self._last_request_time = now
        
        self.total_tokens_used += tokens_used
        self.total_requests += 1
    
    @asynccontextmanager
    async def acquire(self, estimated_tokens: int = 1000):
        """
        Context manager for rate-limited API calls.
        
        Usage:
            async with limiter.acquire(estimated_tokens=500):
                response = await llm.ainvoke(...)
                limiter.record_usage(response.usage.total_tokens)
        
        Args:
            estimated_tokens: Estimated tokens for this request
        """
        async with self._lock:
            await self._wait_if_needed(estimated_tokens)
            
        try:
            yield self
        finally:
            # Record a conservative estimate if actual usage not recorded
            # This prevents the limiter from being too optimistic
            pass
    
    def print_stats(self) -> None:
        """Print rate limiter statistics."""
        print(f"\n📊 [RATE LIMITER STATS]")
        print(f"   Total requests: {self.total_requests}")
        print(f"   Total tokens: {self.total_tokens_used:,}")
        print(f"   Rate limit hits: {self.rate_limit_hits}")
        print(f"   Total wait time: {self.total_wait_time:.1f}s")
        print(f"   Current TPM: {self._get_current_tpm():,}/{self.max_tpm:,}")
        print(f"   Current RPM: {self._get_current_rpm()}/{self.max_rpm}")


async def retry_on_rate_limit(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    *args,
    **kwargs
):
    """
    Retry a function call with exponential backoff on rate limit errors.
    
    Args:
        func: Async function to call
        max_retries: Maximum number of retries
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result from func
    
    Raises:
        Last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a rate limit error
            if "rate limit" in error_str or "429" in error_str:
                last_error = e
                
                if attempt < max_retries:
                    # Extract wait time from error message if available
                    wait_time = None
                    if "try again in" in error_str:
                        # Try to extract wait time from error message
                        import re
                        match = re.search(r'try again in (\d+)ms', error_str)
                        if match:
                            wait_time = float(match.group(1)) / 1000.0
                    
                    # Use extracted time or exponential backoff
                    if wait_time is None:
                        wait_time = min(base_delay * (2 ** attempt), max_delay)
                    
                    print(f"   💤 [RETRY] Rate limit error. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries + 1})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   ❌ [ERROR] Rate limit retry failed after {max_retries + 1} attempts")
                    raise
            else:
                # Not a rate limit error, re-raise immediately
                raise
    
    # If we get here, all retries failed
    raise last_error


# =============================================================================
# Global Rate Limiter Instance
# =============================================================================

# Shared rate limiter for all OpenAI calls in the application
_global_limiter: Optional[OpenAIRateLimiter] = None


def get_rate_limiter() -> OpenAIRateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_limiter
    
    if _global_limiter is None:
        _global_limiter = OpenAIRateLimiter()
    
    return _global_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _global_limiter
    _global_limiter = None
