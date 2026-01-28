# editor/project_extractor.py
"""
Project Information Extractor for ADUmedia

Uses GPT-4o-mini to extract project details from article summaries.
This information is used for project-level deduplication.

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key
    LANGCHAIN_TRACING_V2    - Set to "true" to enable LangSmith
    LANGCHAIN_API_KEY       - LangSmith API key
    LANGCHAIN_PROJECT       - LangSmith project name
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class ProjectLocation(BaseModel):
    """Location information."""
    city: Optional[str] = None
    country: Optional[str] = None


class ExtractedProject(BaseModel):
    """Result of project extraction."""
    is_project: bool = Field(description="Whether this is about a specific project")
    project_name: Optional[str] = Field(None, description="Name of the building/project")
    architect: Optional[str] = Field(None, description="Architect firm name")
    location: ProjectLocation = Field(default_factory=ProjectLocation)
    project_type: Optional[str] = Field(None, description="Type of project")
    project_status: Optional[str] = Field(None, description="Status of project")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# =============================================================================
# Project Extractor Class
# =============================================================================

class ProjectExtractor:
    """
    Extracts project information from article summaries using AI.
    
    Used for:
    1. Linking articles to unique projects
    2. Detecting when same project is covered by multiple sources
    3. Detecting project updates (same project, 3+ months later)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        langsmith_project: Optional[str] = None
    ):
        """
        Initialize the project extractor.
        
        Args:
            model: OpenAI model to use (gpt-4o-mini is cost-effective)
            temperature: Low temperature for consistent extraction
            langsmith_project: LangSmith project name for tracing
        """
        # Configure LangSmith tracing
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            project = langsmith_project or os.getenv("LANGCHAIN_PROJECT", "adumedia-extractor")
            os.environ["LANGCHAIN_PROJECT"] = project
            print(f"[EXTRACTOR] LangSmith tracing: {project}")
        
        # Initialize LLM with JSON mode
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Load prompt template
        self.prompt_template = self._load_prompt()
        
        print(f"[EXTRACTOR] Initialized with {model}")
    
    def _load_prompt(self) -> str:
        """Load the extraction prompt template."""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "prompts" / "extract_project_info.txt",
            Path("prompts/extract_project_info.txt"),
            Path("/home/claude/prompts/extract_project_info.txt"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.read_text(encoding="utf-8")
        
        # Fallback: inline prompt
        return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Return default prompt if file not found."""
        return """You are an architecture news analyst. Extract project information from the article summary.

RULES:
- Do NOT use any emoji
- Only extract explicitly mentioned information
- Use null if not clearly stated
- For news/interviews not about specific buildings, return is_project: false

ARTICLE SUMMARY:
{summary}

METADATA:
Title: {title}
Source: {source_name}

Return ONLY valid JSON:
{{
    "is_project": true/false,
    "project_name": "Name or null",
    "architect": "Firm name or null",
    "location": {{"city": "City or null", "country": "Country or null"}},
    "project_type": "residential|commercial|cultural|educational|healthcare|hospitality|mixed_use|infrastructure|landscape|interior|masterplan|installation|other|null",
    "project_status": "completed|under_construction|announced|competition|renovation|null",
    "confidence": 0.0-1.0
}}"""
    
    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Normalize text for matching."""
        if not text:
            return ""
        
        import re
        # Lowercase, remove special chars, collapse whitespace
        normalized = text.lower().strip()
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    @staticmethod
    def generate_project_hash(name: Optional[str], architect: Optional[str]) -> str:
        """Generate hash for project matching."""
        normalized_name = ProjectExtractor.normalize_text(name)
        normalized_architect = ProjectExtractor.normalize_text(architect)
        combined = f"{normalized_name}|{normalized_architect}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def extract(
        self,
        summary: str,
        title: str,
        source_name: str,
        url: str = "",
        run_name: Optional[str] = None
    ) -> ExtractedProject:
        """
        Extract project information from an article.
        
        Args:
            summary: AI-generated summary of the article
            title: Original article title
            source_name: Source publication name
            url: Article URL (for context)
            run_name: Optional name for LangSmith trace
            
        Returns:
            ExtractedProject with extracted information
        """
        # Build prompt
        prompt = self.prompt_template.format(
            summary=summary,
            title=title,
            source_name=source_name,
            url=url
        )
        
        # Configure run metadata
        config = RunnableConfig(
            run_name=run_name or f"extract-{source_name[:10]}",
            tags=["extraction", "project-info", source_name],
            metadata={
                "source": source_name,
                "title_preview": title[:50] if title else "",
            }
        )
        
        try:
            # Call LLM
            messages = [("human", prompt)]
            response = await self.llm.ainvoke(messages, config=config)
            
            # Parse response
            result = json.loads(response.content)
            return ExtractedProject(**result)
            
        except Exception as e:
            print(f"[EXTRACTOR] Error extracting project info: {e}")
            # Return empty extraction on error
            return ExtractedProject(
                is_project=False,
                confidence=0.0
            )
    
    async def extract_batch(
        self,
        articles: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[ExtractedProject]:
        """
        Extract project info from multiple articles.
        
        Args:
            articles: List of article dicts with summary, title, source_name
            batch_size: Number of concurrent extractions
            
        Returns:
            List of ExtractedProject results (same order as input)
        """
        import asyncio
        
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [
                self.extract(
                    summary=article.get("ai_summary", ""),
                    title=article.get("title", ""),
                    source_name=article.get("source_name", "Unknown"),
                    url=article.get("link", "")
                )
                for article in batch
            ]
            
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"[EXTRACTOR] Batch item {i+j} failed: {result}")
                    results.append(ExtractedProject(is_project=False, confidence=0.0))
                else:
                    results.append(result)
            
            print(f"   [EXTRACTOR] Processed {min(i + batch_size, len(articles))}/{len(articles)} articles")
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_project_info(
    summary: str,
    title: str,
    source_name: str
) -> ExtractedProject:
    """
    Quick function to extract project info from a single article.
    
    Args:
        summary: Article summary
        title: Article title
        source_name: Source name
        
    Returns:
        ExtractedProject
    """
    extractor = ProjectExtractor()
    return await extractor.extract(summary, title, source_name)
