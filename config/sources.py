# config/sources.py
"""
Source Configuration for Telegram Publisher

Combined source registry for looking up display names from URLs.
Includes all sources from both RSS and Custom Scraper pipelines.
"""

from urllib.parse import urlparse
from typing import Optional


# =============================================================================
# Source Registry
# =============================================================================

# Maps domain patterns to display names
# Combined from RSS sources and Custom Scraper sources

SOURCE_NAMES = {
    # =========================================================================
    # RSS SOURCES - Tier 1 Global Primary
    # =========================================================================
    "archdaily.com": "ArchDaily",
    "dezeen.com": "Dezeen",
    "designboom.com": "Designboom",
    "architectsjournal.co.uk": "The Architects' Journal",
    "archpaper.com": "The Architect's Newspaper",

    # =========================================================================
    # RSS SOURCES - Tier 2 North America
    # =========================================================================
    "canadianarchitect.com": "Canadian Architect",
    "design-milk.com": "Design Milk",
    "leibal.com": "Leibal",
    "constructionspecifier.com": "The Construction Specifier",
    "architecturalrecord.com": "Architectural Record",
    "nextcity.org": "Next City",
    "placesjournal.org": "Places Journal",
    "planetizen.com": "Planetizen",

    # =========================================================================
    # RSS SOURCES - Tier 2 Europe
    # =========================================================================
    "architectural-review.com": "The Architectural Review",
    "archi.ru": "Archi.ru",

    # =========================================================================
    # RSS SOURCES - Tier 2 Asia-Pacific
    # =========================================================================
    "yellowtrace.com.au": "Yellowtrace",
    "architectureau.com": "ArchitectureAU",
    "architecturenow.co.nz": "Architecture Now",
    "architectureupdate.in": "Architecture Update",
    "indesignlive.sg": "Indesign Live Singapore",

    # =========================================================================
    # RSS SOURCES - Tier 2 Latin America
    # =========================================================================
    "archdaily.com.br": "ArchDaily Brasil",
    "arquine.com": "Arquine",

    # =========================================================================
    # RSS SOURCES - Tier 2 Middle East / Computational
    # =========================================================================
    "parametric-architecture.com": "Parametric Architecture",

    # =========================================================================
    # CUSTOM SCRAPER SOURCES - Middle East
    # =========================================================================
    "identity.ae": "Identity Magazine",

    # =========================================================================
    # CUSTOM SCRAPER SOURCES - Asia-Pacific
    # =========================================================================
    "archiposition.com": "Archiposition",
    "gooood.cn": "Gooood",
    "japan-architects.com": "Japan Architects",

    # =========================================================================
    # CUSTOM SCRAPER SOURCES - Europe
    # =========================================================================
    "prorus.ru": "ProRus",
    "bauwelt.de": "Bauwelt",
    "domusweb.it": "Domus",
    "metalocus.es": "Metalocus",

    # =========================================================================
    # CUSTOM SCRAPER SOURCES - North America
    # =========================================================================
    "metropolismag.com": "Metropolis",
    "landscapearchitecturemagazine.org": "Landscape Architecture Magazine",

    # =========================================================================
    # CUSTOM SCRAPER SOURCES - International
    # =========================================================================
    "worldlandscapearchitect.com": "World Landscape Architect",
}


def get_source_name(url: str) -> str:
    """
    Get display name for a source URL.

    Args:
        url: Article URL

    Returns:
        Human-readable source name
    """
    if not url:
        return "Source"

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Try exact match first
        if domain in SOURCE_NAMES:
            return SOURCE_NAMES[domain]

        # Try partial match (for subdomains)
        for pattern, name in SOURCE_NAMES.items():
            if domain.endswith(pattern) or pattern in domain:
                return name

        # Fallback: capitalize domain
        return domain.split(".")[0].capitalize()

    except Exception:
        return "Source"


def get_source_id_from_url(url: str) -> str:
    """
    Extract source ID from URL.

    Args:
        url: Article URL

    Returns:
        Source ID (domain without TLD)
    """
    if not url:
        return "unknown"

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www.
        if domain.startswith("www."):
            domain = domain[4:]

        # Return first part of domain
        return domain.split(".")[0]

    except Exception:
        return "unknown"