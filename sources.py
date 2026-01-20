# config/sources.py
"""
Source Configuration for Telegram Publisher

Minimal source registry for looking up display names from URLs.
"""

from urllib.parse import urlparse
from typing import Optional


# =============================================================================
# Source Registry
# =============================================================================

# Maps domain patterns to display names
SOURCE_NAMES = {
    # Tier 1 - Major International
    "archdaily.com": "ArchDaily",
    "dezeen.com": "Dezeen",
    "designboom.com": "Designboom",
    "arch2o.com": "Arch2O",
    "archello.com": "Archello",
    "architecturaldigest.com": "AD",
    "architizer.com": "Architizer",
    "archpaper.com": "Architect's Newspaper",
    "world-architects.com": "World Architects",
    "worldarchitecture.org": "World Architecture",
    "e-architect.com": "e-architect",
    "architectmagazine.com": "Architect Magazine",
    
    # Tier 2 - Regional/Specialty
    "afasiaarchzine.com": "Afasia",
    "metalocus.es": "Metalocus",
    "archinect.com": "Archinect",
    "archiscene.net": "ArchiSCENE",
    "aasarchitecture.com": "AAS Architecture",
    "architonic.com": "Architonic",
    "baunetz.de": "BauNetz",
    "detail.de": "Detail",
    
    # Tier 3 - Niche/Regional
    "gooood.cn": "Gooood",
    "archiposition.com": "Archiposition",
    "prorus.org": "ProRus",
    "bauwelt.de": "Bauwelt",
    "identity.ae": "Identity",
    "landezine.com": "Landezine",
    "landscapearchitecturemagazine.org": "LAM",
    "thearchitectureinsight.com": "Architecture Insight",
    
    # Additional
    "floornature.com": "Floornature",
    "divisare.com": "Divisare",
    "domusweb.it": "Domus",
    "frameweb.com": "Frame",
    "abitare.it": "Abitare",
    "archilovers.com": "Archilovers",
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
        return "Unknown"
    
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
        return "Unknown"


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
