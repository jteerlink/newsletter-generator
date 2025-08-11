# src/scrapers/content_analyzer.py
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    Analyzes content for quality, categorization, and metadata enrichment.
    Handles content deduplication, quality scoring, source reliability assessment,
    and metadata enrichment for Phase 1.2.
    """

    def __init__(self):
        # Quality scoring weights for each factor (tuned based on project
        # needs)
        self.quality_weights = {
            "word_count": 0.3,      # Heavier weight for content length
            "readability": 0.1,     # Lower weight for readability
            "source_reliability": 0.15,  # Trust in the source
            "freshness": 0.15,      # Recency of content
            "completeness": 0.3,    # How complete the content/metadata is
        }

        # Source reliability scores (domain knowledge; can be tuned as needed)
        self.source_reliability_scores = {
            "arxiv.org": 0.95,
            "nature.com": 0.9,
            "science.org": 0.9,
            "ieee.org": 0.85,
            "acm.org": 0.85,
            "google.com": 0.7,
            "github.com": 0.8,
            "medium.com": 0.6,
            "substack.com": 0.65,
            "techcrunch.com": 0.7,
            "wired.com": 0.75,
            "theverge.com": 0.7,
        }

        # Content categories and keywords
        self.category_keywords = {
            "AI/ML": [
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "neural network",
                "AI",
                "ML",
                "algorithm",
            ],
            "Research": [
                "research",
                "study",
                "paper",
                "academic",
                "scientific",
                "experiment",
            ],
            "Industry": [
                "company",
                "startup",
                "business",
                "industry",
                "market",
                "product",
            ],
            "Technology": [
                "technology",
                "tech",
                "software",
                "hardware",
                "system",
                "platform",
            ],
            "Conference": [
                "conference",
                "workshop",
                "symposium",
                "presentation",
                "talk",
            ],
            "Tutorial": ["tutorial", "guide", "how-to", "step-by-step", "explanation"],
            "News": ["announcement", "news", "update", "release", "launch"],
        }

    def analyze_content(self, content: str,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive content analysis including quality scoring, categorization, and metadata enrichment.
        """
        analysis = {
            "quality_score": 0.0,
            "category": "Unknown",
            "tags": [],
            "word_count": 0,
            "readability_score": 0.0,
            "source_reliability": 0.5,
            "freshness_score": 0.0,
            "completeness_score": 0.0,
            "content_hash": "",
            "is_duplicate": False,
            "enriched_metadata": {},
        }

        # Basic content analysis
        analysis["word_count"] = self._count_words(content)
        analysis["readability_score"] = self._calculate_readability(content)
        analysis["content_hash"] = self._generate_content_hash(content)

        # Categorization and tagging
        analysis["category"] = self._categorize_content(content, metadata)
        analysis["tags"] = self._extract_tags(content, metadata)

        # Quality scoring
        analysis["source_reliability"] = self._assess_source_reliability(
            metadata)
        analysis["freshness_score"] = self._calculate_freshness(metadata)
        analysis["completeness_score"] = self._assess_completeness(
            content, metadata)
        analysis["quality_score"] = self._calculate_quality_score(analysis)

        # Metadata enrichment
        analysis["enriched_metadata"] = self._enrich_metadata(
            metadata, analysis)
        # Ensure analyzed_at is present at the top level
        analysis["analyzed_at"] = analysis["enriched_metadata"].get(
            "analyzed_at")

        return analysis

    def _count_words(self, content: str) -> int:
        """Count words in content."""
        if not content:
            return 0
        words = re.findall(r"\b\w+\b", content.lower())
        return len(words)

    def _calculate_readability(self, content: str) -> float:
        """
        Calculate readability score using Flesch Reading Ease.
        Returns score between 0 and 1, where 1 is most readable.
        """
        if not content:
            return 0.0

        sentences = re.split(r"[.!?]+", content)
        words = re.findall(r"\b\w+\b", content.lower())
        syllables = self._count_syllables(content)

        if len(sentences) == 0 or len(words) == 0:
            return 0.0

        # Flesch Reading Ease formula
        flesch_score = (
            206.835
            - (1.015 * (len(words) / len(sentences)))
            - (84.6 * (syllables / len(words)))
        )

        # Normalize to 0-1 range (Flesch typically ranges from 0-100)
        # This makes it easier to combine with other scores
        return max(0.0, min(1.0, flesch_score / 100.0))

    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting."""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False

        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel

        return max(1, count)

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _categorize_content(
            self, content: str, metadata: Dict[str, Any]) -> str:
        """Categorize content based on keywords and metadata."""
        content_lower = content.lower()

        # Check each category
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    return category

        # Fallback based on metadata
        if metadata.get("category"):
            return metadata["category"]

        return "General"

    def _extract_tags(self,
                      content: str,
                      metadata: Dict[str,
                                     Any]) -> List[str]:
        """Extract relevant tags from content and metadata."""
        tags = set()

        # Extract from metadata
        if metadata.get("tags"):
            if isinstance(metadata["tags"], str):
                try:
                    tags.update(json.loads(metadata["tags"]))
                except BaseException:
                    tags.add(metadata["tags"])
            elif isinstance(metadata["tags"], list):
                tags.update(metadata["tags"])

        # Extract from content (simple keyword extraction)
        content_lower = content.lower()
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    tags.add(keyword)

        return list(tags)[:10]  # Limit to 10 tags

    def _assess_source_reliability(self, metadata: Dict[str, Any]) -> float:
        """Assess source reliability based on domain and metadata."""
        url = metadata.get("url", "")
        source = metadata.get("source", "")

        # Check domain-based reliability
        domain = urlparse(url).netloc if url else ""
        for reliable_domain, score in self.source_reliability_scores.items():
            if reliable_domain in domain:
                return score

        # Check source name
        for reliable_domain, score in self.source_reliability_scores.items():
            if (
                reliable_domain.replace(".org", "").replace(".com", "")
                in source.lower()
            ):
                return score

        # Default score for unknown sources
        return 0.5

    def _calculate_freshness(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on publication date."""
        published = metadata.get("published")
        if not published:
            return 0.5

        try:
            if isinstance(published, str):
                published_date = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                )
            else:
                published_date = published

            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            days_old = (now - published_date).days

            # Exponential decay: newer content gets higher scores
            freshness = max(0.1, min(1.0, 2.0 ** (-days_old / 30)))
            return freshness

        except Exception as e:
            logger.warning(f"Error calculating freshness: {e}")
            return 0.5

    def _assess_completeness(
            self, content: str, metadata: Dict[str, Any]) -> float:
        """Assess content completeness based on various factors."""
        score = 0.0

        # Each factor below adds a fixed increment to the completeness score
        # These increments are 'magic values' chosen for reasonable weighting
        # Content present: +0.3
        if content and len(content.strip()) > 0:
            score += 0.3

        # Title present: +0.2
        if metadata.get("title") and len(metadata["title"].strip()) > 0:
            score += 0.2

        # Description present: +0.2
        if metadata.get("description") and len(
                metadata["description"].strip()) > 0:
            score += 0.2

        # Word count: +0.2 for >100, +0.1 for >50
        word_count = self._count_words(content)
        if word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1

        # Author present: +0.1
        if metadata.get("author"):
            score += 0.1

        return min(1.0, score)

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score using weighted factors. Penalize short/incomplete content."""
        score = 0.0
        # Word count score (normalized)
        word_count_score = min(1.0, analysis["word_count"] / 1000.0)
        score += word_count_score * self.quality_weights["word_count"]
        # Readability score
        score += analysis["readability_score"] * \
            self.quality_weights["readability"]
        # Source reliability
        score += (analysis["source_reliability"] *
                  self.quality_weights["source_reliability"])
        # Freshness
        score += analysis["freshness_score"] * \
            self.quality_weights["freshness"]
        # Completeness
        score += analysis["completeness_score"] * \
            self.quality_weights["completeness"]
        # Cap score for very short content (<20 words)
        # This prevents high scores for trivial/insufficient content
        if analysis["word_count"] < 20:
            score = min(score, 0.4)
        # Cap score for incomplete content (<0.5 completeness)
        if analysis["completeness_score"] < 0.5:
            score = min(score, 0.5)
        return round(score, 3)

    def _enrich_metadata(
        self, metadata: Dict[str, Any], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich metadata with analysis results."""
        enriched = metadata.copy()

        enriched.update(
            {
                "quality_score": analysis["quality_score"],
                "category": analysis["category"],
                "tags": analysis["tags"],
                "word_count": analysis["word_count"],
                "readability_score": analysis["readability_score"],
                "source_reliability": analysis["source_reliability"],
                "freshness_score": analysis["freshness_score"],
                "completeness_score": analysis["completeness_score"],
                "content_hash": analysis["content_hash"],
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        return enriched

    def detect_duplicates(
            self,
            content_hash: str,
            existing_hashes: List[str]) -> bool:
        """Check if content is a duplicate based on hash."""
        return content_hash in existing_hashes

    def filter_by_quality(
        self, articles: List[Dict[str, Any]], min_quality: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filter articles by minimum quality score."""
        return [
            article
            for article in articles
            if article.get("quality_score", 0) >= min_quality
        ]

    def sort_by_relevance(
        self, articles: List[Dict[str, Any]], query: str = None
    ) -> List[Dict[str, Any]]:
        """Sort articles by relevance (quality score and freshness)."""

        def relevance_score(article):
            quality = article.get("quality_score", 0)
            freshness = article.get("freshness_score", 0.5)
            return quality * 0.7 + freshness * 0.3

        return sorted(articles, key=relevance_score, reverse=True)
