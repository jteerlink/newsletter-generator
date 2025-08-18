"""
Readability Agent for Phase 2 Multi-Agent System

This agent analyzes and improves content readability. It calculates readability
metrics (Flesch Reading Ease normalized 0-1, sentence/word complexity), provides
audience-specific suggestions, and can perform light automatic simplifications
in FULL processing mode.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base_agent import (
    AgentConfiguration,
    BaseSpecializedAgent,
    ProcessingContext,
    ProcessingMode,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ReadabilityMetrics:
    """Container for readability metrics."""
    normalized_flesch: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    sentence_count: int
    word_count: int


class ReadabilityAgent(BaseSpecializedAgent):
    """
    Specialized agent for readability optimization.

    Capabilities:
    - Compute readability metrics (Flesch Reading Ease normalized to 0-1)
    - Detect long sentences, complex words, and dense paragraphs
    - Provide audience-specific suggestions
    - Perform light simplifications in FULL mode
    """

    def __init__(self, config: Optional[AgentConfiguration] = None):
        super().__init__("Readability", config)
        logger.info("Readability Agent initialized")

    def _process_internal(self, context: ProcessingContext) -> ProcessingResult:
        content = context.content
        processing_mode = context.processing_mode

        metrics = self._compute_readability_metrics(content)
        suggestions: List[str] = []
        warnings: List[str] = []

        # Heuristic issues and suggestions
        if metrics.normalized_flesch < 0.6:
            warnings.append("Content may be difficult to read for a general audience")
            suggestions.append("Shorten sentences and replace complex words with simpler alternatives")

        if metrics.avg_sentence_length > 22:
            suggestions.append("Reduce average sentence length below ~20 words")

        complex_words = self._find_complex_words(content)
        if complex_words:
            suggestions.append(f"Consider simplifying complex words: {', '.join(complex_words[:8])}...")

        # Audience-specific guidance
        audience = (context.audience or "General Tech Audience").lower()
        if "business" in audience:
            suggestions.append("Reduce technical jargon; emphasize impact and outcomes")
        elif "engineer" in audience or "developer" in audience:
            suggestions.append("Keep concise but precise; prefer concrete examples over abstract phrasing")

        # Mobile readability heuristics
        mobile_warnings, mobile_suggestions = self._assess_mobile_readability(content)
        warnings.extend(mobile_warnings)
        suggestions.extend(mobile_suggestions)

        # Optional light transformation
        processed_content = content
        if processing_mode == ProcessingMode.FULL and metrics.normalized_flesch < 0.7:
            processed_content = self._lightly_simplify_content(content)

        quality_score = max(0.0, min(1.0, metrics.normalized_flesch))

        return ProcessingResult(
            success=True,
            processed_content=processed_content,
            quality_score=quality_score,
            confidence_score=quality_score,
            suggestions=suggestions,
            warnings=warnings,
            metadata={
                "readability_metrics": metrics.__dict__,
                "processing_mode": processing_mode.value,
            },
        )

    def _process_fallback(self, context: ProcessingContext) -> ProcessingResult:
        # Minimal analysis only
        metrics = self._compute_readability_metrics(context.content)
        return ProcessingResult(
            success=True,
            processed_content=context.content,
            quality_score=metrics.normalized_flesch,
            confidence_score=metrics.normalized_flesch,
            suggestions=["Fallback mode: Limited readability analysis performed"],
            warnings=[],
            metadata={"readability_metrics": metrics.__dict__, "processing_mode": "fallback"},
        )

    # --- Internal helpers ---

    def _compute_readability_metrics(self, text: str) -> ReadabilityMetrics:
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return ReadabilityMetrics(0.0, 0.0, 0.0, 0, 0)

        syllables = self._count_syllables(text)
        avg_sentence_len = word_count / sentences
        avg_syllables_per_word = syllables / max(1, word_count)

        flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)
        normalized = max(0.0, min(1.0, flesch / 100.0))
        return ReadabilityMetrics(
            normalized_flesch=normalized,
            avg_sentence_length=avg_sentence_len,
            avg_syllables_per_word=avg_syllables_per_word,
            sentence_count=sentences,
            word_count=word_count,
        )

    def _count_syllables(self, text: str) -> int:
        # Simple heuristic syllable counter
        text = text.lower()
        words = re.findall(r"[a-zA-Z]+", text)
        vowels = "aeiouy"
        count = 0
        for word in words:
            syllables = 0
            previous_char_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_char_was_vowel:
                    syllables += 1
                previous_char_was_vowel = is_vowel
            if word.endswith("e") and syllables > 1:
                syllables -= 1
            count += max(1, syllables)
        return count

    def _find_complex_words(self, text: str) -> List[str]:
        # Identify long words as a proxy for complexity
        words = re.findall(r"[A-Za-z]{10,}", text)
        # Deduplicate preserving order
        seen: set = set()
        unique = []
        for w in words:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(w)
        return unique

    def _lightly_simplify_content(self, text: str) -> str:
        # Break overly long sentences and replace a few common jargon terms
        replacements = {
            "utilize": "use",
            "leverage": "use",
            "approximately": "about",
            "facilitate": "help",
            "demonstrate": "show",
        }

        simplified = text
        for src, dst in replacements.items():
            simplified = re.sub(rf"\b{re.escape(src)}\b", dst, simplified, flags=re.IGNORECASE)

        # Break sentences longer than ~35 words by inserting a period near the middle
        def break_long_sentence(sentence: str) -> str:
            words = sentence.split()
            if len(words) <= 35:
                return sentence
            mid = len(words) // 2
            return " ".join(words[:mid]) + ". " + " ".join(words[mid:])

        sentences = re.split(r"(?<=[.!?])\s+", simplified)
        sentences = [break_long_sentence(s) for s in sentences]
        return " ".join(sentences)

    def _assess_mobile_readability(self, text: str) -> tuple[List[str], List[str]]:
        warnings: List[str] = []
        suggestions: List[str] = []
        # Paragraph length
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        long_paras = [p for p in paragraphs if len(p.split()) > 120]
        if long_paras:
            warnings.append("Some paragraphs are long for mobile screens")
            suggestions.append("Break long paragraphs into smaller chunks (<100 words)")
        # Headings and bullets density
        headings = re.findall(r"^#+\s+", text, flags=re.MULTILINE)
        bullets = re.findall(r"^\s*[-*]\s+", text, flags=re.MULTILINE)
        if len(headings) < 2:
            suggestions.append("Add more headings for scannability")
        if len(bullets) < 2:
            suggestions.append("Use bullet lists for dense information")
        # Approximate line length by word count per sentence fragment
        long_lines = [s for s in re.split(r"\n", text) if len(s.split()) > 30]
        if long_lines:
            suggestions.append("Insert line breaks for long lines to improve mobile legibility")
        return warnings, suggestions


__all__ = ["ReadabilityAgent", "ReadabilityMetrics"]


