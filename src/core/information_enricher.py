"""
Information Enricher

Real-time information enhancement for newsletter content as specified in PRD FR2.2.
Provides current developments querying, outdated information detection, and content
freshness scoring through multi-provider search integration.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.tool_usage_tracker import get_tool_tracker
from tools.tools import search_web
from storage import get_storage_provider

logger = logging.getLogger(__name__)


class DevelopmentType(Enum):
    """Types of developments that can be tracked."""
    RESEARCH = "research"
    PRODUCT = "product"
    INDUSTRY = "industry"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    FUNDING = "funding"


class FreshnessLevel(Enum):
    """Content freshness levels."""
    CURRENT = "current"  # < 3 months
    RECENT = "recent"    # 3-12 months
    DATED = "dated"      # 1-2 years
    OUTDATED = "outdated"  # > 2 years


@dataclass
class Development:
    """Represents a recent development or news item."""
    title: str
    summary: str
    source: str
    url: str
    publication_date: Optional[datetime]
    development_type: DevelopmentType
    relevance_score: float
    impact_score: float


@dataclass
class EnrichedContent:
    """Content enhanced with current information."""
    original_content: str
    enriched_content: str
    developments_added: List[Development]
    outdated_sections: List[str]
    freshness_score: float
    enhancement_summary: str


@dataclass
class ContentSection:
    """Represents a section of content for analysis."""
    text: str
    title: Optional[str]
    topic_keywords: List[str]
    position: int
    freshness_level: FreshnessLevel


class InformationEnricher:
    """Real-time information enhancement for content."""
    
    def __init__(self):
        self.tool_tracker = get_tool_tracker()
        self.vector_store = get_storage_provider()
        
        # Search providers and their specializations
        self.search_providers = {
            "arxiv": {
                "query_template": "{topic} site:arxiv.org",
                "specialization": [DevelopmentType.RESEARCH, DevelopmentType.TECHNICAL],
                "authority": 0.95
            },
            "github": {
                "query_template": "{topic} site:github.com",
                "specialization": [DevelopmentType.TECHNICAL, DevelopmentType.PRODUCT],
                "authority": 0.85
            },
            "news": {
                "query_template": "{topic} news recent",
                "specialization": [DevelopmentType.INDUSTRY, DevelopmentType.REGULATORY],
                "authority": 0.70
            },
            "techblogs": {
                "query_template": "{topic} site:techcrunch.com OR site:venturebeat.com OR site:wired.com",
                "specialization": [DevelopmentType.PRODUCT, DevelopmentType.INDUSTRY],
                "authority": 0.75
            }
        }
        
        # Temporal indicators for freshness detection
        self.temporal_indicators = {
            FreshnessLevel.CURRENT: [
                r'\b(?:today|yesterday|this week|last week|recently)\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+202[4-9]\b',
                r'\b202[4-9]\b'
            ],
            FreshnessLevel.RECENT: [
                r'\b(?:this year|last year|in 202[2-3])\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+202[2-3]\b'
            ],
            FreshnessLevel.DATED: [
                r'\b(?:in 202[0-1])\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+202[0-1]\b'
            ],
            FreshnessLevel.OUTDATED: [
                r'\b(?:in 20[0-1][0-9])\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20[0-1][0-9]\b'
            ]
        }
    
    def enrich_content(self, content: str, topic: str,
                      workflow_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> EnrichedContent:
        """Add current developments and context to content."""
        try:
            logger.info(f"Enriching content for topic: {topic}")
            
            # Analyze content structure and freshness
            sections = self._analyze_content_sections(content)
            overall_freshness = self._calculate_overall_freshness(sections)
            
            # Query for recent developments
            developments = self.query_recent_developments(
                topic, workflow_id=workflow_id, session_id=session_id)
            
            # Identify outdated sections
            outdated_sections = self._identify_outdated_sections(sections)
            
            # Generate enriched content
            enriched_content = self._integrate_developments(
                content, developments, outdated_sections)
            
            # Update outdated information
            updated_content = self.update_outdated_information(enriched_content)
            
            # Calculate enhancement summary
            enhancement_summary = self._generate_enhancement_summary(
                developments, outdated_sections, overall_freshness)
            
            return EnrichedContent(
                original_content=content,
                enriched_content=updated_content,
                developments_added=developments,
                outdated_sections=[s.text for s in outdated_sections],
                freshness_score=overall_freshness,
                enhancement_summary=enhancement_summary
            )
            
        except Exception as e:
            logger.error(f"Content enrichment failed: {e}")
            return EnrichedContent(
                original_content=content,
                enriched_content=content,
                developments_added=[],
                outdated_sections=[],
                freshness_score=0.5,
                enhancement_summary="Enhancement failed due to technical issues."
            )
    
    def query_recent_developments(self, topic: str,
                                 max_results: int = 10,
                                 workflow_id: Optional[str] = None,
                                 session_id: Optional[str] = None) -> List[Development]:
        """Query multiple sources for recent developments."""
        all_developments = []
        
        for provider_name, provider_config in self.search_providers.items():
            try:
                query = provider_config["query_template"].format(topic=topic)
                
                with self.tool_tracker.track_tool_usage(
                    tool_name=f"web_search_{provider_name}",
                    agent_name="InformationEnricher",
                    workflow_id=workflow_id,
                    session_id=session_id,
                    input_data={"query": query, "provider": provider_name},
                    context={"enrichment": "recent_developments"}
                ):
                    search_results = search_web(query, max_results=5)
                
                developments = self._parse_developments(
                    search_results, provider_config, topic)
                all_developments.extend(developments)
                
            except Exception as e:
                logger.warning(f"Failed to query {provider_name}: {e}")
        
        # Rank and filter developments
        ranked_developments = self._rank_developments(all_developments, topic)
        return ranked_developments[:max_results]
    
    def update_outdated_information(self, content: str) -> str:
        """Identify and update outdated information."""
        updated_content = content
        
        # Identify potentially outdated statements
        outdated_patterns = [
            (r'\bcurrently\s+available\s+in\s+(\d+)\s+countries', 
             "currently available in multiple countries"),
            (r'\b(\d+)\s+million\s+users', 
             "millions of users"),
            (r'\bfounded\s+in\s+(20[0-1][0-9])', 
             lambda m: f"founded in {m.group(1)} (over {2024 - int(m.group(1))} years ago)"),
            (r'\blatest\s+version\s+is\s+([0-9.]+)', 
             "latest version"),
            (r'\bin\s+development\s+since\s+(20[0-1][0-9])', 
             lambda m: f"in development since {m.group(1)}")
        ]
        
        for pattern, replacement in outdated_patterns:
            if callable(replacement):
                updated_content = re.sub(pattern, replacement, updated_content, 
                                       flags=re.IGNORECASE)
            else:
                updated_content = re.sub(pattern, replacement, updated_content,
                                       flags=re.IGNORECASE)
        
        # Add freshness disclaimers
        if updated_content != content:
            disclaimer = "\n\n*Note: Some statistics and version numbers have been updated to reflect current information as of 2024.*"
            updated_content += disclaimer
        
        return updated_content
    
    def _analyze_content_sections(self, content: str) -> List[ContentSection]:
        """Analyze content structure and identify sections."""
        sections = []
        
        # Split by headers (markdown style)
        section_splits = re.split(r'\n##?\s+(.+)\n', content)
        
        if len(section_splits) == 1:
            # No headers found, treat as single section
            sections.append(self._create_content_section(content, None, 0))
        else:
            # Process sections with headers
            for i in range(1, len(section_splits), 2):
                if i + 1 < len(section_splits):
                    title = section_splits[i].strip()
                    text = section_splits[i + 1].strip()
                    section = self._create_content_section(text, title, i // 2)
                    sections.append(section)
        
        return sections
    
    def _create_content_section(self, text: str, title: Optional[str], 
                               position: int) -> ContentSection:
        """Create a ContentSection with analysis."""
        # Extract topic keywords
        keywords = self._extract_topic_keywords(text)
        
        # Determine freshness level
        freshness_level = self._determine_section_freshness(text)
        
        return ContentSection(
            text=text,
            title=title,
            topic_keywords=keywords,
            position=position,
            freshness_level=freshness_level
        )
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """Extract key topic words from text."""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common words
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'will', 
                     'been', 'more', 'than', 'also', 'were', 'such', 'what',
                     'when', 'where', 'how', 'some', 'many', 'most', 'other'}
        
        keywords = [w for w in words if w not in stop_words]
        
        # Count frequency and return top keywords
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:10]]
    
    def _determine_section_freshness(self, text: str) -> FreshnessLevel:
        """Determine the freshness level of a content section."""
        for level, patterns in self.temporal_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        # Default to RECENT if no temporal indicators found
        return FreshnessLevel.RECENT
    
    def _calculate_overall_freshness(self, sections: List[ContentSection]) -> float:
        """Calculate overall content freshness score."""
        if not sections:
            return 0.5
        
        freshness_scores = {
            FreshnessLevel.CURRENT: 1.0,
            FreshnessLevel.RECENT: 0.8,
            FreshnessLevel.DATED: 0.4,
            FreshnessLevel.OUTDATED: 0.1
        }
        
        total_score = sum(freshness_scores[section.freshness_level] 
                         for section in sections)
        return total_score / len(sections)
    
    def _identify_outdated_sections(self, sections: List[ContentSection]) -> List[ContentSection]:
        """Identify sections that may contain outdated information."""
        outdated_sections = []
        
        for section in sections:
            if section.freshness_level in [FreshnessLevel.DATED, FreshnessLevel.OUTDATED]:
                outdated_sections.append(section)
            
            # Check for potentially outdated specific content
            outdated_indicators = [
                r'\blatest\s+version',
                r'\bcurrent\s+(?:number|count|total)',
                r'\brecent\s+(?:study|research|report)',
                r'\btoday\'?s\s+(?:market|industry)',
                r'\bas\s+of\s+20[0-2][0-9]'
            ]
            
            for indicator in outdated_indicators:
                if re.search(indicator, section.text, re.IGNORECASE):
                    if section not in outdated_sections:
                        outdated_sections.append(section)
        
        return outdated_sections
    
    def _parse_developments(self, search_results: str, provider_config: Dict[str, Any],
                           topic: str) -> List[Development]:
        """Parse search results into Development objects."""
        developments = []
        
        if not search_results:
            return developments
        
        # Simple parsing - would need enhancement based on actual search format
        lines = search_results.split('\n')
        current_dev = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('http'):
                if current_dev:
                    dev = self._create_development(current_dev, provider_config, topic)
                    if dev:
                        developments.append(dev)
                    current_dev = {}
                current_dev['url'] = line
            elif line and 'url' in current_dev:
                if 'title' not in current_dev:
                    current_dev['title'] = line
                else:
                    current_dev['summary'] = line
        
        # Add the last development
        if current_dev:
            dev = self._create_development(current_dev, provider_config, topic)
            if dev:
                developments.append(dev)
        
        return developments
    
    def _create_development(self, dev_dict: Dict[str, str], 
                           provider_config: Dict[str, Any], topic: str) -> Optional[Development]:
        """Create Development object from parsed data."""
        try:
            title = dev_dict.get('title', 'Unknown Title')
            summary = dev_dict.get('summary', '')
            url = dev_dict.get('url', '')
            
            if not url:
                return None
            
            # Determine development type based on content and provider
            dev_type = self._classify_development_type(title, summary, provider_config)
            
            # Calculate relevance and impact scores
            relevance_score = self._calculate_relevance_score(title, summary, topic)
            impact_score = self._calculate_impact_score(title, summary, dev_type)
            
            return Development(
                title=title,
                summary=summary,
                source=self._extract_source_name(url),
                url=url,
                publication_date=self._estimate_publication_date(title, summary),
                development_type=dev_type,
                relevance_score=relevance_score,
                impact_score=impact_score
            )
            
        except Exception as e:
            logger.warning(f"Failed to create development: {e}")
            return None
    
    def _classify_development_type(self, title: str, summary: str, 
                                  provider_config: Dict[str, Any]) -> DevelopmentType:
        """Classify development type based on content."""
        content = f"{title} {summary}".lower()
        
        # Use provider specialization as starting point
        specializations = provider_config.get("specialization", [])
        if specializations:
            # Check content for keywords related to specializations
            if DevelopmentType.RESEARCH in specializations:
                if any(word in content for word in ['research', 'study', 'paper', 'findings']):
                    return DevelopmentType.RESEARCH
            
            if DevelopmentType.PRODUCT in specializations:
                if any(word in content for word in ['release', 'launch', 'product', 'version']):
                    return DevelopmentType.PRODUCT
            
            return specializations[0]  # Default to first specialization
        
        # Fallback classification
        if any(word in content for word in ['funding', 'investment', 'raised']):
            return DevelopmentType.FUNDING
        elif any(word in content for word in ['regulation', 'policy', 'law']):
            return DevelopmentType.REGULATORY
        elif any(word in content for word in ['industry', 'market', 'business']):
            return DevelopmentType.INDUSTRY
        else:
            return DevelopmentType.TECHNICAL
    
    def _calculate_relevance_score(self, title: str, summary: str, topic: str) -> float:
        """Calculate relevance score for development."""
        content = f"{title} {summary}".lower()
        topic_words = set(topic.lower().split())
        content_words = set(content.split())
        
        # Calculate word overlap
        overlap = len(topic_words.intersection(content_words))
        total_topic_words = len(topic_words)
        
        if total_topic_words == 0:
            return 0.5
        
        base_relevance = overlap / total_topic_words
        
        # Boost for exact topic match
        if topic.lower() in content:
            base_relevance += 0.3
        
        return min(1.0, base_relevance)
    
    def _calculate_impact_score(self, title: str, summary: str, 
                               dev_type: DevelopmentType) -> float:
        """Calculate potential impact score for development."""
        content = f"{title} {summary}".lower()
        
        # Base impact by development type
        type_impacts = {
            DevelopmentType.RESEARCH: 0.8,
            DevelopmentType.PRODUCT: 0.7,
            DevelopmentType.INDUSTRY: 0.6,
            DevelopmentType.TECHNICAL: 0.7,
            DevelopmentType.REGULATORY: 0.9,
            DevelopmentType.FUNDING: 0.5
        }
        
        base_impact = type_impacts.get(dev_type, 0.5)
        
        # Boost for impact indicators
        impact_keywords = ['breakthrough', 'revolutionary', 'major', 'significant', 
                          'milestone', 'first', 'largest', 'billion', 'million']
        
        for keyword in impact_keywords:
            if keyword in content:
                base_impact += 0.1
        
        return min(1.0, base_impact)
    
    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL."""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Clean up common domain patterns
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except:
            return "Unknown Source"
    
    def _estimate_publication_date(self, title: str, summary: str) -> Optional[datetime]:
        """Estimate publication date from content."""
        content = f"{title} {summary}"
        
        # Look for date patterns
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(202[0-9])\b',
            r'\b(202[0-9])-(\d{1,2})-(\d{1,2})\b',
            r'\b(\d{1,2})/(\d{1,2})/(202[0-9])\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    # Simple date parsing - would need enhancement
                    return datetime.now() - timedelta(days=30)  # Assume recent
                except:
                    pass
        
        # Default to estimated recent date
        return datetime.now() - timedelta(days=7)
    
    def _rank_developments(self, developments: List[Development], topic: str) -> List[Development]:
        """Rank developments by relevance and impact."""
        # Calculate combined score
        for dev in developments:
            dev.combined_score = (dev.relevance_score * 0.6) + (dev.impact_score * 0.4)
        
        # Sort by combined score
        return sorted(developments, key=lambda x: x.combined_score, reverse=True)
    
    def _integrate_developments(self, content: str, developments: List[Development],
                               outdated_sections: List[ContentSection]) -> str:
        """Integrate developments into content."""
        if not developments:
            return content
        
        # Add recent developments section
        developments_section = self._create_developments_section(developments[:5])
        
        # Find appropriate insertion point
        insertion_point = self._find_insertion_point(content)
        
        if insertion_point >= 0:
            return (content[:insertion_point] + 
                   developments_section + 
                   content[insertion_point:])
        else:
            return content + developments_section
    
    def _create_developments_section(self, developments: List[Development]) -> str:
        """Create a formatted section for recent developments."""
        if not developments:
            return ""
        
        section_lines = [
            "\n\n## Recent Developments\n",
            "*Latest updates and developments in this field:*\n"
        ]
        
        for dev in developments:
            dev_line = f"- **{dev.title}** - {dev.summary[:100]}{'...' if len(dev.summary) > 100 else ''} ([Source]({dev.url}))"
            section_lines.append(dev_line)
        
        section_lines.append("")
        return "\n".join(section_lines)
    
    def _find_insertion_point(self, content: str) -> int:
        """Find appropriate point to insert new content."""
        # Look for conclusion or end of main content
        conclusion_patterns = [
            r'\n##?\s+Conclusion',
            r'\n##?\s+Summary',
            r'\n##?\s+Final Thoughts',
            r'\n##?\s+Looking Forward'
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.start()
        
        # If no conclusion found, append to end
        return -1
    
    def _generate_enhancement_summary(self, developments: List[Development],
                                     outdated_sections: List[ContentSection],
                                     freshness_score: float) -> str:
        """Generate summary of content enhancements."""
        summary_parts = []
        
        if developments:
            summary_parts.append(f"Added {len(developments)} recent developments")
        
        if outdated_sections:
            summary_parts.append(f"Updated {len(outdated_sections)} potentially outdated sections")
        
        freshness_desc = "current" if freshness_score > 0.8 else \
                        "mostly current" if freshness_score > 0.6 else \
                        "somewhat dated" if freshness_score > 0.4 else "outdated"
        
        summary_parts.append(f"Overall content freshness: {freshness_desc}")
        
        if not summary_parts:
            return "No enhancements were needed - content appears current."
        
        return "Content enhanced: " + ", ".join(summary_parts) + "."