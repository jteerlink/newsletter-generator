"""
Phase 1: Daily Quick Pipeline Implementation

Implements 5 specialized agents for rapid newsletter generation:
1. NewsAggregatorAgent - Collect and filter from 40+ sources
2. ContentCuratorAgent - Score and select content for 5-minute reads 
3. QuickBitesAgent - Format content following newsletter style examples
4. SubjectLineAgent - Generate compelling subject lines <50 characters
5. NewsletterAssemblerAgent - Mobile-first final assembly

Based on hybrid_newsletter_system_plan.md requirements.
"""

from __future__ import annotations
import logging
import yaml
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from src.core.core import query_llm
from src.tools.tools import search_web, search_knowledge_base
from src.scrapers.crawl4ai_web_scraper import Crawl4AIScraper
from src.scrapers.rss_extractor import RSSExtractor

logger = logging.getLogger(__name__)

@dataclass
class ContentItem:
    """Individual content item from sources"""
    title: str
    url: str
    content: str
    source: str
    category: str
    timestamp: datetime
    technical_relevance_score: float = 0.0
    practical_applicability_score: float = 0.0
    innovation_significance_score: float = 0.0

@dataclass 
class CuratedContent:
    """Curated content selected for newsletter"""
    news_breakthroughs: List[ContentItem]
    tools_tutorials: List[ContentItem] 
    quick_hits: List[ContentItem]
    estimated_read_time: int

class TechnicalRelevanceScorer:
    """Scores content for AI/ML professional relevance"""
    
    def score_technical_relevance(self, content_item: ContentItem) -> float:
        """Score content for technical professional relevance (0.0-1.0)"""
        scoring_prompt = f"""
        You are evaluating content for AI/ML technical professionals. Rate this content's technical relevance on a scale of 0.0 to 1.0.

        SCORING CRITERIA:
        - Practical applicability for AI/ML engineers (0.3 weight)
        - Technical accuracy and depth (0.25 weight)  
        - Innovation significance (0.2 weight)
        - Implementation value (0.15 weight)
        - Career relevance (0.1 weight)

        CONTENT TO EVALUATE:
        Title: {content_item.title}
        Source: {content_item.source}
        Content preview: {content_item.content[:500]}...

        Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85)
        """
        
        try:
            response = query_llm(scoring_prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default neutral score

class NewsAggregatorAgent:
    """Automated daily news collection from 40+ curated sources"""
    
    def __init__(self):
        self.sources_config = self._load_sources_config()
        self.crawl4ai_scraper = Crawl4AIScraper()
        self.rss_extractor = RSSExtractor()
        self.relevance_scorer = TechnicalRelevanceScorer()
    
    def _load_sources_config(self) -> Dict:
        """Load sources configuration from sources.yaml"""
        sources_path = Path("src/sources.yaml")
        with open(sources_path, 'r') as f:
            return yaml.safe_load(f)
    
    def aggregate_daily_news(self) -> List[ContentItem]:
        """Main news aggregation workflow"""
        logger.info("Starting daily news aggregation from 40+ sources")
        
        # 1. Fetch from all active sources
        raw_articles = self._fetch_from_all_sources()
        
        # 2. Filter for technical relevance
        filtered_articles = self._filter_for_technical_audience(raw_articles)
        
        # 3. Categorize by content pillars
        categorized_articles = self._categorize_by_pillars(filtered_articles)
        
        # 4. Identify trending topics
        trending_topics = self._identify_trending_topics(filtered_articles)
        
        logger.info(f"Aggregated {len(categorized_articles)} relevant articles")
        return categorized_articles
    
    def _fetch_from_all_sources(self) -> List[ContentItem]:
        """Fetch content from all active sources"""
        articles = []
        
        for source in self.sources_config['sources']:
            if not source.get('active', True):
                continue
                
            try:
                if source['type'] == 'rss':
                    source_articles = self._fetch_rss_content(source)
                else:
                    source_articles = self._fetch_web_content(source)
                    
                articles.extend(source_articles)
                
            except Exception as e:
                logger.error(f"Error fetching from {source['name']}: {e}")
                
        return articles
    
    def _fetch_rss_content(self, source: Dict) -> List[ContentItem]:
        """Fetch content from RSS sources"""
        try:
            # Use existing RSS extractor
            rss_data = self.rss_extractor.extract_feed(source['rss_url'])
            
            articles = []
            for item in rss_data.get('items', [])[:5]:  # Limit to 5 per source
                articles.append(ContentItem(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    content=item.get('description', ''),
                    source=source['name'],
                    category=source['category'],
                    timestamp=datetime.now()
                ))
            return articles
            
        except Exception as e:
            logger.error(f"RSS fetch error for {source['name']}: {e}")
            return []
    
    def _fetch_web_content(self, source: Dict) -> List[ContentItem]:
        """Fetch content from web sources using crawl4ai"""
        try:
            # Use existing crawl4ai scraper  
            scrape_result = self.crawl4ai_scraper.scrape_url(source['url'])
            
            if scrape_result and scrape_result.get('articles'):
                articles = []
                for article in scrape_result['articles'][:3]:  # Limit to 3 per source
                    articles.append(ContentItem(
                        title=article.get('title', ''),
                        url=article.get('url', source['url']),
                        content=article.get('content', ''),
                        source=source['name'],
                        category=source['category'],
                        timestamp=datetime.now()
                    ))
                return articles
            return []
            
        except Exception as e:
            logger.error(f"Web scrape error for {source['name']}: {e}")
            return []
    
    def _filter_for_technical_audience(self, articles: List[ContentItem]) -> List[ContentItem]:
        """Filter articles for technical professional relevance"""
        filtered = []
        
        for article in articles:
            # Score technical relevance
            relevance_score = self.relevance_scorer.score_technical_relevance(article)
            article.technical_relevance_score = relevance_score
            
            # Only keep articles above threshold
            if relevance_score >= 0.6:  # Technical threshold
                filtered.append(article)
        
        # Sort by relevance score
        filtered.sort(key=lambda x: x.technical_relevance_score, reverse=True)
        return filtered
    
    def _categorize_by_pillars(self, articles: List[ContentItem]) -> List[ContentItem]:
        """Categorize articles by content pillars"""
        categorization_prompt = """
        Categorize this article into one of these content pillars for AI/ML professionals:

        1. news_breakthroughs - Latest industry news, research findings, product announcements
        2. tools_tutorials - AI/ML tools, platforms, tutorials, how-to guides  
        3. quick_hits - Brief updates, smaller announcements, secondary news

        Article: {title}
        Content: {content_preview}

        Respond with ONLY the pillar name: news_breakthroughs, tools_tutorials, or quick_hits
        """
        
        for article in articles:
            try:
                prompt = categorization_prompt.format(
                    title=article.title,
                    content_preview=article.content[:300]
                )
                
                category = query_llm(prompt).strip().lower()
                if category in ['news_breakthroughs', 'tools_tutorials', 'quick_hits']:
                    article.category = category
                else:
                    article.category = 'quick_hits'  # Default
                    
            except Exception as e:
                logger.error(f"Categorization error: {e}")
                article.category = 'quick_hits'
        
        return articles
    
    def _identify_trending_topics(self, articles: List[ContentItem]) -> List[str]:
        """Identify trending topics from articles"""
        # Simple trending topic extraction
        trending_prompt = f"""
        Identify the top 3 trending topics from these AI/ML articles:

        Articles: {[article.title for article in articles[:10]]}

        Respond with 3 trending topics, one per line:
        """
        
        try:
            response = query_llm(trending_prompt)
            topics = [line.strip() for line in response.split('\n') if line.strip()]
            return topics[:3]
        except:
            return ["AI Research", "ML Tools", "Tech Announcements"]

class ContentCuratorAgent:
    """Intelligent content selection for 5-minute read target"""
    
    def __init__(self):
        self.priority_scorer = TechnicalRelevanceScorer()
    
    def curate_for_quick_consumption(self, aggregated_content: List[ContentItem]) -> CuratedContent:
        """Curate content for 5-minute read target"""
        logger.info("Curating content for quick consumption")
        
        # Separate by category
        news_items = [item for item in aggregated_content if item.category == 'news_breakthroughs']
        tools_items = [item for item in aggregated_content if item.category == 'tools_tutorials']
        quick_items = [item for item in aggregated_content if item.category == 'quick_hits']
        
        # Select top content for each pillar based on plan requirements
        curated = CuratedContent(
            news_breakthroughs=news_items[:3],  # Top 3 news items
            tools_tutorials=tools_items[:2],    # Top 2 tool features
            quick_hits=quick_items[:12],        # 8-12 quick hits
            estimated_read_time=5  # Target 5 minutes
        )
        
        # Validate read time target
        self._validate_read_time_target(curated)
        
        return curated
    
    def _validate_read_time_target(self, curated: CuratedContent) -> bool:
        """Ensure content meets 5-minute read time target"""
        # Estimate read time (average 200 words per minute)
        total_words = 0
        
        for item in curated.news_breakthroughs + curated.tools_tutorials:
            total_words += len(item.content.split())
        
        # Quick hits are shorter
        total_words += len(curated.quick_hits) * 20  # ~20 words per quick hit
        
        estimated_minutes = total_words / 200
        curated.estimated_read_time = int(estimated_minutes)
        
        return estimated_minutes <= 6  # Allow slight buffer

class QuickBitesAgent:
    """Generate scannable, digestible content with visual appeal"""
    
    def __init__(self):
        self.style_examples = self._load_newsletter_examples()
    
    def _load_newsletter_examples(self) -> Dict[str, str]:
        """Load reference examples for consistent style and tone"""
        try:
            with open("News and Tools Example.md", 'r') as f:
                daily_example = f.read()
            
            with open("Deep Dive & Analysis Example.md", 'r') as f:
                deep_dive_example = f.read()
                
            return {
                'daily_format_example': daily_example,
                'deep_dive_format_example': deep_dive_example
            }
        except:
            return {'daily_format_example': '', 'deep_dive_format_example': ''}
    
    def generate_news_breakthroughs(self, news_content: List[ContentItem]) -> List[str]:
        """Create News & Breakthroughs section following exact style"""
        
        news_prompt_template = """
        You are generating content for "The AI Engineer's Daily Byte" newsletter's "⚡ News & Breakthroughs" section.
        
        EXACT FORMAT TO FOLLOW for each news item:
        1. Headline with emoji: "[Descriptive Headline] **[Relevant Emoji]**"
        2. Secondary point: "**PLUS:** [Additional insight]" OR "**ALSO:** [Related development]"
        3. Technical insight: "**Technical Takeaway:** [Technical explanation for practitioners]"
        4. Deep explanation: "**Deep Dive:** [Comprehensive technical analysis with implications]"
        
        TONE & STYLE REQUIREMENTS:
        - Write for technical professionals (AI/ML engineers, data scientists, developers)
        - Balance technical accuracy with accessibility
        - Include specific technical details (architectures, algorithms, performance metrics)
        - Explain practical implications and industry impact
        - Use confident, knowledgeable tone without being overly academic
        
        CONTENT REQUIREMENTS:
        - 150-250 words per news item
        - Focus on innovation significance and practical applications
        - Include industry context and competitive landscape
        
        Generate content for this news story:
        Title: {title}
        Content: {content}
        Source: {source}
        """
        
        formatted_news = []
        for story in news_content:
            prompt = news_prompt_template.format(
                title=story.title,
                content=story.content[:1000],  # Limit content length
                source=story.source
            )
            
            try:
                formatted_content = query_llm(prompt)
                formatted_news.append(formatted_content)
            except Exception as e:
                logger.error(f"Error formatting news item: {e}")
                formatted_news.append(f"### {story.title}\n\n{story.content[:200]}...")
        
        return formatted_news
    
    def generate_tools_tutorials(self, tools_content: List[ContentItem]) -> List[str]:
        """Create Tools & Tutorials section following exact style"""
        
        tools_prompt_template = """
        You are generating content for "The AI Engineer's Daily Byte" newsletter's "🛠️ Tools & Tutorials" section.
        
        EXACT FORMAT TO FOLLOW for each tool/tutorial:
        1. Headline with emoji: "[Tool/Tutorial Name]: [Key Benefit] **[Relevant Emoji]**"
        2. Tutorial label: "**TUTORIAL:** [Brief description of what readers will learn]"
        3. Relevance: "**Why it Matters for You:** [Practical importance for technical professionals]"
        4. Step-by-step guide: "**Quick Start & [Specific Topic]:**"
           - Numbered steps with specific commands
           - Code snippets with proper formatting
           - Installation instructions
        5. Advanced tip: "**Pro Tip:** [Advanced insight or best practice]"
        
        TONE & STYLE REQUIREMENTS:
        - Hands-on, practical approach
        - Include actual code snippets and commands
        - Explain both what to do and why it works
        - Include version numbers and specific dependencies
        
        TECHNICAL REQUIREMENTS:
        - All code must be syntactically correct and runnable
        - Include import statements and setup requirements
        - Provide realistic, practical examples
        
        Generate content for this tool/tutorial:
        Title: {title}
        Content: {content}
        Source: {source}
        """
        
        formatted_tools = []
        for tool in tools_content:
            prompt = tools_prompt_template.format(
                title=tool.title,
                content=tool.content[:1000],
                source=tool.source
            )
            
            try:
                formatted_content = query_llm(prompt)
                formatted_tools.append(formatted_content)
            except Exception as e:
                logger.error(f"Error formatting tool item: {e}")
                formatted_tools.append(f"### {tool.title}\n\n{tool.content[:200]}...")
        
        return formatted_tools
    
    def generate_quick_hits(self, secondary_content: List[ContentItem]) -> List[str]:
        """Create Quick Hits bullet section"""
        quick_hits = []
        
        for item in secondary_content[:12]:  # Max 12 items
            # Extract company/action format
            hit_prompt = f"""
            Create a one-line quick hit in this format:
            "**[Company]** [action/announcement] - [brief description]"
            
            Example: "**OpenAI** announces GPT-5 preview - Enhanced reasoning capabilities for complex problem-solving"
            
            Source content:
            Title: {item.title}
            Content: {item.content[:200]}
            
            Respond with ONLY the formatted quick hit line.
            """
            
            try:
                formatted_hit = query_llm(hit_prompt).strip()
                quick_hits.append(formatted_hit)
            except:
                # Fallback format
                company = item.source.split()[0] if item.source else "Tech"
                quick_hits.append(f"**{company}** {item.title[:50]}...")
        
        return quick_hits

class SubjectLineAgent:
    """Email optimization for maximum open rates"""
    
    def generate_compelling_subject_line(self, newsletter_content: CuratedContent) -> Dict[str, Any]:
        """Create irresistible subject lines under 50 characters"""
        
        if not newsletter_content.news_breakthroughs:
            return {
                'subject_line': "AI Updates & Tools 🚀",
                'preview_text': "Latest developments in AI/ML",
                'character_count': 19
            }
        
        top_story = newsletter_content.news_breakthroughs[0]
        
        subject_variants = [
            self._create_urgency_subject(top_story),
            self._create_curiosity_subject(top_story),
            self._create_value_subject(top_story),
            self._create_number_subject(newsletter_content)
        ]
        
        # Select shortest variant under 50 characters
        valid_subjects = [s for s in subject_variants if len(s) <= 50]
        best_subject = valid_subjects[0] if valid_subjects else "AI Engineer's Update 🤖"
        
        return {
            'subject_line': best_subject,
            'preview_text': self._generate_preview_text(newsletter_content),
            'character_count': len(best_subject)
        }
    
    def _create_urgency_subject(self, story: ContentItem) -> str:
        """Create urgency-based subject line"""
        keywords = story.title.split()[:3]
        return f"Breaking: {' '.join(keywords)} 🚨"
    
    def _create_curiosity_subject(self, story: ContentItem) -> str:
        """Create curiosity-based subject line"""
        return f"This AI breakthrough changes everything 🤯"
    
    def _create_value_subject(self, story: ContentItem) -> str:
        """Create value-based subject line"""
        return f"5 AI tools that boost productivity 10x ⚡"
    
    def _create_number_subject(self, content: CuratedContent) -> str:
        """Create number-based subject line"""
        total_items = len(content.news_breakthroughs) + len(content.tools_tutorials)
        return f"{total_items} major AI updates today 📊"
    
    def _generate_preview_text(self, content: CuratedContent) -> str:
        """Generate compelling preview text (80 characters max)"""
        if content.news_breakthroughs:
            preview = content.news_breakthroughs[0].title[:75] + "..."
            return preview[:80]
        return "Latest AI developments and tools for technical professionals"

class NewsletterAssemblerAgent:
    """Final newsletter assembly with mobile-first design"""
    
    def assemble_daily_newsletter(self, subject_line: Dict, news: List[str], 
                                tools: List[str], quick_hits: List[str]) -> Dict[str, str]:
        """Assemble complete daily newsletter"""
        
        # Generate header and intro
        header = self._generate_engaging_header()
        intro = self._generate_lead_intro(news[0] if news else "")
        toc = self._generate_scannable_toc(news, tools)
        footer = self._generate_engagement_footer()
        
        # Assemble newsletter structure
        newsletter_content = f"""# **The AI Engineer's Daily Byte**

**Issue #{datetime.now().strftime('%Y-%m-%d')}**

{intro}

{toc}

## **⚡ News & Breakthroughs**

{chr(10).join(news)}

## **🛠️ Tools & Tutorials**

{chr(10).join(tools)}

## **⚡ Quick Hits**

{chr(10).join(f'• {hit}' for hit in quick_hits)}

{footer}
"""
        
        # Format for multiple channels
        formatted_outputs = {
            'markdown': newsletter_content,
            'html': self._convert_to_html(newsletter_content),
            'notion': self._format_for_notion(newsletter_content),
            'subject_line': subject_line['subject_line'],
            'preview_text': subject_line['preview_text']
        }
        
        return formatted_outputs
    
    def _generate_engaging_header(self) -> str:
        """Generate newsletter header"""
        return f"**Daily Tech & AI Update - {datetime.now().strftime('%B %d, %Y')}**"
    
    def _generate_lead_intro(self, top_story: str) -> str:
        """Generate lead introduction"""
        intro_prompt = f"""
        Create a 2-3 sentence engaging introduction for today's AI newsletter.
        
        Top story preview: {top_story[:200]}
        
        Keep it conversational and highlight the value for technical professionals.
        """
        
        try:
            return query_llm(intro_prompt)
        except:
            return "Welcome to today's essential AI updates for technical professionals. Here's what's moving the industry forward."
    
    def _generate_scannable_toc(self, news: List[str], tools: List[str]) -> str:
        """Generate table of contents"""
        return f"""📋 **Today's Update:**
• {len(news)} News & Breakthroughs
• {len(tools)} Tools & Tutorials  
• Quick Hits from across the AI landscape"""
    
    def _generate_engagement_footer(self) -> str:
        """Generate engagement footer"""
        return """---

**What did you think of today's newsletter?** Reply with your thoughts!

**Forward to a colleague** who'd benefit from staying current with AI developments.

*Stay tuned for tomorrow's edition with more AI insights and tools!*"""
    
    def _convert_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (basic implementation)"""
        # Basic markdown to HTML conversion
        html = markdown
        html = html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        html = html.replace('## ', '<h2>').replace('\n', '</h2>\n')
        html = html.replace('### ', '<h3>').replace('\n', '</h3>\n')
        html = html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        html = html.replace('• ', '<li>').replace('\n', '</li>\n')
        return f"<html><body>{html}</body></html>"
    
    def _format_for_notion(self, content: str) -> str:
        """Format content for Notion publishing"""
        # Notion-specific formatting adjustments
        notion_content = content.replace('**', '')  # Remove bold markers
        notion_content = notion_content.replace('# ', '## ')  # Adjust headers
        return notion_content

# Main Daily Pipeline Orchestrator
class DailyQuickPipeline:
    """Orchestrates the complete daily newsletter generation pipeline"""
    
    def __init__(self):
        self.news_aggregator = NewsAggregatorAgent()
        self.content_curator = ContentCuratorAgent()
        self.quick_bites = QuickBitesAgent()
        self.subject_line = SubjectLineAgent()
        self.assembler = NewsletterAssemblerAgent()
    
    def generate_daily_newsletter(self) -> Dict[str, Any]:
        """Generate complete daily newsletter following 5-minute read target"""
        logger.info("Starting daily newsletter generation pipeline")
        
        try:
            # Step 1: Aggregate news from all sources
            aggregated_content = self.news_aggregator.aggregate_daily_news()
            
            # Step 2: Curate content for quick consumption
            curated_content = self.content_curator.curate_for_quick_consumption(aggregated_content)
            
            # Step 3: Format content sections
            formatted_news = self.quick_bites.generate_news_breakthroughs(curated_content.news_breakthroughs)
            formatted_tools = self.quick_bites.generate_tools_tutorials(curated_content.tools_tutorials)
            formatted_quick_hits = self.quick_bites.generate_quick_hits(curated_content.quick_hits)
            
            # Step 4: Generate subject line
            subject_line_data = self.subject_line.generate_compelling_subject_line(curated_content)
            
            # Step 5: Assemble final newsletter
            newsletter = self.assembler.assemble_daily_newsletter(
                subject_line_data, formatted_news, formatted_tools, formatted_quick_hits
            )
            
            # Add metadata
            newsletter['metadata'] = {
                'generation_time': datetime.now().isoformat(),
                'estimated_read_time': curated_content.estimated_read_time,
                'content_sources': len(aggregated_content),
                'pipeline_version': 'Phase1_Daily_Quick'
            }
            
            logger.info("Daily newsletter generation completed successfully")
            return newsletter
            
        except Exception as e:
            logger.error(f"Daily newsletter generation failed: {e}")
            return {
                'error': f"Newsletter generation failed: {str(e)}",
                'markdown': "# Newsletter Generation Error\n\nPlease check logs for details.",
                'metadata': {'generation_time': datetime.now().isoformat()}
            }

if __name__ == "__main__":
    # Test the daily pipeline
    pipeline = DailyQuickPipeline()
    newsletter = pipeline.generate_daily_newsletter()
    
    if 'error' not in newsletter:
        print("✅ Daily newsletter generated successfully!")
        print(f"📊 Read time: {newsletter['metadata']['estimated_read_time']} minutes")
        print(f"📰 Subject: {newsletter['subject_line']}")
        print(f"📝 Content length: {len(newsletter['markdown'])} characters")
    else:
        print(f"❌ Error: {newsletter['error']}") 