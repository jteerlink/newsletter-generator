"""
Tool-Augmented Newsletter Generation Module

This module replaces the basic LLM generation with intelligent tool-augmented
newsletter creation using the full suite of enhancement components with
comprehensive monitoring and template compliance validation.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import the monitoring decorator
from core.generation_monitor import monitor_generation_timeout

logger = logging.getLogger(__name__)


@monitor_generation_timeout(timeout=600)  # 10 minute timeout for full generation
def execute_tool_augmented_generation(topic: str, audience: str, 
                                    tool_usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Execute newsletter generation with full tool usage enhancement and monitoring."""
    from core.claim_validator import ClaimExtractor, SourceValidator, CitationGenerator
    from core.information_enricher import InformationEnricher
    from core.section_aware_refinement import ToolAugmentedRefinementLoop, SectionType
    from core.advanced_quality_gates import AdvancedQualityGate
    from core.tool_analytics import ToolEffectivenessAnalyzer, ToolType
    from core.tool_cache import get_tool_cache
    from storage import get_storage_provider
    from tools.enhanced_search import MultiProviderSearchEngine
    from core.source_ranker import SourceAuthorityRanker
    from core.core import query_llm
    from core.generation_monitor import get_generation_monitor, GenerationCheckpoint, GenerationStatus
    from core.template_compliance import validate_newsletter_compliance, ComplianceLevel
    
    start_time = time.time()
    session_id = f"session_{int(time.time())}"
    workflow_id = f"workflow_{topic.replace(' ', '_').lower()}_{int(time.time())}"
    
    # Initialize generation monitoring
    monitor = get_generation_monitor()
    
    # Create checkpoints for each phase (updated for balanced targets)
    checkpoints = [
        GenerationCheckpoint("Research Phase", 200),
        GenerationCheckpoint("Initial Content Generation", 2000),  # Main content generation
        GenerationCheckpoint("Tool-Augmented Refinement", 500),   # Refinement additions
        GenerationCheckpoint("Claim Validation", 300),           # Validation improvements
        GenerationCheckpoint("Quality Assessment", 200),         # Quality metadata
        GenerationCheckpoint("Final Analytics", 100)             # Analytics footer
    ]
    
    generation_metadata = monitor.create_generation_metadata(
        session_id, workflow_id, topic, audience, checkpoints
    )
    
    logger.info("Initializing tool-augmented generation pipeline with monitoring")
    
    # Initialize tool components
    claim_extractor = ClaimExtractor()
    source_validator = SourceValidator()
    citation_generator = CitationGenerator()
    info_enricher = InformationEnricher()
    search_engine = MultiProviderSearchEngine()
    source_ranker = SourceAuthorityRanker()
    refinement_loop = ToolAugmentedRefinementLoop()
    quality_gate = AdvancedQualityGate(enforcement_level="enforcing")
    tool_analytics = ToolEffectivenessAnalyzer()
    tool_cache = get_tool_cache()
    
    try:
        # Phase 1: Research and Information Gathering
        logger.info("Phase 1: Research and information gathering")
        checkpoint_1 = checkpoints[0]
        checkpoint_1.mark_started()
        
        # Vector database search for context
        vector_results = []
        try:
            vector_store = get_storage_provider()
            vector_query = f"{topic} latest developments trends analysis"
            vector_results = vector_store.search(query=vector_query, top_k=10)
            tool_usage_metrics['vector_queries'] += 1
            
            # Cache vector results
            tool_cache.cache_vector_query(
                vector_query, vector_results, top_k=10,
                agent_name="NewsletterGenerator",
                session_id=session_id, workflow_id=workflow_id
            )
            
            logger.info(f"Retrieved {len(vector_results)} vector database results")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
        
        # Multi-provider web search
        search_results = []
        try:
            # Use correct provider names that match the actual search providers
            search_results = search_engine.intelligent_search(
                f"{topic} recent developments 2024 2025",
                providers=['duckduckgo', 'news', 'arxiv']
            )
            tool_usage_metrics['web_searches'] += 1
            tool_usage_metrics['search_providers'] = ['duckduckgo', 'news', 'arxiv']
            
            # Cache search results
            if search_results:
                tool_cache.cache_search_results(
                    f"{topic} recent developments", [r.content for r in search_results],
                    provider="multi", agent_name="NewsletterGenerator",
                    session_id=session_id, workflow_id=workflow_id
                )
            
            logger.info(f"Retrieved {len(search_results)} web search results")
        except Exception as e:
            logger.warning(f"Enhanced web search failed: {e}")
            
            # Fallback to basic unified search provider
            try:
                from tools.search_provider import get_unified_search_provider
                fallback_provider = get_unified_search_provider()
                fallback_results = fallback_provider.search(f"{topic} recent developments 2024 2025", max_results=5)
                
                if fallback_results:
                    # Convert to expected format
                    search_results = []
                    for result in fallback_results:
                        search_results.append(type('SearchResult', (), {
                            'title': result.title,
                            'url': result.url, 
                            'content': result.snippet,
                            'source': result.source
                        })())
                    
                    tool_usage_metrics['web_searches'] += 1
                    tool_usage_metrics['search_providers'] = ['unified_fallback']
                    logger.info(f"Fallback search retrieved {len(search_results)} results")
                
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
        
        # Complete Phase 1 checkpoint
        research_word_count = len(' '.join([str(r) for r in vector_results + search_results]).split())
        checkpoint_1.mark_completed(research_word_count, f"Vector: {len(vector_results)}, Web: {len(search_results)}")
        generation_metadata.add_checkpoint_data(checkpoint_1)
        
        # Phase 2: Content Generation with Context
        logger.info("Phase 2: Initial content generation")
        checkpoint_2 = checkpoints[1]
        checkpoint_2.mark_started()
        
        # Build enhanced context from research
        research_context = ""
        if vector_results:
            vector_context = "\n".join([f"- {getattr(r, 'content', str(r))[:200]}" for r in vector_results[:5]])
            research_context += f"\n\nVector Database Context:\n{vector_context}"
        
        if search_results:
            search_context = "\n".join([f"- {r.title}: {r.content[:200]}" for r in search_results[:5]])
            research_context += f"\n\nWeb Search Context:\n{search_context}"
        
        # Enhanced prompt with research context and article-style writing guidelines
        enhanced_prompt = f"""
        Write a comprehensive, engaging newsletter article about {topic} for {audience}.
        
        Research Context:
        {research_context}
        
        CRITICAL WRITING STYLE REQUIREMENTS:
        - Write primarily in a flowing, article-style narrative format
        - Use paragraphs and prose as the main content structure
        - LIMITED structured elements are acceptable when they enhance clarity:
          • Use bulleted lists ONLY for 3-5 key points or technical specifications
          • Include tables ONLY when comparing specific data/features (max 1-2 per section)
          • Structured lists should support, not replace, narrative explanation
        - Create a conversational yet professional tone
        - Use smooth transitions between ideas and concepts
        - Tell a story with your technical content while using strategic formatting
        - Make complex topics accessible through explanation, examples, and selective formatting
        
        Structure (write each as balanced narrative with selective formatting):
        1. **Brief Introduction** (300-400 words) - Start with an engaging hook, explain why this topic matters now, and provide context for {audience}
        
        2. **Key Developments & Technical Foundation** (500-700 words) - Discuss recent developments and core technical concepts in narrative form. Use a small bulleted list (3-5 points) only if it clarifies complex technical specifications
        
        3. **Deep Technical Analysis** (500-800 words) - Provide in-depth technical insights through explanatory prose. Include examples woven into the narrative. One comparison table is acceptable if comparing specific features or performance metrics
        
        4. **Real-World Applications & Practical Implications** (400-500 words) - Discuss practical applications and implications for {audience} primarily in article format with occasional strategic formatting
        
        5. **Future Outlook & Strategic Insights** (150-250 words) - Analyze trends and provide predictions through thoughtful narrative
        
        6. **Conclusion & Call to Action** (150-250 words) - Synthesize key points and provide actionable takeaways
        
        Requirements:
        - Target length: 2100-2900 words (comprehensive but focused article)
        - Use factual information from the research context
        - Write primarily in connected paragraphs with smooth transitions  
        - Include specific examples and data points woven into the narrative
        - Maintain technical depth while keeping it accessible and readable
        - Use bulleted lists sparingly (3-5 items max) only for clarity enhancement
        - Include max 1-2 tables per article, only for direct comparisons
        - Balance narrative flow with strategic use of formatting for readability
        - Create a professional article that combines storytelling with clear information structure
        
        Generate comprehensive, well-researched content in article format:
        """
        
        initial_content = query_llm(enhanced_prompt)
        
        if not initial_content or len(initial_content) < 100:
            checkpoint_2.mark_failed("Generated content is insufficient")
            raise Exception("Generated content is insufficient")
        
        # Complete Phase 2 checkpoint
        initial_word_count = len(initial_content.split())
        checkpoint_2.mark_completed(initial_word_count, initial_content[:200])
        generation_metadata.add_checkpoint_data(checkpoint_2)
        
        # Phase 3: Tool-Augmented Refinement
        logger.info("Phase 3: Tool-augmented refinement")
        checkpoint_3 = checkpoints[2]
        checkpoint_3.mark_started()
        
        refined_content = refinement_loop.refine_with_tools(
            initial_content, SectionType.ANALYSIS,
            workflow_id=workflow_id, session_id=session_id
        )
        
        # Complete Phase 3 checkpoint
        refined_word_count = len(refined_content.split())
        checkpoint_3.mark_completed(refined_word_count, refined_content[:200])
        generation_metadata.add_checkpoint_data(checkpoint_3)
        
        # Phase 4: Claim Validation and Enhancement
        logger.info("Phase 4: Claim validation and information enrichment")
        checkpoint_4 = checkpoints[3]
        checkpoint_4.mark_started()
        
        # Extract and validate claims
        validated_claims = []
        try:
            claims = claim_extractor.extract_claims(refined_content)
            
            for claim in claims[:10]:  # Limit to top 10 claims
                try:
                    validation_result = source_validator.validate_claim(
                        claim, workflow_id=workflow_id, session_id=session_id)
                    
                    if validation_result.validation_status == "supported":
                        validated_claims.append(validation_result)
                        
                except Exception as e:
                    logger.warning(f"Claim validation failed for: {claim.text[:50]}... - {e}")
            
            tool_usage_metrics['verified_claims'] = validated_claims
            logger.info(f"Validated {len(validated_claims)} claims")
            
        except Exception as e:
            logger.warning(f"Claim validation process failed: {e}")
        
        # Information enrichment
        try:
            enriched_content = info_enricher.enrich_content(refined_content, topic)
            if enriched_content and enriched_content.enhanced_content:
                refined_content = enriched_content.enhanced_content
                logger.info("Content successfully enriched with recent developments")
        except Exception as e:
            logger.warning(f"Information enrichment failed: {e}")
        
        # Complete Phase 4 checkpoint
        validation_word_count = len(refined_content.split())
        checkpoint_4.mark_completed(validation_word_count, f"Claims validated: {len(validated_claims)}")
        generation_metadata.add_checkpoint_data(checkpoint_4)
        
        # Phase 5: Quality Assessment and Template Compliance
        logger.info("Phase 5: Quality assessment and template compliance validation")
        checkpoint_5 = checkpoints[4]
        checkpoint_5.mark_started()
        
        # Prepare metadata for quality assessment
        quality_metadata = {
            'tool_usage': tool_usage_metrics,
            'generation_time_ms': (time.time() - start_time) * 1000,
            'target_length': 1500,
            'session_id': session_id,
            'workflow_id': workflow_id,
            'agent_type': 'NewsletterGenerator'
        }
        
        # Run template compliance validation (CRITICAL)
        template_compliance_report = validate_newsletter_compliance(
            refined_content, 
            template_type="technical_deep_dive",
            compliance_level=ComplianceLevel.STANDARD
        )
        
        logger.info(f"Template compliance: {template_compliance_report.overall_score:.2f} "
                   f"({'PASS' if template_compliance_report.is_compliant else 'FAIL'})")
        
        # If template compliance fails, this is a critical issue
        if not template_compliance_report.is_compliant:
            error_msg = f"Template compliance failed: {template_compliance_report.overall_score:.2f}"
            for issue in template_compliance_report.critical_issues:
                logger.error(f"Critical compliance issue: {issue}")
            checkpoint_5.mark_failed(error_msg)
            generation_metadata.add_checkpoint_data(checkpoint_5)
            
            # Return with compliance failure
            return {
                'success': False,
                'error': error_msg,
                'content': refined_content,
                'execution_time': time.time() - start_time,
                'tool_usage': tool_usage_metrics,
                'template_compliance': template_compliance_report.__dict__,
                'compliance_issues': template_compliance_report.critical_issues
            }
        
        # Run quality assessment
        quality_report = None
        try:
            quality_report = quality_gate.validate_with_level(
                refined_content, tool_usage_metrics, quality_metadata)
            
            logger.info(f"Quality assessment completed: {quality_report.overall_score:.2f}")
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
        
        # Complete Phase 5 checkpoint
        compliance_score = template_compliance_report.overall_score
        checkpoint_5.mark_completed(200, f"Compliance: {compliance_score:.2f}")
        generation_metadata.add_checkpoint_data(checkpoint_5)
        
        # Calculate tool integration score
        tool_integration_score = 0.0
        if tool_usage_metrics['vector_queries'] > 0:
            tool_integration_score += 0.3
        if tool_usage_metrics['web_searches'] > 0:
            tool_integration_score += 0.3
        if len(tool_usage_metrics['verified_claims']) > 0:
            tool_integration_score += 0.4
        
        tool_usage_metrics['tool_integration_score'] = tool_integration_score
        
        # Phase 6: Analytics and Performance Tracking  
        logger.info("Phase 6: Recording analytics")
        checkpoint_6 = checkpoints[5]
        checkpoint_6.mark_started()
        
        # Record tool usage metrics
        try:
            tool_analytics.record_tool_usage(
                ToolType.VECTOR_SEARCH, "NewsletterGenerator",
                execution_time_ms=100, success=tool_usage_metrics['vector_queries'] > 0,
                input_size=len(topic), output_size=len(vector_results),
                session_id=session_id, workflow_id=workflow_id
            )
            
            tool_analytics.record_tool_usage(
                ToolType.WEB_SEARCH, "NewsletterGenerator",
                execution_time_ms=500, success=tool_usage_metrics['web_searches'] > 0,
                input_size=len(topic), output_size=len(search_results),
                session_id=session_id, workflow_id=workflow_id
            )
            
            if validated_claims:
                tool_analytics.record_tool_usage(
                    ToolType.CLAIM_VALIDATION, "NewsletterGenerator",
                    execution_time_ms=1000, success=True,
                    input_size=len(claims) if 'claims' in locals() else 0, 
                    output_size=len(validated_claims),
                    session_id=session_id, workflow_id=workflow_id
                )
        
        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")
        
        # Complete Phase 6 checkpoint and finalize generation
        checkpoint_6.mark_completed(100, "Analytics recorded")
        generation_metadata.add_checkpoint_data(checkpoint_6)
        
        # Finalize generation monitoring
        monitor.finalize_generation(session_id, success=True)
        
        execution_time = time.time() - start_time
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"newsletter_{topic.replace(' ', '_').lower()}_{timestamp}.md"
        output_path = f"output/{output_filename}"
        
        # Ensure output directory exists
        import os
        os.makedirs("output", exist_ok=True)
        
        # Add tool usage footer to content
        enhanced_content = refined_content + f"""

---

*📊 Newsletter Generation Analytics:*
- *Vector Database Queries: {tool_usage_metrics['vector_queries']}*
- *Web Search Queries: {tool_usage_metrics['web_searches']}*
- *Verified Claims: {len(tool_usage_metrics['verified_claims'])}*
- *Search Providers: {', '.join(tool_usage_metrics['search_providers'])}*
- *Tool Integration Score: {tool_integration_score:.1%}*
- *Generation Time: {execution_time:.1f}s*

*🤖 Generated with Enhanced AI Newsletter System - Tool-Augmented Intelligence*
"""
        
        # Save the newsletter content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(enhanced_content)
        
        logger.info(f"Tool-augmented newsletter generation completed successfully in {execution_time:.2f} seconds")
        logger.info(f"Tool integration score: {tool_integration_score:.1%}")
        
        return {
            'success': True,
            'output_file': output_path,
            'content': enhanced_content,
            'execution_time': execution_time,
            'tool_usage': tool_usage_metrics,
            'quality_report': quality_report.__dict__ if quality_report else None,
            'template_compliance': template_compliance_report.__dict__ if template_compliance_report else None,
            'generation_metadata': generation_metadata.__dict__ if generation_metadata else None,
            'session_id': session_id,
            'workflow_id': workflow_id,
            'tool_integration_score': tool_integration_score,
            'checkpoints_completed': sum(1 for cp in checkpoints if cp.status == GenerationStatus.COMPLETED),
            'total_checkpoints': len(checkpoints),
            'generation_mode': 'tool_augmented_monitored'
        }
        
    except Exception as e:
        logger.error(f"Tool-augmented generation failed: {e}")
        
        # Finalize generation monitoring with failure
        try:
            monitor.finalize_generation(session_id, success=False)
            execution_time = time.time() - start_time
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'tool_usage': tool_usage_metrics,
                'generation_metadata': generation_metadata.__dict__ if 'generation_metadata' in locals() else None,
                'session_id': session_id,
                'workflow_id': workflow_id,
                'generation_mode': 'tool_augmented_monitored_failed'
            }
        except Exception as monitor_error:
            logger.error(f"Failed to finalize generation monitoring: {monitor_error}")
            raise e


def execute_basic_generation(topic: str, audience: str, 
                           tool_usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback to basic generation when tool components unavailable."""
    from core.core import query_llm
    import os
    
    start_time = time.time()
    
    # Create a comprehensive prompt for newsletter generation
    prompt = f"""
    Write a newsletter about {topic} for {audience}.

    Include:
    1. Brief introduction
    2. Key developments and trends
    3. Technical insights
    4. Practical implications
    5. Future outlook

    Make it informative and well-structured for technical professionals.
    Target length: 800-1500 words.
    """

    logger.info("Generating newsletter content using basic LLM generation...")
    content = query_llm(prompt)

    if not content or len(content) < 100:
        raise Exception("Generated content is insufficient")

    execution_time = time.time() - start_time

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"newsletter_{topic.replace(' ', '_').lower()}_{timestamp}.md"
    output_path = f"output/{output_filename}"

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Add basic analytics footer
    enhanced_content = content + f"""

---

*📊 Newsletter Generation Analytics:*
- *Generation Mode: Basic LLM (Tool components not available)*
- *Generation Time: {execution_time:.1f}s*

*🤖 Generated with AI Newsletter System*
"""

    # Save the newsletter content
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(enhanced_content)

    logger.info(f"Basic newsletter generation completed successfully in {execution_time:.2f} seconds")

    return {
        'success': True,
        'output_file': output_path,
        'content': enhanced_content,
        'execution_time': execution_time,
        'tool_usage': tool_usage_metrics,
        'generation_mode': 'basic'
    }