"""
Content Validator Module for Newsletter Generation System
Handles repetition detection, fact-checking, and content quality validation
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from difflib import SequenceMatcher
import hashlib

logger = logging.getLogger(__name__)

class ContentValidator:
    """Validates newsletter content for quality, accuracy, and uniqueness."""
    
    def __init__(self):
        self.suspicious_expert_patterns = [
            r"Dr\. [A-Z][a-z]+ [A-Z][a-z]+.*?(MIT|Stanford|Harvard|Berkeley|Cambridge|Oxford)",
            r"Professor [A-Z][a-z]+ [A-Z][a-z]+.*?(Co-Founder|Chief Scientist|Director)",
            r"[A-Z][a-z]+ [A-Z][a-z]+.*?(AI Researcher|Machine Learning Expert|Technology Leader)",
        ]
        
        self.generic_expert_quotes = [
            "must be designed with transparency",
            "critical aspect of developing effective",
            "most important thing about",
            "key to success is",
            "future of AI depends on",
        ]
        
        self.factual_claim_patterns = [
            r"(\d+)% of (?:companies|organizations|users|people)",
            r"studies show that",
            r"research indicates",
            r"according to (?:recent|new) (?:study|research|survey)",
            r"data reveals that",
            r"statistics show",
        ]
    
    def validate_content(self, content: str) -> Dict[str, Any]:
        """
        Comprehensive content validation.
        
        Args:
            content: The newsletter content to validate
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        results = {
            'repetition_analysis': self.detect_repetition(content),
            'expert_quote_analysis': self.analyze_expert_quotes(content),
            'fact_check_analysis': self.analyze_factual_claims(content),
            'quality_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def detect_repetition(self, content: str) -> Dict[str, Any]:
        """
        Detect repetitive content patterns.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with repetition analysis results
        """
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate sentence similarity
        repetitive_sentences = []
        sentence_groups = []
        
        for i, sentence1 in enumerate(sentences):
            similar_sentences = []
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                similarity = self._calculate_similarity(sentence1, sentence2)
                if similarity > 0.7:  # High similarity threshold
                    similar_sentences.append((j, sentence2, similarity))
            
            if similar_sentences:
                group = [i] + [idx for idx, _, _ in similar_sentences]
                sentence_groups.append(group)
        
        # Detect repeated concepts/phrases
        concept_repetition = self._detect_concept_repetition(content)
        
        # Detect section repetition
        section_repetition = self._detect_section_repetition(content)
        
        repetition_score = self._calculate_repetition_score(
            len(repetitive_sentences), 
            len(concept_repetition), 
            len(section_repetition),
            len(sentences)
        )
        
        return {
            'repetition_score': repetition_score,
            'repetitive_sentences': repetitive_sentences,
            'concept_repetition': concept_repetition,
            'section_repetition': section_repetition,
            'sentence_groups': sentence_groups,
            'total_sentences': len(sentences)
        }
    
    def analyze_expert_quotes(self, content: str) -> Dict[str, Any]:
        """
        Analyze expert quotes for authenticity and value.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with expert quote analysis
        """
        # Find potential expert quotes
        expert_quotes = []
        
        # Look for quote patterns
        quote_patterns = [
            r'"([^"]*)"[^"]*?(?:says?|said|explains?|explained|notes?|noted|comments?|commented|states?|stated)\s+([^.]*?)(?:\.|,)',
            r'([^.]*?)(?:says?|said|explains?|explained|notes?|noted|comments?|commented|states?|stated):\s*"([^"]*)"',
            r'According to ([^,]*?),\s*"([^"]*)"',
        ]
        
        for pattern in quote_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    quote_text = match.group(1) if '"' in match.group(0)[:10] else match.group(2)
                    attribution = match.group(2) if '"' in match.group(0)[:10] else match.group(1)
                    expert_quotes.append({
                        'quote': quote_text,
                        'attribution': attribution,
                        'full_match': match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Analyze quotes for suspiciousness
        suspicious_quotes = []
        for quote in expert_quotes:
            suspicion_score = self._calculate_quote_suspicion(quote)
            if suspicion_score > 0.6:
                suspicious_quotes.append({
                    **quote,
                    'suspicion_score': suspicion_score,
                    'reasons': self._get_suspicion_reasons(quote)
                })
        
        return {
            'total_quotes': len(expert_quotes),
            'suspicious_quotes': suspicious_quotes,
            'expert_quotes': expert_quotes,
            'authenticity_score': 1.0 - (len(suspicious_quotes) / max(len(expert_quotes), 1))
        }
    
    def analyze_factual_claims(self, content: str) -> Dict[str, Any]:
        """
        Analyze factual claims for verification needs.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with factual claim analysis
        """
        claims = []
        
        # Find statistical claims
        for pattern in self.factual_claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                claims.append({
                    'claim': match.group(0),
                    'context': context,
                    'type': 'statistical',
                    'position': match.start(),
                    'needs_verification': True
                })
        
        # Find unsupported claims
        unsupported_indicators = [
            r"clearly shows",
            r"obviously",
            r"it's well known that",
            r"everyone knows",
            r"studies prove",
            r"experts agree",
        ]
        
        for indicator in unsupported_indicators:
            matches = re.finditer(indicator, content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                claims.append({
                    'claim': match.group(0),
                    'context': context,
                    'type': 'unsupported',
                    'position': match.start(),
                    'needs_verification': True
                })
        
        verification_score = max(0, 1.0 - (len(claims) / max(len(content.split()), 1)) * 10)
        
        return {
            'claims': claims,
            'statistical_claims': [c for c in claims if c['type'] == 'statistical'],
            'unsupported_claims': [c for c in claims if c['type'] == 'unsupported'],
            'verification_score': verification_score,
            'needs_review': len(claims) > 5
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _detect_concept_repetition(self, content: str) -> List[Dict[str, Any]]:
        """Detect repeated concepts or key phrases."""
        # Extract key phrases (3+ words)
        phrases = re.findall(r'\b\w+\s+\w+\s+\w+\b', content.lower())
        phrase_counts = Counter(phrases)
        
        # Find phrases that appear multiple times
        repeated_concepts = []
        for phrase, count in phrase_counts.items():
            if count > 2:  # Appears more than twice
                repeated_concepts.append({
                    'phrase': phrase,
                    'count': count,
                    'type': 'concept_repetition'
                })
        
        return repeated_concepts
    
    def _detect_section_repetition(self, content: str) -> List[Dict[str, Any]]:
        """Detect repeated sections or paragraphs."""
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        repeated_sections = []
        for i, para1 in enumerate(paragraphs):
            for j, para2 in enumerate(paragraphs[i+1:], i+1):
                if len(para1) > 100 and len(para2) > 100:  # Only check substantial paragraphs
                    similarity = self._calculate_similarity(para1, para2)
                    if similarity > 0.6:
                        repeated_sections.append({
                            'section1_index': i,
                            'section2_index': j,
                            'similarity': similarity,
                            'type': 'section_repetition'
                        })
        
        return repeated_sections
    
    def _calculate_repetition_score(self, repetitive_sentences: int, concept_repetition: int, 
                                   section_repetition: int, total_sentences: int) -> float:
        """Calculate overall repetition score (0-1, where 1 is no repetition)."""
        if total_sentences == 0:
            return 1.0
        
        sentence_penalty = repetitive_sentences / total_sentences
        concept_penalty = min(concept_repetition / 10, 0.5)
        section_penalty = min(section_repetition / 5, 0.3)
        
        total_penalty = sentence_penalty + concept_penalty + section_penalty
        return max(0, 1.0 - total_penalty)
    
    def _calculate_quote_suspicion(self, quote: Dict[str, Any]) -> float:
        """Calculate suspicion score for an expert quote."""
        suspicion_score = 0
        
        # Check for suspicious patterns in attribution
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote['attribution'], re.IGNORECASE):
                suspicion_score += 0.4
        
        # Check for generic quote content
        for generic in self.generic_expert_quotes:
            if generic.lower() in quote['quote'].lower():
                suspicion_score += 0.3
        
        # Check for overly convenient attributions
        if any(word in quote['attribution'].lower() for word in ['co-founder', 'chief scientist', 'director']):
            suspicion_score += 0.2
        
        # Check for vague affiliations
        if not any(word in quote['attribution'].lower() for word in ['university', 'company', 'institute', 'lab']):
            suspicion_score += 0.1
        
        return min(suspicion_score, 1.0)
    
    def _get_suspicion_reasons(self, quote: Dict[str, Any]) -> List[str]:
        """Get reasons why a quote is considered suspicious."""
        reasons = []
        
        # Check attribution patterns
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote['attribution'], re.IGNORECASE):
                reasons.append("Suspicious expert attribution pattern")
        
        # Check generic content
        for generic in self.generic_expert_quotes:
            if generic.lower() in quote['quote'].lower():
                reasons.append("Generic or common quote content")
        
        # Check for convenient titles
        if any(word in quote['attribution'].lower() for word in ['co-founder', 'chief scientist']):
            reasons.append("Overly convenient expert title")
        
        # Check for vague affiliations
        if not any(word in quote['attribution'].lower() for word in ['university', 'company', 'institute']):
            reasons.append("Vague or missing institutional affiliation")
        
        return reasons
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall content quality score."""
        repetition_score = results['repetition_analysis']['repetition_score']
        authenticity_score = results['expert_quote_analysis']['authenticity_score']
        verification_score = results['fact_check_analysis']['verification_score']
        
        # Weighted average
        quality_score = (
            repetition_score * 0.4 +
            authenticity_score * 0.3 +
            verification_score * 0.3
        )
        
        return round(quality_score, 2)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Repetition recommendations
        rep_score = results['repetition_analysis']['repetition_score']
        if rep_score < 0.7:
            recommendations.append("Reduce repetitive content and vary language/examples")
        
        # Expert quote recommendations
        suspicious_quotes = results['expert_quote_analysis']['suspicious_quotes']
        if suspicious_quotes:
            recommendations.append(f"Remove or verify {len(suspicious_quotes)} suspicious expert quotes")
        
        # Fact checking recommendations
        claims = results['fact_check_analysis']['claims']
        if len(claims) > 5:
            recommendations.append("Verify factual claims with credible sources")
        
        # Overall quality recommendations
        if results['quality_score'] < 0.6:
            recommendations.append("Content needs significant improvement before publication")
        elif results['quality_score'] < 0.8:
            recommendations.append("Content has some issues that should be addressed")
        
        return recommendations
    
    def clean_content(self, content: str, validation_results: Dict[str, Any]) -> str:
        """
        Clean content based on validation results.
        
        Args:
            content: Original content
            validation_results: Results from validate_content()
            
        Returns:
            Cleaned content
        """
        cleaned_content = content
        
        # Remove suspicious expert quotes
        suspicious_quotes = validation_results['expert_quote_analysis']['suspicious_quotes']
        for quote in suspicious_quotes:
            # Remove the entire quote and attribution
            cleaned_content = cleaned_content.replace(quote['full_match'], '')
        
        # Add warnings for unverified claims
        claims = validation_results['fact_check_analysis']['claims']
        for claim in claims:
            if claim['needs_verification']:
                # Add a note about verification needed
                original_claim = claim['claim']
                noted_claim = f"{original_claim} [Note: This claim requires verification]"
                cleaned_content = cleaned_content.replace(original_claim, noted_claim)
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = re.sub(r' +', ' ', cleaned_content)
        
        return cleaned_content.strip() 
    
    def validate_expert_quotes(self, content: str) -> Dict[str, Any]:
        """Alias for analyze_expert_quotes for backward compatibility."""
        return self.analyze_expert_quotes(content)
    
    def basic_fact_check(self, content: str) -> Dict[str, Any]:
        """
        Basic fact-checking that returns unverifiable claims.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with unverifiable claims
        """
        analysis = self.analyze_factual_claims(content)
        return {
            'unverifiable_claims': [claim['claim'] for claim in analysis['claims']],
            'verification_score': analysis['verification_score'],
            'needs_review': analysis['needs_review']
        }
    
    def assess_content_quality(self, content: str) -> Dict[str, Any]:
        """Comprehensive content quality assessment with detailed metrics."""
        
        # Core quality metrics
        quality_metrics = {
            "technical_accuracy": self._assess_technical_accuracy(content),
            "information_density": self._assess_information_density(content),
            "readability_score": self._assess_readability(content),
            "engagement_factors": self._assess_engagement_factors(content),
            "structure_quality": self._assess_structure_quality(content),
            "ai_ml_relevance": self._assess_ai_ml_relevance(content),
            "code_quality": self._assess_code_quality(content),
            "citation_quality": self._assess_citation_quality(content),
            "practical_value": self._assess_practical_value(content),
            "innovation_factor": self._assess_innovation_factor(content)
        }
        
        # Calculate overall quality score
        weighted_score = self._calculate_weighted_quality_score(quality_metrics)
        
        # Generate quality insights
        quality_insights = self._generate_quality_insights(quality_metrics, content)
        
        return {
            "overall_score": weighted_score,
            "quality_metrics": quality_metrics,
            "quality_insights": quality_insights,
            "grade": self._assign_quality_grade(weighted_score),
            "improvement_recommendations": self._generate_improvement_recommendations(quality_metrics),
            "strengths": self._identify_strengths(quality_metrics),
            "weaknesses": self._identify_weaknesses(quality_metrics)
        }
    
    def _assess_technical_accuracy(self, content: str) -> float:
        """Assess technical accuracy and correctness."""
        score = 8.0  # Base score
        
        # Check for technical inconsistencies
        technical_terms = [
            "neural network", "machine learning", "deep learning", "algorithm",
            "model", "training", "inference", "gradient", "optimization",
            "transformer", "attention", "embedding", "layer", "activation"
        ]
        
        term_count = sum(1 for term in technical_terms if term in content.lower())
        
        # Higher technical term usage indicates technical depth
        if term_count > 20:
            score += 1.0
        elif term_count > 10:
            score += 0.5
        
        # Check for common technical errors/misconceptions
        error_patterns = [
            "100% accurate", "perfect accuracy", "never fails",
            "always works", "completely eliminates", "totally removes"
        ]
        
        for pattern in error_patterns:
            if pattern in content.lower():
                score -= 0.5
        
        return min(10.0, max(0.0, score))
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density and value per word."""
        words = content.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Count informational elements
        informational_patterns = [
            r'\d+%', r'\d+\.\d+', r'\$\d+', r'\d+x', r'\d+ times',
            r'study shows', r'research indicates', r'according to',
            r'for example', r'such as', r'including', r'specifically'
        ]
        
        info_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                        for pattern in informational_patterns)
        
        # Calculate density score
        density_ratio = info_count / (word_count / 100)  # Info elements per 100 words
        
        # Score based on density
        if density_ratio > 5:
            return 10.0
        elif density_ratio > 3:
            return 8.0
        elif density_ratio > 1:
            return 6.0
        else:
            return 4.0
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability and accessibility."""
        sentences = content.split('.')
        words = content.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Count complex words (3+ syllables, simplified)
        complex_words = sum(1 for word in words if len(word) > 8)
        complex_word_ratio = complex_words / len(words)
        
        # Readability score (simplified Flesch-Kincaid)
        score = 8.0
        
        # Adjust for sentence length
        if avg_sentence_length > 25:
            score -= 2.0
        elif avg_sentence_length > 20:
            score -= 1.0
        elif avg_sentence_length < 10:
            score -= 1.0
        
        # Adjust for complex words
        if complex_word_ratio > 0.3:
            score -= 1.0
        elif complex_word_ratio > 0.2:
            score -= 0.5
        
        return min(10.0, max(0.0, score))
    
    def _assess_engagement_factors(self, content: str) -> float:
        """Assess engagement and reader interest factors."""
        score = 5.0  # Base score
        
        # Check for engaging elements
        engaging_patterns = [
            r'imagine', r'picture this', r'consider', r'what if',
            r'surprisingly', r'remarkable', r'fascinating', r'incredible',
            r'breakthrough', r'revolutionary', r'cutting-edge', r'innovative'
        ]
        
        engagement_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                              for pattern in engaging_patterns)
        
        # Check for storytelling elements
        story_patterns = [
            r'story', r'example', r'case study', r'real-world', r'scenario',
            r'experience', r'journey', r'challenge', r'success', r'failure'
        ]
        
        story_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                         for pattern in story_patterns)
        
        # Score based on engaging elements
        total_engaging = engagement_count + story_count
        if total_engaging > 10:
            score += 3.0
        elif total_engaging > 5:
            score += 2.0
        elif total_engaging > 2:
            score += 1.0
        
        # Check for questions (engagement technique)
        question_count = content.count('?')
        if question_count > 5:
            score += 1.0
        elif question_count > 2:
            score += 0.5
        
        return min(10.0, max(0.0, score))
    
    def _assess_structure_quality(self, content: str) -> float:
        """Assess content structure and organization."""
        score = 6.0  # Base score
        
        # Check for clear sections/headers
        header_patterns = [
            r'\*\*[^*]+\*\*', r'##?\s+[^\n]+', r'#\s+[^\n]+',
            r'[A-Z][^.!?]*:', r'\d+\.\s+[A-Z]'
        ]
        
        header_count = sum(len(re.findall(pattern, content)) 
                          for pattern in header_patterns)
        
        # Score based on structure
        if header_count > 8:
            score += 2.0
        elif header_count > 4:
            score += 1.5
        elif header_count > 2:
            score += 1.0
        
        # Check for logical flow indicators
        flow_patterns = [
            r'first', r'second', r'third', r'next', r'then', r'finally',
            r'however', r'therefore', r'moreover', r'furthermore',
            r'in addition', r'on the other hand', r'as a result'
        ]
        
        flow_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                        for pattern in flow_patterns)
        
        if flow_count > 5:
            score += 1.0
        elif flow_count > 2:
            score += 0.5
        
        return min(10.0, max(0.0, score))
    
    def _assess_ai_ml_relevance(self, content: str) -> float:
        """Assess relevance to AI/ML domain."""
        ai_ml_keywords = [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "algorithm", "model", "training", "data",
            "prediction", "classification", "regression", "clustering",
            "supervised", "unsupervised", "reinforcement", "gradient",
            "optimization", "loss function", "backpropagation", "transformer",
            "attention", "embedding", "feature", "dataset", "validation",
            "accuracy", "precision", "recall", "f1-score", "bias", "variance"
        ]
        
        keyword_count = sum(1 for keyword in ai_ml_keywords 
                           if keyword in content.lower())
        
        # AI/ML frameworks and tools
        tools_keywords = [
            "pytorch", "tensorflow", "keras", "scikit-learn", "pandas",
            "numpy", "jupyter", "python", "r", "gpu", "cuda", "docker",
            "kubernetes", "aws", "azure", "gcp", "mlops", "hugging face"
        ]
        
        tools_count = sum(1 for tool in tools_keywords 
                         if tool in content.lower())
        
        total_relevance = keyword_count + tools_count
        
        if total_relevance > 20:
            return 10.0
        elif total_relevance > 15:
            return 8.5
        elif total_relevance > 10:
            return 7.0
        elif total_relevance > 5:
            return 5.5
        else:
            return 3.0
    
    def _assess_code_quality(self, content: str) -> float:
        """Assess quality of code examples and technical content."""
        
        # Check for code blocks
        code_patterns = [
            r'```[\s\S]*?```',  # Triple backtick blocks
            r'`[^`]+`',         # Inline code
            r'def\s+\w+\(',     # Function definitions
            r'import\s+\w+',    # Import statements
            r'class\s+\w+',     # Class definitions
        ]
        
        code_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                        for pattern in code_patterns)
        
        if code_count == 0:
            return 5.0  # Neutral score for no code
        
        # Check for good coding practices in examples
        good_practices = [
            r'#\s+', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''",  # Comments and docstrings
            r'if\s+__name__\s*==\s*["\']__main__["\']',      # Main guard
            r'try:', r'except:', r'finally:',                # Error handling
            r'with\s+open\(',                                # Context managers
        ]
        
        practice_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                           for pattern in good_practices)
        
        # Score based on code quality indicators
        base_score = 6.0
        
        if practice_count > 3:
            base_score += 2.0
        elif practice_count > 1:
            base_score += 1.0
        
        # Check for code explanations
        explanation_patterns = [
            r'this code', r'the function', r'the algorithm',
            r'here we', r'this line', r'the following'
        ]
        
        explanation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                               for pattern in explanation_patterns)
        
        if explanation_count > 3:
            base_score += 1.0
        
        return min(10.0, max(0.0, base_score))
    
    def _assess_citation_quality(self, content: str) -> float:
        """Assess quality of citations and references."""
        
        # Check for various citation formats
        citation_patterns = [
            r'\[\d+\]',                          # Numbered citations
            r'\([^)]*\d{4}[^)]*\)',             # Year citations
            r'according to [^.]*',               # Attribution phrases
            r'research shows', r'study finds',   # Research references
            r'https?://[^\s]+',                  # URLs
            r'doi:', r'arxiv:', r'github.com',   # Specific platforms
        ]
        
        citation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                           for pattern in citation_patterns)
        
        # Check for credible sources
        credible_sources = [
            'nature', 'science', 'ieee', 'acm', 'arxiv', 'github',
            'google', 'microsoft', 'openai', 'deepmind', 'stanford',
            'mit', 'harvard', 'berkeley', 'carnegie mellon'
        ]
        
        source_count = sum(1 for source in credible_sources 
                          if source in content.lower())
        
        total_citations = citation_count + source_count
        
        if total_citations > 10:
            return 9.0
        elif total_citations > 5:
            return 7.5
        elif total_citations > 2:
            return 6.0
        elif total_citations > 0:
            return 4.0
        else:
            return 2.0
    
    def _assess_practical_value(self, content: str) -> float:
        """Assess practical value and actionability."""
        
        practical_patterns = [
            r'how to', r'step by step', r'implementation',
            r'practical', r'actionable', r'real-world',
            r'use case', r'application', r'deploy',
            r'production', r'best practice', r'tip',
            r'recommendation', r'guideline', r'framework'
        ]
        
        practical_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                             for pattern in practical_patterns)
        
        # Check for specific implementation details
        implementation_patterns = [
            r'install', r'setup', r'configure', r'run',
            r'execute', r'parameter', r'hyperparameter',
            r'optimize', r'tune', r'debug', r'troubleshoot'
        ]
        
        implementation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                                  for pattern in implementation_patterns)
        
        total_practical = practical_count + implementation_count
        
        if total_practical > 15:
            return 10.0
        elif total_practical > 10:
            return 8.0
        elif total_practical > 5:
            return 6.0
        elif total_practical > 2:
            return 4.0
        else:
            return 2.0
    
    def _assess_innovation_factor(self, content: str) -> float:
        """Assess innovation and cutting-edge content."""
        
        innovation_patterns = [
            r'breakthrough', r'novel', r'innovative', r'cutting-edge',
            r'state-of-the-art', r'revolutionary', r'groundbreaking',
            r'emerging', r'next-generation', r'advanced', r'sophisticated',
            r'recent', r'latest', r'new', r'modern', r'contemporary'
        ]
        
        innovation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                              for pattern in innovation_patterns)
        
        # Check for recent developments (years)
        recent_years = [str(year) for year in range(2020, 2025)]
        year_mentions = sum(1 for year in recent_years if year in content)
        
        # Check for trending technologies
        trending_tech = [
            'gpt', 'bert', 'transformer', 'attention', 'llm',
            'generative ai', 'diffusion', 'stable diffusion',
            'chatgpt', 'claude', 'gemini', 'multimodal'
        ]
        
        trend_count = sum(1 for tech in trending_tech 
                         if tech in content.lower())
        
        total_innovation = innovation_count + year_mentions + trend_count
        
        if total_innovation > 15:
            return 10.0
        elif total_innovation > 10:
            return 8.0
        elif total_innovation > 5:
            return 6.0
        elif total_innovation > 2:
            return 4.0
        else:
            return 2.0
    
    def _calculate_weighted_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        
        weights = {
            "technical_accuracy": 0.15,
            "information_density": 0.12,
            "readability_score": 0.10,
            "engagement_factors": 0.12,
            "structure_quality": 0.10,
            "ai_ml_relevance": 0.15,
            "code_quality": 0.10,
            "citation_quality": 0.08,
            "practical_value": 0.10,
            "innovation_factor": 0.08
        }
        
        weighted_score = sum(metrics[metric] * weight 
                           for metric, weight in weights.items())
        
        return min(10.0, max(0.0, weighted_score))
    
    def _generate_quality_insights(self, metrics: Dict[str, float], content: str) -> List[str]:
        """Generate insights about content quality."""
        
        insights = []
        
        # Technical accuracy insights
        if metrics["technical_accuracy"] > 8.5:
            insights.append("Excellent technical accuracy with appropriate use of AI/ML terminology")
        elif metrics["technical_accuracy"] < 6.0:
            insights.append("Technical accuracy could be improved - check for oversimplified claims")
        
        # Information density insights
        if metrics["information_density"] > 8.0:
            insights.append("High information density with rich data and examples")
        elif metrics["information_density"] < 5.0:
            insights.append("Low information density - consider adding more specific data and examples")
        
        # AI/ML relevance insights
        if metrics["ai_ml_relevance"] > 8.0:
            insights.append("Strong AI/ML focus with comprehensive coverage of relevant concepts")
        elif metrics["ai_ml_relevance"] < 6.0:
            insights.append("Limited AI/ML relevance - consider adding more domain-specific content")
        
        # Code quality insights
        if metrics["code_quality"] > 8.0:
            insights.append("Excellent code examples with good practices and explanations")
        elif metrics["code_quality"] < 6.0:
            insights.append("Code quality could be improved - add more examples and explanations")
        
        # Practical value insights
        if metrics["practical_value"] > 8.0:
            insights.append("High practical value with actionable insights and implementation details")
        elif metrics["practical_value"] < 5.0:
            insights.append("Limited practical value - consider adding more actionable content")
        
        return insights
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign letter grade based on quality score."""
        
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7.0:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6.0:
            return "C+"
        elif score >= 5.5:
            return "C"
        elif score >= 5.0:
            return "C-"
        elif score >= 4.0:
            return "D"
        else:
            return "F"
    
    def _generate_improvement_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate specific improvement recommendations."""
        
        recommendations = []
        
        # Check each metric for improvement opportunities
        for metric, score in metrics.items():
            if score < 6.0:  # Below average
                if metric == "technical_accuracy":
                    recommendations.append("Improve technical accuracy by fact-checking claims and using precise terminology")
                elif metric == "information_density":
                    recommendations.append("Increase information density by adding more data points, statistics, and specific examples")
                elif metric == "readability_score":
                    recommendations.append("Improve readability by using shorter sentences and clearer explanations")
                elif metric == "engagement_factors":
                    recommendations.append("Enhance engagement by adding more storytelling elements and real-world examples")
                elif metric == "structure_quality":
                    recommendations.append("Improve structure by adding clear headings and logical flow indicators")
                elif metric == "ai_ml_relevance":
                    recommendations.append("Increase AI/ML relevance by adding more domain-specific concepts and terminology")
                elif metric == "code_quality":
                    recommendations.append("Improve code quality by adding more examples with explanations and best practices")
                elif metric == "citation_quality":
                    recommendations.append("Enhance citations by adding more references to credible sources and research")
                elif metric == "practical_value":
                    recommendations.append("Increase practical value by adding more actionable insights and implementation details")
                elif metric == "innovation_factor":
                    recommendations.append("Boost innovation factor by including more recent developments and cutting-edge concepts")
        
        return recommendations
    
    def _identify_strengths(self, metrics: Dict[str, float]) -> List[str]:
        """Identify content strengths."""
        
        strengths = []
        
        for metric, score in metrics.items():
            if score >= 8.0:  # Strong areas
                if metric == "technical_accuracy":
                    strengths.append("Strong technical accuracy")
                elif metric == "information_density":
                    strengths.append("High information density")
                elif metric == "readability_score":
                    strengths.append("Excellent readability")
                elif metric == "engagement_factors":
                    strengths.append("Highly engaging content")
                elif metric == "structure_quality":
                    strengths.append("Well-structured organization")
                elif metric == "ai_ml_relevance":
                    strengths.append("Strong AI/ML focus")
                elif metric == "code_quality":
                    strengths.append("Excellent code examples")
                elif metric == "citation_quality":
                    strengths.append("Well-cited content")
                elif metric == "practical_value":
                    strengths.append("High practical value")
                elif metric == "innovation_factor":
                    strengths.append("Cutting-edge content")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict[str, float]) -> List[str]:
        """Identify content weaknesses."""
        
        weaknesses = []
        
        for metric, score in metrics.items():
            if score < 6.0:  # Weak areas
                if metric == "technical_accuracy":
                    weaknesses.append("Technical accuracy needs improvement")
                elif metric == "information_density":
                    weaknesses.append("Low information density")
                elif metric == "readability_score":
                    weaknesses.append("Readability issues")
                elif metric == "engagement_factors":
                    weaknesses.append("Limited engagement elements")
                elif metric == "structure_quality":
                    weaknesses.append("Poor structure and organization")
                elif metric == "ai_ml_relevance":
                    weaknesses.append("Limited AI/ML relevance")
                elif metric == "code_quality":
                    weaknesses.append("Poor code examples")
                elif metric == "citation_quality":
                    weaknesses.append("Insufficient citations")
                elif metric == "practical_value":
                    weaknesses.append("Limited practical value")
                elif metric == "innovation_factor":
                    weaknesses.append("Lacks innovation and recent developments")
        
        return weaknesses
    
    def create_quality_gate(self, minimum_score: float = 7.0) -> Dict[str, Any]:
        """Create a quality gate configuration."""
        
        return {
            "minimum_overall_score": minimum_score,
            "required_metrics": {
                "technical_accuracy": 6.0,
                "ai_ml_relevance": 7.0,
                "practical_value": 6.0,
                "structure_quality": 6.0
            },
            "blocking_conditions": [
                "repetition_score > 0.6",
                "unverifiable_claims > 5",
                "suspicious_quotes > 3"
            ],
            "warning_conditions": [
                "overall_score < 7.5",
                "readability_score < 6.0",
                "engagement_factors < 6.0"
            ]
        }
    
    def evaluate_quality_gate(self, content: str, quality_gate: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate content against quality gate criteria."""
        
        if quality_gate is None:
            quality_gate = self.create_quality_gate()
        
        # Get comprehensive quality assessment
        quality_assessment = self.assess_content_quality(content)
        
        # Check repetition and other validations
        repetition_analysis = self.detect_repetition(content)
        expert_analysis = self.analyze_expert_quotes(content)
        factual_analysis = self.analyze_factual_claims(content)
        
        gate_results = {
            "passed": True,
            "overall_score": quality_assessment["overall_score"],
            "grade": quality_assessment["grade"],
            "blocking_issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check minimum overall score
        if quality_assessment["overall_score"] < quality_gate["minimum_overall_score"]:
            gate_results["passed"] = False
            gate_results["blocking_issues"].append(
                f"Overall score ({quality_assessment['overall_score']:.2f}) below minimum ({quality_gate['minimum_overall_score']:.2f})"
            )
        
        # Check required metrics
        for metric, min_score in quality_gate["required_metrics"].items():
            if quality_assessment["quality_metrics"][metric] < min_score:
                gate_results["passed"] = False
                gate_results["blocking_issues"].append(
                    f"{metric} score ({quality_assessment['quality_metrics'][metric]:.2f}) below required ({min_score:.2f})"
                )
        
        # Check blocking conditions
        if repetition_analysis["repetition_score"] > 0.6:
            gate_results["passed"] = False
            gate_results["blocking_issues"].append("High repetition detected")
        
        if len(factual_analysis["claims"]) > 5:
            gate_results["passed"] = False
            gate_results["blocking_issues"].append("Too many unverifiable claims")
        
        if expert_analysis["suspicious_quotes"] > 3:
            gate_results["passed"] = False
            gate_results["blocking_issues"].append("Too many suspicious expert quotes")
        
        # Add warnings
        if quality_assessment["overall_score"] < 7.5:
            gate_results["warnings"].append("Overall quality score is below recommended level")
        
        # Add recommendations
        gate_results["recommendations"].extend(quality_assessment["improvement_recommendations"])
        
        return gate_results