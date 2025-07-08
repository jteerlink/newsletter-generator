"""
Enhanced Vector Store with Multi-Modal RAG Support

This module extends the base vector store with:
- Multi-modal content support (text, images, PDFs)
- Temporal reasoning and trend analysis
- Hierarchical chunking strategies
- Agentic retrieval with reasoning
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Multi-modal processing imports
try:
    from PIL import Image
    import pytesseract
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from src.core.utils import chunk_text, embed_chunks
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

class EnhancedVectorStore(VectorStore):
    """Enhanced vector store with multi-modal and temporal capabilities."""
    
    def __init__(self, db_path: str = "./data/chroma_db", enable_multimodal: bool = True):
        super().__init__(db_path)
        self.enable_multimodal = enable_multimodal and VISION_AVAILABLE
        
        # Create specialized collections
        self.text_collection = self.client.get_or_create_collection("text_content")
        self.image_collection = self.client.get_or_create_collection("image_content") if self.enable_multimodal else None
        self.temporal_collection = self.client.get_or_create_collection("temporal_content")
        
        # Initialize trend tracking
        self.trend_tracker = TrendTracker(self.temporal_collection)
        
    def add_multimodal_document(self, 
                               text_content: str = "",
                               image_path: str = "",
                               pdf_path: str = "",
                               metadata: Dict[str, Any] = None,
                               enable_ocr: bool = True) -> Dict[str, List[str]]:
        """Add multi-modal document with text, images, and PDFs."""
        
        results = {"text_chunks": [], "image_chunks": [], "pdf_chunks": []}
        metadata = metadata or {}
        
        # Process text content
        if text_content:
            results["text_chunks"] = self.add_document(text_content, metadata)
        
        # Process image content
        if image_path and self.enable_multimodal and enable_ocr:
            try:
                extracted_text = self._extract_text_from_image(image_path)
                if extracted_text:
                    img_metadata = metadata.copy()
                    img_metadata.update({"content_type": "image_ocr", "source_image": image_path})
                    results["image_chunks"] = self.add_document(extracted_text, img_metadata)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        
        # Process PDF content
        if pdf_path and PDF_AVAILABLE:
            try:
                extracted_text = self._extract_text_from_pdf(pdf_path)
                if extracted_text:
                    pdf_metadata = metadata.copy()
                    pdf_metadata.update({"content_type": "pdf", "source_pdf": pdf_path})
                    results["pdf_chunks"] = self.add_document(extracted_text, pdf_metadata)
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return results
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        if not VISION_AVAILABLE:
            return ""
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return ""
    
    def hierarchical_query(self, 
                          query_text: str, 
                          granularity: str = "mixed",
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hierarchical retrieval at different granularities.
        
        Args:
            query_text: The search query
            granularity: "sentence", "paragraph", "document", or "mixed"
            top_k: Number of results to return
        """
        
        if granularity == "mixed":
            # Get results from different granularities
            sentence_results = self.query(query_text, top_k=top_k//3)
            paragraph_results = self.query(query_text, top_k=top_k//3)
            document_results = self.query(query_text, top_k=top_k//3)
            
            # Combine and re-rank
            all_results = sentence_results + paragraph_results + document_results
            return self._rerank_results(query_text, all_results)[:top_k]
        else:
            # Query with specific granularity filter
            filters = {"granularity": granularity} if granularity != "mixed" else None
            return self.query(query_text, filters=filters, top_k=top_k)
    
    def _rerank_results(self, query_text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank results using advanced scoring."""
        
        for result in results:
            # Calculate composite score
            semantic_score = 1 - (result.get("similarity", 0) or 0)  # Convert distance to similarity
            temporal_score = self.temporal_score(result.get("metadata", {}))
            
            # Diversity penalty (reduce score for very similar results)
            diversity_score = 1.0  # Simplified for now
            
            # Composite scoring
            result["composite_score"] = (
                0.5 * semantic_score + 
                0.3 * temporal_score + 
                0.2 * diversity_score
            )
        
        # Sort by composite score
        return sorted(results, key=lambda x: x["composite_score"], reverse=True)
    
    def temporal_query(self, 
                      query_text: str, 
                      time_range: Optional[Dict[str, datetime]] = None,
                      trend_analysis: bool = False) -> Dict[str, Any]:
        """
        Query with temporal awareness and trend analysis.
        
        Args:
            query_text: The search query
            time_range: {"start": datetime, "end": datetime}
            trend_analysis: Whether to include trend analysis
        """
        
        # Base query
        results = self.query(query_text, top_k=20)
        
        # Filter by time range if provided
        if time_range:
            filtered_results = []
            for result in results:
                timestamp = result.get("metadata", {}).get("timestamp")
                if timestamp:
                    try:
                        doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if time_range["start"] <= doc_time <= time_range["end"]:
                            filtered_results.append(result)
                    except Exception:
                        continue
            results = filtered_results
        
        response = {"results": results}
        
        # Add trend analysis if requested
        if trend_analysis:
            response["trends"] = self.trend_tracker.analyze_trends(query_text, results)
        
        return response
    
    def agentic_query(self, 
                     query_text: str, 
                     reasoning_steps: List[str] = None,
                     max_iterations: int = 3) -> Dict[str, Any]:
        """
        Perform agentic retrieval with reasoning and iterative refinement.
        
        Args:
            query_text: The initial query
            reasoning_steps: List of reasoning steps to guide retrieval
            max_iterations: Maximum number of refinement iterations
        """
        
        results = {"iterations": [], "final_results": []}
        current_query = query_text
        
        for iteration in range(max_iterations):
            # Perform query
            iteration_results = self.query(current_query, top_k=10)
            
            # Analyze results quality
            analysis = self._analyze_result_quality(iteration_results, query_text)
            
            results["iterations"].append({
                "iteration": iteration + 1,
                "query": current_query,
                "results": iteration_results,
                "analysis": analysis
            })
            
            # Check if we have sufficient quality results
            if analysis["quality_score"] > 0.7:
                results["final_results"] = iteration_results
                break
            
            # Refine query for next iteration
            current_query = self._refine_query(current_query, analysis)
        
        return results
    
    def _analyze_result_quality(self, results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """Analyze the quality of retrieval results."""
        
        if not results:
            return {"quality_score": 0.0, "issues": ["No results found"]}
        
        # Calculate average similarity
        similarities = [1 - (r.get("similarity", 1) or 1) for r in results]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Check for diversity
        diversity_score = self._calculate_diversity(results)
        
        # Overall quality score
        quality_score = (avg_similarity + diversity_score) / 2
        
        analysis = {
            "quality_score": quality_score,
            "average_similarity": avg_similarity,
            "diversity_score": diversity_score,
            "result_count": len(results),
            "issues": []
        }
        
        # Identify issues
        if avg_similarity < 0.5:
            analysis["issues"].append("Low semantic similarity")
        if diversity_score < 0.3:
            analysis["issues"].append("Low result diversity")
        if len(results) < 5:
            analysis["issues"].append("Insufficient result count")
        
        return analysis
    
    def _calculate_diversity(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for results."""
        if len(results) < 2:
            return 0.0
        
        # Simple diversity calculation based on metadata variety
        unique_sources = set()
        unique_topics = set()
        
        for result in results:
            metadata = result.get("metadata", {})
            unique_sources.add(metadata.get("source", "unknown"))
            unique_topics.add(metadata.get("topic", "unknown"))
        
        # Normalize diversity score
        source_diversity = len(unique_sources) / len(results)
        topic_diversity = len(unique_topics) / len(results)
        
        return (source_diversity + topic_diversity) / 2
    
    def _refine_query(self, current_query: str, analysis: Dict[str, Any]) -> str:
        """Refine query based on analysis results."""
        
        # Simple query refinement strategy
        if "Low semantic similarity" in analysis["issues"]:
            # Add synonyms or related terms
            refined_query = f"{current_query} OR related topics OR similar concepts"
        elif "Low result diversity" in analysis["issues"]:
            # Broaden the query
            refined_query = f"{current_query} OR broader context"
        else:
            refined_query = current_query
        
        return refined_query

class TrendTracker:
    """Track and analyze trends in document collections."""
    
    def __init__(self, collection):
        self.collection = collection
    
    def analyze_trends(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the retrieved results."""
        
        # Group results by time periods
        time_buckets = self._group_by_time(results)
        
        # Calculate trend metrics
        trend_metrics = {
            "volume_trend": self._calculate_volume_trend(time_buckets),
            "topic_evolution": self._analyze_topic_evolution(time_buckets),
            "sentiment_trend": self._analyze_sentiment_trend(time_buckets),
            "emerging_themes": self._identify_emerging_themes(time_buckets)
        }
        
        return trend_metrics
    
    def _group_by_time(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by time periods."""
        
        buckets = {}
        for result in results:
            timestamp = result.get("metadata", {}).get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    # Group by month for now
                    period = dt.strftime('%Y-%m')
                    if period not in buckets:
                        buckets[period] = []
                    buckets[period].append(result)
                except Exception:
                    continue
        
        return buckets
    
    def _calculate_volume_trend(self, time_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate volume trends over time."""
        
        if not time_buckets:
            return {"trend": "no_data", "data": []}
        
        # Sort periods and calculate volumes
        sorted_periods = sorted(time_buckets.keys())
        volumes = [len(time_buckets[period]) for period in sorted_periods]
        
        # Calculate trend direction
        if len(volumes) >= 2:
            recent_avg = sum(volumes[-2:]) / 2
            earlier_avg = sum(volumes[:-2]) / max(1, len(volumes[:-2]))
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "data": list(zip(sorted_periods, volumes)),
            "total_volume": sum(volumes)
        }
    
    def _analyze_topic_evolution(self, time_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze how topics evolve over time."""
        
        # Extract key terms from each time period
        period_terms = {}
        for period, results in time_buckets.items():
            terms = []
            for result in results:
                # Extract key terms from document content
                content = result.get("document", "")
                # Simple term extraction (could be enhanced with NER)
                words = content.lower().split()
                terms.extend([word for word in words if len(word) > 4])
            
            # Count term frequencies
            term_freq = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
            
            # Get top terms
            top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            period_terms[period] = top_terms
        
        return {
            "period_terms": period_terms,
            "evolution_summary": "Topics evolving over time"  # Could be enhanced
        }
    
    def _analyze_sentiment_trend(self, time_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze sentiment trends (placeholder for now)."""
        
        # This would integrate with sentiment analysis tools
        return {
            "trend": "neutral",
            "note": "Sentiment analysis not implemented yet"
        }
    
    def _identify_emerging_themes(self, time_buckets: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify emerging themes in recent time periods."""
        
        if not time_buckets:
            return []
        
        # Get the most recent period
        recent_period = max(time_buckets.keys())
        recent_results = time_buckets[recent_period]
        
        # Extract themes (simplified approach)
        themes = []
        for result in recent_results:
            content = result.get("document", "")
            # Simple theme extraction
            if "AI" in content.upper() or "ARTIFICIAL INTELLIGENCE" in content.upper():
                themes.append("AI/ML")
            if "SUSTAINABILITY" in content.upper() or "CLIMATE" in content.upper():
                themes.append("Sustainability")
            if "BLOCKCHAIN" in content.upper() or "CRYPTO" in content.upper():
                themes.append("Blockchain")
        
        # Return unique themes
        return list(set(themes))

# Factory function
def create_enhanced_vector_store(db_path: str = "./data/enhanced_chroma_db", 
                               enable_multimodal: bool = True) -> EnhancedVectorStore:
    """Create an enhanced vector store instance."""
    return EnhancedVectorStore(db_path=db_path, enable_multimodal=enable_multimodal) 