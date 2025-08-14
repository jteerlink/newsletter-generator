"""
Generation Timeout Monitoring and Management System

This module provides comprehensive monitoring, timeout management, and checkpoint
tracking for newsletter generation to prevent incomplete content delivery.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class GenerationStatus(Enum):
    """Status of newsletter generation process."""
    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    GENERATING = "generating"
    REFINING = "refining"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class GenerationCheckpoint:
    """Represents a checkpoint in the generation process."""
    section_name: str
    target_words: int
    status: GenerationStatus = GenerationStatus.INITIALIZING
    actual_words: int = 0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    content_preview: str = ""
    error_message: Optional[str] = None
    validation_score: float = 0.0
    
    def mark_started(self) -> None:
        """Mark checkpoint as started."""
        self.start_time = time.time()
        self.status = GenerationStatus.GENERATING
        logger.info(f"Checkpoint started: {self.section_name}")
    
    def mark_completed(self, word_count: int, content_preview: str = "", validation_score: float = 0.0) -> None:
        """Mark checkpoint as completed."""
        self.completion_time = time.time()
        self.actual_words = word_count
        self.content_preview = content_preview[:200]  # First 200 chars
        self.validation_score = validation_score
        self.status = GenerationStatus.COMPLETED
        
        duration = self.completion_time - (self.start_time or 0)
        logger.info(f"Checkpoint completed: {self.section_name} ({word_count} words, {duration:.2f}s)")
    
    def mark_failed(self, error: str) -> None:
        """Mark checkpoint as failed."""
        self.completion_time = time.time()
        self.error_message = error
        self.status = GenerationStatus.FAILED
        logger.error(f"Checkpoint failed: {self.section_name} - {error}")
    
    @property
    def duration(self) -> float:
        """Get duration of checkpoint processing."""
        if self.start_time is None:
            return 0.0
        end_time = self.completion_time or time.time()
        return end_time - self.start_time
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage based on word count."""
        if self.target_words == 0:
            return 1.0 if self.status == GenerationStatus.COMPLETED else 0.0
        return min(1.0, self.actual_words / self.target_words)


@dataclass  
class GenerationMetadata:
    """Metadata for the entire generation process."""
    session_id: str
    workflow_id: str
    topic: str
    audience: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: GenerationStatus = GenerationStatus.INITIALIZING
    checkpoints: List[GenerationCheckpoint] = field(default_factory=list)
    timeout_events: List[Dict[str, Any]] = field(default_factory=list)
    error_events: List[Dict[str, Any]] = field(default_factory=list)
    total_words_generated: int = 0
    total_words_target: int = 0
    
    @property
    def total_duration(self) -> float:
        """Get total generation duration."""
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    @property
    def overall_progress(self) -> float:
        """Get overall progress percentage."""
        if not self.checkpoints:
            return 0.0
        
        completed_checkpoints = sum(1 for cp in self.checkpoints if cp.status == GenerationStatus.COMPLETED)
        return completed_checkpoints / len(self.checkpoints)
    
    @property
    def word_count_progress(self) -> float:
        """Get progress based on word count."""
        if self.total_words_target == 0:
            return 0.0
        return min(1.0, self.total_words_generated / self.total_words_target)
    
    def add_checkpoint_data(self, checkpoint: GenerationCheckpoint) -> None:
        """Add checkpoint completion data to metadata."""
        self.total_words_generated = sum(cp.actual_words for cp in self.checkpoints)
    
    def add_timeout_event(self, phase: str, duration: float, details: Dict[str, Any]) -> None:
        """Record a timeout event."""
        event = {
            'timestamp': time.time(),
            'phase': phase,
            'duration': duration,
            'details': details
        }
        self.timeout_events.append(event)
        logger.warning(f"Timeout event recorded: {phase} ({duration:.2f}s)")
    
    def add_error_event(self, phase: str, error: str, details: Dict[str, Any]) -> None:
        """Record an error event."""
        event = {
            'timestamp': time.time(),
            'phase': phase,
            'error': error,
            'details': details
        }
        self.error_events.append(event)
        logger.error(f"Error event recorded: {phase} - {error}")


class GenerationTimeoutException(Exception):
    """Exception raised when generation times out."""
    def __init__(self, message: str, metadata: Optional[GenerationMetadata] = None):
        super().__init__(message)
        self.metadata = metadata


class GenerationMonitor:
    """Monitors and manages generation process with timeout handling."""
    
    def __init__(self, default_timeout: int = 300, checkpoint_timeout: int = 120):
        """
        Initialize generation monitor.
        
        Args:
            default_timeout: Default timeout for entire generation process (seconds)
            checkpoint_timeout: Timeout for individual checkpoints (seconds)
        """
        self.default_timeout = default_timeout
        self.checkpoint_timeout = checkpoint_timeout
        self.active_generations: Dict[str, GenerationMetadata] = {}
    
    def monitor_generation_timeout(self, timeout: Optional[int] = None):
        """Decorator to monitor generation function timeouts."""
        actual_timeout = timeout or self.default_timeout
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                function_name = func.__name__
                
                try:
                    logger.info(f"Starting monitored generation: {function_name} (timeout: {actual_timeout}s)")
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    logger.info(f"Generation completed successfully: {function_name} ({duration:.2f}s)")
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    if "timeout" in str(e).lower() or duration >= actual_timeout:
                        error_msg = f"Generation timeout after {duration:.2f}s (limit: {actual_timeout}s)"
                        logger.error(f"TIMEOUT: {function_name} - {error_msg}")
                        raise GenerationTimeoutException(error_msg)
                    else:
                        logger.error(f"Generation failed: {function_name} - {str(e)} ({duration:.2f}s)")
                        raise
            
            return wrapper
        return decorator
    
    def create_generation_metadata(self, session_id: str, workflow_id: str, 
                                  topic: str, audience: str, 
                                  checkpoints: List[GenerationCheckpoint]) -> GenerationMetadata:
        """Create and register generation metadata."""
        metadata = GenerationMetadata(
            session_id=session_id,
            workflow_id=workflow_id,
            topic=topic,
            audience=audience,
            checkpoints=checkpoints,
            total_words_target=sum(cp.target_words for cp in checkpoints)
        )
        
        self.active_generations[session_id] = metadata
        logger.info(f"Generation metadata created: {session_id} ({len(checkpoints)} checkpoints)")
        
        return metadata
    
    def monitor_checkpoint_execution(self, checkpoint: GenerationCheckpoint, 
                                   func: Callable, *args, **kwargs) -> Any:
        """Monitor execution of a single checkpoint with timeout handling."""
        checkpoint.mark_started()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Analyze result to extract word count and validation
            word_count = self._estimate_word_count(result)
            content_preview = self._extract_content_preview(result)
            validation_score = self._calculate_validation_score(result, checkpoint)
            
            checkpoint.mark_completed(word_count, content_preview, validation_score)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            if duration >= self.checkpoint_timeout:
                error_msg = f"Checkpoint timeout after {duration:.2f}s (limit: {self.checkpoint_timeout}s)"
                checkpoint.mark_failed(error_msg)
                raise GenerationTimeoutException(error_msg)
            else:
                checkpoint.mark_failed(str(e))
                raise
    
    def get_generation_status(self, session_id: str) -> Optional[GenerationMetadata]:
        """Get current status of a generation process."""
        return self.active_generations.get(session_id)
    
    def finalize_generation(self, session_id: str, success: bool = True) -> Optional[GenerationMetadata]:
        """Finalize generation and cleanup metadata."""
        metadata = self.active_generations.get(session_id)
        if metadata:
            metadata.end_time = time.time()
            metadata.status = GenerationStatus.COMPLETED if success else GenerationStatus.FAILED
            
            # Generate final report
            self._generate_completion_report(metadata)
            
            # Keep in active generations for a short time for status queries
            # In production, you might want to move to a completed_generations dict
            
        return metadata
    
    def _estimate_word_count(self, content: Any) -> int:
        """Estimate word count from generated content."""
        if isinstance(content, str):
            return len(content.split())
        elif isinstance(content, dict) and 'content' in content:
            return len(str(content['content']).split())
        elif hasattr(content, 'content'):
            return len(str(content.content).split())
        else:
            return len(str(content).split())
    
    def _extract_content_preview(self, content: Any) -> str:
        """Extract preview of generated content."""
        if isinstance(content, str):
            return content[:200]
        elif isinstance(content, dict) and 'content' in content:
            return str(content['content'])[:200]
        elif hasattr(content, 'content'):
            return str(content.content)[:200]
        else:
            return str(content)[:200]
    
    def _calculate_validation_score(self, content: Any, checkpoint: GenerationCheckpoint) -> float:
        """Calculate basic validation score for checkpoint content."""
        estimated_words = self._estimate_word_count(content)
        
        # Basic validation: word count vs target
        word_count_score = min(1.0, estimated_words / max(1, checkpoint.target_words))
        
        # Basic completeness check
        content_str = str(content)
        completeness_score = 1.0
        
        # Check for incomplete sentences
        if content_str.rstrip().endswith((',', 'and', 'the', 'a')):
            completeness_score *= 0.5
        
        # Check for minimum content length
        if len(content_str.strip()) < 50:
            completeness_score *= 0.3
        
        return (word_count_score * 0.7) + (completeness_score * 0.3)
    
    def _generate_completion_report(self, metadata: GenerationMetadata) -> None:
        """Generate completion report with diagnostics."""
        report = {
            'session_id': metadata.session_id,
            'topic': metadata.topic,
            'total_duration': metadata.total_duration,
            'overall_progress': metadata.overall_progress,
            'word_count_progress': metadata.word_count_progress,
            'checkpoints_completed': sum(1 for cp in metadata.checkpoints if cp.status == GenerationStatus.COMPLETED),
            'checkpoints_failed': sum(1 for cp in metadata.checkpoints if cp.status == GenerationStatus.FAILED),
            'timeout_events': len(metadata.timeout_events),
            'error_events': len(metadata.error_events)
        }
        
        logger.info(f"Generation completion report: {report}")
        
        # Log detailed checkpoint information
        for i, checkpoint in enumerate(metadata.checkpoints):
            logger.info(f"  Checkpoint {i+1}: {checkpoint.section_name} - {checkpoint.status.value} "
                       f"({checkpoint.actual_words}/{checkpoint.target_words} words, {checkpoint.duration:.2f}s)")


# Global monitor instance
_global_monitor = GenerationMonitor()


def get_generation_monitor() -> GenerationMonitor:
    """Get global generation monitor instance."""
    return _global_monitor


def monitor_generation_timeout(timeout: Optional[int] = None):
    """Decorator to monitor generation function timeouts."""
    return _global_monitor.monitor_generation_timeout(timeout)