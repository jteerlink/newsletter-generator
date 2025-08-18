"""
Base Agent Framework for Phase 2 Multi-Agent System

This module provides the foundational agent architecture for specialized
newsletter processing agents, including error handling, circuit breakers,
and performance monitoring.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class ProcessingMode(Enum):
    """Agent processing modes."""
    FULL = "full"           # Complete processing with all features
    FAST = "fast"           # Optimized processing with reduced features
    FALLBACK = "fallback"   # Minimal processing for reliability


@dataclass
class AgentMetrics:
    """Performance and reliability metrics for an agent."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_breaker_trips: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        return 100.0 - self.success_rate


@dataclass
class AgentConfiguration:
    """Configuration settings for agent behavior."""
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    health_check_interval_seconds: int = 30
    
    # Performance settings
    max_response_time_ms: int = 30000  # 30 seconds
    enable_performance_monitoring: bool = True
    
    # Processing settings
    default_processing_mode: ProcessingMode = ProcessingMode.FULL
    enable_fallback_mode: bool = True
    
    # Quality settings
    minimum_quality_threshold: float = 0.7
    enable_quality_validation: bool = True


@dataclass
class ProcessingContext:
    """Context information for agent processing."""
    content: str
    section_type: Optional[str] = None
    audience: Optional[str] = None
    technical_level: Optional[str] = None
    word_count_target: Optional[int] = None
    processing_mode: ProcessingMode = ProcessingMode.FULL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'content': self.content,
            'section_type': self.section_type,
            'audience': self.audience,
            'technical_level': self.technical_level,
            'word_count_target': self.word_count_target,
            'processing_mode': self.processing_mode.value,
            'metadata': self.metadata
        }


@dataclass
class ProcessingResult:
    """Result from agent processing operation."""
    success: bool
    processed_content: Optional[str] = None
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    processing_time_ms: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'success': self.success,
            'processed_content': self.processed_content,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'processing_time_ms': self.processing_time_ms,
            'suggestions': self.suggestions,
            'warnings': self.warnings,
            'errors': self.errors,
            'metadata': self.metadata
        }


class CircuitBreaker:
    """Circuit breaker implementation for agent reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN - too many recent failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class BaseSpecializedAgent(ABC):
    """Base class for all specialized newsletter processing agents."""
    
    def __init__(self, name: str, config: Optional[AgentConfiguration] = None):
        """Initialize the agent with configuration."""
        self.name = name
        self.config = config or AgentConfiguration()
        self.status = AgentStatus.HEALTHY
        self.metrics = AgentMetrics()
        self.circuit_breaker = CircuitBreaker(
            self.config.failure_threshold,
            self.config.recovery_timeout_seconds
        )
        
        logger.info(f"Initialized {self.name} agent")
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Main processing method with error handling and metrics."""
        start_time = time.time()
        
        try:
            # Validate input
            if not self._validate_input(context):
                return ProcessingResult(
                    success=False,
                    errors=["Invalid input context"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Execute with circuit breaker protection
            result = self.circuit_breaker.call(self._process_internal, context)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            self._update_metrics_success(processing_time_ms)
            
            # Validate quality if enabled
            if self.config.enable_quality_validation and result.quality_score is not None:
                if result.quality_score < self.config.minimum_quality_threshold:
                    result.warnings.append(f"Quality score {result.quality_score:.3f} below threshold {self.config.minimum_quality_threshold}")
            
            logger.debug(f"{self.name} processed content successfully in {processing_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics_failure()
            
            logger.error(f"{self.name} processing failed: {e}")
            
            # Try fallback processing if enabled
            if self.config.enable_fallback_mode and context.processing_mode != ProcessingMode.FALLBACK:
                logger.info(f"{self.name} attempting fallback processing")
                fallback_context = ProcessingContext(
                    content=context.content,
                    section_type=context.section_type,
                    processing_mode=ProcessingMode.FALLBACK,
                    metadata=context.metadata
                )
                return self._process_fallback(fallback_context)
            
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                processing_time_ms=processing_time_ms
            )
    
    @abstractmethod
    def _process_internal(self, context: ProcessingContext) -> ProcessingResult:
        """Internal processing implementation - must be overridden by subclasses."""
        pass
    
    def _process_fallback(self, context: ProcessingContext) -> ProcessingResult:
        """Fallback processing implementation - can be overridden by subclasses."""
        logger.warning(f"{self.name} using default fallback: returning original content")
        return ProcessingResult(
            success=True,
            processed_content=context.content,
            quality_score=0.5,  # Neutral fallback score
            confidence_score=0.3,  # Low confidence for fallback
            suggestions=["Fallback processing used - consider investigating agent failure"],
            metadata={"processing_mode": "fallback"}
        )
    
    def _validate_input(self, context: ProcessingContext) -> bool:
        """Validate input context - can be overridden by subclasses."""
        return context.content is not None and len(context.content.strip()) > 0
    
    def _update_metrics_success(self, processing_time_ms: float):
        """Update metrics for successful processing."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = datetime.now()
        self.metrics.consecutive_failures = 0
        
        # Update running average of response time
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time_ms = processing_time_ms
        else:
            self.metrics.average_response_time_ms = (
                (self.metrics.average_response_time_ms * (self.metrics.total_requests - 1) + processing_time_ms)
                / self.metrics.total_requests
            )
        
        # Update status based on performance
        if processing_time_ms > self.config.max_response_time_ms:
            self.status = AgentStatus.DEGRADED
        elif self.status == AgentStatus.DEGRADED and self.metrics.consecutive_failures == 0:
            self.status = AgentStatus.HEALTHY
    
    def _update_metrics_failure(self):
        """Update metrics for failed processing."""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = datetime.now()
        self.metrics.consecutive_failures += 1
        
        # Update status based on failure pattern
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            self.status = AgentStatus.FAILED
        elif self.metrics.consecutive_failures > 1:
            self.status = AgentStatus.DEGRADED
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the agent."""
        return {
            'name': self.name,
            'status': self.status.value,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': self.metrics.success_rate,
                'failure_rate': self.metrics.failure_rate,
                'average_response_time_ms': self.metrics.average_response_time_ms,
                'consecutive_failures': self.metrics.consecutive_failures,
                'circuit_breaker_state': self.circuit_breaker.state
            },
            'configuration': {
                'failure_threshold': self.config.failure_threshold,
                'max_response_time_ms': self.config.max_response_time_ms,
                'enable_fallback_mode': self.config.enable_fallback_mode
            },
            'last_activity': {
                'last_success': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                'last_failure': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None
            }
        }
    
    def reset_metrics(self):
        """Reset agent metrics - useful for testing or maintenance."""
        logger.info(f"Resetting metrics for {self.name}")
        self.metrics = AgentMetrics()
        self.status = AgentStatus.HEALTHY
        self.circuit_breaker = CircuitBreaker(
            self.config.failure_threshold,
            self.config.recovery_timeout_seconds
        )


# Export main classes
__all__ = [
    'BaseSpecializedAgent',
    'AgentStatus',
    'ProcessingMode',
    'AgentMetrics',
    'AgentConfiguration',
    'ProcessingContext',
    'ProcessingResult',
    'CircuitBreaker'
]