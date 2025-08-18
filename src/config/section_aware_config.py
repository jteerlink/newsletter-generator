"""
Configuration Management for Section-Aware Newsletter Generation

This module provides configuration management for the section-aware newsletter
generation system, including settings for prompts, quality thresholds, and
processing parameters.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import will be done dynamically to avoid circular imports
# from ..core.section_aware_prompts import SectionType
# from ..core.section_quality_metrics import QualityDimension

logger = logging.getLogger(__name__)


@dataclass
class SectionWeights:
    """Quality dimension weights for a specific section type."""
    clarity: float = 1.0
    relevance: float = 1.0
    completeness: float = 1.0
    accuracy: float = 1.0
    engagement: float = 1.0
    structure: float = 1.0
    consistency: float = 1.0
    readability: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> SectionWeights:
        """Create from dictionary format."""
        return cls(**data)


@dataclass
class PromptConfiguration:
    """Configuration for section-aware prompt generation."""
    default_tone: str = "professional"
    default_technical_level: str = "intermediate"
    default_word_count: int = 3000
    include_examples: bool = False
    include_citations: bool = False
    
    # Section-specific word count multipliers
    section_word_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "introduction": 0.15,  # 15% of total
        "news": 0.20,          # 20% of total
        "analysis": 0.35,      # 35% of total
        "tutorial": 0.25,      # 25% of total
        "conclusion": 0.05     # 5% of total
    })
    
    # Audience-specific customizations
    audience_customizations: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "AI/ML Engineers": {
            "technical_level": "expert",
            "include_examples": True,
            "focus_areas": ["implementation", "performance", "scalability"]
        },
        "Data Scientists": {
            "technical_level": "expert", 
            "include_citations": True,
            "focus_areas": ["methodology", "analysis", "validation"]
        },
        "Software Developers": {
            "technical_level": "intermediate",
            "include_examples": True,
            "focus_areas": ["practical_implementation", "code_examples", "best_practices"]
        },
        "Business Professionals": {
            "technical_level": "beginner",
            "include_citations": False,
            "focus_areas": ["business_impact", "strategic_implications", "roi"]
        },
        "Research Community": {
            "technical_level": "expert",
            "include_citations": True,
            "focus_areas": ["methodology", "experimental_results", "theoretical_foundations"]
        }
    })


@dataclass
class QualityConfiguration:
    """Configuration for section-aware quality assessment."""
    # Global quality thresholds
    global_quality_threshold: float = 0.8
    
    # Section-specific quality thresholds
    section_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "introduction": 0.8,
        "news": 0.75,
        "analysis": 0.85,
        "tutorial": 0.8,
        "conclusion": 0.7
    })
    
    # Section-specific quality weights
    section_weights: Dict[str, SectionWeights] = field(default_factory=lambda: {
        "introduction": SectionWeights(
            engagement=2.0, clarity=1.5, relevance=1.5, structure=1.0
        ),
        "news": SectionWeights(
            accuracy=2.0, relevance=1.8, clarity=1.5, engagement=1.2
        ),
        "analysis": SectionWeights(
            accuracy=2.0, completeness=1.8, clarity=1.5, structure=1.2
        ),
        "tutorial": SectionWeights(
            clarity=2.0, completeness=1.8, structure=1.5, accuracy=1.5
        ),
        "conclusion": SectionWeights(
            completeness=1.8, engagement=1.5, clarity=1.2, relevance=1.0
        )
    })
    
    # Readability scoring parameters
    readability_config: Dict[str, Any] = field(default_factory=lambda: {
        "target_flesch_score": 60.0,  # College level
        "max_sentence_length": 25,
        "max_paragraph_length": 150,
        "preferred_syllables_per_word": 1.5
    })


@dataclass
class RefinementConfiguration:
    """Configuration for section-aware refinement process."""
    max_iterations: int = 3
    quality_improvement_threshold: float = 0.05
    
    # Refinement pass quality thresholds
    pass_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "structure": 0.7,
        "content": 0.8,
        "style": 0.85,
        "technical": 0.9,
        "final": 0.95
    })
    
    # Section-specific refinement strategies
    section_strategies: Dict[str, List[str]] = field(default_factory=lambda: {
        "introduction": ["engagement", "clarity", "structure"],
        "news": ["accuracy", "relevance", "structure"],
        "analysis": ["accuracy", "completeness", "clarity"],
        "tutorial": ["clarity", "structure", "completeness"],
        "conclusion": ["completeness", "engagement", "clarity"]
    })
    
    # Refinement timeout settings
    timeout_settings: Dict[str, int] = field(default_factory=lambda: {
        "per_section": 120,  # seconds
        "total_process": 600,  # seconds
        "llm_request": 30  # seconds
    })


@dataclass
class ContinuityConfiguration:
    """Configuration for continuity validation."""
    # Transition quality thresholds
    transition_quality_threshold: float = 0.6
    style_consistency_threshold: float = 0.7
    redundancy_threshold: float = 0.7
    
    # Style analysis parameters
    style_analysis: Dict[str, Any] = field(default_factory=lambda: {
        "formality_variance_threshold": 0.05,
        "technical_density_variance_threshold": 0.03,
        "sentence_length_variance_threshold": 5.0,
        "pronoun_ratio_variance_threshold": 0.02
    })
    
    # Transition scoring weights
    transition_weights: Dict[str, float] = field(default_factory=lambda: {
        "explicit_indicators": 0.3,
        "contextual_appropriateness": 0.4,
        "logical_flow": 0.2,
        "topic_continuity": 0.1
    })
    
    # Issue severity thresholds
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8,
        "critical": 0.9
    })


@dataclass
class SectionAwareConfig:
    """Master configuration for section-aware newsletter generation."""
    # Component configurations
    prompts: PromptConfiguration = field(default_factory=PromptConfiguration)
    quality: QualityConfiguration = field(default_factory=QualityConfiguration)
    refinement: RefinementConfiguration = field(default_factory=RefinementConfiguration)
    continuity: ContinuityConfiguration = field(default_factory=ContinuityConfiguration)
    
    # System settings
    system: Dict[str, Any] = field(default_factory=lambda: {
        "log_level": "INFO",
        "enable_metrics": True,
        "enable_caching": True,
        "cache_ttl": 3600,
        "max_concurrent_operations": 3,
        "enable_performance_monitoring": True
    })
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "section_aware_prompts": True,
        "multi_pass_refinement": True,
        "quality_metrics": True,
        "continuity_validation": True,
        "performance_optimization": True,
        "advanced_analytics": False
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SectionAwareConfig:
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        config_data = data.copy()
        
        if 'prompts' in config_data:
            config_data['prompts'] = PromptConfiguration(**config_data['prompts'])
        
        if 'quality' in config_data:
            quality_data = config_data['quality'].copy()
            if 'section_weights' in quality_data:
                quality_data['section_weights'] = {
                    k: SectionWeights.from_dict(v) 
                    for k, v in quality_data['section_weights'].items()
                }
            config_data['quality'] = QualityConfiguration(**quality_data)
        
        if 'refinement' in config_data:
            config_data['refinement'] = RefinementConfiguration(**config_data['refinement'])
        
        if 'continuity' in config_data:
            config_data['continuity'] = ContinuityConfiguration(**config_data['continuity'])
        
        return cls(**config_data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate quality thresholds
        for section, threshold in self.quality.section_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                issues.append(f"Invalid quality threshold for {section}: {threshold}")
        
        # Validate refinement iterations
        if self.refinement.max_iterations < 1 or self.refinement.max_iterations > 10:
            issues.append(f"Invalid max_iterations: {self.refinement.max_iterations}")
        
        # Validate word count multipliers
        total_multiplier = sum(self.prompts.section_word_multipliers.values())
        if abs(total_multiplier - 1.0) > 0.1:
            issues.append(f"Section word multipliers sum to {total_multiplier}, should be ~1.0")
        
        # Validate timeout settings
        for setting, value in self.refinement.timeout_settings.items():
            if value <= 0:
                issues.append(f"Invalid timeout setting {setting}: {value}")
        
        return issues


class ConfigurationManager:
    """Manages loading, saving, and validation of section-aware configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        self._config: Optional[SectionAwareConfig] = None
        logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
    
    def load_config(self, config_file: Optional[str] = None) -> SectionAwareConfig:
        """Load configuration from file or environment."""
        if config_file:
            config_path = Path(config_file)
        else:
            # Try multiple default locations
            possible_files = [
                self.config_dir / "section_aware_config.yaml",
                self.config_dir / "section_aware_config.json",
                self.config_dir / "config.yaml",
                self.config_dir / "config.json"
            ]
            
            config_path = None
            for path in possible_files:
                if path.exists():
                    config_path = path
                    break
        
        if config_path and config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            return self._load_from_file(config_path)
        else:
            logger.info("No configuration file found, using defaults with environment overrides")
            return self._load_from_environment()
    
    def _load_from_file(self, config_path: Path) -> SectionAwareConfig:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            config = SectionAwareConfig.from_dict(data)
            
            # Apply environment overrides
            self._apply_environment_overrides(config)
            
            # Validate configuration
            issues = config.validate()
            if issues:
                logger.warning(f"Configuration validation issues: {issues}")
            
            self._config = config
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Falling back to default configuration")
            return self._load_from_environment()
    
    def _load_from_environment(self) -> SectionAwareConfig:
        """Load configuration from environment variables and defaults."""
        config = SectionAwareConfig()
        self._apply_environment_overrides(config)
        self._config = config
        return config
    
    def _apply_environment_overrides(self, config: SectionAwareConfig) -> None:
        """Apply environment variable overrides to configuration."""
        # System settings
        if os.getenv('LOG_LEVEL'):
            config.system['log_level'] = os.getenv('LOG_LEVEL')
        
        if os.getenv('ENABLE_METRICS'):
            config.system['enable_metrics'] = os.getenv('ENABLE_METRICS').lower() == 'true'
        
        if os.getenv('MAX_CONCURRENT_OPERATIONS'):
            try:
                config.system['max_concurrent_operations'] = int(os.getenv('MAX_CONCURRENT_OPERATIONS'))
            except ValueError:
                pass
        
        # Quality settings
        if os.getenv('GLOBAL_QUALITY_THRESHOLD'):
            try:
                config.quality.global_quality_threshold = float(os.getenv('GLOBAL_QUALITY_THRESHOLD'))
            except ValueError:
                pass
        
        # Refinement settings
        if os.getenv('MAX_REFINEMENT_ITERATIONS'):
            try:
                config.refinement.max_iterations = int(os.getenv('MAX_REFINEMENT_ITERATIONS'))
            except ValueError:
                pass
        
        # Feature flags
        feature_flags = {
            'ENABLE_SECTION_AWARE_PROMPTS': 'section_aware_prompts',
            'ENABLE_MULTI_PASS_REFINEMENT': 'multi_pass_refinement',
            'ENABLE_QUALITY_METRICS': 'quality_metrics',
            'ENABLE_CONTINUITY_VALIDATION': 'continuity_validation',
            'ENABLE_PERFORMANCE_OPTIMIZATION': 'performance_optimization',
            'ENABLE_ADVANCED_ANALYTICS': 'advanced_analytics'
        }
        
        for env_var, feature_key in feature_flags.items():
            if os.getenv(env_var):
                config.features[feature_key] = os.getenv(env_var).lower() == 'true'
    
    def save_config(self, config: SectionAwareConfig, 
                   config_file: Optional[str] = None, 
                   format_type: str = 'yaml') -> str:
        """Save configuration to file."""
        if config_file:
            config_path = Path(config_file)
        else:
            extension = 'yaml' if format_type == 'yaml' else 'json'
            config_path = self.config_dir / f"section_aware_config.{extension}"
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_data = config.to_dict()
            
            with open(config_path, 'w') as f:
                if format_type == 'yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to: {config_path}")
            return str(config_path)
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def get_current_config(self) -> Optional[SectionAwareConfig]:
        """Get currently loaded configuration."""
        return self._config
    
    def create_default_config_file(self, config_file: Optional[str] = None) -> str:
        """Create a default configuration file."""
        default_config = SectionAwareConfig()
        return self.save_config(default_config, config_file)
    
    def get_section_config(self, section_type) -> Dict[str, Any]:
        """Get configuration specific to a section type."""
        if not self._config:
            self._config = self.load_config()
        
        section_key = section_type.value if hasattr(section_type, 'value') else str(section_type)
        
        return {
            'quality_threshold': self._config.quality.section_thresholds.get(section_key, 0.8),
            'quality_weights': (
                self._config.quality.section_weights.get(section_key, SectionWeights()).to_dict()
            ),
            'word_multiplier': self._config.prompts.section_word_multipliers.get(section_key, 0.2),
            'refinement_strategies': self._config.refinement.section_strategies.get(section_key, []),
            'enabled_features': {
                key: value for key, value in self._config.features.items() 
                if value
            }
        }


# Global configuration manager instance
_config_manager = ConfigurationManager()

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return _config_manager

def get_current_config() -> SectionAwareConfig:
    """Get the current configuration, loading if necessary."""
    config = _config_manager.get_current_config()
    if config is None:
        config = _config_manager.load_config()
    return config

def reload_config(config_file: Optional[str] = None) -> SectionAwareConfig:
    """Reload configuration from file."""
    return _config_manager.load_config(config_file)


# Export main classes and functions
__all__ = [
    'SectionAwareConfig',
    'PromptConfiguration',
    'QualityConfiguration', 
    'RefinementConfiguration',
    'ContinuityConfiguration',
    'SectionWeights',
    'ConfigurationManager',
    'get_config_manager',
    'get_current_config',
    'reload_config'
]