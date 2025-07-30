"""
Unified Storage System

This module provides a unified storage system that consolidates all vector database
and storage functionality into a single, cohesive interface.
"""

from .base import (
    StorageProvider, StorageConfig, DocumentMetadata, SearchResult, 
    StorageStats, StorageType, DataType, StorageManager
)
from .vector_store import ChromaStorageProvider
from .memory_store import MemoryStorageProvider
from .data_manager import DataManager, DataVersion
from .migration import MigrationManager, MigrationPlan, DataFormatConverter
from .legacy_wrappers import get_db_collection, add_text_to_db, search_vector_db

__all__ = [
    # Base classes and interfaces
    'StorageProvider',
    'StorageConfig', 
    'DocumentMetadata',
    'SearchResult',
    'StorageStats',
    'StorageType',
    'DataType',
    'StorageManager',
    
    # Storage providers
    'ChromaStorageProvider',
    'MemoryStorageProvider',
    
    # Data management
    'DataManager',
    'DataVersion',
    
    # Migration tools
    'MigrationManager',
    'MigrationPlan',
    'DataFormatConverter',
    
    # Legacy wrappers
    'get_db_collection',
    'add_text_to_db',
    'search_vector_db'
] 