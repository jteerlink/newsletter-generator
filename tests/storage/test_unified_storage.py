"""
Tests for the unified storage system.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.storage import (
    StorageProvider, StorageConfig, DocumentMetadata, SearchResult,
    StorageStats, StorageType, DataType,
    ChromaStorageProvider, MemoryStorageProvider,
    DataManager
)


class TestStorageConfig:
    """Test storage configuration."""
    
    def test_storage_config_creation(self):
        """Test creating storage configuration."""
        config = StorageConfig(
            storage_type=StorageType.CHROMA,
            db_path="./test_data",
            collection_name="test_collection",
            embedding_model="test-model",
            chunk_size=500,
            chunk_overlap=50
        )
        
        assert config.storage_type == StorageType.CHROMA
        assert config.db_path == "./test_data"
        assert config.collection_name == "test_collection"
        assert config.embedding_model == "test-model"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.enable_cache is True
        assert config.cache_size == 1000


class TestDocumentMetadata:
    """Test document metadata."""
    
    def test_document_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now(),
            author="Test Author",
            tags=["test", "document"]
        )
        
        assert metadata.doc_id == "test_doc_123"
        assert metadata.title == "Test Document"
        assert metadata.source == "https://example.com"
        assert metadata.content_type == DataType.TEXT
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "document"]
        assert metadata.version == "1.0"
        assert metadata.checksum is not None
    
    def test_document_metadata_defaults(self):
        """Test document metadata with defaults."""
        metadata = DocumentMetadata(
            doc_id="test_doc_456",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        assert metadata.author is None
        assert metadata.tags == []
        assert metadata.version == "1.0"
        assert metadata.checksum is not None


class TestSearchResult:
    """Test search result."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        result = SearchResult(
            doc_id="test_doc_123",
            content="This is test content",
            metadata=metadata,
            similarity_score=0.85,
            rank=1,
            highlights=["test", "content"]
        )
        
        assert result.doc_id == "test_doc_123"
        assert result.content == "This is test content"
        assert result.metadata == metadata
        assert result.similarity_score == 0.85
        assert result.rank == 1
        assert result.highlights == ["test", "content"]
    
    def test_search_result_defaults(self):
        """Test search result with defaults."""
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        result = SearchResult(
            doc_id="test_doc_123",
            content="This is test content",
            metadata=metadata,
            similarity_score=0.85,
            rank=1
        )
        
        assert result.highlights == []


class TestMemoryStorageProvider:
    """Test memory storage provider."""
    
    @pytest.fixture
    def memory_provider(self):
        """Create a memory storage provider for testing."""
        config = StorageConfig(
            storage_type=StorageType.MEMORY,
            db_path="./test_memory_db"
        )
        return MemoryStorageProvider(config)
    
    def test_memory_provider_initialization(self, memory_provider):
        """Test memory provider initialization."""
        assert memory_provider.initialize() is True
        assert memory_provider.initialized is True
    
    def test_memory_provider_add_document(self, memory_provider):
        """Test adding document to memory provider."""
        memory_provider.initialize()
        
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        doc_id = memory_provider.add_document("This is test content", metadata)
        
        assert doc_id == "test_doc_123"
        assert len(memory_provider.documents) == 1
        assert "test_doc_123" in memory_provider.documents
    
    def test_memory_provider_search(self, memory_provider):
        """Test searching in memory provider."""
        memory_provider.initialize()
        
        # Add test documents
        metadata1 = DocumentMetadata(
            doc_id="doc1",
            title="AI and Machine Learning",
            source="https://example.com/ai",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        metadata2 = DocumentMetadata(
            doc_id="doc2", 
            title="Python Programming",
            source="https://example.com/python",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        memory_provider.add_document("AI and machine learning are transforming technology", metadata1)
        memory_provider.add_document("Python is a powerful programming language", metadata2)
        
        # Search for AI-related content
        results = memory_provider.search("AI machine learning", top_k=5)
        
        assert len(results) > 0
        assert all(isinstance(result, SearchResult) for result in results)
        assert results[0].similarity_score > 0  # Should have some similarity
    
    def test_memory_provider_get_document(self, memory_provider):
        """Test getting document from memory provider."""
        memory_provider.initialize()
        
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        memory_provider.add_document("This is test content", metadata)
        
        result = memory_provider.get_document("test_doc_123")
        
        assert result is not None
        content, retrieved_metadata = result
        assert content == "This is test content"
        assert retrieved_metadata.doc_id == "test_doc_123"
    
    def test_memory_provider_delete_document(self, memory_provider):
        """Test deleting document from memory provider."""
        memory_provider.initialize()
        
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        memory_provider.add_document("This is test content", metadata)
        assert len(memory_provider.documents) == 1
        
        success = memory_provider.delete_document("test_doc_123")
        assert success is True
        assert len(memory_provider.documents) == 0
    
    def test_memory_provider_stats(self, memory_provider):
        """Test getting stats from memory provider."""
        memory_provider.initialize()
        
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        memory_provider.add_document("This is test content", metadata)
        
        stats = memory_provider.get_stats()
        
        assert isinstance(stats, StorageStats)
        assert stats.total_documents == 1
        assert stats.total_chunks == 1
        assert stats.storage_size_bytes > 0


class TestDataManager:
    """Test data manager."""
    
    @pytest.fixture
    def memory_provider(self):
        """Create a memory storage provider for testing."""
        config = StorageConfig(
            storage_type=StorageType.MEMORY,
            db_path="./test_memory_db"
        )
        provider = MemoryStorageProvider(config)
        provider.initialize()
        return provider
    
    @pytest.fixture
    def data_manager(self, memory_provider):
        """Create a data manager for testing."""
        manager = DataManager(memory_provider, backup_dir="./test_backups")
        # Clean up any existing versions
        manager.versions.clear()
        manager._save_versions()
        return manager
    
    def test_data_manager_creation(self, data_manager):
        """Test data manager creation."""
        assert data_manager.storage_provider is not None
        assert data_manager.backup_dir.exists()
        assert data_manager.max_versions == 10
    
    def test_data_manager_create_backup(self, data_manager, memory_provider):
        """Test creating backup."""
        # Add some test data
        metadata = DocumentMetadata(
            doc_id="test_doc_123",
            title="Test Document",
            source="https://example.com",
            content_type=DataType.TEXT,
            timestamp=datetime.now()
        )
        
        memory_provider.add_document("This is test content", metadata)
        
        # Create backup
        version_id = data_manager.create_backup("Test backup")
        
        assert version_id is not None
        assert version_id.startswith("backup_")
        assert len(data_manager.versions) == 1
    
    def test_data_manager_list_versions(self, data_manager):
        """Test listing versions."""
        # Create multiple backups
        data_manager.create_backup("First backup")
        data_manager.create_backup("Second backup")
        
        versions = data_manager.list_versions()
        
        assert len(versions) == 2
        assert versions[0].description == "Second backup"  # Most recent first
        assert versions[1].description == "First backup"
    
    def test_data_manager_get_version_info(self, data_manager):
        """Test getting version info."""
        version_id = data_manager.create_backup("Test backup")
        
        version_info = data_manager.get_version_info(version_id)
        
        assert version_info is not None
        assert version_info.version_id == version_id
        assert version_info.description == "Test backup"
    
    def test_data_manager_get_backup_stats(self, data_manager):
        """Test getting backup stats."""
        data_manager.create_backup("Test backup 1")
        data_manager.create_backup("Test backup 2")
        
        stats = data_manager.get_backup_stats()
        
        assert stats["total_versions"] == 2
        assert stats["max_versions"] == 10
        assert stats["backup_directory"] == "test_backups"



