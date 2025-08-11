"""
Migration Tools

This module provides tools for migrating data between different storage providers
and handling data format conversions.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import DataType, DocumentMetadata, StorageConfig, StorageProvider

logger = logging.getLogger(__name__)


class MigrationPlan:
    """Represents a migration plan between storage providers."""

    def __init__(self, source_provider: StorageProvider,
                 target_provider: StorageProvider,
                 migration_id: str = None):
        self.source_provider = source_provider
        self.target_provider = target_provider
        self.migration_id = migration_id or f"migration_{
            datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.total_documents = 0
        self.migrated_documents = 0
        self.failed_documents = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "source_provider": self.source_provider.__class__.__name__,
            "target_provider": self.target_provider.__class__.__name__,
            "total_documents": self.total_documents,
            "migrated_documents": self.migrated_documents,
            "failed_documents": self.failed_documents,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "errors": self.errors}


class MigrationManager:
    """Manages data migration between storage providers."""

    def __init__(self, migrations_dir: str = "./data/migrations"):
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.migration_history: List[MigrationPlan] = []
        self._load_migration_history()

    def _load_migration_history(self):
        """Load migration history from disk."""
        history_file = self.migrations_dir / "migration_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.migration_history = [
                        MigrationPlan(
                            source_provider=None,  # Will be set during migration
                            target_provider=None,
                            migration_id=item["migration_id"]
                        ) for item in data.get("migrations", [])
                    ]
            except Exception as e:
                logger.error(f"Failed to load migration history: {e}")
                self.migration_history = []

    def _save_migration_history(self):
        """Save migration history to disk."""
        history_file = self.migrations_dir / "migration_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump({
                    "migrations": [m.to_dict() for m in self.migration_history]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save migration history: {e}")

    def create_migration_plan(self,
                              source_provider: StorageProvider,
                              target_provider: StorageProvider,
                              filters: Optional[Dict[str,
                                                     Any]] = None) -> MigrationPlan:
        """Create a migration plan between two storage providers."""
        plan = MigrationPlan(source_provider, target_provider)

        # Estimate total documents
        try:
            source_stats = source_provider.get_stats()
            plan.total_documents = source_stats.total_documents
        except Exception as e:
            logger.warning(f"Could not get source stats: {e}")
            plan.total_documents = 0

        return plan

    def execute_migration(
            self,
            plan: MigrationPlan,
            batch_size: int = 100,
            progress_callback: Optional[Callable] = None) -> bool:
        """Execute a migration plan."""
        plan.start_time = datetime.now()

        try:
            logger.info(f"Starting migration {plan.migration_id}")

            # Initialize target provider if needed
            if not plan.target_provider.initialize():
                raise Exception("Failed to initialize target provider")

            # Get all documents from source
            all_documents = self._get_all_documents(plan.source_provider)
            plan.total_documents = len(all_documents)

            # Process documents in batches
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]

                for doc_content, doc_metadata in batch:
                    try:
                        # Add to target provider
                        plan.target_provider.add_document(
                            doc_content, doc_metadata)
                        plan.migrated_documents += 1

                        if progress_callback:
                            progress = (
                                plan.migrated_documents / plan.total_documents) * 100
                            progress_callback(
                                progress, plan.migrated_documents, plan.total_documents)

                    except Exception as e:
                        plan.failed_documents += 1
                        error_msg = f"Failed to migrate document {
                            doc_metadata.doc_id}: {e}"
                        plan.errors.append(error_msg)
                        logger.error(error_msg)

                # Log progress
                logger.info(
                    f"Migrated {plan.migrated_documents}/{plan.total_documents} documents")

            plan.end_time = datetime.now()

            # Save to history
            self.migration_history.append(plan)
            self._save_migration_history()

            logger.info(
                f"Migration {
                    plan.migration_id} completed: " f"{
                    plan.migrated_documents} successful, {
                    plan.failed_documents} failed")

            return plan.failed_documents == 0

        except Exception as e:
            plan.end_time = datetime.now()
            plan.errors.append(f"Migration failed: {e}")
            logger.error(f"Migration {plan.migration_id} failed: {e}")
            return False

    def _get_all_documents(self, provider: StorageProvider) -> List[tuple]:
        """Get all documents from a provider."""
        documents = []

        try:
            # This is a simplified approach - in practice, you'd need to implement
            # a method to iterate through all documents in the provider
            # For now, we'll use the migration method if available
            if hasattr(provider, 'migrate'):
                # Use the provider's own migration method
                temp_provider = DummyStorageProvider()
                provider.migrate(temp_provider)
                documents = temp_provider.get_all_documents()
            else:
                logger.warning(
                    "Provider does not support document enumeration")

        except Exception as e:
            logger.error(f"Failed to get documents from provider: {e}")

        return documents

    def validate_migration(self, plan: MigrationPlan) -> Dict[str, Any]:
        """Validate a completed migration."""
        validation_result = {
            "migration_id": plan.migration_id,
            "source_documents": 0,
            "target_documents": 0,
            "validation_passed": False,
            "errors": []
        }

        try:
            # Get document counts
            source_stats = plan.source_provider.get_stats()
            target_stats = plan.target_provider.get_stats()

            validation_result["source_documents"] = source_stats.total_documents
            validation_result["target_documents"] = target_stats.total_documents

            # Basic validation
            if plan.migrated_documents == plan.total_documents:
                validation_result["validation_passed"] = True
            else:
                validation_result["errors"].append(
                    f"Document count mismatch: migrated {
                        plan.migrated_documents}, " f"expected {
                        plan.total_documents}")

            # Check for failed documents
            if plan.failed_documents > 0:
                validation_result["errors"].append(
                    f"{plan.failed_documents} documents failed to migrate"
                )

        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {e}")

        return validation_result

    def rollback_migration(self, plan: MigrationPlan) -> bool:
        """Rollback a migration by clearing the target provider."""
        try:
            logger.info(f"Rolling back migration {plan.migration_id}")

            # Clear target provider (this would need to be implemented by each
            # provider)
            if hasattr(plan.target_provider, 'clear_all'):
                plan.target_provider.clear_all()
            else:
                logger.warning(
                    "Target provider does not support clear_all operation")

            # Remove from history
            self.migration_history = [
                m for m in self.migration_history if m.migration_id != plan.migration_id]
            self._save_migration_history()

            logger.info(
                f"Rollback completed for migration {
                    plan.migration_id}")
            return True

        except Exception as e:
            logger.error(
                f"Rollback failed for migration {
                    plan.migration_id}: {e}")
            return False

    def get_migration_history(self) -> List[MigrationPlan]:
        """Get migration history."""
        return sorted(
            self.migration_history,
            key=lambda m: m.start_time or datetime.min,
            reverse=True)

    def get_migration_stats(self) -> Dict[str, Any]:
        """Get overall migration statistics."""
        total_migrations = len(self.migration_history)
        successful_migrations = sum(
            1 for m in self.migration_history if m.failed_documents == 0)
        total_documents_migrated = sum(
            m.migrated_documents for m in self.migration_history)
        total_documents_failed = sum(
            m.failed_documents for m in self.migration_history)

        return {
            "total_migrations": total_migrations,
            "successful_migrations": successful_migrations,
            "failed_migrations": total_migrations -
            successful_migrations,
            "total_documents_migrated": total_documents_migrated,
            "total_documents_failed": total_documents_failed,
            "success_rate": (
                successful_migrations /
                total_migrations *
                100) if total_migrations > 0 else 0}


class DummyStorageProvider:
    """Dummy storage provider for migration testing."""

    def __init__(self):
        self.documents = []

    def add_document(self, content: str, metadata: DocumentMetadata) -> str:
        """Add document to dummy storage."""
        self.documents.append((content, metadata))
        return metadata.doc_id

    def get_all_documents(self) -> List[tuple]:
        """Get all documents from dummy storage."""
        return self.documents.copy()


class DataFormatConverter:
    """Converts data between different formats."""

    @staticmethod
    def convert_metadata_format(old_metadata: Dict[str, Any],
                                target_format: str) -> DocumentMetadata:
        """Convert metadata to the unified format."""
        # Extract common fields
        doc_id = old_metadata.get("doc_id") or old_metadata.get(
            "id") or str(uuid.uuid4())
        title = old_metadata.get(
            "title") or old_metadata.get("name") or "Untitled"
        source = old_metadata.get(
            "source") or old_metadata.get("url") or "unknown"

        # Parse timestamp
        timestamp_str = old_metadata.get(
            "timestamp") or old_metadata.get("date")
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
            except BaseException:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Parse content type
        content_type_str = old_metadata.get(
            "content_type") or old_metadata.get("type") or "text"
        try:
            content_type = DataType(content_type_str)
        except ValueError:
            content_type = DataType.TEXT

        # Extract tags
        tags = old_metadata.get("tags") or old_metadata.get("categories") or []
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        return DocumentMetadata(
            doc_id=doc_id,
            title=title,
            source=source,
            content_type=content_type,
            timestamp=timestamp,
            author=old_metadata.get("author"),
            tags=tags,
            version=old_metadata.get("version", "1.0"),
            checksum=old_metadata.get("checksum")
        )

    @staticmethod
    def convert_to_legacy_format(metadata: DocumentMetadata) -> Dict[str, Any]:
        """Convert unified metadata to legacy format."""
        return {
            "doc_id": metadata.doc_id,
            "title": metadata.title,
            "source": metadata.source,
            "content_type": metadata.content_type.value,
            "timestamp": metadata.timestamp.isoformat(),
            "author": metadata.author,
            "tags": metadata.tags,
            "version": metadata.version,
            "checksum": metadata.checksum
        }
