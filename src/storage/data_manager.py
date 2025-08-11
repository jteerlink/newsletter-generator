"""
Data Management System

This module provides comprehensive data management capabilities including
versioning, backup/restore, data migration, and monitoring.
"""

import hashlib
import json
import logging
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import StorageConfig, StorageProvider, StorageStats
from .memory_store import MemoryStorageProvider

logger = logging.getLogger(__name__)


class DataVersion:
    """Represents a version of the data."""

    def __init__(self, version_id: str, timestamp: datetime,
                 description: str = "", metadata: Dict[str, Any] = None):
        self.version_id = version_id
        self.timestamp = timestamp
        self.description = description
        self.metadata = metadata or {}
        self.checksum = None
        self.size_bytes = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        version = cls(
            version_id=data["version_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        version.checksum = data.get("checksum")
        version.size_bytes = data.get("size_bytes", 0)
        return version


class DataManager:
    """Manages data versioning, backup, and restore operations."""

    def __init__(self, storage_provider: StorageProvider,
                 backup_dir: str = "./data/backups",
                 max_versions: int = 10):
        self.storage_provider = storage_provider
        self.backup_dir = Path(backup_dir)
        self.max_versions = max_versions
        self.versions_file = self.backup_dir / "versions.json"
        self.versions: List[DataVersion] = []
        self.backup_lock = threading.Lock()

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Load existing versions
        self._load_versions()

    def _load_versions(self):
        """Load existing version information."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    self.versions = [
                        DataVersion.from_dict(v) for v in data.get(
                            "versions", [])]
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
                self.versions = []
        else:
            self.versions = []

    def _save_versions(self):
        """Save version information to disk."""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump({
                    "versions": [v.to_dict() for v in self.versions]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")

    def create_backup(self, description: str = "",
                      metadata: Dict[str, Any] = None) -> str:
        """Create a backup of the current data."""
        with self.backup_lock:
            try:
                # Generate version ID
                version_id = f"backup_{
                    datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

                # Create backup using storage provider
                if not self.storage_provider.backup():
                    raise Exception("Storage provider backup failed")

                # For memory storage providers, create a simple backup file
                if isinstance(self.storage_provider, MemoryStorageProvider):
                    backup_file = self.backup_dir / \
                        f"memory_backup_{version_id}.json"
                    backup_data = {
                        "version_id": version_id,
                        "timestamp": datetime.now().isoformat(),
                        "description": description,
                        "documents_count": len(
                            self.storage_provider.documents),
                        "storage_type": "memory"}
                    with open(backup_file, 'w') as f:
                        json.dump(backup_data, f, indent=2)

                    # Create version record
                    version = DataVersion(
                        version_id=version_id,
                        timestamp=datetime.now(),
                        description=description,
                        metadata=metadata or {}
                    )

                    # Calculate checksum and size
                    version.checksum = self._calculate_checksum(backup_file)
                    version.size_bytes = self._calculate_size(backup_file)

                    # Add to versions list
                    self.versions.append(version)

                    # Clean up old versions
                    self._cleanup_old_versions()

                    # Save version information
                    self._save_versions()

                    logger.info(f"Created memory backup {
                                version_id} at {backup_file}")
                    return version_id
                else:
                    # Get backup path (assuming it's the most recent backup)
                    db_path_name = Path(
                        self.storage_provider.config.db_path).name
                    backup_paths = list(
                        self.backup_dir.glob(
                            f"{db_path_name}_backup_*"))
                    if not backup_paths:
                        raise Exception(
                            "No backup created by storage provider")

                    latest_backup = max(
                        backup_paths, key=lambda p: p.stat().st_mtime)

                    # Create version record
                    version = DataVersion(
                        version_id=version_id,
                        timestamp=datetime.now(),
                        description=description,
                        metadata=metadata or {}
                    )

                    # Calculate checksum and size
                    version.checksum = self._calculate_checksum(latest_backup)
                    version.size_bytes = self._calculate_size(latest_backup)

                    # Add to versions list
                    self.versions.append(version)

                    # Clean up old versions
                    self._cleanup_old_versions()

                    # Save version information
                    self._save_versions()

                    logger.info(
                        f"Created backup {version_id} at {latest_backup}")
                    return version_id

            except Exception as e:
                logger.error(f"Backup creation failed: {e}")
                raise

    def restore_backup(self, version_id: str) -> bool:
        """Restore from a specific backup version."""
        with self.backup_lock:
            try:
                # Find the version
                version = next(
                    (v for v in self.versions if v.version_id == version_id), None)
                if not version:
                    raise Exception(f"Version {version_id} not found")

                # Find backup file
                db_path_name = Path(self.storage_provider.config.db_path).name
                backup_paths = list(
                    self.backup_dir.glob(
                        f"{db_path_name}_backup_*"))
                if not backup_paths:
                    raise Exception("No backup files found")

                # Find the backup that matches the version timestamp
                target_backup = None
                for backup_path in backup_paths:
                    backup_time = datetime.fromtimestamp(
                        backup_path.stat().st_mtime)
                    if abs(
                            (backup_time - version.timestamp).total_seconds()) < 60:  # Within 1 minute
                        target_backup = backup_path
                        break

                if not target_backup:
                    raise Exception(
                        f"Backup file for version {version_id} not found")

                # Verify checksum
                current_checksum = self._calculate_checksum(target_backup)
                if current_checksum != version.checksum:
                    raise Exception(
                        f"Checksum mismatch for version {version_id}")

                # Restore using storage provider
                if not self.storage_provider.restore(str(target_backup)):
                    raise Exception("Storage provider restore failed")

                logger.info(
                    f"Restored backup {version_id} from {target_backup}")
                return True

            except Exception as e:
                logger.error(f"Restore failed: {e}")
                return False

    def list_versions(self) -> List[DataVersion]:
        """List all available backup versions."""
        return sorted(self.versions, key=lambda v: v.timestamp, reverse=True)

    def get_version_info(self, version_id: str) -> Optional[DataVersion]:
        """Get information about a specific version."""
        return next(
            (v for v in self.versions if v.version_id == version_id),
            None)

    def delete_version(self, version_id: str) -> bool:
        """Delete a specific backup version."""
        try:
            version = self.get_version_info(version_id)
            if not version:
                return False

            # Find and delete backup file
            db_path_name = Path(self.storage_provider.config.db_path).name
            backup_paths = list(
                self.backup_dir.glob(
                    f"{db_path_name}_backup_*"))
            for backup_path in backup_paths:
                backup_time = datetime.fromtimestamp(
                    backup_path.stat().st_mtime)
                if abs((backup_time - version.timestamp).total_seconds()) < 60:
                    backup_path.unlink()
                    break

            # Remove from versions list
            self.versions = [
                v for v in self.versions if v.version_id != version_id]
            self._save_versions()

            logger.info(f"Deleted version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file or directory."""
        sha256_hash = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        else:
            # For directories, hash the contents
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of a file or directory."""
        if path.is_file():
            return path.stat().st_size
        else:
            return sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _cleanup_old_versions(self):
        """Remove old versions beyond the maximum limit."""
        if len(self.versions) > self.max_versions:
            # Sort by timestamp and keep the most recent
            self.versions.sort(key=lambda v: v.timestamp, reverse=True)
            old_versions = self.versions[self.max_versions:]

            for version in old_versions:
                self.delete_version(version.version_id)

    def export_backup(self, version_id: str, export_path: str) -> bool:
        """Export a backup to a compressed file."""
        try:
            version = self.get_version_info(version_id)
            if not version:
                return False

            # Find backup file
            db_path_name = Path(self.storage_provider.config.db_path).name
            backup_paths = list(
                self.backup_dir.glob(
                    f"{db_path_name}_backup_*"))
            target_backup = None
            for backup_path in backup_paths:
                backup_time = datetime.fromtimestamp(
                    backup_path.stat().st_mtime)
                if abs((backup_time - version.timestamp).total_seconds()) < 60:
                    target_backup = backup_path
                    break

            if not target_backup:
                return False

            # Create compressed export
            export_path = Path(export_path)
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if target_backup.is_file():
                    zipf.write(target_backup, target_backup.name)
                else:
                    for file_path in target_backup.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(target_backup)
                            zipf.write(file_path, arcname)

            logger.info(f"Exported backup {version_id} to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_backup(self, import_path: str, description: str = "") -> str:
        """Import a backup from a compressed file."""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                raise Exception(f"Import file {import_path} does not exist")

            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    zipf.extractall(temp_dir)

                # Find the backup directory
                temp_path = Path(temp_dir)
                db_path_name = Path(self.storage_provider.config.db_path).name
                backup_dirs = list(temp_path.glob(f"{db_path_name}_backup_*"))

                if not backup_dirs:
                    raise Exception("No valid backup found in import file")

                backup_dir = backup_dirs[0]

                # Create version record
                version_id = f"import_{
                    datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                version = DataVersion(
                    version_id=version_id,
                    timestamp=datetime.now(),
                    description=description
                )

                # Calculate checksum and size
                version.checksum = self._calculate_checksum(backup_dir)
                version.size_bytes = self._calculate_size(backup_dir)

                # Copy to backup directory
                target_backup = self.backup_dir / backup_dir.name
                shutil.copytree(backup_dir, target_backup)

                # Add to versions list
                self.versions.append(version)
                self._cleanup_old_versions()
                self._save_versions()

                logger.info(f"Imported backup {version_id} from {import_path}")
                return version_id

        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise

    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about backups."""
        total_size = sum(v.size_bytes for v in self.versions)
        oldest_backup = min(
            self.versions,
            key=lambda v: v.timestamp) if self.versions else None
        newest_backup = max(
            self.versions,
            key=lambda v: v.timestamp) if self.versions else None

        return {
            "total_versions": len(
                self.versions),
            "total_size_bytes": total_size,
            "oldest_backup": oldest_backup.timestamp.isoformat() if oldest_backup else None,
            "newest_backup": newest_backup.timestamp.isoformat() if newest_backup else None,
            "backup_directory": str(
                self.backup_dir),
            "max_versions": self.max_versions}
