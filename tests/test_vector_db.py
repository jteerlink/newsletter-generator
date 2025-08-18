import os
import shutil
import tempfile

import pytest

from src.core.utils import chunk_text, embed_chunks
from src.storage import add_text_to_db, get_db_collection


def test_chunk_text_basic():
    text = "This is a test. " * 100
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 1


def test_embed_chunks_basic():
    chunks = ["This is a test chunk.", "Another chunk of text."]
    embeddings = embed_chunks(chunks)
    assert hasattr(embeddings, "__len__")
    assert len(embeddings) == len(chunks)


def test_add_text_to_db_creates_chunks():
    text = "This is a test. " * 100
    # Use a temp directory for ChromaDB
    with tempfile.TemporaryDirectory() as tmpdir:
        collection = get_db_collection(path=tmpdir, name="test_collection")
        add_text_to_db(text, collection=collection, chunk_size=50, chunk_overlap=10)
        # Check that documents were added
        results = collection.get()
        assert "documents" in results
        assert len(results["documents"]) > 1 