"""ChromaDB helper utilities (Phase 2)."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
import uuid
from src.core.utils import chunk_text, embed_chunks

# Default DB path and collection name
DEFAULT_DB_PATH = "./data/chroma_db"
DEFAULT_COLLECTION_NAME = "newsletter_content"  # Unified collection name for all ingestion and queries


def get_db_collection(path=DEFAULT_DB_PATH, name=DEFAULT_COLLECTION_NAME):
    """
    Return a persistent ChromaDB collection object.

    Args:
        path (str): Path to the ChromaDB directory.
        name (str): Name of the collection.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=name)
    return collection


def add_text_to_db(text, collection=None, chunk_size=1000, chunk_overlap=100):
    """
    Chunk the text, embed each chunk, and add to the ChromaDB collection.

    Args:
        text (str): The text to ingest.
        collection: The ChromaDB collection object.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    if collection is None:
        collection = get_db_collection()
    # Chunk the text using the shared utility
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print("No chunks to add.")
        return
    # Embed all chunks using the shared utility
    embeddings = embed_chunks(chunks)
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": "ingest.py"}] * len(chunks),
    )
    print(f"Added {len(chunks)} chunks to the vector database.")
