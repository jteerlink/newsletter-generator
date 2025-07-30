"""Simple CLI stub for ingesting documents into the vector store (Phase 2)."""

from __future__ import annotations

import argparse
import sys
import os

from src.storage import get_db_collection, add_text_to_db


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    collection = get_db_collection()
    add_text_to_db(text, collection=collection)
    print(f"Ingested {file_path} into the vector database.")


if __name__ == "__main__":
    # Inspect the contents of the default vector DB
    from src.storage import get_db_collection
    collection = get_db_collection()
    print("\n=== Vector DB Collection Info ===")
    print(f"Collection name: {collection.name}")
    print(f"Number of documents: {collection.count()}")
    results = collection.get()
    for i, doc in enumerate(results.get("documents", [])):
        print(f"\nDocument {i+1}:")
        print(f"ID: {results['ids'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Document: {doc[:200]}{'...' if len(doc) > 200 else ''}")
    main()
