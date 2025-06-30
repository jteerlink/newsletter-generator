"""Simple CLI stub for ingesting documents into the vector store (Phase 2)."""
from __future__ import annotations

import argparse
import sys

from vector_db import get_db_collection, add_text_to_db


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Ingest a text file into the vector DB")
    parser.add_argument("file", help="Path to the text/markdown file to ingest")
    args = parser.parse_args()

    path = args.file
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        sys.stderr.write(f"File not found: {path}\n")
        sys.exit(1)

    collection = get_db_collection()
    add_text_to_db(text, collection)
    print("Ingestion complete (stub).")


if __name__ == "__main__":
    main() 