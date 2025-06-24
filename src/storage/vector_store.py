# src/storage/vector_store.py
import chromadb
from chromadb.config import Settings

def setup_chroma():
    client = chromadb.PersistentClient(
        path="./data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    return client
