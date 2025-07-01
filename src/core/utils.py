from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text using RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to chunk.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def embed_chunks(chunks: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Embed a list of text chunks using a sentence-transformers model.

    Args:
        chunks (List[str]): List of text chunks.
        model_name (str): Name of the sentence-transformers model.

    Returns:
        List[List[float]]: List of embeddings.
    """
    model = SentenceTransformer(model_name)
    return model.encode(chunks, show_progress_bar=True) 