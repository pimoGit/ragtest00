from pathlib import Path

import chromadb

CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
CHROMA_COLLECTION_NAME = "document_chunks"


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
    )
