from google.genai import types

from chroma_config import get_chroma_collection
from config import get_genai_client

MODEL_NAME = "gemini-embedding-001"
query = "Come migliorare nel tempo?"


def main() -> None:
    collection = get_chroma_collection()
    collection_size = collection.count()
    if collection_size == 0:
        raise RuntimeError("La collection è vuota. Crea prima gli embedding dei documenti.")

    client = get_genai_client()
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Il modello non ha restituito un embedding per la query.")

    results = collection.query(
        query_embeddings=[list(response.embeddings[0].values)],
        # n_results=min(3, collection_size),
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    for index, (document, metadata, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        print(f"Risultato {index} (distanza: {distance:.4f})")
        print(f"Metadati: {metadata}")
        print(document)
        print("-" * 30)


if __name__ == "__main__":
    main()
