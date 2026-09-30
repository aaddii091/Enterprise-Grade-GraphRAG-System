"""Qdrant + embeddings - the vector half of retrieval.

LlamaIndex owns this layer: HuggingFaceEmbedding for vectors, SentenceSplitter for
chunking, QdrantVectorStore for persistence. The collection itself is created
explicitly so the dimension and distance metric are ours, not inferred.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src import config

log = logging.getLogger(__name__)

_client: QdrantClient | None = None
_embed_model: Any = None
_vector_store: Any = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    return _client


def verify() -> None:
    try:
        client().get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {config.QDRANT_URL}.\n"
            "  Start it:  docker compose up -d qdrant\n"
            f"  Detail:    {exc}"
        ) from exc


def embed_model():
    """Lazily load bge-small (~130 MB resident).

    Deliberately lazy: --status and --reset must not pay the model load cost.
    """
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        log.info("Loading embedding model %s ...", config.EMBED_MODEL)
        _embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    return _embed_model


def ensure_collection() -> None:
    """Create the collection with our dimension/metric if it does not exist."""
    existing = {c.name for c in client().get_collections().collections}
    if config.QDRANT_COLLECTION in existing:
        return
    log.info(
        "Creating Qdrant collection %s (dim=%d, cosine)",
        config.QDRANT_COLLECTION,
        config.EMBED_DIM,
    )
    client().create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=config.EMBED_DIM, distance=qmodels.Distance.COSINE
        ),
    )


def vector_store():
    global _vector_store
    if _vector_store is None:
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        ensure_collection()
        _vector_store = QdrantVectorStore(
            client=client(), collection_name=config.QDRANT_COLLECTION
        )
    return _vector_store


def splitter():
    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )


def build_node(chunk_id: str, text: str, metadata: dict[str, Any]):
    """A TextNode with a deterministic ID and its embedding already attached."""
    from llama_index.core.schema import TextNode

    node = TextNode(id_=chunk_id, text=text, metadata=metadata)
    node.embedding = embed_model().get_text_embedding(text)
    return node


def add_nodes(nodes: list[Any]) -> None:
    if nodes:
        vector_store().add(nodes)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Top-k passages for a natural-language query."""
    from llama_index.core.vector_stores import VectorStoreQuery

    k = top_k if top_k is not None else config.TOP_K
    embedding = embed_model().get_query_embedding(query)
    result = vector_store().query(
        VectorStoreQuery(query_embedding=embedding, similarity_top_k=k)
    )

    hits: list[dict[str, Any]] = []
    for i, node in enumerate(result.nodes or []):
        score = None
        if result.similarities and i < len(result.similarities):
            score = result.similarities[i]
        meta = node.metadata or {}
        hits.append(
            {
                "chunk_id": node.node_id,
                "text": node.get_content(),
                "score": score,
                "source": meta.get("filename", meta.get("source", "unknown")),
                "page_no": meta.get("page_no"),
            }
        )
    return hits


def fetch_by_ids(chunk_ids: list[str]) -> list[dict[str, Any]]:
    """Retrieve specific chunks by ID - used to expand graph hits into passages."""
    if not chunk_ids:
        return []
    try:
        points = client().retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=chunk_ids,
            with_payload=True,
        )
    except Exception as exc:
        log.debug("fetch_by_ids failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        out.append(
            {
                "chunk_id": str(point.id),
                "text": payload.get("text", ""),
                "score": None,
                "source": payload.get("filename", payload.get("source", "unknown")),
                "page_no": payload.get("page_no"),
            }
        )
    return out


def count() -> int:
    try:
        return int(client().count(config.QDRANT_COLLECTION, exact=True).count)
    except Exception:
        return 0


def wipe() -> None:
    """Drop the collection. Used by --reset."""
    global _vector_store
    try:
        client().delete_collection(config.QDRANT_COLLECTION)
    except Exception as exc:
        log.debug("delete_collection: %s", exc)
    _vector_store = None
