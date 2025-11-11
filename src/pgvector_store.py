from langchain_core.documents import Document
from langchain_postgres import PGVector
from typing import List, Dict, Any

from .config import PG_CONN, PG_URL
from .embeddings import get_embeddings


def make_pgvector(embedding, collection_name: str, user_jsonb: bool = False):
    return PGVector(
        connection=PG_CONN,
        embeddings=embedding,
        collection_name=collection_name,
        use_jsonb=user_jsonb,
    )


    