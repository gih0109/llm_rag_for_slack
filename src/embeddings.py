from typing import Any, Literal
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(emb_model_name: str, device: Literal["cpu", "cuda"]):
    return HuggingFaceEmbeddings(
        model_name=emb_model_name,
        model_kwargs={"device": device}, # CUDA면 "cuda"
        encode_kwargs={"normalize_embeddings": True}
    )

