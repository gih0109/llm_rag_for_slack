from pathlib import Path
import os
from typing import Optional, Literal

from src.config import (
    # RAW_DIR, 
    CHANNEL_NAME,
    SLACK_JSON_PATH,
    OPENAI_API_KEY, 
    RELEVANT_JUDGE_MODEL, 
    TAG_MODEL, 
    TEMPERATURE,
    MERGE_MSG,
    MERGE_MSG_LEN,
    MERGE_MSG_WINDOW,
    EMB_MODEL_NAME,
    DEVICE,
)

from src.slack_json2chunk import iter_chunks_from_json
from src.llm_chains import build_judge_relevant_llm_chain, build_tag_llm_chain
from src.pgvector_store import make_pgvector, add_documents_stream_pgvector


def _ensure_openai_key():
    if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def make_slack_pgvector(
        raw_json_path,
        channel_name,
        merge_short_msg: bool,
        merge_msg_len: int,
        merge_msg_window_sec: int,
        judge_chain_model_name: str,
        tag_chain_model_name: str,
        llm_temperature: float,
        emb_model_name: str,
        emb_device: Literal["cpu", "cuda"],
        pgvector_use_jsonb: bool = True
    ):
    # build chain
    judge_chain = build_judge_relevant_llm_chain(
        llm_model_name=judge_chain_model_name,
        llm_model_temp=llm_temperature,
    )
    tag_chain = build_tag_llm_chain(
        llm_model_name=tag_chain_model_name,
        llm_model_temp=llm_temperature,
    )
    print("building chain success")

    # pgvector
    relevant_vectorstore = make_pgvector(
        emb_model_name=emb_model_name,
        emb_model_device=emb_device,
        collection_name=channel_name,
        user_jsonb=pgvector_use_jsonb,
    )
    tag_vectorstore = make_pgvector(
        emb_model_name=emb_model_name,
        emb_model_device=emb_device,
        collection_name="tag",
        user_jsonb=pgvector_use_jsonb,
    )
    print("pgvector connected")

    # make chunk iterator
    chunk_iter = iter_chunks_from_json(
        path=raw_json_path,
        default_channel=channel_name,
        merge_short_msg=merge_short_msg,
        merge_window_sec=merge_msg_window_sec,
        merge_msg_len=merge_msg_len,
    )

    print("starting add document stream")
    # save data to pgvetor
    add_documents_stream_pgvector(
        judge_relevant_chain=judge_chain,
        tag_chain=tag_chain,
        channel_name=channel_name,
        chunk_iter=chunk_iter,
        relevant_vectorstore=relevant_vectorstore,
        tag_vectorstore=tag_vectorstore
    )

    print("finished")


def main():
    _ensure_openai_key()

    make_slack_pgvector(
        raw_json_path=SLACK_JSON_PATH,
        channel_name=CHANNEL_NAME,
        merge_short_msg=MERGE_MSG,
        merge_msg_len=MERGE_MSG_LEN,
        merge_msg_window_sec=MERGE_MSG_WINDOW,
        judge_chain_model_name=RELEVANT_JUDGE_MODEL,
        tag_chain_model_name=TAG_MODEL,
        emb_model_name=EMB_MODEL_NAME,
        emb_device=DEVICE,
        llm_temperature=TEMPERATURE,
    )

    
if __name__ == "__main__":
    main()

