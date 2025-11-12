from langchain_core.documents import Document
from langchain_postgres import PGVector
from typing import List, Dict, Any, Iterable

from .config import PG_CONN, PG_URL
from .embeddings import get_embeddings


def make_pgvector(
        emb_model_name: str, 
        emb_model_device: str, 
        collection_name: str, 
        user_jsonb: bool = False
    ):
    """
    pgvector 생성기
    """
    embeddings = get_embeddings(emb_model_name, device=emb_model_device)
    return PGVector(
        connection=PG_CONN,
        embeddings=embeddings,
        collection_name=collection_name,
        use_jsonb=user_jsonb,
    )


def _format_thread_text(msg_list: list):
    """
    스레드 메세지 텍스트를 한줄로 formatting
    예시: [datetime] user: text
    """
    lines = []
    for text, metadata in msg_list:
        datetime = metadata.get("datetime")
        user = metadata.get("user", "unknown_user")
        lines.append(f"[{datetime}] {user}: {text}")
        
    return "\n\n".join(lines)


def _format_thread_metadata(metadata: str):
    """
    Chroma 는 metadata 값으로 str/int/float/bool/None 값만 지원하므로 
    metadata 내 dict, list 값을 str/int/float/bool/None 으로 변환
    """
    ALLOWED_TYPE = (str, int, float, bool, type(None))
    result = {}
    for key, val in metadata.items():
        if key == "reactions":
            result[key] = ",".join(f"{react_key}*{react_val}" for react_key, react_val in val.items())

        elif isinstance(val, ALLOWED_TYPE):
            result[key] = val

        elif isinstance(val, (list, tuple, set)):
            result[key] = ",".join(map(str, val))

        else:
            result[key] = str(val)
    return result


def add_documents_stream_pgvector(
        judge_relevant_chain,
        tag_chain,
        channel_name: str,
        chunk_iter: Iterable[Any],
        relevant_vectorstore: PGVector | None = None,
        tag_vectorstore: PGVector | None = None,
    ) -> None:
    """
    chunk iter로부터 slack message chunk 를 1건식 받아 이전 message 와 관련성을 확인 후
    관련있을 시 이전 message 와 합쳐 메세지 그룹으로 묶는다. 관련없을 시 이전 chunk 를 relevant_vectorstore 에 적재한다.
    또한 관련 메세지 그룹에서 tag 를 추출하여 tag_vectorstore 에 적재한다.

    Args:
        - judge_relevant_chain : 이전 메세지와 현재 메세지의 관련성 확인 langchain llm chain
        - tag_chain : 현재 메세지 묶음에서 주요 태그를 추출하는 langchain llm chain
        - chunk_iter (Iterable) : slack message chunk iterator
        - relevant_vectorstore (PGVector) : 관련 메세지 그룹을 저장하기 위한 postgreSQL Vectorstore
        - tag_vectorstore (PGVector) : 관련 메세지 그룹의 tag 를 저장하기 위한 postgreSQL Vectorstore
    """

    prev_thread_ts_list = []
    relevant_msg_list = []

    for data in chunk_iter:
        text, metadata = data["text"], data["metadata"]
        thread_id = metadata["thread_ts"] if "thread_ts" in metadata else metadata.get("ts", "")

        # relevant 판별
        if relevant_msg_list and thread_id and (thread_id not in prev_thread_ts_list):
            # 이전 스레드와 현재 스레드 관련성 판별
            relevant_text = _format_thread_text(relevant_msg_list)
            relevant_cond = judge_relevant_chain.invoke({
                "prev_msg": relevant_text,
                "cur_msg": text
            }).related # True or False

            if relevant_cond is False:
                # tag 추출
                tag_list = tag_chain.invoke({"thread": relevant_text}).tags

                doc_metadata = metadata
                doc_metadata["tag"] = ",".join(tag_list)

                primary_key = f"{channel_name}|{prev_thread_ts_list[0]}"
                # thread_db[primary_key] = relevant_text
                
                # make docs
                doc_metadata = _format_thread_metadata(doc_metadata)
                relevant_doc = Document(page_content=relevant_text, metadata=doc_metadata)
                tag_doc = Document(page_content=",".join(tag_list), metadata={"primary_key": primary_key})
                
                # relevant_vectorstore 에 저장
                if relevant_vectorstore is not None:
                    r_key = f"doc|{primary_key}"
                    relevant_vectorstore.add_documents(documents=[relevant_doc], ids=[r_key])
                if tag_vectorstore is not None:
                    t_key = f"tag|{primary_key}"
                    tag_vectorstore.add_documents(documents=[tag_doc], ids=[t_key])

                # 초기화
                relevant_msg_list = [(text, metadata)]
                # prev_thread_id 갱신
                prev_thread_ts_list = [thread_id]

            else:
                relevant_msg_list.append((text, metadata))
                prev_thread_ts_list.append(thread_id)

        else:
            relevant_msg_list.append((text, metadata))
            if len(prev_thread_ts_list) == 0:
                prev_thread_ts_list = [thread_id]

