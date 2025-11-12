# llm_rag_for_slack

Slack에서 수집한 메시지 JSON을 LLM 기반 RAG 파이프라인에 적재하기 위한 **최소 동작 코드** 입니다.
현재 버전은 **JSON → 청크화 → 주제 단절(관련성 판별) → 태그 추출 → Document 생성 → (선택) PGVector 적재** 흐름을 중심으로 구성되어 있습니다.


## 주요 기능

- **Slack JSON 로더/청크러**: `src/slack_json2chunk.py`
  - `load_message(...)`: Slack JSON에서 스레드 단위로 메시지를 읽어들입니다.
  - `iter_chunks_from_json(...)`: 짧은 메시지를 시간 창(`merge_msg_window_sec`) 내에서 병합하여 **의미 단위 청크**를 만듭니다.
  - 최종 산출물은 `{"text": ..., "metadata": {...}}` 형태의 스트림으로 제공됩니다.
- **LLM 체인**: `src/llm_chains.py`
  - **관련성 판별 체인**: 연속된 메시지가 같은 주제를 계속하는지 이진 분류합니다.
  - **태그 추출 체인**: 스레드/청크의 핵심 태그를 도출합니다.
  - `ChatOpenAI`를 사용하며, Pydantic을 통한 **구조적 출력**을 사용하도록 설계되어 있습니다.
- **임베딩 & 벡터스토어 어댑터**:
  - `src/embeddings.py`: `HuggingFaceEmbeddings` 기반 임베딩 로더 (예: `BAAI/bge-m3`)
  - `src/pgvector_store.py`: LangChain용 `PGVector` 초기화 헬퍼와 적재 스텁
- **실행 엔트리포인트**: `main.py`
  - `config.yaml` + 환경변수 기반으로 파이프라인을 구동하는 스캐폴드가 포함되어 있습니다.

> 참고: `src/slack_fetch.py`는 현재 비어 있습니다. Slack API로 직접 수집하는 모듈은 후속 구현 대상입니다.


## 디렉터리 구조

```text
.
├─ config.yaml                # 프로젝트 설정
├─ main.py                    # 엔트리포인트 (파이프라인 실행)
├─ src/
│  ├─ config.py               # 설정 로더 (YAML/ENV → 상수)
│  ├─ embeddings.py           # 임베딩 로더 (HuggingFace)
│  ├─ llm_chains.py           # 관련성 판별/태그 추출 체인
│  ├─ pgvector_store.py       # PGVector 초기화/적재 스텁
│  └─ slack_json2chunk.py     # JSON 로더/청크러
├─ .gitignore                 # .env 등 민감정보 제외
└─ README.md                  # 이 문서
```

## Requrirements

- python >= 3.13
- Packages
```
pip install \
  "langchain>=0.2" "langchain-openai>=0.2" "langchain-community>=0.2" \
  "langchain-postgres>=0.0.9" "psycopg[binary]" \
  "pydantic>=2.0" python-dotenv orjson \
  transformers sentence-transformers
```
- postgreSQL 서버

## 빠른 시작

1. 가상환경 & 필요 라이브러리 설치

2. `.env` 설정
    - 루트에 `.env` 를 두고 아래처럼 설정합니다
    ```
    OPENAI_API_KEY=sk-xxxxxxxx...
    ```

3. `config.yaml` 작성

    아래 예시를 참고해 config.yaml 을 작성합니다.
    ```yaml
    paths:
      slack_json_path: "./data/slack_messages.json"  # Slack JSON 경로
      channel_name: "project_mvp"

    openai:
      relevant_judge_model: "gpt-4o-mini"
      tag_model: "gpt-4o-mini"
      temperature: 0.0

    embedding:
      emb_model_name: "BAAI/bge-m3"
      device: "cpu"  # 또는 "cuda"

    database:
      host: "localhost"
      port: 5432
      db_name: "dev"
      user_name: "postgres"
      password: "postgres"
      collection_name: "slack_threads"
      use_jsonb: true

    chunk:
      merge_short_msg: true
      merge_msg_len: 200
      merge_msg_window_sec: 300
    ```

4. Slack JSON 준비

- 현 버전은 Slack API 로부터 JSON 을 수집하는 기능을 개발중입니다.

- 다음 JSON 형태를 입력으로 사용합니다.
    ```json
    {
    "text": "넵 감사합니다!",
    "metadata": {
        "ts": "1761712119.795709",
        "thread_ts": "1761711626.506369",
        "client_msg_id": "ae20548d-9fab-4576-aa9d-9d5f26bd5205",
        "user": "U095DKTM19U",
        "channel": "project_mvp",
        "is_dm": false,
        "datetime": "2025-10-29T13:28:39.795709+09:00",
        "reactions": {},
        "links": [],
        "files": [],
        "reply_count": null,
        "parent_ts": "1761711626.506369"
    }
    }
    ```


5. 실행

    ```
    python main.py
    ```
