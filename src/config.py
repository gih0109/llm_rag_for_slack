from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml


DEFAURT_CONFIG_PATH = Path("./test_config.yaml")


def _load_yaml(yaml_path: str) -> dict:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError("config yaml 파일을 찾을 수 없습니다.")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = config if config is not None else {}
    return config



cfg = _load_yaml(DEFAURT_CONFIG_PATH)

paths_cfg = cfg.get("paths", {})
slack_cfg = cfg.get("slack", {})
openai_cfg = cfg.get("openai", {})
postgresql_cfg = cfg.get("postgresql", {})
embedding_cfg = cfg.get("embedding", {})
chunk_cfg = cfg.get("chunk", {})

# 경로
# DATA_DIR = paths_cfg.get("data_dir")
# RAW_DIR = paths_cfg.get("raw_dir")
# CHUNK_DIR = paths_cfg.get("chunk_dir")
CHANNEL_NAME = paths_cfg.get("channel_name")
SLACK_JSON_PATH = paths_cfg.get("slack_json_path")

# slack
SLACK_BOT_TOKEN = slack_cfg.get("bot_token")
SLACK_USER_TOKEN = slack_cfg.get("user_token")

# openai
OPENAI_API_KEY = openai_cfg.get("openai_api_key")
RELEVANT_JUDGE_MODEL = openai_cfg.get("relevant_judge_model")
TAG_MODEL = openai_cfg.get("tag_model")
TEMPERATURE = openai_cfg.get("temperature")

# postgresql
user_name = postgresql_cfg.get("user_name")
password = postgresql_cfg.get("password")
host = postgresql_cfg.get("host")
port = postgresql_cfg.get("port")
db_name = postgresql_cfg.get("db_name")
PG_CONN = f"postgresql+psycopg://{user_name}:{password}@{host}:{port}/{db_name}"
PG_URL = f"postgresql://{user_name}:{password}@{host}:{port}/{db_name}"

# embedding
EMB_MODEL_NAME = embedding_cfg.get("emb_model_name")
DEVICE = embedding_cfg.get("device")

# chunk
MERGE_MSG = chunk_cfg.get("merge_short_msg")
MERGE_MSG_LEN = chunk_cfg.get("merge_msg_len")
MERGE_MSG_WINDOW = chunk_cfg.get("merge_msg_window_sec")




