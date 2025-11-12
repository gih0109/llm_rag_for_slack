import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

try:
    import orjson as json
except:
    import json


def load_message(slack_json_path, reverse=True):
    """
    json 파일을 읽고 각 쓰레드를 yield 로 반환

    Args:
        - slack_json_path (str): slack json 파일 경로
        - json 읽을 시 역순으로 읽을지 결정 (시간순 정렬 확인 필요)
    """
    with open(slack_json_path, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
    
    r = -1 if reverse is True else 1
    for message in json_data[::r]:
        yield message


def ts2kst(ts: str, timezone_str: str = "Asia/Seoul") -> str:
    """
    slack 의 ts 를 YY-MM-DD HH-MM-SS 형태로 변환

    Args:
        - ts (str): slack ts
        - timezone (str): 시간대 위치, default="Asia/Seoul"
    Returns:
        - UTC time (str): UTC + timezone "{YYYY-MM-DD} {HH:MM:SS} {timezone}"
    """
    ts = float(ts)
    datetime_kst = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ZoneInfo(timezone_str))
    return datetime_kst.strftime("%Y-%m-%d %H:%M:%S %Z")


def extract_links(text: str):
    """<https://url|title> 패턴 및 일반 URL 간단 추출"""
    links = []
    for m in re.finditer(r"<(https?://[^>|]+)\|([^>]+)>", text or ""):
        links.append({"title": m.group(2), "url": m.group(1)})
    for m in re.finditer(r"(?<!<)(https?://\S+)", text or ""):
        url = m.group(1).rstrip(").,]}")
        links.append({"title": None, "url": url})
    # 중복 제거
    seen = set(); uniq = []
    for l in links:
        if l["url"] not in seen:
            uniq.append(l); seen.add(l["url"])
    return uniq


def covert_message_to_dict(raw, default_channel = None) -> dict:
    """
    slack json raw 정보를 dictionary 로 변환
    """
    text = raw.get("text", "") or ""
    ts = raw.get("ts") or raw.get("event_ts") or ""
    thread_ts = raw.get("thread_ts") or ts
    user = raw.get("user") or raw.get("username") or raw.get("bot_id") or "unknown"
    channel = raw.get("channel") or default_channel
    client_msg_id = raw.get("client_msg_id")
    reactions_raw = raw.get("reactions") or []
    reactions = {r.get("name"): r.get("count", 1) for r in reactions_raw if r.get("name")}
    files = []
    for f in raw.get("files") or []:
        files.append({
            "name": f.get("name"),
            "url": f.get("url_private") or f.get("permalink"),
            "mimetype": f.get("mimetype"),
        })
    links = extract_links(text)
    data = {
        # "text": text,
        "ts": ts,
        "thread_ts": thread_ts,
        "client_msg_id": client_msg_id,
        "user": user,
        "channel": channel,
        "is_dm": bool(channel and channel.startswith("D")),
        "datetime": ts2kst(ts) if ts else None,
        "reactions": reactions,
        "links": links,
        "files": files,
        "reply_count": raw.get("reply_count"),
        "parent_ts": thread_ts if thread_ts != ts else None,
    }
    return (text, data)


def process_message(raw, default_channel: str | None = None):
    """
    slack thread 내 메세지 및 정보를 dictionary 로 변환 후 list 로 return
    """
    data_list = [covert_message_to_dict(raw, default_channel)]
    # thread 있을 시
    if len(raw.get("thread")) > 0:
        for thread_raw in raw["thread"]:
            data = covert_message_to_dict(thread_raw, default_channel)
            data_list.append(data)

    return data_list



def iter_chunks_from_json(
        path: str,
        default_channel: str | None = None, 
        merge_short_msg: bool = False, 
        merge_window_sec: int | None = 300, 
        merge_msg_len: int = 20
    ):
    """
    JSON에서 메시지 단위 청킹.
    merge_short=True면 같은 사용자 & merge_window_sec 이내 merge_msg_len 이하 단문을 이전 메시지에 붙인다.

    Args:
        - path (str) : slack message json 경로
        - default_channel (Optional, str) : 현재 채널, default=None
        - merge_short_msg (bool) : 단문을 이전 메세지에 붙일지 결정, default=False
        - merge_window_sec (int) : 단문을 이전 메세지에 붙일 시 임계 시간(초), default=300
        - merge_msg_len (int) : 단문 판별 문자열 길이

    Returns:
        - message (str) : 메세지 텍스트
        - metadata (dict) : metadata
        -
    """
    prev = None  # (text, meta)
    for raw in load_message(slack_json_path=path, reverse=True):
        # prev = None 
        for data in process_message(raw, default_channel=default_channel):
            text, metadata = data
            text = text.strip()
            if not text:
                continue

            if merge_short_msg and len(text) < merge_msg_len and prev:
                ts_cur = float(metadata.get("ts", "0"))
                ts_prev = float(prev["metadata"].get("ts", "0"))

                cond = (
                    metadata["user"] == prev["metadata"]["user"]
                    # and metadata["thread_ts"] == prev["metadata"]["thread_ts"]
                    and abs(ts_cur - ts_prev) <= merge_window_sec
                )

                if cond:
                    prev["text"] = prev["text"] + "\n" + text
                    continue
            
            if prev:
                yield prev
            
            prev = {"text": text, "metadata": metadata}

    if prev:
        yield prev

