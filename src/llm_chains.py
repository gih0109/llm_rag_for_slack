from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from typing import List, Dict, Any, Literal


class JudgeReleventMsgOutput(BaseModel):
    related: bool = Field(..., description="현재 메시지가 이전 메시지와 같은 대화 주제를 계속하면 True, 아니면 False")


JUDGE_RELEVANT_SYS_TEMPLATE = """당신은 대화 관련성 이진 분류기입니다. 
내부적으로 어떤 추론을 하더라도, 출력은 반드시 제공된 JSON 스키마에 “정확히” 일치해야 합니다. 
어떠한 설명/사유/마크다운/여분 텍스트도 출력하지 마십시오. 오직 JSON 한 객체만 출력하십시오.
"""

JUDGE_RELEVANT_USER_TEMPLATE  = """당신의 과업: 두 메시지가 같은 대화 주제를 “계속”하는지 판별하십시오.

[정의]
- 관련 있음(true): 현재 메시지가 이전 메시지의 주제/요청/이슈에 “직접적으로” 이어지거나 그 처리·확인·보완에 해당함
- 관련 없음(false): 새로운 주제로 전환되었거나, 이전 메시지와 의미상 연결을 합리적으로 확정하기 어려움

[판정 규칙]
- 관련 있음에 해당하는 대표 패턴
  1) 후속 질문/답변/정정/추가 정보/상태 업데이트
  2) 동일 작업·이슈·엔티티(문서/링크/파일/사람/티켓/PR/커밋/에러코드 등) 재참조
  3) 이전 지시/요청/제안에 대한 확인·수락·거절·일정·진행 보고
  4) 지시어/대명사(그거/이거/위 내용 등)로 자연스럽게 이어짐
  5) 이전에 요청/언급된 링크·파일을 실제로 첨부·공유함
  6) 동일 문제(버그/로그/오류 재현 등)의 연속 대응

- 관련 없음에 해당하는 대표 패턴
  1) 주제가 바뀐 새 질문/공지/잡담
  2) 무관한 새 링크/파일/밈 공유
  3) 일반적 안부/수다/스몰톡만 있는 경우
  4) 이전 메시지와 연결 고리가 불명확한 단독 감탄사

[경계 상황 처리]
- “네/넵/확인했습니다/감사합니다” 등 짧은 수락·확인은 **직전 메시지가 지시/요청/답변일 때**만 관련 있음(true)
- “이거/그거/위 내용” 등 지시어가 이전 메시지를 합리적으로 가리키면 관련 있음(true)
- 이모지/반응 텍스트(예: 👍)만 있어도 이전 메시지에 대한 동의·확인 의도로 읽히면 관련 있음(true)
- 링크/파일만 있는 메시지도 이전에 요청되었거나 그 이슈에 부속이면 관련 있음(true)
- 모호하면 관련 없음(false)

[판정 우선순위]
1) 명시적 연결 표현(확인/수락/후속 보고) > 2) 동일 엔티티·리소스 재언급 > 3) 언어적 암시(지시어·맥락)
- 여전히 불명확하면 false로 판정

[출력 형식]
- 스키마: {{"related": true|false}}
- 오직 위 JSON 한 객체만 출력(여분 텍스트/따옴표/설명 금지)

[참고 예시 — 출력 금지]
- 예시1 (관련 있음): 
  이전: "PR-123 오늘 검토 부탁드립니다." / 현재: "네, 오늘 안에 확인하겠습니다." → {{"related": true}}
- 예시2 (관련 없음):
  이전: "DB 마이그레이션 일정 공유합니다." / 현재: "점심 뭐 드실래요?" → {{"related": false}}
- 예시3 (관련 있음, 지시어):
  이전: "어제 보낸 오류 로그(NRE) 원인 분석 부탁." / 현재: "그거 재현했고 패치 올렸습니다." → {{"related": true}}
- 예시4 (관련 있음, 링크만):
  이전: "기획안 링크 공유해 주세요." / 현재: "https://example.com/plan" → {{"related": true}}
- 예시5 (관련 있음, 링크만):
  이전: "공유하신 구글 링크 참고하겠습니다." / 현재: "https://notion.com/plan" → {{"related": false}}
- 예시6 (관련 없음, 단독 감탄):
  이전: "오늘 날씨 좋네요." / 현재: "감사합니다!" → {{"related": false}}
- 예시7 (관련 있음, 반응):
  이전: "배포 완료했습니다." / 현재: "👍" → {{"related": true}}

[입력]
이전 메시지: {prev_msg}
현재 메시지: {cur_msg}

[출력]
오직 JSON 한 객체만 출력하십시오.
"""
judge_relevant_msg_prompt = ChatPromptTemplate([
    ("system", JUDGE_RELEVANT_SYS_TEMPLATE),
    ("user", JUDGE_RELEVANT_USER_TEMPLATE),
])


class TagList(BaseModel):
    tags: List[str]


TAGS_SYSTEM_PROMPT_V4 = """
당신은 슬랙 스레드의 핵심 주제를 뽑는 태그 추출기다.

목표: 스레드에서 다루는 핵심 주제를 "태그"로 추출하라.
규칙:
- 본문에 실제로 등장한 용어를 우선 사용하되, 의미가 같은 변형은 하나로 통합.
- 제품/라이브러리/서비스명, 오류코드, 정책/권한명, 결정/작업 항목을 우선.
- 일반어/사담(예: 감사합니다, 오늘, 확인요)은 제외.
- 고유명사는 원형/대소문자 보존, 그 외는 소문자 단어 1~3개로 간결하게.
- 중요도 높은 것부터 내림차순으로 정렬.
출력: JSON 배열만 반환. 예: ["rag 평가", "slack 토큰 스코프", "im:history 권한", "dm 수집 자동화"]

스레드:
{thread}
"""
tag_thread_prompt = ChatPromptTemplate.from_template(TAGS_SYSTEM_PROMPT_V4)


def build_judge_relevant_llm_chain(
        llm_model_name: str, 
        llm_model_temp: float,
        llm_timeout: int = 61,
        llm_max_retry: int = 6,
    ):

    llm = ChatOpenAI(
        model=llm_model_name,
        temperature=llm_model_temp,
        timeout=llm_timeout,
        max_retries=llm_max_retry,
    )

    chain = judge_relevant_msg_prompt | llm.with_structured_output(JudgeReleventMsgOutput)
    return chain


def build_tag_llm_chain(
        # prompt,
        # model_provider: Literal["openai", "google"], 
        llm_model_name: str, 
        llm_model_temp: float,
        llm_timeout: int = 61,
        llm_max_retry: int = 6,
    ):

    llm = ChatOpenAI(
        model=llm_model_name,
        temperature=llm_model_temp,
        timeout=llm_timeout,
        max_retries=llm_max_retry,
    )

    chain = tag_thread_prompt | llm.with_structured_output(TagList)
    return chain
