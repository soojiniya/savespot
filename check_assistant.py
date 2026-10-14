"""SaveSpot AI 질문 기능 점검용 스크립트 (프로젝트 기능 코드와는 별개입니다).

사용법 (프로젝트 폴더에서, FastAPI 서버가 http://127.0.0.1:8000 에서 실행 중일 때):
    python check_assistant.py

질문마다 응답 모드, 호출된 tool, 답변을 출력하고 규칙 위반이 의심되면 [주의]로 표시합니다.
- 금지 표현은 '단정하는 문장'에서만 잡습니다. "0개라는 뜻은 아닙니다"처럼 부정하는 문장은 통과입니다.
- 개념 질문은 tool 없이 답해야 하고, 지역 TOP 값 같은 관계없는 데이터가 붙으면 안 됩니다.
- 0.9109… 같은 내부 원값, 'CCTV N개/N대' 표현, 전체 지역 수 없는 순위 표기를 잡습니다.
API 키는 출력하지 않습니다.
"""
import json
import re
import urllib.request

API = "http://127.0.0.1:8000/api/assistant/query"

# (질문, 개념 질문 여부: True면 tool 호출 없이 답해야 함)
QUESTIONS = [
    ("속초시 어린이 교통안전 상태 알려줘", False),
    ("AI 상대적 고위험 지역 TOP5 알려줘", False),
    ("모델에서 중요한 변수는 뭐야?", False),
    ("시설 데이터가 없는 지역은 시설이 0개야?", True),
    ("사고다발지역에 없으면 안전한 지역이야?", True),
    ("AI 점수가 0.9면 사고가 날 확률이 90%야?", True),
    ("정선군 시설 현황 알려줘", False),  # 49개 시설분석 지역
    ("속초시 시설 현황 알려줘", False),  # 49개 밖 지역
]

# 문장 안에서 이런 단정이 나오면 위반 의심. 같은 문장에 부정 표현이 있으면 통과.
ASSERTION_FLAGS = [
    (r"확률(이|은|는)?\s*\d+(\.\d+)?\s*%", "AI 점수를 확률(%)로 표현"),
    (r"사고(가)?\s*(날|발생할)\s*확률", "AI 점수를 사고 발생 확률로 표현"),
    (r"시설(이|은|는)?\s*0\s*개", "시설 미제공을 0개로 표현"),
    (r"안전한\s*(지역|곳)", "사고다발지역 미선정을 안전으로 표현"),
    (r"공식\s*(교통안전\s*|지역\s*평가\s*)?순위", "AI 순위를 공식 순위로 표현"),
    (r"(공식\s*)?정책\s*우선\s*순위", "개선 검토 점수를 정책 우선순위로 표현"),
]
NEGATION = re.compile(
    r"아닙니다|아니에요|아니다|아니며|아니고|아니라|아닌|아님|아니요|않습니다|않아요|않으며|않고|않는|"
    r"없습니다(?=.*(뜻|의미))|뜻은\s*아|의미는\s*아|의미하지\s*않|뜻하지\s*않|"
    r"판단하지\s*않|단정하지\s*않|해석하지\s*않|금지"
)

# 문장과 상관없이 나오면 바로 위반 의심인 표현
ALWAYS_FLAGS = [
    (r"CCTV(?:\s*(?:설치)?\s*(?:수|대수|개수|장비))?(?:은|는|이|가)?\s*[:：]?\s*\d[\d,]*\s*(개|대)", "CCTV를 장비 개수/대수로 표현"),
    (r"(?<![\d.])0\.\d{4,}", "내부 원값(소수 4자리 이상) 노출"),
]
RANK_WITHOUT_TOTAL = re.compile(r"(?:AI\s*)?순위(?:는|가)?\s*[:：]?\s*(\d{1,3})위")
RANK_WITH_TOTAL = re.compile(r"\d+개\s*지역\s*중\s*\d{1,3}위")
TOP_REGION_HINTS = ["부산광역시 중구", "양양군", "서귀포시", "영덕군", "태백시"]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+| / ", text)
    return [p.strip() for p in parts if p.strip()]


def check(question: str, concept: bool, response: dict) -> list[str]:
    answer = response.get("answer") or ""
    meta = response.get("metadata") or {}
    issues: list[str] = []

    for sentence in sentences(answer):
        negated = bool(NEGATION.search(sentence))
        for pattern, label in ASSERTION_FLAGS:
            if re.search(pattern, sentence) and not negated:
                issues.append(f"{label}: “{sentence[:60]}”")

    for pattern, label in ALWAYS_FLAGS:
        if re.search(pattern, answer):
            issues.append(label)

    for match in RANK_WITHOUT_TOTAL.finditer(answer):
        window = answer[max(0, match.start() - 12) : match.end()]
        if not RANK_WITH_TOTAL.search(window):
            issues.append(f"순위에 전체 지역 수 없음: “{match.group(0)}”")

    if concept:
        tools = meta.get("tool_names") or []
        if tools:
            issues.append(f"개념 질문인데 tool 호출됨: {tools}")
        if any(hint in answer for hint in TOP_REGION_HINTS):
            issues.append("개념 질문 답변에 관계없는 지역 데이터 포함")

    if meta.get("mode") == "provider_error":
        issues.append("OpenAI 호출 오류 (서버 로그 확인)")
    return issues


def ask(question: str) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as res:
        return json.load(res)


if __name__ == "__main__":
    failed = 0
    for question, concept in QUESTIONS:
        response = ask(question)
        meta = response.get("metadata") or {}
        issues = check(question, concept, response)
        failed += bool(issues)
        print("=" * 80)
        print("질문 :", question)
        print("모드 :", meta.get("mode"), "| tool:", meta.get("tool_names") or [])
        print("출처 :", response.get("sources"))
        print("답변 :", response.get("answer"))
        print("판정 :", "[주의] " + " / ".join(issues) if issues else "통과")
    print("=" * 80)
    print(f"결과: {len(QUESTIONS) - failed}/{len(QUESTIONS)} 통과")
