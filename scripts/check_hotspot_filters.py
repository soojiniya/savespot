"""현황지도 시도/시군구 필터 점검 (읽기 전용, 데이터 수정 없음).

사용법 (프로젝트 폴더에서, FastAPI 서버가 http://127.0.0.1:8000 에서 실행 중일 때):
    python scripts/check_hotspot_filters.py

MongoDB `hotspots_map` 문서를 직접 세어(파이썬에서 시도명 정규화) `/api/hotspots`
응답 건수와 비교합니다. 접속 정보는 출력하지 않습니다.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.db import get_database  # noqa: E402
from backend.main import canonical_sido, canonical_sigungu  # noqa: E402

API = "http://127.0.0.1:8000"
SIDOS = ["경기도", "서울특별시", "충청남도"]


def api(path: str, **params: str) -> dict:
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as res:
        return json.load(res)


docs = list(get_database()["hotspots_map"].find({}, {"_id": 0, "sido": 1, "sigungu": 1, "source_type": 1}))
by_sido = Counter(canonical_sido(d.get("sido"), d.get("sigungu")) for d in docs)
by_pair = Counter((canonical_sido(d.get("sido"), d.get("sigungu")), canonical_sigungu(d.get("sigungu"))) for d in docs)
print(f"MongoDB hotspots_map 전체: {len(docs)}건  /  API 전체: {api('/api/hotspots', limit='1')['pagination']['total']}건")
print("원본 시도 표기:", ", ".join(f"{k}({v})" for k, v in Counter(d.get('sido') for d in docs).most_common()))

options = api("/api/hotspots/filter-options")
failed = 0
for sido in SIDOS:
    api_total = api("/api/hotspots", sido=sido, limit="1")["pagination"]["total"]
    ok = api_total == by_sido[sido] and api_total > 0
    failed += not ok
    print(f"[{'OK' if ok else '불일치'}] {sido}: MongoDB {by_sido[sido]}건 / API {api_total}건")
    sigungu = (options["sigungus_by_sido"].get(sido) or [None])[0]
    if sigungu:
        api_sgg = api("/api/hotspots", sido=sido, sigungu=sigungu, limit="1")["pagination"]["total"]
        ok = api_sgg == by_pair[(sido, sigungu)] and api_sgg > 0
        failed += not ok
        print(f"   [{'OK' if ok else '불일치'}] {sido} {sigungu}: MongoDB {by_pair[(sido, sigungu)]}건 / API {api_sgg}건")

empty = [s for s in options["sidos"] if api("/api/hotspots", sido=s, limit="1")["pagination"]["total"] == 0]
failed += bool(empty)
print("필터 옵션 중 0건인 시도:", empty or "없음")
print("결과:", "모두 일치" if not failed else f"{failed}건 확인 필요")
