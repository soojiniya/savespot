"""Natural-language assistant orchestration for child traffic safety data."""

from __future__ import annotations

import json
import re
from typing import Any

from backend.services.assistant_tools import (
    TOOLS,
    all_ai_regions,
    compare_regions,
    get_facility_analysis,
    get_high_risk_top,
    get_improvement_candidates,
    get_model_importance,
    get_model_info,
    get_region_analysis,
    match_regions,
)
from backend.db import get_database
from backend.services.llm_client import LlmClient


SYSTEM_PROMPT = """
너는 어린이 교통안전 프로젝트 전용 데이터 분석 어시스턴트다.
반드시 tool을 통해 조회한 프로젝트 데이터만 근거로 답한다.
모르는 정보는 추측하지 않는다.
AI 상대적 고위험 점수는 실제 사고 발생 가능성 수치가 아니다.
AI 순위는 공식 지역 평가 순위가 아니다.
시설 데이터가 없는 지역은 시설이 없는 것이 아니다.
사고다발지역 선정지점이 없다고 안전한 지역을 의미하지 않는다.
전국 AI 분석은 229개 지역, 시설 상세분석은 데이터가 확보된 49개 지역 기준이다.
개선 검토 점수는 공식 정책 우선순위가 아니라 분석용 heuristic이다.
시설 설치 시 사고가 몇 % 감소하는지 임의로 예측하지 않는다.
인과관계를 주장하지 않는다.
데이터 범위가 부족하면 그 사실을 답변에 명시한다.
가능하면 숫자와 분석 범위를 함께 설명한다.
한국어로 간결하게 답한다.
""".strip()


FINAL_ANSWER_PROMPT = """
최종 답변은 반드시 tool 결과의 실제 값만 사용한다.
지역명, 순위, 사고건수, 사고율, AI 상대적 고위험 점수 등 핵심 수치를 tool 결과와 같은 값으로 포함한다.
predicted_high_risk_probability_2026는 실제 사고 확률이 아니라 AI 상대적 고위험 점수로만 설명한다.
없는 값은 만들지 말고, 데이터 범위 제한을 짧게 덧붙인다.
tool 결과에 표시용 문자열(예: "91.1점", "37.3", "229개 지역 중 25위", "CCTV 설치 보호구역 16곳 (설치율 88.9%)")이 있으면 그 문자열을 그대로 쓴다.
AI 상대적 고위험 점수는 0~100점, 소수점 1자리로만 쓰고 0.9109 같은 내부 원값을 쓰지 않는다.
사고율은 소수점 1자리로 쓴다. 순위는 "229개 지역 중 25위"처럼 전체 지역 수와 함께 쓴다.
CCTV 값은 CCTV 장비 대수가 아니라 CCTV가 설치된 어린이보호구역 수다. 반드시 "CCTV 설치 보호구역 N곳 (설치율 X%)"로 쓰고 "CCTV N개", "CCTV N대"라고 쓰지 않는다.
""".strip()


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_region_analysis",
            "description": "229개 AI 결과 기준 지역 단위 사고/AI 분석 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"region_name": {"type": "string"}},
                "required": ["region_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_regions",
            "description": "2~4개 지역의 229개 공통 지표를 비교합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4,
                    }
                },
                "required": ["region_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_high_risk_top",
            "description": "AI 상대적 고위험 점수 상위 지역을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facility_analysis",
            "description": "49개 시설 상세분석 가능 지역의 시설 현황을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"region_name": {"type": "string"}},
                "required": ["region_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_improvement_candidates",
            "description": "49개 지역 기준 개선 검토 후보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "모델 종류, target, 분석 범위, feature 목록과 metrics를 조회합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_importance",
            "description": "Permutation importance 변수 중요도를 조회합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def score_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    score = value * 100 if value <= 1 else value
    return f"{score:.1f}점"


def number_text(value: Any, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}{suffix}"
    return f"{int(value):,}{suffix}"


def rate_text(value: Any) -> str:
    """사고율: 항상 소수점 1자리."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:,.1f}"


def rank_text(rank: Any, total: int | None = None) -> str:
    """순위: '229개 지역 중 25위'."""
    if not isinstance(rank, (int, float)):
        return "—"
    total = total or len(all_ai_regions())
    return f"{total}개 지역 중 {int(rank)}위"


def cctv_text(zone_count: Any, install_rate: Any) -> str:
    """CCTV 값은 CCTV가 설치된 어린이보호구역 수 (장비 대수 아님)."""
    if not isinstance(zone_count, (int, float)):
        return "CCTV 설치 보호구역 —"
    rate = f" (설치율 {pct_text(install_rate)})" if isinstance(install_rate, (int, float)) else ""
    return f"CCTV 설치 보호구역 {int(zone_count):,}곳{rate}"


def hotspot_counts() -> dict[str, int] | None:
    """Selected-hotspot counts read from the same collection the map uses."""
    try:
        collection = get_database()["hotspots_map"]
        counts = {
            source: collection.count_documents({"source_type": source})
            for source in ("schoolzone_hotspot", "child_pedestrian_hotspot")
        }
    except Exception:
        return None
    counts["total"] = counts["schoolzone_hotspot"] + counts["child_pedestrian_hotspot"]
    return counts if counts["total"] else None


def pct_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"


def importance_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.3f}"


def extract_limit(question: str, default: int = 5) -> int:
    match = re.search(r"(\d+)\s*(곳|개|위|명|top|TOP)?", question)
    if not match:
        return default
    return max(1, min(int(match.group(1)), 20))


def extract_region_candidates(question: str) -> list[str]:
    normalized = re.sub(r"(와|과|랑|하고|및|,|/)", " ", question)
    token_matches: list[str] = []
    for token in re.findall(r"[가-힣]+(?:시|군|구)", normalized):
        for region in match_regions(token):
            if region not in token_matches:
                token_matches.append(region)
    if token_matches:
        return token_matches

    compact = re.sub(r"\s+", "", question)
    matches: list[str] = []
    for size in range(min(12, len(compact)), 1, -1):
        for start in range(0, len(compact) - size + 1):
            piece = compact[start : start + size]
            if not any(token in piece for token in ("시", "군", "구")):
                continue
            for region in match_regions(piece):
                if region not in matches:
                    matches.append(region)
    return matches


def extract_region_inputs(question: str) -> list[str]:
    normalized = re.sub(r"(와|과|랑|하고|및|,|/)", " ", question)
    raw_tokens = re.findall(r"[가-힣]+(?:시|군|구)", normalized)
    inputs: list[str] = []
    for token in raw_tokens:
        matched = match_regions(token)
        if matched:
            for region in matched:
                if region not in inputs:
                    inputs.append(region)
        elif token not in inputs:
            inputs.append(token)
    return inputs or extract_region_candidates(question)


def actions_for(region: str | None = None) -> list[dict[str, Any]]:
    if not region:
        return []
    return [
        {"type": "open_region", "label": "지역분석 보기", "region": region},
        {"type": "open_map", "label": "지도에서 보기", "region": region},
    ]


def topic_particle(text: str) -> str:
    if not text:
        return "은"
    last = text[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3:
        return "은" if (code - 0xAC00) % 28 else "는"
    return "은"


def interpretation_answer(question: str) -> dict[str, Any] | None:
    """Concept questions answered from the project's interpretation rules only.

    No tool is called, so unrelated region data (e.g. the TOP 1 region) is never
    attached. Returns None when the question is not a pure concept question.
    """
    q = question.strip()
    lower = q.lower()
    has_region = bool(extract_region_inputs(q))

    if "확률" in q or ("%" in q and ("점수" in q or "ai" in lower)):
        return {
            "answer": "아니요. AI 상대적 고위험 점수는 실제 사고 발생 가능성 수치가 아닙니다. 전국 229개 시군구 안에서 다음 연도 상대적 고위험 신호를 비교하기 위한 0~100점의 분석 점수이며, 전국 AI 순위도 공식 지역 평가 순위가 아닙니다.",
            "intent": "interpretation",
            "region": None,
            "sources": ["model_target_definition"],
            "actions": [{"type": "open_guide", "label": "데이터안내 보기"}],
        }

    if ("사고다발" in q or "선정" in q) and "안전" in q:
        counts = hotspot_counts()
        scope = (
            f"공식 기준으로 선정된 {counts['total']:,}개 지점(어린이보호구역 {counts['schoolzone_hotspot']:,}개 + 보행어린이 {counts['child_pedestrian_hotspot']:,}개)"
            if counts
            else "공식 기준으로 선정된 지점"
        )
        return {
            "answer": f"아니요. 사고다발지역 선정지점은 {scope}이며, 모든 사고 지점이나 전체 사고건수를 뜻하지 않습니다. 선정지점이 없다고 해서 안전한 지역이라는 의미는 아닙니다.",
            "intent": "interpretation",
            "region": None,
            "sources": ["hotspots_map"],
            "actions": [{"type": "open_guide", "label": "데이터안내 보기"}],
        }

    if "시설" in q and ("0개" in q or "없" in q or "미제공" in q) and not has_region:
        return {
            "answer": "시설 상세분석은 횡단보도·신호등·어린이보호구역·CCTV 4종 시설 데이터가 함께 확보된 49개 지역에만 제공됩니다. 시설 데이터가 없는 지역은 시설이 0개라는 뜻이 아니라 현재 상세 시설 데이터 제공 대상이 아니라는 의미입니다.",
            "intent": "interpretation",
            "region": None,
            "sources": ["eda_main_49"],
            "actions": [{"type": "open_guide", "label": "데이터안내 보기"}],
        }

    return None


def fallback_answer(question: str) -> dict[str, Any]:
    q = question.strip()
    limit = extract_limit(q)
    lower = q.lower()
    regions = extract_region_candidates(q)
    region_inputs = extract_region_inputs(q)

    interpretation = interpretation_answer(q)
    if interpretation:
        return interpretation

    if "중요" in q and ("변수" in q or "feature" in lower):
        data = get_model_importance()
        top = data["items"][:5]
        lines = [
            f"{i + 1}. {item.get('feature')}: {importance_text(item.get('importance_mean'))}"
            for i, item in enumerate(top)
        ]
        return {
            "answer": "모델 변수 중요도는 permutation importance 기준입니다.\n" + "\n".join(lines),
            "intent": "model_importance",
            "region": None,
            "sources": [data["source"]],
            "actions": [{"type": "open_ai", "label": "AI 분석 보기"}],
        }

    if "모델" in q:
        data = get_model_info()
        return {
            "answer": f"현재 최종 모델은 {data['final_model']}입니다. Target은 “{data['target_definition']}”입니다. 분석 단위는 지역 단위이며, 전국 229개 AI 분석과 49개 시설 상세분석은 범위가 다릅니다.",
            "intent": "model_info",
            "region": None,
            "sources": [data["source"]],
            "actions": [{"type": "open_ai", "label": "AI 분석 보기"}],
        }

    if "개선" in q and ("후보" in q or "검토" in q or "top" in lower):
        data = get_improvement_candidates(limit)
        lines = [
            f"{item.get('improvement_priority_rank')}. {item.get('sido_sgg_nm')} — 개선 검토 점수 {score_text(item.get('improvement_priority_score'))}"
            for item in data["items"]
        ]
        return {
            "answer": "49개 시설분석 지역 기준 개선 검토 후보입니다. 공식 정책 우선순위는 아닙니다.\n" + "\n".join(lines),
            "intent": "improvement_candidates",
            "region": None,
            "sources": [data["source"]],
            "actions": [{"type": "open_ai", "label": "AI 분석 보기"}],
        }

    if ("top" in lower or "높" in q or "상위" in q) and ("AI" in q or "위험" in q or "점수" in q):
        data = get_high_risk_top(limit)
        lines = [
            f"{item.get('risk_rank_2026')}. {item.get('sido_sgg_nm')} — AI 상대적 고위험 점수 {score_text(item.get('predicted_high_risk_probability_2026'))}"
            for item in data["items"]
        ]
        return {
            "answer": f"전국 229개 지역 기준 AI 예측 기반 TOP {len(data['items'])}입니다. 점수는 실제 사고 발생 가능성 수치가 아니며, 순위는 공식 지역 평가 순위가 아닙니다.\n" + "\n".join(lines),
            "intent": "high_risk_top",
            "region": None,
            "sources": [data["source"]],
            "actions": [{"type": "open_ai", "label": "AI 분석 보기"}],
        }

    if len(region_inputs) >= 2 or "비교" in q:
        data = compare_regions(region_inputs[:4])
        if data["items"]:
            lines = [
                f"- {item['region']}: 사고 {number_text(item.get('accident_count'), '건')}, 사고율 {rate_text(item.get('accident_per_10k_child'))}, AI 상대적 고위험 점수 {score_text(item.get('predicted_high_risk_probability_2026'))}, 전국 AI 순위 {rank_text(item.get('risk_rank_2026'))}"
                for item in data["items"]
            ]
            if data.get("unresolved"):
                lines.append(
                    "확인하지 못한 지역: "
                    + ", ".join(item.get("input", "") for item in data["unresolved"])
                )
            return {
                "answer": "전국 229개 지역 공통지표 기준 비교입니다.\n" + "\n".join(lines),
                "intent": "region_compare",
                "region": None,
                "sources": [data["source"]],
                "actions": [{"type": "open_region", "label": "지역분석 보기", "region": data["items"][0]["region"]}],
            }

    if regions:
        region = regions[0]
        if "시설" in q or "cctv" in lower or "신호" in q or "횡단" in q:
            data = get_facility_analysis(region)
            if data["status"] == "not_available":
                return {
                    "answer": f"{data['region']}{topic_particle(data['region'])} 현재 49개 시설 상세분석 데이터 제공 대상이 아닙니다. 시설이 없는 것이 아니라 상세 시설 데이터가 제공되지 않는 상태입니다.",
                    "intent": "facility_analysis",
                    "region": data["region"],
                    "sources": [],
                    "actions": actions_for(data["region"]),
                }
            if data["status"] == "ok":
                return {
                    "answer": f"{data['region']} 시설 현황입니다. 횡단보도 {number_text(data.get('crosswalk'), '개')}, 신호등 {number_text(data.get('traffic_light'), '개')}, 어린이보호구역 {number_text(data.get('school_zone'), '개')}, {cctv_text(data.get('cctv'), data.get('cctv_install_rate'))}입니다. 시설 점수와 개선 검토 지표는 49개 지역 기준 분석용 지표입니다.",
                    "intent": "facility_analysis",
                    "region": data["region"],
                    "sources": [data["source"]],
                    "actions": actions_for(data["region"]),
                }
        data = get_region_analysis(region)
        if data["status"] == "ok":
            facility_note = (
                "시설 상세분석 데이터도 제공됩니다."
                if data["facility_data_available"]
                else "이 지역은 시설 상세분석 49개 지역에 포함되지 않습니다."
            )
            return {
                "answer": f"{data['region']}의 직전 연도 어린이 사고건수는 {number_text(data.get('accident_count'), '건')}, 부상자수는 {number_text(data.get('injury_count'), '명')}, 사망자수는 {number_text(data.get('death_count'), '명')}입니다. 어린이 1만 명당 사고율은 {rate_text(data.get('accident_per_10k_child'))}이고, AI 상대적 고위험 점수는 {score_text(data.get('predicted_high_risk_probability_2026'))}, 전국 AI 순위는 {rank_text(data.get('risk_rank_2026'))}입니다. {facility_note} AI 점수는 실제 사고 발생 가능성 수치가 아니며, 전국 AI 순위는 공식 지역 평가 순위가 아닙니다.",
                "intent": "region_analysis",
                "region": data["region"],
                "sources": [data["source"]],
                "actions": actions_for(data["region"]),
            }
        return {
            "answer": data.get("message", "현재 분석 데이터에서 해당 지역을 확인하지 못했습니다."),
            "intent": "region_analysis",
            "region": None,
            "sources": [],
            "actions": [],
            "metadata": data,
        }

    return {
        "answer": "현재 분석 데이터에서 질문 의도를 확인하지 못했습니다. 지역명, 비교할 지역, AI TOP, 개선 검토 후보, 모델 설명처럼 구체적으로 질문해 주세요.",
        "intent": "unknown",
        "region": None,
        "sources": [],
        "actions": [],
    }


def mark_fallback(result: dict[str, Any], reason: str) -> dict[str, Any]:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(
        {
            "mode": "fallback_tool_answers",
            "provider": "local",
            "fallback_reason": reason,
        }
    )
    result["metadata"] = metadata
    return result


def region_view(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("status") not in (None, "ok"):
        return item
    view = {
        "status": item.get("status", "ok"),
        "region": item.get("region") or item.get("sido_sgg_nm"),
        "accident_count": number_text(item.get("accident_count", item.get("prev_accident_count")), "건"),
        "injury_count": number_text(item.get("injury_count", item.get("prev_injury_count")), "명"),
        "death_count": number_text(item.get("death_count", item.get("prev_death_count")), "명"),
        "accident_per_10k_child": rate_text(item.get("accident_per_10k_child", item.get("prev_accident_per_10k_child"))),
        "ai_relative_risk_score": score_text(item.get("predicted_high_risk_probability_2026")),
        "national_ai_rank": rank_text(item.get("risk_rank_2026")),
        "score_note": "AI 상대적 고위험 점수(0~100점)이며 실제 사고 발생 가능성 수치가 아님. 순위는 공식 평가 순위가 아님.",
    }
    if "facility_data_available" in item:
        view["facility_data_available"] = item["facility_data_available"]
        facility = item.get("facility_summary")
        if isinstance(facility, dict):
            view["facility_summary"] = facility_view(
                {
                    "status": "ok",
                    "region": view["region"],
                    "crosswalk": facility.get("crosswalk_count"),
                    "traffic_light": facility.get("traffic_light_count"),
                    "school_zone": facility.get("schoolzone_count"),
                    "cctv": facility.get("cctv_installed_zone_count"),
                    "cctv_install_rate": facility.get("cctv_install_rate"),
                    "infrastructure_supply_score": facility.get("infrastructure_score"),
                    "infrastructure_deficiency": facility.get("infrastructure_deficiency_score"),
                    "improvement_priority_score": facility.get("improvement_priority_score"),
                    "improvement_priority_rank": facility.get("improvement_priority_rank"),
                }
            )
        else:
            view["facility_note"] = "49개 시설 상세분석 대상이 아님. 시설이 0개라는 뜻이 아니라 상세 시설 데이터 미제공."
    return view


def facility_view(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("status") != "ok":
        return item
    view = {
        "status": "ok",
        "region": item.get("region"),
        "facility_data_available": True,
        "crosswalk": number_text(item.get("crosswalk"), "개"),
        "traffic_light": number_text(item.get("traffic_light"), "개"),
        "school_zone": number_text(item.get("school_zone"), "개"),
        "cctv": cctv_text(item.get("cctv"), item.get("cctv_install_rate")),
        "infrastructure_supply_score": score_text(item.get("infrastructure_supply_score")),
        "infrastructure_deficiency": score_text(item.get("infrastructure_deficiency")),
        "scope_note": "49개 시설분석 지역 기준 분석용 지표",
    }
    if isinstance(item.get("improvement_priority_rank"), (int, float)):
        view["improvement_priority"] = (
            f"개선 검토 점수 {score_text(item.get('improvement_priority_score'))} · "
            f"49개 지역 중 {int(item['improvement_priority_rank'])}위 (공식 정책 우선순위 아님)"
        )
    return view


def llm_view(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Display-ready copy of a tool result for the LLM.

    Raw values stay inside the tools/API; the model only sees the strings the
    UI shows (0~100점 scores, 1-decimal rates, '229개 지역 중 N위', CCTV 보호구역 곳 수).
    """
    if not isinstance(result, dict):
        return result
    if name == "get_region_analysis":
        return region_view(result)
    if name == "compare_regions":
        return {**result, "items": [region_view(item) for item in result.get("items", [])]}
    if name == "get_facility_analysis":
        return facility_view(result)
    if name == "get_high_risk_top":
        return {
            "status": result.get("status"),
            "total": result.get("total"),
            "items": [
                {
                    "national_ai_rank": rank_text(row.get("risk_rank_2026"), result.get("total")),
                    "region": row.get("sido_sgg_nm"),
                    "ai_relative_risk_score": score_text(row.get("predicted_high_risk_probability_2026")),
                    "accident_count": number_text(row.get("prev_accident_count"), "건"),
                    "accident_per_10k_child": rate_text(row.get("prev_accident_per_10k_child")),
                }
                for row in result.get("items", [])
            ],
            "score_note": "AI 상대적 고위험 점수(0~100점)이며 실제 사고 발생 가능성 수치가 아님. 순위는 공식 평가 순위가 아님.",
        }
    if name == "get_improvement_candidates":
        return {
            "status": result.get("status"),
            "total": result.get("total"),
            "note": result.get("note"),
            "items": [
                {
                    "improvement_priority_rank": f"{result.get('total') or 49}개 지역 중 {row.get('improvement_priority_rank')}위",
                    "region": row.get("sido_sgg_nm"),
                    "improvement_priority_score": score_text(row.get("improvement_priority_score")),
                    "ai_relative_risk_score": score_text(row.get("predicted_high_risk_probability_2026")),
                    "infrastructure_deficiency": score_text(row.get("infrastructure_deficiency_score")),
                    "cctv_install_rate": pct_text(row.get("cctv_install_rate")),
                }
                for row in result.get("items", [])
            ],
        }
    if name == "get_model_importance":
        return {
            **result,
            "items": [
                {"feature": row.get("feature"), "importance_mean": importance_text(row.get("importance_mean"))}
                for row in result.get("items", [])
            ],
        }
    if name == "get_model_info":
        return {
            **result,
            "metrics": [
                {
                    key: (f"{value:.3f}" if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in result.get("metrics", [])
            ],
        }
    return result


RAW_DECIMAL = re.compile(r"(?<![\d.])0\.\d{4,}")
CCTV_AS_UNITS = re.compile(
    r"CCTV(?:\s*(?:설치)?\s*(?:수|대수|개수|장비))?(?:은|는|이|가)?\s*[:：]?\s*(\d[\d,]*)\s*(?:개|대)"
)
CCTV_ZONES_AS_UNITS = re.compile(r"(CCTV\s*설치\s*(?:어린이)?보호구역\s*[:：]?\s*\d[\d,]*)\s*(?:개|대)")


def sanitize_llm_text(text: str, tool_results: list[dict[str, Any]]) -> str:
    """Last safety net on the model's wording (values themselves are unchanged)."""
    if not text:
        return text
    # CCTV counts are school zones with CCTV, never device units.
    text = CCTV_AS_UNITS.sub(lambda m: f"CCTV 설치 보호구역 {m.group(1)}곳", text)
    text = CCTV_ZONES_AS_UNITS.sub(lambda m: f"{m.group(1)}곳", text)

    # Any internal probability that slipped through is shown as the UI score.
    known: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "predicted_high_risk_probability_2026" and isinstance(inner, (int, float)):
                    known.append(float(inner))
                else:
                    collect(inner)
        elif isinstance(value, list):
            for inner in value:
                collect(inner)

    for item in tool_results:
        collect(item.get("result"))

    def to_score(match: re.Match[str]) -> str:
        raw = float(match.group(0))
        for value in known:
            if abs(value - raw) < 5e-4:
                return score_text(value)
        # a probability of a region outside this answer's tool results
        if len(match.group(0)) >= 7:
            for row in all_ai_regions():
                value = row.get("predicted_high_risk_probability_2026")
                if isinstance(value, (int, float)) and abs(value - raw) < 5e-5:
                    return score_text(value)
        return match.group(0)

    return RAW_DECIMAL.sub(to_score, text)


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        return {"status": "error", "message": "지원하지 않는 tool입니다."}
    return TOOLS[name](**arguments)


def responses_tool_schemas() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for schema in TOOL_SCHEMAS:
        function = schema.get("function", {})
        tools.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return tools


def response_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def response_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif content.get("type") == "text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def verified_numbers_summary(tool_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in tool_results:
        tool = item.get("tool")
        result = item.get("result")
        if not isinstance(result, dict):
            continue

        if tool == "get_region_analysis" and result.get("status") == "ok":
            lines.append(
                "근거 수치: "
                f"{result.get('region')} · 사고 {result.get('accident_count')}건 · "
                f"부상 {result.get('injury_count')}명 · 사망 {result.get('death_count')}명 · "
                f"사고율 {rate_text(result.get('accident_per_10k_child'))} · "
                f"AI 상대적 고위험 점수 {score_text(result.get('predicted_high_risk_probability_2026'))} · "
                f"전국 AI 순위 {rank_text(result.get('risk_rank_2026'))}"
            )
        elif tool == "get_facility_analysis" and result.get("status") == "ok":
            lines.append(
                "근거 수치: "
                f"{result.get('region')} · 횡단보도 {number_text(result.get('crosswalk'), '개')} · "
                f"신호등 {number_text(result.get('traffic_light'), '개')} · "
                f"어린이보호구역 {number_text(result.get('school_zone'), '개')} · "
                f"{cctv_text(result.get('cctv'), result.get('cctv_install_rate'))}"
            )
        elif tool == "get_high_risk_top":
            top_lines = [
                f"{row.get('risk_rank_2026')}. {row.get('sido_sgg_nm')} {score_text(row.get('predicted_high_risk_probability_2026'))}"
                for row in result.get("items", [])[:5]
            ]
            if top_lines:
                lines.append("근거 수치: " + " · ".join(top_lines))
        elif tool == "get_model_importance":
            top_lines = [
                f"{row.get('feature')} {importance_text(row.get('importance_mean'))}"
                for row in result.get("items", [])[:5]
            ]
            if top_lines:
                lines.append("근거 수치: " + " · ".join(top_lines))
    return "\n".join(lines)


def llm_answer(question: str, client: LlmClient) -> dict[str, Any]:
    response = client.create_response(
        input_data=question,
        instructions=SYSTEM_PROMPT,
        tools=responses_tool_schemas(),
        tool_choice="required",
    )
    tool_results: list[dict[str, Any]] = []
    response_id = response.get("id")

    for _ in range(3):
        calls = response_function_calls(response)
        if not calls:
            break

        function_outputs: list[dict[str, Any]] = []
        for call in calls:
            name = call.get("name")
            raw_args = call.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            result = run_tool(name, args)
            tool_results.append({"tool": name, "result": result})
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": json.dumps(llm_view(name, result), ensure_ascii=False),
                }
            )

        response = client.create_response(
            input_data=function_outputs,
            instructions=f"{SYSTEM_PROMPT}\n\n{FINAL_ANSWER_PROMPT}",
            previous_response_id=response_id,
        )
        response_id = response.get("id") or response_id

    content = sanitize_llm_text(response_output_text(response), tool_results)
    verified_numbers = verified_numbers_summary(tool_results)
    if verified_numbers:
        content = f"{content}\n\n{verified_numbers}" if content else verified_numbers
    region = None
    for item in tool_results:
        result = item["result"]
        if isinstance(result, dict) and result.get("region"):
            region = result["region"]
            break
    return {
        "answer": content,
        "intent": "llm_tool_answer",
        "region": region,
        "sources": [item["result"].get("source") for item in tool_results if isinstance(item["result"], dict) and item["result"].get("source")],
        "actions": actions_for(region),
        "metadata": {
            "mode": "llm_tool_calling",
            "provider": "openai",
            "model": client.model,
            "tool_call_count": len(tool_results),
            "tool_names": [item["tool"] for item in tool_results],
            "tools": tool_results,
        },
    }


ASSISTANT_DISABLED_MESSAGE = "데모 버전에서는 AI 자연어 분석 기능이 비활성화되어 있습니다."


def assistant_status() -> dict[str, Any]:
    """Public on/off state for the UI. Never exposes keys or model secrets."""
    client = LlmClient()
    if not client.enabled:
        return {"enabled": False, "mode": "disabled", "message": ASSISTANT_DISABLED_MESSAGE}
    return {"enabled": True, "mode": "openai" if client.available else "local_fallback"}


def answer_question(question: str) -> dict[str, Any]:
    client = LlmClient()
    # AI_ASSISTANT_ENABLED=false: answer with the notice only — no tools, no OpenAI request.
    if not client.enabled:
        return {
            "answer": ASSISTANT_DISABLED_MESSAGE,
            "intent": "disabled",
            "region": None,
            "sources": [],
            "actions": [],
            "metadata": {
                "mode": "disabled",
                "provider": "none",
            },
        }
    interpretation = interpretation_answer(question)
    if interpretation:
        interpretation["metadata"] = {
            "mode": "rule_based_interpretation",
            "provider": "local",
            "tool_call_count": 0,
            "tool_names": [],
        }
        return interpretation
    if client.available:
        try:
            return llm_answer(question, client)
        except Exception as exc:
            # Do not fall back when a key is configured. This makes provider
            # integration failures visible without exposing credentials or raw
            # upstream error details.
            return {
                "answer": "OpenAI LLM 호출 중 오류가 발생했습니다. API 키 값은 노출하지 않으며, 서버 로그와 네트워크 설정을 확인해 주세요.",
                "intent": "provider_error",
                "region": None,
                "sources": [],
                "actions": [],
                "metadata": {
                    "mode": "provider_error",
                    "provider": "openai",
                    "model": client.model,
                    "error_type": type(exc).__name__,
                },
            }
    return mark_fallback(fallback_answer(question), "missing_openai_api_key")


