"""Read-only tool functions used by the natural-language assistant.

The assistant is intentionally limited to the same project data sources used by
the public API. It never receives MongoDB credentials and never runs arbitrary
database queries.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo.errors import PyMongoError

from backend.db import get_database


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"

MODEL_FEATURES = [
    "prev_child_population",
    "prev_vehicle_count",
    "prev_accident_count",
    "prev_injury_count",
    "prev_death_count",
    "prev_accident_per_10k_child",
    "prev_vehicles_per_100_child",
    "accident_count_change_1y",
    "accident_rate_change_1y",
    "accident_count_mean_2y",
]

MODEL_TARGET_DESCRIPTION = (
    "다음 연도 시군구의 어린이 인구 1만 명당 사고율이 해당 연도 상위 25% 이상인지 예측하는 지역 단위 분류 모델입니다."
)


class AssistantDataSourceUnavailableError(RuntimeError):
    """MongoDB connectivity failure, distinct from a region with no facility row."""


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: clean_value(value) for key, value in record.items() if key != "_id"}


@lru_cache(maxsize=8)
def csv_records(filename: str) -> tuple[dict[str, Any], ...]:
    path = RESULTS_DIR / filename
    if not path.exists():
        path = DATA_PROCESSED_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    return tuple(clean_record(record) for record in df.to_dict(orient="records"))


def compact(text: str | None) -> str:
    return "".join(str(text or "").split()).lower()


def all_ai_regions() -> list[dict[str, Any]]:
    records = [
        clean_record(record)
        for record in get_database()["prediction_2026_229"].find({}, {"_id": 0})
    ]
    records.sort(key=lambda record: record.get("risk_rank_2026") or 10**9)
    return records


def improvement_candidate_record(region: str) -> dict[str, Any] | None:
    record = get_database()["infrastructure_blind_spots_49"].find_one(
        {"sido_sgg_nm": region},
        {"_id": 0},
    )
    return clean_record(record) if record else None


def match_regions(region_name: str) -> list[str]:
    """Match full or partial Korean region names against 229 AI regions."""
    query = compact(region_name)
    if not query:
        return []
    exact: list[str] = []
    partial: list[str] = []
    for record in all_ai_regions():
        full = str(record.get("sido_sgg_nm") or "")
        pieces = full.split()
        sigungu = pieces[-1] if pieces else full
        full_compact = compact(full)
        sgg_compact = compact(sigungu)
        if query == full_compact or query == sgg_compact:
            exact.append(full)
        elif query in full_compact or query in sgg_compact:
            partial.append(full)
    return exact or partial


def resolve_region(region_name: str) -> dict[str, Any]:
    matches = match_regions(region_name)
    if not matches:
        return {
            "status": "not_found",
            "message": "현재 분석 데이터에서 해당 지역을 확인하지 못했습니다.",
            "matches": [],
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "message": "지역명이 여러 곳과 일치합니다. 시도와 시군구를 함께 입력해 주세요.",
            "matches": matches[:10],
        }
    return {"status": "ok", "region": matches[0], "matches": matches}


def facility_record(region: str) -> dict[str, Any] | None:
    try:
        db = get_database()
        doc = db["eda_main_49"].find_one(
            {"$or": [{"sido_sgg_nm": region}, {"region_name": region}]},
            {
                "_id": 0,
                "sido_sgg_nm": 1,
                "SIDO_NM": 1,
                "SIGUNGU_NM": 1,
                "crosswalk_count": 1,
                "traffic_light_count": 1,
                "schoolzone_count": 1,
                "cctv_installed_zone_count": 1,
                "cctv_install_rate": 1,
                "infrastructure_score": 1,
                "risk_score": 1,
                "priority_score": 1,
            },
        )
    except (PyMongoError, RuntimeError) as exc:
        raise AssistantDataSourceUnavailableError(
            "시설 상세분석 데이터베이스에 연결할 수 없습니다."
        ) from exc
    if not doc:
        return None
    item = clean_record(doc)
    try:
        candidate = improvement_candidate_record(region)
    except (PyMongoError, RuntimeError) as exc:
        raise AssistantDataSourceUnavailableError(
            "개선 검토 후보 데이터베이스에 연결할 수 없습니다."
        ) from exc
    if candidate:
        item.update(
            {
                "infrastructure_deficiency_score": candidate.get(
                    "infrastructure_deficiency_score"
                ),
                "improvement_priority_score": candidate.get(
                    "improvement_priority_score"
                ),
                "improvement_priority_rank": candidate.get(
                    "improvement_priority_rank"
                ),
            }
        )
    return item


def get_region_analysis(region_name: str) -> dict[str, Any]:
    resolved = resolve_region(region_name)
    if resolved["status"] != "ok":
        return resolved
    region = resolved["region"]
    record = next(
        item for item in all_ai_regions() if item.get("sido_sgg_nm") == region
    )
    try:
        facility = facility_record(region)
    except AssistantDataSourceUnavailableError as exc:
        return {
            "status": "data_source_unavailable",
            "region": region,
            "facility_data_available": None,
            "message": str(exc),
        }
    return {
        "status": "ok",
        "region": region,
        "accident_count": record.get("prev_accident_count"),
        "injury_count": record.get("prev_injury_count"),
        "death_count": record.get("prev_death_count"),
        "child_population": record.get("prev_child_population"),
        "vehicle_count": record.get("prev_vehicle_count"),
        "accident_per_10k_child": record.get("prev_accident_per_10k_child"),
        "accident_count_change_1y": record.get("accident_count_change_1y"),
        "accident_rate_change_1y": record.get("accident_rate_change_1y"),
        "accident_count_mean_2y": record.get("accident_count_mean_2y"),
        "predicted_high_risk_probability_2026": record.get(
            "predicted_high_risk_probability_2026"
        ),
        "risk_rank_2026": record.get("risk_rank_2026"),
        "facility_data_available": facility is not None,
        "facility_summary": facility,
        "source": "prediction_2026_229",
    }


def compare_regions(region_names: list[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in region_names[:4]:
        resolved = resolve_region(name)
        if resolved["status"] != "ok":
            unresolved.append({"input": name, **resolved})
            continue
        region = resolved["region"]
        if region in seen:
            continue
        seen.add(region)
        items.append(get_region_analysis(region))
    return {
        "status": "ok" if items else "not_found",
        "items": items,
        "unresolved": unresolved,
        "source": "prediction_2026_229",
    }


def get_high_risk_top(limit: int = 5) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 5), 20))
    records = all_ai_regions()
    return {
        "status": "ok",
        "items": records[:safe_limit],
        "total": len(records),
        "source": "prediction_2026_229",
    }


def get_facility_analysis(region_name: str) -> dict[str, Any]:
    resolved = resolve_region(region_name)
    if resolved["status"] != "ok":
        return resolved
    region = resolved["region"]
    try:
        facility = facility_record(region)
    except AssistantDataSourceUnavailableError as exc:
        return {
            "status": "data_source_unavailable",
            "region": region,
            "facility_data_available": None,
            "message": str(exc),
        }
    if facility is None:
        return {
            "status": "not_available",
            "region": region,
            "facility_data_available": False,
            "message": "시설이 없는 것이 아니라 현재 시설 상세분석 데이터 제공 대상이 아닙니다.",
        }
    return {
        "status": "ok",
        "region": region,
        "facility_data_available": True,
        "crosswalk": facility.get("crosswalk_count"),
        "traffic_light": facility.get("traffic_light_count"),
        "school_zone": facility.get("schoolzone_count"),
        "cctv": facility.get("cctv_installed_zone_count"),
        "cctv_meaning": "CCTV가 설치된 어린이보호구역 수이며 CCTV 대수가 아닙니다.",
        "cctv_install_rate": facility.get("cctv_install_rate"),
        "infrastructure_supply_score": facility.get("infrastructure_score"),
        "infrastructure_deficiency": facility.get(
            "infrastructure_deficiency_score"
        ),
        "improvement_priority_score": facility.get(
            "improvement_priority_score"
        ),
        "improvement_priority_rank": facility.get("improvement_priority_rank"),
        "source": "eda_main_49 + infrastructure_blind_spots_49",
    }


def get_improvement_candidates(limit: int = 5) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 5), 20))
    records = [
        clean_record(record)
        for record in get_database()["infrastructure_blind_spots_49"].find({}, {"_id": 0})
    ]
    records.sort(key=lambda record: record.get("improvement_priority_rank") or 10**9)
    return {
        "status": "ok",
        "items": records[:safe_limit],
        "total": len(records),
        "source": "infrastructure_blind_spots_49",
        "note": "49개 시설분석 지역 기준 분석용 개선 검토 후보입니다. 공식 정책 우선순위가 아닙니다.",
    }


def get_model_info() -> dict[str, Any]:
    metrics = list(csv_records("model_average_metrics.csv"))
    return {
        "status": "ok",
        "final_model": "Logistic Regression",
        "target_definition": MODEL_TARGET_DESCRIPTION,
        "analysis_scope": "전국 229개 시군구 AI 분석, 49개 지역 시설 상세분석",
        "feature_list": MODEL_FEATURES,
        "metrics": metrics,
        "source": "models/high_risk_classifier.joblib + results/model_average_metrics.csv",
    }


def get_model_importance() -> dict[str, Any]:
    return {
        "status": "ok",
        "items": list(csv_records("feature_importance_permutation_2025.csv")),
        "explanation_method": "permutation_importance",
        "shap_available": False,
        "source": "results/feature_importance_permutation_2025.csv",
    }


TOOLS = {
    "get_region_analysis": get_region_analysis,
    "compare_regions": compare_regions,
    "get_high_risk_top": get_high_risk_top,
    "get_facility_analysis": get_facility_analysis,
    "get_improvement_candidates": get_improvement_candidates,
    "get_model_info": get_model_info,
    "get_model_importance": get_model_importance,
}

