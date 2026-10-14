"""Read-only API for child traffic safety MongoDB collections and model outputs."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from backend.db import close_client, get_client, get_database
from backend.services.assistant_service import answer_question, assistant_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

HOTSPOT_FIELDS = {
    "_id": 0,
    "hotspot_id": 1,
    "source_type": 1,
    "location_name": 1,
    "accident_year": 1,
    "year_verification_status": 1,
    "sido": 1,
    "sigungu": 1,
    "latitude": 1,
    "longitude": 1,
    "accident_count": 1,
    "casualty_count": 1,
    "death_count": 1,
    "serious_injury_count": 1,
    "minor_injury_count": 1,
    "injury_report_count": 1,
    "geometry": 1,
}

REGION_BASE_FIELDS = {
    "_id": 0,
    "SIDO_NM": 1,
    "SIGUNGU_NM": 1,
    "sido_sgg_nm": 1,
    "child_accident_count": 1,
    "child_death_count": 1,
    "child_injury_count": 1,
    "child_population_0_12": 1,
    "crosswalk_count": 1,
    "traffic_light_count": 1,
    "schoolzone_count": 1,
    "cctv_installed_zone_count": 1,
    "schoolzone_data_available": 1,
    "cctv_install_rate": 1,
}

REGION_EDA_FIELDS = {
    **REGION_BASE_FIELDS,
    "accident_per_10k_child": 1,
    "crosswalk_per_1000_child": 1,
    "traffic_light_per_1000_child": 1,
    "schoolzone_per_1000_child": 1,
    "region_name": 1,
    "crosswalk_score": 1,
    "traffic_light_score": 1,
    "schoolzone_score": 1,
    "cctv_score": 1,
    "infrastructure_score": 1,
    "risk_score": 1,
    "priority_score": 1,
}

MAX_LIMIT = 500
DEFAULT_LIMIT = 50
MAX_COMPARE_REGIONS = 10

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
DATA_INTERPRETATION_WARNING = (
    "사고다발지역은 공식 기준으로 선정된 지점 자료이며 전체 사고자료, 학교별 위험도, 안전/위험 순위가 아닙니다. "
    "risk_score, priority_score, improvement_priority_score는 분석용 파생값이며 공식 점수가 아닙니다."
)

app = FastAPI(
    title="Child Traffic Safety API",
    version="0.1.0",
    description=(
        "Read-only API for selected child traffic accident hotspot locations. "
        "Hotspots are official selected locations, not all crashes or school risk scores."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://savespot-amber.vercel.app",
    ],
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AssistantQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AssistantQueryResponse(BaseModel):
    answer: str
    intent: str
    region: str | None = None
    sources: list[Any] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_client()


def bounded_limit(limit: int) -> int:
    return min(max(limit, 1), MAX_LIMIT)


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


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Required data file is missing: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return [clean_record(record) for record in df.to_dict(orient="records")]


@lru_cache(maxsize=8)
def cached_csv_records(relative_path: str) -> tuple[dict[str, Any], ...]:
    return tuple(read_csv_records(PROJECT_ROOT / relative_path))


@lru_cache(maxsize=1)
def model_metadata() -> dict[str, Any]:
    model_path = MODELS_DIR / "high_risk_classifier.joblib"
    if not model_path.exists():
        return {
            "model_available": False,
            "model_name": None,
            "feature_cols": MODEL_FEATURES,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(model_path)
    return {
        "model_available": True,
        "model_name": bundle.get("model_name"),
        "feature_cols": bundle.get("feature_cols", MODEL_FEATURES),
    }


def parse_accident_year(value: str | None) -> int | None | str:
    if value is None:
        return "__not_set__"
    normalized = value.strip().lower()
    if normalized in {"", "null", "none", "unknown"}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="accident_year must be an integer or null") from exc


def hotspot_query(
    source_type: str | None = None,
    accident_year: str | None = None,
    sido: str | None = None,
    sigungu: str | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if source_type:
        query["source_type"] = source_type
    parsed_year = parse_accident_year(accident_year)
    if parsed_year != "__not_set__":
        query["accident_year"] = parsed_year
    if sido:
        query["sido"] = sido
    if sigungu:
        query["sigungu"] = sigungu
    return query


def clean_document(doc: dict[str, Any]) -> dict[str, Any]:
    return clean_record(doc)


def make_region_query(sido: str | None = None, sigungu: str | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if sido:
        query["SIDO_NM"] = sido
    if sigungu:
        query["SIGUNGU_NM"] = sigungu
    return query


def region_identifier(record: dict[str, Any]) -> str | None:
    return record.get("sido_sgg_nm") or record.get("region_name")


def source_metadata(source: str, count: int) -> dict[str, Any]:
    return {
        "source": source,
        "count": count,
        "interpretation_warning": DATA_INTERPRETATION_WARNING,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        get_client().admin.command("ping")
        db = get_database()
        return {
            "status": "ok",
            "mongodb_connected": True,
            "database_accessible": db.name == "child_traffic_safety" or bool(db.name),
            "database_name": db.name,
            "note": "read-only API; credentials are not exposed",
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "mongodb_connected": False,
            "database_accessible": False,
            "error_type": type(exc).__name__,
        }


@app.get("/api/assistant/status")
def get_assistant_status() -> dict[str, Any]:
    """Whether the natural-language assistant is enabled (no secrets returned)."""
    return assistant_status()


@app.post("/api/assistant/query", response_model=AssistantQueryResponse)
def query_assistant(payload: AssistantQueryRequest) -> dict[str, Any]:
    """Answer a natural-language question using read-only project tools."""
    return answer_question(payload.question.strip())


@app.get("/api/regions")
def get_regions(
    sido: str | None = Query(default=None),
    sigungu: str | None = Query(default=None),
    include_scores: bool = Query(default=False),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return 49 regional rows used for facility/status views."""
    try:
        db = get_database()
        collection_name = "eda_main_49" if include_scores else "final_main_49"
        fields = REGION_EDA_FIELDS if include_scores else REGION_BASE_FIELDS
        query = make_region_query(sido=sido, sigungu=sigungu)
        collection = db[collection_name]
        total = collection.count_documents(query)
        cursor = (
            collection.find(query, fields)
            .sort([("SIDO_NM", 1), ("SIGUNGU_NM", 1)])
            .skip(offset)
            .limit(bounded_limit(limit))
        )
        items = [clean_document(doc) for doc in cursor]
        return {
            "items": items,
            "pagination": {
                "limit": bounded_limit(limit),
                "offset": offset,
                "count": len(items),
                "total": total,
            },
            "metadata": source_metadata(collection_name, total),
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc


@app.get("/api/regions/compare")
def compare_regions(
    regions: list[str] = Query(..., description="Repeat this query parameter with sido_sgg_nm values."),
) -> dict[str, Any]:
    if len(regions) > MAX_COMPARE_REGIONS:
        raise HTTPException(status_code=422, detail=f"regions supports up to {MAX_COMPARE_REGIONS} items")
    try:
        db = get_database()
        cursor = db["eda_main_49"].find({"sido_sgg_nm": {"$in": regions}}, REGION_EDA_FIELDS)
        records_by_id = {region_identifier(clean_document(doc)): clean_document(doc) for doc in cursor}
        items = [records_by_id[region] for region in regions if region in records_by_id]
        missing = [region for region in regions if region not in records_by_id]
        return {
            "items": items,
            "missing": missing,
            "metadata": source_metadata("eda_main_49", len(items)),
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc


@app.get("/api/regions/{region_id}")
def get_region(region_id: str) -> dict[str, Any]:
    try:
        db = get_database()
        query = {"$or": [{"sido_sgg_nm": region_id}, {"region_name": region_id}]}
        region = db["eda_main_49"].find_one(query, REGION_EDA_FIELDS)
        if region is None:
            raise HTTPException(status_code=404, detail="region not found in eda_main_49")
        speed_bump = db["final_speed_bump_35"].find_one(
            {"sido_sgg_nm": region_identifier(region)},
            {"_id": 0, "speed_bump_count": 1},
        )
        item = clean_document(region)
        item["speed_bump_count"] = clean_value(speed_bump.get("speed_bump_count")) if speed_bump else None
        return {
            "item": item,
            "metadata": source_metadata("eda_main_49 + final_speed_bump_35", 1),
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc


@app.get("/api/ai/high-risk")
def get_ai_high_risk(
    min_probability: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    records = list(cached_csv_records("results/predicted_high_risk_2026_229.csv"))
    filtered = [
        record
        for record in records
        if (record.get("predicted_high_risk_probability_2026") or 0) >= min_probability
    ]
    filtered.sort(key=lambda record: record.get("risk_rank_2026") or 10**9)
    safe_limit = bounded_limit(limit)
    items = filtered[offset : offset + safe_limit]
    return {
        "items": items,
        "pagination": {
            "limit": safe_limit,
            "offset": offset,
            "count": len(items),
            "total": len(filtered),
        },
        "model": {
            **model_metadata(),
            "target": MODEL_TARGET_DESCRIPTION,
            "prediction_year": 2026,
        },
        "metadata": source_metadata("results/predicted_high_risk_2026_229.csv", len(filtered)),
    }


@app.get("/api/ai/top10")
def get_ai_top10(
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    records = list(cached_csv_records("results/predicted_high_risk_2026_229.csv"))
    records.sort(key=lambda record: record.get("risk_rank_2026") or 10**9)
    items = records[:limit]
    return {
        "items": items,
        "model": {
            **model_metadata(),
            "target": MODEL_TARGET_DESCRIPTION,
            "prediction_year": 2026,
        },
        "metadata": source_metadata("results/predicted_high_risk_2026_229.csv", len(items)),
    }


@app.get("/api/facilities")
def get_facilities(
    sido: str | None = Query(default=None),
    sigungu: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        db = get_database()
        query = make_region_query(sido=sido, sigungu=sigungu)
        total = db["eda_main_49"].count_documents(query)
        cursor = (
            db["eda_main_49"]
            .find(
                query,
                {
                    "_id": 0,
                    "SIDO_NM": 1,
                    "SIGUNGU_NM": 1,
                    "sido_sgg_nm": 1,
                    "child_population_0_12": 1,
                    "crosswalk_count": 1,
                    "traffic_light_count": 1,
                    "schoolzone_count": 1,
                    "cctv_installed_zone_count": 1,
                    "cctv_install_rate": 1,
                    "crosswalk_per_1000_child": 1,
                    "traffic_light_per_1000_child": 1,
                    "schoolzone_per_1000_child": 1,
                    "infrastructure_score": 1,
                },
            )
            .sort([("SIDO_NM", 1), ("SIGUNGU_NM", 1)])
            .skip(offset)
            .limit(bounded_limit(limit))
        )
        speed_bump_records = {
            doc["sido_sgg_nm"]: doc.get("speed_bump_count")
            for doc in db["final_speed_bump_35"].find({}, {"_id": 0, "sido_sgg_nm": 1, "speed_bump_count": 1})
        }
        items = []
        for doc in cursor:
            item = clean_document(doc)
            item["speed_bump_count"] = clean_value(speed_bump_records.get(item.get("sido_sgg_nm")))
            items.append(item)
        return {
            "items": items,
            "pagination": {
                "limit": bounded_limit(limit),
                "offset": offset,
                "count": len(items),
                "total": total,
            },
            "metadata": source_metadata("eda_main_49 + final_speed_bump_35", total),
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc


@app.get("/api/improvement-candidates")
def get_improvement_candidates(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    records = list(cached_csv_records("results/infrastructure_blind_spots_49.csv"))
    records.sort(key=lambda record: record.get("improvement_priority_rank") or 10**9)
    safe_limit = bounded_limit(limit)
    items = records[offset : offset + safe_limit]
    return {
        "items": items,
        "pagination": {
            "limit": safe_limit,
            "offset": offset,
            "count": len(items),
            "total": len(records),
        },
        "score_definition": (
            "predicted_high_risk_probability_2026와 infrastructure_deficiency_score를 결합한 49개 지역 분석용 개선 검토 후보 점수입니다."
        ),
        "metadata": source_metadata("results/infrastructure_blind_spots_49.csv", len(records)),
    }


@app.get("/api/model/info")
def get_model_info() -> dict[str, Any]:
    metrics = cached_csv_records("results/model_average_metrics.csv")
    return {
        "model": {
            **model_metadata(),
            "target": MODEL_TARGET_DESCRIPTION,
            "prediction_unit": "시군구 단위",
            "positive_label": "해당 연도 어린이 인구 1만 명당 사고율 상위 25% 이상",
            "not_supported": [
                "학교별 위험도",
                "사고다발지역 지점별 AI 위험도",
                "안전시설 수 변경에 따른 재예측",
            ],
        },
        "metrics": list(metrics),
        "metadata": source_metadata("models/high_risk_classifier.joblib + results/model_average_metrics.csv", len(metrics)),
    }


@app.get("/api/model/importance")
def get_model_importance() -> dict[str, Any]:
    items = list(cached_csv_records("results/feature_importance_permutation_2025.csv"))
    return {
        "items": items,
        "explanation_method": "permutation_importance",
        "shap_available": False,
        "model": {
            **model_metadata(),
            "target": MODEL_TARGET_DESCRIPTION,
        },
        "metadata": source_metadata("results/feature_importance_permutation_2025.csv", len(items)),
    }


@app.get("/api/hotspots")
def get_hotspots(
    source_type: str | None = Query(default=None),
    accident_year: str | None = Query(default=None, description="Integer year, or null/unknown for schoolzone records"),
    sido: str | None = Query(default=None),
    sigungu: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        db = get_database()
        collection = db["hotspots_map"]
        query = hotspot_query(source_type=source_type, accident_year=accident_year, sido=sido, sigungu=sigungu)
        safe_limit = bounded_limit(limit)
        total = collection.count_documents(query)
        cursor = (
            collection.find(query, HOTSPOT_FIELDS)
            .sort([("source_type", 1), ("sido", 1), ("sigungu", 1), ("hotspot_id", 1)])
            .skip(offset)
            .limit(safe_limit)
        )
        items = [clean_document(doc) for doc in cursor]
        return {
            "items": items,
            "pagination": {
                "limit": safe_limit,
                "offset": offset,
                "count": len(items),
                "total": total,
            },
            "meaning": "주변 및 지도 응답은 공식 기준으로 선정된 사고다발지역 지점이며 전체 사고자료나 학교별 위험도가 아닙니다.",
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc


@app.get("/api/hotspots/nearby")
def get_nearby_hotspots(
    lat: float = Query(..., ge=33, le=39.5),
    lng: float = Query(..., ge=124, le=132),
    radius_m: int = Query(..., gt=0, le=10000),
    source_type: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> dict[str, Any]:
    try:
        db = get_database()
        query: dict[str, Any] = {}
        if source_type:
            query["source_type"] = source_type

        pipeline: list[dict[str, Any]] = [
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": [lng, lat]},
                    "distanceField": "distance_m",
                    "maxDistance": radius_m,
                    "spherical": True,
                    "query": query,
                }
            },
            {"$limit": bounded_limit(limit)},
            {
                "$project": {
                    **HOTSPOT_FIELDS,
                    "distance_m": {"$round": ["$distance_m", 2]},
                }
            },
        ]
        items = [clean_document(doc) for doc in db["hotspots_map"].aggregate(pipeline)]
        return {
            "items": items,
            "query": {
                "lat": lat,
                "lng": lng,
                "radius_m": radius_m,
                "source_type": source_type,
                "limit": bounded_limit(limit),
            },
            "meaning": "nearby 결과는 입력 지점 주변의 공식 사고다발지역 지점이며 전체 사고자료나 학교별 위험도가 아닙니다.",
        }
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail={"error_type": type(exc).__name__}) from exc
