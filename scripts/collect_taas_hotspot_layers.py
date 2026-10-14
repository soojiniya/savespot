"""Collect and normalize TAAS hotspot layers for map use.

This script intentionally treats hotspot data as selected hotspot locations,
not as all child traffic crashes or school-level risk labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from shapely.geometry import Point, shape
except Exception:  # pragma: no cover - validation reports this if unavailable.
    Point = None
    shape = None


ROOT = Path(__file__).resolve().parents[1]
SCHOOLZONE_RAW = ROOT / "data_raw" / "taas_schoolzone_hotspots.csv"
CACHE_DIR = ROOT / "data_raw" / "api_cache" / "child_pedestrian_hotspots"
PROCESSED_DIR = ROOT / "data_processed" / "taas" / "hotspots"
DOC_DIR = ROOT / "docs" / "taas"
CODELIST_URL = "https://opendata.koroad.or.kr/api/down/accidenthazard_codelist_down.jsp"
CHILD_PEDESTRIAN_ENDPOINT = "https://apis.data.go.kr/B552061/frequentzoneChild/getRestFrequentzoneChild"
CODELIST_XLSX = CACHE_DIR / "AccidentHazard_CodeList.xlsx"
ENV_KEY_NAME = "DATA_GO_KR_CHILD_PEDESTRIAN_API_KEY"

SAMPLE_REGIONS = [
    ("capital", "서울특별시", "강남구"),
    ("chungcheong", "충청남도", "천안시동남구"),
    ("yeongnam", "부산광역시", "부산진구"),
    ("honam", "광주광역시(구)", "광산구"),
    ("jeju", "제주특별자치도", "제주시"),
]


@dataclass(frozen=True)
class RegionCode:
    sido_name: str
    sigungu_name: str
    sido_code: str
    gugun_code: str

    @property
    def label(self) -> str:
        return f"{self.sido_name} {self.sigungu_name}".strip()


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def load_env_key() -> str:
    env_path = ROOT / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    api_key = values.get(ENV_KEY_NAME) or os.environ.get(ENV_KEY_NAME)
    if not api_key:
        raise RuntimeError(f"{ENV_KEY_NAME} is not available in .env or environment.")
    return api_key


def download_codelist(force: bool = False) -> Path:
    if CODELIST_XLSX.exists() and not force:
        return CODELIST_XLSX
    response = requests.get(CODELIST_URL, timeout=30)
    response.raise_for_status()
    CODELIST_XLSX.write_bytes(response.content)
    return CODELIST_XLSX


def parse_region_codes(codelist_path: Path) -> pd.DataFrame:
    sido = pd.read_excel(codelist_path, sheet_name="Sido 요청값")
    gugun = pd.read_excel(codelist_path, sheet_name="Gugun 요청값")

    sido = sido.rename(columns={"종류": "sido_name", "Sido 입력값": "sido_code"})
    sido = sido[["sido_name", "sido_code"]].dropna()
    sido = sido[sido["sido_name"].astype(str).str.strip().ne("")]
    sido["sido_name"] = sido["sido_name"].apply(normalize_code_name)
    sido["sido_code"] = sido["sido_code"].astype(int).astype(str).str.zfill(2)

    gugun = gugun.rename(columns={"시도": "sido_name", "구군명칭": "sigungu_name", "gugun 입력값": "gugun_code"})
    gugun["sido_name"] = gugun["sido_name"].ffill()
    gugun = gugun[["sido_name", "sigungu_name", "gugun_code"]].dropna()
    gugun["sido_name"] = gugun["sido_name"].apply(normalize_code_name)
    gugun["sigungu_name"] = gugun["sigungu_name"].astype(str).str.strip()
    gugun["gugun_code"] = gugun["gugun_code"].astype(int).astype(str).str.zfill(3)

    regions = gugun.merge(sido, on="sido_name", how="left")
    missing = regions["sido_code"].isna()
    if missing.any():
        missing_names = sorted(regions.loc[missing, "sido_name"].unique())
        raise RuntimeError(f"Missing sido codes for: {missing_names}")

    regions = regions[["sido_name", "sigungu_name", "sido_code", "gugun_code"]]
    regions = regions.drop_duplicates(["sido_code", "gugun_code"]).sort_values(["sido_code", "gugun_code"])
    regions.to_csv(PROCESSED_DIR / "taas_hotspot_region_codes.csv", index=False, encoding="utf-8-sig")
    return regions


def pick_sample_regions(regions: pd.DataFrame) -> list[RegionCode]:
    picked: list[RegionCode] = []
    for group, sido_name, sigungu_name in SAMPLE_REGIONS:
        rows = regions[(regions["sido_name"] == sido_name) & (regions["sigungu_name"] == sigungu_name)]
        if rows.empty:
            raise RuntimeError(f"Sample region not found in code list: {group} {sido_name} {sigungu_name}")
        row = rows.iloc[0]
        picked.append(
            RegionCode(
                sido_name=str(row["sido_name"]),
                sigungu_name=str(row["sigungu_name"]),
                sido_code=str(row["sido_code"]).zfill(2),
                gugun_code=str(row["gugun_code"]).zfill(3),
            )
        )
    return picked


def request_cache_path(year: int, sido_code: str, gugun_code: str, page: int) -> Path:
    return CACHE_DIR / f"year={year}" / f"sido={sido_code}_gugun={gugun_code}_page={page}.json"


def normalize_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if isinstance(items, dict):
        items = items.get("item")
    if items is None or items == "":
        return []
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]
    return []


def safe_response_summary(payload: dict[str, Any], http_status: int) -> dict[str, Any]:
    return {
        "http_status": http_status,
        "resultCode": payload.get("resultCode"),
        "resultMsg": payload.get("resultMsg"),
        "totalCount": payload.get("totalCount"),
        "numOfRows": payload.get("numOfRows"),
        "pageNo": payload.get("pageNo"),
        "record_count_in_page": len(normalize_items(payload)),
    }


def fetch_api_page(
    api_key: str,
    year: int,
    region: RegionCode,
    page: int = 1,
    num_rows: int = 100,
    use_cache: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_path = request_cache_path(year, region.sido_code, region.gugun_code, page)
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload, {**safe_response_summary(payload, 200), "from_cache": True}

    params = {
        "serviceKey": api_key,
        "searchYearCd": str(year),
        "siDo": region.sido_code,
        "guGun": region.gugun_code,
        "type": "json",
        "numOfRows": str(num_rows),
        "pageNo": str(page),
    }
    response = requests.get(CHILD_PEDESTRIAN_ENDPOINT, params=params, timeout=30)
    try:
        payload = response.json()
    except Exception:
        payload = {
            "resultCode": "NON_JSON",
            "resultMsg": response.text[:200],
            "items": None,
            "totalCount": 0,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.05)
    return payload, {**safe_response_summary(payload, response.status_code), "from_cache": False}


def verify_year_candidates(api_key: str, samples: list[RegionCode], use_cache: bool) -> tuple[pd.DataFrame, list[int]]:
    rows: list[dict[str, Any]] = []
    for year in range(2012, 2026):
        for region in samples:
            try:
                payload, summary = fetch_api_page(api_key, year, region, page=1, num_rows=1, use_cache=use_cache)
                rows.append(
                    {
                        "year": year,
                        "sample_region": region.label,
                        "sido_code": region.sido_code,
                        "gugun_code": region.gugun_code,
                        "http_status": summary["http_status"],
                        "resultCode": summary["resultCode"],
                        "resultMsg": summary["resultMsg"],
                        "totalCount": summary["totalCount"],
                        "has_record": len(normalize_items(payload)) > 0,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "year": year,
                        "sample_region": region.label,
                        "sido_code": region.sido_code,
                        "gugun_code": region.gugun_code,
                        "http_status": None,
                        "resultCode": "EXCEPTION",
                        "resultMsg": type(exc).__name__,
                        "totalCount": None,
                        "has_record": False,
                    }
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(PROCESSED_DIR / "child_pedestrian_api_year_sample_detail.csv", index=False, encoding="utf-8-sig")

    summary = (
        detail.groupby("year", as_index=False)
        .agg(
            tested_region_count=("sample_region", "nunique"),
            resultCode_00_count=("resultCode", lambda s: int((s == "00").sum())),
            resultCode_03_count=("resultCode", lambda s: int((s == "03").sum())),
            actual_record_region_count=("has_record", "sum"),
        )
        .assign(nationwide_collection_candidate=lambda d: d["actual_record_region_count"] > 0)
    )
    summary.to_csv(PROCESSED_DIR / "child_pedestrian_api_year_sample_summary.csv", index=False, encoding="utf-8-sig")
    candidates = summary.loc[summary["nationwide_collection_candidate"], "year"].astype(int).tolist()
    return summary, candidates


def collect_child_pedestrian(
    api_key: str,
    years: list[int],
    regions: pd.DataFrame,
    use_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    region_objs = [
        RegionCode(str(r.sido_name), str(r.sigungu_name), str(r.sido_code).zfill(2), str(r.gugun_code).zfill(3))
        for r in regions.itertuples(index=False)
    ]

    for year in years:
        for region in region_objs:
            page = 1
            while True:
                try:
                    payload, summary = fetch_api_page(api_key, year, region, page=page, num_rows=100, use_cache=use_cache)
                except Exception as exc:
                    logs.append(
                        {
                            "year": year,
                            "sido": region.sido_name,
                            "sigungu": region.sigungu_name,
                            "sido_code": region.sido_code,
                            "gugun_code": region.gugun_code,
                            "page": page,
                            "http_status": None,
                            "resultCode": "EXCEPTION",
                            "resultMsg": type(exc).__name__,
                            "totalCount": None,
                            "record_count_in_page": 0,
                            "from_cache": False,
                        }
                    )
                    break

                logs.append(
                    {
                        "year": year,
                        "sido": region.sido_name,
                        "sigungu": region.sigungu_name,
                        "sido_code": region.sido_code,
                        "gugun_code": region.gugun_code,
                        "page": page,
                        **summary,
                    }
                )
                items = normalize_items(payload)
                for item in items:
                    item = dict(item)
                    item["_searchYearCd"] = year
                    item["_request_sido_code"] = region.sido_code
                    item["_request_gugun_code"] = region.gugun_code
                    item["_request_sido_name"] = region.sido_name
                    item["_request_sigungu_name"] = region.sigungu_name
                    records.append(item)

                total = int(payload.get("totalCount") or 0)
                num_rows = int(payload.get("numOfRows") or 100)
                if summary["resultCode"] == "03" or total <= page * num_rows or not items:
                    break
                page += 1

    raw = pd.DataFrame(records)
    log = pd.DataFrame(logs)
    if not raw.empty:
        raw = raw.drop_duplicates(["_searchYearCd", "afos_fid", "spot_cd"], keep="first")
    raw.to_csv(PROCESSED_DIR / "child_pedestrian_hotspots_raw_collected.csv", index=False, encoding="utf-8-sig")
    log.to_csv(PROCESSED_DIR / "child_pedestrian_api_collection_log.csv", index=False, encoding="utf-8-sig")
    return raw, log


def parse_sido_sigungu(value: Any) -> tuple[str | None, str | None]:
    if pd.isna(value):
        return None, None
    cleaned = re.sub(r"\d+$", "", str(value).strip())
    parts = cleaned.split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def normalize_schoolzone() -> pd.DataFrame:
    raw = pd.read_csv(SCHOOLZONE_RAW, encoding="utf-8")
    fid_unique = not raw["사고다발지fid"].duplicated().any()
    if fid_unique:
        hotspot_id = raw["사고다발지fid"].astype(str)
    else:
        hotspot_id = raw["사고다발지fid"].astype(str) + "_" + raw["지점코드"].astype(str)

    region = raw["시도시군구명"].apply(parse_sido_sigungu)
    normalized = pd.DataFrame(
        {
            "hotspot_id": hotspot_id,
            "source_type": "schoolzone_hotspot",
            "accident_year": pd.NA,
            "year_verification_status": "unknown",
            "sido": [x[0] for x in region],
            "sigungu": [x[1] for x in region],
            "location_name": raw["지점명"],
            "latitude": pd.to_numeric(raw["위도"], errors="coerce"),
            "longitude": pd.to_numeric(raw["경도"], errors="coerce"),
            "geometry": raw["다발지역폴리곤"],
            "accident_count": pd.to_numeric(raw["사고건수"], errors="coerce"),
            "casualty_count": pd.to_numeric(raw["사상자수"], errors="coerce"),
            "death_count": pd.to_numeric(raw["사망자수"], errors="coerce"),
            "serious_injury_count": pd.to_numeric(raw["중상자수"], errors="coerce"),
            "minor_injury_count": pd.to_numeric(raw["경상자수"], errors="coerce"),
            "injury_report_count": pd.to_numeric(raw["부상신고자수"], errors="coerce"),
            "source_file": "data_raw/taas_schoolzone_hotspots.csv",
            "source_record_id": raw["사고다발지fid"].astype(str),
            "source_spot_code": raw["지점코드"].astype(str),
            "source_region_name": raw["시도시군구명"],
        }
    )
    normalized.to_csv(PROCESSED_DIR / "schoolzone_hotspots_processed.csv", index=False, encoding="utf-8-sig")
    return normalized


def normalize_child_pedestrian(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        normalized = pd.DataFrame(
            columns=[
                "hotspot_id",
                "source_type",
                "accident_year",
                "year_verification_status",
                "sido",
                "sigungu",
                "location_name",
                "latitude",
                "longitude",
                "geometry",
                "accident_count",
                "casualty_count",
                "death_count",
                "serious_injury_count",
                "minor_injury_count",
                "injury_report_count",
                "source_file",
                "source_record_id",
                "source_spot_code",
                "source_region_name",
            ]
        )
    else:
        normalized = pd.DataFrame(
            {
                "hotspot_id": raw["afos_fid"].astype(str),
                "source_type": "child_pedestrian_hotspot",
                "accident_year": pd.to_numeric(raw["_searchYearCd"], errors="coerce").astype("Int64"),
                "year_verification_status": "official_api_parameter",
                "sido": raw["_request_sido_name"],
                "sigungu": raw["_request_sigungu_name"],
                "location_name": raw["spot_nm"],
                "latitude": pd.to_numeric(raw["la_crd"], errors="coerce"),
                "longitude": pd.to_numeric(raw["lo_crd"], errors="coerce"),
                "geometry": raw["geom_json"],
                "accident_count": pd.to_numeric(raw["occrrnc_cnt"], errors="coerce"),
                "casualty_count": pd.to_numeric(raw["caslt_cnt"], errors="coerce"),
                "death_count": pd.to_numeric(raw["dth_dnv_cnt"], errors="coerce"),
                "serious_injury_count": pd.to_numeric(raw["se_dnv_cnt"], errors="coerce"),
                "minor_injury_count": pd.to_numeric(raw["sl_dnv_cnt"], errors="coerce"),
                "injury_report_count": pd.to_numeric(raw["wnd_dnv_cnt"], errors="coerce"),
                "source_file": "data_raw/api_cache/child_pedestrian_hotspots",
                "source_record_id": raw["afos_fid"].astype(str),
                "source_spot_code": raw["spot_cd"].astype(str),
                "source_region_name": raw["sido_sgg_nm"],
            }
        )
    normalized.to_csv(PROCESSED_DIR / "child_pedestrian_hotspots_processed.csv", index=False, encoding="utf-8-sig")
    return normalized


def feature_properties(row: pd.Series) -> dict[str, Any]:
    value = row.to_dict()
    props = {
        "hotspot_id": value.get("hotspot_id"),
        "source_type": value.get("source_type"),
        "location_name": value.get("location_name"),
        "accident_year": None if pd.isna(value.get("accident_year")) else int(value.get("accident_year")),
        "year_verification_status": value.get("year_verification_status"),
        "accident_count": None if pd.isna(value.get("accident_count")) else int(value.get("accident_count")),
        "casualty_count": None if pd.isna(value.get("casualty_count")) else int(value.get("casualty_count")),
        "death_count": None if pd.isna(value.get("death_count")) else int(value.get("death_count")),
        "serious_injury_count": None
        if pd.isna(value.get("serious_injury_count"))
        else int(value.get("serious_injury_count")),
        "minor_injury_count": None if pd.isna(value.get("minor_injury_count")) else int(value.get("minor_injury_count")),
        "injury_report_count": None
        if pd.isna(value.get("injury_report_count"))
        else int(value.get("injury_report_count")),
        "sido": value.get("sido"),
        "sigungu": value.get("sigungu"),
    }
    return props


def write_geojson(df: pd.DataFrame) -> tuple[Path, Path]:
    point_features = []
    polygon_features = []
    for _, row in df.iterrows():
        props = feature_properties(row)
        lon = row["longitude"]
        lat = row["latitude"]
        if pd.notna(lon) and pd.notna(lat):
            point_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": props,
                }
            )
        try:
            geom = json.loads(row["geometry"]) if isinstance(row["geometry"], str) else row["geometry"]
            polygon_features.append({"type": "Feature", "geometry": geom, "properties": props})
        except Exception:
            continue

    point_path = PROCESSED_DIR / "taas_hotspots_map_points.geojson"
    polygon_path = PROCESSED_DIR / "taas_hotspots_map_polygons.geojson"
    point_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": point_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    polygon_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": polygon_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return point_path, polygon_path


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_데이터 없음_"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).values.tolist()]
    return "\n".join([header, sep, *rows])


def validate_map_data(df: pd.DataFrame, child_df: pd.DataFrame) -> pd.DataFrame:
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    korea_bounds = lon.between(124, 132) & lat.between(33, 39.5)

    geometry_errors = 0
    for geom_text in df["geometry"]:
        try:
            geom = json.loads(geom_text) if isinstance(geom_text, str) else geom_text
            if shape is not None:
                _ = shape(geom)
        except Exception:
            geometry_errors += 1

    checks = [
        ("combined_hotspot_id_duplicate_rows", int(df.duplicated("hotspot_id", keep=False).sum())),
        (
            "source_type_hotspot_id_duplicate_rows",
            int(df.duplicated(["source_type", "hotspot_id"], keep=False).sum()),
        ),
        ("coordinate_null_rows", int((lon.isna() | lat.isna()).sum())),
        ("coordinate_zero_rows", int(((lon == 0) | (lat == 0)).sum())),
        ("coordinate_outside_korea_rows", int((~korea_bounds).sum())),
        ("geometry_parse_error_rows", geometry_errors),
        ("negative_accident_count_rows", int((pd.to_numeric(df["accident_count"], errors="coerce") < 0).sum())),
        ("missing_source_type_rows", int(df["source_type"].isna().sum() + df["source_type"].astype(str).str.strip().eq("").sum())),
        (
            "missing_year_verification_status_rows",
            int(
                df["year_verification_status"].isna().sum()
                + df["year_verification_status"].astype(str).str.strip().eq("").sum()
            ),
        ),
        ("child_pedestrian_missing_accident_year_rows", int(child_df["accident_year"].isna().sum())),
    ]
    validation = pd.DataFrame(checks, columns=["check_name", "issue_count"])
    validation["status"] = validation["issue_count"].apply(lambda x: "PASS" if x == 0 else "WARN")
    validation.to_csv(PROCESSED_DIR / "taas_hotspots_map_validation.csv", index=False, encoding="utf-8-sig")
    return validation


def write_report(
    year_summary: pd.DataFrame,
    candidates: list[int],
    collection_log: pd.DataFrame,
    schoolzone_df: pd.DataFrame,
    child_df: pd.DataFrame,
    map_df: pd.DataFrame,
    validation: pd.DataFrame,
    point_path: Path,
    polygon_path: Path,
) -> Path:
    yearly_counts = (
        child_df.groupby("accident_year", dropna=False).size().reset_index(name="record_count")
        if not child_df.empty
        else pd.DataFrame(columns=["accident_year", "record_count"])
    )
    source_counts = map_df["source_type"].value_counts(dropna=False).rename_axis("source_type").reset_index(name="count")
    no_data_requests = int((collection_log["resultCode"] == "03").sum()) if not collection_log.empty else 0
    failed_requests = int((~collection_log["resultCode"].isin(["00", "03"])).sum()) if not collection_log.empty else 0

    report = [
        "# TAAS 사고다발지역 지도 레이어 수집 및 검증",
        "",
        "이 자료는 전체 어린이 교통사고 발생 원자료가 아니라, 공식 기준에 따라 선정된 사고다발지역 지점 자료다. 학교별 전체 사고건수, 학교별 실제 총 사고발생 건수, 학교별 AI 위험도, 또는 사고다발지역에 없는 학교가 안전하다는 판단으로 해석하지 않는다.",
        "",
        "## 연도 검증",
        "",
        f"- 샘플 기반 전국 수집 후보 연도: {', '.join(map(str, candidates)) if candidates else '없음'}",
        "- 후보 기준: 대표 시군구 샘플 중 하나 이상에서 `resultCode=00`과 실제 레코드가 반환된 연도.",
        "",
        markdown_table(year_summary),
        "",
        "## 전국 수집 요약",
        "",
        f"- 보행어린이 총 레코드 수: {len(child_df):,}",
        f"- 데이터 없음 요청 수(resultCode 03): {no_data_requests:,}",
        f"- 실패 요청 수(00/03 외): {failed_requests:,}",
        "",
        markdown_table(yearly_counts),
        "",
        "## 지도 레이어 요약",
        "",
        f"- 어린이보호구역 사고다발지역 행 수: {len(schoolzone_df):,}",
        f"- 지도용 통합 행 수: {len(map_df):,}",
        "",
        markdown_table(source_counts),
        "",
        "## 검증 결과",
        "",
        markdown_table(validation),
        "",
        "## 생성 파일",
        "",
        f"- 지도용 CSV: `{(PROCESSED_DIR / 'taas_hotspots_map.csv').relative_to(ROOT)}`",
        f"- Point GeoJSON: `{point_path.relative_to(ROOT)}`",
        f"- Polygon GeoJSON: `{polygon_path.relative_to(ROOT)}`",
        f"- 지역코드 목록: `{(PROCESSED_DIR / 'taas_hotspot_region_codes.csv').relative_to(ROOT)}`",
        f"- 수집 로그: `{(PROCESSED_DIR / 'child_pedestrian_api_collection_log.csv').relative_to(ROOT)}`",
        "",
        "## 남은 한계",
        "",
        "- 어린이보호구역 CSV에는 명시적 사고연도 컬럼이 없어 `accident_year`를 비워 두고 `year_verification_status='unknown'`으로 유지했다.",
        "- 보행어린이 API 연도 후보는 대표 지역 샘플 검증에 기반한다. 후보에 들지 않은 연도는 이번 원칙상 전국 수집에서 제외했다.",
        "- 서로 같은 위치에 있는 두 지점도 같은 사건으로 병합하지 않았다.",
        "",
    ]
    report_path = DOC_DIR / "taas_hotspot_map_layers.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def run(force_code_download: bool = False, no_cache: bool = False) -> dict[str, Any]:
    ensure_dirs()
    api_key = load_env_key()
    codelist_path = download_codelist(force=force_code_download)
    regions = parse_region_codes(codelist_path)
    samples = pick_sample_regions(regions)
    year_summary, candidates = verify_year_candidates(api_key, samples, use_cache=not no_cache)
    child_raw, collection_log = collect_child_pedestrian(api_key, candidates, regions, use_cache=not no_cache)
    schoolzone_df = normalize_schoolzone()
    child_df = normalize_child_pedestrian(child_raw)

    map_df = pd.concat([schoolzone_df, child_df], ignore_index=True)
    map_df["map_feature_id"] = map_df["source_type"].astype(str) + ":" + map_df["hotspot_id"].astype(str)
    map_csv = PROCESSED_DIR / "taas_hotspots_map.csv"
    map_df.to_csv(map_csv, index=False, encoding="utf-8-sig")

    point_path, polygon_path = write_geojson(map_df)
    validation = validate_map_data(map_df, child_df)
    report_path = write_report(
        year_summary,
        candidates,
        collection_log,
        schoolzone_df,
        child_df,
        map_df,
        validation,
        point_path,
        polygon_path,
    )

    by_year = (
        child_df.groupby("accident_year").size().astype(int).to_dict()
        if not child_df.empty
        else {}
    )
    by_sido = (
        child_df.groupby("sido").size().astype(int).to_dict()
        if not child_df.empty
        else {}
    )
    source_counts = map_df["source_type"].value_counts().astype(int).to_dict()

    summary = {
        "candidate_years": candidates,
        "child_pedestrian_records_by_year": {str(k): int(v) for k, v in by_year.items()},
        "child_pedestrian_records_by_sido": {str(k): int(v) for k, v in by_sido.items()},
        "child_pedestrian_total_records": int(len(child_df)),
        "schoolzone_records": int(len(schoolzone_df)),
        "schoolzone_hotspot_id_unique": not schoolzone_df["hotspot_id"].duplicated().any(),
        "child_pedestrian_hotspot_id_unique": not child_df["hotspot_id"].duplicated().any(),
        "no_data_request_count": int((collection_log["resultCode"] == "03").sum()) if not collection_log.empty else 0,
        "failed_request_count": int((~collection_log["resultCode"].isin(["00", "03"])).sum()) if not collection_log.empty else 0,
        "source_type_counts": source_counts,
        "validation_issues": validation.set_index("check_name")["issue_count"].astype(int).to_dict(),
        "files": {
            "region_codes": str(PROCESSED_DIR / "taas_hotspot_region_codes.csv"),
            "year_sample_detail": str(PROCESSED_DIR / "child_pedestrian_api_year_sample_detail.csv"),
            "year_sample_summary": str(PROCESSED_DIR / "child_pedestrian_api_year_sample_summary.csv"),
            "collection_log": str(PROCESSED_DIR / "child_pedestrian_api_collection_log.csv"),
            "child_raw": str(PROCESSED_DIR / "child_pedestrian_hotspots_raw_collected.csv"),
            "schoolzone_processed": str(PROCESSED_DIR / "schoolzone_hotspots_processed.csv"),
            "child_processed": str(PROCESSED_DIR / "child_pedestrian_hotspots_processed.csv"),
            "map_csv": str(map_csv),
            "point_geojson": str(point_path),
            "polygon_geojson": str(polygon_path),
            "validation": str(PROCESSED_DIR / "taas_hotspots_map_validation.csv"),
            "report": str(report_path),
        },
    }
    summary_path = PROCESSED_DIR / "taas_hotspots_collection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-code-download", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    summary = run(force_code_download=args.force_code_download, no_cache=args.no_cache)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
