"""Atomically refresh the MongoDB collections used by the SaveSpot backend.

CSV source files are read-only. Each target is loaded into a staging
collection, verified, then swapped into place. If staging fails, the existing
production collection remains untouched.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo import MongoClient


ROOT = Path(__file__).resolve().parents[1]

# Keep this plan aligned with the collections consumed by FastAPI. The panel
# CSV remains a local modelling artifact because the current public API does
# not query it.
LOAD_PLAN: list[tuple[str, Path, int]] = [
    ("prediction_2026_229", ROOT / "results" / "predicted_high_risk_2026_229.csv", 229),
    ("infrastructure_blind_spots_49", ROOT / "results" / "infrastructure_blind_spots_49.csv", 49),
    ("final_main_49", ROOT / "data_processed" / "final_main_49.csv", 49),
    ("final_speed_bump_35", ROOT / "data_processed" / "final_speed_bump_35.csv", 35),
    ("eda_main_49", ROOT / "data_processed" / "eda_main_49.csv", 49),
    ("hotspots_map", ROOT / "data_processed" / "taas" / "hotspots" / "taas_hotspots_map.csv", 1803),
]


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def clean_value(value: Any) -> Any:
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def frame_to_records(frame: pd.DataFrame, collection_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        record = {str(key): clean_value(value) for key, value in row.items()}
        if collection_name == "hotspots_map":
            lat, lon = record.get("latitude"), record.get("longitude")
            record["location"] = (
                {"type": "Point", "coordinates": [float(lon), float(lat)]}
                if lat is not None and lon is not None
                else None
            )
        records.append(record)
    return records


def create_indexes(db: Any) -> None:
    db["hotspots_map"].create_index([("location", "2dsphere")], name="location_2dsphere")
    db["hotspots_map"].create_index([("source_type", 1), ("sido", 1), ("sigungu", 1)], name="filter_lookup")
    for name in ("prediction_2026_229", "infrastructure_blind_spots_49", "final_main_49", "eda_main_49"):
        db[name].create_index([("sido_sgg_nm", 1)], name="region_lookup")
    db["final_speed_bump_35"].create_index([("sido_sgg_nm", 1)], name="region_lookup")


def replace_collection(
    db: Any,
    collection_name: str,
    records: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    staging_name = f"__staging_{collection_name}_{uuid.uuid4().hex[:12]}"
    stage = db[staging_name]
    try:
        if records:
            stage.insert_many(records, ordered=True)
        staged_count = stage.count_documents({})
        if staged_count != expected_count:
            raise RuntimeError(f"COUNT_MISMATCH expected={expected_count} actual={staged_count}")
        db.client.admin.command(
            "renameCollection",
            f"{db.name}.{staging_name}",
            to=f"{db.name}.{collection_name}",
            dropTarget=True,
        )
        final_count = db[collection_name].count_documents({})
        if final_count != expected_count:
            raise RuntimeError(
                f"POST_SWAP_COUNT_MISMATCH expected={expected_count} actual={final_count}"
            )
        return {"success": True, "expected_count": expected_count, "document_count": final_count}
    except Exception:
        # Before a successful rename, only the disposable staging collection is changed.
        stage.drop()
        raise


def run() -> dict[str, Any]:
    env = load_env()
    uri, db_name = env.get("MONGODB_URI"), env.get("MONGODB_DB_NAME")
    if not uri:
        raise RuntimeError("MISSING_MONGODB_URI")
    if not db_name:
        raise RuntimeError("MISSING_MONGODB_DB_NAME")

    client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000, socketTimeoutMS=10000)
    results: list[dict[str, Any]] = []
    try:
        client.admin.command("ping")
        db = client[db_name]
        for collection_name, csv_path, expected_count in LOAD_PLAN:
            item: dict[str, Any] = {
                "collection": collection_name,
                "csv_path": str(csv_path),
                "expected_count": expected_count,
                "success": False,
            }
            try:
                if not csv_path.exists():
                    raise FileNotFoundError(csv_path.name)
                frame = pd.read_csv(csv_path, encoding="utf-8-sig")
                item["csv_rows"] = len(frame)
                if len(frame) != expected_count:
                    raise RuntimeError(
                        f"SOURCE_COUNT_MISMATCH expected={expected_count} actual={len(frame)}"
                    )
                item.update(
                    replace_collection(
                        db, collection_name, frame_to_records(frame, collection_name), expected_count
                    )
                )
            except Exception as exc:
                item["error_type"] = type(exc).__name__
                item["error"] = str(exc)
            results.append(item)
            if not item["success"]:
                raise RuntimeError(
                    f"LOAD_FAILED collection={collection_name} error_type={item['error_type']}"
                )
        create_indexes(db)
        summary = {"database": db_name, "results": results, "success": True}
    finally:
        client.close()

    (ROOT / "data_processed" / "mongodb_load_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
