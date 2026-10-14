"""Load prepared project CSV files into MongoDB Atlas.

Safety rules:
- never deletes, drops, updates, or overwrites collections
- inserts only when the target collection is empty
- does not print URI or credentials
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError


ROOT = Path(__file__).resolve().parents[1]

LOAD_PLAN = [
    ("final_main_49", ROOT / "data_processed" / "final_main_49.csv"),
    ("final_speed_bump_35", ROOT / "data_processed" / "final_speed_bump_35.csv"),
    ("eda_main_49", ROOT / "data_processed" / "eda_main_49.csv"),
    ("hotspots_map", ROOT / "data_processed" / "taas" / "hotspots" / "taas_hotspots_map.csv"),
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
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def frame_to_records(df: pd.DataFrame, collection_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        record = {str(k): clean_value(v) for k, v in row.items()}
        if collection_name == "hotspots_map":
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                record["location"] = {"type": "Point", "coordinates": [float(lon), float(lat)]}
            else:
                record["location"] = None
        records.append(record)
    return records


def duplicate_summary(df: pd.DataFrame, collection_name: str) -> dict[str, int]:
    out = {"full_duplicate_rows": int(df.duplicated().sum())}
    if collection_name == "hotspots_map" and "map_feature_id" in df.columns:
        out["map_feature_id_duplicate_rows"] = int(df.duplicated("map_feature_id", keep=False).sum())
    if "sido_sgg_nm" in df.columns:
        out["sido_sgg_nm_duplicate_rows"] = int(df.duplicated("sido_sgg_nm", keep=False).sum())
    return out


def run() -> dict[str, Any]:
    env = load_env()
    uri = env.get("MONGODB_URI")
    db_name = env.get("MONGODB_DB_NAME")
    if not uri:
        raise RuntimeError("MISSING_MONGODB_URI")
    if not db_name:
        raise RuntimeError("MISSING_MONGODB_DB_NAME")

    client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000, socketTimeoutMS=10000)
    client.admin.command("ping")
    db = client[db_name]

    results: list[dict[str, Any]] = []
    loaded_collections: list[str] = []

    for collection_name, csv_path in LOAD_PLAN:
        item: dict[str, Any] = {
            "collection": collection_name,
            "csv_path": str(csv_path),
            "csv_exists": csv_path.exists(),
            "csv_rows": None,
            "csv_columns": [],
            "pre_existing_document_count": None,
            "inserted_count": 0,
            "post_insert_document_count": None,
            "success": False,
            "skipped": False,
            "error_type": None,
        }
        try:
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            item["csv_rows"] = int(len(df))
            item["csv_columns"] = list(df.columns)
            item["duplicates"] = duplicate_summary(df, collection_name)

            collection = db[collection_name]
            existing_count = collection.count_documents({})
            item["pre_existing_document_count"] = int(existing_count)
            if existing_count != 0:
                item["skipped"] = True
                item["error_type"] = "COLLECTION_NOT_EMPTY"
                item["post_insert_document_count"] = int(existing_count)
                results.append(item)
                continue

            records = frame_to_records(df, collection_name)
            if records:
                insert_result = collection.insert_many(records, ordered=True)
                item["inserted_count"] = len(insert_result.inserted_ids)
            item["post_insert_document_count"] = int(collection.count_documents({}))
            item["success"] = item["post_insert_document_count"] == item["csv_rows"]
            if item["success"]:
                loaded_collections.append(collection_name)
        except Exception as exc:
            item["error_type"] = type(exc).__name__
        results.append(item)

    index_created = False
    index_error_type = None
    if any(item["collection"] == "hotspots_map" and item["success"] for item in results):
        try:
            db["hotspots_map"].create_index([("location", "2dsphere")], name="location_2dsphere")
            index_created = True
        except Exception as exc:
            index_error_type = type(exc).__name__

    hotspots_source_counts: dict[str, int] = {}
    if "hotspots_map" in db.list_collection_names():
        pipeline = [{"$group": {"_id": "$source_type", "count": {"$sum": 1}}}]
        hotspots_source_counts = {str(row["_id"]): int(row["count"]) for row in db["hotspots_map"].aggregate(pipeline)}

    final_collections = sorted(db.list_collection_names())
    client.close()

    summary = {
        "db_name": db_name,
        "results": results,
        "hotspots_map_source_type_counts": hotspots_source_counts,
        "hotspots_map_location_point_created": any(
            item["collection"] == "hotspots_map" and item["success"] for item in results
        ),
        "hotspots_map_2dsphere_index_created": index_created,
        "hotspots_map_2dsphere_index_error_type": index_error_type,
        "final_collections": final_collections,
    }

    out_path = ROOT / "data_processed" / "mongodb_load_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
