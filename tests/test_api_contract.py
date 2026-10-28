"""Minimum offline API contract checks using the project CSVs as Mongo fixtures.

These tests deliberately do not contact Atlas. They protect API response
shapes, collection counts, and the map filter domain in CI or offline review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend import main
from backend.services import assistant_tools


ROOT = Path(__file__).resolve().parents[1]


class Cursor(list[dict[str, Any]]):
    def sort(self, fields: list[tuple[str, int]]) -> "Cursor":
        for field, direction in reversed(fields):
            super().sort(
                key=lambda row: (row.get(field) is None, str(row.get(field) or "")),
                reverse=direction < 0,
            )
        return self

    def skip(self, count: int) -> "Cursor":
        return Cursor(self[count:])

    def limit(self, count: int) -> "Cursor":
        return Cursor(self[:count])


def matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(row, branch) for branch in expected):
                return False
            continue
        if isinstance(expected, dict) and "$nin" in expected:
            if row.get(key) in expected["$nin"]:
                return False
        elif isinstance(expected, dict) and "$in" in expected:
            if row.get(key) not in expected["$in"]:
                return False
        elif row.get(key) != expected:
            return False
    return True


class Collection:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def find(self, query: dict[str, Any], _projection: Any = None) -> Cursor:
        return Cursor([dict(row) for row in self.rows if matches(row, query)])

    def count_documents(self, query: dict[str, Any]) -> int:
        return sum(matches(row, query) for row in self.rows)

    def find_one(self, query: dict[str, Any], _projection: Any = None) -> dict[str, Any] | None:
        return next(iter(self.find(query)), None)

    def aggregate(self, pipeline: list[dict[str, Any]]) -> Cursor:
        rows = [dict(row) for row in self.rows]
        for stage in pipeline:
            if "$match" in stage:
                rows = [row for row in rows if matches(row, stage["$match"])]
            if "$group" in stage:
                seen = {(row.get("sido"), row.get("sigungu")) for row in rows}
                rows = [{"_id": {"sido": sido, "sigungu": sigungu}} for sido, sigungu in seen]
            if "$sort" in stage:
                rows.sort(
                    key=lambda row: (
                        str(row.get("_id", {}).get("sido") or ""),
                        str(row.get("_id", {}).get("sigungu") or ""),
                    )
                )
        return Cursor(rows)


class Database(dict[str, Collection]):
    name = "child_traffic_safety"


class Client:
    class Admin:
        @staticmethod
        def command(name: str) -> dict[str, int]:
            assert name == "ping"
            return {"ok": 1}

    admin = Admin()


def records(path: str) -> list[dict[str, Any]]:
    return pd.read_csv(ROOT / path, encoding="utf-8-sig").to_dict("records")


@pytest.fixture()
def database(monkeypatch: pytest.MonkeyPatch) -> Database:
    db = Database(
        {
            "prediction_2026_229": Collection(records("results/predicted_high_risk_2026_229.csv")),
            "infrastructure_blind_spots_49": Collection(records("results/infrastructure_blind_spots_49.csv")),
            "final_main_49": Collection(records("data_processed/final_main_49.csv")),
            "final_speed_bump_35": Collection(records("data_processed/final_speed_bump_35.csv")),
            "eda_main_49": Collection(records("data_processed/eda_main_49.csv")),
            "hotspots_map": Collection(records("data_processed/taas/hotspots/taas_hotspots_map.csv")),
        }
    )
    monkeypatch.setattr(main, "get_database", lambda: db)
    monkeypatch.setattr(main, "get_client", lambda: Client())
    monkeypatch.setattr(assistant_tools, "get_database", lambda: db)
    return db


def test_health(database: Database) -> None:
    assert main.health()["status"] == "ok"


def test_analysis_api_counts(database: Database) -> None:
    assert main.get_ai_high_risk(min_probability=0, limit=500, offset=0)["pagination"]["total"] == 229
    assert len(main.get_ai_top10(limit=10)["items"]) == 10
    assert main.get_improvement_candidates(limit=500, offset=0)["pagination"]["total"] == 49
    assert main.get_regions(sido=None, sigungu=None, include_scores=False, limit=500, offset=0)["pagination"]["total"] == 49
    assert main.get_facilities(sido=None, sigungu=None, limit=500, offset=0)["pagination"]["total"] == 49


def test_hotspots_and_complete_filter_domain(database: Database) -> None:
    hotspots = main.get_hotspots(
        source_type=None, accident_year=None, sido=None, sigungu=None, limit=500, offset=0
    )
    assert hotspots["pagination"]["total"] == 1803
    options = main.get_hotspot_filter_options(source_type=None, accident_year=None)
    source = records("data_processed/taas/hotspots/taas_hotspots_map.csv")
    expected_pairs = {
        (row["sido"], row["sigungu"])
        for row in source
        if isinstance(row.get("sido"), str) and isinstance(row.get("sigungu"), str)
    }
    # options use normalised names (경기 / 경기도 -> 경기도); every stored pair must
    # map onto exactly one offered pair
    actual_pairs = {
        (sido, sigungu)
        for sido, sigungus in options["sigungus_by_sido"].items()
        for sigungu in sigungus
    }
    normalised_pairs = {
        (main.canonical_sido(sido, sigungu), main.canonical_sigungu(sigungu))
        for sido, sigungu in expected_pairs
    }
    assert actual_pairs == normalised_pairs


def test_collection_fixture_counts(database: Database) -> None:
    expected = {
        "prediction_2026_229": 229,
        "infrastructure_blind_spots_49": 49,
        "final_main_49": 49,
        "final_speed_bump_35": 35,
        "eda_main_49": 49,
        "hotspots_map": 1803,
    }
    assert {name: collection.count_documents({}) for name, collection in database.items()} == expected


def test_assistant_facility_source_matches_api(database: Database) -> None:
    row = database["eda_main_49"].rows[0]
    region = row["sido_sgg_nm"]
    assistant_item = assistant_tools.get_facility_analysis(region)
    assert assistant_item["status"] == "ok"
    assert assistant_item["crosswalk"] == row["crosswalk_count"]
    assert assistant_item["traffic_light"] == row["traffic_light_count"]
    assert assistant_item["school_zone"] == row["schoolzone_count"]


# ---------------------------------------------------------------------------
# Map region filters: filter-options and /api/hotspots must agree
# ---------------------------------------------------------------------------
def hotspot_total(**filters: Any) -> int:
    params: dict[str, Any] = {
        "source_type": None, "accident_year": None, "sido": None, "sigungu": None, "limit": 1, "offset": 0,
    }
    params.update(filters)
    return main.get_hotspots(**params)["pagination"]["total"]


def source_rows() -> list[dict[str, Any]]:
    return records("data_processed/taas/hotspots/taas_hotspots_map.csv")


def test_hotspots_total_and_type_split(database: Database) -> None:
    assert hotspot_total() == 1803
    assert hotspot_total(source_type="schoolzone_hotspot") == 646
    assert hotspot_total(source_type="child_pedestrian_hotspot") == 1157


@pytest.mark.parametrize("empty", ["", "all", "전체", "전국", "  "])
def test_empty_region_values_are_not_filters(database: Database, empty: str) -> None:
    assert hotspot_total(sido=empty, sigungu=empty) == 1803


@pytest.mark.parametrize(
    ("sido", "short_names"),
    [("경기도", {"경기도", "경기"}), ("서울특별시", {"서울특별시", "서울"}), ("충청남도", {"충청남도", "충남"})],
)
def test_sido_filter_matches_source_rows(database: Database, sido: str, short_names: set[str]) -> None:
    expected = sum(row["sido"] in short_names for row in source_rows())
    assert expected > 0
    assert hotspot_total(sido=sido) == expected
    # the short spelling resolves to the same documents
    assert hotspot_total(sido=sorted(short_names, key=len)[0]) == expected


def test_sido_and_sigungu_filter(database: Database) -> None:
    options = main.get_hotspot_filter_options(source_type=None, accident_year=None)
    for sido in ("경기도", "서울특별시", "충청남도"):
        sigungu = options["sigungus_by_sido"][sido][0]
        total = hotspot_total(sido=sido, sigungu=sigungu)
        assert 0 < total < hotspot_total(sido=sido)
        items = main.get_hotspots(
            source_type=None, accident_year=None, sido=sido, sigungu=sigungu, limit=500, offset=0
        )["items"]
        assert len(items) == total
        assert {main.canonical_sido(i["sido"], i["sigungu"]) for i in items} == {sido}
        assert {main.canonical_sigungu(i["sigungu"]) for i in items} == {sigungu}


def test_type_and_year_combined_with_sido(database: Database) -> None:
    sido = "경기도"
    schoolzone = hotspot_total(source_type="schoolzone_hotspot", sido=sido)
    pedestrian = hotspot_total(source_type="child_pedestrian_hotspot", sido=sido)
    assert schoolzone > 0 and pedestrian > 0
    assert schoolzone + pedestrian == hotspot_total(sido=sido)
    years = {
        int(row["accident_year"])
        for row in source_rows()
        if row["source_type"] == "child_pedestrian_hotspot"
        and main.canonical_sido(row["sido"], row["sigungu"]) == sido
        and not pd.isna(row["accident_year"])
    }
    by_year = sum(
        hotspot_total(source_type="child_pedestrian_hotspot", accident_year=str(year), sido=sido) for year in years
    )
    assert by_year == pedestrian


def test_unknown_region_returns_nothing(database: Database) -> None:
    assert hotspot_total(sido="없는광역시") == 0
    assert hotspot_total(sido="경기도", sigungu="없는구") == 0


def test_every_offered_option_returns_results(database: Database) -> None:
    for source_type in (None, "schoolzone_hotspot", "child_pedestrian_hotspot"):
        options = main.get_hotspot_filter_options(source_type=source_type, accident_year=None)
        assert options["sidos"]
        for sido in options["sidos"]:
            assert hotspot_total(source_type=source_type, sido=sido) > 0, (source_type, sido)
            for sigungu in options["sigungus_by_sido"][sido]:
                assert hotspot_total(source_type=source_type, sido=sido, sigungu=sigungu) > 0, (
                    source_type, sido, sigungu,
                )


def test_offered_sidos_partition_all_hotspots(database: Database) -> None:
    options = main.get_hotspot_filter_options(source_type=None, accident_year=None)
    assert sum(hotspot_total(sido=sido) for sido in options["sidos"]) == 1803
