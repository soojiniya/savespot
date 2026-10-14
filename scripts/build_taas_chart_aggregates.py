from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAAS_DIR = PROJECT_ROOT / "data_processed" / "taas"
DOC_PATH = PROJECT_ROOT / "docs" / "taas" / "taas_preprocessing_validation.md"
CHART_DIR = TAAS_DIR / "charts"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TAAS_DIR / name, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CHART_DIR / name, index=False, encoding="utf-8-sig")


def add_ratio(df: pd.DataFrame, numerator: str, denominator: pd.Series, ratio_col: str) -> pd.DataFrame:
    result = df.copy()
    result[ratio_col] = result[numerator] / result["연도"].map(denominator)
    return result


def compare_frame(name: str, expected: pd.DataFrame, actual: pd.DataFrame, key_cols: list[str], metric_cols: list[str]) -> dict:
    merged = expected.merge(actual, on=key_cols, how="outer", suffixes=("_expected", "_actual"))
    problems = []
    for _, row in merged.iterrows():
        key = ", ".join(f"{col}={row[col]}" for col in key_cols)
        for metric in metric_cols:
            expected_value = row.get(f"{metric}_expected")
            actual_value = row.get(f"{metric}_actual")
            if pd.isna(expected_value) and pd.isna(actual_value):
                continue
            if pd.isna(expected_value) or pd.isna(actual_value) or int(expected_value) != int(actual_value):
                problems.append(f"{key} {metric}: expected={expected_value}, actual={actual_value}")
    return {"검증항목": name, "상태": "통과" if not problems else "확인필요", "상세": "; ".join(problems) if problems else "일치"}


def main() -> None:
    weekday = read_csv("taas_weekday_time_2020_2025.csv")
    weekday_totals = read_csv("taas_weekday_time_totals_2020_2025.csv")
    type_detail = read_csv("taas_accident_type_time_2020_2025.csv")
    type_totals = read_csv("taas_accident_type_time_totals_2020_2025.csv")

    metric_cols = ["사고건수", "사망자수", "부상자수"]
    annual = (
        weekday_totals[(weekday_totals["요일"] == "합계") & (weekday_totals["시간대"] == "합계")]
        [["연도", *metric_cols]]
        .sort_values("연도")
        .reset_index(drop=True)
    )
    annual_denominator = annual.set_index("연도")["사고건수"]

    weekday_year = (
        weekday.groupby(["연도", "요일", "요일 정렬값"], as_index=False)["사고건수"]
        .sum()
        .sort_values(["연도", "요일 정렬값"])
        .reset_index(drop=True)
    )
    weekday_year = add_ratio(weekday_year, "사고건수", annual_denominator, "연도내_사고건수비율")

    time_year = (
        weekday.groupby(["연도", "시간대", "시작시각", "종료시각"], as_index=False)["사고건수"]
        .sum()
        .sort_values(["연도", "시작시각"])
        .reset_index(drop=True)
    )
    time_year = add_ratio(time_year, "사고건수", annual_denominator, "연도내_사고건수비율")

    weekday_time = (
        weekday[["연도", "요일", "요일 정렬값", "시간대", "시작시각", "종료시각", "사고건수"]]
        .sort_values(["연도", "요일 정렬값", "시작시각"])
        .reset_index(drop=True)
    )

    major = type_totals[
        (type_totals["시간대"] == "합계")
        & (type_totals["집계수준"].isin(["대분류소계", "대분류단일"]))
    ][["연도", "사고유형 대분류", "집계수준", "사고건수"]].copy()
    major = major.sort_values(["연도", "사고건수", "사고유형 대분류"], ascending=[True, False, True]).reset_index(drop=True)
    major = add_ratio(major, "사고건수", annual_denominator, "연도내_사고건수비율")

    car_person_total = major[major["사고유형 대분류"] == "차대사람"].set_index("연도")["사고건수"]
    car_person_middle = type_totals[
        (type_totals["시간대"] == "합계")
        & (type_totals["사고유형 대분류"] == "차대사람")
        & (type_totals["집계수준"] == "중분류상세")
    ][["연도", "사고유형 중분류", "집계수준", "사고건수"]].copy()
    car_person_middle = car_person_middle.sort_values(["연도", "사고건수", "사고유형 중분류"], ascending=[True, False, True]).reset_index(drop=True)
    car_person_middle = add_ratio(car_person_middle, "사고건수", car_person_total, "연도별_차대사람내_사고건수비율")

    write_csv(annual, "taas_chart_yearly_totals.csv")
    write_csv(weekday_year, "taas_chart_year_weekday.csv")
    write_csv(time_year, "taas_chart_year_time.csv")
    write_csv(weekday_time, "taas_chart_year_weekday_time.csv")
    write_csv(major, "taas_chart_year_accident_type_major.csv")
    write_csv(car_person_middle, "taas_chart_year_car_person_middle.csv")

    validation_rows = []
    validation_rows.append(
        compare_frame(
            "연도별 차트 총계와 검증 총계 일치",
            annual,
            weekday.groupby("연도", as_index=False)[metric_cols].sum(),
            ["연도"],
            metric_cols,
        )
    )
    validation_rows.append(
        compare_frame(
            "요일별 차트 사고건수 합계와 연도별 총계 일치",
            annual[["연도", "사고건수"]],
            weekday_year.groupby("연도", as_index=False)["사고건수"].sum(),
            ["연도"],
            ["사고건수"],
        )
    )
    validation_rows.append(
        compare_frame(
            "시간대별 차트 사고건수 합계와 연도별 총계 일치",
            annual[["연도", "사고건수"]],
            time_year.groupby("연도", as_index=False)["사고건수"].sum(),
            ["연도"],
            ["사고건수"],
        )
    )
    validation_rows.append(
        compare_frame(
            "요일x시간대 차트 사고건수 합계와 연도별 총계 일치",
            annual[["연도", "사고건수"]],
            weekday_time.groupby("연도", as_index=False)["사고건수"].sum(),
            ["연도"],
            ["사고건수"],
        )
    )
    validation_rows.append(
        compare_frame(
            "사고유형 대분류 차트 사고건수 합계와 연도별 총계 일치",
            annual[["연도", "사고건수"]],
            major.groupby("연도", as_index=False)["사고건수"].sum(),
            ["연도"],
            ["사고건수"],
        )
    )
    validation_rows.append(
        compare_frame(
            "차대사람 중분류 차트 사고건수 합계와 차대사람 총계 일치",
            major[major["사고유형 대분류"] == "차대사람"][["연도", "사고건수"]],
            car_person_middle.groupby("연도", as_index=False)["사고건수"].sum(),
            ["연도"],
            ["사고건수"],
        )
    )

    validation = pd.DataFrame(validation_rows)
    write_csv(validation, "taas_chart_aggregation_validation.csv")

    row_counts = {
        "taas_chart_yearly_totals.csv": len(annual),
        "taas_chart_year_weekday.csv": len(weekday_year),
        "taas_chart_year_time.csv": len(time_year),
        "taas_chart_year_weekday_time.csv": len(weekday_time),
        "taas_chart_year_accident_type_major.csv": len(major),
        "taas_chart_year_car_person_middle.csv": len(car_person_middle),
        "taas_chart_aggregation_validation.csv": len(validation),
    }

    top_time = time_year.loc[time_year.groupby("연도")["사고건수"].idxmax()][["연도", "시간대", "사고건수", "연도내_사고건수비율"]]
    top_weekday = weekday_year.loc[weekday_year.groupby("연도")["사고건수"].idxmax()][["연도", "요일", "사고건수", "연도내_사고건수비율"]]
    top_major = major.loc[major.groupby("연도")["사고건수"].idxmax()][["연도", "사고유형 대분류", "사고건수", "연도내_사고건수비율"]]
    top_car_person = car_person_middle.loc[
        car_person_middle.groupby("연도")["사고건수"].idxmax()
    ][["연도", "사고유형 중분류", "사고건수", "연도별_차대사람내_사고건수비율"]]

    summary = {
        "row_counts": row_counts,
        "validation": validation_rows,
        "top_time_by_year": top_time.to_dict(orient="records"),
        "top_weekday_by_year": top_weekday.to_dict(orient="records"),
        "top_major_by_year": top_major.to_dict(orient="records"),
        "top_car_person_middle_by_year": top_car_person.to_dict(orient="records"),
    }
    (CHART_DIR / "taas_chart_aggregation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    doc_addition = """

## 차트용 집계 CSV

`data_processed/taas/charts/`에 서비스 시각화용 집계를 저장했다. 모든 비율은 0~1 값이다.

- `taas_chart_yearly_totals.csv`: 연도별 사고건수, 사망자수, 부상자수. 분모 없음.
- `taas_chart_year_weekday.csv`: 연도별 요일별 사고건수와 `연도내_사고건수비율`. 분모는 같은 연도의 전체 사고건수다.
- `taas_chart_year_time.csv`: 연도별 시간대별 사고건수와 `연도내_사고건수비율`. 분모는 같은 연도의 전체 사고건수다.
- `taas_chart_year_weekday_time.csv`: 연도별 요일 x 시간대 사고건수. 비율은 사용 화면의 필요에 따라 위 두 집계의 분모를 선택하도록 별도 계산하지 않았다.
- `taas_chart_year_accident_type_major.csv`: 연도별 사고유형 대분류별 사고건수와 `연도내_사고건수비율`. 분모는 같은 연도의 전체 사고건수다. 원본의 대분류소계와 대분류단일 행만 사용한다.
- `taas_chart_year_car_person_middle.csv`: 연도별 차대사람 사고의 중분류별 사고건수와 `연도별_차대사람내_사고건수비율`. 분모는 같은 연도의 차대사람 사고건수다.
- `taas_chart_aggregation_validation.csv`: 차트용 집계가 전처리 검증 총계와 일치하는지 확인한 결과다.

총계, 소계, 세부 항목을 함께 합산하지 않았다. 사고유형의 연도별 계층 차이는 원본에 없는 항목을 만들지 않는 기존 원칙을 유지했다.
"""
    current_doc = DOC_PATH.read_text(encoding="utf-8")
    if "## 차트용 집계 CSV" not in current_doc:
        DOC_PATH.write_text(current_doc.rstrip() + doc_addition + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
