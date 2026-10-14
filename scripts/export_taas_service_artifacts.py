from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "work" / "matplotlib_config"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


CHART_DIR = PROJECT_ROOT / "data_processed" / "taas" / "charts"
RESULT_DIR = PROJECT_ROOT / "results" / "taas"
SERVICE_DIR = PROJECT_ROOT / "data_processed" / "taas" / "service"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(CHART_DIR / name, encoding="utf-8-sig")


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def setup_korean_font() -> str:
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams.get("font.family", ["unknown"])[0]


def save_charts(yearly, weekday, time, weekday_time, major, car_person) -> dict[str, str]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(yearly["연도"], yearly["사고건수"], marker="o", linewidth=2.2, label="사고건수")
    ax1.set_title("전국 TAAS 어린이 교통사고 연도별 집계 (2020~2025, 건/명)")
    ax1.set_xlabel("연도")
    ax1.set_ylabel("사고건수(건)")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(yearly["연도"], yearly["사망자수"], marker="s", color="crimson", label="사망자수")
    ax2.plot(yearly["연도"], yearly["부상자수"], marker="^", color="darkorange", label="부상자수")
    ax2.set_ylabel("사상자수(명)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    paths["yearly_totals"] = str(RESULT_DIR / "taas_yearly_totals_2020_2025.png")
    write_png(Path(paths["yearly_totals"]))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for day, part in weekday.sort_values(["요일 정렬값", "연도"]).groupby("요일"):
        ax.plot(part["연도"], part["연도내_사고건수비율"] * 100, marker="o", label=day)
    ax.set_title("전국 TAAS 연도별 요일 사고건수 비율 (분모: 해당 연도 전체 사고건수)")
    ax.set_xlabel("연도")
    ax.set_ylabel("비율(%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    paths["weekday_ratio"] = str(RESULT_DIR / "taas_weekday_ratio_2020_2025.png")
    write_png(Path(paths["weekday_ratio"]))

    fig, ax = plt.subplots(figsize=(11, 5))
    for year, part in time.groupby("연도"):
        part = part.sort_values("시작시각")
        ax.plot(part["시간대"], part["연도내_사고건수비율"] * 100, marker="o", label=str(year))
    ax.set_title("전국 TAAS 연도별 시간대 사고건수 비율 (분모: 해당 연도 전체 사고건수)")
    ax.set_xlabel("시간대")
    ax.set_ylabel("비율(%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="연도", ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    paths["time_ratio"] = str(RESULT_DIR / "taas_time_ratio_2020_2025.png")
    write_png(Path(paths["time_ratio"]))

    selected_year = 2025
    weekday_order = weekday.sort_values("요일 정렬값")["요일"].drop_duplicates().tolist()
    time_order = time.sort_values("시작시각")["시간대"].drop_duplicates().tolist()
    heat = weekday_time[weekday_time["연도"] == selected_year].pivot(index="요일", columns="시간대", values="사고건수")
    heat = heat.reindex(index=weekday_order, columns=time_order)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    image = ax.imshow(heat.values, cmap="YlOrRd")
    ax.set_xticks(range(len(time_order)), time_order, rotation=45, ha="right")
    ax.set_yticks(range(len(weekday_order)), weekday_order)
    ax.set_title("전국 TAAS 2025년 요일 x 시간대 사고건수 (단위: 건)")
    for i, day in enumerate(weekday_order):
        for j, band in enumerate(time_order):
            ax.text(j, i, int(heat.loc[day, band]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="사고건수(건)")
    paths["weekday_time_heatmap_2025"] = str(RESULT_DIR / "taas_weekday_time_heatmap_2025.png")
    write_png(Path(paths["weekday_time_heatmap_2025"]))

    pivot_major = major.pivot(index="연도", columns="사고유형 대분류", values="연도내_사고건수비율") * 100
    ax = pivot_major.plot(kind="bar", stacked=True, figsize=(10, 4.8), colormap="tab20")
    ax.set_title("전국 TAAS 연도별 사고유형 대분류 비율 (분모: 해당 연도 전체 사고건수)")
    ax.set_xlabel("연도")
    ax.set_ylabel("비율(%)")
    ax.legend(title="사고유형 대분류", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    paths["accident_type_major_ratio"] = str(RESULT_DIR / "taas_accident_type_major_ratio_2020_2025.png")
    write_png(Path(paths["accident_type_major_ratio"]))

    pivot_cp = car_person.pivot(index="연도", columns="사고유형 중분류", values="연도별_차대사람내_사고건수비율") * 100
    ax = pivot_cp.plot(kind="bar", stacked=True, figsize=(10, 4.8), colormap="Set2")
    ax.set_title("전국 TAAS 연도별 차대사람 중분류 비율 (분모: 해당 연도 차대사람 사고건수)")
    ax.set_xlabel("연도")
    ax.set_ylabel("비율(%)")
    ax.legend(title="사고유형 중분류", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    paths["car_person_middle_ratio"] = str(RESULT_DIR / "taas_car_person_middle_ratio_2020_2025.png")
    write_png(Path(paths["car_person_middle_ratio"]))

    return paths


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def build_service_json(yearly, weekday, time, weekday_time, major, car_person) -> dict:
    years = {}
    for year in yearly["연도"].astype(int).tolist():
        total = yearly[yearly["연도"] == year].iloc[0].to_dict()
        years[str(year)] = {
            "total": {
                "accidents": int(total["사고건수"]),
                "deaths": int(total["사망자수"]),
                "injuries": int(total["부상자수"]),
            },
            "weekday": records(
                weekday[weekday["연도"] == year][["요일", "요일 정렬값", "사고건수", "연도내_사고건수비율"]]
            ),
            "time_bands": records(
                time[time["연도"] == year][["시간대", "시작시각", "종료시각", "사고건수", "연도내_사고건수비율"]]
            ),
            "weekday_time": records(
                weekday_time[weekday_time["연도"] == year][["요일", "요일 정렬값", "시간대", "시작시각", "종료시각", "사고건수"]]
            ),
            "accident_type_major": records(
                major[major["연도"] == year][["사고유형 대분류", "집계수준", "사고건수", "연도내_사고건수비율"]]
            ),
            "car_person_middle": records(
                car_person[car_person["연도"] == year][["사고유형 중분류", "집계수준", "사고건수", "연도별_차대사람내_사고건수비율"]]
            ),
        }

    return {
        "metadata": {
            "title": "TAAS 전국 어린이 교통사고 통계 서비스용 데이터",
            "scope": "전국 집계 통계",
            "period": "2020~2025",
            "population": "12세 이하 어린이 사상자가 발생한 교통사고",
            "source": "TAAS 원본 엑셀 4개(data_raw)에서 전처리한 전국 집계",
            "units": {
                "accidents": "건",
                "deaths": "명",
                "injuries": "명",
                "ratios": "0~1 실수",
            },
            "ratio_denominators": {
                "연도내_사고건수비율": "같은 연도의 전체 사고건수",
                "연도별_차대사람내_사고건수비율": "같은 연도의 차대사람 사고건수",
            },
            "limitations": [
                "지역별 수치, 학교별 수치, 좌표를 포함하지 않는다.",
                "전국 수치를 시군구별 예측 결과에 복제해 붙이지 않는다.",
                "어린이 전체 교통사고 통계이며 보행 어린이 사고만을 뜻하지 않는다.",
                "시간대별 사고건수는 개인별 사고 확률이 아니다.",
            ],
        },
        "years": years,
    }


def validate_service_json(data: dict, yearly, weekday, time, weekday_time, major, car_person) -> pd.DataFrame:
    rows = []
    for _, total in yearly.iterrows():
        year = str(int(total["연도"]))
        item = data["years"][year]
        checks = [
            ("연도별 사고건수", int(total["사고건수"]), item["total"]["accidents"]),
            ("연도별 사망자수", int(total["사망자수"]), item["total"]["deaths"]),
            ("연도별 부상자수", int(total["부상자수"]), item["total"]["injuries"]),
            ("요일 사고건수 합계", int(weekday[weekday["연도"] == int(year)]["사고건수"].sum()), sum(int(row["사고건수"]) for row in item["weekday"])),
            ("시간대 사고건수 합계", int(time[time["연도"] == int(year)]["사고건수"].sum()), sum(int(row["사고건수"]) for row in item["time_bands"])),
            ("요일x시간대 사고건수 합계", int(weekday_time[weekday_time["연도"] == int(year)]["사고건수"].sum()), sum(int(row["사고건수"]) for row in item["weekday_time"])),
            ("사고유형 대분류 사고건수 합계", int(major[major["연도"] == int(year)]["사고건수"].sum()), sum(int(row["사고건수"]) for row in item["accident_type_major"])),
            ("차대사람 중분류 사고건수 합계", int(car_person[car_person["연도"] == int(year)]["사고건수"].sum()), sum(int(row["사고건수"]) for row in item["car_person_middle"])),
        ]
        for check_name, expected, actual in checks:
            rows.append(
                {
                    "연도": int(year),
                    "검증항목": check_name,
                    "상태": "통과" if expected == actual else "확인필요",
                    "기대값": expected,
                    "실제값": actual,
                    "상세": "일치" if expected == actual else "불일치",
                }
            )

        ratio_checks = [
            ("요일 비율 합계", sum(float(row["연도내_사고건수비율"]) for row in item["weekday"])),
            ("시간대 비율 합계", sum(float(row["연도내_사고건수비율"]) for row in item["time_bands"])),
            ("사고유형 대분류 비율 합계", sum(float(row["연도내_사고건수비율"]) for row in item["accident_type_major"])),
            ("차대사람 중분류 비율 합계", sum(float(row["연도별_차대사람내_사고건수비율"]) for row in item["car_person_middle"])),
        ]
        ratio_tolerance = 1e-9
        for check_name, actual in ratio_checks:
            passed = abs(actual - 1.0) < ratio_tolerance
            rows.append(
                {
                    "연도": int(year),
                    "검증항목": check_name,
                    "상태": "통과" if passed else "확인필요",
                    "기대값": 1.0,
                    "실제값": actual,
                    "상세": "일치" if passed else "비율 합계 확인 필요",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    font_name = setup_korean_font()
    yearly = load("taas_chart_yearly_totals.csv")
    weekday = load("taas_chart_year_weekday.csv")
    time = load("taas_chart_year_time.csv")
    weekday_time = load("taas_chart_year_weekday_time.csv")
    major = load("taas_chart_year_accident_type_major.csv")
    car_person = load("taas_chart_year_car_person_middle.csv")

    chart_paths = save_charts(yearly, weekday, time, weekday_time, major, car_person)

    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    service_data = build_service_json(yearly, weekday, time, weekday_time, major, car_person)
    service_path = SERVICE_DIR / "taas_national_statistics_service.json"
    service_path.write_text(json.dumps(service_data, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = validate_service_json(service_data, yearly, weekday, time, weekday_time, major, car_person)
    validation_path = SERVICE_DIR / "taas_service_json_validation.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    validation_json_path = SERVICE_DIR / "taas_service_json_validation.json"
    validation_json_path.write_text(validation.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "font": font_name,
        "charts": chart_paths,
        "service_json": str(service_path),
        "validation_csv": str(validation_path),
        "validation_status": validation["상태"].value_counts().to_dict(),
    }
    (SERVICE_DIR / "taas_service_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
