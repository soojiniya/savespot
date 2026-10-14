# TAAS 사고다발지역 지도 레이어 수집 및 검증

이 자료는 전체 어린이 교통사고 발생 원자료가 아니라, 공식 기준에 따라 선정된 사고다발지역 지점 자료다. 학교별 전체 사고건수, 학교별 실제 총 사고발생 건수, 학교별 AI 위험도, 또는 사고다발지역에 없는 학교가 안전하다는 판단으로 해석하지 않는다.

## 연도 검증

- 샘플 기반 전국 수집 후보 연도: 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
- 후보 기준: 대표 시군구 샘플 중 하나 이상에서 `resultCode=00`과 실제 레코드가 반환된 연도.

| year | tested_region_count | resultCode_00_count | resultCode_03_count | actual_record_region_count | nationwide_collection_candidate |
| --- | --- | --- | --- | --- | --- |
| 2012 | 5 | 2 | 3 | 2 | True |
| 2013 | 5 | 4 | 1 | 4 | True |
| 2014 | 5 | 4 | 1 | 4 | True |
| 2015 | 5 | 3 | 2 | 3 | True |
| 2016 | 5 | 2 | 3 | 2 | True |
| 2017 | 5 | 4 | 1 | 4 | True |
| 2018 | 5 | 4 | 1 | 4 | True |
| 2019 | 5 | 4 | 1 | 4 | True |
| 2020 | 5 | 2 | 3 | 2 | True |
| 2021 | 5 | 3 | 2 | 3 | True |
| 2022 | 5 | 2 | 3 | 2 | True |
| 2023 | 5 | 4 | 1 | 4 | True |
| 2024 | 5 | 4 | 1 | 4 | True |
| 2025 | 5 | 2 | 3 | 2 | True |

## 전국 수집 요약

- 보행어린이 총 레코드 수: 1,157
- 데이터 없음 요청 수(resultCode 03): 3,359
- 실패 요청 수(00/03 외): 0

| accident_year | record_count |
| --- | --- |
| 2012 | 176 |
| 2013 | 132 |
| 2014 | 112 |
| 2015 | 107 |
| 2016 | 75 |
| 2017 | 72 |
| 2018 | 73 |
| 2019 | 76 |
| 2020 | 27 |
| 2021 | 35 |
| 2022 | 54 |
| 2023 | 56 |
| 2024 | 87 |
| 2025 | 75 |

## 지도 레이어 요약

- 어린이보호구역 사고다발지역 행 수: 646
- 지도용 통합 행 수: 1,803

| source_type | count |
| --- | --- |
| child_pedestrian_hotspot | 1157 |
| schoolzone_hotspot | 646 |

## 검증 결과

| check_name | issue_count | status |
| --- | --- | --- |
| combined_hotspot_id_duplicate_rows | 0 | PASS |
| source_type_hotspot_id_duplicate_rows | 0 | PASS |
| coordinate_null_rows | 0 | PASS |
| coordinate_zero_rows | 0 | PASS |
| coordinate_outside_korea_rows | 0 | PASS |
| geometry_parse_error_rows | 0 | PASS |
| negative_accident_count_rows | 0 | PASS |
| missing_source_type_rows | 0 | PASS |
| missing_year_verification_status_rows | 0 | PASS |
| child_pedestrian_missing_accident_year_rows | 0 | PASS |

## 생성 파일

- 지도용 CSV: `data_processed\taas\hotspots\taas_hotspots_map.csv`
- Point GeoJSON: `data_processed\taas\hotspots\taas_hotspots_map_points.geojson`
- Polygon GeoJSON: `data_processed\taas\hotspots\taas_hotspots_map_polygons.geojson`
- 지역코드 목록: `data_processed\taas\hotspots\taas_hotspot_region_codes.csv`
- 수집 로그: `data_processed\taas\hotspots\child_pedestrian_api_collection_log.csv`

## 남은 한계

- 어린이보호구역 CSV에는 명시적 사고연도 컬럼이 없어 `accident_year`를 비워 두고 `year_verification_status='unknown'`으로 유지했다.
- 보행어린이 API 연도 후보는 대표 지역 샘플 검증에 기반한다. 후보에 들지 않은 연도는 이번 원칙상 전국 수집에서 제외했다.
- 서로 같은 위치에 있는 두 지점도 같은 사건으로 병합하지 않았다.
