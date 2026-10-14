# 어린이 교통사고 고위험지역 예측 및 교통안전 인프라 사각지대 탐지 AI

## 1. 최종 프로젝트 목표
과거 어린이 교통사고 패턴, 어린이 인구, 자동차 등록대수를 학습해 다음 연도의 **상대적 고위험 시군구**를 예측하고, 상세 교통안전시설 데이터가 확보된 49개 지역에서는 현재 인프라 수준과 비교해 **현장점검·개선 우선지역**을 제안한다.

## 2. 기존 프로젝트에서 바뀐 점
- 기존 49개 지역 회귀모델은 `11_기존49개_베이스라인모델.ipynb`로 보존했다.
- 새 메인 AI는 전국 229개 시군구의 2020~2025 연도별 데이터를 사용한다.
- 미래예측 평가에서는 랜덤 K-Fold 대신 시간 순서를 지키는 walk-forward 검증을 사용했다.
- 2026년 시설자료를 2025년 사고 예측 입력으로 사용하지 않았다. 시설자료는 AI 예측 이후 사각지대 판별 단계에서만 사용한다.

## 3. 데이터
- 어린이 교통사고: 2020~2025, 229개 시군구
- 0~12세 어린이 인구: 2020~2025, 연도별
- 자동차 등록대수: 2020~2025, 각 연도 12월 기준
- 상세 인프라: 횡단보도·신호등·보호구역·CCTV 공통 확보 49개 지역
- 과속방지턱: 35개 지역 추가자료
- TAAS 전국 집계 통계: 2020~2025 어린이 사고의 요일별·시간대별, 사고유형별·시간대별 집계

## 4. AI 입력 변수
- 직전 연도 어린이 인구
- 직전 연도 자동차 등록대수
- 직전 연도 사고/부상/사망 건수
- 직전 연도 어린이 1만 명당 사고율
- 어린이 대비 자동차 등록대수
- 사고건수 1년 증감
- 사고율 1년 증감
- 최근 2년 평균 사고건수

## 5. 고위험 정의
각 목표 연도의 어린이 1만 명당 사고율 상위 25%를 상대적 고위험지역으로 정의했다. 따라서 모델 출력은 '실제 사고가 날 확률'이라기보다 **상대적 고위험지역에 속할 가능성을 추정한 점수**로 해석하는 것이 안전하다.

## 6. 모델 평가
- Logistic Regression: ROC-AUC 0.875, PR-AUC 0.666, F1 0.644
- Random Forest: ROC-AUC 0.842, PR-AUC 0.583, F1 0.626
- HistGradientBoosting: ROC-AUC 0.822, PR-AUC 0.519, F1 0.563

최종 선택: **Logistic Regression**

## 7. 2026 AI 고위험 예측 상위 10개
1. 부산광역시 중구 — 위험확률 1.000
2. 강원특별자치도 양양군 — 위험확률 0.998
3. 경상북도 영덕군 — 위험확률 0.991
4. 강원특별자치도 태백시 — 위험확률 0.990
5. 전라남도 담양군 — 위험확률 0.989
6. 제주특별자치도 서귀포시 — 위험확률 0.989
7. 충청남도 태안군 — 위험확률 0.983
8. 충청북도 제천시 — 위험확률 0.982
9. 전북특별자치도 무주군 — 위험확률 0.980
10. 대구광역시 군위군 — 위험확률 0.969

## 8. 인프라 사각지대 개선 우선순위 상위 10개 (49개 상세지역 내)
1. 강원특별자치도 정선군 — 개선우선점수 0.757
2. 강원특별자치도 태백시 — 개선우선점수 0.750
3. 충청남도 태안군 — 개선우선점수 0.749
4. 경상남도 통영시 — 개선우선점수 0.733
5. 충청북도 제천시 — 개선우선점수 0.728
6. 경상북도 영덕군 — 개선우선점수 0.725
7. 전라남도 담양군 — 개선우선점수 0.709
8. 경상북도 상주시 — 개선우선점수 0.697
9. 전라남도 여수시 — 개선우선점수 0.638
10. 전라남도 해남군 — 개선우선점수 0.620

## 9. 실행 순서
1. 01~10: 기존 수집/전처리/EDA 재현
2. 11: 기존 49개 베이스라인 모델 확인
3. 12: 연도별 229개 패널데이터 확인
4. 13: 고위험지역 예측 AI 및 검증결과 확인
5. 14: AI 위험도 + 인프라 부족도 결합
6. 15: 최종 결과 시각화
7. 16: TAAS 전국 사고 특성 EDA 확인

## 10. TAAS 전국 집계 통계 추가 자료

`data_raw`에 보관한 TAAS 원본 엑셀 4개를 사용해 전국 어린이 교통사고 집계 통계를 별도로 전처리했다.

- 요일별·시간대별 어린이 사고: 2020~2024년, 2025년
- 사고유형별·시간대별 어린이 사고: 2020~2024년, 2025년
- 범위: 12세 이하 어린이 사상자가 발생한 전국 교통사고 집계
- 단위: 사고건수는 건, 사망자수와 부상자수는 명
- 한계: 지역명, 학교명, 좌표가 없는 전국 집계 통계

실행 방법:

```bash
python scripts/preprocess_taas_national_stats.py
python scripts/build_taas_chart_aggregates.py
```

생성 파일:

- `data_processed/taas/taas_weekday_time_2020_2025.csv`
- `data_processed/taas/taas_accident_type_time_2020_2025.csv`
- `data_processed/taas/taas_weekday_time_totals_2020_2025.csv`
- `data_processed/taas/taas_accident_type_time_totals_2020_2025.csv`
- `data_processed/taas/taas_validation_summary.csv`
- `data_processed/taas/charts/taas_chart_yearly_totals.csv`
- `data_processed/taas/charts/taas_chart_year_weekday.csv`
- `data_processed/taas/charts/taas_chart_year_time.csv`
- `data_processed/taas/charts/taas_chart_year_weekday_time.csv`
- `data_processed/taas/charts/taas_chart_year_accident_type_major.csv`
- `data_processed/taas/charts/taas_chart_year_car_person_middle.csv`
- `data_processed/taas/charts/taas_chart_aggregation_validation.csv`
- `data_processed/taas/service/taas_national_statistics_service.json`
- `data_processed/taas/service/taas_service_json_validation.csv`
- `results/taas/taas_yearly_totals_2020_2025.png`
- `results/taas/taas_weekday_ratio_2020_2025.png`
- `results/taas/taas_time_ratio_2020_2025.png`
- `results/taas/taas_weekday_time_heatmap_2025.png`
- `results/taas/taas_accident_type_major_ratio_2020_2025.png`
- `results/taas/taas_car_person_middle_ratio_2020_2025.png`
- `docs/taas/taas_preprocessing_validation.md`
- `notebook/16_TAAS_전국_사고특성_EDA.ipynb`

차트용 비율은 모두 0~1 값이다. 요일별, 시간대별, 사고유형 대분류별 비율의 분모는 같은 연도의 전체 사고건수다. 차대사람 중분류별 비율의 분모는 같은 연도의 차대사람 사고건수다. 총계, 소계, 세부 항목은 중복 집계하지 않는다.

## 11. 주의사항
- 49개 상세 인프라 지역은 전국의 무작위 표본이 아니다. 전국 대표성을 주장하지 않는다.
- 인프라 자료의 기준시점과 사고자료의 기준시점이 다르므로 인과관계를 주장하지 않는다.
- API 미제공 지역을 시설 0개로 처리하지 않는다.
- AI 결과는 정책 결정을 자동으로 내리는 도구가 아니라 현장점검 우선순위를 돕는 보조도구다.
- TAAS 전국 집계 통계는 시군구별 예측 결과에 복제해 붙이지 않는다.
- 어린이 전체 사고를 보행 어린이 사고로 표기하지 않는다.
