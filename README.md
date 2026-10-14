# SaveSpot

## 어린이 교통사고 고위험지역 예측 및 교통안전 인프라 사각지대 탐지 AI

어린이 교통사고, 어린이 인구, 자동차 등록대수 기반의 지역 단위 AI 분석과 공식 사고다발지역 지도, 시설 상세분석을 함께 제공하는 데이터 탐색 서비스입니다.

- 배포 주소: https://savespot-amber.vercel.app
- 공개 데모: AI 자연어 분석 기능은 공개 배포 환경에서 비활성화되어 있습니다.

## 기술 스택

- Frontend: React + Vite
- Backend: FastAPI
- Database: MongoDB Atlas
- Modeling: scikit-learn

## 1. 최종 프로젝트 목표

과거 어린이 교통사고 패턴, 어린이 인구, 자동차 등록대수를 학습해 다음 연도의 **상대적 고위험 시군구**를 분석하고, 상세 교통안전시설 데이터가 확보된 49개 지역에서는 인프라 부족 신호와 함께 **현장점검·개선 검토 후보**를 제시합니다.

이 서비스의 AI 모델 출력은 실제 사고 발생 확률이 아닙니다. 다음 연도 어린이 1만 명당 사고율 기준 전국 상위 25% 상대적 고위험지역에 속할 가능성을 나타내는 **AI 상대적 고위험 점수**이며, 지역 간 비교를 위한 분석 점수입니다.

## 2. 분석 범위와 데이터 의미

| 구분 | 범위 | 의미 |
| --- | --- | --- |
| AI 분석 | 전국 229개 시군구 | 2020~2025 어린이 교통사고, 어린이 인구, 자동차 등록대수 기반의 지역 단위 분석 |
| 현황지도 | 공식 사고다발지역 선정지점 1,803개 | `schoolzone_hotspot` 646개 + `child_pedestrian_hotspot` 1,157개. 전체 어린이 교통사고 건수나 전체 사고 위치 목록이 아님 |
| 시설 상세분석 | 49개 지역 | 횡단보도·신호등·어린이보호구역·CCTV 4종 시설 데이터가 공통 확보된 지역만 포함 |

49개 시설 상세분석 지역은 229개 AI 분석 범위의 하위 데이터 범위입니다. 49개 밖 지역은 시설이 0개인 것이 아니라 상세 시설 비교 데이터가 미제공된 상태입니다.

## 3. 기존 프로젝트에서 바뀐 점

- 기존 49개 지역 회귀모델은 `11_기존49개_베이스라인모델.ipynb`로 보존했습니다.
- 새 메인 AI는 전국 229개 시군구의 2020~2025 연도별 데이터를 사용합니다.
- 미래예측 평가에서는 랜덤 K-Fold 대신 시간 순서를 지키는 walk-forward 검증을 사용했습니다.
- 2026년 시설자료를 2025년 사고 예측 입력으로 사용하지 않았습니다. 시설자료는 AI 예측 이후 사각지대 판별 단계에서만 사용합니다.

## 4. 데이터

- 어린이 교통사고: 2020~2025, 229개 시군구
- 0~12세 어린이 인구: 2020~2025, 연도별
- 자동차 등록대수: 2020~2025, 각 연도 12월 기준
- 상세 인프라: 횡단보도·신호등·어린이보호구역·CCTV 공통 확보 49개 지역
- 과속방지턱: 35개 지역 추가자료
- TAAS 전국 집계 통계: 2020~2025 어린이 사고의 요일별·시간대별, 사고유형별·시간대별 집계

## 5. AI 입력 변수와 고위험 정의

### 입력 변수

- 직전 연도 어린이 인구
- 직전 연도 자동차 등록대수
- 직전 연도 사고·부상·사망 건수
- 직전 연도 어린이 1만 명당 사고율
- 어린이 대비 자동차 등록대수
- 사고건수 1년 증감
- 사고율 1년 증감
- 최근 2년 평균 사고건수

### 고위험 정의와 해석

각 목표 연도의 어린이 1만 명당 사고율 상위 25%를 상대적 고위험지역으로 정의했습니다. 모델은 다음 연도에 해당 지역이 이 상대적 고위험군에 속할 가능성을 **AI 상대적 고위험 점수**로 산출합니다.

- AI 상대적 고위험 점수는 실제 사고 발생 확률이 아닙니다.
- 전국 AI 순위는 공식 교통안전 평가 순위가 아니라 프로젝트의 분석용 순위입니다.
- 학교별 위험도나 사고다발지역 지점별 AI 위험도를 산출하지 않습니다.

## 6. 모델 평가

`results/model_average_metrics.csv`의 평균 성능 기준입니다.

- Logistic Regression: ROC-AUC 0.875, PR-AUC 0.666, F1 0.644, Recall 0.816
- Random Forest: ROC-AUC 0.843, PR-AUC 0.586, F1 0.631, Recall 0.799
- HistGradientBoosting: ROC-AUC 0.822, PR-AUC 0.519, F1 0.563, Recall 0.592

최종 선택 모델: **Logistic Regression**

## 7. 2026 AI 상대적 고위험 점수 상위 10개

`results/predicted_high_risk_2026_229.csv` 기준입니다.

| 순위 | 지역 | AI 상대적 고위험 점수 |
| ---: | --- | ---: |
| 1 | 부산광역시 중구 | 1.000 |
| 2 | 강원특별자치도 양양군 | 0.996 |
| 3 | 제주특별자치도 서귀포시 | 0.989 |
| 4 | 경상북도 영덕군 | 0.985 |
| 5 | 강원특별자치도 태백시 | 0.983 |
| 6 | 전라남도 담양군 | 0.982 |
| 7 | 충청북도 제천시 | 0.977 |
| 8 | 충청남도 태안군 | 0.975 |
| 9 | 전북특별자치도 무주군 | 0.966 |
| 10 | 서울특별시 중구 | 0.955 |

## 8. 인프라 사각지대 개선 검토 후보 상위 10개

`results/infrastructure_blind_spots_49.csv` 기준이며, 49개 시설 상세분석 지역 안에서만 산출합니다.

개선 검토 점수는 공식 정책 점수나 확정 우선순위가 아닙니다. AI 상대적 고위험 점수와 인프라 부족도를 결합해 현장 검토 대상을 좁히기 위한 **분석용 휴리스틱 점수**입니다.

| 순위 | 지역 | 개선 검토 점수 |
| ---: | --- | ---: |
| 1 | 강원특별자치도 정선군 | 0.747 |
| 2 | 강원특별자치도 태백시 | 0.747 |
| 3 | 충청남도 태안군 | 0.745 |
| 4 | 경상남도 통영시 | 0.728 |
| 5 | 충청북도 제천시 | 0.726 |
| 6 | 경상북도 영덕군 | 0.722 |
| 7 | 전라남도 담양군 | 0.705 |
| 8 | 경상북도 상주시 | 0.688 |
| 9 | 전라남도 여수시 | 0.647 |
| 10 | 전라남도 해남군 | 0.612 |

## 9. 로컬 실행 방법

### 백엔드

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

MongoDB Atlas와 LLM을 사용하는 환경에서는 필요한 환경변수를 로컬 `.env`에 설정합니다. 비밀 키와 연결 문자열은 저장소에 포함하거나 출력하지 않습니다.

### 프론트엔드

새 터미널에서 실행합니다.

```bash
cd frontend
npm install
npm run dev
```

기본 개발 주소는 `http://127.0.0.1:5173`이며, 백엔드는 `http://127.0.0.1:8000`을 사용합니다.

## 10. 분석·재현 실행 순서

1. 01~10: 기존 수집/전처리/EDA 재현
2. 11: 기존 49개 베이스라인 모델 확인
3. 12: 연도별 229개 패널데이터 확인
4. 13: 고위험지역 예측 AI 및 검증결과 확인
5. 14: AI 상대적 고위험 점수와 인프라 부족도 결합
6. 15: 최종 결과 시각화
7. 16: TAAS 전국 사고 특성 EDA 확인

## 11. TAAS 전국 집계 통계 추가 자료

`data_raw`에 보관한 TAAS 원본 엑셀 4개를 사용해 전국 어린이 교통사고 집계 통계를 별도로 전처리했습니다.

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

차트용 비율은 모두 0~1 값입니다. 요일별, 시간대별, 사고유형 대분류별 비율의 분모는 같은 연도의 전체 사고건수입니다. 차대사람 중분류별 비율의 분모는 같은 연도의 차대사람 사고건수입니다. 총계, 소계, 세부 항목은 중복 집계하지 않습니다.

## 12. 주의사항

- 49개 상세 인프라 지역은 전국의 무작위 표본이 아니며, 전국 대표성을 주장하지 않습니다.
- 인프라 자료의 기준시점과 사고자료의 기준시점이 다르므로 인과관계를 주장하지 않습니다.
- 시설 상세분석 미제공 지역을 시설 0개로 처리하지 않습니다.
- 공식 사고다발지역 선정지점은 전체 어린이 교통사고 건수나 전체 사고 위치 목록이 아닙니다.
- 사고다발지역에 선정되지 않았다고 안전한 지역을 의미하지 않습니다.
- AI 결과는 정책 결정을 자동으로 내리는 도구가 아니라 현장점검과 추가 검토를 돕는 보조도구입니다.
- 개선 검토 점수는 공식 정책 우선순위가 아닙니다.
- TAAS 전국 집계 통계는 시군구별 예측 결과에 복제해 붙이지 않습니다.
- 어린이 전체 사고를 보행 어린이 사고로 표기하지 않습니다.
