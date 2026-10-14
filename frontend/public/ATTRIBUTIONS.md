# 지도 경계 데이터 출처

`korea-states-topo.json`은 KOSTAT 2018 기반으로 단순화된 17개 시도 경계 TopoJSON을 사용합니다.

- Source: https://github.com/ken2025-art/korea-map-topo
- Original source: https://github.com/SeoulTech/korea-maps
- Data source: Statistics Korea (KOSTAT), 2018
- License: repository README states the KOSTAT data is free to reuse and remix

# 랜딩 지도 데이터 (`landing/korea-scroll.json`)

랜딩페이지의 점 지도, 부산광역시 구·군 경계, 229개 시군구 중심점은 아래 시군구 경계 데이터를 단순화해 만든 시각화용 데이터입니다. 측량 수준의 정확도를 주장하지 않습니다.

- Source: npm package `korea-sigungu-geocoding` v1.3.1 (`dist/geoData.js`, 시군구 경계 GeoJSON)
- License: MIT
- 가공: 경계 단순화, 일정 간격 점 샘플링(전국 약 5.5km, 부산 약 0.47km), 시군구 중심점 계산

랜딩에 표시되는 AI 점수·순위와 사고다발지역 지점 좌표는 이 파일이 아니라 프로젝트 API(`/api/ai/high-risk`, `/api/hotspots`)에서 실행 시 불러옵니다.
