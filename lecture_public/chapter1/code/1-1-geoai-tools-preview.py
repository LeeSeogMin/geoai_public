"""
1-1. GeoAI 도구 맛보기
=====================
GeoAI 워크플로우에서 자주 사용되는 핵심 Python 라이브러리를 살펴본다.
- GeoPandas: 벡터 데이터 처리
- 이후 장에서 Rasterio, laspy 등을 본격적으로 활용한다.

실행 방법 (프로젝트 루트의 통합 가상환경 사용):
    python -m venv .venv
    source .venv/bin/activate          # macOS/Linux
    pip install -r lecture_practice/requirements.txt
    python lecture_practice/chapter1/code/1-1-geoai-tools-preview.py
"""

import geopandas as gpd
import os
from pathlib import Path

import pandas as pd

# ============================================================
# 1. 벡터 데이터 탐색 — GeoPandas
# ============================================================
# Natural Earth 프로젝트의 110m 해상도 세계 국가 경계 데이터를 로드한다.
# 이 데이터는 177개국의 경계, 인구, 대륙 등 속성 정보를 포함한다.
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
countries = gpd.read_file(url)

print("=" * 60)
print("GeoPandas 벡터 데이터 탐색")
print("=" * 60)

# 기본 정보 확인
print(f"\n전체 국가 수: {len(countries)}")
print(f"좌표계(CRS): {countries.crs}")
print(f"기하 유형: {countries.geom_type.unique().tolist()}")

# 대륙별 국가 수 집계
continent_counts = (
    countries.groupby("CONTINENT").size().sort_values(ascending=False)
)
print("\n대륙별 국가 수:")
for continent, count in continent_counts.items():
    print(f"  {continent}: {count}개")

# ============================================================
# 2. 아시아 국가 필터링 및 공간 분석
# ============================================================
# 대륙 기준으로 아시아 국가만 추출한다.
asia = countries[countries["CONTINENT"] == "Asia"].copy()
print(f"\n아시아 국가 수: {len(asia)}")

# 면적 기준 상위 5개국 (추정 면적, 경위도 좌표계이므로 근사값)
# 정확한 면적 계산은 등면적 투영 좌표계로 변환 후 수행해야 한다.
asia["area_approx"] = asia.geometry.area
top5 = asia.nlargest(5, "area_approx")[["NAME", "POP_EST"]]
print("\n아시아 면적 상위 5개국 (근사값 기준):")
for _, row in top5.iterrows():
    pop_m = row["POP_EST"] / 1e6
    print(f"  {row['NAME']}: 인구 약 {pop_m:.0f}백만 명")

# ============================================================
# 3. 공간 데이터의 기본 속성 확인
# ============================================================
# GeoDataFrame은 pandas DataFrame에 geometry 컬럼이 추가된 구조이다.
print(f"\n데이터 형태: {countries.shape}")
print(f"메모리 사용량: {countries.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\n경계 상자(Bounding Box):")
bounds = countries.total_bounds  # [minx, miny, maxx, maxy]
print(f"  경도: {bounds[0]:.2f} ~ {bounds[2]:.2f}")
print(f"  위도: {bounds[1]:.2f} ~ {bounds[3]:.2f}")

print("\n[완료] GeoPandas를 활용한 벡터 데이터 탐색을 마쳤다.")
print("다음 장에서 좌표 변환, 공간 연산, 래스터 처리를 본격적으로 학습한다.")

# ============================================================
# 4. 결과 저장: CSV 및 GeoJSON
# ============================================================
# 결과를 저장할 폴더 (lecture_practice/chapter1/results)
results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 1) 대륙별 국가 수 -> CSV
continent_counts_df = continent_counts.to_frame(name="country_count")
continent_counts_csv = results_dir / "continent_counts.csv"
continent_counts_df.to_csv(continent_counts_csv, index=True)
print(f"저장됨: {continent_counts_csv}")

# 2) 아시아 상위 5개국 정보 -> CSV
top5_csv = results_dir / "asia_top5_by_area_approx.csv"
top5.to_csv(top5_csv, index=False)
print(f"저장됨: {top5_csv}")

# 3) 아시아 국가 GeoJSON (속성 + 지오메트리)
asia_geojson = results_dir / "asia_countries.geojson"
asia.to_file(asia_geojson, driver="GeoJSON")
print(f"저장됨: {asia_geojson}")

# 4) 전체 국가 GeoJSON (경계 데이터)
countries_geojson = results_dir / "world_countries_110m.geojson"
countries.to_file(countries_geojson, driver="GeoJSON")
print(f"저장됨: {countries_geojson}")
