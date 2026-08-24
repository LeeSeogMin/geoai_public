"""
4-1. 공간 피처 추출 실습
========================
건물 필지 데이터에서 형태학적·위상적·거리 기반 피처를 추출한다.
데이터는 미리 준비한 건물 필지·지하철역 파일을 불러와 사용한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python lecture_practice/chapter4/code/4-0-simdata-prep.py   # 최초 1회: 데이터 준비
    python lecture_practice/chapter4/code/4-1-spatial-features.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILDINGS_PATH = DATA_DIR / "buildings.geojson"
if not BUILDINGS_PATH.exists():
    raise SystemExit(
        f"데이터가 없습니다: {BUILDINGS_PATH}\n"
        "먼저 실행: python 4-0-simdata-prep.py"
    )

print("=" * 60)
print("공간 피처 추출 실습")
print("=" * 60)

# ============================================================
# 1. 건물 필지 데이터 불러오기
# ============================================================
print("\n--- 1. 건물 필지 데이터 불러오기 ---")

# 서울 강남 지역 (UTM 52N 좌표, 미터 단위)
base_x, base_y = 323000, 4160000  # 대략 강남역 부근

gdf = gpd.read_file(BUILDINGS_PATH)
n_buildings = len(gdf)
print(f"  불러온 건물: {len(gdf)}개 (좌표계 {gdf.crs})")

# ============================================================
# 2. 기하 유효성 검사
# ============================================================
print("\n--- 2. 기하 유효성 검사 ---")

invalid_count = (~gdf.geometry.is_valid).sum()
print(f"  유효하지 않은 기하 객체: {invalid_count}개")
gdf["geometry"] = gdf.geometry.make_valid()
print(f"  수정 후 유효하지 않은 객체: {(~gdf.geometry.is_valid).sum()}개")

# ============================================================
# 3. 형태학적 피처 추출
# ============================================================
print("\n--- 3. 형태학적 피처 ---")

gdf["area_m2"] = gdf.geometry.area
gdf["perimeter_m"] = gdf.geometry.length
gdf["compactness"] = 4 * np.pi * gdf["area_m2"] / (gdf["perimeter_m"] ** 2)

def calc_aspect_ratio(geom):
    minx, miny, maxx, maxy = geom.bounds
    width, height = maxx - minx, maxy - miny
    return max(width, height) / max(min(width, height), 1e-6)

gdf["aspect_ratio"] = gdf.geometry.apply(calc_aspect_ratio)

print(f"  면적 범위: {gdf['area_m2'].min():.1f} ~ {gdf['area_m2'].max():.1f} m²")
print(f"  둘레 범위: {gdf['perimeter_m'].min():.1f} ~ {gdf['perimeter_m'].max():.1f} m")
print(f"  조밀도 범위: {gdf['compactness'].min():.3f} ~ {gdf['compactness'].max():.3f}")
print(f"  종횡비 범위: {gdf['aspect_ratio'].min():.2f} ~ {gdf['aspect_ratio'].max():.2f}")

# ============================================================
# 4. 거리 기반 피처
# ============================================================
print("\n--- 4. 거리 기반 피처 ---")

# 지하철역 위치 불러오기
stations = gpd.read_file(DATA_DIR / "stations.geojson")

# 최근접 지하철역 거리
def nearest_station_dist(geom, stations_gdf):
    centroid = geom.centroid
    distances = stations_gdf.geometry.distance(centroid)
    return distances.min()

gdf["dist_to_station"] = gdf.geometry.apply(
    lambda g: nearest_station_dist(g, stations)
)

# 중심점(base_x, base_y)까지 거리
center = Point(base_x, base_y)
gdf["dist_to_center"] = gdf.geometry.centroid.distance(center)

print(f"  지하철역 거리: {gdf['dist_to_station'].min():.0f} ~ {gdf['dist_to_station'].max():.0f} m")
print(f"  중심 거리: {gdf['dist_to_center'].min():.0f} ~ {gdf['dist_to_center'].max():.0f} m")

# ============================================================
# 5. 위상적 피처 (10m 버퍼 내 인접 건물 수)
# ============================================================
print("\n--- 5. 위상적 피처 (인접 건물 수) ---")

# 효율적 인접 건물 수 계산 (STRtree 활용)
from shapely import STRtree

tree = STRtree(gdf.geometry.values)
buffer_dist = 50  # 50m

neighbor_counts = []
for geom in gdf.geometry:
    buffered = geom.buffer(buffer_dist)
    idxs = tree.query(buffered)
    # 자기 자신 제외
    count = sum(1 for i in idxs if gdf.geometry.iloc[i].intersects(buffered)) - 1
    neighbor_counts.append(max(count, 0))

gdf["neighbors_50m"] = neighbor_counts

print(f"  50m 내 인접 건물 수: {gdf['neighbors_50m'].min()} ~ {gdf['neighbors_50m'].max()}")
print(f"  평균 인접 건물 수: {gdf['neighbors_50m'].mean():.1f}")

# ============================================================
# 6. 피처 요약 통계
# ============================================================
print("\n--- 6. 피처 요약 ---")

features = ["area_m2", "perimeter_m", "compactness", "aspect_ratio",
            "dist_to_station", "dist_to_center", "neighbors_50m"]

print(f"  {'피처':18s} | {'평균':>10s} | {'표준편차':>10s} | {'최소':>10s} | {'최대':>10s}")
print(f"  {'-'*18} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
for feat in features:
    vals = gdf[feat]
    print(f"  {feat:18s} | {vals.mean():>10.1f} | {vals.std():>10.1f} | {vals.min():>10.1f} | {vals.max():>10.1f}")

print(f"\n  총 {len(features)}개 공간 피처 추출 완료")
print("\n[완료] 공간 피처 추출 실습을 마쳤다.")
