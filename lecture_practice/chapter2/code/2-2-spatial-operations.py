"""
2-2. 공간 연산 실습
==================
GeoPandas를 활용한 버퍼, 오버레이, 공간 조인 연산을 수행한다.
Natural Earth 세계 국가 데이터를 사용한다.

실행 방법:
    cd lecture_practice/chapter2
    python -m venv venv
    source venv/bin/activate
    pip install -r code/requirements.txt
    python code/2-2-spatial-operations.py
"""

import geopandas as gpd
from shapely.geometry import Point, box
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. 데이터 로드
# ============================================================
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
countries = gpd.read_file(url)

print("=" * 60)
print("GeoPandas 공간 연산 실습")
print("=" * 60)

# ============================================================
# 1. 버퍼(Buffer) — 포인트 주변 영역 생성
# ============================================================
print("\n--- 1. 버퍼(Buffer) ---")

# 주요 도시 포인트 (경위도)
city_data = {
    "이름": ["서울", "도쿄", "베이징", "방콕"],
    "geometry": [
        Point(127.0, 37.5),
        Point(139.7, 35.7),
        Point(116.4, 39.9),
        Point(100.5, 13.8),
    ]
}
cities = gpd.GeoDataFrame(city_data, crs="EPSG:4326")

# Mollweide 등면적 투영으로 변환 후 500km 버퍼 생성
cities_moll = cities.to_crs("+proj=moll")
cities_moll["buffer_500km"] = cities_moll.geometry.buffer(500_000)  # 500km

for _, row in cities_moll.iterrows():
    area_km2 = row["buffer_500km"].area / 1e6
    print(f"  {row['이름']}: 500km 버퍼 면적 = {area_km2:,.0f} km²")

# 이론적 면적: π × 500² ≈ 785,398 km²
import math
theoretical = math.pi * 500**2
print(f"  이론 면적 (π×500²): {theoretical:,.0f} km²")

# ============================================================
# 2. 오버레이(Overlay) — 국가 교집합
# ============================================================
print("\n--- 2. 오버레이(Overlay) ---")

# 아시아 국가 추출
asia = countries[countries["CONTINENT"] == "Asia"].copy()

# 서울 500km 버퍼와 아시아 국가의 교집합
buffer_gdf = gpd.GeoDataFrame(
    {"이름": ["서울 500km"]},
    geometry=[cities_moll[cities_moll["이름"] == "서울"]["buffer_500km"].values[0]],
    crs="+proj=moll"
)
asia_moll = asia.to_crs("+proj=moll")

intersected = gpd.overlay(asia_moll[["NAME", "geometry"]], buffer_gdf, how="intersection")
print(f"  서울 500km 반경 내 아시아 국가: {len(intersected)}개")
for _, row in intersected.iterrows():
    area_km2 = row.geometry.area / 1e6
    print(f"    - {row['NAME']}: 교집합 면적 {area_km2:,.0f} km²")

# ============================================================
# 3. 공간 조인(Spatial Join) — 포인트가 어느 나라에 속하는지
# ============================================================
print("\n--- 3. 공간 조인(Spatial Join) ---")

# 더 많은 도시 추가
more_cities_data = {
    "도시": ["서울", "도쿄", "베이징", "방콕", "뉴델리", "시드니", "런던", "파리"],
    "geometry": [
        Point(127.0, 37.5), Point(139.7, 35.7),
        Point(116.4, 39.9), Point(100.5, 13.8),
        Point(77.2, 28.6), Point(151.2, -33.9),
        Point(-0.1, 51.5), Point(2.3, 48.9),
    ]
}
more_cities = gpd.GeoDataFrame(more_cities_data, crs="EPSG:4326")

# 공간 조인: 각 도시가 어느 나라에 속하는지
joined = gpd.sjoin(
    more_cities,
    countries[["NAME", "CONTINENT", "POP_EST", "geometry"]],
    how="left",
    predicate="within"
)
print(f"  {len(joined)}개 도시 → 국가 매칭 완료")
for _, row in joined.iterrows():
    pop_m = row["POP_EST"] / 1e6
    print(f"    {row['도시']} → {row['NAME']} ({row['CONTINENT']}, 인구 {pop_m:.0f}백만)")

# ============================================================
# 4. 공간 필터링 — Bounding Box로 영역 추출
# ============================================================
print("\n--- 4. 공간 필터링 (Bounding Box) ---")

# 동아시아 영역 (경도 100-145, 위도 20-50)
east_asia_box = box(100, 20, 145, 50)
east_asia = countries[countries.intersects(east_asia_box)]
print(f"  동아시아 영역(100-145°E, 20-50°N) 내 국가: {len(east_asia)}개")
for _, row in east_asia.iterrows():
    print(f"    - {row['NAME']}")

# ============================================================
# 5. 면적 계산 — 좌표계의 중요성
# ============================================================
print("\n--- 5. 좌표계에 따른 면적 차이 ---")

korea = countries[countries["NAME"] == "South Korea"].copy()
# 경위도 좌표계에서 면적 (부정확: 도² 단위)
area_4326 = korea.geometry.area.values[0]
# 등면적 투영에서 면적 (정확)
korea_moll = korea.to_crs("+proj=moll")
area_moll = korea_moll.geometry.area.values[0] / 1e6  # km²

print(f"  한국 면적:")
print(f"    EPSG:4326 (경위도): {area_4326:.4f} (도² — 의미 없음)")
print(f"    Mollweide (등면적): {area_moll:,.0f} km²")
print(f"    실제 면적 참고값:   ~100,210 km²")

print("\n[완료] 공간 연산 실습을 마쳤다.")
