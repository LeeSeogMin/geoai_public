"""
3-2. 클라우드 네이티브 포맷 실습
===============================
GeoParquet와 일반 포맷의 성능 비교, STAC Item 메타데이터 분석을 수행한다.

실행 방법:
    cd practice/chapter3
    python -m venv venv
    source venv/bin/activate
    pip install -r code/requirements.txt
    python code/3-2-cloud-formats.py
"""

import geopandas as gpd
import tempfile
import os
import time
from shapely.geometry import box, Point
import numpy as np

# pyarrow를 미리 임포트한다.
# (지연 임포트 비용이 첫 Parquet 연산 시간에 잡혀 I/O 비교를 왜곡하는 것을 방지)
import pyarrow  # noqa: F401

print("=" * 60)
print("클라우드 네이티브 포맷 비교 실습")
print("=" * 60)

# ============================================================
# 1. 테스트 데이터 생성 — 대량 포인트
# ============================================================
print("\n--- 1. 테스트 데이터 생성 ---")

np.random.seed(42)
n_points = 50000

# 서울 지역 랜덤 포인트 생성
lons = np.random.uniform(126.8, 127.2, n_points)
lats = np.random.uniform(37.4, 37.7, n_points)

gdf = gpd.GeoDataFrame(
    {
        "id": range(n_points),
        "category": np.random.choice(["건물", "도로", "녹지", "수면"], n_points),
        "height": np.random.uniform(0, 50, n_points),
        "area_m2": np.random.uniform(10, 5000, n_points),
        "year_built": np.random.randint(1970, 2024, n_points),
    },
    geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
    crs="EPSG:4326",
)
print(f"  생성된 데이터: {len(gdf):,}개 포인트, {len(gdf.columns)}개 컬럼")

tmpdir = tempfile.mkdtemp()

# ------------------------------------------------------------
# 워밍업: 각 포맷의 드라이버(GDAL/fiona, pyarrow)를 미리 초기화한다.
# 일회성 초기화 비용을 측정에서 제외해 순수 I/O 성능을 공정하게 비교하기 위함.
# ------------------------------------------------------------
_warm = gdf.head(100)
for ext, kw in [("geojson", {"driver": "GeoJSON"}), ("shp", {}), ("parquet", None)]:
    _p = os.path.join(tmpdir, f"_warmup.{ext}")
    if kw is None:
        _warm.to_parquet(_p)
        gpd.read_parquet(_p)
    else:
        _warm.to_file(_p, **kw)
        gpd.read_file(_p)

# ============================================================
# 2. 포맷별 저장 및 파일 크기 비교
# ============================================================
print("\n--- 2. 포맷별 저장 비교 ---")

results = {}

# GeoJSON
path_json = os.path.join(tmpdir, "test.geojson")
t0 = time.time()
gdf.to_file(path_json, driver="GeoJSON")
t_json = time.time() - t0
size_json = os.path.getsize(path_json) / 1024 / 1024
results["GeoJSON"] = {"size_mb": size_json, "write_s": t_json}

# Shapefile
path_shp = os.path.join(tmpdir, "test.shp")
t0 = time.time()
gdf.to_file(path_shp)
t_shp = time.time() - t0
# shapefile은 여러 파일로 구성
shp_files = [f for f in os.listdir(tmpdir) if f.startswith("test.")]
size_shp = sum(os.path.getsize(os.path.join(tmpdir, f)) for f in shp_files) / 1024 / 1024
results["Shapefile"] = {"size_mb": size_shp, "write_s": t_shp}

# GeoParquet
path_pq = os.path.join(tmpdir, "test.parquet")
t0 = time.time()
gdf.to_parquet(path_pq)
t_pq = time.time() - t0
size_pq = os.path.getsize(path_pq) / 1024 / 1024
results["GeoParquet"] = {"size_mb": size_pq, "write_s": t_pq}

print(f"  {'포맷':12s} | {'파일 크기':>10s} | {'저장 시간':>10s}")
print(f"  {'-'*12} | {'-'*10} | {'-'*10}")
for fmt, r in results.items():
    print(f"  {fmt:12s} | {r['size_mb']:>8.2f} MB | {r['write_s']:>8.3f}초")

# ============================================================
# 3. 포맷별 읽기 성능 비교
# ============================================================
print("\n--- 3. 포맷별 읽기 성능 비교 ---")

# GeoJSON 읽기
t0 = time.time()
_ = gpd.read_file(path_json)
t_read_json = time.time() - t0

# Shapefile 읽기
t0 = time.time()
_ = gpd.read_file(path_shp)
t_read_shp = time.time() - t0

# GeoParquet 읽기
t0 = time.time()
_ = gpd.read_parquet(path_pq)
t_read_pq = time.time() - t0

print(f"  GeoJSON:    {t_read_json:.3f}초")
print(f"  Shapefile:  {t_read_shp:.3f}초")
print(f"  GeoParquet: {t_read_pq:.3f}초")

if t_read_pq > 0:
    speedup_json = t_read_json / t_read_pq
    speedup_shp = t_read_shp / t_read_pq
    print(f"\n  GeoParquet 대비 속도:")
    print(f"    vs GeoJSON:   {speedup_json:.1f}배 빠름")
    print(f"    vs Shapefile: {speedup_shp:.1f}배 빠름")

# ============================================================
# 4. GeoParquet 공간 필터링 (bbox covering 포함 저장)
# ============================================================
print("\n--- 4. GeoParquet 공간 필터링 ---")

# bbox covering 컬럼 포함하여 저장 (GeoParquet 1.1+ 기능)
path_pq_bbox = os.path.join(tmpdir, "test_bbox.parquet")
gdf.to_parquet(path_pq_bbox, write_covering_bbox=True)

# 서울 강남 영역만 읽기
gangnam_bbox = (127.0, 37.48, 127.1, 37.53)

t0 = time.time()
gdf_filtered = gpd.read_parquet(path_pq_bbox, bbox=gangnam_bbox)
t_filter = time.time() - t0

print(f"  전체 데이터: {len(gdf):,}개")
print(f"  필터링 결과: {len(gdf_filtered):,}개 (강남 영역)")
print(f"  필터링 시간: {t_filter:.3f}초")
print(f"  필터링 비율: {len(gdf_filtered)/len(gdf)*100:.1f}%")

# ============================================================
# 5. 컬럼 선택 읽기 (Column Pushdown)
# ============================================================
print("\n--- 5. 컬럼 선택 읽기 ---")

# 전체 컬럼 읽기
t0 = time.time()
_ = gpd.read_parquet(path_pq)
t_all_cols = time.time() - t0

# 2개 컬럼만 읽기
t0 = time.time()
_ = gpd.read_parquet(path_pq, columns=["category", "geometry"])
t_two_cols = time.time() - t0

print(f"  전체 컬럼 ({len(gdf.columns)}개): {t_all_cols:.3f}초")
print(f"  2개 컬럼만:            {t_two_cols:.3f}초")
if t_two_cols > 0:
    print(f"  속도 개선: {t_all_cols/t_two_cols:.1f}배")

# ============================================================
# 6. 포맷 선택 가이드 요약
# ============================================================
print("\n--- 6. 포맷 선택 가이드 ---")
print("  래스터 (2D 위성영상)     → COG (Cloud Optimized GeoTIFF)")
print("  래스터 (다차원 시계열)   → Zarr")
print("  벡터 (대용량 포인트/폴리곤) → GeoParquet")
print("  벡터 (웹 전송, 소규모)  → GeoJSON")
print("  벡터 (레거시 호환)      → Shapefile (비권장)")

# 정리
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

print("\n[완료] 클라우드 네이티브 포맷 비교 실습을 마쳤다.")
