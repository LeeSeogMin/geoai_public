"""
4-0. 실습 데이터 준비: 공간단위 분석용 데이터셋 저장
====================================================
이 장의 4-1·4-2·4-5 실습이 쓰는 데이터셋을 미리 만들어 `data/` 폴더에
저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고,
이후 각 실습 코드는 저장된 파일을 불러와 분석만 수행한다.

여기서 만드는 세 데이터는 실제 공개 데이터를 그대로 구하기 어려운 대상이라
현실적인 값과 공간 구조를 갖도록 합성한 교육용 데이터다(개인 식별 불가).
- 건물 필지(형태·입지 피처 실습, 4-1)
- 공간 포인트 관측(공간 자기상관·공간 CV 실습, 4-2)
- 격자 사회경제·취약성(정책 취약성 예측 실습, 4-5)

위성영상처럼 시뮬레이션이 어려운 데이터는 4-0-data-download.py가 실제로
내려받는다. 두 준비 스크립트를 모두 실행하면 이 장의 데이터가 갖춰진다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter4/code/4-0-simdata-prep.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.affinity import rotate
from shapely.geometry import Point, box

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 공통 기준 좌표계: UTM 52N (미터 단위)
CRS = "EPSG:32652"


def prepare_buildings():
    """건물 필지와 지하철역 위치를 만들어 저장한다(4-1 형태·거리 피처)."""
    np.random.seed(42)
    n_buildings = 200
    base_x, base_y = 323000, 4160000  # 대략 강남역 부근

    buildings = []
    for _ in range(n_buildings):
        cx = base_x + np.random.uniform(-3000, 3000)
        cy = base_y + np.random.uniform(-3000, 3000)
        w = np.random.uniform(10, 60)
        h = np.random.uniform(10, 60)
        angle = np.random.uniform(0, 45)
        buildings.append(rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), angle))

    gdf = gpd.GeoDataFrame({"id": range(n_buildings)}, geometry=buildings, crs=CRS)
    gdf.to_file(DATA_DIR / "buildings.geojson", driver="GeoJSON")

    stations = gpd.GeoDataFrame(
        {"name": ["역A", "역B", "역C"]},
        geometry=[Point(base_x - 1000, base_y),
                  Point(base_x + 1500, base_y + 500),
                  Point(base_x, base_y - 2000)],
        crs=CRS)
    stations.to_file(DATA_DIR / "stations.geojson", driver="GeoJSON")
    print(f"  건물 {len(gdf)}개 → buildings.geojson, 지하철역 {len(stations)}개 → stations.geojson")


def prepare_spatial_points():
    """공간 트렌드+핫스팟을 가진 관측 포인트를 만들어 저장한다(4-2)."""
    np.random.seed(42)
    n_points = 500
    x = np.random.uniform(310000, 340000, n_points)
    y = np.random.uniform(4150000, 4180000, n_points)

    x_norm = (x - x.min()) / (x.max() - x.min())
    spatial_trend = x_norm * 50

    centers = np.array([[320000, 4165000], [315000, 4155000], [330000, 4170000]])
    intensities = [30, 25, 35]
    cluster_effect = np.zeros(n_points)
    for (cx, cy), intensity in zip(centers, intensities):
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        cluster_effect += intensity * np.exp(-dist / 3000)

    target = spatial_trend + cluster_effect + np.random.normal(0, 5, n_points)
    gdf = gpd.GeoDataFrame(
        {"target": target, "x_coord": x, "y_coord": y},
        geometry=[Point(xi, yi) for xi, yi in zip(x, y)], crs=CRS)
    gdf.to_file(DATA_DIR / "spatial_points.geojson", driver="GeoJSON")
    print(f"  공간 포인트 {len(gdf)}개 → spatial_points.geojson")


def prepare_grid():
    """격자 사회경제·접근성 지표와 취약성(관측 피처 + 잠재 공간장)을 저장한다(4-5).

    취약성에는 관측 피처로 설명되는 부분과, 피처에 없는 잠재 공간장(지역 고유의
    공동체·역사·미세환경 등)이 함께 들어간다. 이 잠재 구조가 뒤의 데이터 누수
    시연에서 좌표 보간으로 회복되는 '누수의 원천'이 된다.
    """
    rng = np.random.default_rng(42)
    grid = 20
    n = grid * grid
    rows, cols = np.divmod(np.arange(n), grid)
    xs = cols / (grid - 1)
    ys = rows / (grid - 1)
    U = np.sin(rows / 3.0) + np.cos(cols / 2.6) + rng.normal(0, 0.3, n)
    U = (U - U.mean()) / U.std()

    elderly = np.clip(0.20 + 0.06 * U + rng.normal(0, 0.04, n), 0.05, 0.45)
    single_hh = np.clip(0.30 + 0.05 * U + rng.normal(0, 0.05, n), 0.10, 0.60)
    basic = np.clip(0.07 + 0.03 * U + rng.normal(0, 0.02, n), 0.01, 0.20)
    dist_hosp = np.clip(800 + 500 * U + rng.normal(0, 250, n), 100, 4000)
    dist_transit = np.clip(400 + 300 * U + rng.normal(0, 150, n), 50, 2500)
    ndvi = np.clip(0.4 - 0.1 * U + rng.normal(0, 0.08, n), 0.05, 0.85)

    socio = (60 * elderly + 30 * single_hh + 80 * basic
             + 0.004 * dist_hosp + 0.003 * dist_transit
             + 0.01 * np.maximum(dist_hosp - 1500, 0) * elderly)
    latent = np.sin(2 * np.pi * 1.5 * xs) * np.cos(2 * np.pi * 1.2 * ys)
    latent = (latent - latent.mean()) / latent.std()
    vuln = np.clip(socio + 10.0 * latent + rng.normal(0, 1.0, n), 0, None)

    df = pd.DataFrame({
        "cell": np.arange(n), "row": rows, "col": cols, "x": xs, "y": ys,
        "elderly": elderly, "single_hh": single_hh, "basic": basic,
        "dist_hosp": dist_hosp, "dist_transit": dist_transit, "ndvi": ndvi,
        "vulnerability": vuln,
    })
    # 부동소수점을 정확히 보존하도록 Parquet로 저장(모델 재현성 확보)
    df.to_parquet(DATA_DIR / "grid_vulnerability.parquet", index=False)
    print(f"  격자 {len(df)}개(20×20) → grid_vulnerability.parquet")


def main():
    print("=" * 60)
    print("실습 데이터 준비 (4-1 · 4-2 · 4-5)")
    print("=" * 60)
    prepare_buildings()
    prepare_spatial_points()
    prepare_grid()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
