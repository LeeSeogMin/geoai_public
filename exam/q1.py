"""
문제 1 (1장 + 2장). 역 3km 안 시설 수
=====================================
어느 구청이 지하철역 한 곳에서 직선 3km 안에 있는 어린이집(모의 400곳)을 센다.
역과 시설 좌표는 학번으로 정해지며, 시험용 모의 자료다.

실행: python exam/q1.py --id 학번
수정본은 q1_fixed.py 로 저장해 같은 방식으로 실행한다.
"""

import numpy as np
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point

from common import key, parse_id, rng_of, seed_of

STATIONS = {  # (경도, 위도) 시험용 근사값
    "옥수": (127.0183, 37.5406), "신림": (126.9297, 37.4842),
    "화곡": (126.8400, 37.5410), "잠실": (127.1001, 37.5133),
    "노원": (127.0564, 37.6552), "홍대입구": (126.9249, 37.5572),
}

sid = parse_id()
name = list(STATIONS)[seed_of(sid) % len(STATIONS)]
lon0, lat0 = STATIONS[name]
rng = rng_of(sid, 1)
n = 400
lon = lon0 + rng.uniform(-0.07, 0.07, n)
lat = lat0 + rng.uniform(-0.055, 0.055, n)
fac = gpd.GeoDataFrame({"id": np.arange(n)}, geometry=gpd.points_from_xy(lon, lat),
                       crs="EPSG:4326")
station = Point(lon0, lat0)

print(f"역: {name}")
print(f"[확인 1] 객체 수 {len(fac)}, 좌표계 {fac.crs}, 기하 유형 {fac.geom_type.unique()[0]}")
print(f"[확인 2] 범위(경도) {lon.min():.4f} ~ {lon.max():.4f}, (위도) {lat.min():.4f} ~ {lat.max():.4f}")

# 3km에 해당하는 도 단위 값으로 0.027을 쓴다
buf = station.buffer(0.027)
inside = fac[fac.within(buf)]
area_km2 = buf.area * 111.0 ** 2   # 도 -> km 환산

# 실제 거리 확인용: 두 점을 미터 좌표계로 옮겨 측정한다
to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
sx, sy = to_m.transform(lon0, lat0)
fx, fy = to_m.transform(fac.geometry.x.values, fac.geometry.y.values)
dist_m = np.hypot(fx - sx, fy - sy)
true_ids = set(fac.index[dist_m <= 3000])
missed = len(true_ids - set(inside.index))

print(f"3km 안 시설 수: {len(inside)}곳")
print(f"권역 면적: {area_km2:.2f} km2")
print(f"실제 3km 안인데 빠진 시설: {missed}곳")
key("count_3km", len(inside))
key("area_km2", float(area_km2))
key("missed", missed)
