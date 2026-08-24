"""
2-1. 좌표 변환 실습
==================
WGS84 경위도를 UTM 52N으로 변환하고, 역변환 검증 및 거리 계산을 수행한다.

실행 방법:
    cd lecture_practice/chapter2
    python -m venv venv
    source venv/bin/activate          # macOS/Linux
    pip install -r code/requirements.txt
    python code/2-1-coordinate-transform.py
"""

from pyproj import Transformer, CRS
import math

# ============================================================
# 1. WGS84 → UTM 52N 좌표 변환
# ============================================================
# always_xy=True: (경도, 위도) 순서로 입력
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
reverse_transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)

print("=" * 60)
print("좌표 변환: WGS84(EPSG:4326) → UTM 52N(EPSG:32652)")
print("=" * 60)

# 한국 주요 도시 좌표 (경도, 위도)
cities = {
    "서울": (127.0, 37.5),
    "부산": (129.0, 35.1),
    "제주": (126.5, 33.5),
    "인천": (126.7, 37.45),
}

print("\n한국 주요 도시 UTM 좌표:")
utm_coords = {}
for name, (lon, lat) in cities.items():
    x, y = transformer.transform(lon, lat)
    utm_coords[name] = (x, y)
    print(f"  {name}: ({lon}°E, {lat}°N) → ({x:,.2f}m E, {y:,.2f}m N)")

# ============================================================
# 2. 역변환 검증
# ============================================================
print("\n역변환 검증:")
x_seoul, y_seoul = utm_coords["서울"]
lon_back, lat_back = reverse_transformer.transform(x_seoul, y_seoul)
print(f"  원본: (127.0, 37.5)")
print(f"  복원: ({lon_back:.10f}, {lat_back:.10f})")
print(f"  오차: {abs(127.0 - lon_back):.15f}도")

# ============================================================
# 3. 도시 간 거리 계산 비교
# ============================================================
print("\n도시 간 거리 계산:")

# UTM 유클리드 거리
def utm_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) / 1000  # km

# Haversine 거리 (경위도 기반)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # 지구 반지름 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

pairs = [("서울", "부산"), ("서울", "제주"), ("서울", "인천")]
for c1, c2 in pairs:
    d_utm = utm_distance(utm_coords[c1], utm_coords[c2])
    d_hav = haversine(*cities[c1], *cities[c2])
    diff_pct = abs(d_utm - d_hav) / d_hav * 100
    print(f"  {c1}-{c2}: UTM {d_utm:.2f}km vs Haversine {d_hav:.2f}km (차이 {diff_pct:.2f}%)")

# ============================================================
# 4. 좌표계 정보 확인
# ============================================================
print("\n좌표계 상세 정보:")
crs_wgs84 = CRS.from_epsg(4326)
crs_utm52n = CRS.from_epsg(32652)
print(f"  WGS84: {crs_wgs84.name}")
print(f"    유형: {'Geographic' if crs_wgs84.is_geographic else 'Projected'}")
print(f"    타원체: {crs_wgs84.ellipsoid.name}")
print(f"    장반경: {crs_wgs84.ellipsoid.semi_major_metre:,.3f}m")
print(f"  UTM 52N: {crs_utm52n.name}")
print(f"    유형: {'Geographic' if crs_utm52n.is_geographic else 'Projected'}")
print(f"    단위: {crs_utm52n.axis_info[0].unit_name}")

print("\n[완료] 좌표 변환 실습을 마쳤다.")
