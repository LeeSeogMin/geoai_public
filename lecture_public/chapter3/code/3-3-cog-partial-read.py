"""
3-3. COG 부분 읽기(HTTP Range Request) 실습
==========================================
STAC로 검색한 Sentinel-2 COG(Cloud Optimized GeoTIFF) 밴드를 대상으로,
전체 파일을 내려받지 않고 관심 영역만 읽을 때 실제로 무엇이 달라지는지 측정한다.

- 전체 파일 크기는 HTTP HEAD 요청으로만 확인한다(본문은 받지 않음).
- 전체 해상도(폭×높이)는 rasterio가 COG 헤더만 읽어서 얻는다(전체 픽셀은 받지 않음).
- 실제로 픽셀을 내려받는 것은 강남 소구역(약 2km×2km) 윈도우 읽기 1회뿐이고,
  받은 결과만 GeoTIFF로 저장한다. 268MB 전체 파일은 여전히 받지 않는다.
- 픽셀 비율은 정확한 값이지만, 압축된 COG의 실제 전송 바이트는 타일 배치에 따라
  픽셀 비율과 정확히 비례하지 않으므로 "근사"로 해석한다.
- 마지막으로 Green·Blue 밴드를 같은 윈도우로 추가 읽어 RGB로 합성하고,
  percentile 대비 스트레치를 적용해 VSCode 등 일반 이미지 뷰어로 바로 볼 수 있는
  PNG 미리보기를 만든다. 스트레치 없는 단일 밴드 PNG도 함께 저장해 비교한다.

실행 방법:
    cd lecture_practice/chapter3
    python code/3-3-cog-partial-read.py
"""

import time
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

matplotlib.use("Agg")  # 화면 없이 PNG 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "gangnam_b04_window.tif"
RAW_PNG_PATH = DATA_DIR / "gangnam_b04_raw_preview.png"
RGB_PNG_PATH = DATA_DIR / "gangnam_rgb_stretched.png"


def percentile_stretch(band, low=2, high=98):
    """반사율 DN을 이미지 뷰어용 0~255 대비로 변환한다(percentile 스트레치)."""
    lo, hi = np.percentile(band, [low, high])
    scaled = np.clip((band.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def fetch_items(search, retries=3, backoff=5):
    """STAC 검색 결과를 가져온다. 일시적 서버 지연은 지수 백오프로 재시도."""
    for attempt in range(1, retries + 1):
        try:
            return list(search.item_collection())
        except APIError as e:
            if attempt == retries:
                raise
            print(f"    (재시도 {attempt}/{retries - 1}: {type(e).__name__}) "
                  f"{backoff * attempt}초 대기 후 재요청")
            time.sleep(backoff * attempt)


print("=" * 60)
print("COG 부분 읽기(Range Request) 실습")
print("=" * 60)

# ============================================================
# 1. STAC 검색: 서울 지역 저운량 Sentinel-2 장면 1개 선택
# ============================================================
print("\n--- 1. STAC 검색 ---")

seoul_bbox = [126.8, 37.4, 127.2, 37.7]
gangnam_bbox = [127.02, 37.49, 127.05, 37.51]  # 경도min, 위도min, 경도max, 위도max

# Planetary Computer의 Asset href는 서명(SAS 토큰)이 있어야 접근 가능하다.
# modifier=pc.sign_inplace로 검색 결과의 모든 href를 자동으로 서명한다.
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=seoul_bbox,
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)
items = fetch_items(search)


# Sentinel-2는 타일 단위로 나뉘어 있어, 검색 bbox와 겹치더라도 실제로
# 읽으려는 강남 소구역을 포함하지 않는 타일이 섞여 있을 수 있다.
# 4단계에서 읽을 소구역을 완전히 포함하는 타일만 후보로 남긴다.
def covers_gangnam(it):
    b = it.bbox
    return (b[0] <= gangnam_bbox[0] and b[1] <= gangnam_bbox[1]
            and b[2] >= gangnam_bbox[2] and b[3] >= gangnam_bbox[3])


candidates = [it for it in items if covers_gangnam(it)]
if not candidates:
    raise RuntimeError("강남 소구역을 포함하는 타일이 검색 결과에 없습니다.")
item = min(candidates, key=lambda it: it.properties["eo:cloud_cover"])
print(f"  검색 결과 {len(items)}개 중 강남 소구역을 포함하며 "
      f"구름이 가장 적은 장면 선택: {item.id}")
print(f"  취득일: {item.datetime:%Y-%m-%d}, 구름 비율: "
      f"{item.properties['eo:cloud_cover']:.1f}%")

band = "B04"
href = item.assets[band].href
print(f"  대상 Asset: {band} (Red, 10m)")

# ============================================================
# 2. 전체 파일 크기 확인 (HTTP HEAD — 본문은 받지 않음)
# ============================================================
print("\n--- 2. 전체 COG 파일 크기 (HTTP HEAD) ---")

req = Request(href, method="HEAD")
with urlopen(req) as resp:
    total_bytes = int(resp.headers.get("Content-Length", 0))
print(f"  전체 파일 크기: {total_bytes / 1e6:.1f} MB")
print("  (본문을 내려받지 않고 헤더만 조회했다)")

# ============================================================
# 3. 전체 해상도 확인 (COG 헤더만 읽음 — 전체 픽셀은 받지 않음)
# ============================================================
print("\n--- 3. 전체 해상도 (COG 헤더 조회) ---")

with rasterio.open(href) as src:
    full_width, full_height = src.width, src.height
    crs = src.crs
    full_transform = src.transform
    nodata = src.nodata
    dtype = src.dtypes[0]
total_pixels = full_width * full_height
print(f"  전체 크기: {full_height}행 × {full_width}열 "
      f"({total_pixels:,}픽셀)")
print(f"  좌표계: {crs}")
print("  (rasterio.open()은 GDAL이 파일 헤더만 읽어 얻는다 — 픽셀 데이터는 아직 없다)")

# ============================================================
# 4. 관심 영역만 윈도우 읽기 (강남 소구역, 약 2km×2km)
# ============================================================
print("\n--- 4. 관심 영역 윈도우 읽기 (Range Request) ---")

transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x0, y0 = transformer.transform(gangnam_bbox[0], gangnam_bbox[1])
x1, y1 = transformer.transform(gangnam_bbox[2], gangnam_bbox[3])

t0 = time.time()
with rasterio.open(href) as src:
    window = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                          src.transform)
    win_data = src.read(1, window=window)
t_window = time.time() - t0

win_pixels = win_data.size
print(f"  읽은 영역: {win_data.shape[0]}행 × {win_data.shape[1]}열 "
      f"({win_pixels:,}픽셀)")
print(f"  소요 시간: {t_window:.2f}초")
print(f"  픽셀 최소~최대값: {win_data.min()}~{win_data.max()} "
      "(값이 있는 반사율 DN — 정상적으로 데이터를 받았다는 확인)")

# ============================================================
# 5. 받은 소구역만 GeoTIFF로 저장 (전체 파일은 여전히 저장하지 않음)
# ============================================================
print("\n--- 5. 소구역 저장 ---")

DATA_DIR.mkdir(exist_ok=True)
win_transform = window_transform(window, full_transform)
profile = {
    "driver": "GTiff", "dtype": dtype, "count": 1,
    "height": win_data.shape[0], "width": win_data.shape[1],
    "crs": crs, "transform": win_transform, "nodata": nodata,
    "compress": "deflate", "predictor": 2,
}
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(win_data, 1)
    dst.set_band_description(1, f"{item.id} {band} 강남 소구역 크롭")
out_size_kb = OUT_PATH.stat().st_size / 1024
print(f"  저장: {OUT_PATH.relative_to(DATA_DIR.parent.parent)} "
      f"({out_size_kb:.1f} KB)")
print(f"  (강남 소구역 크롭만 저장했다 — 전체 {total_bytes / 1e6:.1f} MB 파일은 여전히 받지 않았다)")

# ============================================================
# 6. 비교: 전체 vs 부분 읽기
# ============================================================
print("\n--- 6. 비교 ---")

pixel_ratio = win_pixels / total_pixels * 100
print(f"  전체 파일 크기: {total_bytes / 1e6:.1f} MB ({total_pixels:,}픽셀)")
print(f"  실제로 받은 픽셀: {win_pixels:,}개 (전체의 {pixel_ratio:.3f}%)")
print(f"  윈도우 읽기 소요 시간: {t_window:.2f}초")
print(
    "  → 픽셀 비율은 정확한 값이지만, COG는 타일 단위로 압축되어 있어 "
    "실제 전송 바이트가 픽셀 비율과 정확히 비례하지는 않는다(근사)."
)
print(
    "  → 그럼에도 전체 파일을 먼저 내려받는 절차 없이, 필요한 소구역만 "
    f"{t_window:.2f}초 만에 바로 받을 수 있었다는 점이 핵심이다."
)

# ============================================================
# 7. RGB 합성 미리보기 생성 (VSCode 등 일반 이미지 뷰어로 확인용)
# ============================================================
print("\n--- 7. RGB 합성 미리보기 ---")

# 7-1. 비교 기준: B04(Red) 1개 밴드를 대비 스트레치 없이 그대로 그레이스케일로 저장
raw_gray = percentile_stretch(win_data, low=0, high=100)  # 0~100% = 스트레치 없음(최소~최대 그대로)
plt.imsave(RAW_PNG_PATH, raw_gray, cmap="gray")
print(f"  저장(스트레치 없음, 단일 밴드 B04): {RAW_PNG_PATH.name}")

# 7-2. Green(B03)·Blue(B02) 밴드를 같은 윈도우로 추가 읽기
t0 = time.time()
with rasterio.open(item.assets["B03"].href) as src:
    green = src.read(1, window=window)
with rasterio.open(item.assets["B02"].href) as src:
    blue = src.read(1, window=window)
t_rgb_fetch = time.time() - t0
print(f"  Green(B03)·Blue(B02) 추가 읽기: {t_rgb_fetch:.2f}초 "
      f"(B04와 마찬가지로 강남 소구역만 Range Request로 수신)")

# 7-3. 밴드별 2~98 percentile 스트레치 후 RGB로 합성
rgb = np.dstack([
    percentile_stretch(win_data),  # Red   = B04
    percentile_stretch(green),     # Green = B03
    percentile_stretch(blue),      # Blue  = B02
])
plt.imsave(RGB_PNG_PATH, rgb)
print(f"  저장(2~98% 스트레치, RGB 합성): {RGB_PNG_PATH.name}")
print("  Red=B04, Green=B03, Blue=B02 순서로 합성했다.")
print(
    f"  → {RAW_PNG_PATH.name}(스트레치 없는 단일 밴드)과 "
    f"{RGB_PNG_PATH.name}(스트레치+RGB 합성)을 VSCode에서 나란히 열어 "
    "선명도·색상 차이를 직접 비교할 수 있다."
)

print("\n[완료] COG 부분 읽기 실습을 마쳤다.")
