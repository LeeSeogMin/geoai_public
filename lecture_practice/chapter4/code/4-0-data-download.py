"""
4-0. 실습 데이터 다운로드: Sentinel-2 L2A 클립 + ESA WorldCover 라벨
====================================================================
Microsoft Planetary Computer STAC API에서 춘천 일대의 저운량
Sentinel-2 L2A 장면을 검색하고, 연구지역 bbox를 10m 공통 격자로
정합한 12밴드 GeoTIFF 클립을 저장한다. 같은 격자로 장면분류(SCL)
래스터와 ESA WorldCover 2021 토지피복 라벨도 저장한다.

- Sentinel-2 MSI 센서는 13개 밴드를 관측하지만, L2A(대기보정)
  산출물에는 지표 정보가 없는 B10(권운)이 제외되어 12개 밴드가 제공된다.
- 20m·60m 밴드는 10m 격자로 쌍선형(bilinear) 보간해 정합한다.
- 파일이 이미 존재하면 다운로드를 건너뛰고 요약만 출력한다(재실행 안전).

데이터 출처·라이선스:
- Sentinel-2 L2A: ESA Copernicus (CC BY-SA 3.0 IGO),
  https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a
- ESA WorldCover 2021 v200: ESA (CC BY 4.0),
  https://planetarycomputer.microsoft.com/dataset/esa-worldcover

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    pip install -r lecture_practice/requirements.txt
    python lecture_practice/chapter4/code/4-0-data-download.py
"""

import json
import time
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

# ============================================================
# 0. 설정: 연구지역·시점·밴드
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

S2_PATH = DATA_DIR / "sentinel2_l2a_12band_clip.tif"
SCL_PATH = DATA_DIR / "sentinel2_scl_clip.tif"
WC_PATH = DATA_DIR / "worldcover_2021_clip.tif"
META_PATH = DATA_DIR / "clip_metadata.json"

# 연구지역: 춘천 일대 (의암호 수계 + 시가지 + 농경지 + 산림 공존)
AOI_BBOX = [127.63, 37.82, 127.78, 37.94]  # 경도min, 위도min, 경도max, 위도max

# 시점: 2021년 (라벨로 쓰는 ESA WorldCover 2021과 같은 해)
DATE_RANGE = "2021-04-01/2021-11-30"
MAX_CLOUD = 5  # 장면 전체 구름 비율 상한(%)

# L2A 제공 12밴드 (B10 권운 밴드는 L2A에서 제외됨)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
         "B07", "B08", "B8A", "B09", "B11", "B12"]
BAND_INFO = {  # (중심파장 nm, 원본 해상도 m, 설명)
    "B01": (443, 60, "Coastal aerosol"),
    "B02": (490, 10, "Blue"),
    "B03": (560, 10, "Green"),
    "B04": (665, 10, "Red"),
    "B05": (705, 20, "Red Edge 1"),
    "B06": (740, 20, "Red Edge 2"),
    "B07": (783, 20, "Red Edge 3"),
    "B08": (842, 10, "NIR"),
    "B8A": (865, 20, "NIR narrow"),
    "B09": (945, 60, "Water vapour"),
    "B11": (1610, 20, "SWIR 1"),
    "B12": (2190, 20, "SWIR 2"),
}
TARGET_RES = 10  # 공통 해상도(m)


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


def read_to_grid(href, dst_crs, dst_transform, width, height, resampling):
    """원격 COG의 연구지역 창을 목표 격자(10m)로 정합해 읽는다."""
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=dst_crs, transform=dst_transform,
                       width=width, height=height,
                       resampling=resampling) as vrt:
            return vrt.read(1)


def main():
    print("=" * 60)
    print("실습 데이터 다운로드: Sentinel-2 L2A + ESA WorldCover")
    print("=" * 60)

    already = S2_PATH.exists() and SCL_PATH.exists() and WC_PATH.exists()

    # ============================================================
    # 1. STAC 검색: 저운량 L2A 장면 선택
    # ============================================================
    print("\n--- 1. STAC 검색 (Planetary Computer) ---")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,   # 자산 URL 서명(무료, 인증키 불필요)
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=AOI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    items = fetch_items(search)
    print(f"  검색 조건: {DATE_RANGE}, 장면 구름 < {MAX_CLOUD}%")
    print(f"  검색 결과: {len(items)}개 장면")

    # 연구지역 bbox를 완전히 포함하는 장면 중 구름이 가장 적은 것 선택
    def covers_aoi(item):
        b = item.bbox
        return (b[0] <= AOI_BBOX[0] and b[1] <= AOI_BBOX[1]
                and b[2] >= AOI_BBOX[2] and b[3] >= AOI_BBOX[3])

    candidates = [it for it in items if covers_aoi(it)]
    candidates.sort(key=lambda it: it.properties["eo:cloud_cover"])
    if not candidates:
        raise RuntimeError("연구지역을 완전히 덮는 장면이 없습니다. bbox·기간을 조정하세요.")
    item = candidates[0]
    props = item.properties
    print(f"  선택 장면: {item.id}")
    print(f"    취득일: {item.datetime:%Y-%m-%d %H:%M} UTC")
    print(f"    구름 비율: {props['eo:cloud_cover']:.2f}%")
    print(f"    처리 베이스라인: {props.get('s2:processing_baseline', 'N/A')}")

    # ============================================================
    # 2. 목표 격자 정의: UTM 좌표계, 10m 공통 해상도
    # ============================================================
    epsg = props.get("proj:epsg") or props.get("proj:code", "").replace("EPSG:", "")
    dst_crs = f"EPSG:{epsg}"
    tf = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    x0, y0 = tf.transform(AOI_BBOX[0], AOI_BBOX[1])
    x1, y1 = tf.transform(AOI_BBOX[2], AOI_BBOX[3])
    # 10m 격자에 스냅
    xmin = np.floor(x0 / TARGET_RES) * TARGET_RES
    ymax = np.ceil(y1 / TARGET_RES) * TARGET_RES
    width = int(np.ceil((x1 - xmin) / TARGET_RES))
    height = int(np.ceil((ymax - y0) / TARGET_RES))
    dst_transform = from_origin(xmin, ymax, TARGET_RES, TARGET_RES)
    print(f"\n--- 2. 목표 격자 ---")
    print(f"  좌표계: {dst_crs} (UTM)")
    print(f"  크기: {height}행 × {width}열, 해상도 {TARGET_RES}m")
    print(f"  bounds: ({xmin:.0f}, {ymax - height * TARGET_RES:.0f}) ~ "
          f"({xmin + width * TARGET_RES:.0f}, {ymax:.0f})")

    if already:
        print("\n  데이터 파일이 이미 존재하므로 다운로드를 건너뛴다.")
    else:
        # ============================================================
        # 3. 12밴드 다운로드: 20m·60m 밴드는 10m로 쌍선형 보간
        # ============================================================
        print("\n--- 3. 12밴드 클립 다운로드 ---")
        cube = np.zeros((len(BANDS), height, width), dtype=np.uint16)
        for i, band in enumerate(BANDS):
            wl, res, desc = BAND_INFO[band]
            href = item.assets[band].href
            cube[i] = read_to_grid(href, dst_crs, dst_transform, width, height,
                                   Resampling.bilinear)
            note = "원본 10m" if res == TARGET_RES else f"{res}m→10m 쌍선형 보간"
            print(f"  {band:4s} ({wl:4d}nm, {desc:15s}): {note}")

        profile = {
            "driver": "GTiff", "dtype": "uint16", "count": len(BANDS),
            "height": height, "width": width, "crs": dst_crs,
            "transform": dst_transform, "nodata": 0,
            "compress": "deflate", "predictor": 2,
        }
        with rasterio.open(S2_PATH, "w", **profile) as dst:
            dst.write(cube)
            for i, band in enumerate(BANDS, start=1):
                wl, res, desc = BAND_INFO[band]
                dst.set_band_description(i, f"{band} {desc} ({wl}nm, 원본 {res}m)")
        print(f"  저장: {S2_PATH.name} ({S2_PATH.stat().st_size / 1e6:.1f} MB)")

        # ============================================================
        # 4. SCL(장면분류)·WorldCover 라벨: 같은 격자, 최근접 보간
        # ============================================================
        print("\n--- 4. SCL·WorldCover 클립 다운로드 ---")
        scl = read_to_grid(item.assets["SCL"].href, dst_crs, dst_transform,
                           width, height, Resampling.nearest)
        profile.update(dtype="uint8", count=1, nodata=0, predictor=1)
        with rasterio.open(SCL_PATH, "w", **profile) as dst:
            dst.write(scl.astype(np.uint8), 1)
            dst.set_band_description(1, "Sentinel-2 L2A Scene Classification (SCL)")
        print(f"  저장: {SCL_PATH.name} (SCL, 범주형이므로 최근접 보간)")

        wc_search = catalog.search(collections=["esa-worldcover"], bbox=AOI_BBOX,
                                   datetime="2021-01-01/2021-12-31")
        wc_items = fetch_items(wc_search)
        wc_item = wc_items[0]
        wc = read_to_grid(wc_item.assets["map"].href, dst_crs, dst_transform,
                          width, height, Resampling.nearest)
        with rasterio.open(WC_PATH, "w", **profile) as dst:
            dst.write(wc.astype(np.uint8), 1)
            dst.set_band_description(1, "ESA WorldCover 2021 v200 (10m)")
        print(f"  저장: {WC_PATH.name} (WorldCover {wc_item.id})")

        # ============================================================
        # 5. 메타데이터 기록 (재현성·출처·라이선스)
        # ============================================================
        meta = {
            "sentinel2_item_id": item.id,
            "datetime_utc": item.datetime.isoformat(),
            "cloud_cover_percent": props["eo:cloud_cover"],
            "processing_baseline": props.get("s2:processing_baseline"),
            "collection": "sentinel-2-l2a (Microsoft Planetary Computer)",
            "stac_api": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "license_s2": "Copernicus Sentinel data 2021 (ESA), CC BY-SA 3.0 IGO",
            "worldcover_item_id": wc_item.id,
            "license_worldcover": "ESA WorldCover 2021 v200, CC BY 4.0",
            "aoi_bbox_wgs84": AOI_BBOX,
            "crs": dst_crs,
            "resolution_m": TARGET_RES,
            "shape": [len(BANDS), height, width],
            "bands": {b: {"wavelength_nm": BAND_INFO[b][0],
                          "native_res_m": BAND_INFO[b][1],
                          "name": BAND_INFO[b][2]} for b in BANDS},
            "resampling": "20m/60m 밴드는 10m로 bilinear, SCL·WorldCover는 nearest",
            "dn_to_reflectance": ("reflectance = DN / 10000. 처리 베이스라인 04.00 미만에는 "
                                  "BOA 오프셋이 없다(이 장면 baseline "
                                  f"{props.get('s2:processing_baseline')})."),
            "note_b10": "L2A에는 B10(권운)이 없어 12밴드. 13밴드 원본은 L1C(TOA).",
        }
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"  저장: {META_PATH.name}")

    # ============================================================
    # 6. 저장 파일 검증 요약
    # ============================================================
    print("\n--- 5. 저장 파일 요약 ---")
    for path in [S2_PATH, SCL_PATH, WC_PATH]:
        with rasterio.open(path) as src:
            print(f"  {path.name}: {src.count}밴드 {src.height}×{src.width}, "
                  f"{src.crs}, {src.res[0]:.0f}m, dtype={src.dtypes[0]}")

    print("\n[완료] 실습 데이터 다운로드를 마쳤다.")


if __name__ == "__main__":
    main()
