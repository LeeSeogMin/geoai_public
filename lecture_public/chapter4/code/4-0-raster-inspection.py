"""
4-0. 실제 위성영상 파일 열어보기 (래스터 검사)
=============================================
4-0-data-download.py로 저장한 실제 Sentinel-2 L2A 12밴드 클립을 열어,
학습자가 원자료의 구조를 직접 확인한다. 밴드 수·행렬 크기·좌표계·해상도·
경계·nodata·밴드명·밴드별 통계·특정 픽셀의 밴드값 벡터·파생지수(NDVI·NDWI)를
모두 실제 파일에서 읽어 출력한다.

핵심: 화면에 보이는 RGB 영상이 아니라, (행, 열) 위치마다 밴드별 숫자가
저장된 3차원 데이터 큐브가 원자료라는 사실을 눈으로 확인하는 것이 목적이다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter4/code/4-0-raster-inspection.py
    # 데이터가 없으면 먼저: python practice/chapter4/code/4-0-data-download.py
"""

from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
S2_PATH = DATA_DIR / "sentinel2_l2a_12band_clip.tif"
SCL_PATH = DATA_DIR / "sentinel2_scl_clip.tif"

if not S2_PATH.exists():
    raise SystemExit(
        f"데이터가 없습니다: {S2_PATH}\n"
        "먼저 실행: python 4-0-data-download.py"
    )

print("=" * 60)
print("실제 Sentinel-2 L2A 파일 열어보기")
print("=" * 60)

# ============================================================
# 1. 파일 수준 메타데이터
# ============================================================
print("\n--- 1. 파일 메타데이터 ---")
with rasterio.open(S2_PATH) as src:
    print(f"  파일 경로: {S2_PATH}")
    print(f"  밴드 수: {src.count}")
    print(f"  행×열: {src.height} × {src.width}")
    print(f"  좌표계: {src.crs} ({src.crs.to_authority()})")
    print(f"  공간 해상도: {src.res[0]:.0f}m × {src.res[1]:.0f}m")
    print(f"  bounds: {tuple(round(v, 1) for v in src.bounds)}")
    print(f"  nodata: {src.nodata}")
    print(f"  dtype: {src.dtypes[0]}")

    # 밴드 설명(밴드명)
    print("\n  밴드 목록:")
    for i in range(1, src.count + 1):
        print(f"    밴드 {i:2d}: {src.descriptions[i - 1]}")

    # ============================================================
    # 2. 전체 데이터 큐브 읽기
    # ============================================================
    print("\n--- 2. 데이터 큐브 ---")
    cube = src.read()  # shape: (밴드, 행, 열)
    print(f"  배열 shape (밴드, 행, 열): {cube.shape}")
    print(f"  메모리 크기: {cube.nbytes / 1e6:.1f} MB")

    # nodata(0) 마스크: 클립 경계 밖 결측
    valid = cube[1] > 0  # B02(Blue)가 0이 아니면 유효 픽셀
    print(f"  유효 픽셀: {valid.sum():,} / {valid.size:,} "
          f"({100 * valid.sum() / valid.size:.1f}%)")

    # ============================================================
    # 3. 밴드별 통계 (반사율 = DN / 10000)
    # ============================================================
    print("\n--- 3. 밴드별 통계 (유효 픽셀, 반사율 단위) ---")
    print(f"  {'밴드':22s} | {'최소':>7s} | {'평균':>7s} | {'최대':>7s}")
    print(f"  {'-'*22} | {'-'*7} | {'-'*7} | {'-'*7}")
    for i in range(src.count):
        vals = cube[i][valid] / 10000.0
        name = src.descriptions[i].split("(")[0].strip()
        print(f"  {name:22s} | {vals.min():7.3f} | {vals.mean():7.3f} | {vals.max():7.3f}")

# ============================================================
# 4. 특정 픽셀의 12개 밴드값 벡터
# ============================================================
print("\n--- 4. 특정 픽셀의 밴드값 벡터 ---")
# 영상 기하 중심의 픽셀을 고른다 (유효 영역이면 그대로 사용)
row, col = cube.shape[1] // 2, cube.shape[2] // 2
pixel = cube[:, row, col]  # 12개 밴드값
print(f"  픽셀 (행={row}, 열={col})의 원시 DN 값:")
band_short = ["B01", "B02", "B03", "B04", "B05", "B06",
              "B07", "B08", "B8A", "B09", "B11", "B12"]
for name, dn in zip(band_short, pixel):
    print(f"    {name}: {int(dn):5d}  (반사율 {dn / 10000:.3f})")

# ============================================================
# 5. 파생지수: NDVI, NDWI (실제 밴드값으로 계산)
# ============================================================
print("\n--- 5. 파생지수 (실제 밴드값 계산) ---")
# B04=Red(인덱스3), B08=NIR(인덱스7), B03=Green(인덱스2)
red = pixel[3] / 10000.0
nir = pixel[7] / 10000.0
green = pixel[2] / 10000.0
ndvi = (nir - red) / (nir + red)
ndwi = (green - nir) / (green + nir)
print(f"  이 픽셀 NDVI = (NIR {nir:.3f} - Red {red:.3f}) / (합) = {ndvi:.3f}")
print(f"  이 픽셀 NDWI = (Green {green:.3f} - NIR {nir:.3f}) / (합) = {ndwi:.3f}")

# 전체 영상의 NDVI 분포
red_all = cube[3][valid] / 10000.0
nir_all = cube[7][valid] / 10000.0
ndvi_all = (nir_all - red_all) / (nir_all + red_all + 1e-9)
print(f"  영상 전체 NDVI: 최소 {ndvi_all.min():.3f}, "
      f"평균 {ndvi_all.mean():.3f}, 최대 {ndvi_all.max():.3f}")
print(f"    NDVI > 0.4 (식생) 비율: {100 * (ndvi_all > 0.4).mean():.1f}%")

# ============================================================
# 6. 장면분류(SCL)로 본 구름·수역 비율
# ============================================================
print("\n--- 6. 장면분류(SCL) 요약 ---")
with rasterio.open(SCL_PATH) as src:
    scl = src.read(1)
SCL_LABELS = {4: "식생", 5: "나지", 6: "수역", 8: "구름(중)",
              9: "구름(고)", 10: "권운", 3: "구름그림자"}
cloud = np.isin(scl, [8, 9, 10, 3]).sum()
print(f"  구름·그림자 픽셀: {cloud:,} ({100 * cloud / scl.size:.2f}%)")
for code in [4, 5, 6]:
    n = (scl == code).sum()
    print(f"  SCL {code} {SCL_LABELS[code]}: {n:,} ({100 * n / scl.size:.1f}%)")

print("\n[완료] 실제 파일 검사를 마쳤다. 원자료는 (밴드, 행, 열) 숫자 배열이다.")
