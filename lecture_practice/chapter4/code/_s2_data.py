"""
_s2_data.py — 실제 Sentinel-2 클립 공통 로더 (Ch4·Ch5 공용)
============================================================
4-0-data-download.py가 저장한 실제 12밴드 클립·SCL·WorldCover 라벨을
읽어, 픽셀 표(전통 ML)와 패치(CNN)로 변환하는 함수를 모은다.
파일명이 '_'로 시작하므로 실행 증거 게이트(run_and_capture)의
실행 대상에서 제외된다(라이브러리 모듈).

밴드 순서(12): B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
파생지수는 이 인덱스를 기준으로 계산한다.
"""

from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
S2_PATH = DATA_DIR / "sentinel2_l2a_12band_clip.tif"
SCL_PATH = DATA_DIR / "sentinel2_scl_clip.tif"
WC_PATH = DATA_DIR / "worldcover_2021_clip.tif"

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06",
              "B07", "B08", "B8A", "B09", "B11", "B12"]
IDX = {b: i for i, b in enumerate(BAND_NAMES)}

# ESA WorldCover 코드 → 실습 클래스 (희소 클래스는 제외)
WC_CLASSES = {10: "산림", 30: "초지", 40: "경작지", 50: "시가지", 80: "수역"}
# 구름·그림자·권운·포화 SCL 코드 (마스킹 대상)
SCL_INVALID = [0, 1, 3, 8, 9, 10]


def _require_data():
    if not S2_PATH.exists():
        raise SystemExit(
            f"데이터가 없습니다: {S2_PATH}\n먼저 실행: python 4-0-data-download.py"
        )


def load_reflectance():
    """12밴드 반사율 큐브(float32, 0~1), 유효마스크, SCL을 반환한다."""
    _require_data()
    with rasterio.open(S2_PATH) as src:
        cube = src.read().astype(np.float32) / 10000.0  # (12, H, W)
        transform = src.transform
    with rasterio.open(SCL_PATH) as src:
        scl = src.read(1)
    with rasterio.open(WC_PATH) as src:
        wc = src.read(1)
    # 유효 픽셀: nodata(0) 아니고, 구름·그림자 아님
    filled = cube[IDX["B02"]] > 0
    clear = ~np.isin(scl, SCL_INVALID)
    valid = filled & clear
    return cube, valid, scl, wc, transform


def spectral_indices(cube):
    """정규화 분광지수 3종(NDVI·NDWI·NDBI)을 (3, H, W)로 계산한다."""
    red, nir = cube[IDX["B04"]], cube[IDX["B08"]]
    green, swir = cube[IDX["B03"]], cube[IDX["B11"]]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)      # 식생
    ndwi = (green - nir) / (green + nir + eps)  # 수분·수역
    ndbi = (swir - nir) / (swir + nir + eps)    # 건조·시가지
    return np.stack([ndvi, ndwi, ndbi])


def build_pixel_table(n_per_class=1500, seed=42):
    """
    유효 픽셀에서 클래스별로 균형 표본을 뽑아 픽셀 표를 만든다.
    반환: X(피처: 12밴드+NDVI+NDWI+NDBI), y(0..K-1), coords(m), 부가정보 dict
    """
    cube, valid, scl, wc, transform = load_reflectance()
    idx3 = spectral_indices(cube)
    feat = np.concatenate([cube, idx3], axis=0)  # (15, H, W)
    feat_names = BAND_NAMES + ["NDVI", "NDWI", "NDBI"]

    rng = np.random.default_rng(seed)
    codes = sorted(WC_CLASSES.keys())
    Xs, ys, rows, cols = [], [], [], []
    for cls_idx, code in enumerate(codes):
        mask = valid & (wc == code)
        rr, cc = np.where(mask)
        if len(rr) == 0:
            continue
        take = min(n_per_class, len(rr))
        sel = rng.choice(len(rr), size=take, replace=False)
        r, c = rr[sel], cc[sel]
        Xs.append(feat[:, r, c].T)          # (take, 15)
        ys.append(np.full(take, cls_idx))
        rows.append(r)
        cols.append(c)

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    r_all = np.concatenate(rows)
    c_all = np.concatenate(cols)
    # 픽셀 (행,열) → 지도 좌표(m). 공간 블록 CV의 그룹 정의에 사용
    xs_m, ys_m = rasterio.transform.xy(transform, r_all, c_all)
    coords = np.column_stack([xs_m, ys_m]).astype(np.float64)

    info = {
        "feature_names": feat_names,
        "class_codes": codes,
        "class_names": [WC_CLASSES[c] for c in codes],
    }
    return X, y, coords, info


def extract_patches(patch=32, stride=32, purity=0.6, seed=42):
    """
    영상을 격자로 잘라 패치를 만들고, 패치 중심 영역의 WorldCover
    다수결로 라벨을 붙인다. 순도(purity) 미만이거나 구름·nodata가
    섞인 패치는 버린다.
    반환: patches(N,12,P,P), labels(N,), tile_ids(N,), info
    """
    cube, valid, scl, wc, transform = load_reflectance()
    C, H, W = cube.shape
    codes = sorted(WC_CLASSES.keys())
    code_to_idx = {c: i for i, c in enumerate(codes)}
    codes_arr = np.array(codes)

    rng = np.random.default_rng(seed)
    patches, labels, tiles = [], [], []
    # 타일: 영상을 6×6 큰 블록으로 나눠 각 패치에 타일 ID 부여(공간 CV용)
    tile_h, tile_w = H / 6, W / 6

    for r0 in range(0, H - patch + 1, stride):
        for c0 in range(0, W - patch + 1, stride):
            vwin = valid[r0:r0 + patch, c0:c0 + patch]
            if vwin.mean() < 0.95:          # 구름·경계 섞인 패치 제외
                continue
            wwin = wc[r0:r0 + patch, c0:c0 + patch]
            # 중심 절반 영역의 다수 클래스
            q = patch // 4
            center = wwin[q:patch - q, q:patch - q]
            center = center[np.isin(center, codes)]
            if center.size == 0:
                continue
            vals, cnts = np.unique(center, return_counts=True)
            major, frac = vals[cnts.argmax()], cnts.max() / center.size
            if frac < purity:               # 순도 낮은(혼합) 패치 제외
                continue
            patches.append(cube[:, r0:r0 + patch, c0:c0 + patch])
            labels.append(code_to_idx[int(major)])
            tiles.append(int(r0 // tile_h) * 6 + int(c0 // tile_w))

    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels)
    tiles = np.array(tiles)
    info = {"class_codes": codes,
            "class_names": [WC_CLASSES[c] for c in codes]}
    return patches, labels, tiles, info
