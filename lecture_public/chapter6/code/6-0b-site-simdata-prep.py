"""
6-0b. 실습 데이터 준비: 개발 가능 부지 탐색용 도시 블록 래스터
==============================================================
이 장의 6-2 실습(6.7 분석 2)이 쓰는 데이터셋을 미리 만들어 `data/`
폴더에 저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를
준비하고, 6-2 코드는 저장된 파일을 불러와 객체화·형상 피처·필터·보정만
수행한다.

왜 합성인가: 도시 전역의 유휴 부지 정답 대장은 공개되어 있지 않고, 부지
경계를 사람이 확인한 라벨은 상업 데이터다. 여기서는 정답을 알고 있는
가상의 도시 블록을 만들어, 세그멘테이션 출력의 오차가 후보 선별을 어떻게
바꾸는지 채점할 수 있게 한다. 무거운 세그멘테이션 모델을 학습하는 대신
그 '출력(부지 마스크)'을 시뮬레이션하는 방식은 6-1과 같다.

저장 데이터(512×512 래스터, 픽셀 1m = 1㎡ — 약 26헥타르의 도시 블록):
- road         : 도로 마스크(bool)
- building     : 기존 건물 마스크(bool)
- vacant_true  : 유휴 부지 정답 마스크(bool)
- parcel_id    : 유휴 부지 정답 필지 번호(int, 0=배경)
- pred         : 세그멘테이션 예측 마스크(bool) — 오차 3종이 모두 들어간 본 실험
- pred_c1_noerode : 대조군 C1 — 경계 침식만 제거
- pred_c2_nofp    : 대조군 C2 — 오탐만 제거
- pred_c3_nosplit : 대조군 C3 — 과분할만 제거

심어 둔 오차 3종(6-2가 각각을 대조군으로 분리해 검증한다):
1. 경계 침식 — 예측 마스크에서 필지 가장자리를 한 겹(일부는 두 겹) 깎는다.
   면적을 체계적으로 과소 추정하게 만드는 원인이다.
2. 오탐 — 산발 얼룩과, 나대지처럼 보이는 큰 오탐 덩어리 몇 개(주차장·공사장을
   유휴 부지로 오인한 상황).
3. 과분할 — 중간 크기 필지 몇 곳을 얇은 띠로 갈라 두 조각으로 나눈다.
   가로수 그늘이나 수목열이 필지를 가로지를 때 마스크가 끊기는 상황이다.

임계 근처 필지군: 최소 면적 요건(1,000㎡) 언저리인 950~1,150㎡ 필지를
여럿 심어 둔다. 경계 침식이 면적을 10% 남짓 깎으면 이들이 통째로 임계
아래로 내려가므로, 오차가 결정을 뒤집는 지점이 눈에 보인다.

재현성: 6-1이 쓰는 6-0-simdata-prep.py와 **완전히 분리된 난수 생성기**
(default_rng(20260813))를 쓴다. 이 파일을 추가·수정해도 6-1의 결과는
바뀌지 않는다.

실행 방법 (프로젝트 루트, 통합 .venv):
    python practice/chapter6/code/6-0b-site-simdata-prep.py
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260813)   # 6-0과 독립된 시드

H = W = 512          # 래스터 크기(px)
PIXEL_M = 1.0        # 픽셀 한 변(m) → 픽셀 하나 = 1㎡
MARGIN = 6           # 래스터 가장자리 여백(경계 침식이 테두리에서 일어나지 않게)
GAP = 3              # 객체 사이 최소 간격(m)

# 오차 파라미터
P_DEEP_EROSION = 0.25   # 두 겹 침식이 일어나는 필지 비율
FP_RATE = 0.005         # 산발 오탐 씨앗 픽셀 비율
N_BIG_FP = 5            # 큰 오탐 덩어리(주차장·공사장 오인) 개수
N_SPLIT = 3             # 과분할시킬 필지 수

ROAD_ACCESS_M = 1.5     # '접도'로 볼 도로까지의 거리(m) — 도로에 맞붙은 필지
LANDLOCK_M = 14.0       # '맹지'로 볼 도로까지의 거리(m)


# ---------------------------------------------------------------- 도로망
def build_roads():
    """격자형 도로 + 사선 도로 1개."""
    road = np.zeros((H, W), bool)
    for center, half in [(24, 5), (160, 5), (300, 6), (440, 5)]:     # 가로 도로
        road[center - half:center + half + 1, :] = True
    for center, half in [(28, 6), (172, 5), (320, 6), (468, 5)]:     # 세로 도로
        road[:, center - half:center + half + 1] = True

    # 사선 도로 하나(격자만 있으면 필지 형상이 지나치게 규칙적이 된다)
    yy, xx = np.ogrid[:H, :W]
    for t in np.linspace(0.0, 1.0, 800):
        cy = int(round(40 + t * (470 - 40)))
        cx = int(round(430 + t * (60 - 430)))
        road |= (yy - cy) ** 2 + (xx - cx) ** 2 <= 4 ** 2
    return road


# ---------------------------------------------------------- 객체 배치 도구
def rect_patch(h, w, notch=None):
    """h×w 직사각형(또는 한 모서리를 파낸 L자) 패치."""
    m = np.ones((h, w), bool)
    if notch is not None:
        nh, nw, corner = notch
        nh, nw = min(nh, h - 4), min(nw, w - 4)
        if nh > 0 and nw > 0:
            if corner == 0:
                m[:nh, :nw] = False
            elif corner == 1:
                m[:nh, w - nw:] = False
            elif corner == 2:
                m[h - nh:, :nw] = False
            else:
                m[h - nh:, w - nw:] = False
    return m


def try_place(blocked, dist_road, patch, access, rng, tries=6000):
    """겹치지 않고 접도 조건을 만족하는 자리를 찾는다. 실패하면 None."""
    h, w = patch.shape
    for _ in range(tries):
        y0 = int(rng.integers(MARGIN, H - h - MARGIN))
        x0 = int(rng.integers(MARGIN, W - w - MARGIN))
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        if np.any(blocked[sl] & patch):
            continue
        d = dist_road[sl][patch].min()
        if access and d > ROAD_ACCESS_M:
            continue
        if (not access) and d < LANDLOCK_M:
            continue
        return y0, x0
    return None


def place_objects(specs, blocked, dist_road, rng):
    """specs를 순서대로 배치한다. (라벨 배열, 배치 성공 기록, 실패 수) 반환."""
    canvas = np.zeros((H, W), np.int32)
    placed, failed = [], 0
    for h, w, notch, access in specs:
        patch = rect_patch(h, w, notch)
        got = try_place(blocked, dist_road, patch, access, rng)
        if got is None:
            failed += 1
            continue
        y0, x0 = got
        full = np.zeros((H, W), bool)
        full[y0:y0 + h, x0:x0 + w] = patch
        pid = len(placed) + 1
        canvas[full] = pid
        # 객체 사이 간격을 확보해 연결요소가 서로 붙지 않게 한다
        blocked |= ndimage.binary_dilation(full, iterations=GAP)
        placed.append({"pid": pid, "access": access, "area": int(full.sum())})
    return canvas, placed, failed


# ------------------------------------------------------------------ 설계도
def vacant_specs(rng):
    """유휴 부지 설계. 임계(1,000㎡) 근처 필지군을 의도적으로 몰아 둔다."""
    specs = []

    # (a) 임계 근처 8필지: 참 면적 950~1,150㎡, 거의 정방형
    for _ in range(8):
        target = int(rng.integers(950, 1151))
        w = int(rng.integers(28, 37))
        specs.append((max(20, int(round(target / w))), w, None, True))

    # (b) 넉넉히 큰 7필지: 1,600~4,000㎡. 일부는 L자 형상
    for _ in range(7):
        target = int(rng.integers(1600, 4001))
        w = int(rng.integers(38, 60))
        h = max(30, int(round(target / w)))
        notch = None
        if rng.random() < 0.5:
            notch = (max(6, h // 3), max(6, w // 3), int(rng.integers(0, 4)))
        specs.append((h, w, notch, True))

    # (c) 요건 미달 소필지 8개: 300~850㎡
    for _ in range(8):
        target = int(rng.integers(300, 851))
        w = int(rng.integers(16, 28))
        specs.append((max(12, int(round(target / w))), w, None, True))

    # (d) 폭이 좁은 자투리 4개: 면적은 1,000㎡ 안팎이어도 최소폭에서 걸러져야 한다
    for _ in range(4):
        short = int(rng.integers(9, 13))
        long_ = int(rng.integers(95, 119))
        specs.append((long_, short, None, True) if rng.random() < 0.5
                     else (short, long_, None, True))

    # (e) 맹지 5개: 면적은 충분하지만 도로에 닿지 않는다
    for _ in range(5):
        target = int(rng.integers(1100, 2600))
        w = int(rng.integers(30, 46))
        specs.append((max(24, int(round(target / w))), w, None, False))

    # 제약이 센 것부터(맹지 → 큰 필지) 배치해야 자리가 남는다
    return sorted(specs, key=lambda s: (s[3], -s[0] * s[1]))


def confuser_specs(rng, n=N_BIG_FP):
    """유휴 부지로 오인되는 큰 오탐 덩어리(포장 주차장·공사장 부지)."""
    specs = []
    for _ in range(n):
        target = int(rng.integers(900, 2401))
        w = int(rng.integers(28, 48))
        specs.append((max(20, int(round(target / w))), w, None, True))
    return sorted(specs, key=lambda s: -s[0] * s[1])


def building_specs(rng, n=40):
    """기존 건물. 면적 200~1,300㎡, 종횡비 다양."""
    specs = []
    for _ in range(n):
        target = int(rng.integers(200, 1301))
        w = int(rng.integers(12, 40))
        h = min(max(10, int(round(target / w))), 55)
        notch = None
        if rng.random() < 0.25:
            notch = (max(5, h // 3), max(5, w // 3), int(rng.integers(0, 4)))
        specs.append((h, w, notch, bool(rng.random() < 0.85)))
    return sorted(specs, key=lambda s: -s[0] * s[1])


# ------------------------------------------------------ 세그멘테이션 오차
def erode_per_parcel(parcel_id, deep_ids):
    """필지마다 가장자리를 한 겹(deep_ids는 두 겹) 깎는다."""
    out = np.zeros(parcel_id.shape, bool)
    for pid in range(1, int(parcel_id.max()) + 1):
        m = parcel_id == pid
        it = 2 if pid in deep_ids else 1
        out |= ndimage.binary_erosion(m, iterations=it, border_value=0)
    return out


def make_split_band(parcel_id, split_ids, band_w=2):
    """지정 필지를 가로지르는 폭 2m의 띠(그늘·수목열이 마스크를 끊는 상황)."""
    band = np.zeros(parcel_id.shape, bool)
    rows = np.arange(H)[:, None]
    for pid in split_ids:
        m = parcel_id == pid
        ys, _ = np.nonzero(m)
        cy = int((ys.min() + ys.max()) / 2)
        band |= m & (rows >= cy) & (rows < cy + band_w)
    return band


def make_speckle(rng):
    """산발 오탐 얼룩(잡음이 만드는 작은 거짓 덩어리)."""
    fp = rng.random((H, W)) < FP_RATE
    return ndimage.binary_dilation(fp, iterations=1)


def main():
    print("=" * 64)
    print("실습 데이터 준비: 개발 가능 부지 탐색 (6-2, 6.7 분석 2)")
    print("=" * 64)

    road = build_roads()
    dist_road = ndimage.distance_transform_edt(~road)
    blocked = road.copy()

    # 유휴 부지 → 오탐 덩어리 → 기존 건물 순으로 배치한다(큰 것부터 자리를 잡아야
    # 뒤에 오는 작은 객체가 틈새에 들어간다)
    vac_canvas, vac_placed, vac_failed = place_objects(
        vacant_specs(RNG), blocked, dist_road, RNG)
    conf_canvas, conf_placed, conf_failed = place_objects(
        confuser_specs(RNG), blocked, dist_road, RNG)
    bld_canvas, bld_placed, bld_failed = place_objects(
        building_specs(RNG), blocked, dist_road, RNG)

    building = bld_canvas > 0
    confuser = conf_canvas > 0
    parcel_id = vac_canvas
    vacant_true = parcel_id > 0
    n_parcel = int(parcel_id.max())

    # ---- 오차 1: 경계 침식 (일부 필지는 두 겹)
    n_deep = max(1, int(round(n_parcel * P_DEEP_EROSION)))
    deep_ids = set(int(i) for i in RNG.choice(np.arange(1, n_parcel + 1),
                                              size=n_deep, replace=False))
    eroded = erode_per_parcel(parcel_id, deep_ids)

    # ---- 오차 3: 과분할. 중간 크기(1,500~2,600㎡) 필지에서 고른다.
    #      반으로 갈리면 두 조각이 모두 임계 아래로 내려가는 크기대다.
    sizes = np.array([int((parcel_id == p).sum()) for p in range(1, n_parcel + 1)])
    band_ids = [p for p in np.argsort(-sizes) + 1 if 1500 <= sizes[p - 1] <= 2600]
    split_ids = [int(p) for p in band_ids[:N_SPLIT]]
    split_band = make_split_band(parcel_id, split_ids)

    # ---- 오차 2: 오탐 (배치 단계에서 자리를 잡아 둔 큰 덩어리 + 산발 얼룩)
    fp_mask = confuser | make_speckle(RNG)
    n_big_fp = len(conf_placed)

    pred = (eroded & ~split_band) | fp_mask
    pred_c1 = (vacant_true & ~split_band) | fp_mask      # 침식만 제거
    pred_c2 = eroded & ~split_band                        # 오탐만 제거
    pred_c3 = eroded | fp_mask                            # 과분할만 제거

    out_path = DATA_DIR / "urban_block.npz"
    np.savez(out_path, road=road, building=building, vacant_true=vacant_true,
             parcel_id=parcel_id, pred=pred, pred_c1_noerode=pred_c1,
             pred_c2_nofp=pred_c2, pred_c3_nosplit=pred_c3,
             deep_erosion_ids=np.array(sorted(deep_ids), dtype=np.int32),
             split_ids=np.array(split_ids, dtype=np.int32))

    n_landlock = sum(1 for r in vac_placed if not r["access"])
    print(f"  래스터 {H}×{W} (픽셀 {PIXEL_M:.0f}m = 1㎡, 총 "
          f"{H * W / 10000:.1f}ha)")
    print(f"  도로 {int(road.sum()):,}㎡ | 기존 건물 {len(bld_placed)}동 "
          f"({int(building.sum()):,}㎡, 배치 실패 {bld_failed}) | "
          f"오탐 덩어리 {len(conf_placed)}개(배치 실패 {conf_failed})")
    print(f"  유휴 부지 정답 {n_parcel}필지 ({int(vacant_true.sum()):,}㎡, "
          f"배치 실패 {vac_failed}) | 그중 맹지 {n_landlock}필지")
    print(f"  필지 면적 최소 {sizes.min():,}㎡ / 중앙 {int(np.median(sizes)):,}㎡ "
          f"/ 최대 {sizes.max():,}㎡")
    print(f"  오차 심기 — 두 겹 침식 {len(deep_ids)}필지 | 과분할 "
          f"{len(split_ids)}필지(id {split_ids}) | 큰 오탐 {n_big_fp}개 + 산발 얼룩")
    print(f"  예측 마스크 {int(pred.sum()):,}㎡ "
          f"(정답 대비 {pred.sum() / vacant_true.sum():.3f}배)")
    print(f"  → {out_path.name} 저장 완료 (대조군 마스크 3종 포함)")
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
