"""
6장 실습(6.7): 변화 탐지 결과의 평가·거짓변화 필터링·행정구역 집계
=================================================================
질문: "U-Net/SAM이 산출한 변화 마스크를 어떻게 신뢰하고, 어떻게 정책 우선순위로 바꾸는가?"

세그멘테이션·변화탐지의 결론은 '픽셀'이 아니라 '구역별 변화량·우선순위'다.
이 실습은 그 전 과정을 한 사이클로 통합한다:
  세그멘테이션 출력(변화 마스크) → IoU/정밀도 평가 → 거짓변화 필터링 →
  행정구역별 변화 면적 집계 → 복구·규제 우선순위

재현성 원칙: 무거운 U-Net/SAM 학습 대신, 그 '출력(변화 마스크)'을 미리 준비한
  데이터(6-0-simdata-prep.py가 생성·저장)에서 불러온다
  (모델 자체는 5장 CNN·6.1~6.3 본문에서 다룸). 평가·필터링·집계는 실제 계산값.

거짓 변화(false change): 구름·그림자·계절·센서 차이로 생기는 가짜 변화.
  대응: NDVI 차분 임계 + 최소 변화 면적(연결요소) 필터. 정밀도가 오르는지로 검증.

데이터: 교육용 시뮬레이션(128×128 래스터, 픽셀 10m=0.01ha). seed 42.
실행:
    python 6-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 6-1-change-detection-policy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MASKS_PATH = DATA_DIR / "change_masks.npz"
if not MASKS_PATH.exists():
    raise SystemExit(
        f"데이터가 없습니다: {MASKS_PATH}\n"
        "먼저 실행: python 6-0-simdata-prep.py"
    )

H = W = 128                         # 래스터 크기
PIXEL_M = 10.0                     # 픽셀 한 변(m)
PIXEL_HA = (PIXEL_M ** 2) / 10000.0  # 픽셀 면적(ha) = 0.01
N_ADMIN = 4                         # 행정구역 4×4 = 16개


def iou_prf(pred, true):
    """예측·진짜 변화 마스크의 IoU·정밀도·재현율·Dice."""
    pred, true = pred.astype(bool), true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    iou = inter / union if union else 0.0
    prec = inter / pred.sum() if pred.sum() else 0.0
    rec = inter / true.sum() if true.sum() else 0.0
    dice = 2 * inter / (pred.sum() + true.sum()) if (pred.sum() + true.sum()) else 0.0
    return iou, prec, rec, dice


def load_masks():
    """미리 준비한 NDVI 래스터와 변화 마스크를 불러온다(6-0-simdata-prep.py 생성)."""
    d = np.load(MASKS_PATH)
    return d["ndvi_t1"], d["ndvi_t2"], d["true_change"], d["pred"]


def main():
    print("=" * 64)
    print("변화 탐지 결과의 평가·거짓변화 필터링·행정 집계 (6-1, 6.7 실습)")
    print("=" * 64)
    ndvi_t1, ndvi_t2, true_change, pred = load_masks()
    print(f"\n래스터 {H}×{W}(픽셀 {PIXEL_M:.0f}m, {PIXEL_HA}ha) | "
          f"진짜 변화 {true_change.sum()}px, 예측 변화 {pred.sum()}px")

    # -------- [6.2 평가] 필터 전 --------
    iou0, p0, r0, d0 = iou_prf(pred, true_change)
    print("\n[6.2 평가] 거짓변화 필터링 전 (예측 vs 진짜)")
    print(f"  IoU {iou0:.3f} | 정밀도 {p0:.3f} | 재현율 {r0:.3f} | Dice {d0:.3f}")
    print("  → 정밀도가 낮다 = 거짓양성(구름·계절 잡음)이 섞여 있다.")

    # -------- 거짓변화 필터링 --------
    dndvi = np.abs(ndvi_t2 - ndvi_t1)
    ndvi_ok = dndvi > 0.25                       # ① NDVI 차분 임계(진짜 토지변화는 큰 ΔNDVI)
    pred_f1 = np.logical_and(pred, ndvi_ok)
    # ② 최소 변화 면적: 연결요소 라벨링 → 작은 얼룩 제거(1ha=100px 미만 제거)
    min_px = int(1.0 / PIXEL_HA)                 # 1 ha = 100 px
    lbl, n = ndimage.label(pred_f1)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s >= min_px}
    pred_filt = np.isin(lbl, list(keep)) if keep else np.zeros_like(pred_f1)

    iou1, p1, r1, d1 = iou_prf(pred_filt, true_change)
    print(f"\n[거짓변화 필터] NDVI 차분>{0.25} + 최소면적≥1ha({min_px}px)")
    print(f"  필터 후: IoU {iou1:.3f} | 정밀도 {p1:.3f} | 재현율 {r1:.3f} | Dice {d1:.3f}")
    print(f"  → 정밀도 {p0:.3f} → {p1:.3f} (개선 {p1-p0:+.3f}), IoU {iou0:.3f} → {iou1:.3f}")
    print("  ※ 거짓양성 제거로 정밀도가 오른다. 재현율은 약간의 경계 손실 대가.")

    # -------- [6.6 정책 집계] 행정구역별 변화 면적 --------
    print("\n[6.6 정책] 행정구역(4×4=16)별 개발 변화 면적·우선순위")
    zone = (np.arange(H)[:, None] // (H // N_ADMIN)) * N_ADMIN + (np.arange(W)[None, :] // (W // N_ADMIN))
    rows = []
    for z in range(N_ADMIN * N_ADMIN):
        zmask = zone == z
        changed_px = int(np.logical_and(pred_filt, zmask).sum())
        rows.append({"admin_zone": z, "changed_px": changed_px,
                     "changed_ha": round(changed_px * PIXEL_HA, 2)})
    adf = pd.DataFrame(rows).sort_values("changed_ha", ascending=False)
    adf["priority_rank"] = range(1, len(adf) + 1)
    print(adf[adf["changed_ha"] > 0].to_string(index=False))

    csv_path = RESULTS_DIR / "change_admin_priority.csv"
    adf.to_csv(csv_path, index=False)
    total_ha = adf["changed_ha"].sum()
    print(f"\n  탐지된 개발 변화 총 {total_ha:.1f}ha, 우선순위 저장 → {csv_path.name}")
    print("  ※ 정책 연결: 구역별 변화 면적은 벌채 규제·복구·도시확장 관리의 우선순위 근거.")
    print("    필터 전 수치를 그대로 쓰면 거짓변화로 우선순위가 왜곡된다.")

    print("\n[완료] 변화 탐지 결과의 평가·필터·정책 집계를 마쳤다.")


if __name__ == "__main__":
    main()
