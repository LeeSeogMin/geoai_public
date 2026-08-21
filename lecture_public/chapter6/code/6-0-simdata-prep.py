"""
6-0. 실습 데이터 준비: 변화 탐지 마스크·NDVI 래스터 저장
========================================================
이 장의 6-1 실습(6.7 분석 예제)이 쓰는 데이터셋을 미리 만들어 `data/`
폴더에 저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를
준비하고, 이후 6-1 코드는 저장된 파일을 불러와 평가·필터링·집계만 수행한다.

여기서 만드는 데이터는 실제 위성 세그멘테이션 파이프라인의 산출물을 그대로
구하기 어려운 대상이라, 현실적인 값과 공간 구조를 갖도록 합성한 교육용
데이터다. 무거운 U-Net/SAM 학습을 실행하는 대신 그 '출력(변화 마스크)'을
시뮬레이션한다(모델 자체는 5장 CNN·6.1~6.3 본문에서 다룸). 이렇게 만든
데이터에는 정답을 알고 있는 다섯 개의 개발 패치(진짜 변화), 계절성 잡음(거짓
변화의 원천), 경계 침식(거짓음성), 산발 얼룩(거짓양성)이 의도적으로 들어가
있어, 6-1의 평가·필터링·집계가 각각 무엇을 회복·제거하는지 검증할 수 있다.

저장 데이터(128×128 래스터, 픽셀 10m=0.01ha):
- ndvi_t1, ndvi_t2 : 두 시점의 NDVI(float 배열)
- true_change      : 정답 변화 마스크(bool 배열)
- pred             : U-Net/SAM 예측 변화 마스크(bool 배열)

재현성: 시드가 걸린 전용 난수 생성기(default_rng(42))로 생성하므로, 이
스크립트를 다시 실행해도 동일한 배열이 나온다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter6/code/6-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

H = W = 128                          # 래스터 크기
PIXEL_M = 10.0                       # 픽셀 한 변(m)
PIXEL_HA = (PIXEL_M ** 2) / 10000.0  # 픽셀 면적(ha) = 0.01


def disk(cy, cx, r):
    """중심 (cy,cx), 반지름 r의 원형 마스크."""
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def simulate():
    """t1/t2 NDVI와 '진짜 변화'(산림→개발) + U-Net/SAM '예측 변화'(거짓양성·음성) 생성."""
    # t1 NDVI: 산림(고NDVI) 배경 + 약간의 공간 변동
    base = 0.65 + 0.08 * np.sin(np.linspace(0, 3, H))[:, None] + RNG.normal(0, 0.03, (H, W))
    ndvi_t1 = np.clip(base, 0.1, 0.9)

    # 진짜 변화: 5개 개발 패치(산림 → 도시/나지) → NDVI 급감
    true_change = np.zeros((H, W), bool)
    centers = [(30, 30, 12), (40, 95, 10), (95, 40, 14), (100, 100, 9), (70, 70, 11)]
    ndvi_t2 = ndvi_t1.copy()
    for cy, cx, r in centers:
        m = disk(cy, cx, r)
        true_change |= m
        ndvi_t2[m] = np.clip(0.15 + RNG.normal(0, 0.03, m.sum()), 0.05, 0.4)  # 개발 후 저NDVI

    # 계절성 잡음: 전역적으로 NDVI 소폭 변동(거짓 변화의 원천, ΔNDVI 작음)
    ndvi_t2 += RNG.normal(0, 0.04, (H, W))
    ndvi_t2 = np.clip(ndvi_t2, 0.05, 0.9)

    # U-Net/SAM '예측 변화': 진짜 변화 + 거짓양성(작은 산발 잡음) + 거짓음성(경계 침식)
    pred = true_change.copy()
    pred = ndimage.binary_erosion(pred, iterations=1)             # 거짓음성: 경계 일부 누락
    fp = RNG.random((H, W)) < 0.02                                # 거짓양성: 2% 산발 픽셀
    fp = ndimage.binary_dilation(fp, iterations=1)               # 작은 얼룩으로
    pred = np.logical_or(pred, fp)
    return ndvi_t1, ndvi_t2, true_change, pred


def main():
    print("=" * 60)
    print("실습 데이터 준비 (6-1, 6.7 분석 예제)")
    print("=" * 60)
    ndvi_t1, ndvi_t2, true_change, pred = simulate()

    # 부동소수점을 정확히 보존하도록 .npz(비압축)로 저장(평가·필터 재현성 확보)
    out_path = DATA_DIR / "change_masks.npz"
    np.savez(out_path, ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
             true_change=true_change, pred=pred)
    print(f"  래스터 {H}×{W}(픽셀 {PIXEL_M:.0f}m, {PIXEL_HA}ha) | "
          f"진짜 변화 {int(true_change.sum())}px, 예측 변화 {int(pred.sum())}px")
    print(f"  → {out_path.name} 저장 완료")
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
