"""
문제 5 (6장). 거짓변화 필터는 얼마나 도움이 되는가
====================================================
변화 탐지 모델이 낸 예측 마스크(pred)를 진짜 변화(true_change)와 비교한다.
128x128 래스터 가운데 학번으로 정한 96x96 구역을 쓴다. 픽셀 한 변은 10m이다.
필터는 NDVI 차분 0.25 초과 + 최소 면적 1ha 이상이다.

보고서의 문장: "정확도가 높고 필터를 거친 뒤에도 유지되므로 이 모델은 믿을 만하다."

실행: python exam/q5.py --id 학번
"""

import numpy as np
from scipy import ndimage

from common import PRACTICE, key, parse_id, rng_of

sid = parse_id()
rng = rng_of(sid, 5)
d = np.load(PRACTICE / "chapter6" / "data" / "change_masks.npz")
r0, c0 = (int(v) for v in rng.integers(0, 33, 2))
sl = (slice(r0, r0 + 96), slice(c0, c0 + 96))
ndvi1, ndvi2 = d["ndvi_t1"][sl], d["ndvi_t2"][sl]
true = d["true_change"][sl].astype(bool)
pred = d["pred"][sl].astype(bool)

PIXEL_M = 10.0
PIXEL_HA = PIXEL_M / 10000.0          # 픽셀 한 개의 면적(ha)


def score(p, t):
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    acc = (tp + tn) / p.size
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return acc, iou, prec, rec


print(f"구역 시작 위치 ({r0}, {c0}), 진짜 변화 {true.sum()}px, 예측 변화 {pred.sum()}px")
a0, i0, p0, r0_ = score(pred, true)
print(f"필터 전: 정확도 {a0:.3f} | IoU {i0:.3f} | 정밀도 {p0:.3f} | 재현율 {r0_:.3f}")

ok = np.abs(ndvi2 - ndvi1) > 0.25
f1 = np.logical_and(pred, ok)
min_px = int(1.0 / PIXEL_HA)            # 1ha에 해당하는 픽셀 수
lbl, n = ndimage.label(f1)
sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
filt = np.isin(lbl, keep)

a1, i1, p1, r1 = score(filt, true)
print(f"최소 면적 기준: {min_px}px")
print(f"필터 후: 정확도 {a1:.3f} | IoU {i1:.3f} | 정밀도 {p1:.3f} | 재현율 {r1:.3f}")
print("보고: 정확도가 높게 유지되므로 믿을 만하다.")
key("min_px", min_px)
key("acc_after", float(a1))
key("iou_before", float(i0))
key("iou_after", float(i1))
key("precision_before", float(p0))
key("precision_after", float(p1))
key("recall_before", float(r0_))
key("recall_after", float(r1))
