"""
문제 4 (5장). CNN을 도입해도 되는가
====================================
토지피복 분류에서 세 방법의 폴드별 정확도(5개 공간 타일 폴드)가 아래처럼 나왔다.
이 값은 시험용 모의 점수이며, 실제 CNN을 학습해 얻은 값이 아니다. 학번마다 다르다.

  R1  고정 임계 규칙  (학습 없음)
  L   로지스틱 회귀  (값싼 기준선)
  CNN 합성곱 신경망  (연 개발·운영비가 L보다 더 든다)

도입 조건(기관이 정한 값)
  연 판독 건수 200,000건, 오분류 1건 처리비 2,000원, CNN 추가비용 연 3,000,000원

보고서의 문장: "평균 정확도가 가장 높은 방법을 도입한다."

실행: python exam/q4.py --id 학번
"""

import numpy as np

from common import key, parse_id, rng_of

ANNUAL_CASES = 200_000
COST_PER_ERROR = 2_000
CNN_EXTRA_COST = 3_000_000

sid = parse_id()
rng = rng_of(sid, 4)
fold_shift = rng.normal(0, 0.012, 5)            # 폴드마다 어려운 정도(모든 방법에 공통)
delta = rng.uniform(-0.005, 0.045)              # CNN이 L보다 낫거나 못한 정도
acc = {
    "R1": 0.588 + fold_shift + rng.normal(0, 0.008, 5),
    "L": 0.947 + fold_shift + rng.normal(0, 0.008, 5),
    "CNN": 0.947 + delta + fold_shift + rng.normal(0, 0.008, 5),
}

print(f"{'방법':<5}{'폴드1':>8}{'폴드2':>8}{'폴드3':>8}{'폴드4':>8}{'폴드5':>8}{'평균':>8}")
for m, a in acc.items():
    print(f"{m:<5}" + "".join(f"{v:>8.3f}" for v in a) + f"{a.mean():>8.3f}")

best = max(acc, key=lambda m: acc[m].mean())
print(f"평균이 가장 높은 방법: {best}")
print(f"결론: {best} 도입")
key("best_by_mean", best)
key("decision", "CNN 도입" if best == "CNN" else "CNN 도입 안 함")
