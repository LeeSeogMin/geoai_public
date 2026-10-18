"""
문제 2 (3장). 지도에 0개로 나온 업종을 비었다고 판정해도 되는가
=================================================================
서울 412개 동에서, 상가정보(등록부)에 있는 점포 중 지도(OSM)에 나타난 비율이 포착률이다.
다섯 업종의 포착률과 동별 점포 수는 학번으로 정해지는 모의 자료다.

실행: python exam/q2.py --id 학번
"""

import numpy as np

from common import key, parse_id, rng_of

KINDS = ["서점", "안경", "화장품", "약국", "카페"]
N_DONG = 412

sid = parse_id()
rng = rng_of(sid, 2)
lam = rng.uniform(0.3, 3.5, len(KINDS))
registry = [rng.negative_binomial(1.5, 1.5 / (1.5 + l), N_DONG) for l in lam]  # 동별 등록 점포 수
p_true = rng.uniform(0.04, 0.6, len(KINDS))
over = int(rng.integers(0, len(KINDS)))          # 대응표 문제가 있는 업종 하나
osm_total = []
for k, reg in enumerate(registry):
    r = int(reg.sum())
    osm_total.append(int(r * rng.uniform(1.03, 1.15)) if k == over else int(rng.binomial(r, p_true[k])))

CUT_BIZ, CUT_PUB = 0.80, 0.20

print(f"{'업종':<6}{'등록':>7}{'지도':>7}{'포착률':>8}{'0개 동 비율':>12}{'사후확률':>10}  판정")
for k, name in enumerate(KINDS):
    reg = registry[k]
    p = osm_total[k] / int(reg.sum())
    prior0 = float((reg == 0).mean())
    posterior = prior0                      # 0개 동의 비율을 그대로 쓴다
    verdict = "공백 판정" if posterior >= CUT_BIZ else "판정 불가"
    print(f"{name:<6}{int(reg.sum()):>7}{osm_total[k]:>7}{p:>8.3f}{prior0:>12.3f}{posterior:>10.3f}  {verdict}")
    key(f"p_{name}", float(p))
    key(f"post_{name}", float(posterior))
    key(f"verdict_{name}", verdict)
