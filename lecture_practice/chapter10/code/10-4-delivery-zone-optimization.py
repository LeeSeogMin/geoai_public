"""
10장 실습 4(비즈니스): 배달 권역 설계와 원가 컷라인 — 최적화와 단위 경제
=====================================================================
비즈니스 질문: "거점을 어디에 몇 개 두고, 어디까지 배달하면 남는가?"

10-1이 "주민 → 시설" 방향으로 형평(사각지대)을 물었다면, 이 분석은
"거점 → 고객" 방향으로 수익성을 묻는다. 계산 도구(거리·도달시간)는 같고
목적함수가 다르다. 예측 모델을 쓰지 않는다 — 이 문제는 예측이 아니라 결정이다.

무엇을 계산하는가
  [1] 최적화: 같은 후보에서 목적함수를 바꾸면 거점 배치가 달라지는가
      · p-median      수요가중 총 도달시간 최소화 (ReVelle & Swain, 1970)
      · 최대커버(MCLP) T분 내 커버 수요 최대화 (Church & ReVelle, 1974)
      · 용량제약       라이더 시간 상한 아래 배차 + 미배차 수요 노출
  [2] 원가 컷라인: 건당 기여이익이 0이 되는 주행시간(손익분기 등시선)
  [3] 공급 제약: 평시 → 피크에서 이용률·배차 대기가 컷라인을 얼마나 당기는가
  [4] 중첩: 컷라인 밖 격자와 생활 인프라 열위 격자가 겹치는가 (+ 대조군)
  [5] 퇴화 검사: 최적화가 단순 규칙("음식점 많은 셀", "수요 큰 셀")을 되풀이하는가
  [6] 민감도: 임률·수수료·묶음·라이더 수·거리정의에서 결론이 버티는가

데이터: 실데이터 스냅샷(`10-0b-delivery-data-prep.py`가 만든다).
  서울 상가(상권)정보 + 생활인구(행정동). 격자 500m.

원가 파라미터는 **공개 통계가 없다.** 아래 CONFIG의 값은 전부 가정이며,
결론은 절대 금액이 아니라 **순위와 민감도 범위**로 읽는다.

실행:
    python 10-0b-delivery-data-prep.py      # 최초 1회: 스냅샷 생성(원자료 필요)
    python 10-4-delivery-zone-optimization.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

# 크로스 플랫폼 한글 폰트: 없으면 라벨만 □로 보이고 수치에는 영향 없음.
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# =====================================================================
# CONFIG — 값의 출처를 셋으로 구분한다: [실데이터] [표준] [가정]
# =====================================================================
P_HUBS = 10                 # [가정] 운영 거점 수
N_CAND = 40                 # [가정] 후보 거점 = 음식점 밀집 상위 N개 셀(픽업이 음식점에서 일어난다)
T_COVER = 5.0               # [가정] 커버 임계 — 거점→고객 편도 주행 5분(픽업·전달 시간은 별도)
SPEED_OFFPEAK = 22.0        # [가정] 이륜차 평균 속도(km/h), 평시
SPEED_PEAK = 16.0           # [가정] 피크 혼잡 시 속도
DETOUR = 1.35               # [가정] 우회계수(직선거리 → 실주행거리). 도로망 대신 쓰는 근사
WALK_KMH, WALK_DETOUR = 4.0, 1.2   # [가정] 생활 인프라 도보 접근 계산용

ORDER_RATE = 1.2            # [가정] 평시 시간당 주문 = 생활인구 1,000명당 1.2건
ORDER_RATE_PEAK = 2.5       # [가정] 저녁 식사 시간대에는 같은 인구가 더 많이 주문한다
FEE_PER_ORDER = 4500.0      # [가정] 건당 수수료 수입(원)
WAGE_PER_MIN = 300.0        # [가정] 라이더 시간 임률(원/분) ≈ 시간당 18,000원
FIXED_MIN = 6.0             # [가정] 픽업·전달 고정시간(분/건)
VAR_COST = 400.0            # [가정] 건당 변동비(연료·통신·보험 등, 원)
BUNDLE_MAX_GAIN = 1.0       # [가정] 묶음배송으로 주행시간이 최대 2배까지 나뉜다(계수 1→2)
DENSITY_REF = 6.0           # [가정] 묶음이 최대가 되는 셀 주문밀도(건/시간)
RIDERS_PER_HUB = 25         # [가정] 평시 거점당 라이더 수
PEAK_RIDER_MULT = 2.0       # [가정] 피크에는 라이더가 더 나온다(그래도 수요 증가를 다 못 따라간다)
WAIT_COEF = 1.0             # [가정] 대기 근사 계수(분) — ρ/(1−ρ)에 곱한다(서비스 품질 지표)
SURGE_MAX = 0.6             # [가정] 이용률이 100%에 이르면 유효 임률이 최대 60% 할증된다
RHO_CAP = 0.95              # [표준] 대기 근사의 발산 방지 상한

HUB_CAPACITY_MIN = RIDERS_PER_HUB * 60.0   # 거점당 시간당 라이더 가용 분


# =====================================================================
# 1. 스냅샷 읽기
# =====================================================================
def load_snapshot() -> tuple[pd.DataFrame, np.ndarray, dict]:
    grid_path = DATA_DIR / "delivery_grid.parquet"
    infra_path = DATA_DIR / "delivery_infra.parquet"
    meta_path = DATA_DIR / "delivery_snapshot_meta.json"
    if not grid_path.exists():
        raise SystemExit(
            f"스냅샷이 없습니다: {grid_path}\n"
            "먼저 실행: python 10-0b-delivery-data-prep.py\n"
            "(원자료 내려받는 곳은 10-0b 파일 머리말 참조)")
    cells = pd.read_parquet(grid_path)
    infra = pd.read_parquet(infra_path)[["x_m", "y_m"]].to_numpy()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return cells, infra, meta


# =====================================================================
# 2. 거리·시간
# =====================================================================
def travel_time(cand_xy: np.ndarray, cell_xy: np.ndarray, speed: float,
                metric: str = "euclid") -> np.ndarray:
    """후보 거점 → 격자 편도 주행시간(분). 도로망 대신 두 가지 근사를 쓴다.

    euclid    : 직선거리 × 우회계수 (우회계수 하나로 도로망을 대신한다)
    manhattan : 가로망 근사(|Δx|+|Δy|) — 우회를 거리 정의로 흡수하므로 계수를 곱하지 않는다
    """
    dx = cand_xy[:, 0][:, None] - cell_xy[:, 0][None, :]
    dy = cand_xy[:, 1][:, None] - cell_xy[:, 1][None, :]
    if metric == "manhattan":
        dist_km = (np.abs(dx) + np.abs(dy)) / 1000.0
    else:
        dist_km = np.sqrt(dx ** 2 + dy ** 2) / 1000.0 * DETOUR
    return dist_km / speed * 60.0


def nearest_walk_minutes(cell_xy: np.ndarray, poi_xy: np.ndarray) -> np.ndarray:
    """각 격자에서 가장 가까운 생활 인프라 POI까지 도보 시간(분)."""
    if len(poi_xy) == 0:
        return np.full(len(cell_xy), np.nan)
    d = np.sqrt((cell_xy[:, 0][:, None] - poi_xy[:, 0][None, :]) ** 2
                + (cell_xy[:, 1][:, None] - poi_xy[:, 1][None, :]) ** 2).min(axis=1)
    return d / 1000.0 * WALK_DETOUR / WALK_KMH * 60.0


# =====================================================================
# 3. 최적화 — 정확해(MILP, HiGHS)와 휴리스틱(Teitz–Bart)
# =====================================================================
def _milp():
    """scipy.optimize.milp을 쓸 수 있으면 돌려준다(scipy>=1.9)."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        return milp, LinearConstraint, Bounds
    except ImportError:
        return None, None, None


def _assignment_rows(J: int, I: int) -> sparse.csr_matrix:
    """Σ_j x_ji (+ u_i) = 1 제약 행렬. 변수 순서는 [y(J), x(J*I), u(I)]."""
    rows, cols = [], []
    for i in range(I):
        for j in range(J):
            rows.append(i)
            cols.append(J + j * I + i)
        rows.append(i)
        cols.append(J + J * I + i)          # 미배차 슬랙
    data = np.ones(len(rows))
    return sparse.csr_matrix((data, (rows, cols)), shape=(I, J + J * I + I))


def _linking_rows(J: int, I: int) -> sparse.csr_matrix:
    """x_ji − y_j ≤ 0 (닫힌 거점에 배차 금지)."""
    n = J * I
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    data = np.empty(2 * n)
    k = 0
    for j in range(J):
        for i in range(I):
            cols[k], data[k] = J + j * I + i, 1.0
            cols[k + 1], data[k + 1] = j, -1.0
            k += 2
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, J + J * I + I))


def solve_pmedian(t: np.ndarray, d: np.ndarray, p: int,
                  occ: np.ndarray | None = None, cap: float | None = None,
                  penalty_min: float | None = None) -> dict:
    """수요가중 총 도달시간 최소화. occ·cap을 주면 용량 제약 + 미배차 슬랙을 넣는다.

    x는 [0,1] 연속으로 둔다. 무제약에서는 최적해가 최근접 배차가 되므로 정확하고,
    용량 제약에서는 한 격자를 두 거점이 나눠 담당할 수 있게 된다(배달에서는 현실적).
    """
    milp, LinearConstraint, Bounds = _milp()
    J, I = t.shape
    if milp is None:
        return solve_pmedian_heuristic(t, d, p)

    pen = penalty_min if penalty_min is not None else 3.0 * float(t.max())
    c = np.concatenate([np.zeros(J), (d[None, :] * t).ravel(), pen * d])

    cons = [LinearConstraint(_assignment_rows(J, I), 1.0, 1.0),
            LinearConstraint(_linking_rows(J, I), -np.inf, 0.0)]
    card = sparse.csr_matrix((np.ones(J), (np.zeros(J, dtype=int), np.arange(J))),
                             shape=(1, J + J * I + I))
    cons.append(LinearConstraint(card, p, p))

    if occ is not None and cap is not None:
        rows, cols, data = [], [], []
        for j in range(J):
            for i in range(I):
                rows.append(j)
                cols.append(J + j * I + i)
                data.append(d[i] * occ[j, i])
            rows.append(j)
            cols.append(j)
            data.append(-cap)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(J, J + J * I + I))
        cons.append(LinearConstraint(A, -np.inf, 0.0))

    integrality = np.concatenate([np.ones(J), np.zeros(J * I + I)])
    res = milp(c=c, constraints=cons, integrality=integrality,
               bounds=Bounds(0, 1), options={"time_limit": 300, "presolve": True})
    if not res.success:
        return solve_pmedian_heuristic(t, d, p)

    y = res.x[:J]
    x = res.x[J:J + J * I].reshape(J, I)
    u = res.x[J + J * I:]
    hubs = np.flatnonzero(y > 0.5)
    return {"hubs": hubs, "x": x, "unserved": u, "solver": "MILP(HiGHS)"}


def solve_pmedian_heuristic(t: np.ndarray, d: np.ndarray, p: int) -> dict:
    """탐욕 추가 + 정점 치환(Teitz & Bart, 1968). MILP 대조·퇴화 검사용."""
    J = t.shape[0]
    hubs: list[int] = []
    best = np.full(t.shape[1], np.inf)
    for _ in range(p):
        gains = [(float((d * np.minimum(best, t[j])).sum()), j)
                 for j in range(J) if j not in hubs]
        _, j_best = min(gains)
        hubs.append(j_best)
        best = np.minimum(best, t[j_best])

    improved = True
    while improved:                       # 정점 치환: 든 것과 빠진 것을 하나씩 교환
        improved = False
        cur = float((d * t[hubs].min(axis=0)).sum())
        for out in list(hubs):
            for cand in range(J):
                if cand in hubs:
                    continue
                trial = [h for h in hubs if h != out] + [cand]
                val = float((d * t[trial].min(axis=0)).sum())
                if val < cur - 1e-9:
                    hubs, cur, improved = trial, val, True
                    break
            if improved:
                break
    hubs_arr = np.array(sorted(hubs))
    x = np.zeros(t.shape)
    x[hubs_arr[t[hubs_arr].argmin(axis=0)], np.arange(t.shape[1])] = 1.0
    return {"hubs": hubs_arr, "x": x, "unserved": np.zeros(t.shape[1]),
            "solver": "탐욕+정점치환(Teitz–Bart)"}


def solve_mclp(t: np.ndarray, d: np.ndarray, p: int, threshold: float) -> dict:
    """임계 T분 내 커버 수요 최대화 (Church & ReVelle, 1974)."""
    milp, LinearConstraint, Bounds = _milp()
    J, I = t.shape
    cover = (t <= threshold)
    if milp is None:                       # 탐욕 커버(대체)
        hubs, covered = [], np.zeros(I, dtype=bool)
        for _ in range(p):
            gain = [(float(d[covered | cover[j]].sum()), j)
                    for j in range(J) if j not in hubs]
            _, j_best = max(gain)
            hubs.append(j_best)
            covered |= cover[j_best]
        return {"hubs": np.array(sorted(hubs)), "solver": "탐욕 커버"}

    c = np.concatenate([np.zeros(J), -d])              # 변수: [y(J), z(I)]
    rows, cols, data = [], [], []
    for i in range(I):
        rows.append(i)
        cols.append(J + i)
        data.append(1.0)
        for j in np.flatnonzero(cover[:, i]):
            rows.append(i)
            cols.append(int(j))
            data.append(-1.0)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(I, J + I))
    card = sparse.csr_matrix((np.ones(J), (np.zeros(J, dtype=int), np.arange(J))),
                             shape=(1, J + I))
    res = milp(c=c, constraints=[LinearConstraint(A, -np.inf, 0.0),
                                 LinearConstraint(card, p, p)],
               integrality=np.concatenate([np.ones(J), np.zeros(I)]),
               bounds=Bounds(0, 1), options={"time_limit": 300})
    if not res.success:
        return solve_mclp(t, d, p, threshold)
    return {"hubs": np.flatnonzero(res.x[:J] > 0.5), "solver": "MILP(HiGHS)"}


# =====================================================================
# 4. 단위 경제 — 묶음계수·점유시간·기여이익·컷라인
# =====================================================================
def bundle_factor(orders: np.ndarray) -> np.ndarray:
    """주문 밀도가 높은 셀에서는 주행을 여러 건이 나눈다(묶음배송).

    밀도가 낮으면 계수가 1에 가까워 한 건이 주행을 다 부담한다 —
    라스트마일 원가가 고객 밀도에 좌우된다는 문헌의 방향과 같다(Boyer et al., 2009).
    """
    return 1.0 + BUNDLE_MAX_GAIN * np.minimum(1.0, orders / DENSITY_REF)


def unit_economics(t_assigned: np.ndarray, orders: np.ndarray,
                   fee: float = FEE_PER_ORDER,
                   wage: float | np.ndarray = WAGE_PER_MIN,
                   bundle_gain: float = BUNDLE_MAX_GAIN) -> dict:
    """격자별 건당 점유시간·원가·기여이익.

    점유시간에 배차 대기를 넣지 않는다. 대기는 주문이 기다리는 시간이지 라이더가
    일하는 시간이 아니므로, 대기를 점유시간에 더하면 임금을 두 번 세게 된다.
    혼잡이 원가에 들어오는 경로는 따로 있다 — 유효 임률의 할증(wage 인자).
    """
    bundle = 1.0 + bundle_gain * np.minimum(1.0, orders / DENSITY_REF)
    occupancy = FIXED_MIN + 2.0 * t_assigned / bundle
    cost = wage * occupancy + VAR_COST
    margin = fee - cost
    return {"bundle": bundle, "occupancy": occupancy, "cost": cost,
            "margin": margin, "cell_margin": margin * orders}


def breakeven_oneway_minutes(bundle: float, fee: float = FEE_PER_ORDER,
                             wage: float = WAGE_PER_MIN) -> float:
    """기여이익이 0이 되는 편도 주행시간(분) — 손익분기 등시선의 반지름."""
    slack = fee - VAR_COST - wage * FIXED_MIN
    return slack / wage * bundle / 2.0


def hub_utilization(t: np.ndarray, x: np.ndarray, orders: np.ndarray,
                    hubs: np.ndarray, capacity: float) -> tuple[np.ndarray, np.ndarray]:
    """거점 이용률과 배차 대기 근사.

    이용률 ρ = (배차된 주문이 요구하는 라이더 시간) / (거점의 가용 라이더 시간).
    대기는 ρ/(1−ρ) 형태로 근사한다 — 대기행렬의 표준적인 모양이며 여기서는
    **서비스 품질 지표**로만 쓴다(원가에는 할증 임률로 들어간다).
    """
    bundle = bundle_factor(orders)
    occ = FIXED_MIN + 2.0 * t / bundle[None, :]
    load = np.array([float((orders * x[j] * occ[j]).sum()) for j in hubs])
    rho = load / capacity
    rho_w = np.minimum(rho, RHO_CAP)
    wait = WAIT_COEF * rho_w / (1.0 - rho_w)
    return rho, wait


# =====================================================================
# 5. 시나리오 실행
# =====================================================================
def assigned_time(t: np.ndarray, x: np.ndarray, hubs: np.ndarray) -> np.ndarray:
    """배차된 거점까지의 평균 편도 시간. 분할 배차와 미배차를 함께 처리한다.

    용량 제약에서 배차받지 못한 격자(미배차)는 거리 때문이 아니라 공급 때문에
    남은 것이므로, 최근접 거점까지의 시간으로 채워 원가를 계산한다.
    """
    served = x[hubs].sum(axis=0)
    weighted = np.einsum("ji,ji->i", t[hubs], x[hubs])
    nearest = t[hubs].min(axis=0)
    return np.where(served > 1e-6, weighted / np.where(served > 1e-6, served, 1.0), nearest)


def _weighted_by_hub(values: np.ndarray, orders: np.ndarray, x: np.ndarray,
                     hubs: np.ndarray) -> float:
    """거점별 값을 그 거점이 실제로 받은 주문량으로 가중 평균한다."""
    if not len(hubs):
        return 0.0
    w = np.array([float((orders * x[j]).sum()) for j in hubs])
    return float((values * w).sum() / w.sum()) if w.sum() > 0 else 0.0


def run_scenario(cells: pd.DataFrame, cand_idx: np.ndarray, pop_col: str,
                 speed: float, p: int, capacity: float | None,
                 metric: str = "euclid", label: str = "",
                 order_rate: float = ORDER_RATE) -> dict:
    cell_xy = cells[["x_m", "y_m"]].to_numpy()
    cand_xy = cell_xy[cand_idx]
    t = travel_time(cand_xy, cell_xy, speed, metric)
    orders = cells[pop_col].to_numpy() / 1000.0 * order_rate
    pop = cells[pop_col].to_numpy()

    if capacity is None:
        sol = solve_pmedian(t, orders, p)
        hubs, x = sol["hubs"], sol["x"]
        wait_hub = np.zeros(len(hubs))
        rho = np.zeros(len(hubs))
        wage_cell = np.full(len(cells), WAGE_PER_MIN)
    else:
        bundle = bundle_factor(orders)
        occ = FIXED_MIN + 2.0 * t / bundle[None, :]
        sol = solve_pmedian(t, orders, p, occ=occ, cap=capacity)
        hubs, x = sol["hubs"], sol["x"]
        rho, wait_hub = hub_utilization(t, x, orders, hubs, capacity)
        # 혼잡의 원가 경로: 라이더를 더 붙잡아 두려면 할증을 얹어야 한다(가정).
        served = x[hubs].sum(axis=0)
        rho_cell = np.where(served > 1e-6,
                            np.einsum("k,ki->i", rho, x[hubs]) / np.where(served > 1e-6, served, 1.0),
                            0.0)
        # 할증은 이용률이 포화에 가까워질 때 급해진다 → ρ의 3승을 쓴다(가정).
        wage_cell = WAGE_PER_MIN * (1.0 + SURGE_MAX * np.minimum(rho_cell, 1.0) ** 3)

    t_cell = assigned_time(t, x, hubs)
    econ = unit_economics(t_cell, orders, wage=wage_cell)

    served_share = x[hubs].sum(axis=0)
    outside = econ["margin"] < 0
    # 배차받은 주문에만 가중해 평균 임률을 낸다(미배차 격자는 임률이 없다).
    w = orders * served_share
    wage_served = float((w * wage_cell).sum() / w.sum()) if w.sum() > 0 else WAGE_PER_MIN
    # hubs는 후보 배열 기준 색인이다. 격자 배열 기준 색인이 필요할 때가 많아 함께 둔다.
    return {"label": label, "t": t, "orders": orders, "pop": pop, "hubs": hubs,
            "hub_cells": np.asarray(cand_idx)[hubs], "cand_idx": np.asarray(cand_idx), "x": x,
            "t_cell": t_cell, "econ": econ, "outside": outside, "wage_cell": wage_cell,
            "wait_hub": wait_hub, "rho": rho, "served_share": served_share,
            "wage_served": wage_served,
            "rho_weighted": _weighted_by_hub(rho, orders, x, hubs),
            "wait_weighted": _weighted_by_hub(wait_hub, orders, x, hubs),
            "unserved_share": float((orders * sol["unserved"]).sum() / orders.sum()),
            "solver": sol["solver"],
            "avg_time": float(np.average(t_cell, weights=orders)),
            "cover_share": float(orders[t_cell <= T_COVER].sum() / orders.sum()),
            "outside_pop_share": float(pop[outside].sum() / pop.sum()),
            "outside_cells": int(outside.sum())}


# =====================================================================
# 6. 출력 블록
# =====================================================================
def print_optimization_table(cells, cand_idx, base, mclp_hubs, cap_res, heur, alt):
    print("\n[1] 최적화 — 목적함수가 거점 배치를 바꾼다 (표 10.7)")
    print("-" * 68)
    print(f"  후보 거점 {len(cand_idx)}개(음식점 밀집 상위), 개설 {P_HUBS}개, 해법: {base['solver']}")
    cell_xy = cells[["x_m", "y_m"]].to_numpy()
    t_base = base["t"]
    orders = base["orders"]
    pop = base["pop"]

    def metrics(hubs):
        tt = t_base[hubs].min(axis=0)
        return (float(np.average(tt, weights=orders)),
                float(orders[tt <= T_COVER].sum() / orders.sum()),
                float(pop[tt <= T_COVER].sum() / pop.sum()))

    rows = []
    for name, hubs, extra in [
            ("p-median", base["hubs"], ""),
            (f"최대커버(T={T_COVER:.0f}분)", mclp_hubs, ""),
            ("용량제약 p-median", cap_res["hubs"],
             f"미배차 {cap_res['unserved_share']:.1%}"),
            ("휴리스틱(정점치환)", heur["hubs"], ""),
    ]:
        avg_t, cov_d, cov_p = metrics(np.asarray(hubs))
        overlap = len(set(np.asarray(hubs).tolist()) & set(base["hubs"].tolist()))
        rows.append((name, avg_t, cov_d, cov_p, overlap, extra))

    print(f"  {'해':<20}{'평균 도달(분)':>13}{'T분 커버 수요':>14}{'커버 인구':>11}"
          f"{'p-median과 공통':>16}  비고")
    for name, avg_t, cov_d, cov_p, ov, extra in rows:
        print(f"  {name:<20}{avg_t:>13.2f}{cov_d:>13.1%}{cov_p:>11.1%}"
              f"{ov:>12}/{P_HUBS}  {extra}")

    base_obj = float((orders * t_base[base['hubs']].min(axis=0)).sum())
    heur_obj = float((orders * t_base[heur['hubs']].min(axis=0)).sum())
    print(f"\n  휴리스틱 목적함수 격차: {(heur_obj / base_obj - 1):+.3%} "
          f"(정확해 대비. Teitz–Bart는 전역 최적을 보장하지 않는다)")
    print(f"  해 안정성 — p={alt['p_list']}: 공통 거점 수 {alt['p_overlap']}, "
          f"후보집합 교체 시 공통 {alt['cand_overlap']}/{P_HUBS}")


def print_cutline_table(res, cells):
    econ, orders, pop = res["econ"], res["orders"], res["pop"]
    outside = res["outside"]
    print("\n[2] 원가 컷라인 — 어디까지 배달하면 남는가 (표 10.8)")
    print("-" * 68)
    print(f"  건당 수수료 {FEE_PER_ORDER:,.0f}원, 임률 {WAGE_PER_MIN:,.0f}원/분, "
          f"고정 {FIXED_MIN:.0f}분, 변동비 {VAR_COST:,.0f}원  ← 전부 가정값")
    be_lo = breakeven_oneway_minutes(1.0)
    be_hi = breakeven_oneway_minutes(1.0 + BUNDLE_MAX_GAIN)
    print(f"  손익분기 편도 주행시간: 묶음 없음(계수 1.0) {be_lo:.1f}분 → "
          f"최대 묶음(계수 {1 + BUNDLE_MAX_GAIN:.1f}) {be_hi:.1f}분")
    q = np.percentile(econ["margin"], [10, 25, 50, 75, 90])
    print(f"  건당 기여이익 분위(원): p10 {q[0]:,.0f} | p25 {q[1]:,.0f} | "
          f"중앙 {q[2]:,.0f} | p75 {q[3]:,.0f} | p90 {q[4]:,.0f}")
    print(f"  컷라인 밖: 격자 {int(outside.sum())}개/{len(cells)} "
          f"({outside.mean():.1%}), 인구 {pop[outside].sum():,.0f}명 "
          f"({res['outside_pop_share']:.1%}), 주문 "
          f"{orders[outside].sum() / orders.sum():.1%}")
    if outside.any():
        print(f"  컷라인 밖 격자의 특성: 평균 도달 {res['t_cell'][outside].mean():.1f}분 "
              f"(안쪽 {res['t_cell'][~outside].mean():.1f}분), "
              f"주문밀도 {orders[outside].mean():.2f}건/시간 "
              f"(안쪽 {orders[~outside].mean():.2f}), "
              f"묶음계수 {econ['bundle'][outside].mean():.2f} "
              f"(안쪽 {econ['bundle'][~outside].mean():.2f})")
    frame = pd.DataFrame({"sigungu": cells["sigungu"].to_numpy(),
                          "dong": cells["dong"].to_numpy(),
                          "pop": pop,
                          "outside": outside.astype(int),
                          "out_pop": np.where(outside, pop, 0.0)})
    by_dong = (frame.groupby(["sigungu", "dong"], as_index=False)
               .agg(cells=("outside", "size"), out_cells=("outside", "sum"),
                    out_pop=("out_pop", "sum")))
    hit = by_dong[by_dong["out_cells"] > 0].sort_values("out_cells", ascending=False)
    print(f"  컷라인 밖 격자를 가진 행정동 {len(hit)}개 "
          f"(전체 {cells['dong'].nunique()}개). 상위 5:")
    for _, r in hit.head(5).iterrows():
        print(f"    {r['sigungu']} {r['dong']}: {int(r['out_cells'])}/{int(r['cells'])}개 셀")
    return by_dong


def print_supply_table(base, peak_nocap, peak_cap, peak_variants=None):
    print("\n[3] 공급 제약 — 피크에 컷라인이 얼마나 당겨지는가 (표 10.9)")
    print("-" * 68)
    print(f"  거점당 라이더 평시 {RIDERS_PER_HUB}명(시간당 {HUB_CAPACITY_MIN:,.0f}분) → "
          f"피크 {RIDERS_PER_HUB * PEAK_RIDER_MULT:.0f}명"
          f"(주문은 {ORDER_RATE_PEAK / ORDER_RATE:.1f}배)")
    print(f"  {'시나리오':<26}{'평균 도달':>10}{'평균 이용률':>11}{'배차 대기':>10}"
          f"{'유효 임률':>10}{'미배차':>8}{'컷라인 밖 인구':>15}")
    for r in (base, peak_nocap, peak_cap):
        rho = f"{r['rho_weighted']:.2f}" if r["rho_weighted"] > 0 else "—"
        wait = f"{r['wait_weighted']:.1f}분" if r["wait_weighted"] > 0 else "—"
        print(f"  {r['label']:<26}{r['avg_time']:>9.2f}분{rho:>11}{wait:>10}"
              f"{r['wage_served']:>10,.0f}{r['unserved_share']:>8.1%}"
              f"{r['outside_pop_share']:>14.1%}")
    delta = (peak_cap["outside_pop_share"] - base["outside_pop_share"]) * 100
    gap = (peak_cap["outside_pop_share"] - peak_nocap["outside_pop_share"]) * 100
    print(f"\n  평시 → 피크(용량·할증 반영): 컷라인 밖 인구 {delta:+.1f}%p "
          f"({base['outside_pop_share']:.1%} → {peak_cap['outside_pop_share']:.1%})")
    print(f"  그중 수요 이동·혼잡만의 몫은 {peak_nocap['outside_pop_share']:.1%}이고, "
          f"용량 제약·할증이 더한 몫이 {gap:+.1f}%p다.")
    print("  ※ 대기는 서비스 품질 지표이고, 원가에 들어오는 경로는 유효 임률의 할증이다.")

    if peak_variants:
        print("\n  라이더 공급을 ±40% 흔들면(피크 기준):")
        print(f"    {'라이더 배수':<12}{'평균 이용률':>11}{'배차 대기':>10}"
              f"{'유효 임률':>10}{'미배차':>8}{'컷라인 밖 인구':>15}")
        for mult, r in sorted(peak_variants.items()):
            print(f"    {mult:<12.1f}{r['rho_weighted']:>11.2f}{r['wait_weighted']:>9.1f}분"
                  f"{r['wage_served']:>10,.0f}{r['unserved_share']:>8.1%}"
                  f"{r['outside_pop_share']:>14.1%}")


def print_overlap_table(res, cells, infra_min):
    print("\n[4] 중첩 — 컷라인 밖과 생활 인프라 열위가 겹치는가 (표 10.11)")
    print("-" * 68)
    outside = res["outside"].astype(bool)
    q75 = float(np.percentile(infra_min, 75))
    weak = infra_min >= q75                      # 상대 정의(도달시간 상위 25%)
    weak15 = infra_min > 15.0
    print(f"  생활 인프라(보건의료·체육 POI) 도보 도달시간: 중앙 {np.median(infra_min):.1f}분, "
          f"상위25% 기준 {q75:.1f}분, 15분 초과 격자 {int(weak15.sum())}개")
    print("  ※ 민간 시설이므로 공공 생활SOC를 대신하지 못한다. 기술통계로만 읽는다.")

    inter = int((outside & weak).sum())
    union = int((outside | weak).sum())
    jac = inter / union if union else float("nan")
    print(f"  컷라인 밖 {int(outside.sum())}개 ∩ 인프라 열위 {int(weak.sum())}개 = {inter}개")
    print(f"  Jaccard(교집합/합집합) {jac:.3f} | "
          f"컷라인 밖 중 열위 비율 {inter / max(outside.sum(), 1):.1%} | "
          f"열위 중 컷라인 밖 비율 {inter / max(weak.sum(), 1):.1%}")
    return {"jaccard": jac, "inter": inter, "weak": weak}


def null_controls(cells, cand_idx, res, weak, n_perm=200):
    """대조군 둘. 심은 메커니즘을 하나씩 제거해 결론이 버티는지 본다."""
    print("\n[4-2] 대조군 — 결론이 배분 규칙과 지리적 우연의 산물인가")
    print("-" * 68)
    # A) 균등 배분: 행정동 인구를 POI 밀도 대신 셀 수로 나눈 자료로 다시 계산
    alt = run_scenario(cells, cand_idx, "pop_offpeak_uniform", SPEED_OFFPEAK,
                       P_HUBS, None, label="대조군A 균등배분")
    weak_a = weak
    inter_a = int((alt["outside"].astype(bool) & weak_a).sum())
    union_a = int((alt["outside"].astype(bool) | weak_a).sum())
    print(f"  대조군A(POI 밀도 가중 제거): 컷라인 밖 인구 "
          f"{alt['outside_pop_share']:.1%} (관측 {res['outside_pop_share']:.1%}), "
          f"Jaccard {inter_a / union_a if union_a else float('nan'):.3f}")

    # B) 수요 공간 무작위화: 위치와 수요의 결합만 끊는다(격자 기하는 그대로)
    outside_obs = res["outside"].astype(bool)
    obs_inter = int((outside_obs & weak).sum())
    obs_union = int((outside_obs | weak).sum())
    obs_jac = obs_inter / obs_union if obs_union else np.nan

    cell_xy = cells[["x_m", "y_m"]].to_numpy()
    t = res["t"]
    hubs = res["hubs"]
    pop = cells["pop_offpeak"].to_numpy()
    jacs, out_shares = [], []
    for _ in range(n_perm):
        perm = RNG.permutation(len(pop))
        orders_p = pop[perm] / 1000.0 * ORDER_RATE
        t_cell = t[hubs].min(axis=0)              # 거점 배치는 고정, 수요만 섞는다
        econ = unit_economics(t_cell, orders_p)
        out_p = econ["margin"] < 0
        inter = int((out_p & weak).sum())
        union = int((out_p | weak).sum())
        jacs.append(inter / union if union else np.nan)
        out_shares.append(float(pop[perm][out_p].sum() / pop.sum()))
    jacs = np.array(jacs, dtype=float)
    lo, hi = np.nanpercentile(jacs, [2.5, 97.5])
    pval = float(np.nanmean(jacs >= obs_jac))
    print(f"  대조군B(수요–위치 결합 제거, {n_perm}회 순열): Jaccard 평균 {np.nanmean(jacs):.3f} "
          f"[{lo:.3f}, {hi:.3f}] | 관측 {obs_jac:.3f} | 경험적 p(대조군 ≥ 관측) = {pval:.3f}")
    print(f"     대조군B의 컷라인 밖 인구 비율: 평균 {np.mean(out_shares):.1%} "
          f"(관측 {res['outside_pop_share']:.1%})")
    if pval > 0.05:
        print("  → 관측 중첩이 대조군 분포 안에 있다. 겹침을 '수요 구조의 결과'로 읽지 않는다.")
    else:
        print("  → 관측 중첩이 대조군 분포를 벗어난다. 겹침이 기하만의 산물은 아니다.")
    return {"jac_null_mean": float(np.nanmean(jacs)), "jac_null_lo": float(lo),
            "jac_null_hi": float(hi), "pval": pval,
            "uniform_outside_share": alt["outside_pop_share"]}


def degeneracy_check(cells, cand_idx, res):
    """최적화가 단순 규칙을 되풀이하는지 검사한다(docs/ch14.md 14.6.6의 선례)."""
    print("\n[5] 퇴화 검사 — 최적화가 '음식점 많은 셀 고르기'와 다른 답을 내는가")
    print("-" * 68)
    t, orders, pop = res["t"], res["orders"], res["pop"]
    cand_food = cells["n_food"].to_numpy()[cand_idx]
    cand_orders = orders[cand_idx]

    naive_sets = {
        "규칙1: 음식점 상위 p개": np.argsort(-cand_food)[:P_HUBS],
        "규칙2: 수요 상위 p개": np.argsort(-cand_orders)[:P_HUBS],
    }
    obj_opt = float((orders * t[res["hubs"]].min(axis=0)).sum())
    print(f"  {'방법':<24}{'평균 도달(분)':>13}{'최적 대비':>11}{'공통 거점':>10}"
          f"{'컷라인 밖 인구':>15}")
    tt = t[res["hubs"]].min(axis=0)
    print(f"  {'최적화(p-median)':<24}{np.average(tt, weights=orders):>13.2f}"
          f"{'—':>11}{'—':>10}{res['outside_pop_share']:>14.1%}")
    out = {}
    for name, hubs in naive_sets.items():
        tt = t[hubs].min(axis=0)
        econ = unit_economics(tt, orders)
        outside = econ["margin"] < 0
        share = float(pop[outside].sum() / pop.sum())
        ov = len(set(hubs.tolist()) & set(res["hubs"].tolist()))
        print(f"  {name:<24}{np.average(tt, weights=orders):>13.2f}"
              f"{float((orders * tt).sum()) / obj_opt - 1:>+10.1%}{ov:>10}"
              f"{share:>14.1%}")
        out[name] = {"overlap": ov, "outside_share": share,
                     "obj_gap": float((orders * tt).sum()) / obj_opt - 1}
    worst_gap = min(v["obj_gap"] for v in out.values())
    max_overlap = max(v["overlap"] for v in out.values())
    if worst_gap < 0.05 and max_overlap >= P_HUBS - 2:
        print("  → 판정: 단순 규칙이 최적해를 거의 되풀이한다. 이 문제에서 최적화의 값어치는")
        print("     크지 않다 — 규칙으로 후보를 좁히고 원가 계산에 집중하는 편이 낫다.")
    else:
        print(f"  → 판정: 단순 규칙은 최적해와 다른 답을 낸다(목적함수 최소 격차 {worst_gap:+.1%}, "
              f"공통 거점 최대 {max_overlap}/{P_HUBS}).")
        print("     밀집지에 거점을 몰면 평균 도달시간이 늘고 컷라인 밖 인구가 커진다 →")
        print("     최적화를 돌릴 값어치가 있다.")
    return out


def sensitivity_table(cells, res, infra_min):
    """고정된 거점 배치 위에서 파라미터만 흔든다(최적화 재실행 없음 → 빠르다)."""
    print("\n[6] 민감도 — 결론을 절대 금액이 아니라 범위로 읽는다 (표 10.10)")
    print("-" * 68)
    t_cell, orders, pop = res["t_cell"], res["orders"], res["pop"]
    cell_xy = cells[["x_m", "y_m"]].to_numpy()
    cand_xy = cell_xy[res["hub_cells"]]

    def share(fee=FEE_PER_ORDER, wage=WAGE_PER_MIN, gain=BUNDLE_MAX_GAIN,
              t_override=None):
        tc = t_cell if t_override is None else t_override
        econ = unit_economics(tc, orders, fee=fee, wage=wage, bundle_gain=gain)
        out = econ["margin"] < 0
        return float(pop[out].sum() / pop.sum()), int(out.sum())

    t_manh = travel_time(cand_xy, cell_xy, SPEED_OFFPEAK, "manhattan").min(axis=0)
    t_det12 = t_cell / DETOUR * 1.20
    t_det15 = t_cell / DETOUR * 1.50

    rows = [
        ("기준", share()),
        ("임률 −30%", share(wage=WAGE_PER_MIN * 0.7)),
        ("임률 +30%", share(wage=WAGE_PER_MIN * 1.3)),
        ("수수료 −20%", share(fee=FEE_PER_ORDER * 0.8)),
        ("수수료 +20%", share(fee=FEE_PER_ORDER * 1.2)),
        ("묶음 없음(계수 1.0)", share(gain=0.0)),
        ("묶음 강함(최대 2.5)", share(gain=1.5)),
        ("우회계수 1.20", share(t_override=t_det12)),
        ("우회계수 1.50", share(t_override=t_det15)),
        ("거리정의: 맨해튼", share(t_override=t_manh)),
    ]
    print(f"  {'설정':<24}{'컷라인 밖 인구':>15}{'컷라인 밖 격자':>15}")
    for name, (s, n) in rows:
        print(f"  {name:<24}{s:>14.1%}{n:>13}개")
    vals = [s for _, (s, _) in rows]
    print(f"\n  컷라인 밖 인구 비율 범위: {min(vals):.1%} ~ {max(vals):.1%} "
          f"(기준 {rows[0][1][0]:.1%})")
    print("  → 절대 수준은 가정에 흔들린다. 흔들리지 않는 것은 순서다 — "
          "먼 저밀도 격자가 언제나 먼저 잘린다.")
    return rows


def make_figure(cells, res, peak_cap, path: Path):
    cell_xy = cells[["x_m", "y_m"]].to_numpy() / 1000.0
    hubs_xy = cell_xy[res["hub_cells"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    ax = axes[0]
    m = res["econ"]["margin"]
    sc = ax.scatter(cell_xy[:, 0], cell_xy[:, 1], c=m, s=26, cmap="RdYlGn",
                    vmin=-abs(m).max(), vmax=abs(m).max())
    ax.scatter(hubs_xy[:, 0], hubs_xy[:, 1], marker="*", s=240, c="k",
               label=f"거점 {len(hubs_xy)}개")
    fig.colorbar(sc, ax=ax, label="건당 기여이익(원)")
    ax.set_title("(a) 평시 격자별 건당 기여이익과 거점 배치")
    ax.set_xlabel("동서(km)")
    ax.set_ylabel("남북(km)")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.scatter(cell_xy[:, 0], cell_xy[:, 1], s=18, c="#c9c9c9", label="컷라인 안")
    o1 = res["outside"].astype(bool)
    o2 = peak_cap["outside"].astype(bool)
    ax.scatter(cell_xy[o2, 0], cell_xy[o2, 1], s=34, c="#f4a261", label="피크에 컷라인 밖")
    ax.scatter(cell_xy[o1, 0], cell_xy[o1, 1], s=34, c="#c1121f", label="평시에도 컷라인 밖")
    ax.scatter(hubs_xy[:, 0], hubs_xy[:, 1], marker="*", s=200, c="k")
    ax.set_title("(b) 평시 대 피크의 컷라인 밖 격자")
    ax.set_xlabel("동서(km)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("배달 권역 설계: 최적화된 거점 배치와 원가 컷라인 (실데이터 스냅샷)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =====================================================================
# 7. main
# =====================================================================
def main() -> None:
    print("=" * 68)
    print("배달 권역 설계와 원가 컷라인 (10-4, 비즈니스 · 최적화)")
    print("=" * 68)
    cells, infra_xy, meta = load_snapshot()
    print(f"자료: 서울 {' · '.join(meta.get('대상_자치구', []))} | "
          f"활동 격자 {len(cells)}개(셀 {meta.get('격자_m', 500):.0f}m) | "
          f"행정동 {cells['dong_code'].nunique()}개")
    print(f"  원본: {meta.get('원본_상가정보', '?')} / {meta.get('원본_생활인구', '?')}")
    print(f"  스냅샷 생성일 {meta.get('생성일', '?')}, sha256 "
          f"{str(meta.get('스냅샷_sha256', ''))[:16]}…")
    print(f"  평시 생활인구 {cells['pop_offpeak'].sum():,.0f}명 → "
          f"피크 {cells['pop_peak'].sum():,.0f}명")
    print(f"  주문 환산(가정): 생활인구 1,000명당 시간당 {ORDER_RATE}건 → 평시 "
          f"{cells['pop_offpeak'].sum() / 1000 * ORDER_RATE:,.0f}건/시간")
    # 저녁에 총량이 줄어도 분포는 이동한다 — 도심이 비고 주거지 비중이 오른다.
    share = (cells.groupby("sigungu")[["pop_offpeak", "pop_peak"]].sum()
             / cells[["pop_offpeak", "pop_peak"]].sum())
    print("  자치구별 생활인구 비중(평시 → 피크):")
    for sg, r in share.iterrows():
        print(f"    {sg}: {r['pop_offpeak']:.1%} → {r['pop_peak']:.1%} "
              f"({(r['pop_peak'] - r['pop_offpeak']) * 100:+.1f}%p)")

    # 후보 거점: 음식점 밀집 상위 셀(픽업이 음식점에서 일어난다)
    cand_idx = np.argsort(-cells["n_food"].to_numpy())[:N_CAND]
    cand_alt = np.argsort(-cells["n_poi"].to_numpy())[:N_CAND]   # 해 안정성 점검용

    base = run_scenario(cells, cand_idx, "pop_offpeak", SPEED_OFFPEAK, P_HUBS,
                        None, label="평시 · 용량 무제약")
    mclp = solve_mclp(base["t"], base["orders"], P_HUBS, T_COVER)
    cap_off = run_scenario(cells, cand_idx, "pop_offpeak", SPEED_OFFPEAK, P_HUBS,
                           HUB_CAPACITY_MIN, label="평시 · 용량제약")
    heur = solve_pmedian_heuristic(base["t"], base["orders"], P_HUBS)

    # 해 안정성: p를 바꾸고, 후보집합을 바꿔 본다
    p_sets = {}
    for p in (8, 10, 12):
        p_sets[p] = set(solve_pmedian(base["t"], base["orders"], p)["hubs"].tolist())
    p_overlap = len(p_sets[8] & p_sets[10] & p_sets[12])
    alt_run = run_scenario(cells, cand_alt, "pop_offpeak", SPEED_OFFPEAK, P_HUBS,
                           None, label="후보집합 교체")
    cand_overlap = len(set(base["hub_cells"].tolist())
                       & set(alt_run["hub_cells"].tolist()))
    alt = {"p_list": (8, 10, 12), "p_overlap": p_overlap, "cand_overlap": cand_overlap}

    print_optimization_table(cells, cand_idx, base, mclp["hubs"], cap_off, heur, alt)
    by_dong = print_cutline_table(base, cells)

    peak_nocap = run_scenario(cells, cand_idx, "pop_peak", SPEED_PEAK, P_HUBS,
                              None, label="피크 · 용량 무제약",
                              order_rate=ORDER_RATE_PEAK)
    peak_variants = {
        mult: run_scenario(cells, cand_idx, "pop_peak", SPEED_PEAK, P_HUBS,
                           HUB_CAPACITY_MIN * mult, label=f"피크 · 라이더 ×{mult:.1f}",
                           order_rate=ORDER_RATE_PEAK)
        for mult in (PEAK_RIDER_MULT * 0.6, PEAK_RIDER_MULT, PEAK_RIDER_MULT * 1.4)}
    peak_cap = peak_variants[PEAK_RIDER_MULT]
    peak_cap["label"] = "피크 · 용량제약+할증"
    print_supply_table(base, peak_nocap, peak_cap, peak_variants)

    infra_min = nearest_walk_minutes(cells[["x_m", "y_m"]].to_numpy(), infra_xy)
    ov = print_overlap_table(base, cells, infra_min)
    nulls = null_controls(cells, cand_idx, base, ov["weak"])
    deg = degeneracy_check(cells, cand_idx, base)
    sens = sensitivity_table(cells, base, infra_min)

    # ---------- 산출물 저장 ----------
    out = cells[["cell_id", "sigungu", "dong", "x_m", "y_m", "n_poi", "n_food",
                 "n_infra", "pop_offpeak", "pop_peak"]].copy()
    out["orders_per_hour"] = base["orders"]
    out["hub_minutes"] = base["t_cell"]
    out["bundle"] = base["econ"]["bundle"]
    out["cost_per_order"] = base["econ"]["cost"]
    out["margin_per_order"] = base["econ"]["margin"]
    out["cell_margin_per_hour"] = base["econ"]["cell_margin"]
    out["outside_offpeak"] = base["outside"]
    out["outside_peak"] = peak_cap["outside"]
    out["infra_walk_min"] = infra_min
    out["infra_weak"] = ov["weak"]
    out["is_hub"] = out["cell_id"].isin(cells["cell_id"].to_numpy()[base["hub_cells"]])
    cells_csv = RESULTS_DIR / "delivery_zone_cells.csv"
    out.round(4).to_csv(cells_csv, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {"scenario": r["label"], "avg_minutes": r["avg_time"],
         "cover_share_T": r["cover_share"], "outside_cells": r["outside_cells"],
         "outside_pop_share": r["outside_pop_share"],
         "unserved_share": r["unserved_share"],
         "max_utilization": float(r["rho"].max()) if len(r["rho"]) else 0.0,
         "mean_wait_min": float(r["wait_hub"].mean()) if len(r["wait_hub"]) else 0.0}
        for r in (base, cap_off, peak_nocap, peak_cap)])
    summary_csv = RESULTS_DIR / "delivery_zone_summary.csv"
    summary.round(4).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    fig_path = RESULTS_DIR / "10-4-delivery-zone.png"
    make_figure(cells, base, peak_cap, fig_path)

    print("\n[7] 산출물")
    print("-" * 68)
    print(f"  {cells_csv.name} — 격자별 원가·기여이익·컷라인 판정")
    print(f"  {summary_csv.name} — 시나리오별 요약")
    print(f"  {fig_path.name} — 거점 배치와 컷라인 지도(그림 10.2)")

    print("\n[증명 범위]")
    print("-" * 68)
    print("  증명하는 것: 목적함수(평균 최소화 대 임계 커버 최대화)가 거점 배치를 바꾸는가,")
    print("    거리·주문밀도·용량 제약이 건당 기여이익과 컷라인을 어떻게 움직이는가,")
    print("    그 결론이 원가 가정의 범위에서 얼마나 버티는가.")
    print("  증명하지 못하는 것: 특정 사업자의 실제 손익, 실제 주문 분포와 그 시간 프로파일,")
    print("    실제 도로망 주행시간(우회계수 근사로 대신했다), 공공 생활SOC와의 중첩")
    print("    (여기서 쓴 인프라 지표는 민간 보건의료·체육 POI다).")
    print("=" * 68)
    print("[완료] 배달 권역 최적화와 원가 컷라인 분석을 마쳤다.")


if __name__ == "__main__":
    main()
