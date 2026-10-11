"""
출점 효과 예제(8-3)의 합성 데이터 생성기 — 세 세계를 같은 지리로 만든다
======================================================================
이 파일은 실습 코드가 아니라 **데이터 생성 모듈**이다(파일명이 `_`로 시작하므로
하네스 `scripts/run_and_capture.py`의 실행 대상에서 제외된다). 두 곳에서 쓴다.

- `8-0-simdata-prep.py` — 학습자가 분석할 패널 세 개를 `data/`에 저장한다.
- `8-3-store-opening-donut-did.py` — 1종 오류율 점검을 위해 같은 DGP를
  수백 번 재실행한다(재실행이 없으면 "추론이 깨졌는가"를 물을 수 없다).

왜 합성인가
-----------
참 처치효과를 알아야 추정량을 채점할 수 있다. 카드 매출·POS는 상업 데이터라
공개 대체재가 없고, 있더라도 "정답"이 없어 도넛 설계가 오염을 실제로 걷어냈는지
검증할 길이 없다. 즉 이 데이터는 construct validation(설계가 의도한 것을 정말
재는지 확인하는 절차)용 실험실이며, 국내 상권의 실증이 아니다.

매출 귀속 규칙(중요)
--------------------
격자별 브랜드 매출은 **판매 매장의 상권에 안분 귀속**된다. 고객 거주지 기준으로
집계하면 손님이 기존점에서 신규점으로 옮겨가도 그 격자의 브랜드 지출 총액은
그대로이므로 **자기잠식이 격자 수준에서 아예 보이지 않는다.** 자기잠식을 격자
단위로 관측하려면 매출을 판매 매장 쪽에 붙여야 한다. 이 예제는 그렇게 만든다.

심은 메커니즘 세 개(각각을 하나씩 끈 대조군 세계를 만든다)
----------------------------------------------------------
1. 내생적 배치 — 신규점은 잠재 성장력이 높은 격자에 우선 배치된다.
2. 시변 교란   — 출점하지 않았어도 잠재 성장력이 높은 격자가 사후에 더 성장한다.
3. 자기잠식    — 신규점 300~800m 링의 매출이 신규점으로 이동해 감소한다.

세계 A = 1+2+3 (관측 세계) / 세계 B = 1+2 (잠식 0) / 세계 C = 1만 (효과·잠식 0)
세 세계는 **같은 난수 스트림·같은 매장 배치**를 쓴다. 지리가 같아야 차이를
메커니즘 탓으로 돌릴 수 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ===========================================================
# 설계 파라미터 — 본문·README가 인용하는 값의 단일 출처
# ===========================================================
GRID = 40                      # 40 × 40 격자
CELL_M = 100.0                 # 격자 한 변 100m → 분석 권역 4km × 4km 도심 상권
N_CELLS = GRID * GRID          # 1,600
MONTHS = 24                    # 월 24기
POST_START = 12                # 출점 시점(사전 12기 · 사후 12기)

N_EXISTING = 12                # 브랜드 A 기존 매장
N_NEW = 3                      # 신규 출점
EXIST_MIN_SEP = 400.0          # 기존점끼리 최소 이격
NEW_MIN_SEP = 1200.0           # 신규점끼리 최소 이격(반경이 겹치지 않게)
NEW_NEAR_LO = 300.0            # 신규점은 기존점에서 이 범위 안에 낸다
NEW_NEAR_HI = 600.0            # (프랜차이즈는 물류·관리 반경 안에서 확장한다)

CATCH_M = 500.0                # 매장 상권 반경 = 브랜드 매출이 관측되는 범위
TREAT_R = 300.0                # 신규점 처치 반경
RING_OUT = 500.0               # 자기잠식이 닿는 바깥 경계(처치 반경 바로 밖 한 겹)

CANNIB_CATCH = 900.0           # 기존점 상권과 겹치는 범위 — 잠식은 여기서만 일어난다

TRUE_TAU = 0.40                # 참 총효과(로그 매출), 반경 300m 내 균일
TRUE_KAPPA = 0.09              # 참 자기잠식 최대치(300m 지점)
KAPPA_DECAY = 0.40             # 링 바깥 경계에서 최대치의 (1-0.40)=60%로 감쇠
CONFOUND = 0.30                # 시변 교란 계수: post × 0.30 × latent
TREND = 0.02                   # 월별 공통 추세
NOISE_SD = 0.12                # 월별 관측 잡음
BASE_SD = 0.15                 # 격자 고정 수준의 잔차

EXIST_BETA = 1.5               # 기존점 배치가 잠재 성장력에 쏠리는 정도
NEW_BETA = 1.5                 # 신규점 배치가 잠재 성장력에 쏠리는 정도(내생성의 세기)

# 분석에 쓰는 사전 공변량 — 상권 분석 실무가 실제로 확보하는 변수들이다.
COV_NAMES = ["x_foot", "x_hh", "x_comp", "x_rent", "x_access"]
RING_EDGES = [0.0, 300.0, 500.0, 800.0, 1200.0, np.inf]
RING_LABELS = ["0-300m", "300-500m", "500-800m", "800-1200m", "1200m+"]


def cell_xy() -> tuple[np.ndarray, np.ndarray]:
    """격자 인덱스 → 미터 좌표(x=열, y=행). 좌표계는 평면 직교로 단순화한다."""
    idx = np.arange(N_CELLS)
    return (idx % GRID) * CELL_M, (idx // GRID) * CELL_M


def _smooth_field(rng: np.random.Generator, fx: float, fy: float, sd: float) -> np.ndarray:
    """격자 위의 매끄러운 공간장 → [0, 1]. 가까운 격자가 닮게 만든다(공간 자기상관)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID), indexing="ij")
    f = (np.sin(2 * np.pi * fx * xx) + np.cos(2 * np.pi * fy * yy)
         + sd * rng.normal(0, 1, (GRID, GRID))).reshape(-1)
    return (f - f.min()) / (f.max() - f.min())


def _nearest_dist(xs, ys, picks) -> np.ndarray:
    """각 격자에서 매장 집합까지의 최단거리(미터). 매장이 없으면 무한."""
    if len(picks) == 0:
        return np.full(N_CELLS, np.inf)
    px, py = xs[picks], ys[picks]
    d = np.sqrt((xs[:, None] - px[None, :]) ** 2 + (ys[:, None] - py[None, :]) ** 2)
    return d.min(axis=1)


def _place(rng, latent, xs, ys, k, beta, min_sep, allowed=None) -> list[int]:
    """잠재 성장력이 높은 격자를 우선해 매장 k개를 순차 배치한다.

    beta가 클수록 '좋은 자리'로 쏠린다 = 배치가 내생적이다. min_sep은 매장이
    한 점에 뭉치지 않게 하는 제약이고, allowed는 후보 격자를 제한한다.
    """
    picks: list[int] = []
    for _ in range(k):
        ok = np.ones(N_CELLS, dtype=bool) if allowed is None else allowed.copy()
        if picks:
            ok &= _nearest_dist(xs, ys, picks) >= min_sep
        if not ok.any():
            raise RuntimeError("배치 후보가 소진되었다 — 이격 제약을 완화해야 한다")
        w = np.exp(beta * latent) * ok
        picks.append(int(rng.choice(N_CELLS, p=w / w.sum())))
    return picks


def make_store_panel(seed: int = 42, with_effect: bool = True,
                     with_cannibalization: bool = True) -> dict:
    """출점 패널 하나를 생성한다.

    Parameters
    ----------
    seed : 난수 시드. 같은 시드면 지리(공변량·매장 배치)가 완전히 같다.
    with_effect : False면 참 총효과 τ를 0으로 둔다(효과 0 대조군).
    with_cannibalization : False면 참 자기잠식 κ를 0으로 둔다(잠식 0 대조군).

    Returns
    -------
    dict — panel(장기형 패널), cells(격자 속성·참값), truth(스칼라 참값)
    """
    rng = np.random.default_rng(seed)
    xs, ys = cell_xy()

    # --- 공변량: 셋은 공간적으로 매끄럽고, 둘은 그렇지 않다 ---------------
    x_foot = _smooth_field(rng, 1.3, 1.1, 0.30)      # 유동인구
    x_access = _smooth_field(rng, 2.1, 1.7, 0.30)    # 접근성(역세권·간선도로)
    x_hh = _smooth_field(rng, 0.9, 1.5, 0.35)        # 배후 가구수
    x_comp = rng.uniform(0, 1, N_CELLS)              # 경쟁점 밀도
    raw_rent = 0.6 * x_foot + 0.4 * rng.uniform(0, 1, N_CELLS)
    x_rent = (raw_rent - raw_rent.min()) / (raw_rent.max() - raw_rent.min())  # 임대료 대리

    X = pd.DataFrame({"x_foot": x_foot, "x_hh": x_hh, "x_comp": x_comp,
                      "x_rent": x_rent, "x_access": x_access})

    # --- 잠재 성장력: 관측 공변량의 비선형 함수 ---------------------------
    # 선형 통제로는 완전히 걷히지 않도록 곱·제곱 항을 넣는다. 관측 밖 요인은
    # 넣지 않았으므로 이 세계에서는 '무교란' 가정이 성립한다(현실은 다르다).
    latent = (1.1 * x_foot * x_access + 0.7 * x_foot ** 2
              + 0.5 * x_hh - 0.4 * x_comp)
    latent = (latent - latent.mean()) / latent.std()

    # --- 매장 배치: 기존점 → 신규점(기존점 인접 후보 중에서) ---------------
    exist = _place(rng, latent, xs, ys, N_EXISTING, beta=EXIST_BETA,
                   min_sep=EXIST_MIN_SEP)
    d_exist = _nearest_dist(xs, ys, exist)
    candidate = (d_exist >= NEW_NEAR_LO) & (d_exist <= NEW_NEAR_HI)
    new = _place(rng, latent, xs, ys, N_NEW, beta=NEW_BETA,
                 min_sep=NEW_MIN_SEP, allowed=candidate)

    d_new = _nearest_dist(xs, ys, new)
    d_brand = np.minimum(d_exist, _nearest_dist(xs, ys, new))

    # --- 분석 우주: 브랜드 상권 안에서만 격자 매출이 관측된다 --------------
    in_universe = d_brand <= CATCH_M

    treated = (d_new <= TREAT_R).astype(int)
    in_ring = ((d_new > TREAT_R) & (d_new <= RING_OUT)).astype(int)

    # --- 참값: 총효과는 반경 내 균일, 잠식은 거리 감쇠 ---------------------
    # 잠식은 링 전체가 아니라 **기존점 상권과 겹치는 격자**에서만 일어난다.
    # 빼앗을 매출이 있는 곳에서만 빼앗기기 때문이다. 이 제약이 없으면 잠식이
    # 링 면적 전체에 걸려(링은 처치 원반의 6배 넓이) 순효과가 음수로 뒤집힌다.
    overlap = d_exist <= CANNIB_CATCH
    tau = np.where(treated == 1, TRUE_TAU, 0.0)
    ring_decay = 1.0 - KAPPA_DECAY * np.clip((d_new - TREAT_R) / (RING_OUT - TREAT_R), 0, 1)
    kappa = np.where((in_ring == 1) & overlap, TRUE_KAPPA * ring_decay, 0.0)
    if not with_effect:
        tau = np.zeros_like(tau)
    if not with_cannibalization:
        kappa = np.zeros_like(kappa)

    base = 2.0 + 1.2 * x_foot + 0.6 * x_access + rng.normal(0, BASE_SD, N_CELLS)

    # --- 패널 생성(우주 안 격자만) -----------------------------------------
    uni = np.where(in_universe)[0]
    t = np.arange(MONTHS)
    post = (t >= POST_START).astype(float)
    noise = rng.normal(0, NOISE_SD, (len(uni), MONTHS))

    y = (base[uni][:, None] + TREND * t[None, :]
         + post[None, :] * (CONFOUND * latent[uni])[:, None]
         + post[None, :] * tau[uni][:, None]
         - post[None, :] * kappa[uni][:, None]
         + noise)

    panel = pd.DataFrame({
        "cell": np.repeat(uni, MONTHS),
        "month": np.tile(t, len(uni)),
        "post": np.tile(post, len(uni)).astype(int),
        "log_sales": y.reshape(-1),
    })

    # 경계는 오른쪽 닫힘으로 잡는다 — 처치(d≤300)와 잠식 링(300<d≤500)의 정의와
    # 링 구간이 정확히 일치해야 한다. 왼쪽 닫힘으로 자르면 d=300 격자가 처치이면서
    # 잠식 링 구간에 들어가 프로파일의 두 칸이 섞인다.
    ring_bin = pd.cut(d_new[uni], bins=RING_EDGES, labels=RING_LABELS,
                      right=True, include_lowest=True)
    cells = pd.DataFrame({
        "cell": uni,
        "x_m": xs[uni], "y_m": ys[uni],
        "d_new_m": d_new[uni], "d_exist_m": d_exist[uni], "d_brand_m": d_brand[uni],
        "treated": treated[uni], "in_ring": in_ring[uni],
        "ring_bin": ring_bin.astype(str),
        # 아래 세 열은 채점(construct validation) 전용 참값이다.
        "tau_true": tau[uni], "kappa_true": kappa[uni], "latent": latent[uni],
    })
    for c in COV_NAMES:
        cells[c] = X[c].to_numpy()[uni]

    # --- 스칼라 참값: 격자 가중 평균과 매출지수 기준 순효과 -----------------
    pre_level = np.exp(y[:, :POST_START].mean(axis=1))   # 사전 매출지수
    tmask, rmask = cells["treated"].to_numpy() == 1, cells["in_ring"].to_numpy() == 1
    gain = float(np.sum(pre_level[tmask] * (np.exp(tau[uni][tmask]) - 1.0)))
    loss = float(np.sum(pre_level[rmask] * (np.exp(-kappa[uni][rmask]) - 1.0)))
    zone = tmask | rmask

    truth = {
        "seed": seed,
        "with_effect": with_effect,
        "with_cannibalization": with_cannibalization,
        "n_cells_grid": N_CELLS,
        "n_cells_universe": int(len(uni)),
        "months": MONTHS,
        "panel_rows": int(len(panel)),
        "n_existing": N_EXISTING,
        "n_new": N_NEW,
        "cell_m": CELL_M,
        "treat_radius_m": TREAT_R,
        "ring_out_m": RING_OUT,
        "n_treated_cells": int(tmask.sum()),
        "n_ring_cells": int(rmask.sum()),
        "n_ring_cannibalized": int((rmask & (kappa[uni] > 0)).sum()),
        "n_far_cells": int((~tmask & ~rmask).sum()),
        "true_att_total": float(tau[uni][tmask].mean()) if tmask.any() else 0.0,
        "true_kappa_peak": float(TRUE_KAPPA if with_cannibalization else 0.0),
        "true_kappa_ring_mean": float(kappa[uni][rmask].mean()) if rmask.any() else 0.0,
        "true_gain_index": gain,
        "true_loss_index": loss,
        "true_net_index": gain + loss,
        "zone_pre_sales_index": float(pre_level[zone].sum()),
        "true_net_pct_of_zone": 100.0 * (gain + loss) / float(pre_level[zone].sum()),
        "latent_mean_treated": float(latent[uni][tmask].mean()),
        "latent_mean_ring": float(latent[uni][rmask].mean()),
        "latent_mean_far": float(latent[uni][~tmask & ~rmask].mean()),
        "new_store_cells": [int(i) for i in new],
        "existing_store_cells": [int(i) for i in exist],
    }
    return {"panel": panel, "cells": cells, "truth": truth}
