"""
15장 실습 0: 시장 단위 패널 데이터 준비 — 지역 실험·시장 SCM 공용 DGP
================================================================================
왜 합성 데이터인가
--------------------------------------------------------------------------------
15장의 질문은 "출점 효과가 얼마인가"가 아니라 **"어떤 설계와 어떤 추정량이 참값에
닿는가"**다. 참 효과를 모르면 이 질문에 답할 수 없다. 실데이터로는 도넛을 그었을 때
오염이 실제로 걷혔는지 확인할 길이 없다 — 걷혔는지 알려면 걷어야 할 양을 알아야 한다.

두 번째 이유가 더 무겁다. 국내에 대조할 실증이 없다. 기업의 시장 단위 무작위 실험
공개 사례를 찾지 못했고, 국내 상권 자기잠식의 학술 실증도 확인하지 못했다(리서치
기록: content/research/ch15-business-causal-inference.md §2). 실데이터를 써도
무엇이 맞는지 판정할 기준이 없다.

따라서 이 데이터는 construct validation 용도다. **방법의 성질을 보이고 시장의 사실은
보이지 않는다.**

심은 메커니즘 셋 (각각을 제거한 대조군이 분석 스크립트에서 함께 돌아간다)
--------------------------------------------------------------------------------
  (M1) 참 처치효과      : 처치 시장 매출 로그 +0.049 (약 +5%)
  (M2) 공간 파급        : 처치 시장에 인접한 대조 시장 로그 −0.015 (약 −1.5%)
  (M3) 처치 배정의 내생성: 분석 2에서 진출 시장을 사전 성장률 상위에서 고른다
                          (분석 1은 무작위 배정이므로 M3이 없다)

공간 구조를 왜 격자로 두는가
--------------------------------------------------------------------------------
시장 40개를 8×5 격자에 놓고 상하좌우 인접(rook)으로 이웃을 정의한다. 그리고 **시장의
성질(기저 수준·성장률·요인 적재)을 공간적으로 매끄럽게** 만든다. 즉 가까운 시장끼리
닮는다. 이것은 편의가 아니라 현실의 성질이다(공간 자기상관, 4장).

이 설정이 15.4의 핵심 긴장을 만든다. 사전 궤적이 닮은 시장이 곧 지리적 이웃이므로,
**합성통제가 고르는 최적의 기증 단위가 바로 파급으로 오염된 시장**이 된다. 오염을
피하려고 이웃을 빼면 좋은 합성이 어려워진다. 교환이 설계에서 자동으로 발생한다.

생성물
--------------------------------------------------------------------------------
  data/geoexp_panel.parquet   : (40 시장 × 104 주) 로그 매출 기저 패널 (처치 전 세계)
  data/geoexp_markets.parquet : 시장 속성(격자 좌표·기저 수준·성장률·요인 적재)
  data/geoexp_adjacency.npy   : (40, 40) 인접 행렬 (0/1, 대각 0)
  data/geoexp_truth.json      : 참값과 설계 상수

주의: 이 스크립트는 **처치를 적용하지 않은 기저 패널**만 만든다. 처치효과와 파급은
분석 스크립트가 배정에 따라 얹는다. 그래야 같은 기저 위에서 1,000회 배정을 반복하고
대조군 세계(파급 0, 효과 0)를 만들 수 있다.

실행:
    python 15-0-simdata-prep.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42

# ===========================================================
# 설계 상수
# ===========================================================
GRID_ROWS, GRID_COLS = 4, 10      # 4×10 = 40개 시장(자치구 규모를 상정)
N_MARKETS = GRID_ROWS * GRID_COLS
# 군집 무작위화용 블록 — 2×2로 나누면 정확히 10개 군집이 된다. 격자 크기를 4×10으로
# 잡은 이유가 이것이다(5×8은 2×2로 나누어지지 않는다).
BLOCK_ROWS, BLOCK_COLS = 2, 2
N_WEEKS_PRE = 52                  # 사전 52주
N_WEEKS_POST = 52                 # 사후 52주
N_WEEKS = N_WEEKS_PRE + N_WEEKS_POST

TRUE_EFFECT_LOG = 0.049           # 참 처치효과: 로그 +0.049 ≈ +5.0%
SPILLOVER_LOG = -0.015            # 인접 대조 시장 파급: 로그 −0.015 ≈ −1.5%

N_FACTORS = 3                     # 공통 요인 수(경기·계절 외 지역 공통 충격)
NOISE_SD = 0.035                  # 주간 관측 잡음 표준편차(로그 스케일)


def grid_coords():
    """시장을 격자에 배치하고 (행, 열) 좌표를 돌려준다."""
    rows, cols = np.divmod(np.arange(N_MARKETS), GRID_COLS)
    return rows, cols


def rook_adjacency(rows, cols):
    """상하좌우 인접 행렬. 대각선은 이웃으로 보지 않는다(rook 인접).

    격자를 쓰는 이유는 이웃 관계를 명시적으로 통제할 수 있기 때문이다. 실제
    분석에서는 행정구역 경계의 접합 관계나 임계 거리 안의 시장 쌍으로 만든다.
    """
    A = np.zeros((N_MARKETS, N_MARKETS), dtype=np.int8)
    for i in range(N_MARKETS):
        for j in range(N_MARKETS):
            if i == j:
                continue
            if abs(rows[i] - rows[j]) + abs(cols[i] - cols[j]) == 1:
                A[i, j] = 1
    return A


def spatial_smooth(rng, rows, cols, length_scale=1.6):
    """공간적으로 매끄러운 무작위장을 만든다.

    독립 잡음을 거리 기반 가중으로 평균해 인접 시장끼리 닮게 한다. 결과는 평균 0,
    표준편차 1로 정규화한다. 가우시안 커널의 length_scale이 클수록 넓게 닮는다.
    """
    raw = rng.standard_normal(N_MARKETS)
    d2 = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
    W = np.exp(-d2 / (2 * length_scale ** 2))
    smooth = W @ raw / W.sum(axis=1)
    return (smooth - smooth.mean()) / smooth.std()


def build_panel():
    """기저 패널을 생성한다(처치 없음).

    로그 매출 y_it = 기저수준_i + 성장_i·t + Σ_f 적재_if·요인_ft + 계절_t + 잡음_it

    각 항의 뜻:
      기저수준_i : 시장 규모(큰 상권 vs 작은 상권). 공간적으로 매끄럽다.
      성장_i     : 시장별 추세. 공간적으로 매끄럽다 → 분석 2의 내생 선택이 지리적
                   군집을 고르게 되고, 그 이웃이 곧 좋은 기증 단위가 된다.
      요인_ft    : 전국 공통 충격 3개(경기·물가·날씨 등의 대리). 모든 시장이 서로
                   다른 강도(적재)로 함께 흔들린다 → 짝지음이 정밀도를 높이는 근거.
      계절_t     : 52주 주기의 연간 계절성.
      잡음_it    : 관측 잡음. 이 크기가 검정력을 좌우한다.
    """
    rng = np.random.default_rng(SEED)
    rows, cols = grid_coords()

    # 시장 속성 — 공간적으로 매끄럽게
    level = 10.0 + 0.35 * spatial_smooth(rng, rows, cols)          # 로그 매출 기저
    growth = 0.0010 + 0.0007 * spatial_smooth(rng, rows, cols, 1.3)  # 주당 로그 성장
    loadings = np.column_stack([
        spatial_smooth(rng, rows, cols, 1.5) for _ in range(N_FACTORS)
    ])                                                             # (40, 3)

    t = np.arange(N_WEEKS)
    factors = np.zeros((N_FACTORS, N_WEEKS))
    for f in range(N_FACTORS):
        # 요인은 임의보행에 가깝게(자기상관 있는 공통 충격)
        shock = rng.standard_normal(N_WEEKS) * 0.020
        factors[f] = np.cumsum(shock) - np.cumsum(shock).mean()

    season = 0.045 * np.sin(2 * np.pi * t / 52.0) + 0.020 * np.sin(4 * np.pi * t / 52.0)

    signal = (level[:, None]
              + growth[:, None] * t[None, :]
              + loadings @ factors
              + season[None, :])
    noise = rng.standard_normal((N_MARKETS, N_WEEKS)) * NOISE_SD
    Y = signal + noise

    # 군집 무작위화용 블록 번호 — 2×2 블록. 파급 대부분을 군집 안에 가두려는 설계에 쓴다.
    block_id = (rows // BLOCK_ROWS) * (GRID_COLS // BLOCK_COLS) + (cols // BLOCK_COLS)

    markets = pd.DataFrame({
        "market_id": np.arange(N_MARKETS),
        "grid_row": rows,
        "grid_col": cols,
        "block_id": block_id,
        "level": level,
        "growth": growth,
    })
    for f in range(N_FACTORS):
        markets[f"loading_{f}"] = loadings[:, f]

    return Y, markets, rook_adjacency(rows, cols)


def main():
    Y, markets, A = build_panel()

    # 사전기간 관측 성장률 — 분석 2의 내생적 선택이 쓰는 값(참 성장률이 아니라 관측치)
    t_pre = np.arange(N_WEEKS_PRE)
    t_c = t_pre - t_pre.mean()
    pre_growth_obs = (Y[:, :N_WEEKS_PRE] @ t_c) / (t_c @ t_c)
    markets["pre_growth_observed"] = pre_growth_obs

    panel = pd.DataFrame(Y, columns=[f"w{w:03d}" for w in range(N_WEEKS)])
    panel.insert(0, "market_id", np.arange(N_MARKETS))

    truth = {
        "seed": SEED,
        "n_markets": int(N_MARKETS),
        "grid": [int(GRID_ROWS), int(GRID_COLS)],
        "block": [int(BLOCK_ROWS), int(BLOCK_COLS)],
        "n_blocks": int((GRID_ROWS // BLOCK_ROWS) * (GRID_COLS // BLOCK_COLS)),
        "n_weeks_pre": int(N_WEEKS_PRE),
        "n_weeks_post": int(N_WEEKS_POST),
        "true_effect_log": float(TRUE_EFFECT_LOG),
        "true_effect_pct": float(np.expm1(TRUE_EFFECT_LOG) * 100),
        "spillover_log": float(SPILLOVER_LOG),
        "spillover_pct": float(np.expm1(SPILLOVER_LOG) * 100),
        "noise_sd": float(NOISE_SD),
        "n_factors": int(N_FACTORS),
        "note": ("처치효과·파급은 분석 스크립트가 배정에 따라 얹는다. "
                 "이 패널은 처치 이전의 기저 세계다."),
    }

    panel.to_parquet(DATA_DIR / "geoexp_panel.parquet", index=False)
    markets.to_parquet(DATA_DIR / "geoexp_markets.parquet", index=False)
    np.save(DATA_DIR / "geoexp_adjacency.npy", A)
    (DATA_DIR / "geoexp_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print("15-0 시장 단위 패널 준비 — 지역 실험·시장 SCM 공용 DGP")
    print("=" * 68)
    print(f"시장 = {N_MARKETS}개 ({GRID_ROWS}×{GRID_COLS} 격자, rook 인접)")
    print(f"군집 = {truth['n_blocks']}개 ({BLOCK_ROWS}×{BLOCK_COLS} 블록) — 군집 무작위화 설계용")
    print(f"기간 = {N_WEEKS}주 (사전 {N_WEEKS_PRE} / 사후 {N_WEEKS_POST})")
    print(f"참 처치효과 = 로그 {TRUE_EFFECT_LOG:+.3f} ({truth['true_effect_pct']:+.2f}%)")
    print(f"인접 대조 파급 = 로그 {SPILLOVER_LOG:+.3f} ({truth['spillover_pct']:+.2f}%)")
    print(f"관측 잡음 SD = {NOISE_SD:.3f} (로그 스케일)")
    print()
    deg = A.sum(axis=1)
    print(f"인접 이웃 수: 최소 {deg.min()} / 중앙값 {int(np.median(deg))} / 최대 {deg.max()}")
    print(f"인접 시장 쌍 = {int(A.sum() // 2)}개")
    print()
    # 공간 자기상관 점검 — 성장률이 실제로 공간적으로 매끄러운지
    g = markets["growth"].to_numpy()
    gz = (g - g.mean()) / g.std()
    moran = float((A * np.outer(gz, gz)).sum() / A.sum())
    print(f"성장률의 공간 자기상관(인접 쌍 평균 곱) = {moran:+.3f}")
    print("  → 양수이므로 가까운 시장끼리 성장률이 닮는다. 이 성질 때문에 SCM이")
    print("    고르는 좋은 기증 단위가 곧 파급으로 오염된 이웃이 된다(15.4의 교환).")
    print()
    print(f"사전 관측 성장률 상위 5개 시장: {list(markets.nlargest(5, 'pre_growth_observed')['market_id'])}")
    print()
    print(f"저장: {DATA_DIR / 'geoexp_panel.parquet'}")
    print(f"저장: {DATA_DIR / 'geoexp_markets.parquet'}")
    print(f"저장: {DATA_DIR / 'geoexp_adjacency.npy'}")
    print(f"저장: {DATA_DIR / 'geoexp_truth.json'}")


if __name__ == "__main__":
    main()
