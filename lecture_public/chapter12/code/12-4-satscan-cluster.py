"""
12장 실습 4: 공간 스캔 통계(SaTScan)로 감염병 발생 군집 탐지
=================================================================
감염병 정책 질문: "발생이 비정상적으로 몰린 군집이 어디인가? 그 군집은 우연인가?"

배경(8.5절 연결): 8장은 공간 스캔 통계(SaTScan; Kulldorff, 1997)를 군집을 사전 위치
지정 없이 탐지하는 표준으로 소개했다. 이 실습은 그 방법을 감염병 격자 자료에 '실증'한다.
12장 12-2(SIRS 시차예측)가 "다음 시점 위험"을 예측했다면, 여기서는 "지금 어디가
비정상 군집인가"를 검정한다(예측이 아니라 이상 탐지).

방법(원형 스캔 + 포아송 우도비 + 몬테카를로 순열):
  1. 각 단위를 중심으로 반경을 넓혀 가며 원형 스캔 윈도우를 만든다(인구 상한 50%까지).
  2. 각 윈도우에서 관측 c와 기대 e(= 인구비례)를 비교해 포아송 우도비 통계
     LLR = c·ln(c/e) + (C−c)·ln((C−c)/(C−e))  (단, c>e인 고위험 윈도우만; C=총 발생)
     를 계산하고, 최대 LLR을 갖는 윈도우를 '가장 가능성 높은 군집(MLC)'으로 잡는다.
  3. 귀무가설(위험이 공간적으로 균일)에서 총 발생을 인구비례로 재분배한 순열자료
     999개를 만들어 각각의 최대 LLR을 구하고, 관측 최대 LLR의 p값을 몬테카를로로 얻는다.

검증(construct validation): 참 군집(상대위험 RR을 높인 원형 영역)을 심은 합성 격자에서
  SaTScan이 그 군집을 실제로 탐지하는지(중심·구성단위 회복, 유의성)를 확인한다.

데이터: 미리 준비한 교육용 합성 데이터를 불러와 사용(개인 식별 불가, 격자 집계).
  분석·추정은 실제 계산값. 실제 분석에서는 질병관리청 신고자료·인구를 비식별 집계로
  사용한다(개인정보·목적제한).
  (참 군집을 심은 합성 격자의 설계 의도와 생성은 12-0-simdata-prep.py 참고)

실행:
    python 12-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 12-4-satscan-cluster.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# 몬테카를로 순열 검정 전용 난수생성기(데이터 생성과 분리·재현 가능).
RNG = np.random.default_rng(42)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRID_ROWS, GRID_COLS = 20, 15          # 300개 단위(격자)
N = GRID_ROWS * GRID_COLS
N_SIM = 999                            # 몬테카를로 순열 횟수(최소 p값 = 1/(N_SIM+1))
MAX_POP_FRAC = 0.5                     # 스캔 윈도우 인구 상한(Kulldorff 관례)
# 아래 셋은 데이터에 '심어 둔' 참 군집의 설계값으로, 탐지 결과를 이 참값과 대조해 보고할 때만
# 쓴다(생성은 12-0-simdata-prep.py). 좌표·인구·발생 자체는 저장 파일에서 불러온다.
TRUE_RR = 3.0                          # 참 군집의 상대위험(고위험)
CLUSTER_CENTER = (6.0, 4.0)            # 참 군집 중심(row, col)
CLUSTER_RADIUS = 2.6                   # 참 군집 반경(격자 단위)


def load_cases():
    """미리 준비한 SaTScan 격자 데이터를 불러온다.

    참 군집(중심에서 반경 이내 단위에 상대위험 RR을 높인 영역)을 심은 합성 격자다.
    귀무(균일)에서는 발생이 인구에 비례하고, 참 군집 안에서만 '비정상 밀집'이 나타난다.
    SaTScan이 이 군집을 위치·크기 지정 없이 회복하는지가 검증 목표다.
    반환: coords(N×2 row,col), pop, obs, in_cluster.
    """
    path = DATA_DIR / "satscan_cases.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 12-0-simdata-prep.py")
    df = pd.read_parquet(path).sort_values("unit_id")
    coords = df[["row", "col"]].values.astype(float)
    return coords, df["pop"].values, df["obs"].values, df["in_cluster"].values


def build_windows(coords, pop):
    """각 중심에서 거리순으로 단위를 누적하는 원형 스캔 윈도우를 미리 구성.

    반환: orders(중심별 거리순 인덱스 prefix), e_cumsum(중심별 기대발생 누적).
    기대 e는 인구에만 의존하므로 순열 사이에서 불변 → 한 번만 계산한다.
    """
    P = pop.sum()
    D = cdist(coords, coords)
    orders, e_cumsums = [], []
    # 기대 발생은 전체 총발생이 정해진 뒤 인구비례로 배분되므로 아래 run에서 스케일.
    for ci in range(N):
        order = np.argsort(D[ci])                    # 자기 자신 포함, 가까운 순
        pop_cum = np.cumsum(pop[order])
        # 인구 상한 50%까지의 prefix만 유효 윈도우로 사용
        L = int(np.searchsorted(pop_cum, MAX_POP_FRAC * P, side="right"))
        L = max(L, 1)
        order = order[:L]
        orders.append(order)
        e_cumsums.append(np.cumsum(pop[order]))      # 인구 누적(총발생 곱은 run에서)
    return orders, e_cumsums, P


def poisson_llr(c, e, C):
    """포아송 우도비 로그통계(고위험 군집만 양수). c,e는 배열, C는 총발생 스칼라."""
    out = np.zeros_like(c, dtype=float)
    mask = (c > e) & (c > 0) & (c < C) & (e > 0) & (e < C)
    cc, ee = c[mask], e[mask]
    out[mask] = cc * np.log(cc / ee) + (C - cc) * np.log((C - cc) / (C - ee))
    return out


def scan_max(obs, orders, pop_cumsums, P, C):
    """모든 원형 윈도우에서 최대 LLR과 그 군집(중심·구성단위)을 탐지."""
    best_llr, best_center, best_len = 0.0, -1, 0
    for ci in range(N):
        order = orders[ci]
        c_cum = np.cumsum(obs[order])                # 관측 발생 누적
        e_cum = pop_cumsums[ci] * (C / P)            # 기대 발생 누적(인구비례)
        llr = poisson_llr(c_cum, e_cum, C)
        j = int(np.argmax(llr))
        if llr[j] > best_llr:
            best_llr, best_center, best_len = llr[j], ci, j + 1
    return best_llr, best_center, best_len


def main():
    print("=" * 64)
    print("공간 스캔 통계(SaTScan) 감염병 군집 탐지 (12-4)")
    print("=" * 64)
    coords, pop, obs, in_cluster = load_cases()
    C = float(obs.sum())
    print(f"\n분석 단위: {N}개 격자(20×15), 총발생 C={int(C)}건, 총인구 P={int(pop.sum())}명")
    print(f"참 군집(심은 신호): 중심 {CLUSTER_CENTER}, 반경 {CLUSTER_RADIUS}, "
          f"단위 {int(in_cluster.sum())}개, 상대위험 RR={TRUE_RR}")

    orders, pop_cumsums, P = build_windows(coords, pop)

    # -------- 관측자료에서 가장 가능성 높은 군집(MLC) 탐지 --------
    obs_llr, ci, L = scan_max(obs, orders, pop_cumsums, P, C)
    members = orders[ci][:L]
    c_in = float(obs[members].sum())
    e_in = float(pop[members].sum() * (C / P))
    rr_in = (c_in / e_in) if e_in > 0 else np.nan
    print("\n[군집 탐지] 관측자료의 가장 가능성 높은 군집(MLC)")
    print(f"  중심 단위: (row {int(coords[ci,0])}, col {int(coords[ci,1])}) | 구성단위 {L}개")
    print(f"  관측 c={c_in:.0f}, 기대 e={e_in:.1f}, 상대위험 관측 RR={rr_in:.2f}")
    print(f"  로그우도비 LLR={obs_llr:.2f}")

    # -------- 몬테카를로 순열로 p값 --------
    # 귀무: 위험이 공간 균일 → 총발생 C를 인구비례 확률로 재분배(다항분포)
    p_null = pop / pop.sum()
    ge = 1                                           # 관측 자신 포함(+1 보정)
    sim_max = np.empty(N_SIM)
    for s in range(N_SIM):
        obs_sim = RNG.multinomial(int(C), p_null).astype(float)
        m_llr, _, _ = scan_max(obs_sim, orders, pop_cumsums, P, C)
        sim_max[s] = m_llr
        if m_llr >= obs_llr:
            ge += 1
    p_value = ge / (N_SIM + 1)
    print("\n[유의성] 몬테카를로 순열 검정")
    print(f"  순열 최대 LLR: 중앙값 {np.median(sim_max):.2f}, 95%분위 {np.quantile(sim_max,0.95):.2f}")
    print(f"  관측 LLR {obs_llr:.2f} ≥ 순열 최대 LLR 인 경우 {ge-1}/{N_SIM}")
    print(f"  p값 = {p_value:.4f}  ({'유의(군집 존재)' if p_value < 0.05 else '유의하지 않음'})")

    # -------- construct validation: 참 군집 회복 정도 --------
    detected = np.zeros(N, dtype=bool)
    detected[members] = True
    tp = int((detected & in_cluster).sum())
    recall = tp / max(int(in_cluster.sum()), 1)      # 참 군집 중 탐지 비율
    precision = tp / max(L, 1)                        # 탐지 군집 중 참 비율
    center_hit = bool(in_cluster[ci])
    print("\n[검증] 참 군집 회복(construct validation)")
    print(f"  탐지 군집이 참 군집을 덮은 비율(recall) = {recall:.2f} ({tp}/{int(in_cluster.sum())})")
    print(f"  탐지 군집 중 참 군집 비율(precision) = {precision:.2f} ({tp}/{L})")
    print(f"  MLC 중심이 참 군집 내부인가: {'예' if center_hit else '아니오'}")
    if p_value < 0.05 and recall >= 0.5 and center_hit:
        print("  → SaTScan이 심은 군집을 유의하게 회복했다(방법 검증 성공).")
    else:
        print("  → 회복이 부분적이다. 신호 세기·반경·순열수를 점검한다.")

    # -------- 저장 --------
    out_csv = RESULTS_DIR / "satscan_cluster.csv"
    header = "unit_id,row,col,pop,obs,in_true_cluster,in_detected_cluster"
    rows_out = np.column_stack([
        np.arange(N), coords[:, 0].astype(int), coords[:, 1].astype(int),
        pop.round(0).astype(int), obs.astype(int),
        in_cluster.astype(int), detected.astype(int)])
    np.savetxt(out_csv, rows_out, fmt="%d", delimiter=",", header=header, comments="")
    print(f"\n  군집 탐지 결과 저장 → {out_csv.name}")
    print("  ※ SaTScan은 예측이 아니라 이상 군집의 조기 탐지(감시)다.")
    print("    12-2의 시차예측(다음 주 위험)과 상호보완적으로 쓴다(8.5절·12.4절 연결).")

    print("\n[완료] 공간 스캔 통계 군집 탐지 분석을 마쳤다.")


if __name__ == "__main__":
    main()
