"""
12-5. 미충족 수요와 제약 하 순차 입지 선택 — 서울 동물병원 (비즈니스 분석)
============================================================================
12.3의 공공 접근성 진단과 **같은 도구, 다른 목적**을 다룬다. 공공은 접근성이 나쁜
곳을 '취약지'로 지정하고 예산 배분으로 넘어가지만, 민간은 거기서 멈추지 않는다.
"부족한 곳이 어디인가"를 찾은 뒤에 "그중 몇 곳을, 어떤 순서로 낼 수 있는가"를
계산해야 한다.

이 실습의 축은 **한 곳을 고르는 문제가 아니라 여러 곳을 순서대로 고르는 문제**라는
점이다. 첫 선택이 둘째 선택의 조건을 바꾼다. 두 경로로 그렇게 된다.

  ① 자기잠식 — 이미 낸 지점과 가까우면 새 지점이 끌어오는 손님의 일부가 자사
     지점에서 옮겨 온 것이다. (정의처: 14장 14.3.3. 여기서는 적용만 한다)
  ② 용량 제약 — 지점이 받을 수 있는 수요에 상한이 있으면, 상한을 넘긴 자리의 남은
     수요는 여전히 미충족이다. (정의처: 15장 15.5.5)

그래서 미충족 수요 순위표 상위 K를 그냥 고르는 것과 순차로 고르는 것이 다른 답을
낸다. 이 실습은 그 차이를 숫자로 보이고, **용량 수준에 따라 그 차이가 사라지는 체제**
까지 함께 보고한다.

계산 순서
  [1층]  2SFCA로 현행 접근성 Aᵢ (2SFCA의 정식 설명은 4장 4.2)
  [진단] 미충족 수요 Uᵢ = Dᵢ × max(0, 1 − Aᵢ/A*), A* = 수요가중 평균 접근성
  [선택] 자기잠식·용량 제약 하 탐욕적 순차 선택 (배분은 Huff, 14장 14.3.2)
  [결정] 한계 순증과 컷라인 — 컷라인은 자기 매출이 아니라 순증에 걸린다
  [대조] 인구만 / 미충족수요 순위 / 이격 규칙 / 탐욕 네 전략을 같은 평가자로 채점
  [체제] 용량 수준을 바꾸면 순서가 정보를 잃는 구간이 나타난다
  [검사] 퇴화 검사와 동점 진단
  [민감] 가정값을 흔들어 결론이 버티는지

이 실습은 **예측 모델을 학습하지 않는다.** 12-1·12-2가 이미 공간 시차 + 블록 CV
골격을 쓰고 있어, 그것을 비즈니스에서 되풀이하면 방법이 중복된다. 여기서는 1층
접근성 계산과 최적화, 단위 경제만 쓴다.

데이터: 12-0b-vetcare-data-prep.py가 만든 실데이터(상가정보 동물병원, 서울 행정동
심야 생활인구). 사육률·용량·원가만 가정값이며 민감도로 다룬다.

실행 방법 (프로젝트 루트, 통합 .venv):
    python lecture_practice/chapter12/code/12-0b-vetcare-data-prep.py   # 최초 1회
    python lecture_practice/chapter12/code/12-5-unmet-demand-siting.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 크로스 플랫폼 한글 폰트: 사용 가능한 첫 후보 사용(macOS/Windows/Linux).
# 없으면 그림 라벨의 한글이 □로 보일 뿐 결과·수치에는 영향 없음 → 경고만 억제.
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 검증값·산출값·가정값을 구분해 한 곳에 모은다. 구분하지 않으면 본문에서 가정을
# 사실처럼 인용하게 된다.
# ---------------------------------------------------------------------------
SPEND_PER_ANIMAL = 37_000   # [검증값] 마리당 월 병원비.
                            #   농림축산식품부 2025년 반려동물 양육현황조사(2026-02 발표)
RHO = 0.104                 # [산출값] 인구 1인당 추정 반려동물 수.
                            #   ① 전국 = (반려견 499만 + 반려묘 277만) / 주민등록인구
                            #      5,109만 = 0.152
                            #   ② 서울 보정 = 서울 양육가구 19.5%(2024 서울서베이, 2만 가구)
                            #      ÷ 전국 28.6% = 0.682 → 0.152 × 0.682 = 0.104
                            #   가구당 마리수가 서울과 전국이 같다는 가정이 들어간다.
                            #   아래 [7]에서 이 값의 영향 범위를 수치로 확인한다
CAP_MULT = 1.8              # [가정] 지점 용량 = 서울 현재 평균 담당 마리수 × 이 배수.
                            #   1.8은 '평균보다 상당히 큰 병원'을 뜻한다([1]의 정합성 검산에서
                            #   월 매출 상한으로 환산해 확인한다). 단일 값을 고르는 대신
                            #   [6]에서 1.4·1.0도 함께 돌려 체제를 비교한다
BETA = 2.0                  # [가정] Huff 거리 저항. 추정 자료가 없어 민감도로 대신한다
RADIUS_M = 3_000            # [가정] 2SFCA 권역 반경
MARGIN = 0.60               # [가정] 기여이익률. 15장의 15%는 소매점 값이라 쓰지 않는다
FIXED_COST = 16_000_000     # [가정] 월 고정비(수의사·스태프 인건비 + 임대료)
K = 10                      # 신규 개설 예산(지점 수)
MIN_SPACING_M = 1_000       # 관행적 이격 규칙(대조 팔 N2에만 적용)
DIST_FLOOR_M = 250          # 분석 해상도(후보 격자 250m) 아래 거리는 의미가 없으므로 하한
CUTLINE_SWEEP = (16, 24, 32, 40, 44, 48)   # 월 고정비 스윕(백만원)
CAPACITY_REGIMES = (1.8, 1.4, 1.0)         # 용량 체제
SENS = 0.30                 # 민감도 흔들기 폭(±30%)
TIE_TOL = 1e-6              # 동점 판정 허용오차(마리)


# ===========================================================================
# 자료
# ===========================================================================
def load():
    need = ["vet_clinics.parquet", "vet_demand_dong.parquet", "vet_candidates.parquet"]
    missing = [n for n in need if not (DATA_DIR / n).exists()]
    if missing:
        sys.exit(f"[중단] 데이터 파일이 없다: {missing}\n"
                 "  → python 12-0b-vetcare-data-prep.py 를 먼저 실행한다.")
    return (pd.read_parquet(DATA_DIR / need[0]),
            pd.read_parquet(DATA_DIR / need[1]),
            pd.read_parquet(DATA_DIR / need[2]))


def pairwise_dist(ax, ay, bx, by) -> np.ndarray:
    """투영좌표(m) 두 점집합의 거리 행렬. (len(a), len(b))"""
    return np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])


# ===========================================================================
# 1층: 2SFCA 접근성과 미충족 수요
# ===========================================================================
def two_sfca(D, d_dc, capacity, radius):
    """2단계 부동 집수법(2SFCA). 정식 설명은 4장 4.2.

    1단계: 병원마다 '권역 안 수요 대비 용량 비율' Rⱼ를 구한다. 권역에 사람이 많으면
           같은 용량이라도 1마리에게 돌아가는 몫이 적다 — 혼잡을 반영하는 부분이다.
    2단계: 행정동마다 권역 안 병원들의 Rⱼ를 더한다 = 그 동의 접근성 Aᵢ.
           단위는 '마리당 확보 가능한 월 진료 용량'이다.

    거리는 원 논문처럼 권역(문턱)으로 다룬다. 뒤에 나오는 Huff의 연속 감쇠와 처리가
    다른데, 두 방법이 서로 다른 출처를 따르기 때문이다.
    """
    inside = d_dc <= radius                        # (n_dong, n_clinic)
    served = inside.T @ D                          # 병원별 권역 안 수요
    # 권역에 수요가 없는 병원은 누구에게도 접근성을 주지 못한다(0으로 나누기 방지)
    R = np.where(served > 0, capacity / np.maximum(served, 1e-9), 0.0)
    return inside @ R                              # (n_dong,) 접근성


def unmet_demand(D, A):
    """미충족 수요 Uᵢ = Dᵢ × max(0, 1 − Aᵢ/A*), A* = 수요가중 평균 접근성.

    A*를 임의 분위로 고르지 않고 **수요가중 평균**으로 둔다. 이 값에는 계산으로
    확인되는 성질이 있다 — Σ Dᵢ Aᵢ = (병원 수 × 지점 용량)이므로 A*는 정확히
    '용량 배수'와 같아지고, 따라서 Aᵢ/A*는 용량 가정과 사육률 가정에 **불변**이다.
    확인 못 한 값이 진단을 흔들지 않게 하는 장치다.

    대신 이 U는 **총량 부족이 아니라 분포의 불균형**만 잡는다. "서울에 동물병원이
    모자라다"는 주장은 이 분석에서 하지 않는다.
    """
    a_star = float((D * A).sum() / D.sum())
    return D * np.clip(1.0 - A / max(a_star, 1e-12), 0.0, None), a_star


# ===========================================================================
# 선택: Huff 배분 + 자기잠식 + 용량
# ===========================================================================
def evaluate(sel, D, w_dj, base_B, capacity):
    """선택 집합의 지점별 배분·단독배분·실현 포획과 자기잠식률.

    배분은 Huff 구조(14장 14.3.2)를 그대로 쓴다. 분모에 기존 동물병원 전체와 이미
    선택된 자사 지점이 함께 들어가므로, 경쟁이 별도 변수가 아니라 구조로 들어온다.

    신규 진입 체인이라 기존 병원은 모두 타사다. 그래서 **자기잠식은 선택된 신규
    지점들 사이에서만** 생긴다 — 첫 지점에는 없고 둘째부터 나타난다. 순차 구조가
    여기서 드러난다.
    """
    if len(sel) == 0:
        z = np.zeros(0)
        return dict(alloc=z, solo=z, realized=z, cannibal_rate=0.0)
    W = w_dj[:, sel]                                   # (n_dong, k)
    den = base_B + W.sum(axis=1)
    alloc = (D[:, None] * W / den[:, None]).sum(axis=0)
    solo = np.array([float((D * w_dj[:, s] / (base_B + w_dj[:, s])).sum()) for s in sel])
    realized = np.minimum(alloc, capacity)
    rate = 0.0 if solo.sum() <= 0 else 1.0 - alloc.sum() / solo.sum()
    return dict(alloc=alloc, solo=solo, realized=realized, cannibal_rate=rate)


def marginal_gains(sel, D, w_dj, base_B, capacity):
    """모든 후보에 대해 '하나 더 넣었을 때 실현 포획이 얼마나 늘어나는가'.

    선택된 지점의 배분이 함께 줄어드는 것(자기잠식)과 용량 상한에 걸려 더 늘지 않는
    것(용량 제약)이 이득에서 자동으로 차감된다. 그래서 이미 고른 것이 많아질수록 한
    곳 더 고르는 이득이 줄어든다 — 탐욕 절차가 잘 듣는 구조다.

    반환: (실현 기준 한계이득, 용량 무시 한계이득). 둘째 값은 동점을 가르는 데 쓴다.
    """
    cur = evaluate(sel, D, w_dj, base_B, capacity)
    cur_real, cur_alloc = cur["realized"].sum(), cur["alloc"].sum()

    W = w_dj[:, sel] if len(sel) else None
    base = base_B + (W.sum(axis=1) if len(sel) else 0.0)
    new_den = base[:, None] + w_dj                     # (n_dong, n_cand)
    a_new = (D[:, None] * w_dj / new_den).sum(axis=0)   # 새 지점 배분
    real = np.minimum(a_new, capacity)
    alloc = a_new.copy()
    for c in range(len(sel)):                          # 기존 선택 지점의 새 배분
        a = (D[:, None] * W[:, c][:, None] / new_den).sum(axis=0)
        real = real + np.minimum(a, capacity)
        alloc = alloc + a
    return real - cur_real, alloc - cur_alloc


def greedy(D, w_dj, base_B, capacity, k):
    """제약 하 탐욕적 순차 선택.

    한계 이득이 가장 큰 후보를 차례로 고른다. 용량 상한에 걸리는 자리들끼리는 한계
    이득이 정확히 같아져 **동점**이 생기므로, 동점 수를 기록하고 '용량을 무시했을
    때의 이득'이 큰 쪽(= 상방 여유가 큰 쪽)으로 가른다. 동점 수 자체가 진단이다 —
    동점이 많으면 그 구간의 순서는 정보를 담지 않는다.

    이득이 0 이하면 그 자리에서 멈춘다. K개를 다 못 채우는 것은 오류가 아니라 결과다
    (Kuehn & Hamburger, 1963의 창고 입지 휴리스틱 이후의 관행).
    """
    sel, trace = [], []
    for _ in range(k):
        g, g_unc = marginal_gains(sel, D, w_dj, base_B, capacity)
        g[sel] = -np.inf
        best = float(g.max())
        if best <= 0:
            break
        ties = int((g >= best - TIE_TOL).sum())
        j = int(np.argmax(np.where(g >= best - TIE_TOL, g_unc, -np.inf)))
        sel.append(j)
        trace.append(dict(gain=best, gain_uncapped=float(g_unc[j]), ties=ties))
    return sel, trace


def incremental_gains(order, D, w_dj, base_B, capacity):
    """정해진 순서로 하나씩 추가하며 실현 포획 증가분을 기록한다.

    대조 전략(N0·N1·N2)을 탐욕과 같은 기준으로 채점하기 위한 장치다. 순서를 정하는
    방식만 다르고 평가자는 같아야 비교가 성립한다.
    """
    gains, prev = [], 0.0
    for t in range(1, len(order) + 1):
        cur = evaluate(order[:t], D, w_dj, base_B, capacity)["realized"].sum()
        gains.append(cur - prev)
        prev = cur
    return np.array(gains)


def topk_by(score, k, coords=None, min_spacing=None):
    """점수 상위 k개. min_spacing을 주면 이미 고른 것과 그 거리 안이면 건너뛴다."""
    picked = []
    for j in np.argsort(-score):
        if len(picked) >= k:
            break
        if min_spacing is not None and picked:
            dx = coords[0][j] - coords[0][picked]
            dy = coords[1][j] - coords[1][picked]
            if np.min(np.hypot(dx, dy)) < min_spacing:
                continue
        picked.append(int(j))
    return picked


# ===========================================================================
# 돈 — 용어는 15장 15.5.3을 따른다
# ===========================================================================
def breakeven_animals(margin, fixed_cost, spend=SPEND_PER_ANIMAL):
    """손익분기 포획 마리수 = 고정비 ÷ (마리당 월 병원비 × 기여이익률)."""
    return fixed_cost / (spend * margin)


def contribution(animals, margin, spend=SPEND_PER_ANIMAL):
    """월 기여이익(고정비 차감 전)."""
    return animals * spend * margin


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


# ===========================================================================
# 한 번의 완결 실행(체제·민감도에서 재사용)
# ===========================================================================
def run_once(ctx, *, rho=RHO, beta=BETA, radius=RADIUS_M, cap_mult=CAP_MULT, k=K):
    D = ctx["night_pop"] * rho
    avg_load = D.sum() / ctx["n_clinic"]
    capacity = avg_load * cap_mult

    A = two_sfca(D, ctx["d_dc"], capacity, radius)
    U, a_star = unmet_demand(D, A)

    w_dj = np.maximum(ctx["d_dj"], DIST_FLOOR_M) ** (-beta)
    w_dc = np.maximum(ctx["d_dc"], DIST_FLOOR_M) ** (-beta)
    base_B = w_dc.sum(axis=1)

    inside_j = ctx["d_dj"] <= radius
    sel, trace = greedy(D, w_dj, base_B, capacity, k)
    return dict(D=D, A=A, U=U, a_star=a_star, capacity=capacity, avg_load=avg_load,
                w_dj=w_dj, base_B=base_B, sel=sel, trace=trace,
                pot=inside_j.T @ D, unmet_pot=inside_j.T @ U)


def arm_sets(r, cand, k=K):
    """대조 전략 네 팔의 선택 집합."""
    coords = (cand["x"].to_numpy(float), cand["y"].to_numpy(float))
    return {
        "N0 인구 상위K": topk_by(r["pot"], k),
        "N1 미충족수요 상위K": topk_by(r["unmet_pot"], k),
        f"N2 N1+이격{MIN_SPACING_M//1000}km": topk_by(r["unmet_pot"], k, coords=coords,
                                                   min_spacing=MIN_SPACING_M),
        "G  탐욕 순차": r["sel"],
    }


# ===========================================================================
def main() -> None:
    print("=" * 80)
    print("12-5. 미충족 수요와 제약 하 순차 입지 선택 — 서울 동물병원")
    print("=" * 80)

    clinics, dong, cand = load()
    ctx = dict(
        night_pop=dong["night_pop"].to_numpy(float),
        n_clinic=len(clinics),
        d_dc=pairwise_dist(dong["rep_x"].to_numpy(float), dong["rep_y"].to_numpy(float),
                           clinics["x"].to_numpy(float), clinics["y"].to_numpy(float)),
        d_dj=pairwise_dist(dong["rep_x"].to_numpy(float), dong["rep_y"].to_numpy(float),
                           cand["x"].to_numpy(float), cand["y"].to_numpy(float)),
    )

    print("\n[0] 자료와 가정")
    print(f"  기존 동물병원 {len(clinics)}개 | 수요 단위(행정동) {len(dong)}개 "
          f"| 후보지 {len(cand):,}개")
    print(f"  심야 생활인구 합계 {ctx['night_pop'].sum():,.0f}명 (상주 인구의 대리)")
    print(f"  [검증값] 마리당 월 병원비 {SPEND_PER_ANIMAL:,}원 "
          "(농림축산식품부 2025년 반려동물 양육현황조사)")
    print(f"  [산출값] 인구 1인당 반려동물 ρ={RHO} "
          "(전국 0.152 × 서울/전국 양육가구비 0.682)")
    print(f"  [가정] β={BETA} | 2SFCA 반경 {RADIUS_M:,}m | 용량 배수 {CAP_MULT} "
          f"| 기여이익률 {MARGIN:.0%} | 월 고정비 {FIXED_COST/1e6:.0f}백만원")

    r = run_once(ctx)
    D, A, U, capacity = r["D"], r["A"], r["U"], r["capacity"]

    print("\n[1] 1층: 2SFCA 현행 접근성 (4장 4.2)")
    print(f"  추정 반려동물 수요 합계 {D.sum():,.0f}마리")
    print(f"  병원 1곳당 현재 평균 담당 {r['avg_load']:,.0f}마리 "
          f"→ 지점 용량 {capacity:,.0f}마리/월 (×{CAP_MULT})")
    print(f"  접근성 A(마리당 월 진료 용량): 최소 {A.min():.3f} / 중앙 {np.median(A):.3f} "
          f"/ 최대 {A.max():.3f}")
    print(f"  기준 접근성 A* = 수요가중 평균 = {r['a_star']:.4f} "
          f"(= 용량 배수 {CAP_MULT}. 검산이 맞으면 A/A*는 용량·사육률 가정에 불변)")
    print(f"  접근성 비 A/A*: 최소 {(A/r['a_star']).min():.3f} / "
          f"중앙 {np.median(A/r['a_star']):.3f} / 최대 {(A/r['a_star']).max():.3f}")

    # 경제적 정합성 검산(15장 15.5.4의 규율). 가정값을 돈으로 환산해 실무 감각과
    # 어긋나지 않는지 먼저 본다 — 통계 진단이 못 잡는 종류의 오류를 여기서 잡는다
    market_month = D.sum() * SPEND_PER_ANIMAL
    print("  [정합성 검산] 가정을 돈으로 환산하면")
    print(f"    서울 동물병원 시장 규모(추정) 월 {market_month/1e8:,.0f}억원 "
          f"= 연 {market_month*12/1e8:,.0f}억원")
    print(f"    병원 1곳 평균 월 매출(추정) {r['avg_load']*SPEND_PER_ANIMAL/1e4:,.0f}만원")
    print(f"    지점 용량을 매출로 환산한 상한 {capacity*SPEND_PER_ANIMAL/1e4:,.0f}만원/월")
    print("    앞의 두 값이 실무 감각과 크게 어긋나면 사육률·기여이익률 가정을 먼저")
    print("    고쳐야 한다. 뒤의 최적화를 정교하게 해도 이 어긋남은 교정되지 않는다.")

    print("\n[2] 진단: 미충족 수요 — '골짜기'를 찾는 단계")
    n_unmet = int((U > 0).sum())
    print(f"  미충족 수요가 있는 행정동 {n_unmet}개 / {len(dong)}개 ({n_unmet/len(dong):.1%})")
    print(f"  미충족 수요 합계 {U.sum():,.0f}마리 (전체 수요의 {U.sum()/D.sum():.1%})")
    print("  이 U는 총량 부족이 아니라 **분포의 불균형**만 잡는다(A*가 평균이므로).")
    print("  상위 8 행정동 (행정동 | 자치구 | 수요 | 접근성비 | 미충족):")
    for i in np.argsort(-U)[:8]:
        print(f"    {dong['dong'].iloc[i]:<12s} {dong['sigungu'].iloc[i]:<6s} "
              f"{D[i]:9,.0f} {A[i]/r['a_star']:8.3f} {U[i]:9,.0f}")

    print(f"\n[3] 선택: 자기잠식·용량 제약 하 탐욕적 순차 입지 (K={K})")
    sel = r["sel"]
    ev = evaluate(sel, D, r["w_dj"], r["base_B"], capacity)
    gains = np.array([t["gain"] for t in r["trace"]])
    ties = np.array([t["ties"] for t in r["trace"]])
    be = breakeven_animals(MARGIN, FIXED_COST)
    print(f"  손익분기 포획 = 고정비 ÷ (월 병원비 × 기여이익률) = {be:,.0f}마리/월")
    print("  순서 | 후보ID | 자치구 | 단독배분 | 배분 | 실현 | 용량소진 | 한계순증 | "
          "한계기여이익(백만) | 컷라인 | 동점")
    sigs = []
    for rank, j in enumerate(sel, start=1):
        i_near = int(np.argmin(ctx["d_dj"][:, j]))
        sigs.append(dong["sigungu"].iloc[i_near])
        g = gains[rank - 1]
        print(f"  {rank:>4d} | {int(cand['cand_id'].iloc[j]):>6d} | {sigs[-1]:<6s} | "
              f"{ev['solo'][rank-1]:8,.0f} | {ev['alloc'][rank-1]:6,.0f} | "
              f"{ev['realized'][rank-1]:6,.0f} | "
              f"{ev['realized'][rank-1]/capacity:8.1%} | {g:8,.0f} | "
              f"{contribution(g, MARGIN)/1e6:18.1f} | "
              f"{'통과' if g >= be else '탈락':<6s} | {ties[rank-1]:>5d}")
    n_open = int((gains >= be).sum())
    print(f"  총 배분 {ev['alloc'].sum():,.0f}마리 / 실현 {ev['realized'].sum():,.0f}마리")
    print(f"  자기잠식 손실률 {ev['cannibal_rate']:.1%} "
          f"(단독 합 {ev['solo'].sum():,.0f} → 동시 {ev['alloc'].sum():,.0f})")
    print(f"  한계 순증이 손익분기를 넘는 지점 {n_open}개 / {len(sel)}개 "
          f"→ **개설 가능 {n_open}개**")
    print("  컷라인은 그 지점의 자기 매출이 아니라 **자기잠식을 뺀 한계 순증**에 적용한다.")
    print("  자기 매출로 판정하면 이미 낸 지점에서 옮겨 온 몫을 새 이익으로 세게 된다.")
    fc_star = contribution(gains.min(), MARGIN)
    print(f"  역산 — 월 고정비가 {fc_star/1e6:,.1f}백만원을 넘으면 마지막 지점부터 탈락한다")

    print("\n[4] 돈: 컷라인이 언제 구속하는가 (월 고정비 스윕, 탐욕 선택 기준)")
    print("  월 고정비(백만) | 손익분기(마리) | 개설 가능 | 총 한계기여이익(백만) | "
          "총 월 영업이익(백만)")
    for f_m in CUTLINE_SWEEP:
        b = breakeven_animals(MARGIN, f_m * 1e6)
        ok = gains >= b
        contr = contribution(gains[ok], MARGIN).sum()
        print(f"  {f_m:>14d} | {b:>14,.0f} | {int(ok.sum()):>9d} | {contr/1e6:>21.1f} | "
              f"{(contr - ok.sum()*f_m*1e6)/1e6:>19.1f}")

    print("\n[5] 대조: 네 전략을 같은 평가자로 채점")
    arms = arm_sets(r, cand)
    print("  전략 | 지점 | 단독합 | 총배분 | 실현포획 | 자기잠식률 | 개설가능 | "
          "총영업이익(백만) | G와 Jaccard")
    rows = []
    for label, s in arms.items():
        e = evaluate(s, D, r["w_dj"], r["base_B"], capacity)
        inc = incremental_gains(s, D, r["w_dj"], r["base_B"], capacity)
        ok = inc >= be
        op = contribution(inc[ok], MARGIN).sum() - ok.sum() * FIXED_COST
        rows.append(dict(strategy=label, n=len(s), solo=e["solo"].sum(),
                         alloc=e["alloc"].sum(),
                         realized=e["realized"].sum(), cannibal=e["cannibal_rate"],
                         n_open=int(ok.sum()), operating=op,
                         jaccard_vs_g=jaccard(s, sel)))
        print(f"  {label:<22s} {len(s):>4d} {e['solo'].sum():>8,.0f} "
              f"{e['alloc'].sum():>8,.0f} "
              f"{e['realized'].sum():>9,.0f} {e['cannibal_rate']:>10.1%} "
              f"{int(ok.sum()):>9d} {op/1e6:>17.1f} {jaccard(s, sel):>12.2f}")
    strategy_df = pd.DataFrame(rows)
    g_row = strategy_df[strategy_df["strategy"].str.startswith("G")].iloc[0]
    for tag in ("N0", "N1", "N2"):
        o = strategy_df[strategy_df["strategy"].str.startswith(tag)].iloc[0]
        print(f"  → 탐욕 − {tag}: 실현 {g_row['realized']-o['realized']:+,.0f}마리 "
              f"({(g_row['realized']/o['realized']-1):+.1%}), "
              f"자기잠식 {(g_row['cannibal']-o['cannibal'])*100:+.1f}%p, "
              f"영업이익 {(g_row['operating']-o['operating'])/1e6:+,.1f}백만원/월")

    # 격차의 분해. '자기잠식만의 문제'로 읽으면 원인을 절반만 본 것이 된다
    n1 = strategy_df[strategy_df["strategy"].str.startswith("N1")].iloc[0]
    print("  격차 분해 (탐욕 − 미충족수요 순위, 마리):")
    print(f"    ① 자리 자체의 포획력(단독 합 차이)  {g_row['solo']-n1['solo']:+9,.0f}")
    print(f"    ② 자기잠식 손실 차이               "
          f"{-( (g_row['solo']-g_row['alloc']) - (n1['solo']-n1['alloc']) ):+9,.0f}")
    print(f"    ③ 용량 손실 차이                   "
          f"{-( (g_row['alloc']-g_row['realized']) - (n1['alloc']-n1['realized']) ):+9,.0f}")
    print(f"    합계(실현 차이)                    {g_row['realized']-n1['realized']:+9,.0f}")
    print("  ①이 크다는 것은 **미충족 수요가 큰 동과 포획이 큰 자리가 다르다**는 뜻이다.")
    print("  접근성이 낮은 곳은 대개 배후 수요 자체가 얇아, 진단 지표 상위가 곧 좋은")
    print("  자리는 아니다. 순위표를 결정으로 바꾸면 안 되는 이유가 여기에 하나 더 있다.")

    print("\n[6] 용량 체제: 상한을 낮추면 '어디에'가 '몇 개'로 바뀐다")
    print("  용량배수 | 용량(마리) | 매출상한(만원/월) | G실현 | N1실현 | 차이 | "
          "G상한도달 | 1회차동점 | Jaccard(G,N1)")
    regime_rows = []
    for m in CAPACITY_REGIMES:
        rm = run_once(ctx, cap_mult=m)
        arms_m = arm_sets(rm, cand)
        g_s, n1_s = arms_m["G  탐욕 순차"], arms_m["N1 미충족수요 상위K"]
        eg = evaluate(g_s, rm["D"], rm["w_dj"], rm["base_B"], rm["capacity"])
        en = evaluate(n1_s, rm["D"], rm["w_dj"], rm["base_B"], rm["capacity"])
        capped = int((eg["realized"] >= rm["capacity"] - TIE_TOL).sum())
        t1 = rm["trace"][0]["ties"] if rm["trace"] else 0
        regime_rows.append(dict(
            cap_mult=m, capacity=rm["capacity"],
            revenue_cap_won=rm["capacity"] * SPEND_PER_ANIMAL,
            g_realized=eg["realized"].sum(), n1_realized=en["realized"].sum(),
            diff=eg["realized"].sum() - en["realized"].sum(),
            g_capped=capped, ties_step1=t1, jaccard_g_n1=jaccard(g_s, n1_s)))
        print(f"  {m:>8.1f} | {rm['capacity']:>10,.0f} | "
              f"{rm['capacity']*SPEND_PER_ANIMAL/1e4:>17,.0f} | "
              f"{eg['realized'].sum():>6,.0f} | {en['realized'].sum():>6,.0f} | "
              f"{eg['realized'].sum()-en['realized'].sum():>+6,.0f} | "
              f"{capped:>9d} | {t1:>9,d} | {jaccard(g_s, n1_s):>13.2f}")
    regime_df = pd.DataFrame(regime_rows)
    print("  읽는 법: 용량 배수가 낮아지면 상한 도달 지점이 늘고 1회차 동점 후보가")
    print("  급증한다. 동점이 커진다는 것은 **어느 자리를 먼저 열어도 실현 포획이 같다**는")
    print("  뜻이므로, 그 체제에서는 입지의 우열이 사라지고 '몇 개를 열 것인가'만 남는다.")

    print("\n[7] 퇴화 검사와 동점 진단")
    g1, _ = marginal_gains([], D, r["w_dj"], r["base_B"], capacity)
    print("  (가) 답이 '사람 많은 곳'을 되풀이하는가 — 14장 14.6.6의 검사")
    print(f"      1회차 한계이득 ~ 권역 수요(인구) 상관 {np.corrcoef(g1, r['pot'])[0,1]:+.3f} "
          f"| ~ 권역 미충족수요 상관 {np.corrcoef(g1, r['unmet_pot'])[0,1]:+.3f}")
    print(f"      탐욕 vs 인구 상위K Jaccard {jaccard(sel, arms['N0 인구 상위K']):.2f} "
          f"| vs 미충족수요 상위K Jaccard {jaccard(sel, arms['N1 미충족수요 상위K']):.2f}")
    print("  (나) 순서가 정보를 담고 있는가")
    print(f"      단계별 동점 후보 수 {list(map(int, ties))}")
    print(f"      용량 상한에 걸린 지점 {int((ev['realized'] >= capacity - TIE_TOL).sum())}개"
          f" / {len(sel)}개")
    print(f"      한계 순증 감소 추이 {[f'{g:,.0f}' for g in gains]}")
    print(f"  (다) 선택된 지점의 자치구 {sorted(set(sigs))}")
    # 수요 해상도가 행정동이라, 대표점에 가까운 칸이 그 동의 수요를 크게 가져간다.
    # 선택이 이 해상도 인공물에 끌려간 정도를 수치로 드러낸다(14장 14.6.1의 한계와 같은 뿌리)
    d_near = ctx["d_dj"].min(axis=0)
    print("  (라) 해상도 인공물 점검 — 행정동 대표점까지의 거리(m)")
    print(f"      전체 후보 중앙 {np.median(d_near):,.0f} / 선택 지점 중앙 "
          f"{np.median(d_near[sel]):,.0f} / 선택 지점 최소 {d_near[sel].min():,.0f}")
    print("      선택 지점이 전체보다 뚜렷하게 가깝다면, 그만큼은 '대표점 옆 칸'이라는")
    print("      자료 구조에서 온 이득이다. 좌표를 그대로 후보지로 읽지 말아야 하는 이유다.")

    print("\n[8] 민감도: 가정값을 흔들면 선택이 바뀌는가")
    scenarios = [
        ("기준", {}),
        ("β 1.5", dict(beta=1.5)),
        ("β 2.5", dict(beta=2.5)),
        ("사육률 ρ −30%", dict(rho=RHO * (1 - SENS))),
        ("사육률 ρ +30%", dict(rho=RHO * (1 + SENS))),
        ("용량 배수 1.4", dict(cap_mult=1.4)),
        ("용량 배수 1.0", dict(cap_mult=1.0)),
        ("2SFCA 반경 −30%", dict(radius=RADIUS_M * (1 - SENS))),
        ("2SFCA 반경 +30%", dict(radius=RADIUS_M * (1 + SENS))),
    ]
    print("  시나리오 | 미충족합계 | G실현포획 | 개설가능 | 기준 대비 Jaccard | "
          "G−N1 실현차")
    sens_rows = []
    for label, kw in scenarios:
        rr = run_once(ctx, **kw)
        ee = evaluate(rr["sel"], rr["D"], rr["w_dj"], rr["base_B"], rr["capacity"])
        nn = arm_sets(rr, cand)["N1 미충족수요 상위K"]
        en = evaluate(nn, rr["D"], rr["w_dj"], rr["base_B"], rr["capacity"])
        gg = np.array([t["gain"] for t in rr["trace"]])
        gap = ee["realized"].sum() - en["realized"].sum()
        sens_rows.append(dict(scenario=label, unmet_total=rr["U"].sum(),
                              realized=ee["realized"].sum(),
                              n_open=int((gg >= be).sum()),
                              jaccard_vs_base=jaccard(rr["sel"], sel),
                              gap_vs_n1=gap))
        print(f"  {label:<16s} {rr['U'].sum():>11,.0f} {ee['realized'].sum():>9,.0f} "
              f"{int((gg >= be).sum()):>9d} {jaccard(rr['sel'], sel):>17.2f} "
              f"{gap:>+11,.0f}")
    # 원가 가정은 선택을 바꾸지 않고 컷라인만 옮긴다 — 따로 보고한다
    print("  원가 가정(선택 집합 불변, 컷라인만 이동):")
    for label, mg, fc in [("기여이익률 −30%", MARGIN * (1 - SENS), FIXED_COST),
                          ("기여이익률 +30%", MARGIN * (1 + SENS), FIXED_COST),
                          ("고정비 +30%", MARGIN, FIXED_COST * (1 + SENS)),
                          ("고정비 −30%", MARGIN, FIXED_COST * (1 - SENS))]:
        b2 = breakeven_animals(mg, fc)
        n2 = int((gains >= b2).sum())
        sens_rows.append(dict(scenario=label, unmet_total=U.sum(),
                              realized=ev["realized"].sum(), n_open=n2,
                              jaccard_vs_base=1.0,
                              gap_vs_n1=float(g_row["realized"] - n1["realized"])))
        print(f"    {label:<14s} 손익분기 {b2:>7,.0f}마리 → 개설 가능 {n2}개")
    sens_df = pd.DataFrame(sens_rows)
    print(f"  → 개설 가능 {sens_df['n_open'].min()}~{sens_df['n_open'].max()}개, "
          f"선택 집합 Jaccard {sens_df['jaccard_vs_base'].min():.2f}~1.00")
    print(f"  → G−N1 실현차는 모든 시나리오에서 "
          f"{sens_df['gap_vs_n1'].min():+,.0f}~{sens_df['gap_vs_n1'].max():+,.0f}마리 — "
          "부호가 뒤집히지 않는다면 결론(순차가 순위표보다 낫다)은 가정에 견고하다")
    rj = sens_df.loc[sens_df["scenario"].str.startswith("사육률"), "jaccard_vs_base"]
    print(f"  → 확인 못 한 사육률 ρ: 선택 집합 Jaccard {rj.min():.2f}~{rj.max():.2f} "
          "(선택 불변). ρ는 접근성비와 선택을 바꾸지 않고 금액만 비례로 움직인다")

    # ---------------- 저장 ----------------
    pd.DataFrame({
        "rank": np.arange(1, len(sel) + 1),
        "cand_id": cand["cand_id"].to_numpy()[sel],
        "sigungu": sigs,
        "lon": cand["lon"].to_numpy()[sel],
        "lat": cand["lat"].to_numpy()[sel],
        "solo_animals": ev["solo"],
        "alloc_animals": ev["alloc"],
        "realized_animals": ev["realized"],
        "capacity_util": ev["realized"] / capacity,
        "marginal_animals": gains,
        "marginal_contribution_won": contribution(gains, MARGIN),
        "passes_cutline": gains >= be,
        "ties_at_step": ties,
    }).to_csv(RESULTS_DIR / "vet_site_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "dong_code": dong["dong_code"], "dong": dong["dong"], "sigungu": dong["sigungu"],
        "night_pop": dong["night_pop"], "demand_animals": D,
        "accessibility_2sfca": A, "accessibility_ratio": A / r["a_star"],
        "unmet_animals": U,
    }).sort_values("unmet_animals", ascending=False).to_csv(
        RESULTS_DIR / "vet_unmet_demand.csv", index=False, encoding="utf-8-sig")
    strategy_df.to_csv(RESULTS_DIR / "vet_strategy_comparison.csv", index=False,
                       encoding="utf-8-sig")
    regime_df.to_csv(RESULTS_DIR / "vet_capacity_regimes.csv", index=False,
                     encoding="utf-8-sig")
    sens_df.to_csv(RESULTS_DIR / "vet_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(dong["rep_x"], dong["rep_y"], s=np.sqrt(U) * 2.0 + 4,
                    c=A / r["a_star"], cmap="viridis", alpha=0.85, edgecolor="none")
    ax.scatter(clinics["x"], clinics["y"], s=5, c="0.55", marker="+",
               label=f"기존 동물병원 {len(clinics)}개")
    n1 = arms["N1 미충족수요 상위K"]
    ax.scatter(cand["x"].to_numpy()[n1], cand["y"].to_numpy()[n1], s=60, marker="s",
               facecolor="none", edgecolor="steelblue", linewidth=1.2,
               label=f"미충족수요 순위 상위 {len(n1)}(대조)")
    ax.scatter(cand["x"].to_numpy()[sel], cand["y"].to_numpy()[sel], s=120,
               facecolor="none", edgecolor="crimson", linewidth=1.8,
               label=f"탐욕 순차 선택 {len(sel)}개(숫자=순서)")
    for rank, j in enumerate(sel, start=1):
        ax.annotate(str(rank), (cand["x"].iloc[j], cand["y"].iloc[j]),
                    fontsize=9, color="crimson", ha="center", va="center")
    fig.colorbar(sc, ax=ax, label="접근성비 A/A* (1.0 = 서울 평균)")
    ax.set_title("서울 동물병원 미충족 수요와 제약 하 순차 입지 선택\n"
                 "(점 크기 = 미충족 수요, 색 = 접근성비)")
    ax.set_xlabel("EPSG:5179 X (m)"); ax.set_ylabel("EPSG:5179 Y (m)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "12-5-vet-siting-map.png", dpi=140)
    plt.close(fig)

    print("\n" + "-" * 80)
    print("저장: vet_site_selection.csv, vet_unmet_demand.csv, "
          "vet_strategy_comparison.csv, vet_capacity_regimes.csv, vet_sensitivity.csv, "
          "12-5-vet-siting-map.png")
    print("=" * 80)
    print("[완료] 미충족 수요 진단 → 제약 하 순차 선택 → 컷라인 결정")


if __name__ == "__main__":
    main()
