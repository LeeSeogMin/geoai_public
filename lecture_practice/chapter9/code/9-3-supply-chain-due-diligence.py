"""
9장 실습 3: 공급망 산림 리스크 실사 — 등급 컷오프를 어디에 둘 것인가
================================================================================
비즈니스 질문: 국내 식품·목재 기업이 해외 조달 구역에서 원료를 산다. 현장 실사는
구역당 비용이 들고 한 해에 몇 건만 가능하다. **어느 구역을 실사할 것인가?**

이 실습은 예측 모델을 만들지 않는다. 만들지 않는 이유가 이 예제의 출발점이다.
리스크 '등급'은 프레임워크가 규칙으로 정의하는 값이므로, 자기가 만든 라벨을
지도학습으로 복원한 뒤 "손실률이 중요하다"고 해석하면 순환논증이다. 물어야 할
것은 등급을 맞히는 법이 아니라 **등급의 문턱을 어디에 둘 것인가**이고, 그 답은
데이터가 아니라 **두 오류의 비용**이 정한다.

비대칭 손실(정의처는 15.5절 — 여기서는 적용만 한다):
  - 놓친 리스크(FN): 실사하지 않은 구역에서 실제 위반 발생 → 부족 쪽 손해 Cu = C_FN
  - 헛발질(FP)     : 문제없는 구역에 실사 파견        → 과잉 쪽 손해 Co = C_FP
  위반확률 p인 구역에서 실사의 기대손실은 (1−p)·C_FP, 미실사의 기대손실은 p·C_FN.
  둘이 같아지는 지점이 컷오프다 →  p* = C_FP / (C_FP + C_FN) = 1 − 임계비
  즉 실사 개시 확률 컷오프는 15.5절 임계비의 여집합이다.

  단순화 하나를 명시한다. 실사가 위반을 실제로 잡아낸 경우(TP)의 비용은 0으로 둔다.
  실사비를 쓰긴 했지만 그 지출이 회피한 손해로 상계된다고 보는 표준 설정이다
  (Elkan, 2001의 2×2 비용행렬). 상계가 완전하지 않으면 C_FP·C_FN을 순비용으로
  다시 정의하면 되고, 컷오프 공식의 형태는 그대로다.

비용 단위: 실사 1건 비용을 1단위로 둔다(C_FP = 1). 따라서 비용비 R = C_FN / C_FP
  하나만 움직이며, 절대 금액을 가정하지 않는다.

3층 모델에서의 위치 — 이 분석은 [3층 의사결정]에 무게가 있다:
  [1층 GIS]  조달 구역 단위 산림손실률 집계(위성 파생 지표).
  [2층 AI ]  이 예제에서는 학습 모델을 쓰지 않는다. 위험함수를 선언하고, 그 선언이
             결론을 얼마나 좌우하는지 민감도로 되갚는다(가정을 발견처럼 보고하지 않기).
  [3층 결정] 비용 비대칭·예산 제약 하의 실사 컷오프와 예산 근거.

실행:
    python 9-0-simdata-prep.py                  # 최초 1회: 데이터 준비
    python 9-3-supply-chain-due-diligence.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

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

# ===========================================================
# 0. 데이터 로드 (준비 스크립트가 저장한 합성 조달 구역 자료)
# ===========================================================
# 아래 위험함수 계수는 9-0-simdata-prep.py의 선언값과 동일하다(같은 위험함수).
B0, B_LOSS, B_GOV = -2.30, 1.10, 1.60
LOSS_LOG_MEDIAN, LOSS_LOG_SD = np.log(0.55), 0.95

COST_RATIO_MAIN = 10.0      # 본 분석의 기준 비용비 R = C_FN / C_FP
BUDGET_MAIN = 30            # 기준 예산: 한 해 실사 가능 건수
NULL_DRAWS = 300            # 대조군에서 위반을 다시 뽑는 횟수
SEED = 42

DATA_PATH = DATA_DIR / "sourcing_districts.parquet"
if not DATA_PATH.exists():
    raise SystemExit(
        f"데이터 파일이 없습니다: {DATA_PATH}\n"
        "먼저 `python 9-0-simdata-prep.py`를 실행해 실습 데이터를 준비하세요."
    )
df = pd.read_parquet(DATA_PATH)
N = len(df)

loss_rate = df["loss_rate_pct"].to_numpy()
governance = df["governance"].to_numpy()
forest_area = df["forest_area_ha"].to_numpy()
p_true = df["p_true"].to_numpy()

print("=" * 70)
print("9-3 공급망 산림 리스크 실사 — 비대칭 손실 하의 등급 컷오프")
print("=" * 70)
print(f"조달 구역 {N}개 · 조달국 {df['country'].nunique()}개(표준위험 가정)")
print(f"비용 단위: 실사 1건 = 1단위(C_FP=1). 기준 비용비 R = C_FN/C_FP = {COST_RATIO_MAIN:.0f}")
print(f"기준 예산 K = {BUDGET_MAIN}건 / 연")


# ===========================================================
# 1. 관측 — 손실률 분포와 오른쪽 꼬리
# ===========================================================
def describe_observation():
    qs = [50, 75, 90, 95, 99]
    vals = np.percentile(loss_rate, qs)
    loss_area = forest_area * loss_rate / 100.0
    order = np.argsort(-loss_area)
    top10 = order[: int(N * 0.10)]
    share = loss_area[top10].sum() / loss_area.sum()

    print("\n=== [1단계 관측] 조달 구역 연간 산림손실률 분포 ===")
    print("  분위    " + "  ".join(f"p{q:<3d}" for q in qs))
    print("  손실률% " + "  ".join(f"{v:<5.2f}" for v in vals))
    print(f"  평균 {loss_rate.mean():.3f}% · 최대 {loss_rate.max():.3f}%")
    print(f"  손실 면적 상위 10% 구역({len(top10)}개)이 전체 손실 면적의 {share:.1%}를 차지")
    print("  → 손실은 소수 구역에 쏠린다. 그래서 '어디를 볼 것인가'가 결정 문제가 된다.")
    return vals, share


loss_quantiles, tail_share = describe_observation()


# ===========================================================
# 2. 위험함수 — 추정하지 않고 선언한다
# ===========================================================
def risk_function(loss, gov, slope_scale=1.0):
    """p(위반 | 손실률, 거버넌스). 계수는 선언값이며 추정하지 않는다.

    slope_scale은 보정 민감도용이다. 분석가가 손실률의 기울기를 잘못 잡았을 때
    (실제 세계는 slope_scale=1.0) 결정이 얼마나 나빠지는지 보기 위해 쓴다.
    """
    z = (np.log(loss) - LOSS_LOG_MEDIAN) / LOSS_LOG_SD
    logit = B0 + slope_scale * B_LOSS * z + B_GOV * (0.5 - gov) * 2.0
    return 1.0 / (1.0 + np.exp(-logit))


p_hat = risk_function(loss_rate, governance)          # 완전 보정 기준선
assert np.allclose(p_hat, p_true), "선언 위험함수가 자료 생성 모형과 일치해야 한다"
p_bar = float(p_true.mean())

print("\n=== [2단계 위험함수] 위반확률 p(위반|손실률, 거버넌스) — 선언값 ===")
print(f"  logit p = {B0:+.2f} {B_LOSS:+.2f}·z(log 손실률) {B_GOV:+.2f}·(0.5−거버넌스)·2")
print(f"  참 위반확률: 평균 {p_bar:.3f} · 범위 [{p_true.min():.3f}, {p_true.max():.3f}]")
print(f"  실현 위반 {int(df['violation'].sum())}건 / {N}구역")
print("  주의: 계수는 추정값이 아니라 가정이다. 이 가정이 결론을 얼마나 좌우하는지는")
print("        6단계 보정 민감도와 대조군에서 되갚는다.")


# ===========================================================
# 3. 의사결정 도구 — 기대 총비용과 컷오프
# ===========================================================
def expected_cost(selected, cost_ratio, p=None):
    """기대 총비용(실사 1건 비용 = 1단위). selected는 bool 배열."""
    p = p_true if p is None else p
    sel = np.asarray(selected, dtype=bool)
    return float((1.0 - p[sel]).sum() + p[~sel].sum() * cost_ratio)


def missed_violations(selected, p=None):
    """놓친 위반의 기대 건수."""
    p = p_true if p is None else p
    return float(p[~np.asarray(selected, dtype=bool)].sum())


def select_top_share(score, share):
    """점수 상위 share 비율을 고른다(제도 점검률을 사내 규칙으로 옮겨 쓴 규칙)."""
    k = int(np.floor(N * share))
    sel = np.zeros(N, dtype=bool)
    sel[np.argsort(-score)[:k]] = True
    return sel


def select_top_k(score, k):
    sel = np.zeros(N, dtype=bool)
    sel[np.argsort(-score)[:k]] = True
    return sel


def prob_cutoff(cost_ratio):
    """비용 최적 확률 컷오프 p* = C_FP/(C_FP+C_FN) = 1 − 임계비."""
    return 1.0 / (1.0 + cost_ratio)


# --- 표 9.11: 비용비별 최적 컷오프 -----------------------------------------
print("\n=== [3단계 비용 곡선] 비용비 R에 따른 최적 컷오프와 기대 총비용 ===")
print("  R=C_FN/C_FP  임계비  확률컷오프p*  실사건수  기대총비용  놓친위반(기대건수)")
cost_rows = []
for R in [1, 3, 5, 10, 20, 50, 100]:
    cut = prob_cutoff(R)
    sel = p_hat > cut
    ec = expected_cost(sel, R)
    mv = missed_violations(sel)
    cost_rows.append((R, R / (1 + R), cut, int(sel.sum()), ec, mv))
    print(f"  {R:>10d}  {R / (1 + R):6.3f}  {cut:11.4f}  {int(sel.sum()):8d}"
          f"  {ec:10.1f}  {mv:16.2f}")
cost_df = pd.DataFrame(
    cost_rows,
    columns=["cost_ratio", "critical_ratio", "prob_cutoff", "n_inspect",
             "expected_cost", "missed_violations_expected"],
)
print("  → C_FN이 커질수록 임계비가 1에 가까워지고 확률 컷오프는 0으로 내려간다.")
print("    같은 데이터·같은 예측인데 실사 대상 수가 바뀐다. 바꾸는 것은 비용비다.")


# ===========================================================
# 4. 예산 제약 — 컷오프 문제가 순위 문제로 바뀐다
# ===========================================================
# 실사 1건의 기대 절감 Δ = p·C_FN − (1−p)·C_FP. Δ가 큰 순으로 예산을 쓴다.
R = COST_RATIO_MAIN
delta_hat = p_hat * R - (1.0 - p_hat)
order = np.argsort(-delta_hat)
delta_true_ordered = (p_true * R - (1.0 - p_true))[order]

no_inspect_cost = expected_cost(np.zeros(N, dtype=bool), R)
budget_curve = no_inspect_cost - np.concatenate([[0.0], np.cumsum(delta_true_ordered)])
k_star = int((delta_hat > 0).sum())          # 예산 무제한일 때의 최적 실사 건수
max_saving = no_inspect_cost - budget_curve.min()
k90 = int(np.argmax(no_inspect_cost - budget_curve >= 0.90 * max_saving))

print(f"\n=== [4단계 예산 제약] R={R:.0f}에서 예산 K에 따른 잔여 기대손실 ===")
print(f"  무실사 기대 총비용        : {no_inspect_cost:.1f}")
print(f"  예산 무제한 최적 실사 건수 : K* = {k_star}건 (= p̂ > {prob_cutoff(R):.4f} 구역 수)")
print(f"  그때의 기대 총비용        : {budget_curve[k_star]:.1f} (절감 {max_saving:.1f})")
print(f"  절감의 90% 달성 건수       : K = {k90}건 → 곡선의 무릎")
for k in [0, 10, 20, BUDGET_MAIN, 60, 90, k_star]:
    got = (no_inspect_cost - budget_curve[k]) / max_saving if max_saving > 0 else 0.0
    print(f"    K={k:>3d} → 기대 총비용 {budget_curve[k]:7.1f} · 잔여 {budget_curve[k]:7.1f}"
          f" · 달성 절감 {got:5.1%}")
print(f"  → 예산이 K={BUDGET_MAIN}으로 묶이면 달성 가능한 절감의"
      f" {(no_inspect_cost - budget_curve[BUDGET_MAIN]) / max_saving:.1%}만 얻는다.")
print("    이 곡선이 '실사 예산을 얼마나 더 써야 하는가'의 근거가 된다.")


# ===========================================================
# 5. 규칙 비교 — 제도 기본선, 비용 최적, 예산 상위 K, 무작위
# ===========================================================
# EUDR의 관할당국 점검률(저위험 1% / 표준위험 3% / 고위험 9%)은 '당국이 사업자를
# 표본 점검하는 비율'이며 기업의 사내 실사율이 아니다. 그러나 실무에서 이 숫자가
# 사내 목표율의 앵커로 옮겨 쓰이기 쉬우므로, 그 전용(轉用)을 규칙으로 올려 비교한다.
rng = np.random.default_rng(SEED)

rules = {}
rules["제도 기본선 3%(표준위험 점검률 전용)"] = select_top_share(loss_rate, 0.03)
rules["제도 기본선 9%(고위험 점검률 전용)"] = select_top_share(loss_rate, 0.09)
rules[f"비용 최적 컷오프(p*>{prob_cutoff(R):.4f})"] = p_hat > prob_cutoff(R)
rules[f"예산 상위 K={BUDGET_MAIN}(위험순위)"] = select_top_k(p_hat, BUDGET_MAIN)
rules[f"동수 손실률 컷오프(K={BUDGET_MAIN})"] = select_top_k(loss_rate, BUDGET_MAIN)
rules[f"무작위 K={BUDGET_MAIN}"] = select_top_k(rng.uniform(0, 1, N), BUDGET_MAIN)
rules["전수 실사"] = np.ones(N, dtype=bool)
rules["무실사"] = np.zeros(N, dtype=bool)

print(f"\n=== [5단계 규칙 비교] R={R:.0f} · 예산 K={BUDGET_MAIN} ===")
print(f"  {'실사 규칙':<38} {'건수':>4} {'기대총비용':>10} {'놓친위반':>8} {'무실사대비':>9}")
rule_rows = []
for name, sel in rules.items():
    ec = expected_cost(sel, R)
    mv = missed_violations(sel)
    saving = no_inspect_cost - ec
    rule_rows.append((name, int(sel.sum()), ec, mv, saving))
    print(f"  {name:<38} {int(sel.sum()):>4d} {ec:>10.1f} {mv:>8.2f} {saving:>+9.1f}")
rule_df = pd.DataFrame(
    rule_rows, columns=["rule", "n_inspect", "expected_cost",
                        "missed_violations_expected", "saving_vs_none"]
)
gap = rule_df.set_index("rule")
inst9 = gap.loc["제도 기본선 9%(고위험 점검률 전용)", "expected_cost"]
opt = gap.loc[f"비용 최적 컷오프(p*>{prob_cutoff(R):.4f})", "expected_cost"]
budget_ec = gap.loc[f"예산 상위 K={BUDGET_MAIN}(위험순위)", "expected_cost"]
loss_ec = gap.loc[f"동수 손실률 컷오프(K={BUDGET_MAIN})", "expected_cost"]
rand_ec = gap.loc[f"무작위 K={BUDGET_MAIN}", "expected_cost"]
print(f"  → 제도 점검률 9%를 사내 규칙으로 쓰면 기대 총비용 {inst9:.1f}, 비용 최적은 {opt:.1f}."
      f" 차이 {inst9 - opt:.1f}단위")
print(f"    같은 예산 {BUDGET_MAIN}건이라도 위험순위 {budget_ec:.1f} < 손실률 단일 컷오프"
      f" {loss_ec:.1f} < 무작위 {rand_ec:.1f}")
print("    등급을 손실률 한 변수로 자르면 같은 건수를 써도 비용이 더 든다(거버넌스를 못 본다).")


# ===========================================================
# 6. 보정 민감도 — 최적 컷오프는 보정이 맞을 때만 최적이다
# ===========================================================
print(f"\n=== [6단계 보정 민감도] 위험함수 손실률 기울기를 ±50% 흔든다 (R={R:.0f}) ===")
print(f"  {'기울기':>8} {'실사건수':>8} {'실제 기대총비용':>14} {'최적 대비 초과':>14}")
sens_rows = []
for scale in [0.5, 1.0, 1.5]:
    p_wrong = risk_function(loss_rate, governance, slope_scale=scale)
    sel = p_wrong > prob_cutoff(R)          # 확률 컷오프는 비용이 정하므로 불변
    ec = expected_cost(sel, R)              # 채점은 참 p_true로 한다
    sens_rows.append((scale, int(sel.sum()), ec, ec - opt))
    print(f"  {scale:>8.2f} {int(sel.sum()):>8d} {ec:>14.1f} {ec - opt:>+14.1f}")
sens_df = pd.DataFrame(sens_rows, columns=["slope_scale", "n_inspect",
                                          "expected_cost", "excess_vs_optimal"])
print("  → 확률 컷오프 p*는 비용비가 정하므로 보정과 무관하게 그대로다.")
print("    움직이는 것은 그 컷오프에 걸리는 구역 집합, 즉 실질 손실률 컷오프다.")

# 비용비를 잘못 잡았을 때 — 보정 오차와 어느 쪽이 더 비싼가
print(f"\n=== [6단계-b 비용비 오지정] 참 비용비는 R={R:.0f}인데 R을 다르게 가정했다면 ===")
print(f"  {'가정한 R':>8} {'실사건수':>8} {'실제 기대총비용':>14} {'최적 대비 초과':>14}")
mis_rows = []
for r_assumed in [1, 3, 5, 10, 20, 50]:
    sel = p_hat > prob_cutoff(r_assumed)
    ec = expected_cost(sel, R)          # 채점은 참 비용비 R로 한다
    mis_rows.append((r_assumed, int(sel.sum()), ec, ec - opt))
    print(f"  {r_assumed:>8d} {int(sel.sum()):>8d} {ec:>14.1f} {ec - opt:>+14.1f}")
mis_df = pd.DataFrame(
    mis_rows,
    columns=["assumed_cost_ratio", "n_inspect",
             "expected_cost_at_true_ratio", "excess_vs_optimal"],
)
print("  → 두 민감도를 견주면 결론이 분명하다. 위험함수의 기울기를 ±50% 틀리는 대가는")
print("    작지만, 비용비를 한 자릿수 틀리는 대가는 크다. 기대비용 곡선이 최적점 근처에서")
print("    평평하기 때문이다 — 컷오프 근처 구역은 실사해도 안 해도 손익이 비슷하다.")
print("    그러므로 실무에서 먼저 확보할 것은 더 정교한 위험모형이 아니라 C_FN/C_FP다.")


# ===========================================================
# 7. 대조군(null control) — 심은 메커니즘을 하나씩 제거한다
# ===========================================================
print("\n=== [7단계 대조군] 심은 메커니즘을 제거한 세계에서 이득이 사라지는가 ===")

# (a) 신호 0 대조군: 손실률과 위반의 연결을 끊는다(위반이 어디서나 같은 확률로 발생).
#     결정 규칙은 그대로 두므로, 위험순위가 무작위보다 나은 이유가 정말 신호인지 검증한다.
#     기대값으로 채점하면 두 규칙이 정의상 같아지므로, 위반을 실제로 뽑아 채점한다.
rng_null = np.random.default_rng(2026)
sel_rank = select_top_k(p_hat, BUDGET_MAIN)

print(f"  (a) 신호 0 대조군 — 위반을 {NULL_DRAWS}회 다시 뽑아 실현 비용으로 채점")
print(f"      {'세계':<16} {'위험순위':>10} {'무작위':>10} {'차이(순위−무작위)':>18}")
null_rows = []
for world, p_world in [("신호 있음", p_true), ("신호 0", np.full(N, p_bar))]:
    viol = rng_null.random((NULL_DRAWS, N)) < p_world
    cost_rank = (~viol[:, sel_rank]).sum(1) + viol[:, ~sel_rank].sum(1) * R
    cost_rand = np.empty(NULL_DRAWS)
    for m in range(NULL_DRAWS):
        pick = rng_null.permutation(N)[:BUDGET_MAIN]
        mask = np.zeros(N, dtype=bool)
        mask[pick] = True
        cost_rand[m] = (~viol[m, mask]).sum() + viol[m, ~mask].sum() * R
    diff = cost_rank - cost_rand
    null_rows.append((world, cost_rank.mean(), cost_rand.mean(), diff.mean(), diff.std(ddof=1)))
    print(f"      {world:<16} {cost_rank.mean():>10.1f} {cost_rand.mean():>10.1f}"
          f" {diff.mean():>+11.1f} (SD {diff.std(ddof=1):.1f})")
null_df = pd.DataFrame(null_rows, columns=["world", "cost_rank", "cost_random",
                                           "diff_mean", "diff_sd"])
d_sig, d_null = null_df.loc[0, "diff_mean"], null_df.loc[1, "diff_mean"]
sd_null = null_df.loc[1, "diff_sd"]
print(f"      통과 조건: 신호 0 세계에서 차이 ≈ 0. 실측 {d_null:+.1f}"
      f" (SD {sd_null:.1f}, |차이|/SD = {abs(d_null) / sd_null:.2f})")
print(f"      신호 있는 세계의 차이는 {d_sig:+.1f} — 위험순위의 이득은 절차가 아니라 신호에서 온다.")

# (b) 대칭 비용 대조군: C_FN = C_FP로 두면 컷오프가 p=0.5로 가야 한다.
cut_sym = prob_cutoff(1.0)
sel_sym = p_hat > cut_sym
print("\n  (b) 대칭 비용 대조군 — C_FN = C_FP (R=1)")
print(f"      확률 컷오프 {cut_sym:.4f} (기대 0.5), 실사 건수 {int(sel_sym.sum())}건"
      f" · R={R:.0f}일 때 {int((p_hat > prob_cutoff(R)).sum())}건")
print(f"      통과 조건: 컷오프가 정확히 0.5로 이동 → {'통과' if abs(cut_sym - 0.5) < 1e-12 else '실패'}")
print("      비대칭이 사라지면 점추정을 그대로 쓰는 것과 같아진다(15.5절의 임계비 0.5).")


# ===========================================================
# 8. 그림 — 위험함수와 예산 곡선
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sc = ax.scatter(loss_rate, p_true, c=governance, cmap="viridis", s=22, alpha=0.85)
ax.axhline(prob_cutoff(R), color="crimson", lw=1.6, ls="--",
           label=f"비용 최적 컷오프 p*={prob_cutoff(R):.3f} (R={R:.0f})")
ax.axhline(prob_cutoff(1.0), color="gray", lw=1.4, ls=":",
           label="대칭 비용 컷오프 p*=0.5 (R=1)")
ax.axvline(np.percentile(loss_rate, 91), color="darkorange", lw=1.4, ls="-.",
           label="제도 기본선 9% 손실률 컷오프")
ax.set_xscale("log")
# log 축의 기본 지수 표기는 mathtext(U+2212)를 쓰므로 한글 폰트에서 경고가 난다 → 평문 표기로 고정
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("연간 산림손실률 (%, log 축)")
ax.set_ylabel("참 위반확률 p")
ax.set_title("(a) 위험함수와 세 가지 컷오프")
ax.legend(fontsize=8, loc="upper left")
plt.colorbar(sc, ax=ax, label="거버넌스 지표")

ax = axes[1]
ks = np.arange(N + 1)
ax.plot(ks, budget_curve, lw=2, color="steelblue")
ax.axvline(k_star, color="crimson", ls="--", lw=1.5, label=f"K*={k_star} (컷오프와 일치)")
ax.axvline(k90, color="seagreen", ls="-.", lw=1.5, label=f"절감 90% 지점 K={k90}")
ax.axvline(BUDGET_MAIN, color="dimgray", ls=":", lw=1.5, label=f"현 예산 K={BUDGET_MAIN}")
ax.set_xlabel("실사 예산 K (건)")
ax.set_ylabel("잔여 기대 총비용 (실사 1건 = 1단위)")
ax.set_title(f"(b) 예산에 따른 잔여 기대손실 (R={R:.0f})")
ax.legend(fontsize=9)

plt.tight_layout()
fig_path = RESULTS_DIR / "9-3-due-diligence-cutoff.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\n[그림] 위험함수·컷오프와 예산 곡선 저장 → {fig_path.name}")

# ===========================================================
# 9. 산출물 저장
# ===========================================================
cost_df.to_csv(RESULTS_DIR / "9-3-cost-ratio-cutoff.csv", index=False)
rule_df.to_csv(RESULTS_DIR / "9-3-rule-comparison.csv", index=False)
sens_df.to_csv(RESULTS_DIR / "9-3-calibration-sensitivity.csv", index=False)
mis_df.to_csv(RESULTS_DIR / "9-3-cost-ratio-misspecification.csv", index=False)
null_df.to_csv(RESULTS_DIR / "9-3-null-control.csv", index=False)
print("[저장] 9-3-cost-ratio-cutoff.csv · 9-3-rule-comparison.csv"
      " · 9-3-calibration-sensitivity.csv · 9-3-cost-ratio-misspecification.csv"
      " · 9-3-null-control.csv")

print("\n" + "=" * 70)
print("요약: 등급은 예측할 것이 아니라 정하는 것이다.")
print(f"  비용비 R이 1→100으로 가면 확률 컷오프는 {prob_cutoff(1.0):.3f}→{prob_cutoff(100):.4f}로,")
print(f"  실사 대상은 {int((p_hat > prob_cutoff(1.0)).sum())}→"
      f"{int((p_hat > prob_cutoff(100)).sum())}구역으로 움직인다. 데이터는 그대로다.")
print(f"  제도 점검률 9%를 사내 규칙으로 전용하면 비용 최적보다 {inst9 - opt:.1f}단위를 더 쓴다.")
print("  반면 위험함수 기울기를 ±50% 틀리는 대가는 그보다 훨씬 작다 — 먼저 확보할 것은")
print("  더 정교한 위험모형이 아니라 비용비 C_FN/C_FP다.")
print(f"  예산이 K={BUDGET_MAIN}으로 묶이면 컷오프 문제는 순위 문제가 되고,")
print(f"  잔여 기대손실 {budget_curve[BUDGET_MAIN]:.1f}이 증액 요청의 근거가 된다.")
print("한계: 위험함수의 계수는 선언값이며, 실제 위반확률·C_FN·실사 단가를 이 예제가")
print("  말하지 않는다. 컷오프의 절대값이 아니라 컷오프를 정하는 절차가 산출물이다.")
print("=" * 70)
