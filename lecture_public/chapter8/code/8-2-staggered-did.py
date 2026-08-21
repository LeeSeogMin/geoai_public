"""
8장 심화 예제: 시차 도입(staggered adoption) DID — TWFE는 언제 편향되는가
======================================================================
정책 맥락: 국내 정책은 지자체마다 도입 시점이 다른 경우가 흔하다(예: 사업의
단계적 확대, 조례의 순차 시행). 이때 표준 양방향고정효과(TWFE) DID를 그대로
쓰면 처치효과가 **음(−)의 가중**으로 섞여 편향된다(Goodman-Bacon, 2021).

무엇을 보이는가:
  - 처치 시점이 코호트마다 다르고(시차 도입), 효과가 시간에 따라 커지면(동적 효과)
    TWFE는 이미 처치된 단위를 통제군처럼 써서 진짜값에서 벗어난다.
  - Callaway & Sant'Anna(2021)식 group-time ATT는 아직 처치되지 않은/전혀 처치되지
    않은 단위만 통제군으로 삼아 이 편향을 피한다.

3층 모델:
  [1층 GIS]   단위×연도 패널(도입 코호트 라벨 포함)
  [2층/방법]  TWFE vs group-time ATT 추정량 비교(수동 구현, 패키지 의존 최소화)
  [3층 정책]  "국내 시차 도입 정책은 현대 DID로 재추정" 판단 근거

데이터: 미리 준비한 교육용 시뮬레이션 패널을 불러와 사용한다. 처치효과는 결정론적
        으로 심어 진짜 ATT를 알고 추정량을 비교한다(진짜 overall ATT = 0.7875).

실행:
    python 8-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 8-2-staggered-did.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 렌더링(크로스플랫폼)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 데이터 설계 파라미터(준비 스크립트 8-0과 동일 값 — 분석·해석에 사용)
# 코호트: 조기 도입(g=3), 후기 도입(g=5), 미도입(never=99). 각 100단위.
# 기간 t = 0..7 (예: 2016~2023).
N_PER = 100
T = 8
THETA = 0.30  # 처치 후 1기당 효과 증가분(동적 효과) — 두 코호트 공통

# ---------------------------------------------------------------------------
# 1. 데이터 로드: 미리 준비한 시차 도입 + 동적 효과 패널
# ---------------------------------------------------------------------------
panel_path = DATA_DIR / "staggered_panel.parquet"
if not panel_path.exists():
    raise SystemExit(
        f"데이터가 없습니다: {panel_path}\n먼저 실행: python 8-0-simdata-prep.py")
panel = pd.read_parquet(panel_path)

# 진짜 overall ATT = 처치된 관측의 결정론적 효과 평균
true_att = panel.loc[panel["D"] == 1, "tau"].mean()
print(f"[설계] 진짜 overall ATT (처치 관측의 평균 효과) = {true_att:.4f}")
print(f"[설계] 코호트: 조기 g=3({N_PER}), 후기 g=5({N_PER}), 미도입({N_PER}) / 기간 T={T}")

# ---------------------------------------------------------------------------
# 2. 표준 TWFE DID — 시차 도입 + 동적 효과에서 편향된다
# ---------------------------------------------------------------------------
twfe = smf.ols("y ~ D + C(unit) + C(time)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["unit"]}
)
twfe_att = twfe.params["D"]
twfe_se = twfe.bse["D"]
print("\n[TWFE] 양방향고정효과 DID")
print(f"  ATT 추정 = {twfe_att:.4f} (군집 SE {twfe_se:.4f})")
print(f"  진짜값(0.7875)과 오차 = {twfe_att - true_att:+.4f}")

# ---------------------------------------------------------------------------
# 3. Callaway & Sant'Anna(2021)식 group-time ATT (수동 구현)
#    ATT(g,t) = [ΔY(g-1→t | 코호트 g)] − [ΔY(g-1→t | 미도입 통제군)]
#    통제군은 '전혀 처치되지 않은' 단위(g=99)만 사용 → 오염 없는 통제.
# ---------------------------------------------------------------------------
def cohort_mean(df, g, t):
    return df[(df.cohort == g) & (df.time == t)]["y"].mean()

never = 99
gt_records = []
for g in [3, 5]:
    base = g - 1  # 처치 직전 기준기
    for t in range(g, T):  # 처치 후 기간
        d_treat = cohort_mean(panel, g, t) - cohort_mean(panel, g, base)
        d_ctrl = cohort_mean(panel, never, t) - cohort_mean(panel, never, base)
        att_gt = d_treat - d_ctrl
        n_treated = N_PER  # 코호트 g의 단위 수(t 시점 처치 관측 수)
        gt_records.append({"g": g, "t": t, "event": t - g, "att_gt": att_gt, "w": n_treated})

gt = pd.DataFrame(gt_records)
print("\n[CS] group-time ATT(g,t) (미도입 통제군, 기준기 g-1)")
for _, r in gt.iterrows():
    print(f"  g={int(r.g)} t={int(r.t)} (event {int(r.event):+d}): ATT(g,t) = {r.att_gt:.4f}")

# overall ATT: 처치 관측 수로 가중 평균(= 단순 처치효과 요약)
overall_cs = np.average(gt["att_gt"], weights=gt["w"])
print(f"\n[CS] overall ATT (처치 관측 가중) = {overall_cs:.4f}")
print(f"  진짜값(0.7875)과 오차 = {overall_cs - true_att:+.4f}")

# 이벤트-스터디 집계: 경과기간(event time)별 평균 ATT
es = gt.groupby("event").apply(
    lambda d: np.average(d["att_gt"], weights=d["w"]), include_groups=False
)
print("\n[CS] 이벤트-스터디 (경과기간별 동적 효과)")
for e, v in es.items():
    print(f"  event {int(e):+d}: {v:.4f}  (설계 기대 = {THETA*(e+1):.4f})")

# ---------------------------------------------------------------------------
# 4. 요약 표 + 그림
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("요약: 추정량별 overall ATT (진짜값 = 0.7875)")
print("=" * 60)
print(f"  {'TWFE (양방향고정효과)':<28} {twfe_att:.4f}  (오차 {twfe_att-true_att:+.4f})")
print(f"  {'CS group-time ATT':<28} {overall_cs:.4f}  (오차 {overall_cs-true_att:+.4f})")

# 그림 라벨은 영문(크로스플랫폼 한글 폰트 의존 회피)
fig, ax = plt.subplots(figsize=(6.5, 4.0))
events = sorted(es.index)
ax.plot(events, [es[e] for e in events], "o-", label="CS group-time ATT (by event time)")
ax.plot(events, [THETA * (e + 1) for e in events], "k--", alpha=0.6, label="design true effect")
ax.axhline(twfe_att, color="crimson", ls=":", label=f"TWFE single estimate {twfe_att:.3f}")
ax.axhline(true_att, color="green", ls=":", alpha=0.7, label=f"true overall ATT {true_att:.3f}")
ax.set_xlabel("event time (periods since treatment)")
ax.set_ylabel("treatment effect (ATT)")
ax.set_title("Staggered DID: TWFE bias vs CS group-time ATT")
ax.legend(fontsize=8)
fig.tight_layout()
fig_path = DATA_DIR / "fig_8_2_staggered_did.png"
fig.savefig(fig_path, dpi=110)
print(f"\n[그림] {fig_path}")
