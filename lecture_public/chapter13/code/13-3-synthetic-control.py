"""
13장 실습 3: 지방소멸대응기금 집중 투입의 효과 — 합성통제법(SCM) (인과 척추)
================================================================================
지역균형발전 정책 질문: "한 지자체에 집중 투입된 지방소멸대응기금(거점 조성·정주여건
대규모 투자)이 그 지역의 경제활동을 실제로 끌어올렸는가? 아니면 원래 회복하던 참이었나?"

3층 모델에서의 위치 — 이 분석은 [3층 정책·인과]까지 간다(식별원=합성 반사실):
  [1층 GIS]  단위별(지자체) 야간조명 경제지수 시계열 집계
  [2층 AI ]  없음 — 이 문제의 핵심은 예측이 아니라 '단일 처치의 반사실 구성'이다.
             볼록조합 최적화로 여러 통제 지자체를 가중결합해 합성 대조군을 만든다.
  [3층 정책] 처치 후 (실제 − 합성) 격차 = 집중 투입의 효과. 단, 단일 사례·파급·
             예상 때문에 "전국 기금효과"로 외삽하지 않는다(규칙2·규칙4).

방법 홈: SCM 정식 언급은 8.2·8.4절(준실험 설계). 여기서는 '적용'한다.

교육 의도(construct validation, 10-3 RDD·8-2 staggered DID 선례):
  특정 지자체의 집중 투입 사업을 좌표·연도로 정제·결합하는 것은 대규모 데이터 획득
  과제라, 본 예제는 *참 효과 α를 아는 합성 DGP*로 추정량의 타당성을 검증한다. 검증 대상:
    (a) SCM이 사전기간을 잘 적합하는가 (작은 pre-RMSPE)
    (b) 처치 후 (실제−합성) 격차가 참 효과 αₜ를 복원하는가
    (c) 볼록조합 가중치가 소수 기증 단위에 희소하게 집중되는가 — DGP의 지배적 단위는
        복원하되, 요인 공선성으로 가중치가 비유일이라 나머지는 대체 단위로 분산된다
        (부분 복원; 개별 가중치의 인과 해석 금지; Abadie 2021)
    (d) placebo 추론에서 처치 단위의 RMSPE 비율이 최상위인가 (유사 p값)
  → 합성이지만 '방법이 옳게 작동하는 조건'을 실증한다(요인모형; Abadie et al. 2010).

데이터: 미리 준비한 교육용 시뮬레이션 패널을 불러와 사용(개인 식별 불가, 지자체 단위).
  참 효과 α를 아는 요인모형 DGP로 만든 위조 대조군 자료(합성이 본질)다. 실제 분석에서는
  위성 야간조명(VIIRS)·생활인구·사업체 시계열 + 집중 투입 사업 지정 자료를 사용한다.

실행:
    python 13-0-simdata-prep.py        # 최초 1회: 데이터 준비
    python 13-3-synthetic-control.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

# ===========================================================
# 0. 패널 설계 상수 (요인모형 — Abadie et al. 2010의 SCM 정당화 구조)
# ===========================================================
# 단위 J+1개(처치 1 + 기증 풀 J). 시점 T개, 처치 시점 T0 이후 처치 단위에 참 효과 α.
# 결과 Y = 기저 + λ(단위 적재)·μ(공통요인) + 잡음. 단위: 야간조명 경제지수(기준 100).
# 실제 패널 데이터는 13-0-simdata-prep.py가 이 설계로 생성해 data/에 저장한다.
J = 30                  # 기증 풀(통제 지자체) 수
N_UNITS = J + 1         # 처치 단위 1 + 기증 풀 J
T = 30                  # 총 연도 수
T0 = 20                 # 처치 시점(21년차부터 처치). 사전 20 + 사후 10
TRUE_EFFECT_FINAL = 8.0  # 참 효과: 처치 후 마지막 해 +8 지수(기금 투입 → 경제 상승)
NOISE_SD = 1.3          # 관측 잡음(작아야 사전 적합 양호)


def load_panel():
    """미리 준비한 합성 지자체 패널을 불러온다(요인모형 DGP).

    이 패널은 위조 대조군 시연을 위한 합성 자료가 본질이다: 처치 단위의 기저·적재가
    몇몇 기증 단위의 볼록조합 안에 놓이도록(좋은 합성이 실제로 존재하도록) 설계되고,
    참 효과는 처치 후 선형 증가(αₜ: 0 → TRUE_EFFECT_FINAL)로 심겨 있어 추정량이
    사전기간을 적합하고 효과를 복원하는지 검증할 수 있다.

    반환:
      Y      : (N_UNITS, T) 관측 결과. 행 0 = 처치 단위.
      cov    : (N_UNITS, 2) 시불변 공변량(고령화율·청년인구비 프록시).
      alpha_t: (T,) 처치 단위에 심은 참 효과(사전기간 0).
      w_true : (N_UNITS,) DGP 가중치를 **전체-단위 프레임**(index 0=처치, 1..J=기증)으로
               정렬한 벡터. scm_for_unit이 돌려주는 weights와 같은 프레임이라 바로 비교 가능
               (기증-전용 프레임과의 off-by-one 혼동 방지).
    """
    y_path = DATA_DIR / "scm_Y.parquet"
    u_path = DATA_DIR / "scm_units.parquet"
    a_path = DATA_DIR / "scm_alpha.parquet"
    if not (y_path.exists() and u_path.exists() and a_path.exists()):
        raise SystemExit(
            f"데이터가 없습니다: {y_path}\n먼저 실행: python 13-0-simdata-prep.py")
    Y = pd.read_parquet(y_path).to_numpy()
    units = pd.read_parquet(u_path)
    cov = units[["cov_aging", "cov_youth"]].to_numpy()
    w_true = units["w_true"].to_numpy()
    alpha_t = pd.read_parquet(a_path)["alpha_t"].to_numpy()
    if Y.shape != (N_UNITS, T):
        raise SystemExit(
            f"패널 형태 불일치: {Y.shape} ≠ ({N_UNITS}, {T}). "
            "13-0-simdata-prep.py로 데이터를 다시 준비하세요.")
    return Y, cov, alpha_t, w_true


# ===========================================================
# 1. SCM 추정량 (볼록조합 최적화: Σw=1, w≥0)
# ===========================================================
def fit_synthetic_weights(x_treat, X_donors):
    """예측자 벡터를 볼록조합으로 최적 근사하는 가중치 w를 구한다.

    minimize ‖x_treat − X_donorsᵀ w‖²  s.t.  Σw = 1, w ≥ 0  (SLSQP).
    x_treat : (P,)  처치 단위 예측자(표준화된 사전결과+공변량).
    X_donors: (J, P) 기증 단위 예측자.
    반환: (J,) 볼록조합 가중치.

    교육용 축약: 예측자 가중행렬 V를 항등으로 둔다(표준화 예측자에 동일 가중).
    정식 SCM은 V를 외부 루프에서 최적화(중첩 최적화; 실무는 Synth/pysyncon).
    """
    n = X_donors.shape[0]

    def loss(w):
        r = x_treat - X_donors.T @ w
        return float(r @ r)

    def grad(w):
        r = x_treat - X_donors.T @ w
        return -2.0 * (X_donors @ r)

    w0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
             "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(loss, w0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()  # 수치오차로 어긋난 Σw=1 재정규화


def build_predictors(Y_all, cov_all, t0):
    """예측자 = [사전기간 결과 전체, 공변량]. 단위 간 표준화(V=I 대비 스케일 정규화)."""
    pre = Y_all[:, :t0]                         # (units, t0)
    P = np.hstack([pre, cov_all])               # (units, t0 + n_cov)
    mean, sd = P.mean(0), P.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (P - mean) / sd                      # 표준화 예측자


def rmspe(gap, lo, hi):
    """구간 [lo, hi)의 제곱근평균제곱예측오차."""
    seg = gap[lo:hi]
    return float(np.sqrt(np.mean(seg ** 2)))


def scm_for_unit(Y_all, cov_all, treat_idx, t0):
    """treat_idx를 처치 단위로 두고 나머지를 기증 풀로 SCM 적합.

    반환: dict(weights[전체 units 길이, 자기 자신=0], synth 궤적, gap 궤적,
              pre_rmspe, post_rmspe, ratio).
    """
    donor_mask = np.ones(Y_all.shape[0], dtype=bool)
    donor_mask[treat_idx] = False
    P = build_predictors(Y_all, cov_all, t0)
    w = fit_synthetic_weights(P[treat_idx], P[donor_mask])

    donors_Y = Y_all[donor_mask]
    synth = w @ donors_Y                         # 합성 궤적
    gap = Y_all[treat_idx] - synth
    pre_r = rmspe(gap, 0, t0)
    post_r = rmspe(gap, t0, Y_all.shape[1])
    ratio = post_r / pre_r if pre_r > 0 else np.inf

    # 자기 자신=0을 포함한 전체 길이 가중치 벡터로 되돌림(해석 편의)
    w_full = np.zeros(Y_all.shape[0])
    w_full[donor_mask] = w
    return {"weights": w_full, "synth": synth, "gap": gap,
            "pre_rmspe": pre_r, "post_rmspe": post_r, "ratio": ratio}


# ===========================================================
# 2. 실행: 추정·진단·placebo 추론·시각화
# ===========================================================
def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 68)
    log("13-3 지방소멸대응기금 집중 투입 SCM — 합성 반사실로 정책효과 식별")
    log("=" * 68)
    Y, cov, alpha_t, w_true = load_panel()
    true_post_mean = float(alpha_t[T0:].mean())
    log(f"단위 = {N_UNITS}개 지자체(처치 1 + 기증 풀 {J}),  연도 = {T}년,  처치 시점 = {T0}년차")
    log(f"참 효과 αₜ(설계값): 처치 후 선형 증가 0 → {TRUE_EFFECT_FINAL:.1f} 지수, "
        f"사후 평균 = {true_post_mean:.2f}")
    log("")

    # --- (a) 처치 단위 SCM: 가중치·사전적합 ---
    res = scm_for_unit(Y, cov, treat_idx=0, t0=T0)
    est_post_mean = float(res["gap"][T0:].mean())

    log("[1] 처치 지자체의 합성 대조군 — 볼록조합 가중치 (Σw=1, w≥0)")
    log("-" * 68)
    order = np.argsort(res["weights"])[::-1]
    log(f"{'기증 지자체(단위)':>16} | {'SCM 가중치':>10}")
    shown = 0
    for j in order:
        if res["weights"][j] < 0.005:
            continue
        log(f"{('통제 #' + str(j)):>16} | {res['weights'][j]:>10.3f}")
        shown += 1
    log(f"  (가중치 0.005 미만 {J - shown}개 단위 생략, Σw = {res['weights'].sum():.3f})")
    true_support = np.where(w_true > 0)[0]
    sup_str = "·".join("#" + str(j) for j in true_support)
    w_on_support = float(res["weights"][true_support].sum())
    dom = int(true_support[np.argmax(w_true[true_support])])  # DGP 최대 기여 단위
    log(f"→ 볼록조합 제약이 가중을 소수 단위({shown}개)에 몰아 해석 가능한 합성을 만든다.")
    log(f"  SCM은 DGP의 지배적 기증 단위 #{dom}(참 가중 {w_true[dom]:.2f})에 최대 가중")
    log(f"  {res['weights'][dom]:.3f}을 실어 이를 잘 복원했다. 다만 DGP가 쓴 네 단위({sup_str})에")
    log(f"  실린 SCM 가중 합은 {w_on_support:.3f}로 약 절반이고, 나머지는 요인구조가 비슷한")
    log("  대체 단위(#18·#9·#28 등)로 분산됐다. 즉 SCM 가중치는 유일하게 결정되지 않는다")
    log("  (비유일성; Abadie 2021) — 기증 단위들이 같은 요인구조를 공유하면 여러 조합이")
    log("  동일한 사전 적합을 주기 때문이다. 따라서 개별 가중치를 인과로 해석하지 말고,")
    log("  사전 궤적 재현([2])·효과 복원([3])을 신뢰 근거로 삼는다.")
    log("")

    # --- (b) 사전 적합도 진단 ---
    log("[2] 사전 적합도 진단 (pre-RMSPE) — SCM 신뢰의 전제")
    log("-" * 68)
    log(f"  사전기간(1~{T0}년) RMSPE = {res['pre_rmspe']:.3f} 지수  "
        f"(관측 잡음 SD {NOISE_SD:.1f} 수준)")
    log(f"  사후기간({T0+1}~{T}년) RMSPE = {res['post_rmspe']:.3f} 지수")
    log("→ 사전 RMSPE가 잡음 수준으로 작다 = 합성 대조군이 처치 전 궤적을 잘 재현했다.")
    log("  사전 적합이 나쁘면 처치 후 격차를 효과로 읽을 수 없다(신뢰의 전제).")
    log("")

    # --- (c) 처치효과: 실제 − 합성 격차 ---
    log("[3] 처치효과 추정 — 처치 후 (실제 − 합성) 격차")
    log("-" * 68)
    log(f"{'연차':>6} | {'실제':>8} | {'합성':>8} | {'격차(효과)':>10} | {'참 효과':>8}")
    for tt in range(T0, T):
        log(f"{tt+1:>6} | {Y[0, tt]:>8.2f} | {res['synth'][tt]:>8.2f} | "
            f"{res['gap'][tt]:>10.2f} | {alpha_t[tt]:>8.2f}")
    log(f"  사후 평균 격차(SCM 추정) = {est_post_mean:.2f} 지수  "
        f"(참 사후 평균 {true_post_mean:.2f})")
    log(f"→ SCM 추정 효과가 참 효과를 근사 복원(편의 {est_post_mean - true_post_mean:+.2f}). "
        f"효과는 처치 후 점증한다.")
    log("")

    # --- (d) placebo 추론: 각 통제 단위를 가짜 처치로 ---
    log("[4] Placebo 추론 (Abadie et al. 2010) — 순열 검정으로 유의성")
    log("-" * 68)
    placebo = []  # (unit_idx, gap, pre_rmspe, ratio)
    for j in range(1, N_UNITS):
        rj = scm_for_unit(Y, cov, treat_idx=j, t0=T0)
        placebo.append((j, rj["gap"], rj["pre_rmspe"], rj["ratio"]))

    # RMSPE 비율(post/pre): 사전적합 나쁜 placebo의 거짓 유의를 분모로 보정
    treat_ratio = res["ratio"]
    ratios = np.array([treat_ratio] + [p[3] for p in placebo])
    rank = int((ratios >= treat_ratio).sum())  # 처치 포함 순위(1=최상위)
    p_value = rank / len(ratios)
    log(f"  처치 지자체 RMSPE 비율(post/pre) = {treat_ratio:.2f}")
    pl_ratios = np.array([p[3] for p in placebo])
    log(f"  기증 풀 placebo 비율: 중앙값 {np.median(pl_ratios):.2f}, "
        f"최댓값 {pl_ratios.max():.2f}")
    log(f"  전체 {len(ratios)}개 단위 중 처치 지자체 비율 순위 = {rank}위 "
        f"→ 유사 p값 = {p_value:.3f}")
    if p_value <= 1.0 / len(ratios) + 1e-9:
        log("→ 처치 지자체의 처치후/사전 격차비가 기증 풀에서 가장 극단 → 효과가")
        log("  '우연한 잡음'으로 설명되기 어렵다(단일 사례 순열추론의 최소 p값).")
    else:
        log("→ 처치 지자체 비율이 최상위는 아니다 → 효과의 유의성을 신중히 해석.")
    log("")

    # --- 시각화: (좌) 실제 vs 합성 궤적, (우) placebo 격차 분포 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    years = np.arange(1, T + 1)

    ax1.plot(years, Y[0], color="crimson", lw=2.5, label="처치 지자체(실제)", zorder=4)
    ax1.plot(years, res["synth"], color="navy", lw=2.0, ls="--",
             label="합성 대조군", zorder=3)
    ax1.axvline(T0 + 0.5, color="black", ls=":", lw=1)
    ax1.annotate("집중 투입 시점", xy=(T0 + 0.5, Y[0].min()),
                 xytext=(T0 - 8.5, Y[0].min()), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax1.set_xlabel("연차")
    ax1.set_ylabel("야간조명 경제지수")
    ax1.set_title("실제 대 합성 대조군 궤적")
    ax1.legend(loc="upper left")

    # placebo 격차(회색) + 처치 격차(적색)
    for _, gap_j, pre_j, _ in placebo:
        # 사전적합이 처치의 2배 넘게 나쁜 placebo는 흐리게(비교 왜곡 방지, Abadie 관행)
        alpha = 0.5 if pre_j <= 2 * res["pre_rmspe"] else 0.12
        ax2.plot(years, gap_j, color="gray", lw=0.8, alpha=alpha, zorder=2)
    ax2.plot(years, res["gap"], color="crimson", lw=2.5,
             label="처치 지자체", zorder=4)
    ax2.plot(years, alpha_t, color="seagreen", lw=1.5, ls="--",
             label="참 효과 αₜ", zorder=3)
    ax2.axvline(T0 + 0.5, color="black", ls=":", lw=1)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xlabel("연차")
    ax2.set_ylabel("격차 (실제 − 합성)")
    ax2.set_title("Placebo 격차 분포 (회색=기증 풀)")
    ax2.legend(loc="upper left")

    fig_path = RESULTS_DIR / "13-3-synthetic-control.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    log(f"[그림] 합성 대조 궤적·placebo 분포 저장 → {fig_path.name}")
    log("")

    log("=" * 68)
    log("요약: 단일 처치 단위에도 SCM은 여러 통제 지자체의 볼록조합으로 반사실을")
    log(f"구성해 효과를 식별한다. 사전 적합(pre-RMSPE {res['pre_rmspe']:.2f})이 좋을 때만")
    log(f"신뢰하며, placebo 순열추론(유사 p={p_value:.3f})으로 유의성을 판정한다.")
    log(f"추정 효과 {est_post_mean:.1f} ≈ 참 {true_post_mean:.1f} 지수.")
    log("한계: 단일 사례·파급(spillover)·예상반응·과적합 → 전국 기금효과로 외삽 금지.")
    log("=" * 68)

    # 텍스트 로그도 남김(하네스 stdout 캡처와 별개로 재확인용)
    (RESULTS_DIR / "13-3-scm-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
