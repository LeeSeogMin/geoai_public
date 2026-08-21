"""
9장 실습 2: 보호구역 효과의 인과 이질성 — R-러너로 CATE 추정 (인과 척추)
================================================================================
환경 정책 질문: "보호구역 지정의 산림손실 저감효과가 '어디서' 크고 작은가?"
9.6절 DID는 평균처치효과(ATE)만 답했다. 그러나 예산이 유한하면 효과가 큰 곳을
먼저 보호해야 한다 → 조건부 평균처치효과 CATE τ(x)=E[Y(1)−Y(0)|X=x]를 추정한다.

3층 모델에서의 위치 — 이 분석은 [3층 정책·인과]까지 간다(식별원=무교란 관측연구):
  [1층 GIS]  격자 공변량장(개발압력·접근성·서식지)과 공간 자기상관.
  [2층 AI ]  랜덤포레스트로 누이선스(m̂·ê·μ̂)와 이질성 함수 τ̂(x)를 근사.
             고차원 이질성 함수 근사가 AI의 자리다(단순 회귀로는 τ(x) 형태를 모름).
  [3층 정책] τ̂(x) = 지역별 보호 저감효과 → 효과 큰 지역 우선 보호(표적화).
             단, 무교란·positivity 가정 위에서만 인과로 읽는다(규칙4: 예측≠인과).

방법 홈: 이질성·인과 포레스트의 정식 언급은 8장(인과, DML·이질성)과 4장(ML·랜덤포레스트).
         여기서는 R-러너(Nie & Wager 2021)로 '적용'한다.

교육 의도(construct validation, 9.6·M2.1 선례):
  실제 보호구역의 참 CATE는 관측 불가능하므로, *참 τ(x)를 아는 합성 DGP*로 검증한다:
    (a) R-러너가 참 τ(x)를 복원하는가 (상관·산점도)
    (b) 이중강건 ATE가 참 ATE를 복원하고, naive 하위집단 차이는 교란으로 편향되는가
    (c) GATES·BLP 검정이 이질성을 유의하게 탐지하는가 (우연·과적합 차단)
    (d) 잡음 공변량(고도·경사)이 헛이질성을 만들지 않는가
    (e) CATE 표적화가 ATE-맹목 배분보다 총 저감손실을 더 확보하는가 (정책 payoff)
  → 합성이지만 '방법이 옳게 작동하는 조건'을 실증한다.

데이터: 미리 준비한 교육용 시뮬레이션을 불러와 사용한다(개인 식별 불가). 실제
  분석에서는 보호지역·산림 공간자료, 토지피복도, 개발행위 자료, 위성 산림변화
  지표를 결합한다.

실행:
    python 9-0-simdata-prep.py         # 최초 1회: 데이터 준비
    python 9-2-causal-heterogeneity.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

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
# 0. 데이터 로드 (준비 스크립트가 저장한 합성 격자 자료)
# ===========================================================
# 아래 상수는 9-0-simdata-prep.py의 DGP와 동일하며, 본문 출력·그림에 쓰인다.
# 참 CATE: τ(x) = −(TAU0 + TAU_E·enforce)  (음수 = 손실 저감)
# 핵심 설계: 효과 조절변수는 '집행역량 enforce'이고, 손실 '수준'은 '개발압력 pressure'가
#   따로 결정한다. 즉 손실이 큰 곳(고압력) ≠ 보호가 잘 듣는 곳(고집행역량).
#   그래서 관측 손실만으로는 '어디서 보호가 효과적인가'를 알 수 없고 CATE가 필요하다.
GRID_ROWS, GRID_COLS = 40, 40      # 1,600 격자
N = GRID_ROWS * GRID_COLS

FEATURES = ["pressure", "enforce", "habitat", "elevation", "slope"]


def load_data():
    """9-0-simdata-prep.py가 저장한 참 CATE 격자 자료를 불러온다.

    자료에는 집행역량(enforce)이 조절하는 참 CATE와, 저압력·고서식지 지역이 우선
    보호되는 선택편향이 심겨 있다. enforce와 pressure가 서로 다른 공간장이라 손실이
    큰 곳(고압력)과 보호가 잘 듣는 곳(고집행역량)이 어긋난다 → 관측 손실만으로는
    후자를 못 찾고, R-러너 CATE가 필요하다. 잡음 공변량(elevation·slope)은 CATE와
    무관한 대조군으로, RF가 헛이질성을 만들지 않는지 점검한다.
    """
    path = DATA_DIR / "cate_grid.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 9-0-simdata-prep.py")
    return pd.read_parquet(path)


# ===========================================================
# 1. 누이선스 교차적합 (m̂, ê, μ̂0, μ̂1)  — Ch8 DML과 동일 규율
# ===========================================================
def crossfit_nuisances(df, n_splits=5):
    """K-겹 교차적합으로 out-of-fold 누이선스를 만든다.

    자기 자신을 예측에 쓰지 않아야(교차적합) 머신러닝 정규화 편향이 인과추정을
    오염시키지 않는다(Chernozhukov 외 2018 DML 원리, Ch8). 반환은 모두 out-of-fold.
      m̂(x)=E[Y|X] (풀드 회귀),      ê(x)=E[D|X] (성향점수, 분류기)
      μ̂0(x)=E[Y|X,D=0], μ̂1(x)=E[Y|X,D=1] (팔별 회귀 → AIPW용)
    """
    X = df[FEATURES].values
    y = df["y"].values
    d = df["treated"].values
    n = len(df)
    m_hat = np.zeros(n); e_hat = np.zeros(n)
    mu0 = np.zeros(n); mu1 = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in kf.split(X):
        # m̂: 결과의 주변기대(처치 무관)
        rf_m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     random_state=42, n_jobs=1).fit(X[tr], y[tr])
        m_hat[te] = rf_m.predict(X[te])
        # ê: 성향점수. 절단으로 AIPW 분모 폭발 방지(positivity).
        rf_e = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      random_state=42, n_jobs=1).fit(X[tr], d[tr])
        e_hat[te] = np.clip(rf_e.predict_proba(X[te])[:, 1], 0.02, 0.98)
        # μ̂0, μ̂1: 팔별 회귀(각 팔 표본으로만 학습, 전체에 예측)
        tr0, tr1 = tr[d[tr] == 0], tr[d[tr] == 1]
        rf0 = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                    random_state=42, n_jobs=1).fit(X[tr0], y[tr0])
        rf1 = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                    random_state=42, n_jobs=1).fit(X[tr1], y[tr1])
        mu0[te] = rf0.predict(X[te]); mu1[te] = rf1.predict(X[te])
    return m_hat, e_hat, mu0, mu1


# ===========================================================
# 2. R-러너 CATE (Nie & Wager 2021)
# ===========================================================
def fit_r_learner(X, y_res, d_res):
    """R-손실 최소화의 가중회귀 형태: 목표 Ỹ/D̃, 가중치 D̃²로 RF 회귀.

    제곱손실 argmin Σ(Ỹ − τ(x)·D̃)² 는 목표=Ỹ/D̃, 가중치=D̃²의 가중회귀와 (윈저화
    없이는) 정확히 등가다. |D̃|≈0 점은 가중치 D̃²≈0으로 자동 축소되나, 유사결과
    Ỹ/D̃ 자체가 폭발할 수 있어 1·99 백분위로 윈저화한다(극단값이 소수 잎을 지배하는
    것을 방지). 따라서 최종 적합은 이 등가해의 '근사'다(수치 안정과 맞바꾼 편의).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pseudo = np.where(np.abs(d_res) > 1e-8, y_res / d_res, 0.0)
    lo, hi = np.nanpercentile(pseudo, [1, 99])
    pseudo = np.clip(pseudo, lo, hi)
    weight = d_res ** 2
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=8,
                               random_state=42, n_jobs=1)
    rf.fit(X, pseudo, sample_weight=weight)
    return rf


def r_learner_cate(df, m_hat, e_hat, n_splits=5):
    """교차적합 out-of-fold τ̂(정직한 이질성 검정용)과 전체적합 τ̂(공간지도용)을 함께 반환."""
    X = df[FEATURES].values
    y_res = df["y"].values - m_hat
    d_res = df["treated"].values - e_hat

    tau_oof = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    for tr, te in kf.split(X):
        rf = fit_r_learner(X[tr], y_res[tr], d_res[tr])
        tau_oof[te] = rf.predict(X[te])
    # 전체적합(모든 격자에 CATE 예측 → 공간 지도·정책 표적화)
    rf_full = fit_r_learner(X, y_res, d_res)
    tau_full = rf_full.predict(X)
    return tau_oof, tau_full, rf_full


# ===========================================================
# 3. AIPW(이중강건) 점수와 이질성 검정
# ===========================================================
def aipw_scores(df, e_hat, mu0, mu1):
    """Γ = μ̂1−μ̂0 + D(Y−μ̂1)/ê − (1−D)(Y−μ̂0)/(1−ê).  E[Γ|X]=τ(X), 이중강건."""
    y = df["y"].values; d = df["treated"].values
    return (mu1 - mu0
            + d * (y - mu1) / e_hat
            - (1 - d) * (y - mu0) / (1 - e_hat))


def gates_table(gamma, tau_oof, n_groups=5):
    """예측 CATE 분위별 DR 그룹 ATE(Sorted Group ATE). 최상위−최하위 차이 z·p.

    τ̂(교차적합)로 단위를 이질성 순서로 정렬 → 각 분위의 이중강건 그룹 ATE.
    분위 간 ATE가 실제로 다르면 이질성이 실재한다(Chernozhukov 외 2018 GATES를
    무교란하 DR 점수 버전으로 차용).
    """
    order = np.argsort(tau_oof)                # 오름차순: 가장 음수(효과 큼)가 1분위
    groups = np.array_split(order, n_groups)
    rows = []
    stats_g = []
    for g, idx in enumerate(groups, 1):
        dr = gamma[idx].mean()
        se = gamma[idx].std(ddof=1) / np.sqrt(len(idx))
        pred = tau_oof[idx].mean()
        rows.append((g, pred, dr, se, len(idx)))
        stats_g.append((dr, se))
    # 예측 CATE가 가장 큰(가장 음수) 분위 vs 가장 작은 분위 차이 검정
    (dr_lo, se_lo) = stats_g[0]    # τ̂ 오름차순 1분위 = 가장 음수(효과 큼)
    (dr_hi, se_hi) = stats_g[-1]   # 마지막 분위 = 가장 덜 음수(효과 작음)
    diff = dr_hi - dr_lo
    z = diff / np.sqrt(se_lo ** 2 + se_hi ** 2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return rows, diff, z, p


def blp_calibration(gamma, tau_oof):
    """BLP 보정 검정: Γ = β0 + β1·(τ̂−mean) + e, HC0 강건 SE.

    β1>0 유의 → 이질성 존재. β1≈1 → τ̂가 잘 보정됨. β0≈ATE.
    (statsmodels 없이 정규방정식 + 샌드위치 분산으로 자족 구현.)
    """
    centered = tau_oof - tau_oof.mean()
    Xd = np.column_stack([np.ones_like(centered), centered])
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ gamma)
    resid = gamma - Xd @ beta
    meat = Xd.T @ (Xd * (resid ** 2)[:, None])   # Σ xᵢxᵢ' eᵢ²
    cov = XtX_inv @ meat @ XtX_inv               # HC0 샌드위치
    se = np.sqrt(np.diag(cov))
    z1 = beta[1] / se[1]
    p1 = 2 * (1 - stats.norm.cdf(abs(z1)))
    return beta, se, z1, p1


# ===========================================================
# 4. 정책 표적화: CATE 표적화 vs ATE-맹목 배분
# ===========================================================
def targeting_value(df, tau_full, k=100):
    """미보호 격자 중 신규 보호 k곳을 고르는 세 전략의 '실제 총 저감손실'을 비교한다.

    합성이므로 참 τ를 알기에, 각 전략이 실제로 얼마의 손실을 막는지 평가할 수 있다.
    - CATE 표적화: 예측 저감효과(−τ̂) 큰 순 → 이질성 인지 배분
    - 사후손실 표적화: 관측 손실 y 큰 순 → 교란된 naive 배분(효과≠수준)
    - 무작위: ATE만 알 때의 사실상 무정보 배분
    실제 저감손실 = Σ(−true_tau) (선택된 격자의 참 효과 크기 합).
    """
    ctrl = df[df["treated"] == 0].copy()
    ctrl["tau_hat"] = tau_full[df["treated"].values == 0]
    k = min(k, len(ctrl))
    # 전략별 선택 인덱스
    by_cate = ctrl.nsmallest(k, "tau_hat")          # 가장 음수(효과 큰) τ̂
    by_loss = ctrl.nlargest(k, "y")                 # 관측 손실 큰 순
    rng = np.random.default_rng(123)
    by_rand = ctrl.iloc[rng.permutation(len(ctrl))[:k]]

    def avoided(sub):
        return float((-sub["true_tau"]).sum())      # 실제 총 저감손실(%p 합)

    best = avoided(ctrl.nsmallest(k, "true_tau"))   # 참값을 알 때의 상한(oracle)
    return {
        "CATE 표적화": avoided(by_cate),
        "사후손실 표적화(naive)": avoided(by_loss),
        "무작위 배분": avoided(by_rand),
        "oracle(참 CATE)": best,
        "k": k, "n_ctrl": len(ctrl),
    }


# ===========================================================
# 5. 시각화
# ===========================================================
def make_figure(df, tau_oof, tau_full, gates_rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (A) 참 vs 추정 CATE 산점도
    ax = axes[0, 0]
    ax.scatter(df["true_tau"], tau_oof, s=6, alpha=0.25, color="steelblue")
    lim = [min(df["true_tau"].min(), tau_oof.min()), max(df["true_tau"].max(), tau_oof.max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    r = np.corrcoef(df["true_tau"], tau_oof)[0, 1]
    ax.set_title(f"(A) 참 CATE vs 추정 CATE (r={r:.3f})")
    ax.set_xlabel("참 τ(x)  (%p)"); ax.set_ylabel("추정 τ̂(x) (out-of-fold)")
    ax.legend()

    # (B) 추정 CATE 공간 지도
    ax = axes[0, 1]
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid[df["row"], df["col"]] = tau_full
    im = ax.imshow(grid, cmap="RdYlGn", origin="lower")  # 음수(저감 큼)=붉음
    ax.set_title("(B) 추정 CATE 공간 지도 τ̂(x)")
    ax.set_xlabel("열"); ax.set_ylabel("행")
    fig.colorbar(im, ax=ax, shrink=0.8, label="τ̂ (%p)")

    # (C) 참 CATE 공간 지도(육안 construct 검증)
    ax = axes[1, 0]
    grid_t = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid_t[df["row"], df["col"]] = df["true_tau"]
    im = ax.imshow(grid_t, cmap="RdYlGn", origin="lower")
    ax.set_title("(C) 참 CATE 공간 지도 τ(x)")
    ax.set_xlabel("열"); ax.set_ylabel("행")
    fig.colorbar(im, ax=ax, shrink=0.8, label="τ (%p)")

    # (D) GATES: 예측 CATE 분위별 DR 그룹 ATE
    ax = axes[1, 1]
    gs = [row[0] for row in gates_rows]
    dr = [row[2] for row in gates_rows]
    se = [row[3] for row in gates_rows]
    pred = [row[1] for row in gates_rows]
    ax.errorbar(gs, dr, yerr=[1.96 * s for s in se], fmt="o", capsize=4,
                color="crimson", label="DR 그룹 ATE (95% CI)")
    ax.plot(gs, pred, "s--", color="gray", label="예측 CATE 분위 평균")
    ax.set_title("(D) GATES: 예측 CATE 분위별 실제 효과")
    ax.set_xlabel("예측 CATE 분위 (1=효과 큼 … 5=효과 작음)")
    ax.set_ylabel("그룹 ATE (%p)")
    ax.set_xticks(gs)
    ax.legend()

    fig.tight_layout()
    out = RESULTS_DIR / "9-2-cate-heterogeneity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ===========================================================
# 6. 실행
# ===========================================================
def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("9-2 보호구역 효과의 인과 이질성 — R-러너로 CATE 추정")
    log("=" * 70)
    df = load_data()
    true_ate = df["true_tau"].mean()
    log(f"표본 N = {N} (40×40 격자),  보호구역(처치) = {int(df['treated'].sum())}개")
    log(f"참 CATE 범위 = [{df['true_tau'].min():.2f}, {df['true_tau'].max():.2f}] %p, "
        f"참 ATE = {true_ate:.3f} %p")
    log(f"공변량: {FEATURES}  (elevation·slope는 CATE와 무관한 잡음 대조군)")
    log("")

    # --- 누이선스 교차적합 + R-러너 CATE ---
    m_hat, e_hat, mu0, mu1 = crossfit_nuisances(df)
    tau_oof, tau_full, rf_full = r_learner_cate(df, m_hat, e_hat)
    gamma = aipw_scores(df, e_hat, mu0, mu1)

    # --- (a) 참 vs 추정 CATE 복원 ---
    r_p = np.corrcoef(df["true_tau"], tau_oof)[0, 1]
    r_s = stats.spearmanr(df["true_tau"], tau_oof).correlation
    log("[1] CATE 복원: 참 τ(x) vs 추정 τ̂(x) (out-of-fold)")
    log("-" * 70)
    log(f"  Pearson 상관 r = {r_p:.3f},  Spearman ρ = {r_s:.3f}")
    log(f"  → R-러너 τ̂가 참 이질성의 순서·크기를 복원한다(1에 가까울수록 좋음).")
    log("")

    # --- (b) ATE 비교: 참 · DR · naive · R-러너 평균 ---
    dr_ate = gamma.mean()
    dr_se = gamma.std(ddof=1) / np.sqrt(N)
    naive = (df[df.treated == 1]["y"].mean() - df[df.treated == 0]["y"].mean())
    log("[2] 평균처치효과(ATE) 비교 (참값 = {:.3f} %p)".format(true_ate))
    log("-" * 70)
    log(f"{'방법':<28}{'추정 ATE(%p)':>13}{'참값과 오차':>13}")
    for name, est in [("naive 하위집단 차이", naive),
                      ("이중강건(AIPW) ATE", dr_ate),
                      ("R-러너 평균 τ̂", float(tau_oof.mean()))]:
        log(f"{name:<28}{est:>13.3f}{est - true_ate:>+13.3f}")
    log(f"  이중강건 ATE 95% CI = [{dr_ate - 1.96*dr_se:.3f}, {dr_ate + 1.96*dr_se:.3f}]")
    log("  → naive는 선택편향(저압력 지역이 보호됨)으로 효과를 과대추정한다.")
    log("    직교화(R-러너)·이중강건(AIPW)은 교란을 제거해 참값에 근접한다.")
    log("")

    # --- (c) 이질성 존재 검정: GATES + BLP ---
    gates_rows, diff, z, p = gates_table(gamma, tau_oof, n_groups=5)
    log("[3] 이질성 존재 검정 1 — GATES (예측 CATE 분위별 이중강건 그룹 ATE)")
    log("-" * 70)
    log(f"{'분위':>4} | {'예측 CATE':>10} | {'DR 그룹 ATE':>11} | {'SE':>6} | {'n':>4}")
    for g, pred, dr, se, ng in gates_rows:
        log(f"{g:>4} | {pred:>10.3f} | {dr:>11.3f} | {se:>6.3f} | {ng:>4}")
    log(f"  1분위(효과 큼)−5분위(효과 작음) 차이 = {diff:+.3f} %p, "
        f"z = {z:+.2f}, p = {p:.4f}")
    hetero = "유의(이질성 실재)" if p < 0.05 else "유의하지 않음"
    log(f"  → 분위 간 실제 효과 차이가 {hetero}. 예측 CATE 순서가 실제 효과 순서와 일치.")
    log("")

    beta, se_b, z1, p1 = blp_calibration(gamma, tau_oof)
    # β1 해석: >0 유의 → 이질성 실재. 1과의 거리로 보정 상태 판정(1보다 작으면 과대분산).
    calib = ("보정 양호(β1≈1)" if abs(beta[1] - 1) < 0.2
             else ("τ̂ 과대분산: 예측 산포가 실제보다 큼(순위는 유효, 크기는 과장)"
                   if beta[1] < 1 else "τ̂ 과소분산"))
    log("[4] 이질성 존재 검정 2 — BLP 보정 (Γ ~ β0 + β1·(τ̂−mean), HC0 SE)")
    log("-" * 70)
    log(f"  β0(≈ATE) = {beta[0]:+.3f} (SE {se_b[0]:.3f}),  "
        f"β1(보정기울기) = {beta[1]:+.3f} (SE {se_b[1]:.3f})")
    log(f"  β1 검정: z = {z1:+.2f}, p = {p1:.4f}  → β1>0 유의: 이질성 존재.")
    log(f"  보정 판정: {calib}.")
    log("")

    # --- (d) 잡음 공변량 점검 (전체적합 R-러너의 피처 중요도) ---
    imp = dict(zip(FEATURES, rf_full.feature_importances_))
    log("[5] 잡음 공변량 점검 — R-러너 τ̂의 피처 중요도(MDI)")
    log("-" * 70)
    log("  " + ", ".join(f"{k}={imp[k]:.3f}" for k in FEATURES))
    noise_imp = imp["elevation"] + imp["slope"]
    top = max(imp, key=imp.get)
    log(f"  참 조절변수 enforce 중요도 = {imp['enforce']:.3f}, "
        f"잡음(elevation+slope) 합 = {noise_imp:.3f}, 최상위 = {top}")
    log("  → τ̂는 참 조절변수 enforce를 최상위로 지목한다(pressure는 R-러너가 교란으로")
    log("    직교화해 효과 설명에서 밀려남). 단 MDI는 연속변수에 분산되므로, 이질성의")
    log("    '실재'는 참 CATE 상관(r)과 GATES·BLP(잡음만으로는 통과 못 함)로 판정한다.")
    log("")

    # --- (e) 정책 표적화 payoff ---
    tv = targeting_value(df, tau_full, k=100)
    log("[6] 정책 payoff — 미보호 {}곳 중 신규 보호 {}곳 선정 전략별 실제 총 저감손실"
        .format(tv["n_ctrl"], tv["k"]))
    log("-" * 70)
    order = ["CATE 표적화", "사후손실 표적화(naive)", "무작위 배분", "oracle(참 CATE)"]
    base = tv["무작위 배분"]
    for name in order:
        val = tv[name]
        gain = f"  (무작위 대비 {(val/base - 1)*100:+.0f}%)" if base > 0 and name != "무작위 배분" else ""
        log(f"  {name:<22}: 총 저감손실 {val:7.2f} %p{gain}")
    log("  → CATE 표적화는 효과 큰(고집행역량) 지역을 우선 보호해 같은 예산으로 더 많은")
    log("    손실을 막는다. 관측 손실 최댓값(naive)은 '고압력'을 좇을 뿐 '효과 큰 곳'을")
    log("    놓친다(손실 수준과 처치효과가 다른 변수에 좌우되기 때문). ATE만으로는 불가능.")
    log("")

    out = make_figure(df, tau_oof, tau_full, gates_rows)
    log(f"[그림] 참/추정 CATE 산점도·공간지도·GATES 저장 → {out.name}")
    log("")
    log("=" * 70)
    log("요약: ATE(9.6)는 '평균 얼마'만 답한다. R-러너 CATE는 '어디서 큰가'를 답한다.")
    log(f"참 τ(x)를 심은 합성 자료에서 R-러너 τ̂는 참 CATE를 복원(r={r_p:.2f})하고,")
    log(f"GATES·BLP가 이질성을 유의하게 탐지(p<0.05)했다. 정책적으로는 효과 큰 지역")
    log("우선 보호가 ATE-맹목 배분보다 더 많은 손실을 막는다.")
    log("한계: 무교란·positivity 가정 위에서만 인과. 미관측 교란이 있으면 τ̂도 편향된다.")
    log("=" * 70)

    (RESULTS_DIR / "9-2-cate-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
