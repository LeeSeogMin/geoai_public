"""
12장 실습 3: 고차원·비선형 교란과 이중기계학습(DML) — 왜 순진 OLS로는 안 되는가 (인과 척추)
================================================================================
복지·보건 정책 질문: "긴급복지/방문상담 같은 선제 개입(처치 D)이 위기가구의 위기 악화
(결과 Y)를 실제로 줄였는가? 아니면 원래 더 위험한 가구가 개입을 받았을 뿐인가?"

3층 모델에서의 위치 — 이 분석은 [3층 정책·인과]까지 간다(식별원=관측 공변량 조건부 무교란):
  [1층 GIS]  없음(이 예제는 공간 축이 아니라 '고차원 교란 통제' 축을 다룬다). 실데이터에서는
             47종 위기지표를 읍면동·격자로 집계한 공간 피처가 X에 들어간다.
  [2층 AI ]  랜덤포레스트가 nuisance 함수 ĝ(X)=E[Y|X], m̂(X)=E[D|X]를 유연하게 추정.
             예측력을 키우는 게 목적이 아니라, 비선형 교란을 걷어내 처치효과를 식별하기 위함.
  [3층 정책] 개입의 국소 인과효과 θ. 단, 무교란·positivity 가정 위에서만 성립하며,
             미관측 교란이 있으면 DML도 편향된다(규칙4: 예측≠인과).

방법 홈: DML 정식 설명은 8.4절(ML×인과). 여기서는 '왜 DML이 필요한가'를 실증한다.

교육 의도(construct validation, §12.7 선례):
  실제 복지 미시데이터는 개인정보·목적제한이 강하고 무엇보다 '참 θ'를 알 수 없어 추정량의
  타당성을 직접 검증할 수 없다. 본 예제는 *참 처치효과 θ=−2.0을 아는 합성 DGP*로 검증한다:
    (a) 순진 OLS(Y~D+X 선형)는 g(X) 비선형 오설정 + 개입 대상 선택으로 편향된다
    (b) DML(RF nuisance + 직교화 + 교차적합)은 참 θ≈−2.0을 복원한다
    (c) 교차적합을 빼면(같은 표본에서 nuisance 학습·평가) 과적합 편향이 잔존한다
    (d) fold 분할·시드에 따른 안정성과 부트스트랩/폴드 표준오차
  → 합성이지만 '방법이 옳게 작동하는 조건과, 순진 OLS가 실패하는 조건'을 실증한다.

데이터: 미리 준비한 교육용 합성 표본을 불러와 사용(개인 식별 불가). 실제 분석에서는 복지
  사각지대 발굴시스템의 47종 위기정보(단전·체납·의료비 등)를 읍면동·격자로 비식별 집계한 X,
  긴급복지/방문상담 개입 여부 D, 후속 위기점수 Y를 사용한다(무교란·positivity 성립 논증은
  별도 과제). 참 θ=−2.0을 아는 합성 데이터의 설계(부분선형모형, 공유 비선형 교란 c(X),
  선택편향)와 생성은 12-0-simdata-prep.py에 있다 — 이 예제는 고차원 교란 통제의 시연이라
  참값을 아는 합성이 본질이다.

실행:
    python 12-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 12-3-dml-highdim-confounding.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# 0. 데이터 로드 (부분선형모형 Y = θ·D + g(X) + ε의 합성 표본)
# ===========================================================
# X = 고차원 위기지표(20종). D = 선제 개입 여부(이진). Y = 후속 위기점수(낮을수록 좋음).
# 설계 핵심(생성은 12-0-simdata-prep.py):
#   - 결과 기저 g(X)와 처치확률 m(X)가 '같은 비선형 교란 c(X)'를 공유한다 → 개입(D=1)이
#     위기 심각 영역(c 큰 곳)에 집중(선택편향). 이때 X를 '선형'으로만 통제하면 c(X)의
#     비선형 잔차가 D와 상관돼 θ̂가 편향된다.
#   - c(X)는 '가법 비선형 주효과'(계단·꺾임·단조곡선)라 선형회귀(OLS)로는 못 걷어내지만
#     랜덤포레스트는 정확히 추정한다 → OLS는 편향, DML은 참값 복원.
#   - 참 θ=−2.0(개입이 위기점수를 2.0 낮춤)을 심어 두어, 추정량이 이를 복원하는지 검증한다.
N = 6000
P = 20                  # 위기지표 차원(고차원; 실제로 c에 쓰이는 것은 소수, 나머지는 잡음지표)
TRUE_THETA = -2.0       # 참 처치효과: 개입이 위기점수를 2.0 낮춤(음의 효과=개선)


def load_sample():
    """미리 준비한 합성 표본을 불러온다.

    반환: X(N×P), D(처치), Y(결과), m(참 성향점수), ey_true(참 E[Y|X]=θ·m+g(X)).
    ey_true는 nuisance 추정정확도(참 함수 대비 RMSE) 점검에만 쓴다.
    """
    path = DATA_DIR / "dml_sample.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 12-0-simdata-prep.py")
    df = pd.read_parquet(path)
    X = df[[f"x{j:02d}" for j in range(P)]].values
    return X, df["D"].values, df["Y"].values, df["m"].values, df["ey_true"].values


# ===========================================================
# 1. 순진 OLS 추정량 (Y ~ [1, D, X] 선형) — 오설정 편향 대비용
# ===========================================================
def naive_ols(X, D, Y):
    """Y를 D와 X에 대해 *선형*으로 회귀 → D 계수 = 순진 처치효과 추정.

    g(X)가 비선형이므로 이 선형 통제는 교란을 다 걷어내지 못한다. 그 잔차가 D와 상관돼
    θ̂가 편향된다(함수형태 오설정 + 개입 대상 선택편향). 반환: (θ̂, 표준오차).
    """
    n = len(Y)
    Xmat = np.column_stack([np.ones(n), D, X])           # [1, D, X1..Xp]
    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    resid = Y - Xmat @ beta
    # 이분산-강건(HC0) 표준오차
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    meat = (Xmat * resid[:, None]).T @ (Xmat * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    theta = beta[1]                                      # D 계수
    se = np.sqrt(cov[1, 1])
    return theta, se


# ===========================================================
# 2. DML 부분선형 추정량 (직교화 + 교차적합)
# ===========================================================
def make_learners(seed):
    """nuisance 학습기: 결과회귀 ĝ(RF회귀), 처치확률 m̂(RF분류). 유연·비선형.

    max_features=0.5로 잡음지표 사이에서도 관련 교란변수를 잘 찾도록 하고(가법 주효과
    추정 정확도↑), min_samples_leaf=5로 과적합을 억제한다. nuisance 추정오차가 작아야
    DML이 참 θ를 복원한다(RMSE/표준편차 ≈ 0.3 수준을 목표).
    """
    g_learner = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=0.5,
        random_state=seed, n_jobs=-1,
    )
    m_learner = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features=0.5,
        random_state=seed, n_jobs=-1,
    )
    return g_learner, m_learner


def dml_partial_linear(X, D, Y, n_folds=5, seed=0, crossfit=True):
    """부분선형 DML. Ỹ=Y−ĝ(X), D̃=D−m̂(X) 잔차화 후 θ̂=(D̃ᵀD̃)⁻¹(D̃ᵀỸ).

    crossfit=True : nuisance를 다른 fold에서 학습해 held-out fold에 예측(과적합 편향 제거).
    crossfit=False: 전체 표본에서 nuisance 학습·같은 표본에 예측(과적합 편향 잔존 — 대비용).

    반환: dict(theta, se_plugin, ghat, mhat, resid_y, resid_d).
      se_plugin = Neyman-orthogonal moment의 이분산-강건 표준오차.
    """
    n = len(Y)
    ghat = np.empty(n)   # Ê[Y|X]
    mhat = np.empty(n)   # Ê[D|X] = 처치확률

    if crossfit:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            g_learner, m_learner = make_learners(seed)
            g_learner.fit(X[train_idx], Y[train_idx])
            m_learner.fit(X[train_idx], D[train_idx])
            ghat[test_idx] = g_learner.predict(X[test_idx])
            mhat[test_idx] = m_learner.predict_proba(X[test_idx])[:, 1]
    else:
        # 교차적합 없음: 전체 표본에서 학습하고 같은 표본에 예측(in-sample) → 과적합 편향
        g_learner, m_learner = make_learners(seed)
        g_learner.fit(X, Y)
        m_learner.fit(X, D)
        ghat = g_learner.predict(X)
        mhat = m_learner.predict_proba(X)[:, 1]

    # 직교화(partialling-out): FWL의 비모수 확장
    resid_y = Y - ghat
    resid_d = D - mhat

    # θ̂ = 잔차회귀 기울기(D̃에 대한 Ỹ)
    denom = float(resid_d @ resid_d)
    theta = float(resid_d @ resid_y) / denom

    # Neyman-orthogonal moment ψ = D̃(Ỹ − θ D̃)의 이분산-강건 SE
    psi = resid_d * (resid_y - theta * resid_d)
    var = float(np.sum(psi ** 2)) / (denom ** 2)
    se = np.sqrt(var)

    return {
        "theta": theta, "se": se, "ghat": ghat, "mhat": mhat,
        "resid_y": resid_y, "resid_d": resid_d,
    }


def residual_bootstrap_se(resid_d, resid_y, n_boot=1000, seed=0):
    """잔차 부트스트랩 SE(nuisance 고정): 직교화 잔차쌍 (D̃,Ỹ)를 복원추출해 θ̂ 재계산.

    nuisance(ĝ,m̂)를 고정하고 최종 잔차회귀의 표본변동만 반영하는 보조 SE.
    (완전한 추론은 nuisance 추정 불확실성까지 포함하나, 여기서는 폴드 SE와 상호 점검용.)
    """
    rng = np.random.default_rng(seed)
    n = len(resid_d)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rd, ry = resid_d[idx], resid_y[idx]
        boots[b] = float(rd @ ry) / float(rd @ rd)
    return float(np.std(boots, ddof=1))


# ===========================================================
# 3. 실행: 추정·대비·안정성·시각화
# ===========================================================
def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 72)
    log("12-3 고차원·비선형 교란과 이중기계학습(DML) — 순진 OLS는 왜 편향되는가")
    log("=" * 72)
    log(f"참 처치효과 θ(설계값) = {TRUE_THETA:+.2f}  (개입이 위기점수를 낮춤)")
    log(f"표본 N = {N},  위기지표 차원 p = {P}")
    X, D, Y, m, ey_true = load_sample()
    log(f"처치비율 = {D.mean():.3f}  (성향점수 범위 {m.min():.3f}~{m.max():.3f}, positivity 대략 성립)")
    log("")

    # --- (a) 순진 OLS: 선형 통제 = 오설정 편향 ---
    log("[1] 순진 OLS (Y ~ D + X 선형) — 함수형태 오설정 + 개입 선택편향")
    log("-" * 72)
    ols_theta, ols_se = naive_ols(X, D, Y)
    log(f"  OLS θ̂ = {ols_theta:+.3f}  (SE {ols_se:.3f}),  참값 대비 편향 {ols_theta - TRUE_THETA:+.3f}")
    log("  → g(X)가 비선형이라 선형 통제가 교란을 다 못 걷어낸다. 개입이 위기 심각 영역에")
    log("    집중돼(선택편향), 남은 교란이 θ̂를 0쪽으로 끌어올려 개입 효과를 과소평가한다.")
    log("")

    # --- (b) DML(교차적합 O): 참값 복원 ---
    log("[2] DML(직교화 + 교차적합 K=5) — 참 θ 복원")
    log("-" * 72)
    dml = dml_partial_linear(X, D, Y, n_folds=5, seed=0, crossfit=True)
    boot_se = residual_bootstrap_se(dml["resid_d"], dml["resid_y"], n_boot=1000, seed=1)
    # nuisance 추정 정확도(참 함수 대비): 이것이 작아야 DML이 작동한다는 전제를 로그로 남긴다.
    # 결과 nuisance의 참값 ℓ(X)=E[Y|X]=θ·m(X)+g(X)(partialling-out 대상)와 처치 nuisance m(X)는
    # 데이터 준비 단계에서 함께 저장해 둔 것을 불러와 쓴다(ey_true, m).
    g_nrmse = np.sqrt(np.mean((dml["ghat"] - ey_true) ** 2)) / ey_true.std()
    m_nrmse = np.sqrt(np.mean((dml["mhat"] - m) ** 2)) / m.std()
    log(f"  nuisance 추정정확도(참 함수 대비 RMSE/표준편차): 결과 ĝ={g_nrmse:.2f}, 처치 m̂={m_nrmse:.2f}")
    log(f"  DML θ̂ = {dml['theta']:+.3f}  (폴드 SE {dml['se']:.3f}, 잔차부트스트랩 SE {boot_se:.3f})")
    log(f"  참값 대비 편향 {dml['theta'] - TRUE_THETA:+.3f}  "
        f"95% CI [{dml['theta'] - 1.96*dml['se']:+.3f}, {dml['theta'] + 1.96*dml['se']:+.3f}]")
    log("  → RF가 비선형 nuisance ĝ·m̂을 잡고, 직교화가 그 추정오차에 둔감한 moment를 준다.")
    log("")

    # --- (c) DML(교차적합 X): 과적합 편향 잔존 ---
    log("[3] DML(직교화 O, 교차적합 X) — 과적합 편향 대비")
    log("-" * 72)
    dml_nocf = dml_partial_linear(X, D, Y, seed=0, crossfit=False)
    log(f"  DML(no CF) θ̂ = {dml_nocf['theta']:+.3f}  (SE {dml_nocf['se']:.3f}),  "
        f"참값 대비 편향 {dml_nocf['theta'] - TRUE_THETA:+.3f}")
    log("  → 같은 표본에서 nuisance를 학습·평가하면 잔차가 인위적으로 작아져(과적합) 정규화")
    log("    편향이 θ̂에 남는다. 직교화만으로 부족 — 교차적합이 함께 있어야 한다.")
    log("")

    # --- (d) fold 분할·시드 안정성 (단일 값 확정 금지) ---
    log("[4] 교차적합 fold 분할 안정성 (시드별 재추정)")
    log("-" * 72)
    seed_thetas = []
    for s in range(5):
        r = dml_partial_linear(X, D, Y, n_folds=5, seed=s, crossfit=True)
        seed_thetas.append(r["theta"])
        log(f"  seed={s}: DML θ̂ = {r['theta']:+.3f}")
    seed_thetas = np.array(seed_thetas)
    log(f"→ 시드별 평균 {seed_thetas.mean():+.3f}, 표준편차 {seed_thetas.std(ddof=1):.3f} "
        f"(범위 {seed_thetas.min():+.3f}~{seed_thetas.max():+.3f}).")
    log("  단일 fold 분할 값을 최종치로 확정하지 말고 여러 분할로 안정성을 확인한다.")
    log("")

    # --- 편향 비교 요약표(로그) ---
    log("[요약] 순진 OLS vs DML 편향 비교 (참 θ = %.2f)" % TRUE_THETA)
    log("-" * 72)
    log(f"{'추정량':<26} | {'θ̂':>8} | {'편향':>8} | {'SE':>7}")
    log(f"{'순진 OLS(선형 통제)':<24} | {ols_theta:>8.3f} | {ols_theta-TRUE_THETA:>+8.3f} | {ols_se:>7.3f}")
    log(f"{'DML(교차적합 없음)':<24} | {dml_nocf['theta']:>8.3f} | {dml_nocf['theta']-TRUE_THETA:>+8.3f} | {dml_nocf['se']:>7.3f}")
    log(f"{'DML(교차적합 K=5)':<24} | {dml['theta']:>8.3f} | {dml['theta']-TRUE_THETA:>+8.3f} | {dml['se']:>7.3f}")
    log("")

    # --- 시각화: 직교화 잔차회귀(FWL 기하) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (좌) 직교화 잔차 산점도 D̃ vs Ỹ + 기울기 θ̂
    ax = axes[0]
    rd, ry = dml["resid_d"], dml["resid_y"]
    ax.scatter(rd, ry, s=8, alpha=0.15, color="gray")
    grid = np.linspace(rd.min(), rd.max(), 50)
    ax.plot(grid, dml["theta"] * grid, color="crimson", lw=2.5,
            label=f"DML 기울기 θ̂ = {dml['theta']:.2f}")
    ax.plot(grid, TRUE_THETA * grid, color="black", ls="--", lw=1.5,
            label=f"참 θ = {TRUE_THETA:.2f}")
    ax.set_xlabel("처치 잔차  D̃ = D − m̂(X)")
    ax.set_ylabel("결과 잔차  Ỹ = Y − ĝ(X)")
    ax.set_title("직교화 잔차회귀 (부분선형 DML)")
    ax.legend()

    # (우) 추정량별 편향 막대
    ax = axes[1]
    labels = ["순진 OLS", "DML\n(no CF)", "DML\n(CF K=5)"]
    thetas = [ols_theta, dml_nocf["theta"], dml["theta"]]
    ses = [ols_se, dml_nocf["se"], dml["se"]]
    colors = ["darkorange", "goldenrod", "seagreen"]
    ax.bar(labels, thetas, yerr=[1.96 * s for s in ses], capsize=6, color=colors, alpha=0.85)
    ax.axhline(TRUE_THETA, color="black", ls="--", lw=1.5, label=f"참 θ = {TRUE_THETA:.2f}")
    ax.set_ylabel("처치효과 추정 θ̂ (95% CI)")
    ax.set_title("추정량별 편향 비교")
    ax.legend()

    fig_path = RESULTS_DIR / "12-3-dml-orthogonal-residual.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    log(f"[그림] 직교화 잔차회귀 + 편향 비교 저장 → {fig_path.name}")
    log("")

    log("=" * 72)
    log("요약: 교란이 고차원·비선형이고 처치가 그 위험 영역에 집중될 때, 순진 OLS(선형 통제)는")
    log(f"오설정+선택편향으로 θ̂={ols_theta:+.2f}로 빗나간다. DML은 유연한 ML로 nuisance를 잡고")
    log(f"직교화+교차적합으로 정규화 편향을 제거해 참 θ={TRUE_THETA:+.2f}를 복원한다(θ̂={dml['theta']:+.2f}).")
    log("교차적합을 빼면 과적합 편향이 남는다. 단, 이 회복은 '관측 공변량 조건부 무교란'")
    log("가정 위에서만 성립한다 — 미관측 교란이 있으면 DML도 편향된다(예측≠인과).")
    log("=" * 72)

    # 텍스트 로그도 남김(하네스 stdout 캡처와 별개로 재확인용)
    (RESULTS_DIR / "12-3-dml-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
