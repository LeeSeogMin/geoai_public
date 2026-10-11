"""
8장 분석 예제: 야간조명 + 이중차분법(DID)/이중기계학습(DML)으로 개발사업 효과 측정
======================================================================
정책 질문: "개발사업(예: 산업단지·교통망 투자)이 지역 경제활동을 실제로 높였는가?"

3층 모델 압축:
  [1층 GIS]   읍면동/격자 단위 야간조명 패널 집계, 처치·대조 비교
  [2층 AI]    ★ 이중기계학습(DML): 고차원 공변량을 RandomForest로 통제(공간 피처 포함)
              → 단순 비교/선형통제의 편향을 교정. 공간 자기상관(Moran's I) 진단.
  [3층 정책]  DID 정책효과 추정 + event-study 평행추세 점검

핵심 교훈:
  - 개발사업은 잘 되는(고잠재) 지역에 선택적으로 배치된다(selection on observables).
  - 단순 비교는 이 선택을 정책효과로 오인한다.
  - 공변량이 비선형이면 선형통제로 부족 → ML(DML)이 정당해진다(규칙2).
  - 단, "예측 정확도"가 "인과"를 보장하지 않는다(규칙4): 식별 가정(무교란·평행추세)이 전제다.

데이터: 미리 준비한 교육용 시뮬레이션 패널을 불러와 사용한다(진짜 ATT를 알고
        추정량을 검증하는 통제된 설계). 실제로는 통계청 SGIS 격자통계 + VIIRS
        야간조명 + 행정 개발사업 자료를 결합한다.

실행:
    python 8-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 8-1-nightlight-did-dml.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 렌더링(크로스플랫폼)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 데이터 설계 파라미터(준비 스크립트 8-0과 동일 값 — 분석·해석에 사용)
GRID_COLS = 15
N_REGIONS = 300                       # 20행 × 15열 격자(읍면동 가정)
YEARS = np.arange(2016, 2024)         # 8년
TREAT_YEAR = 2020                     # 개발사업 착수
TRUE_ATT = 0.50                       # 진짜 정책효과: 야간조명 성장 +0.50(로그 radiance)


def load_nightlight():
    """미리 준비한 야간조명 패널·공변량을 불러온다.

    데이터는 사업이 잠재 성장력이 높은 지역에 선택적으로 배치되도록(selection on
    observables) 설계돼 있고, 진짜 정책효과(ATT=+0.50)가 결정론적으로 심어져 있다.
    아래 분석은 이 값을 여러 추정량이 얼마나 잘 회복하는지 비교한다.
    처치 지정(treated)은 패널에 저장돼 있어 격자별 첫 값으로 복원한다.
    """
    panel_path = DATA_DIR / "nightlight_panel.parquet"
    cov_path = DATA_DIR / "nightlight_covariates.parquet"
    if not panel_path.exists() or not cov_path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {panel_path}\n먼저 실행: python 8-0-simdata-prep.py")
    panel = pd.read_parquet(panel_path)
    X = pd.read_parquet(cov_path)
    treated = panel.groupby("region")["treated"].first().to_numpy()
    return panel, X, treated


def morans_i(values):
    """4-이웃 격자 가중치 기반 전역 Moran's I(공간 자기상관)."""
    rows = N_REGIONS // GRID_COLS
    idx = {(i // GRID_COLS, i % GRID_COLS): i for i in range(N_REGIONS)}
    z = values - np.mean(values)
    num = 0.0
    w_sum = 0.0
    for (r, c), i in idx.items():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            j = idx.get((r + dr, c + dc))
            if j is not None:
                num += z[i] * z[j]
                w_sum += 1
    den = np.sum(z ** 2)
    return (N_REGIONS / w_sum) * (num / den)


# ===========================================================
# [1층] 처치·대조 비교
# ===========================================================
def layer1_compare(panel):
    g = panel.groupby(["treated", "post"])["light"].mean().unstack("post")
    g.index = ["대조(사업 없음)", "처치(사업)"]
    g.columns = ["사전(2016-2019)", "사후(2020-2023)"]
    print("=== [1층] 처치/대조 평균 야간조명(로그 radiance) ===")
    print(g.round(3).to_string())
    naive = (panel[(panel.post == 1) & (panel.treated == 1)]["light"].mean()
             - panel[(panel.post == 1) & (panel.treated == 0)]["light"].mean())
    print(f"\n단순 사후 비교(처치-대조): {naive:+.3f}  ← 선택 편향 포함")
    return naive


# ===========================================================
# [3층] DID + event-study(평행추세 점검)
# ===========================================================
def layer3_did(panel):
    res = smf.ols("light ~ treated + post + treated:post", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["region"]})
    coef, se = res.params["treated:post"], res.bse["treated:post"]
    print(f"\n=== [3층] DID 추정 ===")
    print(f"DID(treated:post): {coef:+.3f} (SE={se:.3f}, p={res.pvalues['treated:post']:.4f})")
    print(f"  → naive보다 편향이 줄지만 진짜 ATT({TRUE_ATT:+.2f})와는 여전히 차이. "
          f"처치와 상관된 시변 교란 때문(아래 2층 DML로 통제).")

    # event-study: 연도별 (treated×year) 계수 — 사전 계수 ≈ 0 이어야 평행추세
    panel = panel.copy()
    panel["ry"] = panel["year"] - TREAT_YEAR
    es = smf.ols("light ~ C(ry) + treated + treated:C(ry)", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["region"]})
    pre = [p for p in es.params.index if "treated:C(ry)" in p and "[T.-" in p]
    pre_max = max(abs(es.params[p]) for p in pre) if pre else 0.0
    print(f"event-study 사전(pre-treatment) 상호작용 계수 최대 |값|: {pre_max:.3f} "
          f"({'사전추세 평행(필요조건 충족)' if pre_max < 0.1 else '평행추세 의심'})")
    print("  ※ 사전추세 평행은 필요조건이지 충분조건이 아니다. 처치와 상관된 시변 교란이 "
          "있으면 DID도 편향된다 → 2층 DML로 고차원 통제 필요.")
    return coef, es


# ===========================================================
# [2층 AI] DML: 고차원 공변량 통제로 편향 교정
# ===========================================================
def layer2_dml(panel, X, treated, block):
    # 성장(growth) = 사후 평균 - 사전 평균: 시간불변 요인 차분 제거
    pre = panel[panel.post == 0].groupby("region")["light"].mean()
    post = panel[panel.post == 1].groupby("region")["light"].mean()
    growth = (post - pre).values
    D = treated.astype(float)
    Xm = X.values

    # naive(공변량 무시)
    naive = growth[D == 1].mean() - growth[D == 0].mean()

    # 선형 통제(OLS): 잠재 성장력이 비선형이라 부족할 수 있음
    lin = LinearRegression().fit(np.column_stack([D, Xm]), growth)
    ols_att = lin.coef_[0]

    # DML(부분선형, 교차적합): Y,D를 X에 대해 RandomForest로 잔차화 후 효과 추정
    def cross_resid(target, folds):
        out = np.zeros_like(target, dtype=float)
        for tr, te in folds:
            m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
            m.fit(Xm[tr], target[tr])
            out[te] = target[te] - m.predict(Xm[te])
        return out

    def dml_estimate(folds):
        y_res, d_res = cross_resid(growth, folds), cross_resid(D, folds)
        att = float(np.sum(d_res * y_res) / np.sum(d_res * d_res))
        psi = (y_res - att * d_res) * d_res
        se = float(np.sqrt(np.mean(psi ** 2) / np.mean(d_res ** 2) ** 2 / len(growth)))
        return att, se

    # (a) 무작위 KFold 교차적합: 공간 자기상관이 강하면 인접 격자가 train/test로 갈려
    #     공간 누수(spatial leakage) 위험 — "공간 명시적"이 아니다.
    rand_folds = list(KFold(5, shuffle=True, random_state=42).split(Xm))
    dml_att, dml_se = dml_estimate(rand_folds)

    # (b) 공간 블록 CV(규칙1): 격자 행 밴드(인접 단위)를 통째로 같은 fold에 묶어
    #     train/test를 공간적으로 분리 → 누수 차단. 이것이 공간 명시적 교차적합이다.
    spatial_folds = [(np.where(block != b)[0], np.where(block == b)[0])
                     for b in np.unique(block)]
    dml_sp_att, dml_sp_se = dml_estimate(spatial_folds)

    print("\n=== [2층 AI] 고차원 공변량 통제: 단순 vs 선형 vs DML(무작위 CV vs 공간 블록 CV) ===")
    print(f"{'방법':<30}{'ATT 추정':>10}{'진짜값과 오차':>14}")
    for name, est in [("naive(통제 없음)", naive),
                      ("OLS(선형 통제)", ols_att),
                      ("DML(무작위 KFold CV)", dml_att),
                      ("DML(공간 블록 CV)", dml_sp_att)]:
        print(f"{name:<30}{est:>10.3f}{est - TRUE_ATT:>+14.3f}")
    print(f"DML 표준오차 — 무작위 CV: {dml_se:.3f} · 공간 블록 CV: {dml_sp_se:.3f}  (진짜 ATT={TRUE_ATT:+.2f})")
    print("해석: 사업이 고잠재 지역에 선택 배치되어 무통제는 과대추정. 관측 공변량을 통제하면")
    print("      진짜값에 근접한다. 이 합성 구조에서는 선형(OLS)도 충분했고, DML은 비선형·")
    print("      고차원 교란일 때 정당하다(항상 우월하지 않음 — 규칙2: 정당성 입증 후 사용).")
    print("공간 명시성(규칙1): Moran's I가 큰 데이터에서 무작위 CV는 인접 격자 누수를 부른다.")
    print("      공간 블록 CV는 train/test를 공간적으로 분리해 누수를 막는다 — 두 추정의 차이가 그 신호다.")
    print("주의(규칙4): 통제 후 인과 해석은 '관측 공변량 조건부 평행추세/무교란' 가정 위에서만 성립. 예측력≠인과.")
    return naive, ols_att, dml_att, dml_se, dml_sp_att, dml_sp_se


def plot_event_study(es):
    """event-study 상호작용 계수 플롯(평행추세 시각 점검)."""
    terms = [(p, es.params[p]) for p in es.params.index if "treated:C(ry)" in p]
    rys, vals = [], []
    for p, v in terms:
        try:
            rys.append(int(p.split("[T.")[1].rstrip("]")))
            vals.append(v)
        except (IndexError, ValueError):
            continue
    order = np.argsort(rys)
    rys, vals = np.array(rys)[order], np.array(vals)[order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(-0.5, ls="--", color="#c0392b", label="Project start")
    ax.plot(rys, vals, "o-", color="#2c3e50")
    ax.set_xlabel("Years relative to project start")
    ax.set_ylabel("Treated x year coef. (vs. base year)")
    ax.set_title("Event-study: pre-trend check (pre coefs near 0 = parallel)")
    ax.legend()
    fig.tight_layout()
    out = DATA_DIR / "fig_8_1_event_study.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n그림 저장: {out.relative_to(SCRIPT_DIR.parent.parent.parent)}")


def main():
    print("야간조명 + DID/DML 개발사업 효과 분석 (교육용 시뮬레이션)")
    print(f"격자 {N_REGIONS}개 · {YEARS[0]}~{YEARS[-1]} · 사업착수 {TREAT_YEAR} · 진짜 ATT {TRUE_ATT:+.2f}\n")

    panel, X, treated = load_nightlight()

    # 공간 자기상관 진단(공간 명시성)
    mi_urban = morans_i(X["x_urban"].values)
    mi_x0 = morans_i(X["x0"].values)
    print(f"=== 공간 자기상관 진단 (Moran's I) ===")
    print(f"도시화 공변량 I={mi_urban:.3f}(공간 군집) vs 비공간 공변량 I={mi_x0:.3f}(무작위 근처)\n")

    # 공간 블록(행 밴드): 20행 격자를 4행씩 5개 블록으로 → 공간 블록 CV용
    block = (np.arange(N_REGIONS) // GRID_COLS) // 4

    layer1_compare(panel)
    coef, es = layer3_did(panel)
    layer2_dml(panel, X, treated, block)
    plot_event_study(es)

    print("\n[3층 구조] 1층 집계 → 2층 DML 고차원 통제 → 3층 DID 인과·event-study. "
          "AI(2층)는 비선형 고차원 교란을 통제할 때 정당하며, 식별 가정을 대체하지 않는다.")


if __name__ == "__main__":
    main()
