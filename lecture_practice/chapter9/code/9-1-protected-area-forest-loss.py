"""
9장 실습: 보호구역 정책의 산림손실 저감 효과 분석
=====================================================
환경·기후 정책(보호구역 지정)의 효과를 위성 기반 산림손실 시계열로 평가한다.

핵심 정책 질문: "보호구역 지정이 실제로 산림 벌채를 줄였는가?"

학습 포인트 (Andam et al. 2008, PNAS 의 문제의식):
  - 보호구역은 애초에 개발 압력이 낮은 곳(오지·고지대)에 지정되는 경향이 있다(selection bias).
  - 따라서 "보호구역 안 vs 밖"의 단순 비교(naive)는 정책 효과를 과대추정한다.
  - 이중차분법(DID)과 매칭은 이 편향을 교정해 정책의 '순효과'에 접근한다.

데이터: 미리 준비한 교육용 시뮬레이션 패널을 불러와 사용한다. 보호구역 selection
        bias·평행추세·정책효과·노이즈를 명시적 파라미터로 심어 생성한 자료다.
        **분석·추정 결과는 실제 계산값이다(가짜 아님).**
        실제 분석에서는 Hansen Global Forest Change / Global Forest Watch 데이터를 사용한다.

기본(학부): 보호구역 안/밖 연도별 산림손실률 비교 + 공간단위 우선순위표
심화(대학원): DID 정책효과 추정(군집 강건 표준오차) + 매칭으로 selection bias 교정

실행:
    python 9-0-simdata-prep.py                  # 최초 1회: 데이터 준비
    python 9-1-protected-area-forest-loss.py
"""

from pathlib import Path
import os
import tempfile

_CACHE_DIR = Path(tempfile.gettempdir()) / "geoai_mpl_cache"
_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))
import matplotlib
matplotlib.use("Agg")  # 크로스플랫폼 headless 렌더링 (GUI 없이 PNG 저장)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 경로: 스크립트 기준 상대 경로 (GOTCHAS: pathlib, 하드코딩 금지)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 1. 데이터 로드 (준비 스크립트가 저장한 합성 패널)
# ===========================================================
# 아래 상수는 9-0-simdata-prep.py의 생성 파라미터와 동일하며, 본문 출력·그림·
# 공간 배치에 쓰인다(데이터를 다시 생성하지는 않는다).
N_UNITS = 300          # 분석 단위(산림 격자) 수
N_PROTECTED = 150      # 보호구역(처리군)
YEARS = np.arange(2001, 2021)   # 20년
TREAT_YEAR = 2011      # 보호구역 지정 발효 연도
TRUE_EFFECT = -0.80    # 진짜 정책 효과: 보호로 연간 손실률 0.80%p 감소 (음수=저감)
GRID_COLS = 15         # 300단위를 20행×15열 격자로 배치(공간 명시적 피처용, 결정론적)


def load_panel():
    """9-0-simdata-prep.py가 저장한 산림손실 패널을 불러온다.

    패널에는 '개발 압력(pressure)'이 높을수록 손실이 크고, 보호구역은 압력이
    낮은 곳에 우선 지정되는 selection bias가 심겨 있다 → 단순 비교는 편향된다.
    공통 시간추세(평행추세)와 처리군·사후에만 걸리는 진짜 정책효과, 관측되지 않는
    공간 효과, 노이즈가 함께 들어 있어 아래 DID·매칭·공간 ML의 재료가 된다.
    """
    path = DATA_DIR / "forest_loss_panel.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 9-0-simdata-prep.py")
    return pd.read_parquet(path)


# ===========================================================
# 2. 기본 분석 (학부): 보호구역 안/밖 손실률 비교
# ===========================================================
def basic_comparison(df):
    """연도별·집단별 평균 손실률과 사전/사후 요약."""
    group = (
        df.groupby(["treated", "post"])["loss"].mean().unstack("post")
    )
    group.index = ["보호구역 밖(대조군)", "보호구역 안(처리군)"]
    group.columns = ["사전(2001-2010)", "사후(2011-2020)"]

    print("=== [기본] 보호구역 안/밖 평균 산림손실률 (%) ===")
    print(group.round(3).to_string())

    # naive 추정: 사후 기간 '안 - 밖' 차이 (selection bias로 편향됨)
    post = df[df["post"] == 1]
    naive = (post[post.treated == 1]["loss"].mean()
             - post[post.treated == 0]["loss"].mean())
    print(f"\nnaive 추정(사후 안-밖 차이): {naive:+.3f} %p"
          f"  ← selection bias로 효과 과대추정")
    return group


# ===========================================================
# 3. 심화 분석 (대학원): 이중차분법(DID)
# ===========================================================
def did_estimate(df, label="전체 표본"):
    """DID 회귀: loss ~ treated + post + treated:post.

    treated:post 계수 = 정책의 평균 처리효과(ATT). 단위 군집 강건 표준오차.
    """
    model = smf.ols("loss ~ treated + post + treated:post", data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["unit"]})
    coef = res.params["treated:post"]
    se = res.bse["treated:post"]
    pval = res.pvalues["treated:post"]
    print(f"\n=== [심화] DID 추정 ({label}, n={len(df)}) ===")
    print(f"DID 효과(treated:post): {coef:+.3f} %p  (SE={se:.3f}, p={pval:.4f})")
    return coef, se, pval


def matched_did(df):
    """압력(pressure)이 겹치는 구간으로 표본을 제한(공통 지지)한 뒤 DID 재추정.

    보호구역(저압력)과 비보호(고압력)의 압력 분포가 겹치는 영역만 비교 →
    selection bias 완화. (교육용 단순 매칭: 공통 지지 구간 제한)
    """
    p_treat = df[df.treated == 1]["pressure"]
    p_ctrl = df[df.treated == 0]["pressure"]
    low = max(p_treat.min(), p_ctrl.min())
    high = min(p_treat.max(), p_ctrl.max())
    matched = df[(df.pressure >= low) & (df.pressure <= high)].copy()
    n_units = matched["unit"].nunique()
    print(f"\n공통 지지 구간 압력 [{low:.2f}, {high:.2f}] → 매칭 후 단위 {n_units}개")
    return did_estimate(matched, label="매칭(공통지지) 표본")


# ===========================================================
# 4. 시각화
# ===========================================================
def plot_trends(df):
    """연도별 집단 평균 손실률 추세 (평행추세 + 정책 발효 시점)."""
    yearly = df.groupby(["year", "treated"])["loss"].mean().unstack("treated")
    # 그림 라벨은 크로스플랫폼 폰트 호환을 위해 영문 사용 (GOTCHAS: 한글 폰트 의존 회피)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(yearly.index, yearly[0], "o-", label="Outside PA (control)", color="#c0392b")
    ax.plot(yearly.index, yearly[1], "s-", label="Inside PA (treated)", color="#27ae60")
    ax.axvline(TREAT_YEAR - 0.5, ls="--", color="gray", label=f"PA designated ({TREAT_YEAR})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual forest loss rate (%)")
    ax.set_title("Forest loss: inside vs outside protected areas (DID design)")
    ax.legend()
    fig.tight_layout()
    out = DATA_DIR / "fig_9_1_forest_loss_trends.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n그림 저장: {out.relative_to(SCRIPT_DIR.parent.parent.parent)}")


# ===========================================================
# 5. 공간단위 정책 산출물
# ===========================================================
def _minmax(series):
    """0~1 정규화. 값이 모두 같으면 0을 반환한다."""
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def spatial_policy_outputs(df, did_coef):
    """격자·행정구역 단위의 정책 판단 자료를 생성한다.

    영상 또는 래스터에서 얻은 산림손실률은 최종 산출물이 아니다.
    이 함수는 산림손실 시계열을 분석 단위별 지표와 우선순위표로 바꾼다.
    """
    unit_panel = (
        df.groupby(["unit", "admin_area", "treated", "pressure", "habitat_value", "post"])["loss"]
        .mean()
        .unstack("post")
        .reset_index()
        .rename(columns={0: "pre_loss", 1: "post_loss"})
    )
    unit_panel["loss_change"] = unit_panel["post_loss"] - unit_panel["pre_loss"]
    unit_panel["estimated_avoided_loss_pp"] = np.where(
        unit_panel["treated"] == 1, -did_coef, 0.0
    )

    unit_panel["priority_score"] = (
        0.40 * _minmax(unit_panel["post_loss"])
        + 0.25 * _minmax(unit_panel["pressure"])
        + 0.20 * _minmax(unit_panel["habitat_value"])
        + 0.15 * (1 - unit_panel["treated"])
    )
    unit_panel["priority_rank"] = unit_panel["priority_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    unit_panel = unit_panel.sort_values("priority_rank")

    district = (
        unit_panel.groupby("admin_area")
        .agg(
            n_units=("unit", "count"),
            protected_share=("treated", "mean"),
            mean_pressure=("pressure", "mean"),
            mean_habitat_value=("habitat_value", "mean"),
            pre_loss=("pre_loss", "mean"),
            post_loss=("post_loss", "mean"),
            mean_loss_change=("loss_change", "mean"),
            estimated_avoided_loss_pp=("estimated_avoided_loss_pp", "mean"),
            high_priority_units=("priority_rank", lambda s: int((s <= 30).sum())),
        )
        .reset_index()
    )
    district["priority_score"] = (
        0.35 * _minmax(district["post_loss"])
        + 0.25 * _minmax(district["mean_pressure"])
        + 0.20 * _minmax(district["mean_habitat_value"])
        + 0.20 * _minmax(district["high_priority_units"])
    )
    district["priority_rank"] = district["priority_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    district = district.sort_values("priority_rank")

    unit_out = RESULTS_DIR / "spatial_unit_priority.csv"
    district_out = RESULTS_DIR / "admin_area_priority.csv"
    unit_panel.to_csv(unit_out, index=False)
    district.to_csv(district_out, index=False)

    print("\n=== 공간단위 정책 산출물 ===")
    print(f"격자 단위 우선순위표 저장: {unit_out.relative_to(SCRIPT_DIR.parent.parent.parent)}")
    print(f"행정구역 우선순위표 저장: {district_out.relative_to(SCRIPT_DIR.parent.parent.parent)}")
    print("\n행정구역 우선순위 상위 5개")
    cols = [
        "admin_area",
        "priority_rank",
        "post_loss",
        "mean_pressure",
        "protected_share",
        "high_priority_units",
    ]
    print(district[cols].head(5).round(3).to_string(index=False))
    return unit_panel, district


# ===========================================================
# 5.5 [2층] 공간 명시적 AI — 벌채위험 예측
# ===========================================================
def _spatial_lag(unit_panel):
    """격자 좌표 기반 4-이웃 사전 손실 평균(공간 자기상관 피처).

    GeoAI의 정의적 요건인 '공간 명시성'을 위해, 단위를 격자에 배치하고
    이웃의 사전 손실을 피처로 내장한다(Tobler 제1법칙: 가까운 것이 더 유사).
    """
    p = unit_panel.sort_values("unit").reset_index(drop=True).copy()
    p["row"] = p["unit"] // GRID_COLS
    p["col"] = p["unit"] % GRID_COLS
    loss_by_rc = {(r, c): v for r, c, v in zip(p["row"], p["col"], p["pre_loss"])}
    lags = []
    for r, c in zip(p["row"], p["col"]):
        neigh = [loss_by_rc.get((r + dr, c + dc))
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neigh = [v for v in neigh if v is not None]
        lags.append(float(np.mean(neigh)) if neigh else float(p["pre_loss"].mean()))
    p["spatial_lag_pre_loss"] = lags
    return p


def predict_deforestation_risk(unit_panel):
    """[2층] 공간 명시적 ML로 사후 벌채위험을 예측한다.

    - 입력 피처에 공간 자기상관(이웃 손실)을 내장 → 'GIS+ML'이 아니라 GeoAI(규칙1).
    - 공간 블록 CV(좌/우 분할)로 평가 → 일반 CV의 낙관 편향 방지(공간 데이터 누수 차단).
    - 공간 피처 유/무 비교로 'AI가 정당한가'를 입증(규칙2): 공간 구조가 성능을 올리면 ML 정당.
    - 피처 중요도로 설명가능성 확보(규칙3).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    p = _spatial_lag(unit_panel)
    # 타깃 오염 방지: '보호가 없을 때의 손실 위험'을 추정해야 하므로 통제구역(treated=0)만 학습한다.
    # (처리구역의 post_loss는 정책효과가 섞여 있어 '보호 우선순위 위험'으로 쓸 수 없다.)
    ctrl = p[p["treated"] == 0].copy()
    # 위험은 '입지 특성'으로 추정한다(보호 여부는 피처에서 제외 — 순환 방지).
    feat_spatial = ["pressure", "habitat_value", "spatial_lag_pre_loss"]
    feat_nonspatial = ["pressure", "habitat_value"]
    target = "post_loss"

    def spatial_block_cv(feats):
        """2겹 공간 블록 CV: 격자를 좌/우 블록으로 나눠 번갈아 검증(통제구역만)."""
        r2s, rmses = [], []
        for test_left in (True, False):
            if test_left:
                tr, te = ctrl[ctrl.col >= GRID_COLS // 2], ctrl[ctrl.col < GRID_COLS // 2]
            else:
                tr, te = ctrl[ctrl.col < GRID_COLS // 2], ctrl[ctrl.col >= GRID_COLS // 2]
            m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
            m.fit(tr[feats], tr[target])
            pred = m.predict(te[feats])
            r2s.append(r2_score(te[target], pred))
            rmses.append(np.sqrt(mean_squared_error(te[target], pred)))
        return float(np.mean(r2s)), float(np.mean(rmses))

    r2_sp, rmse_sp = spatial_block_cv(feat_spatial)
    r2_ns, rmse_ns = spatial_block_cv(feat_nonspatial)

    print("\n=== [2층 AI] 공간 명시적 ML 벌채위험 예측 (2겹 공간 블록 CV, 통제구역 학습) ===")
    print(f"공간 피처 포함 : R²={r2_sp:.3f}, RMSE={rmse_sp:.3f}")
    print(f"공간 피처 제외 : R²={r2_ns:.3f}, RMSE={rmse_ns:.3f}")
    gain = r2_sp - r2_ns
    verdict = "공간 ML 정당" if gain >= 0.03 else ("약한 보조 신호" if gain > 0 else "공간 효과 없음→단순모델")
    print(f"→ 공간 시차(미관측 공간효과) 기여: R² {gain:+.3f} ({verdict})")

    # 통제구역으로 학습 → 모든 단위에 '보호 없을 때 예상 손실' 예측 + 설명가능성
    m_full = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
    m_full.fit(ctrl[feat_spatial], ctrl[target])
    p["ml_risk"] = m_full.predict(p[feat_spatial])
    imp = sorted(zip(feat_spatial, m_full.feature_importances_), key=lambda x: -x[1])
    print("피처 중요도(설명가능성):", ", ".join(f"{k}={v:.3f}" for k, v in imp))
    print("주의(규칙4): DID는 '관측' 손실로 추정한다. ML 예측치를 인과추정의 "
          "결과변수로 쓰면 평균수축→효과 과소추정이 생기므로 별도 보정이 필요하다.")

    # -------- [불확실성] 정규화 conformal 예측구간·신뢰등급(OOF 근사) --------
    # ml_risk 점추정만으로 보전 예산을 배분하면 '얼마나 틀릴지 모른 채 확신하는'(7.5절)
    # 위험이 있다. conformal 예측(Lei et al. 2018)은 분포가정 없이 목표 포함확률을 겨냥한
    # 예측구간을 주고, 지역 난이도로 정규화(Romano et al. 2019 계열)하면 이질적 불확실성을
    # 반영한다. 단, 아래는 엄밀한 분할 conformal(학습/보정/시험 3분할)이 아니라 공간 블록
    # CV의 out-of-fold(OOF) 잔차를 σ̂ 적합과 보정에 함께 쓰는 '교육용 근사'다(포함률 다소 낙관적).
    cal = ctrl.reset_index(drop=True)                # 통제구역만 보정표본(정책효과 미오염)
    oof = np.full(len(cal), np.nan)
    for test_left in (True, False):                  # 기존 2겹 좌/우 공간 블록 재사용(누수 없는 OOF)
        te = (cal.col < GRID_COLS // 2) if test_left else (cal.col >= GRID_COLS // 2)
        tr = ~te
        m_oof = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
        m_oof.fit(cal.loc[tr, feat_spatial], cal.loc[tr, target])
        oof[te.values] = m_oof.predict(cal.loc[te, feat_spatial])
    resid = np.abs(cal[target].values - oof)
    n_cal = len(resid)
    # σ̂(x): OOF 절대잔차를 특징으로 회귀(지역 난이도 = 이질적 불확실성 추정)
    sig_model = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=1)
    sig_model.fit(cal[feat_spatial], resid)
    sigma_cal = np.clip(sig_model.predict(cal[feat_spatial]), 1e-3, None)
    # 정규화 conformal 점수의 k번째 순서통계(k=⌈(n+1)(1-α)⌉, 분할 conformal 관례)
    alpha = 0.10
    scores = np.sort(resid / sigma_cal)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = scores[min(k, n_cal) - 1]                    # k>n이면 최댓값(구간 발산 회피)
    cover = float(np.mean(resid <= q * sigma_cal))   # OOF 경험적 포함률(≈0.90 목표)
    # 전체 격자에 구간 반폭·신뢰등급 부여(σ̂를 모든 단위에 적용)
    sigma_all = np.clip(sig_model.predict(p[feat_spatial]), 1e-3, None)
    half = q * sigma_all                             # 단위별 예측구간 반폭(이질적)
    t1, t2 = np.quantile(half, [1 / 3, 2 / 3])       # 신뢰등급: 반폭 3분위(좁을수록 신뢰↑)
    conf = np.where(half <= t1, "높음", np.where(half <= t2, "중간", "낮음"))
    p["pi_half"] = half
    p["confidence"] = conf
    print("\n=== [불확실성] 정규화 conformal 예측구간(90%, OOF 근사)과 신뢰등급 ===")
    print(f"통제구역 보정표본 n={n_cal} · 목표 포함확률 0.90 conformal 계열(교육용 OOF 근사)")
    print(f"OOF 경험적 포함률 = {cover:.3f} (목표 0.90)")
    print(f"구간 반폭(%p): 중앙값 {np.median(half):.3f}, 범위 [{half.min():.3f}, {half.max():.3f}]")
    print(f"신뢰등급 분포: 높음 {(conf=='높음').sum()} · 중간 {(conf=='중간').sum()} · "
          f"낮음 {(conf=='낮음').sum()}")
    print("주의: σ̂를 같은 OOF 잔차로 적합 → 포함률은 다소 낙관적이다. 엄밀한 실무는")
    print("      학습/보정/시험 3분할 split conformal을 쓴다(7.5절·보론).")
    return p[["unit", "ml_risk", "pi_half", "confidence"]]


# ===========================================================
# 6. 메인
# ===========================================================
def main():
    print("보호구역 정책의 산림손실 저감 효과 분석 (교육용 시뮬레이션)")
    print(f"단위 {N_UNITS}개(보호 {N_PROTECTED}) · {YEARS[0]}~{YEARS[-1]} · 지정 {TREAT_YEAR}")
    print(f"진짜 정책효과(설정값): {TRUE_EFFECT:+.2f} %p\n")

    df = load_panel()

    basic_comparison(df)
    naive = (df[(df.post == 1) & (df.treated == 1)]["loss"].mean()
             - df[(df.post == 1) & (df.treated == 0)]["loss"].mean())

    did_coef, _, _ = did_estimate(df)
    m_coef, _, _ = matched_did(df)

    plot_trends(df)
    unit_panel, _district = spatial_policy_outputs(df, did_coef)

    # [2층 AI] 공간 명시적 ML 위험 예측 → 정책 우선순위에 반영
    risk = predict_deforestation_risk(unit_panel)
    merged = unit_panel.merge(risk, on="unit")
    merged["priority_rank_ml"] = merged["ml_risk"].rank(
        ascending=False, method="first").astype(int)
    merged.to_csv(RESULTS_DIR / "spatial_unit_priority.csv", index=False)

    # 가중합 우선순위 vs ML 위험 우선순위 비교 (변화는 '영향'이지 '우월성'이 아님)
    top_w = set(merged.nsmallest(30, "priority_rank")["unit"])
    top_ml = set(merged.nsmallest(30, "priority_rank_ml")["unit"])
    overlap = len(top_w & top_ml)
    print(f"\n가중합 상위30 vs ML위험 상위30 일치: {overlap}/30 "
          f"→ {30 - overlap}곳 차이 (변화 자체는 영향의 증거일 뿐, 우월성은 아래 근거로 판단)")
    # ML에는 새로 들어오고 가중합에는 없던 격자 = ML이 추가로 지목한 곳
    new_in_ml = merged[merged["unit"].isin(top_ml - top_w)].copy()
    cols = ["unit", "admin_area", "post_loss", "pressure", "habitat_value", "treated", "ml_risk"]
    print("ML이 새로 상위30에 올린 격자(검증용 — 손실·압력·서식지 근거 확인):")
    print(new_in_ml.sort_values("ml_risk", ascending=False)[cols].head(6).round(3).to_string(index=False))

    # [불확실성] ml_risk 예측의 conformal 구간을 행정구역 우선순위표에 열로 추가
    unc = (
        merged.groupby("admin_area")
        .agg(
            mean_ml_risk=("ml_risk", "mean"),
            mean_pi_half=("pi_half", "mean"),
            low_conf_units=("confidence", lambda s: int((s == "낮음").sum())),
            n_units=("unit", "count"),
        )
        .reset_index()
        .sort_values("mean_ml_risk", ascending=False)
        .reset_index(drop=True)
    )
    a1, a2 = np.quantile(unc["mean_pi_half"], [1 / 3, 2 / 3])
    unc["confidence"] = np.where(
        unc["mean_pi_half"] <= a1, "높음",
        np.where(unc["mean_pi_half"] <= a2, "중간", "낮음"))
    unc["priority_rank"] = range(1, len(unc) + 1)
    unc.to_csv(RESULTS_DIR / "admin_area_priority_uncertainty.csv", index=False)
    print("\n=== [불확실성] ml_risk 기준 행정구역 우선순위(구간반폭·신뢰등급) ===")
    cols_u = ["priority_rank", "admin_area", "mean_ml_risk",
              "mean_pi_half", "low_conf_units", "confidence"]
    print(unc[cols_u].head(6).round(3).to_string(index=False))

    # 요약: 추정 방법별 정책효과 vs 진짜값
    print(f"\n=== 추정 방법 비교 (진짜 효과 = {TRUE_EFFECT:.2f} %p) ===")
    print(f"{'방법':<22}{'추정 효과(%p)':>14}{'진짜값과 오차':>16}")
    for name, est in [("naive(단순 비교)", naive),
                      ("DID(전체)", did_coef),
                      ("DID(매칭)", m_coef)]:
        print(f"{name:<22}{est:>14.3f}{est - TRUE_EFFECT:>+16.3f}")
    print("\n해석: naive는 selection bias로 효과를 과대(더 음수)추정한다.")
    print("DID는 단위 고정효과로 시간불변 차이를 제거해 진짜값에 근접한다.")


if __name__ == "__main__":
    main()
