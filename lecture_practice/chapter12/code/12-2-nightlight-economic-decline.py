"""
13장 실습 2: 야간조명 경제 프록시와 지역 쇠퇴 예측 (2층 AI 정당 + 측정오차 보정)
==============================================================================
질문: "위성 야간조명으로 지역 경제 쇠퇴를 예측할 수 있는가? 거친 프록시를 그대로 써도 되는가?"

3층 모델:
  [1층 GIS]  시군구별 경제·인구 지표 집계
  [2층 AI ]  야간조명(위성 경제 프록시, Henderson 2012) + 공간 명시적 ML 쇠퇴 예측 +
             측정오차 편의 보정(규칙4) + 공간 피처 유무 성능 비교(=AI 정당성)
  [3층 정책] 시군구 쇠퇴위험 우선순위(투자 배분)

세 가지 교육 포인트:
  (1) 프록시 정당성: 야간조명이 (관측 불가) 진짜 경제활동과 강하게 상관 → 위성 프록시 정당.
  (2) 측정오차 편의(규칙4): 거친 야간조명을 예측변수로 쓰면 회귀계수가 0쪽으로 수축
      (attenuation). 신뢰도 비율로 보정해야 진짜 관계에 근접한다.
  (3) FM buzzword 경고(규칙2): 거친 집계 쇠퇴는 행정지표+야간조명으로 대부분 설명된다.
      고차원 임베딩(FM)은 '신호를 실제로 더한다'가 입증될 때만 정당하다(여기선 시연 안 함).

데이터: 미리 준비한 교육용 시뮬레이션 데이터를 불러와 사용(개인 식별 불가, 시군구).
  분석·추정은 실제 계산값.
실행:
    python 13-0-simdata-prep.py                # 최초 1회: 데이터 준비
    python 13-2-nightlight-economic-decline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRID_ROWS, GRID_COLS = 20, 15
N = GRID_ROWS * GRID_COLS


def _neighbor_mean(rows, cols, values):
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid[rows, cols] = values
    out = np.zeros(N)
    for i, (r, c) in enumerate(zip(rows, cols)):
        neigh = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                neigh.append(grid[rr, cc])
        out[i] = np.nanmean(neigh)
    return out


def load_regions():
    """미리 준비한 야간조명·행정지표·미래 쇠퇴 데이터를 불러온다.

    데이터 설계: 관측되지 않는 진짜 경제활동에서 야간조명(측정오차 포함, 거친 프록시)과
    행정지표, 그리고 인접 spillover를 반영한 미래 쇠퇴율을 만든 교육용 시뮬레이션이다.
    분석은 이 데이터로 프록시 정당성·attenuation 보정·공간 예측을 시연한다.
    """
    path = DATA_DIR / "econ_regions.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 13-0-simdata-prep.py")
    return pd.read_parquet(path)


def std_coef(y, x):
    """표준화 단순회귀 계수(=상관계수). 측정오차 attenuation 비교용."""
    ys = (y - y.mean()) / y.std()
    xs = (x - x.mean()) / x.std()
    return float(np.mean(ys * xs))


def spatial_block_cv_r2(df, feat_cols, target="decline", n_blocks=5):
    block = (df["col"].values / (GRID_COLS / n_blocks)).astype(int)
    block = np.clip(block, 0, n_blocks - 1)
    preds = np.full(len(df), np.nan)
    X, y = df[feat_cols].values, df[target].values
    for b in range(n_blocks):
        te = block == b
        m = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(X[~te], y[~te])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds)


def main():
    print("=" * 64)
    print("야간조명 경제 프록시와 지역 쇠퇴 예측 (13-2, 2층 AI + 규칙4)")
    print("=" * 64)
    df = load_regions()
    print(f"\n분석 단위: {N}개 시군구(20×15), 개인 식별 불가 집계")

    # -------- (1) 프록시 정당성 --------
    print("\n[프록시 검증] 야간조명 ↔ (관측 불가) 진짜 경제활동")
    corr = np.corrcoef(df["nightlight"], df["true_econ"])[0, 1]
    reliability = corr ** 2     # 공유분산(참분산 비율) = corr²
    print(f"  상관 r = {corr:.3f} (공유분산 r² = {reliability:.3f})")
    print(f"  → 야간조명은 경제활동의 유용한 위성 프록시(Henderson 2012). 단, 측정오차 존재.")

    # -------- (2) 측정오차 편의(규칙4) --------
    print("\n[규칙4] 거친 프록시를 예측변수로 쓸 때 측정오차 편의(attenuation)")
    coef_true = std_coef(df["decline"].values, df["true_econ"].values)
    coef_light = std_coef(df["decline"].values, df["nightlight"].values)
    # 표준화 계수(=상관)의 attenuation 인자는 corr(nl,true)=√reliability → corr로 나눠 보정
    corrected = coef_light / corr
    print(f"  쇠퇴~진짜경제 표준화계수(오라클)   = {coef_true:+.3f}")
    print(f"  쇠퇴~야간조명 표준화계수(관측)     = {coef_light:+.3f}  "
          f"(attenuation {(1-abs(coef_light)/abs(coef_true))*100:4.1f}% 0쪽 수축)")
    print(f"  attenuation 보정 후(÷ r)            = {corrected:+.3f}  (진짜계수 복원)")
    print("  → 야간조명을 결과/예측변수로 쓰면 효과를 과소추정. 보정 또는 복수지표 필요.")

    # -------- (3) 공간 명시적 ML 쇠퇴 예측 --------
    print("\n[2층 AI] 쇠퇴 예측: 공간 피처 유무 비교 (공간 블록 CV)")
    df["nl_lag"] = _neighbor_mean(df["row"].values, df["col"].values, df["nightlight"].values)
    df["decline_lag"] = _neighbor_mean(df["row"].values, df["col"].values, df["decline"].values)
    base = ["nightlight", "aging", "biz"]
    spat = base + ["nl_lag", "decline_lag"]
    r2_base = spatial_block_cv_r2(df, base)
    r2_spat = spatial_block_cv_r2(df, spat)
    print(f"  비공간 피처(야간조명+행정지표)  R² = {r2_base:6.3f}")
    print(f"  +공간 시차 피처                 R² = {r2_spat:6.3f}  (개선 {r2_spat-r2_base:+.3f})")
    if r2_spat - r2_base > 0.02:
        print("  → 공간 시차가 쇠퇴 예측을 개선 = 공간 명시적 AI 정당화(규칙1·2).")
    else:
        print("  → 공간 피처 개선 미미 → 저비용 모델로 충분.")
    print("  ※ FM 경고(규칙2): 거친 집계 쇠퇴는 행정지표+야간조명으로 대부분 설명된다.")
    print("    고차원 임베딩(AlphaEarth류)은 '신호를 실제로 더한다'가 입증될 때만 정당.")

    full = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                     max_depth=3, learning_rate=0.05).fit(
        df[spat].values, df["decline"].values)
    imp = pd.Series(full.feature_importances_, index=spat).sort_values(ascending=False)
    print("\n  피처 중요도(설명가능성):")
    for name, val in imp.items():
        print(f"    {name:12s} {val:.3f}")

    # -------- [3층 정책] 쇠퇴위험 우선순위 --------
    print("\n[3층 정책] 시군구 쇠퇴위험 우선순위(투자 배분 보조)")
    df["pred_decline"] = full.predict(df[spat].values)
    df["priority_rank"] = df["pred_decline"].rank(ascending=False).astype(int)
    top = (df.sort_values("pred_decline", ascending=False)
             [["region_id", "nightlight", "aging", "biz", "pred_decline", "priority_rank"]]
             .head(10).round(3))
    print(top.to_string(index=False))

    csv_path = RESULTS_DIR / "econ_decline_priority.csv"
    df.sort_values("pred_decline", ascending=False).round(4).to_csv(csv_path, index=False)
    print(f"\n  쇠퇴위험 우선순위 저장 → {csv_path.name}")

    # -------- [불확실성] 정규화 conformal 예측구간·신뢰등급(OOF 근사) --------
    # 우선순위 점추정만으로 투자를 배분하면 '얼마나 틀릴지 모른 채 확신하는'(7.5절) 위험이 있다.
    # conformal 예측(Lei et al. 2018)은 분포가정 없이 예측구간을 주고, 지역 난이도로
    # 정규화(Romano et al. 2019 계열)하면 이질적 불확실성을 반영한다. 이는 앞의 측정오차
    # 보정(규칙4)과는 '별개의 층'이다. 규칙4는 야간조명 프록시의 측정오차가 계수를 0쪽으로
    # 수축시키는 편의를 바로잡는 것이고, conformal은 (보정 여부와 무관하게) 쇠퇴 예측 자체가
    # 얼마나 불확실한가를 정량화한다. 단, 아래는 엄밀한 분할 conformal(학습/보정/시험 3분할)이
    # 아니라 공간 블록 CV의 OOF 잔차를 σ̂ 적합과 캘리브레이션에 함께 쓰는 '교육용 근사'다
    # (포함률은 다소 낙관적).
    print("\n[불확실성] 정규화 conformal 예측구간(90%, OOF 근사)과 신뢰등급")
    X_all = df[spat].values
    y_all = df["decline"].values
    # 1) 공간 블록 CV의 out-of-fold(OOF) 예측 → 누수 없는 정직한 잔차
    n_blocks = 5
    block = (df["col"].values / (GRID_COLS / n_blocks)).astype(int)
    block = np.clip(block, 0, n_blocks - 1)
    oof = np.full(N, np.nan)
    for b in range(n_blocks):
        te = block == b
        m = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3,
                                      learning_rate=0.05).fit(X_all[~te], y_all[~te])
        oof[te] = m.predict(X_all[te])
    resid = np.abs(y_all - oof)
    # 2) 지역 난이도 σ̂(x): OOF 절대잔차를 특징으로 회귀(이질적 불확실성 추정)
    sigma = np.clip(GradientBoostingRegressor(random_state=1, n_estimators=150, max_depth=3,
                    learning_rate=0.05).fit(X_all, resid).predict(X_all), 1e-3, None)
    # 3) 정규화 conformal 점수의 k번째 순서통계(k=⌈(n+1)(1-α)⌉, 분할 conformal 관례)
    alpha = 0.10
    scores = np.sort(resid / sigma)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    q = scores[min(k, N) - 1]                       # k>N이면 최댓값(구간 무한대 회피)
    half = q * sigma                                # 단위별 예측구간 반폭(이질적)
    cover = float(np.mean(resid <= half))           # OOF 경험적 포함률(≈0.90 목표)
    df["pi_half"] = half
    df["pi_low"] = df["pred_decline"] - half
    df["pi_high"] = df["pred_decline"] + half
    t1, t2 = np.quantile(half, [1 / 3, 2 / 3])      # 신뢰등급: 반폭 3분위(좁을수록 신뢰↑)
    df["confidence"] = np.where(half <= t1, "높음", np.where(half <= t2, "중간", "낮음"))
    n_low = int((df["confidence"] == "낮음").sum())
    print(f"  90% 예측구간 경험적 포함률(OOF) = {cover:.3f}  (목표 0.90)")
    print(f"  구간 반폭(쇠퇴율 %p): 중앙값 {np.median(half):.2f}, "
          f"범위 [{half.min():.2f}, {half.max():.2f}]")
    print(f"  신뢰등급 '낮음' 시군구 = {n_low}개")
    print("  주의: σ̂를 같은 OOF 잔차로 적합했으므로 포함률은 다소 낙관적이다.")
    print("        엄밀한 실무는 학습/보정/시험 3분할 split conformal을 쓴다(7.5절·보론).")

    top_unc = (df.sort_values("pred_decline", ascending=False)
                 [["region_id", "pred_decline", "priority_rank",
                   "pi_half", "pi_low", "pi_high", "confidence"]]
                 .head(10).round(3))
    print("\n  불확실성을 반영한 시군구 쇠퇴위험 우선순위(상위 10):")
    print(top_unc.to_string(index=False))
    print("  → 예측 쇠퇴율이 높아도 신뢰등급이 '낮음'이면, 투자 확정 전에 추가 관측·현장확인으로")
    print("     불확실성을 줄인다(점추정 맹신 방지).")

    print("\n[완료] 야간조명 경제 프록시·쇠퇴 예측 분석을 마쳤다.")


if __name__ == "__main__":
    main()
