"""
11장 실습 3: 건물별 홍수 리스크 등급과 보험료 차등 (비즈니스)
=================================================================
비즈니스 질문: "이 건물의 홍수 리스크 등급은 몇 등급이고, 보험료를 얼마나 차등해야 하는가?
               보험 가입 편향이 등급 산정을 어떻게 왜곡하는가?"

이 분석은 11-1(침수위험 공공 분석)의 비즈니스 대응물이다. 같은 침수 위험 모델의
산출물이 공공에서는 "대응 우선순위표"가 되고 비즈니스에서는 "보험료 등급표"가 된다.
핵심 메시지 세 가지:
  (1) 같은 위험 모델, 다른 가격표 — 산출물이 자원 배분에서 가격으로 전환
  (2) 불확실성의 비즈니스 번역 — conformal 예측구간이 "현장확인 우선"에서
      "보험료 로딩(할증)"으로
  (3) 라벨 편향의 비즈니스 버전 — 보험 가입자만 청구 기록을 남기므로 미가입 지역
      위험이 과소평가된다(11-2의 '순찰이 적발 기록을 만드는 것'과 구조적으로 같다)

데이터: 11-0에서 생성한 insurance_buildings.parquet + flood_terrain.parquet
실행:
    python 11-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 11-3-insurance-risk-grading.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 리스크 등급과 보험료 배수 ──
RISK_TIERS = {1: "최고위험", 2: "고위험", 3: "중위험", 4: "저위험"}
PREMIUM_MULT = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0}
# 불확실성 로딩: 예측구간 반폭이 클수록 보험료에 안전마진(+α) 추가
UNCERTAINTY_LOADING = 0.15  # 저신뢰 건물에 적용할 추가 배수


def load_data():
    """건물 + 격자 데이터를 불러와 결합한다."""
    bld = pd.read_parquet(DATA_DIR / "insurance_buildings.parquet")
    flood = pd.read_parquet(DATA_DIR / "flood_terrain.parquet")

    # 격자 수준 피처를 건물에 연결 (11-1 결과 재사용 논리)
    grid_feats = flood[["cell", "elevation", "slope", "river_dist",
                        "imperv", "rainfall", "flood_depth"]].rename(
        columns={"cell": "grid_id", "flood_depth": "grid_flood_depth_full"})
    bld = bld.merge(grid_feats[["grid_id", "slope", "river_dist", "imperv", "rainfall"]],
                    on="grid_id", how="left", suffixes=("", "_grid"))
    return bld


def assign_risk_tier(prob):
    """침수 확률을 리스크 등급으로 분류한다.
    25%ile, 50%ile, 75%ile 기준으로 4등급 분류(등급 1이 최고위험).
    """
    q = np.nanpercentile(prob[~np.isnan(prob)], [75, 50, 25])
    tiers = np.full(len(prob), 4, dtype=int)
    tiers[prob >= q[0]] = 1
    tiers[(prob >= q[1]) & (prob < q[0])] = 2
    tiers[(prob >= q[2]) & (prob < q[1])] = 3
    return tiers


def build_risk_model(X, y, model_name="model"):
    """그래디언트 부스팅 회귀로 침수 확률을 예측하고 5-fold CV R²를 반환한다."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        mdl = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, subsample=0.8)
        mdl.fit(X[train_idx], y[train_idx])
        preds[test_idx] = mdl.predict(X[test_idx])
    r2 = r2_score(y, preds)
    print(f"  {model_name} CV R²: {r2:.3f}")
    return preds, r2


def main():
    print("=" * 60)
    print("비즈니스 분석: 건물별 홍수 리스크 등급과 보험료 차등")
    print("=" * 60)

    bld = load_data()
    n_total = len(bld)

    # ── 피처 구성 ──
    # 구조 유형을 수치로 인코딩 (RC=0, steel=1, wood=2)
    struct_map = {"RC": 0, "steel": 1, "wood": 2}
    bld["struct_code"] = bld["structure_type"].map(struct_map)

    feature_cols = ["elevation", "slope", "river_dist", "imperv", "rainfall",
                    "floor_type", "building_age", "struct_code"]
    X_all = bld[feature_cols].values

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (a) 완전 모델: 진짜 침수확률로 학습 (채점용 — 현실에는 없는 "신의 모델")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    y_true = bld["true_flood_prob"].values
    pred_full, r2_full = build_risk_model(X_all, y_true, "완전 모델(전체)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (b) 관측 모델: 보험 가입 건물의 청구 기록만으로 학습
    #     (보험사가 실제로 만드는 모델 — 미가입 건물은 라벨이 없다)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    insured_mask = bld["insured"].values.astype(bool)
    y_claim = bld.loc[insured_mask, "claim_observed"].values.astype(float)
    X_insured = X_all[insured_mask]

    # 가입자만으로 학습한 모델
    # 주의: 아래 R²는 '가입자 277채의 청구 기록'을 맞히는 OOF 점수다.
    # 표 11.6의 관측 모델 R²(500채의 진짜 침수확률 대상)와는 채점 대상이 다르므로
    # 두 값을 같은 지표로 읽으면 안 된다.
    pred_obs_insured, r2_obs_insured = build_risk_model(
        X_insured, y_claim, "관측 모델(가입자 277채, 대상=청구 기록)")

    # 관측 모델을 전체 건물에 적용 (전이)
    mdl_obs = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, subsample=0.8)
    mdl_obs.fit(X_insured, y_claim)
    pred_obs_all = mdl_obs.predict(X_all)

    # 완전 모델의 예측은 in-sample이 아니라 교차검증 예측(OOF)을 쓴다.
    # 같은 데이터로 학습하고 같은 데이터로 채점하면 R²가 부풀려진다 — 분석 2에서
    # "자기가 쓴 답안지를 자기가 채점한다"고 비판한 바로 그 문제다.
    pred_full_all = pred_full

    # ── 대조군(null control): 선택편향의 몫을 분리한다 ──
    # 관측 모델의 성능 저하가 '보험 가입 선택' 때문인지, 아니면 표본이 절반으로
    # 줄고 라벨이 청구 기록이라서인지 구분해야 한다. 가입 여부와 무관하게 같은
    # 크기로 무작위 추출해 같은 절차를 반복하면, 선택 이외의 요인만 남는다.
    n_ins = int(insured_mask.sum())
    rng = np.random.default_rng(2024)
    null_r2 = []
    for rep in range(20):
        idx = rng.choice(len(bld), size=n_ins, replace=False)
        y_rep = bld["claim_recorded_latent"].values[idx]
        mdl_rep = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, subsample=0.8)
        mdl_rep.fit(X_all[idx], y_rep)
        null_r2.append(r2_score(y_true, mdl_rep.predict(X_all)))
    null_r2_mean, null_r2_sd = float(np.mean(null_r2)), float(np.std(null_r2))

    # ── 리스크 등급 산정 ──
    tiers_full = assign_risk_tier(pred_full_all)
    tiers_obs = assign_risk_tier(pred_obs_all)

    bld["tier_full"] = tiers_full
    bld["tier_obs"] = tiers_obs
    bld["pred_full"] = pred_full_all
    bld["pred_obs"] = pred_obs_all

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 표 11.5: 관측 모델 리스크 등급별 보험료 차등과 편향
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "─" * 60)
    print("표 11.5  관측 모델 리스크 등급별 보험료 차등과 편향")
    print("─" * 60)
    print(f"{'등급':>4} {'등급명':>8} {'건물수':>6} {'평균예측침수확률':>14} "
          f"{'보험료배수':>8} {'저소득비율':>8} {'미가입비율':>8}")
    print("─" * 60)

    for tier in sorted(RISK_TIERS.keys()):
        mask = bld["tier_obs"] == tier
        n = mask.sum()
        avg_pred = bld.loc[mask, "pred_obs"].mean()
        premium = PREMIUM_MULT[tier]
        low_inc_ratio = bld.loc[mask, "low_income"].mean()
        unins_ratio = (~bld.loc[mask, "insured"]).mean()
        print(f"{tier:>4} {RISK_TIERS[tier]:>8} {n:>6} {avg_pred:>14.3f} "
              f"{premium:>7.1f}x {low_inc_ratio:>8.1%} {unins_ratio:>8.1%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 표 11.6: 완전 모델 vs 관측 모델의 편향
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    r2_full_all = r2_score(y_true, pred_full_all)
    r2_obs_all = r2_score(y_true, pred_obs_all)

    low_mask = bld["low_income"].values
    r2_full_low = r2_score(y_true[low_mask], pred_full_all[low_mask])
    r2_obs_low = r2_score(y_true[low_mask], pred_obs_all[low_mask])

    # 위험 과소평가: 관측 모델이 진짜 확률보다 10%p 이상 낮게 예측한 건물
    underest = (y_true - pred_obs_all) > 0.10
    underest_full = (y_true - pred_full_all) > 0.10

    print("\n" + "─" * 60)
    print("표 11.6  완전 모델 vs 관측 모델의 편향 (전부 교차검증 예측 기준)")
    print("─" * 60)
    print(f"{'지표':>24} {'완전 모델':>10} {'관측 모델':>10}")
    print("─" * 60)
    print(f"{'전체 R²':>24} {r2_full_all:>10.3f} {r2_obs_all:>10.3f}")
    print(f"{'저소득 건물 R²':>24} {r2_full_low:>10.3f} {r2_obs_low:>10.3f}")
    print(f"{'위험 과소평가 비율':>24} {underest_full.mean():>10.1%} {underest.mean():>10.1%}")
    print("─" * 60)
    print(f"  [대조군] 가입 선택을 없앤 동일크기 무작위 표본 R²: "
          f"{null_r2_mean:.3f} (SD {null_r2_sd:.3f}, 20회)")
    print(f"  → 관측 모델({r2_obs_all:.3f})과의 차이 "
          f"{null_r2_mean - r2_obs_all:+.3f}가 '가입 선택' 고유의 몫이다.")
    print(f"  → 완전 모델({r2_full_all:.3f})과 대조군의 차이 "
          f"{r2_full_all - null_r2_mean:+.3f}는 선택이 아니라 라벨(청구 기록)과")
    print(f"     표본 축소에서 온다. 두 몫을 섞어 읽지 않는다.")

    # 저소득 미신고가 만든 라벨 왜곡의 크기
    n_under = int(bld["underreported"].values.sum())
    n_under_ins = int((bld["underreported"].values & insured_mask).sum())
    print(f"\n라벨 왜곡(소액 피해 미신고): 전체 {n_under}건, "
          f"그중 가입자 {n_under_ins}건이 학습에 0으로 들어갔다.")

    # ── 등급 불일치 분석 ──
    mismatch = tiers_full != tiers_obs
    print(f"\n등급 불일치 건물 비율: {mismatch.mean():.1%}")
    # 저소득 건물의 등급 불일치
    mismatch_low = mismatch[low_mask].mean()
    mismatch_high = mismatch[~low_mask].mean()
    print(f"  저소득 건물 등급 불일치: {mismatch_low:.1%}")
    print(f"  고소득 건물 등급 불일치: {mismatch_high:.1%}")

    # 저소득 건물이 관측 모델에서 과소평가되는 방향 확인
    # 관측 등급이 완전 등급보다 낮음(=더 안전하다고 평가) = 위험 과소평가
    undergraded_low = ((tiers_obs > tiers_full) & low_mask).sum()
    overgraded_low = ((tiers_obs < tiers_full) & low_mask).sum()
    print(f"  저소득: 과소평가(덜 위험하다고 판단) {undergraded_low}건, "
          f"과대평가 {overgraded_low}건")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 불확실성 로딩: 예측구간 반폭에 따른 보험료 할증
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 정규화 split conformal — 분석 1(11-1)과 같은 방식이다.
    # 가입자의 out-of-sample 잔차를 국소 규모로 정규화한 뒤, 그 분위수로 구간을
    # 만든다. 예측값의 단조함수로 구간 폭을 '정하는' 것이 아니라, 실제 잔차가
    # 구간 폭을 결정한다. (이전 구현은 σ를 예측값의 단조식으로 두어, "저신뢰
    # 건물이 곧 최고위험 건물"이 결과가 아니라 정의상 참이 되는 문제가 있었다.)
    residuals = np.abs(y_claim - pred_obs_insured)

    # 국소 규모 추정: 잔차 크기를 같은 피처로 회귀해 이분산을 학습한다.
    scale_mdl = GradientBoostingRegressor(
        n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42)
    scale_mdl.fit(X_insured, residuals)
    scale_insured = np.clip(scale_mdl.predict(X_insured), 1e-3, None)
    scale_all = np.clip(scale_mdl.predict(X_all), 1e-3, None)

    # 보정집합의 정규화 부적합 점수 → 90% 분위수
    nonconf = residuals / scale_insured
    n_cal = len(nonconf)
    q_level = min(np.ceil((n_cal + 1) * 0.90) / n_cal, 1.0)
    q_hat = float(np.quantile(nonconf, q_level))
    ci_half = q_hat * scale_all

    # 포함률 점검(가입자 기준) — conformal이 목표 수준을 지키는지 확인
    coverage = float(np.mean(residuals <= q_hat * scale_insured))

    # 저신뢰(CI 반폭 상위 25%) 건물에 로딩 적용
    ci_threshold = np.percentile(ci_half, 75)
    low_confidence = ci_half >= ci_threshold
    # 로딩은 배수에 곱한다(가산이 아니다). 3.0배 × 1.15 = 3.45배 → 실제로 +15%.
    base_premium = np.array([PREMIUM_MULT[t] for t in tiers_obs])
    final_premium = np.where(low_confidence,
                             base_premium * (1 + UNCERTAINTY_LOADING),
                             base_premium)

    # σ와 예측값이 사실상 같은 순위인지 점검(항등식 여부 자가진단)
    from scipy.stats import spearmanr
    rho = float(spearmanr(ci_half, pred_obs_all).statistic)

    print(f"\n불확실성(정규화 conformal):")
    print(f"  q̂={q_hat:.3f}, 가입자 기준 포함률: {coverage:.3f} (목표 0.90)")
    print(f"  CI 반폭 중앙값: {np.median(ci_half):.3f}, "
          f"저신뢰 기준(상위 25%): {ci_threshold:.3f}")
    print(f"  구간폭-예측값 Spearman: {rho:.3f} "
          f"(1.000이면 '저신뢰=고위험'이 항등식)")
    print(f"  저신뢰 건물 수: {low_confidence.sum()}개 "
          f"(보험료 ×{1 + UNCERTAINTY_LOADING:.2f} 적용)")
    print(f"  저신뢰 건물의 최고위험 등급 비율: "
          f"{(tiers_obs[low_confidence] == 1).mean():.1%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공공-비즈니스 대비 요약
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "─" * 60)
    print("공공 vs 비즈니스 대비")
    print("─" * 60)
    print(f"{'차원':>16} {'공공(재난 대비)':>18} {'비즈니스(보험)':>18}")
    print("─" * 60)
    print(f"{'위험지도 용도':>16} {'구조장비·대피소 배치':>18} {'보험료·자산가치':>18}")
    print(f"{'불확실성 대응':>16} {'현장확인 우선배정':>18} {'보험료 로딩(할증)':>18}")
    print(f"{'라벨 편향 유형':>16} {'순찰→적발 편향':>18} {'가입→청구 편향':>18}")
    print(f"{'윤리 쟁점':>16} {'감시·차별적 단속':>18} {'가격차별·레드라이닝':>18}")

    # ── 보험 가입률과 리스크의 역설 ──
    # 실제 고위험(진짜 확률 상위 25%)인데 미가입인 건물
    true_high_risk = y_true >= np.percentile(y_true, 75)
    high_risk_uninsured = true_high_risk & ~insured_mask
    print(f"\n보험의 역설:")
    print(f"  진짜 고위험 건물 중 미가입: "
          f"{high_risk_uninsured.sum()}/{true_high_risk.sum()} "
          f"({high_risk_uninsured.sum()/true_high_risk.sum():.1%})")
    print(f"  이 건물들은 관측 모델 학습에 포함되지 않아 위험이 과소평가된다.")
    print(f"  → 보험 상품이 설계되지 않고 → 가입 불가 → 기록 부재 → 과소평가 지속")
    print(f"  (11-2 '순찰이 없으면 안전하다고 오학습'과 같은 되먹임)")

    print("\n" + "=" * 60)
    print("[완료] 비즈니스 분석: 건물별 홍수 리스크 등급과 보험료 차등")
    print("=" * 60)


if __name__ == "__main__":
    main()
