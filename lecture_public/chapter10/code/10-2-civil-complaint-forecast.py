"""
10장 실습 2: 민원 발생 시공간 예측과 현장대응 우선순위 (2층 AI 하중)
=====================================================================
도시·행정 정책 질문: "다음 주 행정수요(민원)는 어디서 발생할까? 현장 인력·장비를 어디에 먼저 배치할까?"

3층 모델:
  [1층 GIS]  행정동·격자별 민원(도로파임·불법주정차·쓰레기 등) 집계 + 핫스팟
  [2층 AI ]  공간 시차(이웃 민원) + 시간 시차(과거 민원) 피처를 내장한 다음 주 예측 +
             시간 블록 CV + 시공간 피처 유무 성능 비교(=AI 정당성) + 피처 중요도(=설명가능성)
  [3층 정책] 다음 주 행정동(권역)별 현장대응 자원 배치 우선순위 CSV

핵심 설계(공간 명시성 + AI 정당성):
  - 민원은 인접 구간으로 번지고(공간 확산: 도로 파손·무단투기·소음의 권역성),
    과거에 의존(시간 지속성)하는 '비선형 시공간 과정'으로 발생하도록 생성한다.
  - 시간 시차만 쓴 모델 vs 공간 시차를 더한 모델의 다음 주 예측력을 비교한다.
    공간 피처가 유의하게 개선하면 = 2층 AI(공간 명시적)가 정당화된다(규칙2).
    개선이 작으면 그 사실도 본문에 명시한다(AI 강제 금지).

책임성(규칙3, 가장 중요):
  - 민원 자료는 편향된다: 신고 접근성(연령·디지털 친숙도), 반복 민원, 지역 민원문화 차이.
  - 자기실현적 편향: 신고 多 지역에 자원 집중 → 인지·처리↑ → 신고 더 증가 → 모델이
    그 지역을 더 고위험으로 학습하는 되먹임 루프. 예측적 치안의 편향과 동형(Ch11 연결).
  - 예측 점수를 기관 평가지표·행정처분의 근거로 직결하지 않는다. 인간 검토·감사 필수.

데이터: 미리 준비한 민원 시공간 패널(도로 공변량 + 주별 민원 관측)을 불러와
  사용(개인 식별 불가, 격자·권역 집계). 분석·추정은 불러온 데이터로 계산한 실제 값이다.
  실제 분석에서는 국민신문고·범정부 민원분석시스템(국민권익위) 자료를 사용한다.

실행:
    python 10-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 10-2-civil-complaint-forecast.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 1. 격자·기간 파라미터 (도시 격자 × 주)
# ===========================================================
GRID = 16
N = GRID * GRID                 # 256개 격자
T = 40                          # 40주
BURN = 12                       # 앞 12주는 burn-in(정상상태 도달용), 분석은 이후
# 데이터 생성 파라미터(기저 강도·시간 지속성 ALPHA·공간 확산 KAPPA·계절성)와
# 정상성 조건(ALPHA+KAPPA<1)은 10-0-simdata-prep.py에 있다. 여기서는 결과를 불러와
# 분석만 한다.


def neighbor_sum(mat):
    """각 격자의 4-이웃 합. mat: (GRID, GRID)."""
    s = np.zeros_like(mat, dtype=float)
    s[1:, :] += mat[:-1, :]
    s[:-1, :] += mat[1:, :]
    s[:, 1:] += mat[:, :-1]
    s[:, :-1] += mat[:, 1:]
    return s


# 격자별 유효 이웃 수(내부 4, 가장자리 3, 모서리 2) — 이웃 평균 계산용
NCOUNT = neighbor_sum(np.ones((GRID, GRID)))


def neighbor_mean(mat):
    """각 격자의 4-이웃 평균(공간 시차). 합이 아닌 평균이라 발산하지 않는다."""
    return neighbor_sum(mat) / NCOUNT


def load_complaints():
    """미리 준비한 민원 시공간 패널을 불러온다.

    road: 관측 가능한 정적 공변량(도로·활동 밀도) — 예측 피처로 사용.
    obs : 주별 민원 관측(비선형 시공간 과정으로 생성). 민원은 인접 구간으로
          번지고(공간 확산) 과거에 의존(시간 지속성)하도록 설계돼 있다.
    데이터에는 신고 성향(편향)도 반영돼 있으나(민원 多 ≠ 문제 多), 그 자체는
    관측되지 않으므로 저장하지 않는다(규칙3).
    """
    road_path = DATA_DIR / "complaint_road.parquet"
    obs_path = DATA_DIR / "complaint_obs.parquet"
    if not road_path.exists() or not obs_path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {road_path}\n먼저 실행: python 10-0-simdata-prep.py")

    road = (pd.read_parquet(road_path).sort_values("cell")["road"]
            .to_numpy().reshape(GRID, GRID))
    obs_df = pd.read_parquet(obs_path).sort_values(["t", "cell"])
    # 저장은 정수 무손실 → 원 시뮬레이션과 동일한 float 배열로 복원
    obs = obs_df["count"].to_numpy().astype(float).reshape(T, GRID, GRID)
    return obs, road


def build_panel(obs, road):
    """(t, 격자) 패널 + 시공간 피처. target_next = 다음 주 민원 수."""
    recs = []
    rr, cc = np.divmod(np.arange(N), GRID)
    road_flat = road.reshape(-1)
    for t in range(BURN, T - 1):                 # t의 피처로 t+1을 예측
        self_t = obs[t].reshape(-1)
        self_t1 = obs[t - 1].reshape(-1)
        self_t2 = obs[t - 2].reshape(-1)
        neigh_t = neighbor_mean(obs[t]).reshape(-1)
        neigh_t1 = neighbor_mean(obs[t - 1]).reshape(-1)
        target = obs[t + 1].reshape(-1)
        for c in range(N):
            recs.append((t, c, rr[c], cc[c], road_flat[c],
                         self_t[c], self_t1[c], self_t2[c],
                         neigh_t[c], neigh_t1[c], target[c]))
    return pd.DataFrame(recs, columns=[
        "t", "cell", "row", "col", "road",
        "self_t", "self_t1", "self_t2", "neigh_t", "neigh_t1", "target_next"])


def time_block_eval(df, feat_cols):
    """시간 블록 CV: 앞 70% 기간 학습 → 뒤 30% 시점 예측(미래 누수 차단)."""
    cut = int(df["t"].quantile(0.7))
    tr, te = df["t"] <= cut, df["t"] > cut
    m = GradientBoostingRegressor(random_state=42, n_estimators=300,
                                  max_depth=3, learning_rate=0.05)
    m.fit(df.loc[tr, feat_cols], df.loc[tr, "target_next"])
    pred = m.predict(df.loc[te, feat_cols])
    y = df.loc[te, "target_next"]
    return r2_score(y, pred), mean_absolute_error(y, pred), m, te


def main():
    print("=" * 64)
    print("민원 발생 시공간 예측과 현장대응 우선순위 (10-2, 2층 AI)")
    print("=" * 64)
    obs, road = load_complaints()
    df = build_panel(obs, road)
    total = obs[BURN:].sum()
    print(f"\n분석 단위: {N}개 격자 × {T}주(분석 {BURN}~{T-1}주), 권역 8개 집계, 개인 식별 불가")
    print(f"총 민원(분석기간): {int(total):,}건, 주 평균 {total/(T-BURN):.0f}건")

    # -------- [1층 GIS] 권역별 집계·핫스팟 --------
    print("\n[1층 GIS] 권역(행정동 대용)별 누적 민원 집계 — 8개 권역(열 2분할 × 행 4분할)")
    rr, cc = np.divmod(df["cell"].values, GRID)
    # 8 권역: 행을 4구간, 열을 2구간 → 4×2
    df["region"] = (rr // (GRID // 4)) * 2 + (cc // (GRID // 2))
    last_t = df["t"].max()
    cum = (df.groupby("region")["self_t"].sum()).astype(int)
    for reg, v in cum.items():
        print(f"  권역 {reg}: 누적 {v:,}건")

    # -------- [2층 AI] 시공간 피처의 정보 가치 --------
    print("\n[2층 AI] 다음 주 민원 예측: 시공간 피처 유무 비교 (시간 블록 CV 70/30)")
    temporal_only = ["road", "self_t", "self_t1", "self_t2"]
    spatiotemporal = temporal_only + ["neigh_t", "neigh_t1"]

    r2_t, mae_t, _, _ = time_block_eval(df, temporal_only)
    r2_st, mae_st, model_st, te_mask = time_block_eval(df, spatiotemporal)
    print(f"  시간 시차만(자기 과거)   R² = {r2_t:6.3f} | MAE = {mae_t:.3f}")
    print(f"  +공간 시차(이웃 과거)    R² = {r2_st:6.3f} | MAE = {mae_st:.3f}")
    print(f"    (개선  R² {r2_st-r2_t:+.3f}, MAE {mae_t-mae_st:+.3f})")
    if r2_st - r2_t > 0.01:
        print("  → 공간 시차가 다음 주 예측을 개선 = 공간 명시적 AI 정당화(규칙2).")
        print("    민원은 인접 구간으로 번지므로 자기 과거만으로는 확산을 놓친다.")
    else:
        print("  → 공간 피처 개선 미미 → 시간 모델로 충분(AI 강제 금지, 규칙2).")

    imp = pd.Series(model_st.feature_importances_,
                    index=spatiotemporal).sort_values(ascending=False)
    print("\n  피처 중요도(설명가능성):")
    for name, val in imp.items():
        print(f"    {name:9s} {val:.3f}")

    # -------- [3층 정책] 다음 주 권역별 현장대응 우선순위 --------
    print("\n[3층 정책] 다음 주 권역별 현장대응 자원 배치 우선순위")
    # 마지막 시점 피처로 다음 주 격자 예측 → 권역 합계
    last = df[df["t"] == last_t].copy()
    last["pred_next"] = model_st.predict(last[spatiotemporal])
    reg_pred = (last.groupby("region")["pred_next"].sum()
                .sort_values(ascending=False).round(1))
    reg_df = reg_pred.reset_index()
    reg_df.columns = ["region", "pred_next_week_complaints"]
    reg_df["priority_rank"] = reg_df["pred_next_week_complaints"].rank(
        ascending=False).astype(int)
    print(reg_df.to_string(index=False))

    csv_path = RESULTS_DIR / "complaint_response_priority.csv"
    reg_df.to_csv(csv_path, index=False)
    print(f"\n  권역 우선순위 저장 → {csv_path.name}")
    print("  ※ 책임성(규칙3): 이 점수는 현장대응 '배치 보조'일 뿐이다.")
    print("    민원 자료는 신고 편향을 담고(민원 多 ≠ 문제 多), 자원을 신고 많은 곳에만")
    print("    몰면 자기실현적 편향(신고→자원→신고) 루프에 빠진다. 신고 저조 취약지역을")
    print("    별도 점검하고, 점수를 기관 평가·행정처분 근거로 직결하지 않는다.")

    # -------- [불확실성] 정규화 conformal 예측구간·신뢰등급(시간블록 OOF 근사) --------
    # 우선순위 점추정만으로 자원을 배치하면 '얼마나 틀릴지 모른 채 확신하는'(7.5절) 위험이 있다.
    # conformal 예측(Lei et al. 2018)은 분포가정 없이 목표 포함확률을 달성하는 예측구간을
    # 주고, 지역 난이도로 정규화(Romano et al. 2019 계열)하면 이질적 불확실성을 반영한다.
    # 단, 아래는 엄밀한 분할 conformal(학습/보정/시험 3분할)이 아니라, 시간 블록 CV의
    # 뒤 30% out-of-fold(OOF) 잔차를 σ̂ 적합과 캘리브레이션에 함께 쓰는 '교육용 근사'다
    # (같은 잔차를 재사용하므로 포함률은 다소 낙관적). 미래 누수를 막으려 학습(앞 70%)과
    # 분리된 미래 블록(뒤 30%)의 OOF 잔차만 쓴다(기존 시간 블록 구조 그대로).
    print("\n[불확실성] 정규화 conformal 예측구간(90%, 시간블록 OOF 근사)과 신뢰등급")
    cal = df.loc[te_mask]
    X_cal = cal[spatiotemporal]                      # DataFrame(피처명 유지) → 경고 없이 예측
    # 1) 미래 블록 OOF 예측(학습에 안 쓰인 뒤 30% 시점) → 누수 없는 정직한 잔차
    oof = model_st.predict(X_cal)
    resid = np.abs(cal["target_next"].values - oof)
    n_cal = resid.shape[0]
    # 2) 지역 난이도 σ̂(x): OOF 절대잔차를 특징으로 회귀(이질적 불확실성 추정)
    sigma_model = GradientBoostingRegressor(random_state=1, n_estimators=150,
                                            max_depth=3, learning_rate=0.05).fit(X_cal, resid)
    sigma_cal = np.clip(sigma_model.predict(X_cal), 1e-3, None)
    # 3) 정규화 conformal 점수의 k번째 순서통계(k=⌈(n+1)(1-α)⌉, 분할 conformal 관례)
    alpha = 0.10
    scores = np.sort(resid / sigma_cal)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = scores[min(k, n_cal) - 1]                   # k>n이면 최댓값(구간 무한대 회피)
    half_cal = q * sigma_cal                         # 보정 단위별 예측구간 반폭(이질적)
    cover = float(np.mean(resid <= half_cal))        # OOF 경험적 포함률(≈0.90 목표)
    print(f"  90% 예측구간 경험적 포함률(OOF) = {cover:.3f}  (목표 0.90)")
    print(f"  구간 반폭(건): 중앙값 {np.median(half_cal):.2f}, "
          f"범위 [{half_cal.min():.2f}, {half_cal.max():.2f}]")
    print("  주의: σ̂를 같은 OOF 잔차로 적합했으므로 포함률은 다소 낙관적이다.")
    print("        엄밀한 실무는 학습/보정/시험 3분할 split conformal을 쓴다(7.5절·보론).")

    # -------- 다음 주 권역 우선순위표에 불확실성 열 추가 --------
    # 다음 주 격자 예측(last)에 같은 σ̂·q를 적용해 격자별 구간 반폭·신뢰등급을 부여한다.
    last["pi_half"] = q * np.clip(sigma_model.predict(last[spatiotemporal]), 1e-3, None)
    g1, g2 = np.quantile(last["pi_half"], [1 / 3, 2 / 3])   # 신뢰등급: 반폭 3분위(좁을수록 신뢰↑)
    last["confidence"] = np.where(last["pi_half"] <= g1, "높음",
                                  np.where(last["pi_half"] <= g2, "중간", "낮음"))
    unc = (last.groupby("region")
               .agg(mean_pi_half=("pi_half", "mean"),
                    low_conf_cells=("confidence", lambda s: int((s == "낮음").sum())))
               .round(2))
    r1, r2 = np.quantile(unc["mean_pi_half"], [1 / 3, 2 / 3])
    unc["confidence"] = np.where(unc["mean_pi_half"] <= r1, "높음",
                                 np.where(unc["mean_pi_half"] <= r2, "중간", "낮음"))
    reg_unc = reg_df.merge(unc.reset_index(), on="region", how="left")
    print("\n  불확실성을 반영한 다음 주 권역 우선순위:")
    print(reg_unc.to_string(index=False))
    unc_csv = RESULTS_DIR / "complaint_response_priority_uncertainty.csv"
    reg_unc.to_csv(unc_csv, index=False)
    print(f"  다음 주 권역 우선순위(불확실성 포함) 저장 → {unc_csv.name}")
    print("  → 예측 민원이 많아도 신뢰등급이 '낮음'이면, 자원 확정 전에 추가 관측·현장확인으로")
    print("     불확실성을 줄인다(점추정 맹신 방지).")

    print("\n[완료] 민원 시공간 예측·현장대응 우선순위 분석을 마쳤다.")


if __name__ == "__main__":
    main()
