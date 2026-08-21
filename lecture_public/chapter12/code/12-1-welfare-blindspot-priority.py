"""
12장 실습 1: 복지·의료 사각지대 위험 예측과 방문상담 우선순위
=================================================================
사회복지·보건 정책 질문: "복지·의료 사각지대는 어디인가? 어디부터 찾아가야 하는가?"

3층 모델:
  [1층 GIS]  읍면동별 취약계층·위기정보 집계 + 의료시설 이동시간 접근성으로 취약지 식별(학부)
  [2층 AI ]  공간 시차(이웃 위기정보·접근성) 피처를 내장한 사각지대 위험 예측 +
             공간 블록 CV + 공간 피처 유무 성능 비교(=AI 정당성) + 피처 중요도(=설명가능성)
  [3층 정책] 읍면동별 사각지대 위험 우선순위표(방문상담 우선순위) CSV

핵심 설계(공간 명시성):
  - '미관측 지역 고립효과(U)'를 공간 자기상관을 갖도록 주입한다.
  - U는 관측되지 않으므로, 이웃 읍면동의 위기정보·접근성(공간 시차)이 U의 대리변수가 된다.
  - 공간 시차 피처가 (설계된) spillover를 회복함을 보인다 = 방법 검증(규칙2의 비교 설계).
    실데이터에서 2층 AI가 정당한지는 같은 비교를 실데이터로 수행해야 입증된다.

책임성(규칙3): 예측 위험 점수는 방문상담의 '우선순위 보조'일 뿐,
  자동 행정처분(수급 중단 등)의 근거가 되어서는 안 된다(미시간 MiDAS 반면교사).

데이터: 미리 준비한 교육용 합성 데이터를 불러와 사용(개인 식별 불가, 읍면동 집계).
       분석·추정은 실제 계산값. 실제 분석에서는 복지 사각지대 발굴시스템 47종
       위기정보·SGIS 통계격자를 사용한다.
       (합성 데이터의 설계 의도와 생성은 12-0-simdata-prep.py 참고)

실행:
    python 12-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 12-1-welfare-blindspot-priority.py
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

# ===========================================================
# 1. 분석 파라미터 (읍면동 격자)
# ===========================================================
GRID_ROWS, GRID_COLS = 20, 15          # 300개 읍면동을 20×15 격자로 배치(공간 이웃 정의)
N = GRID_ROWS * GRID_COLS
ACCESS_THRESHOLD = 30.0                # 의료시설 30분 초과 = 접근성 취약지(응급의료취약지 기준 참고)


def load_units():
    """미리 준비한 읍면동 사각지대 데이터를 불러온다.

    이 데이터에는 관측 피처(취약계층·위기정보·접근성) 외에, 인접 지역의 위기정보·
    접근성에 의존하는 spillover 구조가 설계돼 들어 있다. 자기 피처만으로는 이 spillover를
    잡지 못하고, 공간 시차 피처가 이를 회복한다(아래 2층 AI 비교로 확인). 데이터 생성의
    설계 의도는 12-0-simdata-prep.py에 있다.
    """
    path = DATA_DIR / "welfare_units.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 12-0-simdata-prep.py")
    return pd.read_parquet(path)


def spatial_lag(df, value_col):
    """4-이웃(상하좌우) 평균 = 공간 시차 피처. 미관측 공간효과의 대리변수."""
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid[df["row"], df["col"]] = df[value_col].values
    lag = np.zeros(N)
    for i, (r, c) in enumerate(zip(df["row"], df["col"])):
        neigh = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                neigh.append(grid[rr, cc])
        lag[i] = np.nanmean(neigh)
    return lag


def spatial_block_folds(df, n_blocks=5):
    """열(col)을 기준으로 공간 블록 분할 → 인접 단위가 같은 fold에 모임(공간 누수 차단)."""
    block = (df["col"].values / (GRID_COLS / n_blocks)).astype(int)
    block = np.clip(block, 0, n_blocks - 1)
    for b in range(n_blocks):
        test = block == b
        yield ~test, test


def cv_r2(X, y, folds):
    preds = np.full(len(y), np.nan)
    for tr, te in folds:
        m = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds)


def main():
    df = load_units()
    print("=" * 64)
    print("복지·의료 사각지대 위험 예측과 방문상담 우선순위 (12-1)")
    print("=" * 64)
    print(f"\n분석 단위: {N}개 읍면동(20×15 격자), 개인 식별 불가 집계")

    # -------- [1층 GIS] 접근성 기반 취약지 단순 식별 --------
    print("\n[1층 GIS] 의료 접근성 취약지 (이동시간 > 30분)")
    vuln = df["medical_access"] > ACCESS_THRESHOLD
    print(f"  접근성 취약 읍면동: {vuln.sum()}개 ({vuln.mean()*100:.1f}%)")
    print(f"  취약지 평균 사각지대 위험: {df.loc[vuln,'blindspot_risk'].mean():.2f}% "
          f"vs 그 외: {df.loc[~vuln,'blindspot_risk'].mean():.2f}%")
    print("  → 접근성만으로는 위험의 일부만 설명(고립·취약계층 결합 필요) → 2층 AI 동기")

    # -------- [2층 AI] 공간 명시적 위험 예측 --------
    print("\n[2층 AI] 사각지대 위험 예측: 공간 피처 유무 + 검증 방식 비교")
    base_cols = ["pop", "elderly", "single", "basic", "crisis",
                 "medical_access", "welfare_access"]
    df["crisis_lag"] = spatial_lag(df, "crisis")
    df["access_lag"] = spatial_lag(df, "medical_access")
    df["elderly_lag"] = spatial_lag(df, "elderly")
    spatial_cols = base_cols + ["crisis_lag", "access_lag", "elderly_lag"]

    y = df["blindspot_risk"].values
    X_base = df[base_cols].values
    X_spat = df[spatial_cols].values
    folds = list(spatial_block_folds(df))

    r2_base = cv_r2(X_base, y, folds)
    r2_spat = cv_r2(X_spat, y, folds)
    # 검증 방식 비교: 랜덤 CV는 공간 누수로 과대추정
    from sklearn.model_selection import KFold
    rand_folds = [(tr, te) for tr, te in KFold(5, shuffle=True,
                  random_state=42).split(X_spat)]
    r2_spat_random = cv_r2(X_spat, y, rand_folds)

    print(f"  공간 블록 CV | 비공간 피처     R² = {r2_base:6.3f}")
    print(f"  공간 블록 CV | +공간 시차 피처 R² = {r2_spat:6.3f}  (개선 {r2_spat-r2_base:+.3f})")
    print(f"  랜덤    CV | +공간 시차 피처 R² = {r2_spat_random:6.3f}  "
          f"(공간 누수 과대추정 {r2_spat_random-r2_spat:+.3f})")
    if r2_spat - r2_base > 0.02:
        print("  → 공간 시차가 (설계된) spillover를 회복 = 방법 검증(실데이터 실증은 별도, 규칙2)")
    else:
        print("  → 공간 피처 개선 미미 → 단순 모델로 충분(AI 강제 금지, 규칙2)")

    # 설명가능성: 피처 중요도(규칙3)
    full = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                     max_depth=3, learning_rate=0.05).fit(X_spat, y)
    imp = pd.Series(full.feature_importances_, index=spatial_cols).sort_values(ascending=False)
    print("\n  피처 중요도 상위 5(설명가능성):")
    for name, val in imp.head(5).items():
        print(f"    {name:16s} {val:.3f}")

    # -------- [3층 정책] 우선순위표 --------
    print("\n[3층 정책] 방문상담 우선순위 (예측 위험 상위)")
    df["pred_risk"] = full.predict(X_spat)
    df["priority_rank"] = df["pred_risk"].rank(ascending=False).astype(int)
    out = (df.sort_values("pred_risk", ascending=False)
             [["unit_id", "row", "col", "pop", "elderly", "single",
               "crisis", "medical_access", "pred_risk", "priority_rank"]]
             .head(20).round(3))
    print(out.head(10).to_string(index=False))
    csv_path = RESULTS_DIR / "welfare_blindspot_priority.csv"
    df.sort_values("pred_risk", ascending=False).round(4).to_csv(csv_path, index=False)
    print(f"\n  전체 우선순위표 저장 → {csv_path.name}")
    print("  ※ 책임성(규칙3): 위 점수는 '방문상담 우선순위 보조'일 뿐,")
    print("    자동 수급중단 등 행정처분 근거로 쓰면 안 된다(미시간 MiDAS 85% 오판 반면교사).")

    # -------- [불확실성] 정규화 conformal 예측구간·신뢰등급(OOF 근사) --------
    # 사각지대 위험 점추정만으로 방문 자원을 배분하면 '얼마나 틀릴지 모른 채 확신하는'
    # (7.5절) 위험이 있다. conformal 예측(Lei et al. 2018)은 분포가정 없이 예측구간을
    # 주고, 지역 난이도로 정규화(Romano et al. 2019 계열)하면 이질적 불확실성을 반영한다.
    # 단, 아래는 엄밀한 분할 conformal(학습/보정/시험 3분할)이 아니라 공간 블록 CV의
    # OOF 잔차를 σ̂ 적합과 캘리브레이션에 함께 쓰는 '교육용 근사'다(포함률은 다소 낙관적).
    print("\n[불확실성] 정규화 conformal 예측구간(90%, OOF 근사)과 신뢰등급")
    # 1) 공간 블록 CV의 out-of-fold(OOF) 예측 → 공간 누수 없는 정직한 잔차
    oof = np.full(N, np.nan)
    for tr, te in folds:
        m = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05).fit(X_spat[tr], y[tr])
        oof[te] = m.predict(X_spat[te])
    resid = np.abs(y - oof)
    # 2) 지역 난이도 σ̂(x): OOF 절대잔차를 특징으로 회귀(이질적 불확실성 추정)
    sigma = np.clip(GradientBoostingRegressor(random_state=1, n_estimators=150, max_depth=3,
                    learning_rate=0.05).fit(X_spat, resid).predict(X_spat), 1e-3, None)
    # 3) 정규화 conformal 점수의 k번째 순서통계(k=⌈(n+1)(1-α)⌉, 분할 conformal 관례)
    alpha = 0.10
    scores = np.sort(resid / sigma)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    q = scores[min(k, N) - 1]                        # k>N이면 최댓값(구간 무한대 회피)
    half = q * sigma                                 # 단위별 예측구간 반폭(이질적, %p)
    cover = float(np.mean(resid <= half))            # OOF 경험적 포함률(≈0.90 목표)
    df["pi_half"] = half
    df["pi_low"] = np.clip(df["pred_risk"] - half, 0, None)
    df["pi_high"] = df["pred_risk"] + half
    t1, t2 = np.quantile(half, [1 / 3, 2 / 3])       # 신뢰등급: 반폭 3분위(좁을수록 신뢰↑)
    df["confidence"] = np.where(half <= t1, "높음", np.where(half <= t2, "중간", "낮음"))
    n_low = int((df["confidence"] == "낮음").sum())
    print(f"  90% 예측구간 경험적 포함률(OOF) = {cover:.3f}  (목표 0.90)")
    print(f"  구간 반폭(%p): 중앙값 {np.median(half):.2f}, 범위 [{half.min():.2f}, {half.max():.2f}]")
    print(f"  신뢰등급 '낮음' 읍면동 = {n_low}개")
    print("  주의: σ̂를 같은 OOF 잔차로 적합했으므로 포함률은 다소 낙관적이다.")
    print("        엄밀한 실무는 학습/보정/시험 3분할 split conformal을 쓴다(7.5절·보론).")

    # -------- 방문상담 우선순위표에 불확실성 열 추가 --------
    unc = (df.sort_values("pred_risk", ascending=False)
             [["unit_id", "row", "col", "pred_risk", "priority_rank",
               "pi_half", "confidence"]].head(10).round(3))
    print("\n  불확실성을 반영한 방문상담 우선순위 상위 10:")
    print(unc.to_string(index=False))
    df.sort_values("pred_risk", ascending=False).round(4).to_csv(
        RESULTS_DIR / "welfare_blindspot_priority_uncertainty.csv", index=False)
    print("  → 신뢰등급 '낮음'은 배제 신호가 아니라 능동 점검 신호다: 사각지대에서")
    print("     저신뢰는 오히려 현장확인을 먼저 해야 할 곳일 수 있다(미시간 MiDAS 반면교사).")

    print("\n[완료] 복지·의료 사각지대 우선순위 분석을 마쳤다.")


if __name__ == "__main__":
    main()
