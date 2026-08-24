"""
4.6 분석 2 (비즈니스): 행정동 기대 공급량 모델과 자치구 단위 공간 검증
=====================================================================
같은 파이프라인(RF → 공간 CV → SHAP)을 공공 취약성이 아니라 상권에 적용한다.
쓰는 자료는 실제 국내 공개 데이터다(4-0-market-data-prep.py 산출물).

**이 모델이 답하는 질문을 정확히 적는다.**
"수요 여건이 이만한 행정동이면 카페가 몇 개쯤 유지되는가."
이것은 매출 예측이 아니다. 점포 수는 이미 시장이 도달한 **공급 균형**이지 수요의
직접 관측이 아니기 때문이다. 그래서 결과를 '유망도'라 부르지 않고 **기대 공급량**
이라 부른다. 실제 점포 수가 기대 공급량보다 적은 동은 '아직 덜 채워진 곳'의 후보이며,
후보일 뿐 결정이 아니다.

네 가지를 확인한다.
  1) 공간 시차를 넣으면 예측이 좋아지는가 (넣고 뺀 비교)
  2) 무작위 CV와 자치구 단위 CV의 차이 — "강남에서 배운 모델이 노원에 통하는가"
  3) 라벨 셔플 대조군 — 모델이 잡은 것이 구조인가 잡음인가
  4) 1)과 2)의 차이가 실행 간 흔들림보다 큰가 — 시드를 바꿔 잡음 바닥을 잰다

실행:
    python 4-6-store-location-supply.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, GroupKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results"
SEED = 42

# 수요·위치 피처만 쓴다. 같은 동의 다른 업종 점포 수(n_store)는 넣지 않는다 —
# 공급량을 피처로 넣으면 "가게 많은 곳에 가게 많다"를 발견처럼 보고하게 된다.
DEMAND_FEATURES = ["day_pop", "night_pop", "day_night_ratio",
                   "weekend_ratio", "peak_ratio", "dist_center_km"]
SPATIAL_FEATURES = ["nbr_cafe_lag", "nbr_daypop_lag"]

FEATURE_KO = {
    "day_pop": "낮 생활인구", "night_pop": "심야 생활인구",
    "day_night_ratio": "주야 인구비", "weekend_ratio": "주말/평일 비",
    "peak_ratio": "시간대 변동", "dist_center_km": "도심 거리",
    "nbr_cafe_lag": "이웃 동 카페 수", "nbr_daypop_lag": "이웃 동 낮인구",
}


def set_korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(name == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=-1)


def cv_r2(X, y, cv, groups=None) -> tuple[np.ndarray, float]:
    """교차검증 예측(OOF)과 그 R²를 돌려준다."""
    pred = cross_val_predict(make_model(), X, y, cv=cv, groups=groups, n_jobs=None)
    return pred, r2_score(y, pred)


def cv_r2_seeded(X, y, cv, groups, seed: int) -> tuple[float, np.ndarray]:
    """지정한 모델 시드로 OOF R²와 폴드별 R²를 함께 낸다.

    시드 반복(6절) 전용이다. make_model()과 달리 random_state를 인자로 받아,
    '모델 시드만 바꿨을 때 결과가 얼마나 움직이는가'를 잴 수 있게 한다.
    OOF R²는 본문 표와 같은 정의(전체 예측을 모아 한 번에 계산)이고,
    폴드별 R²는 흔들림의 크기를 재기 위해 따로 뽑는다.
    """
    mdl = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=seed, n_jobs=-1)
    pred = cross_val_predict(mdl, X, y, cv=cv, groups=groups, n_jobs=None)
    fold_r2 = cross_val_score(mdl, X, y, cv=cv, groups=groups, scoring="r2",
                              n_jobs=None)
    return r2_score(y, pred), np.asarray(fold_r2)


def main() -> None:
    print("=" * 66)
    print("비즈니스 분석: 행정동 기대 공급량과 자치구 단위 공간 검증")
    print("=" * 66)

    src = DATA_DIR / "seoul_dong_market.parquet"
    if not src.exists():
        sys.exit(f"[중단] {src.name}이 없다. 먼저 python 4-0-market-data-prep.py 를 실행한다.")
    df = pd.read_parquet(src)
    print(f"\n자료: 서울 {len(df)}개 행정동, {df['sigungu'].nunique()}개 자치구")
    print(f"  카페 점포 수 — 중앙값 {df['n_cafe'].median():.0f}, "
          f"최대 {df['n_cafe'].max()} ({df.loc[df['n_cafe'].idxmax(), 'dong']})")

    y = df["n_cafe"].values.astype(float)
    groups = df["sigungu"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=5)

    # ── 1. 공간 시차의 값어치: 넣고 뺀 비교 ──────────────────────────────
    print("\n" + "-" * 66)
    print("[1] 공간 시차를 넣으면 나아지는가 (자치구 단위 CV 기준)")
    print("-" * 66)
    X_demand = df[DEMAND_FEATURES].values
    X_full = df[DEMAND_FEATURES + SPATIAL_FEATURES].values

    _, r2_demand = cv_r2(X_demand, y, gkf, groups)
    pred_full_group, r2_full_group = cv_r2(X_full, y, gkf, groups)
    print(f"  수요·위치 피처만      R² = {r2_demand:6.3f}")
    print(f"  + 공간 시차 피처      R² = {r2_full_group:6.3f}"
          f"  (개선 {r2_full_group - r2_demand:+.3f})")

    # ── 2. 검증 설계에 따른 차이 ────────────────────────────────────────
    print("\n" + "-" * 66)
    print("[2] 무작위 CV vs 자치구 단위 CV — 처음 보는 자치구에도 통하는가")
    print("-" * 66)
    pred_random, r2_random = cv_r2(X_full, y, kf)
    print(f"  무작위 5-Fold CV      R² = {r2_random:6.3f}")
    print(f"  자치구 GroupKFold CV  R² = {r2_full_group:6.3f}"
          f"  (과대추정 {r2_random - r2_full_group:+.3f})")

    # 자치구별로 남겨 두었을 때의 성능. 어디서 무너지는지 본다.
    per_gu = []
    for tr, te in gkf.split(X_full, y, groups):
        mdl = make_model().fit(X_full[tr], y[tr])
        held = sorted(set(groups[te]))
        per_gu.append((", ".join(held), r2_score(y[te], mdl.predict(X_full[te])), len(te)))
    print("\n  폴드별(자치구 묶음) 성능:")
    for gu, r2, n in per_gu:
        print(f"    R² {r2:6.3f}  (n={n:3d})  {gu[:52]}")

    # ── 3. 라벨 셔플 대조군 ─────────────────────────────────────────────
    print("\n" + "-" * 66)
    print("[3] 대조군 — 결과변수를 섞으면 성능이 사라지는가")
    print("-" * 66)
    rng = np.random.default_rng(SEED)
    null_r2 = []
    for _ in range(20):
        _, r2n = cv_r2(X_full, rng.permutation(y), gkf, groups)
        null_r2.append(r2n)
    print(f"  라벨 셔플 20회 R² = {np.mean(null_r2):.3f} (SD {np.std(null_r2):.3f}, "
          f"최대 {np.max(null_r2):.3f})")
    print(f"  → 실제 {r2_full_group:.3f} 대비, 모델이 잡은 것은 잡음이 아니라 구조다")

    # ── 4. SHAP 기여도 ──────────────────────────────────────────────────
    print("\n" + "-" * 66)
    print("[4] SHAP — 기대 공급량을 무엇이 끌어올리는가")
    print("-" * 66)
    import shap
    mdl_full = make_model().fit(X_full, y)
    expl = shap.TreeExplainer(mdl_full)
    sv = expl.shap_values(X_full, check_additivity=False)
    names = DEMAND_FEATURES + SPATIAL_FEATURES
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    for _, r in imp.iterrows():
        print(f"  {FEATURE_KO[r['feature']]:<14} {r['mean_abs_shap']:8.3f}")

    # ── 5. 기대 공급량 대비 미달 행정동 ─────────────────────────────────
    print("\n" + "-" * 66)
    print("[5] 기대 공급량보다 점포가 적은 행정동 (자치구 CV 예측 기준)")
    print("-" * 66)
    df["expected"] = pred_full_group
    df["gap"] = df["n_cafe"] - df["expected"]      # 음수 = 기대보다 적다
    cand = df.nsmallest(10, "gap")[["sigungu", "dong", "n_cafe", "expected", "gap", "day_pop"]]
    print(f"  {'자치구':<8}{'행정동':<12}{'실제':>5}{'기대':>8}{'차이':>8}{'낮인구':>10}")
    for _, r in cand.iterrows():
        print(f"  {r['sigungu']:<8}{r['dong']:<12}{r['n_cafe']:>5}"
              f"{r['expected']:>8.1f}{r['gap']:>8.1f}{r['day_pop']:>10,.0f}")

    over = df.nlargest(3, "gap")
    print("\n  반대쪽(기대보다 많은 동) 상위 3:")
    for _, r in over.iterrows():
        print(f"    {r['sigungu']} {r['dong']}: 실제 {r['n_cafe']} vs 기대 {r['expected']:.1f} "
              f"({r['gap']:+.1f})")

    # ── 6. 시드 반복 — 앞의 두 차이가 잡음보다 큰가 ─────────────────────
    # [1]의 '개선 +0.003'과 [2]의 '과대추정 +0.025'는 소수점 셋째~둘째 자리 값이다.
    # 그 자리가 실행마다 흔들리는지 먼저 재지 않으면 부호도 크기도 해석할 수 없다.
    #
    # RandomForest는 random_state를 고정하면 결정적이다. 그래서 '무엇을 바꿔
    # 반복하는가'를 명시해야 한다. 두 가지로 나눈다.
    #   (A) 모델 시드만 교체 — 폴드 배정은 본 실행과 똑같이 고정한다. RF 내부의
    #       부트스트랩 표본과 분기 후보 피처 추첨에서 오는 흔들림만 잡힌다.
    #       시드 42는 본 실행을 그대로 재현하므로 자체 점검이 된다.
    #   (B) 모델 시드 + 폴드 배정 시드를 함께 교체 — 무작위 CV는 KFold의 shuffle
    #       시드를, 자치구 CV는 GroupKFold의 그룹→폴드 배정 시드를 바꾼다. 남이
    #       이 코드를 다시 돌릴 때 실제로 마주하는 흔들림에 더 가깝다.
    #       본 실행의 GroupKFold는 shuffle 없이(=배정 고정) 돌렸으므로, (B)의
    #       결과는 본 실행 수치의 재현이 아니라 잡음 바닥의 척도로 읽는다.
    #
    # 이 블록은 앞의 계산을 건드리지 않는다. 위 [1]~[5]가 모두 끝난 뒤에 덧붙으며,
    # 난수를 쓰는 객체를 새로 만들어 쓰므로 앞 절의 수치는 그대로 보존된다.
    print("\n" + "-" * 66)
    print("[6] 시드를 바꾸면 위 두 차이가 얼마나 흔들리는가 (잡음 바닥)")
    print("-" * 66)
    repeat_seeds = (0, 1, 2, 42)
    for tag, desc in (("A", "모델 시드만 교체 (폴드 배정 고정, 42=본 실행 재현)"),
                      ("B", "모델 시드 + 폴드 배정 시드 동시 교체")):
        print(f"\n  ({tag}) {desc}")
        print(f"    {'시드':>4}{'수요·위치':>10}{'+공간시차':>10}{'개선':>8}"
              f"{'무작위CV':>10}{'과대추정':>9}{'폴드SD(구)':>11}{'폴드SD(무)':>11}")
        gains, overs, sd_grp, sd_rnd = [], [], [], []
        for s in repeat_seeds:
            kf_s = kf if tag == "A" else KFold(n_splits=5, shuffle=True, random_state=s)
            gkf_s = gkf if tag == "A" else GroupKFold(n_splits=5, shuffle=True,
                                                      random_state=s)
            r2_d, _ = cv_r2_seeded(X_demand, y, gkf_s, groups, s)
            r2_g, fold_g = cv_r2_seeded(X_full, y, gkf_s, groups, s)
            r2_r, fold_r = cv_r2_seeded(X_full, y, kf_s, None, s)
            gains.append(r2_g - r2_d)
            overs.append(r2_r - r2_g)
            sd_grp.append(fold_g.std())
            sd_rnd.append(fold_r.std())
            print(f"    {s:>4}{r2_d:>10.3f}{r2_g:>10.3f}{gains[-1]:>+8.3f}"
                  f"{r2_r:>10.3f}{overs[-1]:>+9.3f}{sd_grp[-1]:>11.3f}"
                  f"{sd_rnd[-1]:>11.3f}")
        print(f"    범위 — 공간시차 개선 {min(gains):+.3f} ~ {max(gains):+.3f}"
              f" / 과대추정 {min(overs):+.3f} ~ {max(overs):+.3f}")
        print(f"    폴드 간 R² SD 평균 — 자치구 CV {np.mean(sd_grp):.3f}, "
              f"무작위 CV {np.mean(sd_rnd):.3f}")
        if tag == "B":
            print(f"\n  → 재려는 차이(개선 {r2_full_group - r2_demand:+.3f}, "
                  f"과대추정 {r2_random - r2_full_group:+.3f})를 위 범위·SD와 나란히 놓고,"
                  f"\n    차이가 흔들림보다 큰지 확인한 뒤에만 부호를 해석한다.")

    # ── 7. 산출물 저장 ──────────────────────────────────────────────────
    RESULT_DIR.mkdir(exist_ok=True)
    df.sort_values("gap")[["sigungu", "dong", "dong_code", "n_cafe", "expected", "gap",
                           "day_pop", "night_pop", "dist_center_km"]].to_csv(
        RESULT_DIR / "ch4_dong_supply_gap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "구분": ["수요·위치만(자치구CV)", "+공간시차(자치구CV)", "+공간시차(무작위CV)", "라벨셔플 대조군"],
        "R2": [r2_demand, r2_full_group, r2_random, float(np.mean(null_r2))],
    }).to_csv(RESULT_DIR / "ch4_supply_cv_comparison.csv", index=False, encoding="utf-8-sig")

    set_korean_font()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lim = np.abs(df["gap"]).quantile(0.95)
    sc = ax.scatter(df["x_m"] / 1000, df["y_m"] / 1000, c=df["gap"].clip(-lim, lim),
                    cmap="RdBu", s=np.sqrt(df["day_pop"]) / 12, edgecolor="k", linewidth=0.3)
    ax.scatter([0], [0], marker="*", s=180, c="black", zorder=5)
    ax.annotate("시청", (0, 0), xytext=(4, 4), textcoords="offset points", fontsize=9)
    plt.colorbar(sc, ax=ax, label="실제 − 기대 (점포 수)")
    ax.set_xlabel("도심 기준 동서 거리 (km)")
    ax.set_ylabel("도심 기준 남북 거리 (km)")
    ax.set_title("행정동별 카페 공급 격차 (붉을수록 기대보다 적음)\n점 크기 = 낮 생활인구")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ch4_supply_gap_map.png", dpi=150)
    plt.close(fig)

    print("\n" + "-" * 66)
    print("저장: ch4_dong_supply_gap.csv, ch4_supply_cv_comparison.csv, ch4_supply_gap_map.png")
    print("=" * 66)
    print("[완료] 이 표는 후보 목록이지 출점 결정이 아니다. 새 매장이 자사 기존 매장의")
    print("       손님을 얼마나 가져가는지는 이 모델이 답하지 못한다.")


if __name__ == "__main__":
    main()
