"""
4-3. 앙상블 기법 비교 실습 (실제 Sentinel-2 픽셀 분류)
=====================================================
실제 Sentinel-2 L2A 12밴드 클립에서 픽셀을 뽑아 토지피복을 분류한다.
라벨은 ESA WorldCover 2021(실측 토지피복 지도)에서 가져오고, 성능은
Random Forest·XGBoost·LightGBM으로 비교한다. 핵심은 픽셀을 무작위로
나누는 일반 교차검증과, 공간 블록 단위로 나누는 공간 교차검증의 차이다.

데이터 흐름:
  실제 GeoTIFF(12밴드) → 반사율 변환·구름 마스킹
  → 픽셀별 행 데이터(12밴드 + NDVI·NDWI·NDBI)
  → WorldCover 라벨 결합 → RF/XGB/LGBM → Random CV vs 공간 Block CV

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter4/code/4-3-ensemble-comparison.py
    # 데이터가 없으면 먼저: python practice/chapter4/code/4-0-data-download.py
"""

import time

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from _s2_data import build_pixel_table

print("=" * 60)
print("앙상블 기법 비교: RF vs XGBoost vs LightGBM (실제 Sentinel-2)")
print("=" * 60)

# ============================================================
# 1. 실제 픽셀 표 구성 (WorldCover 라벨)
# ============================================================
print("\n--- 1. 실제 픽셀 표 구성 ---")
X, y, coords, info = build_pixel_table(n_per_class=1500, seed=42)
feature_names = info["feature_names"]
class_names = info["class_names"]

print(f"  표본 수: {len(y):,}개 (클래스별 균형 표집)")
print(f"  피처 {X.shape[1]}개: {', '.join(feature_names)}")
print(f"  클래스 {len(class_names)}개: {class_names}")
for i, name in enumerate(class_names):
    print(f"    {name}: {(y == i).sum():,}개")

# ============================================================
# 2. 공간 블록 ID 부여 (좌표 → 1km 블록)
# ============================================================
print("\n--- 2. 공간 블록 정의 ---")
block_size = 1000  # 1km 블록
block_ids = ((coords[:, 0] // block_size).astype(int) * 100000
             + (coords[:, 1] // block_size).astype(int))
n_blocks = len(np.unique(block_ids))
print(f"  블록 수: {n_blocks}개 (1km 격자)")
print("  같은 블록의 픽셀은 학습·검증에 갈라지지 않게 묶는다(공간 누수 차단).")

# ============================================================
# 3. 모델 정의
# ============================================================
print("\n--- 3. 모델 정의 ---")
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        n_jobs=-1, random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1),
}
for name in models:
    print(f"  {name}: 정의 완료")

# ============================================================
# 4. Random CV vs 공간 Block CV
# ============================================================
print("\n--- 4. Random CV vs 공간 Block CV (Macro-F1) ---")
random_cv = KFold(n_splits=5, shuffle=True, random_state=42)
block_cv = GroupKFold(n_splits=5)

results = {}
for name, model in models.items():
    t0 = time.time()
    r_scores = cross_val_score(model, X, y, cv=random_cv, scoring="f1_macro")
    t_random = time.time() - t0
    t0 = time.time()
    b_scores = cross_val_score(model, X, y, cv=block_cv, groups=block_ids,
                               scoring="f1_macro")
    t_block = time.time() - t0
    results[name] = {"random": r_scores.mean(), "random_std": r_scores.std(),
                     "block": b_scores.mean(), "block_std": b_scores.std(),
                     "time": t_block}
    print(f"\n  {name}:")
    print(f"    Random CV Macro-F1: {r_scores.mean():.3f} +/- {r_scores.std():.3f} ({t_random:.1f}초)")
    print(f"    Block CV  Macro-F1: {b_scores.mean():.3f} +/- {b_scores.std():.3f} ({t_block:.1f}초)")
    print(f"    과대추정(Random-Block): {r_scores.mean() - b_scores.mean():+.3f}")

# ============================================================
# 5. 모델 비교 요약
# ============================================================
print("\n--- 5. 모델 비교 요약 ---")
print(f"  {'모델':15s} | {'Random F1':>10s} | {'Block F1':>10s} | {'과대추정':>8s} | {'학습시간':>8s}")
print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
for name, r in results.items():
    print(f"  {name:15s} | {r['random']:>10.3f} | {r['block']:>10.3f} | "
          f"{r['random'] - r['block']:>+8.3f} | {r['time']:>7.1f}초")

# ============================================================
# 6. Random Forest 변수 중요도
# ============================================================
print("\n--- 6. Random Forest 변수 중요도 ---")
rf = models["Random Forest"]
rf.fit(X, y)
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"  {'피처':8s} | {'중요도':>8s}")
print(f"  {'-'*8} | {'-'*8}")
for idx in sorted_idx[:8]:
    print(f"  {feature_names[idx]:8s} | {importances[idx]:>8.3f}")

# ============================================================
# 7. 클래스별 성능 (공간 Block 홀드아웃)
# ============================================================
print("\n--- 7. 클래스별 성능 (RF, 공간 블록 홀드아웃) ---")
unique_blocks = np.unique(block_ids)
rng = np.random.default_rng(42)
rng.shuffle(unique_blocks)
test_blocks = set(unique_blocks[:len(unique_blocks) // 5])
test_mask = np.array([b in test_blocks for b in block_ids])

rf_eval = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_eval.fit(X[~test_mask], y[~test_mask])
y_pred = rf_eval.predict(X[test_mask])
print(classification_report(y[test_mask], y_pred, target_names=class_names,
                            labels=list(range(len(class_names))),
                            digits=3, zero_division=0))
macro_f1 = f1_score(y[test_mask], y_pred, average="macro",
                    labels=list(range(len(class_names))), zero_division=0)
print(f"  공간 블록 홀드아웃 Macro-F1: {macro_f1:.3f}")

# ============================================================
# 8. 단일 장면의 한계 (정직한 보고)
# ============================================================
print("\n--- 8. 이 실습의 범위와 한계 ---")
print("  - 하나의 장면(2021-04-07, 춘천)에서 학습·검증하므로, 다른 지역·")
print("    다른 계절로의 일반화 성능은 이 수치로 보장되지 않는다.")
print("  - 4월 초 장면이라 경작지 상당수가 나지 상태다. WorldCover의")
print("    연간 '경작지' 라벨과 4월 분광이 어긋나, 경작지 정확도가 낮게 나온다.")
print("  - Block CV는 같은 장면 안에서의 공간 누수만 차단한다. 지역 일반화는")
print("    별도 지역의 독립 검증이 필요하다.")

# ============================================================
# 9. 시드 반복 — 4절의 과대추정이 잡음보다 큰가
# ============================================================
# 4절이 보고한 과대추정(+0.010 ~ +0.013)은 같은 절이 함께 보고한 폴드 간
# 표준편차(Block CV 기준 0.015~0.021)보다 작다. 재려는 차이가 측정의
# 흔들림보다 작으면 부호를 해석하는 순간 틀린 이야기가 되므로, 흔들림의
# 크기를 직접 잰다.
#
# 무엇을 흔들지부터 정해야 한다. 과대추정은 "블록을 통째로 빼면 성능이
# 얼마나 떨어지는가"이므로, 이 값을 좌우하는 축은 **196개 블록을 다섯
# 폴드에 어떻게 배정하는가**다. 어느 블록들이 함께 빠지느냐에 따라 검증
# 지역이 학습 지역에서 얼마나 멀어지는지가 달라지기 때문이다. 모델 시드만
# 흔드는 반복은 이 축을 건드리지 않는다. 그래서 그 반복에서 값이 안정적으로
# 나오더라도 "과대추정이 실재한다"는 근거가 되지 못한다.
#
# 세 가지로 나눠 돌린다.
#   (A) 모델 시드만 교체 — RF의 부트스트랩과 XGB·LGBM의 subsample·
#       colsample만 달라진다. 폴드 배정과 픽셀 표본은 본 실행 그대로다.
#       시드 42는 4절을 그대로 재현하므로 자체 점검이 된다.
#   (B) 모델 시드 + 폴드 배정 시드 교체 — 무작위 CV는 KFold의 shuffle
#       시드를, Block CV는 블록→폴드 배정 시드를 바꾼다. 재려는 축을
#       실제로 흔드는 반복이며, 판정의 근거는 이쪽이다.
#   (C) 픽셀 표집 시드 교체 — 클래스별 1,500개를 다시 뽑는다. 자료가
#       바뀌어도 같은 결론이 나오는지 본다(폴드 배정·모델 시드는 고정).
#
# 이 블록은 앞의 계산을 건드리지 않는다. 1~8절이 모두 끝난 뒤에 덧붙으며,
# 모델 객체와 분할기를 새로 만들어 쓰므로 위 수치는 그대로 보존된다.
print("\n--- 9. 시드 반복: 4절의 과대추정이 잡음보다 큰가 ---")

REPEAT_SEEDS = (0, 1, 2, 42)


def make_models(seed):
    """4절과 같은 하이퍼파라미터에 시드만 갈아 끼운 세 모델을 만든다."""
    return {
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=seed),
        "XGB": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            n_jobs=-1, random_state=seed, verbosity=0),
        "LGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=seed, verbose=-1),
    }


def sweep(X_s, y_s, blocks_s, model_seed, rcv, bcv):
    """세 모델의 과대추정과 폴드 간 SD를 한 번에 낸다."""
    over, sd_r, sd_b = [], [], []
    for mdl in make_models(model_seed).values():
        r = cross_val_score(mdl, X_s, y_s, cv=rcv, scoring="f1_macro")
        b = cross_val_score(mdl, X_s, y_s, cv=bcv, groups=blocks_s,
                            scoring="f1_macro")
        over.append(r.mean() - b.mean())
        sd_r.append(r.std())
        sd_b.append(b.std())
    return over, float(np.mean(sd_r)), float(np.mean(sd_b))


def report(title, rows):
    """(시드, 과대추정 3개, SD 2개) 행들을 표로 찍고 범위를 요약한다."""
    print(f"\n  {title}")
    print(f"    {'시드':>4}{'RF':>9}{'XGB':>9}{'LGBM':>9}"
          f"{'폴드SD(무작위)':>14}{'폴드SD(Block)':>14}")
    pooled = []
    for seed, over, sr, sb in rows:
        pooled.extend(over)
        print(f"    {seed:>4}{over[0]:>+9.3f}{over[1]:>+9.3f}{over[2]:>+9.3f}"
              f"{sr:>14.3f}{sb:>14.3f}")
    print(f"    과대추정 범위(3모델×{len(rows)}시드) "
          f"{min(pooled):+.3f} ~ {max(pooled):+.3f}")
    return pooled


# (A) 모델 시드만 — 폴드 배정과 픽셀 표본은 본 실행 그대로
rows_a = []
for s in REPEAT_SEEDS:
    over, sr, sb = sweep(X, y, block_ids, s, random_cv, block_cv)
    rows_a.append((s, over, sr, sb))
pool_a = report("(A) 모델 시드만 교체 — 폴드 배정 고정 (42 = 4절 재현)", rows_a)

# (B) 모델 시드 + 폴드 배정 시드 — 재려는 축을 흔든다
rows_b = []
for s in REPEAT_SEEDS:
    over, sr, sb = sweep(X, y, block_ids, s,
                         KFold(n_splits=5, shuffle=True, random_state=s),
                         GroupKFold(n_splits=5, shuffle=True, random_state=s))
    rows_b.append((s, over, sr, sb))
pool_b = report("(B) 모델 시드 + 블록→폴드 배정 시드 교체", rows_b)

# (C) 픽셀 표집 시드 — 자료 자체를 다시 뽑는다
rows_c = []
for s in REPEAT_SEEDS:
    X_s, y_s, coords_s, _ = build_pixel_table(n_per_class=1500, seed=s)
    blocks_s = ((coords_s[:, 0] // block_size).astype(int) * 100000
                + (coords_s[:, 1] // block_size).astype(int))
    over, sr, sb = sweep(X_s, y_s, blocks_s, 42, random_cv, block_cv)
    rows_c.append((s, over, sr, sb))
pool_c = report("(C) 픽셀 표집 시드 교체 — 폴드 배정·모델 시드 고정", rows_c)

all_over = pool_a + pool_b + pool_c
print(f"\n  → 세 반복을 합치면 과대추정은 {min(all_over):+.3f} ~ {max(all_over):+.3f} "
      f"사이를 오간다.")
print("    4절이 보고한 +0.010~+0.013은 이 범위 안의 한 점이며, Block CV의")
print("    폴드 간 표준편차보다 작다. 부호를 말하기 전에 이 폭을 먼저 봐야 한다.")

print("\n[완료] 앙상블 기법 비교 실습을 마쳤다.")
