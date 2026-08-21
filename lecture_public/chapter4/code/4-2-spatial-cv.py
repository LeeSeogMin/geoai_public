"""
4-2. 공간 자기상관 분석 및 공간 교차검증 비교
=============================================
미리 준비한 공간 포인트 관측 데이터를 불러와, Moran's I로 공간 자기상관을
확인하고 Random CV와 Block CV의 성능 차이를 비교한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter4/code/4-0-simdata-prep.py   # 최초 1회: 데이터 준비
    python practice/chapter4/code/4-2-spatial-cv.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POINTS_PATH = DATA_DIR / "spatial_points.geojson"
if not POINTS_PATH.exists():
    raise SystemExit(
        f"데이터가 없습니다: {POINTS_PATH}\n"
        "먼저 실행: python 4-0-simdata-prep.py"
    )

print("=" * 60)
print("공간 자기상관 분석 및 공간 CV 비교")
print("=" * 60)

# ============================================================
# 1. 공간 포인트 관측 데이터 불러오기
# ============================================================
print("\n--- 1. 공간 포인트 관측 데이터 불러오기 ---")

gdf = gpd.read_file(POINTS_PATH)
x = gdf["x_coord"].values
y = gdf["y_coord"].values
target = gdf["target"].values

# 접근성 분석에서 참조할 관심 지점(핫스팟 후보) 3곳
centers = np.array([
    [320000, 4165000],
    [315000, 4155000],
    [330000, 4170000],
])

print(f"  불러온 포인트: {len(gdf)}개 (좌표계 {gdf.crs})")
print(f"  타겟 범위: {target.min():.1f} ~ {target.max():.1f}")
print(f"  타겟 평균: {target.mean():.1f}, 표준편차: {target.std():.1f}")

# ============================================================
# 2. Global Moran's I 계산
# ============================================================
print("\n--- 2. Global Moran's I ---")

w = KNN.from_dataframe(gdf, k=8)
w.transform = "r"

# 순열 검정의 난수를 고정해 z-점수·LISA 결과를 재현 가능하게 한다
np.random.seed(42)
mi = Moran(gdf["target"].values, w, permutations=999)

print(f"  Moran's I 값: {mi.I:.4f}")
print(f"  기대값 E[I]: {mi.EI:.4f}")
print(f"  Z-score: {mi.z_sim:.4f}")
print(f"  p-value: {mi.p_sim:.4f}")

if mi.I > 0.7:
    print("  해석: 강한 양의 공간 자기상관 → 공간 CV 필수")
elif mi.I > 0.3:
    print("  해석: 중간 양의 공간 자기상관 → 공간 CV 권장")
else:
    print("  해석: 약한 자기상관 → Random CV 사용 가능")

# ============================================================
# 3. Local Moran's I (LISA)
# ============================================================
print("\n--- 3. Local Moran's I (LISA) ---")

lisa = Moran_Local(gdf["target"].values, w, permutations=999)

gdf["lisa_q"] = lisa.q
gdf["lisa_p"] = lisa.p_sim

# 유의 수준 0.05 기준
sig_mask = gdf["lisa_p"] < 0.05
labels = {1: "HH (Hot Spot)", 2: "LH (Low-High)", 3: "LL (Cold Spot)", 4: "HL (High-Low)"}

print("  유의한 클러스터 분류 (p < 0.05):")
for q_val, label in labels.items():
    count = ((gdf["lisa_q"] == q_val) & sig_mask).sum()
    print(f"    {label}: {count}개")

not_sig = (~sig_mask).sum()
print(f"    Not Significant: {not_sig}개")

# ============================================================
# 4. 피처 구성
# ============================================================
print("\n--- 4. 피처 구성 ---")

# 좌표 + 핫스팟 거리를 피처로
for i, (cx, cy) in enumerate(centers):
    gdf[f"dist_center_{i}"] = np.sqrt((x - cx)**2 + (y - cy)**2)

feature_cols = ["x_coord", "y_coord", "dist_center_0", "dist_center_1", "dist_center_2"]
X = gdf[feature_cols].values
y_target = gdf["target"].values

print(f"  피처 수: {len(feature_cols)}")
print(f"  피처: {feature_cols}")

# ============================================================
# 5. Random CV vs Block CV 비교
# ============================================================
print("\n--- 5. Random CV vs Block CV 비교 ---")

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Random K-Fold CV
random_cv = KFold(n_splits=5, shuffle=True, random_state=42)
random_scores = cross_val_score(model, X, y_target, cv=random_cv, scoring="r2")

print(f"  Random K-Fold CV:")
print(f"    R² scores: [{', '.join(f'{s:.3f}' for s in random_scores)}]")
print(f"    평균 R²: {random_scores.mean():.3f} +/- {random_scores.std():.3f}")

# Block CV (5km 블록)
block_size = 5000
gdf["block_id"] = (
    (gdf.geometry.x // block_size).astype(int) * 10000
    + (gdf.geometry.y // block_size).astype(int)
)

n_blocks = gdf["block_id"].nunique()
print(f"\n  블록 수: {n_blocks}개 (5km 격자)")

block_cv = GroupKFold(n_splits=5)
block_scores = cross_val_score(
    model, X, y_target, cv=block_cv, groups=gdf["block_id"], scoring="r2"
)

print(f"  Spatial Block CV:")
print(f"    R² scores: [{', '.join(f'{s:.3f}' for s in block_scores)}]")
print(f"    평균 R²: {block_scores.mean():.3f} +/- {block_scores.std():.3f}")

# ============================================================
# 6. Cluster CV (K-means 기반)
# ============================================================
print("\n--- 6. Cluster CV ---")

from sklearn.cluster import KMeans

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
gdf["cluster_id"] = kmeans.fit_predict(coords)

cluster_cv = GroupKFold(n_splits=5)
cluster_scores = cross_val_score(
    model, X, y_target, cv=cluster_cv, groups=gdf["cluster_id"], scoring="r2"
)

print(f"  Cluster CV:")
print(f"    R² scores: [{', '.join(f'{s:.3f}' for s in cluster_scores)}]")
print(f"    평균 R²: {cluster_scores.mean():.3f} +/- {cluster_scores.std():.3f}")

# ============================================================
# 7. 결과 비교 요약
# ============================================================
print("\n--- 7. CV 기법 비교 요약 ---")

print(f"  {'기법':15s} | {'평균 R²':>8s} | {'표준편차':>8s} | {'과대추정':>8s}")
print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*8}")

baseline = block_scores.mean()
for name, scores in [
    ("Random CV", random_scores),
    ("Block CV", block_scores),
    ("Cluster CV", cluster_scores),
]:
    diff = scores.mean() - baseline
    print(f"  {name:15s} | {scores.mean():>8.3f} | {scores.std():>8.3f} | {diff:>+8.3f}")

print(f"\n  Random CV - Block CV = {random_scores.mean() - block_scores.mean():.3f}")
print(f"  Random CV가 Block CV 대비 성능을 과대 추정")

if random_scores.mean() > block_scores.mean():
    pct = (random_scores.mean() - block_scores.mean()) / block_scores.mean() * 100
    print(f"  과대 추정 비율: {pct:.1f}%")

# ============================================================
# 8. 시드 반복 — 위 사다리가 실행마다 얼마나 흔들리는가
# ============================================================
# 7절이 보고한 과대추정 +0.071(Random−Block)과 0.520(Random−Cluster)도
# 실행 간 변동을 재고 나서 써야 한다. 눈에 걸리는 것이 이미 하나 있다 —
# Block CV의 폴드 간 표준편차가 0.143으로, 재려는 차이 0.071의 두 배다.
#
# 무엇을 흔들지부터 정한다. 이 사다리가 재는 것은 "학습에 쓰인 지역을
# 통째로 빼면 성능이 얼마나 떨어지는가"이고, 그 답을 정하는 것은
# ① 포인트가 어디에 찍혔는가(자료) ② 어느 지역이 한 폴드로 묶이는가(분할)다.
# 그래서 (C) 생성 시드를 주축으로 두고 (A)·(B)를 대조군으로 놓는다.
#   (A) 모델 시드 — RandomForest의 random_state만 교체.
#   (B) 분할 시드 — 무작위 CV의 shuffle 시드, 블록→폴드 배정 시드,
#       군집 정의(KMeans) 시드를 함께 교체.
#   (C) 생성 시드 — 500개 포인트를 다시 뽑는다. 4-0-simdata-prep.py의
#       prepare_spatial_points()와 같은 식을 메모리에서 다시 계산한다.
#       시드 42가 저장 자료를 그대로 재현하는지 곧바로 대조한다.
#
# 각 축이 실제로 무언가를 바꾸는지도 함께 확인한다. 난수를 쓰지 않는
# 설정에 시드만 갈아 끼우면 표는 길어져도 아무것도 반복되지 않는다.
#
# 이 블록은 앞의 계산을 건드리지 않는다. 1~7절이 모두 끝난 뒤에 덧붙으며
# 자료·분할기·모델을 새로 만들어 쓴다.
print("\n--- 8. 시드 반복: 위 사다리가 실행마다 얼마나 흔들리는가 ---")

REPEAT_SEEDS = (0, 1, 2, 42)
CENTER_INTENSITY = [30, 25, 35]


def regenerate_points(seed):
    """4-0-simdata-prep.py의 prepare_spatial_points()와 같은 식으로 다시 뽑는다.

    원본은 시드 42를 geojson으로 굳혀 두므로, 생성 시드를 바꾸려면 같은 식을
    메모리에서 다시 계산해야 한다. 원본이 전역 np.random.seed()를 쓰지만
    여기서는 앞 절들의 난수 상태를 건드리지 않도록 독립 RandomState를 쓴다
    (같은 알고리즘·같은 시드이므로 수열은 동일하다).
    난수 소비 순서(x → y → 잡음)를 원본과 똑같이 유지해야 한다.
    """
    rs = np.random.RandomState(seed)
    n_points = 500
    gx = rs.uniform(310000, 340000, n_points)
    gy = rs.uniform(4150000, 4180000, n_points)
    x_norm = (gx - gx.min()) / (gx.max() - gx.min())
    spatial_trend = x_norm * 50
    cluster_effect = np.zeros(n_points)
    for (cx, cy), intensity in zip(centers, CENTER_INTENSITY):
        dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        cluster_effect += intensity * np.exp(-dist / 3000)
    tgt = spatial_trend + cluster_effect + rs.normal(0, 5, n_points)
    feats = np.column_stack(
        [gx, gy] + [np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) for cx, cy in centers])
    return feats, tgt, gx, gy


chk_X, chk_y, _, _ = regenerate_points(42)
print(f"  생성식 대조: 시드 42로 다시 뽑은 타깃과 저장 자료의 최대 차이 = "
      f"{np.abs(chk_y - y_target).max():.2e}")
print("    (0에 가까워야 아래 (C) 반복이 원본과 같은 식을 흔든 것이다)")


def ladder(X_l, y_l, px, py, model_seed, kf_l, gkf_l, km_seed):
    """한 벌의 자료·분할·모델 시드에서 Random·Block·Cluster R²와 폴드 SD를 낸다."""
    blocks_l = (px // block_size).astype(int) * 10000 + (py // block_size).astype(int)
    clus_l = KMeans(n_clusters=5, random_state=km_seed, n_init=10).fit_predict(
        np.column_stack([px, py]))
    mdl = RandomForestRegressor(n_estimators=100, max_depth=10,
                                random_state=model_seed)
    r = cross_val_score(mdl, X_l, y_l, cv=kf_l, scoring="r2")
    b = cross_val_score(mdl, X_l, y_l, cv=gkf_l, groups=blocks_l, scoring="r2")
    c = cross_val_score(mdl, X_l, y_l, cv=gkf_l, groups=clus_l, scoring="r2")
    return r.mean(), b.mean(), c.mean(), b.std(), c.std()


def show(title, runs):
    """축 하나의 결과를 표로 찍고 두 차이의 범위를 요약한다."""
    print(f"\n  {title}")
    print(f"    {'시드':>4}{'Random':>9}{'Block':>9}{'Cluster':>9}"
          f"{'R−B':>9}{'R−C':>9}{'폴드SD(B/C)':>15}")
    rb, rc = [], []
    for seed, (r, b, c, sb, sc) in runs:
        rb.append(r - b)
        rc.append(c - r)
        print(f"    {seed:>4}{r:>9.3f}{b:>9.3f}{c:>9.3f}{r - b:>+9.3f}"
              f"{c - r:>+9.3f}{f'{sb:.3f}/{sc:.3f}':>15}")
    print(f"    R−B 범위 {min(rb):+.3f} ~ {max(rb):+.3f}"
          f"  (폭 {max(rb) - min(rb):.3f})")
    print(f"    R−C 범위 {min(rc):+.3f} ~ {max(rc):+.3f}"
          f"  (폭 {max(rc) - min(rc):.3f})")
    if max(rb) - min(rb) < 1e-9 and max(rc) - min(rc) < 1e-9:
        print("    ⚠ 이 축은 아무것도 바꾸지 않았다 — 대조군이 아니라 빈칸이다.")
    return rb, rc


kf0 = KFold(n_splits=5, shuffle=True, random_state=42)
gkf0 = GroupKFold(n_splits=5)
px0, py0 = gdf.geometry.x.values, gdf.geometry.y.values

runs_a = [(s, ladder(X, y_target, px0, py0, s, kf0, gkf0, 42)) for s in REPEAT_SEEDS]
rb_a, rc_a = show("(A) 모델 시드만 교체 — 자료·분할 고정 (42 = 위 실행 재현)", runs_a)

runs_b = [(s, ladder(X, y_target, px0, py0, 42,
                     KFold(n_splits=5, shuffle=True, random_state=s),
                     GroupKFold(n_splits=5, shuffle=True, random_state=s), s))
          for s in REPEAT_SEEDS]
rb_b, rc_b = show("(B) 분할 시드 교체 — 무작위 CV·블록 배정·군집 정의", runs_b)

runs_c = []
for s in REPEAT_SEEDS:
    gX, gy_t, gpx, gpy = regenerate_points(s)
    runs_c.append((s, ladder(gX, gy_t, gpx, gpy, 42, kf0, gkf0, 42)))
rb_c, rc_c = show("(C) 생성 시드 교체 — 포인트를 다시 뽑는다 (주축)", runs_c)

all_rb = rb_a + rb_b + rb_c
all_rc = rc_a + rc_b + rc_c
print(f"\n  → 12번(4시드×3축) 합계 — Random−Block {min(all_rb):+.3f} ~ {max(all_rb):+.3f}, "
      f"Random−Cluster {min(all_rc):+.3f} ~ {max(all_rc):+.3f}")
print("    두 차이는 같은 사다리에서 나왔지만 판정이 다를 수 있다.")
print("    각각의 범위가 0을 품는지, 폴드 SD와 견주어 어느 쪽인지 따로 본다.")

print("\n[완료] 공간 자기상관 분석 및 공간 CV 비교 실습을 마쳤다.")
