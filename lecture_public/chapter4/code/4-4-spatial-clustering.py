"""
4-4. 비지도 군집 실습 (실제 Sentinel-2 픽셀)
============================================
실제 Sentinel-2 L2A 12밴드 클립의 픽셀에 K-means와 DBSCAN을 적용해
정답 라벨 없이 분광 구조를 발견한다. 발견한 군집을 ESA WorldCover
실측 토지피복과 대조해, 비지도 군집이 정답을 보지 않고도 실제
토지피복 구분을 얼마나 재현하는지 조정 랜드 지수로 확인한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python lecture_practice/chapter4/code/4-4-spatial-clustering.py
    # 데이터가 없으면 먼저: python lecture_practice/chapter4/code/4-0-data-download.py
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from _s2_data import build_pixel_table

print("=" * 60)
print("비지도 군집: K-means vs DBSCAN (실제 Sentinel-2)")
print("=" * 60)

# ============================================================
# 1. 실제 픽셀 표본 + 표준화
# ============================================================
print("\n--- 1. 실제 픽셀 표본 ---")
X, y_true, coords, info = build_pixel_table(n_per_class=800, seed=42)
class_names = info["class_names"]
feature_names = info["feature_names"]
print(f"  표본 수: {len(X):,}개, 피처 {X.shape[1]}개")
print(f"  실제 토지피복(WorldCover) 클래스: {class_names}")

# 거리 기반 군집이므로 표준화는 필수
Xs = StandardScaler().fit_transform(X)

# ============================================================
# 2. 엘보·실루엣으로 군집 수 탐색
# ============================================================
print("\n--- 2. 엘보·실루엣 분석 ---")
k_range = range(2, 8)
print(f"  {'K':>3s} | {'Inertia':>12s} | {'실루엣':>8s}")
print(f"  {'-'*3} | {'-'*12} | {'-'*8}")
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lab = km.fit_predict(Xs)
    sil = silhouette_score(Xs, lab)
    sil_scores.append(sil)
    print(f"  {k:>3d} | {km.inertia_:>12.0f} | {sil:>8.3f}")
best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\n  최적 K (실루엣 기준): {best_k}")

# ============================================================
# 3. K-means (실제 클래스 수 = 5로 고정)
# ============================================================
print(f"\n--- 3. K-means (K={len(class_names)}, 실제 클래스 수) ---")
km = KMeans(n_clusters=len(class_names), n_init=10, random_state=42)
km_labels = km.fit_predict(Xs)
print(f"  실루엣 점수: {silhouette_score(Xs, km_labels):.3f}")
for k in range(len(class_names)):
    print(f"    군집 {k}: {(km_labels == k).sum()}개")

# ============================================================
# 4. 군집 vs 실제 토지피복 대조
# ============================================================
print("\n--- 4. 군집 vs 실제 토지피복(WorldCover) ---")
ari = adjusted_rand_score(y_true, km_labels)
nmi = normalized_mutual_info_score(y_true, km_labels)
print(f"  조정 랜드 지수(ARI): {ari:.3f}  (1.0=완전 일치, 0=우연 수준)")
print(f"  정규화 상호정보(NMI): {nmi:.3f}")

print(f"\n  실제 클래스 × K-means 군집 (분할표):")
print(f"  {'':8s}", end="")
for k in range(len(class_names)):
    print(f" | 군집{k:>2d}", end="")
print()
for c, name in enumerate(class_names):
    print(f"  {name:8s}", end="")
    for k in range(len(class_names)):
        print(f" | {((y_true == c) & (km_labels == k)).sum():>5d}", end="")
    print()

# ============================================================
# 5. 군집별 분광 특성 (NDVI·NDWI·NDBI로 해석)
# ============================================================
print("\n--- 5. K-means 군집별 분광 특성 ---")
i_ndvi = feature_names.index("NDVI")
i_ndwi = feature_names.index("NDWI")
i_ndbi = feature_names.index("NDBI")
print(f"  {'군집':>4s} | {'NDVI':>7s} | {'NDWI':>7s} | {'NDBI':>7s} | 해석")
print(f"  {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} |")
for k in range(len(class_names)):
    m = X[km_labels == k].mean(axis=0)
    ndvi, ndwi, ndbi = m[i_ndvi], m[i_ndwi], m[i_ndbi]
    if ndwi > 0.1:
        interp = "수역(높은 NDWI)"
    elif ndvi > 0.5:
        interp = "식생(높은 NDVI)"
    elif ndbi > 0:
        interp = "시가지·나지(양의 NDBI)"
    else:
        interp = "혼합·저식생"
    print(f"  {k:>4d} | {ndvi:>7.3f} | {ndwi:>7.3f} | {ndbi:>7.3f} | {interp}")

# ============================================================
# 6. DBSCAN (밀도 기반 군집·노이즈 분리)
# ============================================================
print("\n--- 6. DBSCAN ---")
db_labels = DBSCAN(eps=1.2, min_samples=15).fit_predict(Xs)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = int((db_labels == -1).sum())
print(f"  발견된 군집 수: {n_clusters}")
print(f"  노이즈 포인트: {n_noise}개 ({100 * n_noise / len(db_labels):.1f}%)")
print("  DBSCAN이 찾은 밀집 지역은 군집화의 산물이지, 통계적으로 유의한")
print("  핫스팟이라는 증명은 아니다(유의성 검정은 8장 LISA·공간 스캔).")

print("\n[완료] 비지도 군집 실습을 마쳤다.")
