"""
문제 3 (4장). 새 지역에서도 이 점수가 나오는가
===============================================
500개 관측점(spatial_points.geojson) 가운데 학번으로 400개를 뽑아,
접근성 지표(target)를 예측하는 랜덤 포레스트의 R2를 보고한다.
보고서의 문장: "이 모델은 아직 조사하지 않은 새 지역에서도 이 점수를 낸다."

피처 사전
  x_coord, y_coord     관측점 좌표(m)
  dist_center_0..2     관심 지점 3곳까지 거리(m)
  survey_score         사후 만족도 조사 점수 -- 지표가 확정된 뒤 현장 조사로 얻음

실행: python exam/q3.py --id 학번
"""

import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from common import PRACTICE, key, parse_id, rng_of

sid = parse_id()
rng = rng_of(sid, 3)
gdf = gpd.read_file(PRACTICE / "chapter4" / "data" / "spatial_points.geojson")
sub = gdf.iloc[rng.choice(len(gdf), 400, replace=False)].reset_index(drop=True)
sub["survey_score"] = sub["target"].values + rng.normal(0, 3, len(sub))

centers = np.array([[320000, 4165000], [315000, 4155000], [330000, 4170000]])
x, y = sub["x_coord"].values, sub["y_coord"].values
for i, (cx, cy) in enumerate(centers):
    sub[f"dist_center_{i}"] = np.hypot(x - cx, y - cy)

features = ["x_coord", "y_coord", "dist_center_0", "dist_center_1", "dist_center_2", "survey_score"]
X, t = sub[features].values, sub["target"].values
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, t, cv=cv, scoring="r2")

print(f"관측점 {len(sub)}개, 피처 {features}")
print(f"5겹 R2: {np.round(scores, 3).tolist()}")
print(f"평균 R2 = {scores.mean():.3f}")
print("보고: 새 지역에서도 이 점수를 낸다.")
key("r2_random", float(scores.mean()))
