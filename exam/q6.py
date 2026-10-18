"""
문제 6 (7장). 발주량을 어떻게 정해야 손실이 줄어드는가
========================================================
점포 6곳 x 품목 2종의 일별 수요(store_demand.parquet)에서, 학번으로 점포 하나와 품목 하나를 정한다.
요일별 평균 수요로 예측하고, 예측값을 그대로 발주한다.
결품 1단위 손실 Cu, 잉여 1단위 손실 Co는 7장 8.1절의 가정값이다.

보고서의 문장: "검증용 30%를 무작위로 떼어 평가했으므로 실제 운영에서도 이 손실이 난다."

실행: python exam/q6.py --id 학번
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PRACTICE, key, parse_id, seed_of

COSTS = {"도시락": (1200, 2000), "우산": (15000, 1500)}   # (Cu, Co)

sid = parse_id()
s = seed_of(sid)
df = pd.read_parquet(PRACTICE / "chapter7" / "data" / "store_demand.parquet")
stores = sorted(df["store_id"].unique())
items = sorted(df["item"].unique())
store, item = stores[s % len(stores)], items[(s // len(stores)) % len(items)]
g = df[(df["store_id"] == store) & (df["item"] == item)].sort_values("day").reset_index(drop=True)
Cu, Co = COSTS[item]
cr = Cu / (Cu + Co)

train, test = train_test_split(g, test_size=0.3, random_state=s % 1000, shuffle=True)
dow_mean = train.groupby("dow")["demand"].mean()
forecast = test["dow"].map(dow_mean).values
order = np.round(forecast)                      # 예측값을 그대로 발주한다

short = np.maximum(test["demand"].values - order, 0)
over = np.maximum(order - test["demand"].values, 0)
cost = (Cu * short + Co * over).mean()

print(f"점포 {store}, 품목 {item}, Cu={Cu}, Co={Co}, 임계비={cr:.4f}")
print(f"학습 {len(train)}일 / 검증 {len(test)}일 (무작위 분할)")
print(f"일평균 손실: {cost:,.0f}원")
print("보고: 이 손실이 실제 운영에서도 난다.")
key("critical_ratio", float(cr))
key("cost_per_day", int(round(cost)))
