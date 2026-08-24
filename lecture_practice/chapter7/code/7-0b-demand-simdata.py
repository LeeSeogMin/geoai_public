"""
7-0b. 실습 데이터 준비 2: 점포별 일 수요 패널 (7-3용)
=========================================================
7-3(매출 예측과 발주량 결정)이 쓰는 점포×품목×일 단위 수요 패널을 만들어
`data/` 폴더에 저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행하고,
7-3은 저장된 파일을 불러와 예측·결정만 수행한다.

왜 합성인가
-----------
점포 단위 일별 판매량·폐기량·원가 구조는 기업 내부 자료이고 공개된 것이 없다.
게다가 발주 정책의 손익을 채점하려면 "그날 실제로 팔릴 수 있었던 수요"의 참값이
있어야 한다. 실제 매출 자료는 발주량에 잘린 값(품절이면 수요를 관측하지 못한다)
이라 참값을 주지 못한다. 그래서 참값을 아는 합성 패널을 쓴다.

주의: 이 파일은 7-0(교통량·레이어 카탈로그)과 **완전히 분리된 난수열**을 쓴다.
7-0의 난수 소비 순서를 건드리면 7-1·7-2의 기존 실행 결과가 흔들리므로,
파일도 시드도 따로 둔다.

무엇을 심었나 (7-3의 대조군이 검사할 메커니즘)
-----------------------------------------------
1. 요일 주기 — 상권(도심/주거/외곽)에 따라 주중·주말 피크가 뒤집힌다.
2. 연 주기 — 우산은 장마철에 크게 오르고, 도시락은 완만하다.
3. 이벤트 — 프로모션(수요 증가), 공휴일(도심은 감소, 주거는 증가).
4. **이분산** — 점포마다 예측 난이도가 다르고(φ), 수요 수준이 높은 날일수록
   잡음의 폭이 크다. 7-3의 정규화 conformal이 이것을 잡아내는지 검사한다.

대조군용 데이터(C1)
-------------------
위 4번만 제거한 등분산 패널을 함께 저장한다. 평균 곡면과 표준정규 난수는
그대로 두고 잡음의 **크기 구조만** 상수로 바꾼다(전체 분산은 맞춘다).
7-3은 두 패널에 같은 파이프라인을 돌려 정규화의 이득이 정말 이분산에서
오는지 확인한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    python lecture_practice/chapter7/code/7-0b-demand-simdata.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 20260813          # 7-0(seed 42)과 겹치지 않는 독립 시드
DAYS = 1460              # 4년치 일 단위 — 보정·시험 구간이 각각 계절을 대부분 덮도록
N_STORES = 6

# 점포 설정: 가상 도시(0~10,000m, 7-1과 같은 무대)에 6개 점포
# scale = 점포 규모, phi = 예측 난이도(잡음 배율) — 이 둘은 서로 독립이다
STORES = [
    # (id,      상권,   x,    y,   규모, 잡음배율)
    ("S1", "도심", 2400, 7600, 1.30, 0.60),
    ("S2", "도심", 3100, 6800, 1.10, 1.50),
    ("S3", "주거", 6900, 5200, 1.00, 0.90),
    ("S4", "주거", 7600, 4300, 0.95, 1.90),
    ("S5", "외곽", 1500, 1800, 0.80, 1.00),
    ("S6", "외곽", 8800, 1200, 0.70, 1.20),
]

# 품목 설정 — 두 품목은 손실 구조가 정반대다(7-3에서 임계비로 계산)
ITEMS = {
    "도시락": dict(base=80.0, noise_a=2.0, noise_b=0.16),
    "우산": dict(base=12.0, noise_a=0.8, noise_b=0.22),
}

# 상권별 요일 배수 (월~일). 도심은 주중 피크, 주거는 주말 피크
DOW_FACTOR = {
    "도심": np.array([1.18, 1.20, 1.18, 1.16, 1.22, 0.62, 0.48]),
    "주거": np.array([0.92, 0.90, 0.92, 0.95, 1.10, 1.25, 1.20]),
    "외곽": np.array([1.00, 0.98, 1.00, 1.02, 1.10, 1.05, 0.92]),
}


def mean_surface(item, store, day, doy, dow, promo, holiday):
    """관측 가능한 요인만으로 만든 평균 수요 곡면(참값의 결정론적 부분)."""
    cfg = ITEMS[item]
    _, region, _, _, scale, _ = store
    level = cfg["base"] * scale

    dow_mult = DOW_FACTOR[region][dow]

    if item == "도시락":
        # 완만한 연 주기(봄·가을 소폭 상승)와 완만한 상승 추세
        season = 1.0 + 0.08 * np.sin(2 * np.pi * (doy - 100) / 365.0)
        trend = 1.0 + 0.00008 * day
        promo_mult = np.where(promo, 1.25, 1.0)
        # 공휴일: 도심 사무상권은 급감, 주거는 소폭 증가
        hol_mult = np.where(holiday, 0.60 if region == "도심" else 1.10, 1.0)
    else:  # 우산 — 장마철에 크게 오른다
        season = 1.0 + 1.20 * np.exp(-(((doy - 190) / 45.0) ** 2)) \
                     + 0.35 * np.exp(-(((doy - 95) / 30.0) ** 2))
        trend = np.ones_like(np.asarray(day, dtype=float))
        promo_mult = np.where(promo, 1.15, 1.0)
        hol_mult = np.ones_like(np.asarray(day, dtype=float))

    return level * dow_mult * season * trend * promo_mult * hol_mult


def build_panel():
    """이분산 패널과 등분산 대조 패널을 함께 만든다."""
    rng = np.random.default_rng(SEED)

    day = np.arange(DAYS)
    doy = day % 365 + 1
    dow = day % 7                       # 0 = 월요일
    # 프로모션: 품목 무관하게 점포 공통 캘린더(약 8%의 날)
    promo = rng.random(DAYS) < 0.08
    # 공휴일: 연 15일을 무작위로 배치(관측 가능한 더미)
    holiday = np.zeros(DAYS, dtype=bool)
    for yr in range(DAYS // 365 + 1):
        picks = rng.choice(365, size=15, replace=False) + yr * 365
        holiday[picks[picks < DAYS]] = True

    rows_het, rows_hom = [], []
    for item in ITEMS:
        cfg = ITEMS[item]
        # 표준정규 난수는 한 번만 뽑아 두 패널에 **똑같이** 쓴다.
        # 이렇게 해야 두 패널의 차이가 오직 '잡음 크기 구조'뿐이다.
        z = rng.standard_normal((N_STORES, DAYS))

        mu = np.zeros((N_STORES, DAYS))
        sd_het = np.zeros((N_STORES, DAYS))
        for i, store in enumerate(STORES):
            phi = store[5]
            mu[i] = mean_surface(item, store, day, doy, dow, promo, holiday)
            # 이분산: 점포 난이도 × (기본 + 수요 수준 비례)
            sd_het[i] = phi * (cfg["noise_a"] + cfg["noise_b"] * mu[i])

        # 등분산 대조: 전체 분산을 맞춘 단일 상수로 대체(구조만 제거)
        sd_hom = np.full_like(sd_het, np.sqrt(np.mean(sd_het ** 2)))

        for label, sd, bucket in (("het", sd_het, rows_het), ("hom", sd_hom, rows_hom)):
            demand = np.clip(np.rint(mu + sd * z), 0, None).astype(int)
            for i, store in enumerate(STORES):
                sid, region, x, y, _, _ = store
                bucket.append(pd.DataFrame({
                    "store_id": sid, "region": region, "x": x, "y": y,
                    "item": item, "day": day, "doy": doy, "dow": dow,
                    "promo": promo.astype(int), "holiday": holiday.astype(int),
                    "demand": demand[i],
                }))
            del label

    return pd.concat(rows_het, ignore_index=True), pd.concat(rows_hom, ignore_index=True)


def summarize(df, title):
    print(f"\n[{title}] {len(df):,}행 = 점포 {df.store_id.nunique()} × "
          f"품목 {df['item'].nunique()} × {df.day.nunique()}일")
    g = df.groupby(["item", "store_id"])["demand"]
    tab = pd.DataFrame({"평균": g.mean().round(1), "표준편차": g.std().round(1)})
    tab["변동계수"] = (tab["표준편차"] / tab["평균"]).round(3)
    print(tab.to_string())


def main():
    print("=" * 66)
    print("실습 데이터 준비 2: 점포별 일 수요 패널 (7-3용)")
    print("=" * 66)
    het, hom = build_panel()

    het.to_parquet(DATA_DIR / "store_demand.parquet", index=False)
    hom.to_parquet(DATA_DIR / "store_demand_homoskedastic.parquet", index=False)

    summarize(het, "본 데이터 — 이분산")
    summarize(hom, "대조군 C1 — 등분산(잡음 크기 구조만 제거)")

    print("\n두 패널의 평균 수요는 같고 잡음의 '크기 구조'만 다르다:")
    for item in ITEMS:
        a = het.loc[het["item"] == item, "demand"].mean()
        b = hom.loc[hom["item"] == item, "demand"].mean()
        sa = het.loc[het["item"] == item, "demand"].std()
        sb = hom.loc[hom["item"] == item, "demand"].std()
        print(f"  {item}: 평균 {a:.2f} vs {b:.2f} / 표준편차 {sa:.2f} vs {sb:.2f}")

    print("\n  → store_demand.parquet, store_demand_homoskedastic.parquet")
    print("[완료] 7-3이 쓸 수요 패널을 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
