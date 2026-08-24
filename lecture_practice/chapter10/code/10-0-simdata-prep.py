"""
10-0. 실습 데이터 준비: 도시·생활권·민원 행정 분석용 데이터셋 저장
=====================================================================
이 장의 10-1·10-2 실습이 쓰는 데이터셋을 미리 만들어 `data/` 폴더에 저장한다.
학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고, 이후 각 실습
코드는 저장된 파일을 불러와 분석만 수행한다.

여기서 만드는 두 데이터는 실제 공개 데이터를 그대로 구하기 어려운 대상이라
현실적인 값과 공간 구조를 갖도록 합성한 교육용 데이터다(개인 식별 불가, 격자 집계).
  - 생활SOC 접근성(격자 인구 + 시설 위치, 10-1)
      실제 분석에서는 SGIS 통계격자 인구 + 생활SOC 시설 위치(공공데이터포털)를 쓴다.
  - 민원 시공간 패널(도로 공변량 + 주별 민원 발생, 10-2)
      실제 분석에서는 국민신문고·범정부 민원분석시스템(국민권익위) 자료를 쓴다.

RDD 실습(10-3)도 참 규제효과 τ를 아는 합성 자료로 추정량을 검증하는 예제다.
데이터(정상 표본·조작 표본)를 여기서 미리 만들어 저장하고, 10-3은 이를 불러와
추정·진단만 수행한다. 참 τ를 안다는 '설계 의도'는 교재·코드 주석에 남긴다.

부동소수점과 격자 구조를 정확히 보존하도록 표 데이터는 Parquet로 저장한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python lecture_practice/chapter10/code/10-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ===========================================================
# 10-1. 생활SOC 접근성: 격자 인구 + 시설 위치
# ===========================================================
SOC_GRID_ROWS, SOC_GRID_COLS = 20, 15   # 300개 격자를 20×15로 배치
SOC_N = SOC_GRID_ROWS * SOC_GRID_COLS
# 생활SOC 4종과 종류별 시설 수(도시 전역에 분산 공급, 종류별 공급량 차이)
SOC_TYPES = {"도서관": 6, "돌봄": 10, "공원": 12, "체육": 8}


def prepare_living_soc():
    """격자 인구와 생활SOC 시설 위치를 만들어 저장한다(10-1).

    핵심 설계(현실적 불평등 주입): 인구는 도심(중앙)에 집중되고, 시설은
    인구가 많은 곳에 우선 공급되는 경향이 있다(예산·수요 논리).
    → 외곽 저밀도 지역이 접근성 사각지대가 되는 구조를 만든다.
    """
    rng = np.random.default_rng(42)
    rows, cols = np.divmod(np.arange(SOC_N), SOC_GRID_COLS)
    cr, cc = (SOC_GRID_ROWS - 1) / 2.0, (SOC_GRID_COLS - 1) / 2.0
    dist_center = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)

    # 인구: 도심 집중(중앙일수록 많음) + 개별 변동
    pop = np.clip(
        18000 * np.exp(-(dist_center / 6.0) ** 2) + rng.normal(0, 2500, SOC_N),
        800, None,
    ).round()

    # 시설 위치: 인구 비례 확률로 배치(공급이 수요를 따라가는 경향 → 외곽 사각지대)
    prob = pop / pop.sum()
    fac_records = []
    for soc, n_fac in SOC_TYPES.items():
        idx = rng.choice(SOC_N, size=n_fac, replace=False, p=prob)
        fac_records.extend((soc, int(c)) for c in idx)

    grid = pd.DataFrame({"cell": np.arange(SOC_N), "row": rows, "col": cols, "pop": pop})
    grid.to_parquet(DATA_DIR / "soc_grid.parquet", index=False)
    fac = pd.DataFrame(fac_records, columns=["soc", "facility_cell"])
    fac.to_parquet(DATA_DIR / "soc_facilities.parquet", index=False)
    print(f"  격자 {len(grid)}개(20×15, 인구 {int(pop.sum()):,}명) → soc_grid.parquet")
    print("  생활SOC 시설: " + ", ".join(f"{k} {v}개소" for k, v in SOC_TYPES.items())
          + " → soc_facilities.parquet")


# ===========================================================
# 10-2. 민원 시공간 패널: 도로 공변량 + 주별 민원 발생
# ===========================================================
CMP_GRID = 16
CMP_N = CMP_GRID * CMP_GRID      # 256개 격자
CMP_T = 40                       # 40주
BASE = 1.2                       # 기저 민원 강도
ROAD_COEF = 4.0                  # 도로·활동 밀도가 민원을 끌어올리는 정도
ALPHA = 0.35                     # 시간 지속성(자기 과거 민원 → 이번 주)
KAPPA = 0.35                     # 공간 확산(이웃 과거 평균 → 이번 주) ★ 공간 신호의 핵심
SEASON_AMP = 0.25                # 계절성 진폭
# 정상성(stationarity) 조건: ALPHA + KAPPA = 0.70 < 1 → 민원 수가 발산하지 않고
# 평형 수준에서 안정. 이웃은 합이 아니라 '평균'을 써야 실효 계수가 1을 넘지 않는다.


def _neighbor_sum(mat):
    """각 격자의 4-이웃 합. mat: (CMP_GRID, CMP_GRID)."""
    s = np.zeros_like(mat, dtype=float)
    s[1:, :] += mat[:-1, :]
    s[:-1, :] += mat[1:, :]
    s[:, 1:] += mat[:, :-1]
    s[:, :-1] += mat[:, 1:]
    return s


# 격자별 유효 이웃 수(내부 4, 가장자리 3, 모서리 2) — 이웃 평균 계산용
_NCOUNT = _neighbor_sum(np.ones((CMP_GRID, CMP_GRID)))


def _neighbor_mean(mat):
    """각 격자의 4-이웃 평균(공간 시차). 합이 아닌 평균이라 발산하지 않는다."""
    return _neighbor_sum(mat) / _NCOUNT


def prepare_complaints():
    """민원 발생을 시공간 과정으로 생성해 저장한다(10-2).

    road: 관측 가능한 정적 공변량(도로·활동 밀도) — 예측 피처로 사용.
    report_bias: 신고 성향(편향) — 관측 라벨에 영향을 주지만 '진짜 수요'와는 다르다.
                 (규칙3: 민원 多가 곧 문제 多는 아니다) — 생성에만 쓰고 저장하지 않는다.
    obs[t] ~ Poisson(BASE + ROAD_COEF*road + ALPHA*obs[t-1] + KAPPA*neigh(obs[t-1]) + 계절성)
             × report_bias
    """
    rng = np.random.default_rng(42)

    def smooth_field(freq_r, freq_c, noise_sd):
        """공간 자기상관을 갖는 0~1 정규화 장(field)."""
        rr, cc = np.meshgrid(np.linspace(0, 1, CMP_GRID), np.linspace(0, 1, CMP_GRID),
                             indexing="ij")
        f = np.sin(2 * np.pi * freq_r * rr) + np.cos(2 * np.pi * freq_c * cc)
        f = f + rng.normal(0, noise_sd, (CMP_GRID, CMP_GRID))
        return (f - f.min()) / (f.max() - f.min())

    road = smooth_field(1.3, 1.1, 0.25)          # 도심·간선도로 패턴(관측 가능)
    report_bias = 0.7 + 0.6 * smooth_field(2.1, 1.7, 0.3)   # 0.7~1.3, 독립 패턴(관측 불가)

    obs = np.zeros((CMP_T, CMP_GRID, CMP_GRID))
    prev = BASE + ROAD_COEF * road               # 초기 강도
    for t in range(CMP_T):
        season = 1.0 + SEASON_AMP * np.sin(2 * np.pi * t / 26.0)   # 반기 계절성
        # 정상 시공간 AR: 이웃은 '평균'(_neighbor_mean)이라 ALPHA+KAPPA<1이면 안정
        lam = (BASE + ROAD_COEF * road
               + ALPHA * prev + KAPPA * _neighbor_mean(prev)) * season
        lam = np.clip(lam * report_bias, 0.05, None)
        obs[t] = rng.poisson(lam)
        prev = obs[t]

    # 도로 공변량(격자 평탄화) → 셀별 1행
    road_df = pd.DataFrame({"cell": np.arange(CMP_N), "road": road.reshape(-1)})
    road_df.to_parquet(DATA_DIR / "complaint_road.parquet", index=False)

    # 민원 관측(주 × 격자) → 긴 형식. poisson은 정수라 count는 int로 저장(무손실).
    ts = np.repeat(np.arange(CMP_T), CMP_N)
    cells = np.tile(np.arange(CMP_N), CMP_T)
    counts = obs.reshape(CMP_T, CMP_N).reshape(-1).astype(np.int64)
    obs_df = pd.DataFrame({"t": ts, "cell": cells, "count": counts})
    obs_df.to_parquet(DATA_DIR / "complaint_obs.parquet", index=False)
    print(f"  도로 공변량 {CMP_N}개 격자 → complaint_road.parquet")
    print(f"  민원 관측 {CMP_T}주 × {CMP_N}격자 = {len(obs_df):,}행 → complaint_obs.parquet")


# ===========================================================
# 10-3. 부동산 규제경계 RDD: 참 τ를 심은 합성 실거래 표본
# ===========================================================
# 배정변수 x = 경계까지 부호거리(km). x<0=규제지역, x>0=비규제. 경계=0.
# 결과 Y = 매끄러운 공간 가격경사 f(x) + 규제효과 τ·1[x<0] + 잡음 (만원/㎡)
RDD_N = 2000
RDD_H_DATA = 3.0            # 데이터 생성 반경(경계 ±3km)
RDD_TRUE_TAU = -60.0        # 참 규제효과(규제지역 60만원/㎡ 하락)
RDD_BASE_PRICE = 1200.0     # 경계 기준가
RDD_NOISE_SD = 55.0         # 거래가 잡음


def _rdd_smooth_gradient(x):
    """경계에서 연속인 공간 가격경사(입지·학군·교통 등 교란의 총합)."""
    return 30.0 * x + 25.0 * np.sin(1.5 * x)


def _make_rdd(rng, manipulation=False):
    """합성 실거래 표본. manipulation=True면 경계 바깥(x>0) 밀도를 부풀린다(정렬)."""
    if not manipulation:
        x = rng.uniform(-RDD_H_DATA, RDD_H_DATA, RDD_N)
    else:
        base = rng.uniform(-RDD_H_DATA, RDD_H_DATA, RDD_N)
        extra = rng.uniform(0.0, 1.0, RDD_N // 3)  # 경계 바로 바깥에 정렬
        x = np.concatenate([base, extra])
    treated = (x < 0).astype(float)
    y = (RDD_BASE_PRICE + _rdd_smooth_gradient(x)
         + RDD_TRUE_TAU * treated + rng.normal(0, RDD_NOISE_SD, len(x)))
    return pd.DataFrame({"x": x, "y": y, "treated": treated})


def prepare_rdd():
    """RDD 정상 표본과 조작(정렬) 표본을 각각 독립 시드로 만들어 저장한다(10-3)."""
    # 정상 표본은 seed 42로 생성(원래 실습의 주 표본과 동일한 난수열)
    normal = _make_rdd(np.random.default_rng(42), manipulation=False)
    normal.to_parquet(DATA_DIR / "rdd_normal.parquet", index=False)
    # 조작 표본은 별도 시드 43으로 재현 가능하게 생성
    manip = _make_rdd(np.random.default_rng(43), manipulation=True)
    manip.to_parquet(DATA_DIR / "rdd_manip.parquet", index=False)
    print(f"  정상 표본 {len(normal)}행 → rdd_normal.parquet, "
          f"조작 표본 {len(manip)}행 → rdd_manip.parquet")


def main():
    print("=" * 60)
    print("실습 데이터 준비 (10-1 · 10-2 · 10-3)")
    print("=" * 60)
    print("[10-1] 생활SOC 접근성")
    prepare_living_soc()
    print("[10-2] 민원 시공간 패널")
    prepare_complaints()
    print("[10-3] 규제경계 RDD 표본")
    prepare_rdd()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
