"""
7-0. 데이터 재생성 스크립트 (교수자용 — 학생은 실행할 필요 없음)
=============================================================
이 장의 7-1·7-2 실습이 쓰는 `data/` 폴더의 파일 네 개(pop_grid.parquet,
libraries.parquet, schools.parquet, traffic_volume.npy)를 만든다.
이 파일들은 저장소에 포함되어 배포하므로, 학생은 이 스크립트를 실행하지
않아도 된다. 데이터 파일이 손상되었거나 다시 만들어야 할 때만 실행한다.
시드가 고정되어 있어 다시 실행해도 같은 파일이 나온다.

이 데이터는 실제 공개 데이터를 그대로 구하기 어려운 대상이라 현실적인
값과 구조를 갖도록 만든 교육용 데이터다(개인 식별 불가). 계산은
실제값으로 이뤄진다.
- 자율 GIS 레이어 카탈로그: 인구격자·도서관·학교 (미터좌표 EPSG:5179, 7-1)
- 합성 교통량 시계열: 일·주 주기 + 이분산 잡음 (7-2)

부동소수점을 정확히 보존하도록 표 데이터는 Parquet, 시계열 배열은 npy로
저장한다(모델 재현성 확보). CSV로 저장하면 절단이 생겨 결과가 어긋난다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python lecture_practice/chapter7/code/7-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_CRS = "EPSG:5179"          # 데이터 좌표계(한국 미터좌표)
HOURS = 1200                    # 총 1200시간(약 50일)


def prepare_gis_catalog():
    """자율 GIS 레이어 카탈로그를 만들어 저장한다(7-1).

    미터좌표(EPSG:5179) 가상의 도시(0~10000m 범위). 세 레이어를 하나의
    난수열(seed 42)에서 순서대로 생성한다 — 인구격자 → 도서관 → 학교.
    """
    rng = np.random.default_rng(42)
    # 인구격자: 20×20, 셀 500m
    gx, gy = np.meshgrid(np.arange(20) * 500 + 250, np.arange(20) * 500 + 250)
    pop = np.clip(rng.normal(1200, 400, 400), 100, None).round()
    pop_grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel(), "pop": pop})
    # 도서관 6개소, 학교 12개소(미터좌표)
    libraries = pd.DataFrame({"x": rng.uniform(0, 10000, 6), "y": rng.uniform(0, 10000, 6)})
    schools = pd.DataFrame({"x": rng.uniform(0, 10000, 12), "y": rng.uniform(0, 10000, 12)})

    pop_grid.to_parquet(DATA_DIR / "pop_grid.parquet", index=False)
    libraries.to_parquet(DATA_DIR / "libraries.parquet", index=False)
    schools.to_parquet(DATA_DIR / "schools.parquet", index=False)
    print(f"  인구격자 {len(pop_grid)}셀 → pop_grid.parquet")
    print(f"  도서관 {len(libraries)}개소 → libraries.parquet, "
          f"학교 {len(schools)}개소 → schools.parquet (좌표계 {DATA_CRS})")


def prepare_traffic():
    """합성 교통량 시계열을 만들어 저장한다(7-2).

    일 주기(24h) + 주 주기(168h) + 이분산 잡음. 낮 피크 시간대일수록 잡음의
    폭이 커서 불확실성 구조가 시간대에 따라 달라진다(이분산). 전역 np.random에
    의존하므로 생성 직전에 seed(42)를 고정해 재현성을 확보한다.
    """
    np.random.seed(42)
    t = np.arange(HOURS)
    daily = 60 + 40 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 2)   # 새벽 저점·낮 피크
    weekly = 15 * np.sin(2 * np.pi * (t % 168) / 168)                 # 주중·주말
    # 이분산 잡음: 낮(피크) 시간대 변동이 더 큼 → 불확실성 구조 주입
    noise_sd = 4 + 6 * (np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 2) > 0.3)
    volume = np.clip(daily + weekly + np.random.normal(0, noise_sd), 0, None)
    volume = volume.astype(np.float32)
    np.save(DATA_DIR / "traffic_volume.npy", volume)
    print(f"  교통량 시계열 {len(volume)}시간 → traffic_volume.npy")


def main():
    print("=" * 60)
    print("실습 데이터 준비 (7-1 · 7-2)")
    print("=" * 60)
    prepare_gis_catalog()
    prepare_traffic()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
