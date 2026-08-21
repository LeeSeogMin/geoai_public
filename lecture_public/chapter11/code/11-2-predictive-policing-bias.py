"""
11장 실습 2: 예측적 치안의 자기실현적 편향 (반면교사)
=================================================================
질문: "범죄위험을 예측해 순찰을 배분하면 안전해질까?" — 이 분석의 답은 '주의하라'다.

이 분석은 GeoAI '성공 사례'가 아니라 **반면교사**다(정책 컴포넌트 Ch11).
예측적 치안(국내 경찰청 Pre-CAS, 미국 PredPol)이 편향된 자료로 학습될 때
어떻게 차별을 재생산하는지를 시뮬레이션으로 입증한다.

핵심 메커니즘(Lum & Isaac 2016, Significance):
  관측되지 않는 '진짜 범죄율'이 있고, 과거 단속은 특정 지역에 편향되어 있다.
  예측은 '기록된 범죄'로 학습되므로 단속 편향을 반영한다.
  순찰을 예측 위험에 비례해 배분 → 과잉순찰지에서 더 많이 적발(discovery) →
  기록↑ → 다음 예측↑ → 같은 곳으로 반복 회귀하는 '되먹임 루프(runaway loop)'.

입증 지표(라운드별):
  - corr(순찰 배분, 진짜 범죄율): 떨어진다(진짜 수요와 괴리).
  - 순찰 집중도 Gini: 오른다(소수 지역에 집중).
  - 집단 간 disparate impact: 오른다(편향 집단을 과잉 표적).
  - 모델 R²(기록 범죄 기준): 높게 유지된다 → '정확해 보이는' 함정.

결론(규칙3, 가장 중요): 정확도(특히 기록 범죄에 대한 정확도)는 형평을 보장하지 않는다.
  공정성·설명가능성이 정확도에 항상 우선한다. Pre-CAS류 시스템은 신고편향 보정·
  공정성 감사·인간 검토·설명가능성이 전제될 때만 제한적으로 허용된다.

데이터: 미리 준비한 '진짜 범죄율' 필드를 불러와 사용(개인 식별 불가, 격자).
  분석·추정은 실제 계산값. 되먹임 시뮬레이션(단속→기록→예측→재배치)의 포아송
  적발 과정은 이 분석 코드가 수행한다.
실행:
    python 11-0-simdata-prep.py                 # 최초 1회: 데이터 준비
    python 11-2-predictive-policing-bias.py
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

GRID = 16
N = GRID * GRID
ROUNDS = 8
BOOST = 1.6        # 과잉순찰지의 적발 증폭(순찰 집중 → 더 많이 기록)


def load_field():
    """미리 준비한 '진짜 범죄율' 필드와 집단·역사적 편향 구조를 불러온다."""
    path = DATA_DIR / "policing_field.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 11-0-simdata-prep.py")
    return pd.read_parquet(path)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main():
    print("=" * 64)
    print("예측적 치안의 자기실현적 편향 (11-2, 반면교사)")
    print("=" * 64)

    # 미리 준비한 필드 로드: 진짜 범죄율(관측 불가), 집단, 역사적 단속 편향
    field = load_field()
    true_rate = field["true_rate"].values
    group_A = field["group_A"].values.astype(bool)  # 과잉단속 집단(좌측 절반)
    hist_bias = field["hist_bias"].values           # 진짜율은 같아도 기록은 편향
    rr = field["row"].values
    cc = field["col"].values

    # 되먹임 시뮬레이션(단속→기록→예측→재배치)의 포아송 적발 과정은 이 분석의 몫이다.
    # 데이터 생성이 11-0으로 분리되어 전역 RNG 상태가 소멸하므로, 시뮬레이션이
    # 재현 가능하도록 여기서 재시드한다.
    RNG = np.random.default_rng(42)

    # 라운드 0: 기록 범죄 = 진짜율 × 역사적 단속 편향
    recorded = RNG.poisson(true_rate * hist_bias).astype(float)

    print(f"\n분석 단위: {N}개 격자(16×16), 라운드 {ROUNDS}회, 개인 식별 불가")
    print(f"진짜 범죄율 평균 {true_rate.mean():.1f}건 (관측 불가) | "
          f"group A(과잉단속) 평균 진짜율 {true_rate[group_A].mean():.1f} vs "
          f"group B {true_rate[~group_A].mean():.1f} → 진짜 위험은 거의 동일")

    feats = np.column_stack([rr, cc])               # 공간 좌표(+이웃) 피처
    history = []
    for t in range(ROUNDS):
        # 공간 ML로 위험 예측(기록 범죄 학습) — '공간 명시적 모델'도 편향을 못 막는다
        model = GradientBoostingRegressor(random_state=42, n_estimators=150,
                                          max_depth=3, learning_rate=0.05)
        # 이웃 기록(공간 시차) 추가
        grid_rec = recorded.reshape(GRID, GRID)
        neigh = np.zeros_like(grid_rec)
        neigh[1:, :] += grid_rec[:-1, :]; neigh[:-1, :] += grid_rec[1:, :]
        neigh[:, 1:] += grid_rec[:, :-1]; neigh[:, :-1] += grid_rec[:, 1:]
        X = np.column_stack([feats, neigh.reshape(-1)])
        model.fit(X, recorded)
        pred = np.clip(model.predict(X), 0, None)
        r2_recorded = r2_score(recorded, pred)      # 기록 기준 정확도(겉보기)

        # 순찰 배분 ∝ 예측 위험(상대 강도, 평균 1)
        patrol = pred / pred.mean()

        # 입증 지표
        corr_true = np.corrcoef(patrol, true_rate)[0, 1]
        g = gini(patrol)
        # disparate impact: group A 순찰비중 / group A 진짜범죄비중
        di = (patrol[group_A].sum() / patrol.sum()) / (true_rate[group_A].sum() / true_rate.sum())
        history.append((t, r2_recorded, corr_true, g, di))

        # 되먹임: 과잉순찰지에서 더 많이 적발 → 기록 갱신
        discovery = 1.0 + BOOST * (patrol - 1.0)
        recorded = RNG.poisson(np.clip(true_rate * discovery, 0.1, None)).astype(float)

    hist = pd.DataFrame(history, columns=["round", "r2_recorded",
                                          "corr_patrol_true", "patrol_gini", "disparate_impact"])
    print("\n[입증] 라운드별 지표 — '정확해 보이는' 모델이 진짜 수요와 괴리된다")
    print(hist.round(3).to_string(index=False))

    first, last = hist.iloc[0], hist.iloc[-1]
    print(f"\n  기록 기준 모델 R²: {first.r2_recorded:.3f} → {last.r2_recorded:.3f} (계속 높음 = 겉보기 정확)")
    print(f"  순찰-진짜위험 상관: {first.corr_patrol_true:.3f} → {last.corr_patrol_true:.3f} "
          f"({last.corr_patrol_true-first.corr_patrol_true:+.3f}, 진짜 수요와 괴리)")
    print(f"  순찰 집중도 Gini : {first.patrol_gini:.3f} → {last.patrol_gini:.3f} "
          f"({last.patrol_gini-first.patrol_gini:+.3f}, 소수 지역 집중)")
    print(f"  disparate impact: {first.disparate_impact:.3f} → {last.disparate_impact:.3f} "
          f"(group A 과잉표적; 1.0=공정, >1=차별)")

    csv_path = RESULTS_DIR / "predictive_policing_bias.csv"
    hist.round(4).to_csv(csv_path, index=False)
    print(f"\n  편향 추이 저장 → {csv_path.name}")

    print("\n[결론 — 반면교사, 규칙3]")
    print("  진짜 범죄율은 두 집단이 거의 같았지만, 편향된 기록으로 학습한 예측은")
    print("  group A를 계속 과잉 표적하고(되먹임), 모델 R²는 높게 유지되어 '정확해' 보인다.")
    print("  → 정확도 ≠ 형평. 공정성·설명가능성이 정확도에 우선한다.")
    print("  → 예측적 치안은 정당화 사례가 아니라, 신고편향 보정·공정성 감사·인간 검토가")
    print("    전제될 때만 제한적으로 허용되는 '주의' 영역이다(Lum & Isaac 2016).")

    print("\n[완료] 예측적 치안 편향 시뮬레이션을 마쳤다.")


if __name__ == "__main__":
    main()
