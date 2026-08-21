"""
14장 분석: 입지 점수와 자기잠식 제약 (Huff 흡인 모형)
======================================================
비즈니스 질문: "강남구에 매장을 하나 더 낸다면 어디인가, 그리고 그 선택이 기존
자사 매장을 얼마나 잡아먹는가?"

산출물은 순위표가 아니라 결정이다. 후보지마다 신규 배분 수요와 기존점 감소분을
계산해 순증(net gain)을 구하고, 자기잠식이 임계를 넘는 후보를 걸러 낸다.

데이터: 14-0이 만든 cafes.parquet + demand_dong.parquet (국내 공개 데이터)
실행:
    python 14-0-data-prep.py                     # 최초 1회
    python 14-1-huff-location-cannibalization.py

설계에서 미리 밝힐 세 가지.
  (1) 매력도 A는 상수에서 출발한다. 상가정보에는 매장 규모·매출이 없어 A를 직접
      잴 수 없다. 상수로 두고 시작한 뒤 다른 대리변수로 바꿔 결론이 흔들리는지
      본다. 대리변수 선택이 결과를 좌우한다는 사실을 감추지 않는다.
  (2) 거리 저항 β도 관측 방문 자료가 없어 추정할 수 없다. 여러 값을 비교한다.
  (3) 수요 해상도는 행정동이다. 경계 파일 없이 격자로 인구를 쪼개면 점포가 많은
      동일수록 격자가 늘어 격자당 인구가 도리어 낮아지는 역전이 생긴다.
      이 분석의 가장 큰 한계이며 본문에서 그대로 논의한다.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BRAND = "스타벅스"        # 자사로 볼 브랜드(강남구 매장 수 1위). 방법 시연이며 해당 기업의 전략 평가가 아니다
GRID_M = 250              # 후보지 격자 크기(m)
BETA_BASE = 2.0           # 거리 저항 기준값
BETAS = (1.5, 2.0, 2.5)   # 민감도 검토용
MIN_DIST_M = 400.0        # 거리 0 방지를 위한 하한(도보 최소 접근 거리 가정)
LAT0 = 37.5               # 강남구 위도대. 경위도 → 미터 근사에 쓴다


def to_meters(lon, lat):
    """위경도를 국지 평면 좌표(m)로 근사한다. 자치구 규모에서는 오차가 작다."""
    k = np.cos(np.radians(LAT0))
    return np.asarray(lon) * 111_320.0 * k, np.asarray(lat) * 110_540.0


def load():
    f1, f2 = DATA_DIR / "cafes.parquet", DATA_DIR / "demand_dong.parquet"
    if not (f1.exists() and f2.exists()):
        sys.exit("[중단] 분석용 파일이 없다. 먼저 python 14-0-data-prep.py 를 실행한다.")
    return pd.read_parquet(f1), pd.read_parquet(f2)


def build_candidates(sx, sy):
    """점포 분포 범위에 후보지 격자를 깐다."""
    xs = np.arange(sx.min(), sx.max() + GRID_M, GRID_M)
    ys = np.arange(sy.min(), sy.max() + GRID_M, GRID_M)
    cx, cy = np.meshgrid(xs, ys)
    cx, cy = cx.ravel(), cy.ravel()
    d = np.hypot(cx[:, None] - sx[None, :], cy[:, None] - sy[None, :])
    keep = d.min(axis=1) <= GRID_M * 1.5      # 시가지 근사
    return cx[keep], cy[keep]


def huff_weights(dx, dy, sx, sy, attract, beta):
    """수요 지점 x 매장의 '끌어당기는 점수' A/d^beta."""
    d = np.hypot(dx[:, None] - sx[None, :], dy[:, None] - sy[None, :])
    np.maximum(d, MIN_DIST_M, out=d)
    return attract[None, :] / d ** beta


def evaluate(pop, w, own_mask, dx, dy, cx, cy, a_new, beta):
    """후보지마다 신규 배분·자기잠식·순증을 계산한다.

    Huff에서 수요 지점 i가 매장 j를 고를 확률은 (A_j/d_ij^b) / sum_k(A_k/d_ik^b)다.
    신규점을 놓으면 분모에 항이 하나 늘 뿐이므로, 분모 D_i와 자사 몫 S_i만 알면
    후보지별 결과가 닫힌 식으로 나온다. 전체를 다시 계산할 필요가 없다.
    """
    D = w.sum(axis=1)                 # 수요 지점별 분모
    S = w[:, own_mask].sum(axis=1)    # 수요 지점별 자사 몫

    dc = np.hypot(dx[:, None] - cx[None, :], dy[:, None] - cy[None, :])
    np.maximum(dc, MIN_DIST_M, out=dc)
    a = a_new / dc ** beta            # 수요 지점 x 후보지

    Dn = D[:, None] + a
    gross = (pop[:, None] * a / Dn).sum(axis=0)
    loss = (pop[:, None] * S[:, None] * (1.0 / D[:, None] - 1.0 / Dn)).sum(axis=0)
    return gross, loss, gross - loss


def main():
    print("=" * 70)
    print("비즈니스 분석: 입지 점수와 자기잠식 제약 (Huff 흡인 모형)")
    print("=" * 70)

    cafes, demand = load()
    cafes = cafes.merge(demand, on="dong_code", how="inner")
    is_own = cafes["name"].str.contains(BRAND, regex=False).values
    print(f"\n대상: 강남구 카페 {len(cafes)}개 | 자사 {is_own.sum()}개 / "
          f"경쟁 {(~is_own).sum()}개 (자사 점포 점유율 {is_own.mean():.1%})")

    sx, sy = to_meters(cafes["lon"].values, cafes["lat"].values)

    # 수요 단위는 행정동. 대표점은 그 동 카페들의 중심(상업 중심의 대리)
    dong = (pd.DataFrame({"dong_code": cafes["dong_code"].values, "x": sx, "y": sy})
            .groupby("dong_code").mean().reset_index()
            .merge(demand, on="dong_code", how="left"))
    dx, dy, pop = dong["x"].values, dong["y"].values, dong["day_pop"].values
    print(f"  수요 단위: 행정동 {len(dong)}개 | 낮 활동 인구 합계 {pop.sum():,.0f}명")

    cx, cy = build_candidates(sx, sy)
    d_own = np.hypot(cx[:, None] - sx[None, is_own],
                     cy[:, None] - sy[None, is_own]).min(axis=1)
    print(f"  후보지: {GRID_M}m 격자 {len(cx)}개 | 자사 최근접 거리 "
          f"중앙값 {np.median(d_own):,.0f}m")

    def run(attract, beta, a_new=1.0):
        w = huff_weights(dx, dy, sx, sy, attract, beta)
        g, l, n = evaluate(pop, w, is_own, dx, dy, cx, cy, a_new, beta)
        tr = np.divide(l, g, out=np.zeros_like(l), where=g > 0)
        return g, l, n, tr

    attract = np.ones(len(cafes))
    gross, loss, net, transfer = run(attract, BETA_BASE)
    res = pd.DataFrame({"x": cx, "y": cy, "dist_own_m": d_own, "gross": gross,
                        "loss": loss, "net": net, "transfer_rate": transfer})
    res["net_rank"] = res["net"].rank(ascending=False).astype(int)

    print("\n" + "-" * 70)
    print(f"표 14.4  후보지 상위 10곳 — 총배분 순위와 순증 순위 (beta={BETA_BASE}, A=상수)")
    print("-" * 70)
    print(f"{'총배분순위':>8} {'자사최근접(m)':>13} {'예상배분(명)':>12} "
          f"{'자기잠식(명)':>12} {'전이율':>8} {'순증(명)':>10} {'순증순위':>8}")
    for i, (_, r) in enumerate(res.sort_values("gross", ascending=False).head(10).iterrows(), 1):
        print(f"{i:>8} {r['dist_own_m']:>13,.0f} {r['gross']:>12,.0f} "
              f"{r['loss']:>12,.0f} {r['transfer_rate']:>7.1%} "
              f"{r['net']:>10,.0f} {r['net_rank']:>8}")

    print("\n" + "-" * 70)
    print("표 14.5  자사 최근접 매장까지의 거리와 전이율")
    print("-" * 70)
    bins = [0, 300, 500, 800, 1200, 2000, np.inf]
    labels = ["~300", "300~500", "500~800", "800~1200", "1200~2000", "2000~"]
    res["band"] = pd.cut(res["dist_own_m"], bins=bins, labels=labels, right=False)
    print(f"{'거리대(m)':>12} {'후보수':>7} {'평균 전이율':>12} {'평균 순증(명)':>14}")
    for lab, grp in res.groupby("band", observed=True):
        print(f"{str(lab):>12} {len(grp):>7} {grp['transfer_rate'].mean():>11.1%} "
              f"{grp['net'].mean():>14,.0f}")
    print(f"  거리~전이율 상관: {np.corrcoef(res['dist_own_m'], res['transfer_rate'])[0, 1]:.3f}")

    print("\n" + "-" * 70)
    print("결정 — 자기잠식 상한은 언제 구속력을 갖는가")
    print("-" * 70)
    bg = res.loc[res["gross"].idxmax()]
    print(f"  총배분 1위: 배분 {bg['gross']:,.0f}명, 전이율 {bg['transfer_rate']:.1%}, "
          f"순증 {bg['net']:,.0f}명")
    print(f"  전이율 분포: 중앙값 {res['transfer_rate'].median():.1%}, "
          f"최대 {res['transfer_rate'].max():.1%}")
    best_net = res["net"].max()
    for cap in (0.30, 0.10, 0.06, 0.05, 0.04):
        ok = res[res["transfer_rate"] <= cap]
        if ok.empty:
            print(f"  상한 {cap:>4.0%}: 통과 후보 없음")
            continue
        b = ok.loc[ok["net"].idxmax()]
        if b["net"] < best_net - 1e-9:
            note = (f"구속함 — 순증 1위 탈락, 대안 순증 {b['net']:,.0f}명 "
                    f"({(b['net'] - best_net) / best_net:+.1%})")
        else:
            note = "구속하지 않음"
        print(f"  상한 {cap:>4.0%}: 통과 {len(ok):>3}/{len(res)} "
              f"({len(ok) / len(res):>5.1%}) | {note}")

    print("\n" + "-" * 70)
    print("표 14.6  민감도 — 거리 저항 beta")
    print("-" * 70)
    runs = {b: run(attract, b) for b in BETAS}
    ref_idx = int(runs[BETA_BASE][2].argmax())
    ref = (cx[ref_idx], cy[ref_idx])
    print(f"{'beta':>6} {'평균 전이율':>12} {'최선 후보 순증(명)':>18} {'기준 대비 이동(m)':>18}")
    for b in BETAS:
        g2, l2, n2, t2 = runs[b]
        idx = int(n2.argmax())
        shift = float(np.hypot(cx[idx] - ref[0], cy[idx] - ref[1]))
        print(f"{b:>6.1f} {t2.mean():>11.1%} {n2[idx]:>18,.0f} {shift:>18,.0f}")

    print("\n" + "-" * 70)
    print("표 14.7  민감도 — 매력도 A의 대리변수")
    print("-" * 70)
    dss = np.hypot(sx[:, None] - sx[None, :], sy[:, None] - sy[None, :])
    cluster = (dss <= 50).sum(axis=1).astype(float)   # 같은 건물·블록의 집적도
    print(f"{'매력도 정의':>24} {'최선 후보 순증(명)':>18} {'기준 대비 이동(m)':>18}")
    base_xy = None
    for label, av in {"상수(기준)": attract, "집적도(50m 내 점포수)": cluster}.items():
        g3, l3, n3, t3 = run(av, BETA_BASE, a_new=float(np.median(av)))
        idx = int(n3.argmax())
        if base_xy is None:
            base_xy, shift = (cx[idx], cy[idx]), 0.0
        else:
            shift = float(np.hypot(cx[idx] - base_xy[0], cy[idx] - base_xy[1]))
        print(f"{label:>24} {n3[idx]:>18,.0f} {shift:>18,.0f}")

    print("\n" + "-" * 70)
    print("퇴화 검사 — 결과가 '사람 많은 곳'을 되풀이하는 데 그치는가")
    print("-" * 70)
    near = np.hypot(cx[:, None] - dx[None, :], cy[:, None] - dy[None, :]).argmin(axis=1)
    near_pop = pop[near]
    print(f"  총배분 ~ 최근접 행정동 낮인구 상관: {np.corrcoef(res['gross'], near_pop)[0, 1]:.3f}")
    print(f"  순증  ~ 최근접 행정동 낮인구 상관: {np.corrcoef(res['net'], near_pop)[0, 1]:.3f}")
    print(f"  순증  ~ 자사 최근접 거리 상관    : {np.corrcoef(res['net'], res['dist_own_m'])[0, 1]:.3f}")

    res.drop(columns=["band"]).to_csv(RESULTS_DIR / "location_candidates.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"\n저장: results/location_candidates.csv ({len(res)}행)")
    print("=" * 70)
    print("[완료] 입지 점수와 자기잠식 제약")


if __name__ == "__main__":
    main()
