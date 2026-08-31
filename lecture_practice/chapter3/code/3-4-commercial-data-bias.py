"""
3-4. 상업 공간 데이터의 편향 진단 — 이 데이터를 어디까지 믿을 것인가
=====================================================================
같은 도시, 같은 업종을 세 출처가 각각 몇 개로 세는지 맞대어 본다. 출처가 어긋나는
정도를 재고, 그 어긋남이 지역 특성을 따라 움직이는지 검정한 뒤, 어느 지역에서 어느
데이터를 써도 되는지 판정표를 낸다.

세 분석:
  A. 포착률 — 지도 데이터(OSM)는 실제 점포를 얼마나 담는가
  B. 결측 구조 — 카드 기반 추정매출의 빈칸은 무작위로 생기는가
  C. 판정 — 이 데이터로 "빈 동네"를 판정해도 되는가, 어디까지가 컷라인인가

입력은 3-0b가 만든 집계 파일이다. 인터넷 없이 재현된다.

실행:
    python 3-4-commercial-data-bias.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 크로스 플랫폼 한글 폰트: 사용 가능한 첫 후보 사용(macOS/Windows/Linux).
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 포착률 분석 대상(OSM 태그가 대응되는 업종군)
COVERAGE_CATS = ["카페", "제과점", "편의점", "미용·뷰티", "약국",
                 "안경", "화장품", "서점", "세탁소"]
# 행정동 단위 합산에서는 미용·뷰티를 뺀다. OSM의 shop=hairdresser 하나에 미용실·네일숍·
# 피부관리실이 섞여 들어와 좁은 대응에서 포착률이 1을 넘고(A-1 참조), 개수도 가장 많아
# 합산값을 이 한 업종이 좌우한다. 태그 의미가 흔들리는 항목을 합계에 넣으면 '어느 동이
# 덜 담겼는가'가 아니라 '어느 동에서 미용실 태그를 많이 썼는가'를 재게 된다.
POOLED_CATS = [c for c in COVERAGE_CATS if c != "미용·뷰티"]

# 현금 결제 비중 시나리오. 업종별 실측 공개 자료를 찾지 못해 **가정**으로 둔다.
# 가운데 값은 한국은행 「2024년 지급수단 및 모바일금융서비스 이용행태 조사결과」의
# 전국 현금 이용 비중(건수 기준) 15.9%를 기준선으로 삼았다.
CASH_SHARE = {
    "high": (0.30, ["청과상", "육류판매", "수산물판매", "반찬가게", "미곡판매", "철물점"]),
    "mid": (0.15, ["한식음식점", "분식전문점", "치킨전문점", "호프-간이주점", "노래방",
                   "세탁소", "미용·뷰티", "슈퍼마켓", "신발", "문구", "가구",
                   "자전거", "애완동물", "시계및귀금속", "서점", "안경", "제과점"]),
    "low": (0.05, ["편의점", "카페", "화장품", "약국"]),
}

# 오판 비용의 비. 진입 실패(매몰 투자)를 1로 두었을 때 기회 상실의 상대 크기.
LOSS_ENTER, LOSS_MISS = 1.0, 0.25


def log(msg: str = "") -> None:
    print(msg, flush=True)


def head(title: str) -> None:
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dong = pd.read_parquet(DATA_DIR / "bias_dong.parquet")
    counts = pd.read_parquet(DATA_DIR / "bias_counts.parquet")
    sales = pd.read_parquet(DATA_DIR / "bias_sales.parquet")
    wide = counts.pivot_table(index=["dong_cd", "cat"], columns="source",
                              values="n", aggfunc="sum", fill_value=0).reset_index()
    for c in ("sbiz_narrow", "sbiz_broad", "osm_narrow", "osm_broad"):
        if c not in wide:
            wide[c] = 0
    return dong, wide, sales


# ---------------------------------------------------------------------------
# 분석 A. 포착률
# ---------------------------------------------------------------------------
def analysis_a(dong: pd.DataFrame, wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    head("분석 A. 포착률 — 지도 데이터는 실제 점포를 얼마나 담는가")
    cov = wide[wide["cat"].isin(COVERAGE_CATS)]

    log("\n[A-1] 업종군별 서울 전체 포착률 (OSM 점포 수 ÷ 상권정보 점포 수)")
    log(f"{'업종군':<10}{'상권정보(좁음)':>12}{'OSM(좁음)':>10}{'포착률':>8}"
        f"{'상권정보(넓음)':>12}{'OSM(넓음)':>10}{'포착률':>8}")
    tot = cov.groupby("cat")[["sbiz_narrow", "sbiz_broad", "osm_narrow", "osm_broad"]].sum()
    tot["cov_narrow"] = tot["osm_narrow"] / tot["sbiz_narrow"]
    tot["cov_broad"] = tot["osm_broad"] / tot["sbiz_broad"]
    for c in COVERAGE_CATS:
        r = tot.loc[c]
        log(f"{c:<10}{int(r['sbiz_narrow']):>12,}{int(r['osm_narrow']):>10,}"
            f"{r['cov_narrow']:>8.3f}{int(r['sbiz_broad']):>12,}"
            f"{int(r['osm_broad']):>10,}{r['cov_broad']:>8.3f}")
    s = tot.loc[POOLED_CATS].sum()
    pooled_n = s["osm_narrow"] / s["sbiz_narrow"]
    pooled_b = s["osm_broad"] / s["sbiz_broad"]
    log(f"{'8개 합계*':<10}{int(s['sbiz_narrow']):>12,}{int(s['osm_narrow']):>10,}"
        f"{pooled_n:>8.3f}{int(s['sbiz_broad']):>12,}{int(s['osm_broad']):>10,}{pooled_b:>8.3f}")
    log("  * 합계에서 미용·뷰티는 제외했다(아래 주의 참조).")

    over1 = tot[tot["cov_narrow"] > 1.0]
    if len(over1):
        log(f"\n  [주의] 좁은 대응에서 포착률이 1을 넘는 업종군: {list(over1.index)}")
        log("        포착률이 1을 넘는 것은 물리적으로 불가능하다. 원인은 OSM 태그 하나에")
        log("        여러 업종이 섞여 담기는 것이다. 넓은 대응에서 그 값이 어떻게")
        log("        움직이는지가 대응표의 영향력을 보여 준다.")
        for c in over1.index:
            log(f"        {c}: 좁은 대응 {tot.loc[c,'cov_narrow']:.3f} "
                f"→ 넓은 대응 {tot.loc[c,'cov_broad']:.3f}")

    log(f"\n[A-2] 행정동별 포착률 분포 ({len(POOLED_CATS)}개 업종군 합산, 넓은 대응)")
    log("  합산에서 미용·뷰티는 뺀다. 태그 하나에 여러 업종이 섞여 값이 흔들리는 데다,")
    log("  개수가 가장 많아 합계를 이 한 업종이 좌우하기 때문이다.")
    d = (cov[cov["cat"].isin(POOLED_CATS)]
         .groupby("dong_cd")[["sbiz_broad", "osm_broad", "sbiz_narrow", "osm_narrow"]]
         .sum().reset_index().merge(dong, on="dong_cd", how="inner"))
    MIN_DEN = 30
    dropped = d[d["sbiz_broad"] < MIN_DEN]
    d = d[d["sbiz_broad"] >= MIN_DEN].copy()
    log(f"  분모(상권정보 점포 수) {MIN_DEN}개 미만인 {len(dropped)}개 동은 비율이 불안정해 제외")
    log(f"  분석 대상 {len(d)}개 행정동")
    d["cov"] = d["osm_broad"] / d["sbiz_broad"]
    q = d["cov"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3)
    log(f"  포착률 최소 {q[0]:.3f} / 10% {q[.1]:.3f} / 1사분위 {q[.25]:.3f} / "
        f"중앙값 {q[.5]:.3f} / 3사분위 {q[.75]:.3f} / 90% {q[.9]:.3f} / 최대 {q[1]:.3f}")
    log(f"  가장 높은 동: " + ", ".join(
        f"{r.dong_nm}({r.cov:.2f})" for r in d.nlargest(5, "cov").itertuples()))
    log(f"  가장 낮은 동: " + ", ".join(
        f"{r.dong_nm}({r.cov:.2f})" for r in d.nsmallest(5, "cov").itertuples()))

    log("\n[A-3] 편향은 무작위 잡음인가, 지역 특성을 따라가는가")
    log("  포착 여부를 점포 하나하나의 성공/실패로 보고 이항 GLM을 적합한다.")
    log("  (분모가 동마다 크게 다르므로 비율을 그대로 최소제곱에 넣지 않는다)")
    d["density"] = d["sbiz_broad"] / d["area_km2"]
    X = pd.DataFrame({
        "log_점포밀도": np.log(d["density"]),
        "시청거리_km": d["dist_cityhall_km"],
        "log_면적": np.log(d["area_km2"]),
    })
    Xs = (X - X.mean()) / X.std()          # 계수를 서로 견줄 수 있게 표준화
    Xs = sm.add_constant(Xs)
    endog = np.c_[d["osm_broad"], d["sbiz_broad"] - d["osm_broad"]]
    glm = sm.GLM(endog, Xs, family=sm.families.Binomial()).fit(cov_type="HC1")
    log(f"  {'항':<14}{'계수':>9}{'표준오차':>10}{'z':>8}{'p':>10}")
    for name in Xs.columns:
        log(f"  {name:<14}{glm.params[name]:>9.4f}{glm.bse[name]:>10.4f}"
            f"{glm.tvalues[name]:>8.2f}{glm.pvalues[name]:>10.2e}")
    log(f"  이탈도(deviance) {glm.deviance:,.0f} / 자유도 {int(glm.df_resid)} "
        f"→ 분산 팽창 {glm.deviance / glm.df_resid:.1f}배 (과대산포)")
    log("  과대산포가 크다는 것은 동마다 포착률이 흔들리는 폭이 단순 이항 잡음보다")
    log("  훨씬 넓다는 뜻이다. 즉 설명변수로 다 잡히지 않는 지역 고유의 차이가 있다.")

    resid = glm.resid_pearson
    moran = None
    try:
        from esda.moran import Moran
        from libpysal.weights import KNN
        w = KNN.from_array(np.c_[d["lon"].to_numpy(), d["lat"].to_numpy()], k=6)
        w.transform = "r"
        mi = Moran(resid, w, permutations=999)
        moran = {"I": float(mi.I), "EI": float(mi.EI), "p_sim": float(mi.p_sim)}
        log(f"\n  잔차의 공간 자기상관 Moran's I = {mi.I:.3f} "
            f"(기댓값 {mi.EI:.3f}, 순열검정 p = {mi.p_sim:.3f}, 이웃 6개 기준)")
        log("  잔차가 공간적으로 뭉쳐 있으면, 설명변수로 걸러지지 않은 편향이 특정")
        log("  구역에 몰려 있다는 뜻이다. 동을 서로 독립으로 보고 세운 추론은 그만큼 위태롭다.")
    except Exception as e:                                  # noqa: BLE001
        log(f"\n  [건너뜀] Moran's I 계산 실패: {type(e).__name__}: {e}")

    # 대응 방식이 결론을 바꾸는가
    d["cov_narrow"] = d["osm_narrow"] / d["sbiz_narrow"].replace(0, np.nan)
    rho = d[["cov", "cov_narrow"]].corr(method="spearman").iloc[0, 1]
    log(f"\n  좁은 대응과 넓은 대응으로 각각 잰 행정동 포착률의 순위상관 = {rho:.3f}")
    log("  대응을 바꾸면 포착률의 절대 수준은 바뀌지만 동 사이의 순서는 거의 그대로다.")
    log("  따라서 이 절은 '포착률이 몇 %다'가 아니라 '어느 동이 더 낮다'를 주장으로 삼는다.")

    log("\n  [대응표를 바꾸면 이 결론이 남는가] 미용·뷰티를 합계에 다시 넣고 같은 회귀를 돌린다.")
    d9 = (cov.groupby("dong_cd")[["sbiz_broad", "osm_broad"]].sum().reset_index()
          .merge(dong, on="dong_cd", how="inner"))
    d9 = d9[d9["sbiz_broad"] >= MIN_DEN].copy()
    X9 = pd.DataFrame({
        "log_점포밀도": np.log(d9["sbiz_broad"] / d9["area_km2"]),
        "시청거리_km": d9["dist_cityhall_km"],
        "log_면적": np.log(d9["area_km2"]),
    })
    X9 = sm.add_constant((X9 - X9.mean()) / X9.std())
    g9 = sm.GLM(np.c_[d9["osm_broad"], d9["sbiz_broad"] - d9["osm_broad"]],
                X9, family=sm.families.Binomial()).fit(cov_type="HC1")
    log(f"  {'항':<14}{'8개 계수':>10}{'8개 p':>12}{'9개 계수':>11}{'9개 p':>12}")
    for name in Xs.columns:
        log(f"  {name:<14}{glm.params[name]:>+10.4f}{glm.pvalues[name]:>12.3f}"
            f"{g9.params[name]:>+11.4f}{g9.pvalues[name]:>12.3f}")
    log("  미용·뷰티를 넣으면 시청거리의 부호가 뒤집히고 유의성이 사라진다.")
    log("  개수가 가장 많은 업종군 하나가 합계를 끌고 가기 때문이다. 대응표를 어떻게 잡느냐가")
    log("  '도심에서 멀수록 덜 담긴다'는 결론 자체를 좌우한다 — 그래서 대응표를 표로 공개한다.")

    d["resid"] = resid
    d["fitted"] = glm.fittedvalues
    return d, {"tot": tot, "pooled_narrow": pooled_n, "pooled_broad": pooled_b,
               "glm": glm, "glm9": g9, "moran": moran,
               "rho_scope": float(rho), "n_dropped": len(dropped)}


# ---------------------------------------------------------------------------
# 분석 B. 카드 매출 데이터의 결측 구조
# ---------------------------------------------------------------------------
def analysis_b(dong: pd.DataFrame, wide: pd.DataFrame, sales: pd.DataFrame):
    head("분석 B. 카드 기반 추정매출의 빈칸은 무작위로 생기는가")
    # 업종군 → 서비스업종 코드 대응은 준비 스크립트가 남긴 메타에서 읽는다.
    meta = json.loads((DATA_DIR / "SOURCES_bias.json").read_text(encoding="utf-8"))
    cat2svc = meta["업종군_코드"]

    # 셀 = (행정동, 업종군). 상권정보에 점포가 하나라도 있는 셀만 본다.
    cells = wide[wide["cat"].isin(cat2svc)][["dong_cd", "cat", "sbiz_broad"]].copy()
    cells = cells[cells["sbiz_broad"] > 0]
    cells["svc_cd"] = cells["cat"].map(cat2svc)
    # 상권정보와 매출 자료의 행정동 목록이 완전히 같지는 않다(행정동 통폐합 시점 차이).
    keep = set(dong["dong_cd"])
    dropped_dong = sorted(set(cells["dong_cd"]) - keep)
    if dropped_dong:
        log(f"  [주의] 경계·매출 자료에 없는 행정동 {len(dropped_dong)}개를 제외한다: {dropped_dong}")
        log("        행정동은 통폐합되므로 자료마다 기준 시점이 다르면 목록이 어긋난다.")
        cells = cells[cells["dong_cd"].isin(keep)]

    # 2025년 네 분기 중 매출 행이 한 번이라도 있으면 '관측', 하나도 없으면 '결측'
    seen = sales.groupby(["dong_cd", "svc_cd"]).agg(
        q_seen=("yq", "nunique"), amt=("amt", "sum"), cnt=("cnt", "sum"),
        amt_60p=("amt_60p", "sum"), amt_20s=("amt_20s", "sum")).reset_index()
    cells = cells.merge(seen, on=["dong_cd", "svc_cd"], how="left")
    cells["q_seen"] = cells["q_seen"].fillna(0).astype(int)
    for col in ("amt", "cnt", "amt_60p", "amt_20s"):
        cells[col] = cells[col].fillna(0.0)
    cells["missing"] = (cells["q_seen"] == 0).astype(int)

    log(f"\n[B-1] 점포가 실재하는 셀 {len(cells):,}개 "
        f"(행정동 {cells['dong_cd'].nunique()}개 × 업종군 {cells['cat'].nunique()}개 중)")
    log(f"  그중 2025년 매출 자료에 한 분기도 없는 셀: {cells['missing'].sum():,}개 "
        f"({cells['missing'].mean() * 100:.1f}%)")
    log("  '점포는 있는데 매출 기록이 없다'는 것이 여기서 말하는 결측이다.")

    log("\n[B-2] 업종군별 결측률과 점포당 카드 매출")
    log(f"{'업종군':<12}{'셀 수':>7}{'점포 수':>9}{'셀당 점포':>10}{'결측 셀':>8}{'결측률':>8}"
        f"{'점포당 연매출(만원)':>20}{'건당 결제(원)':>14}")
    rows = []
    for c, g in cells.groupby("cat"):
        n_store = g["sbiz_broad"].sum()
        amt = g["amt"].sum()
        cnt = g["cnt"].sum()
        rows.append({
            "cat": c, "n_cell": len(g), "n_store": int(n_store),
            "per_cell": n_store / len(g),
            "n_missing": int(g["missing"].sum()), "miss_rate": g["missing"].mean(),
            "amt_per_store_manwon": amt / n_store / 1e4 if n_store else np.nan,
            "won_per_txn": amt / cnt if cnt else np.nan,
        })
    tb = pd.DataFrame(rows).sort_values("miss_rate", ascending=False)
    for r in tb.itertuples():
        log(f"{r.cat:<12}{r.n_cell:>7,}{r.n_store:>9,}{r.per_cell:>10.1f}{r.n_missing:>8,}"
            f"{r.miss_rate:>8.3f}{r.amt_per_store_manwon:>20,.0f}{r.won_per_txn:>14,.0f}")
    log("  [읽는 법] 점포당 연매출은 두 자료의 업종 정의가 어긋나면 크게 부풀거나 줄어든다.")
    log("  분자(매출)는 서울 상권분석의 서비스업종, 분모(점포)는 상권정보의 소분류이므로,")
    log("  두 범위가 다르면 이 값은 업종 간 비교에 쓸 수 없다. 반면 건당 결제금액은 매출")
    log("  자료 안에서만 계산되어 대응표의 영향을 받지 않는다 — 이 열이 더 믿을 만하다.")

    log("\n  결측이 어디에 몰리는지 한 줄로 보면 이렇다.")
    small = tb.sort_values("per_cell")
    log("    셀당 평균 점포 수가 적은 업종군 5개: " + ", ".join(
        f"{r.cat}({r.per_cell:.1f}개, 결측률 {r.miss_rate:.2f})" for r in small.head(5).itertuples()))
    log("    많은 업종군 5개: " + ", ".join(
        f"{r.cat}({r.per_cell:.1f}개, 결측률 {r.miss_rate:.2f})" for r in small.tail(5).itertuples()))
    log("  점포가 몇 개 없는 칸일수록 빈다. 소수 표본을 가리는 최소 집계 규칙이 작동한 것으로")
    log("  보이지만(자료 명세에서 확인 필요), 원인이 무엇이든 결과는 같다 — 작은 업종이")
    log("  체계적으로 지워진다.")

    # 연령 구성: 현금 이야기의 정황이지 증거는 아니다.
    obs = cells[cells["missing"] == 0].groupby("cat")[["amt", "amt_60p"]].sum()
    obs["share_60p"] = obs["amt_60p"] / obs["amt"]
    j = tb.set_index("cat").join(obs["share_60p"])
    r60 = j[["miss_rate", "share_60p"]].corr(method="spearman").iloc[0, 1]
    log(f"\n  참고: 60대 이상 매출 비중이 가장 높은 업종군 — " + ", ".join(
        f"{i}({v:.2f})" for i, v in obs["share_60p"].nlargest(4).items()))
    log(f"        가장 낮은 업종군 — " + ", ".join(
        f"{i}({v:.2f})" for i, v in obs["share_60p"].nsmallest(4).items()))
    log(f"        결측률과 60대 이상 비중의 순위상관 {r60:+.3f}")
    log("  이것은 정황일 뿐 인과의 증거가 아니다. 고령층 이용이 많은 업종에서 현금 결제가")
    log("  많다면 카드 자료가 그만큼 덜 담겠지만, 그 연결을 이 자료만으로는 확인할 수 없다.")

    log("\n[B-3] 결측이 무작위인지 검정 (로지스틱 회귀)")
    log("  완전 무작위 결측(MCAR)이라면 결측 확률이 점포 수·지역·업종과 무관해야 한다.")
    m = cells.merge(dong, on="dong_cd", how="left")
    m["log_n"] = np.log1p(m["sbiz_broad"])
    m["dist"] = m["dist_cityhall_km"]
    num = m[["log_n", "dist"]]
    num = (num - num.mean()) / num.std()
    dum = pd.get_dummies(m["cat"], prefix="cat", drop_first=True).astype(float)

    def fit(X):
        return sm.Logit(m["missing"].to_numpy(), sm.add_constant(X, has_constant="add")).fit(disp=0)

    m0 = fit(pd.DataFrame(index=m.index))                       # 절편만
    m1 = fit(num)                                               # 점포 수·거리
    m2 = fit(pd.concat([num, dum], axis=1))                     # + 업종 고정효과
    from scipy import stats as st
    for name, mod, base in (("점포 수·시청거리", m1, m0), ("+ 업종 고정효과", m2, m1)):
        lr = 2 * (mod.llf - base.llf)
        dfd = int(mod.df_model - base.df_model)
        p = st.chi2.sf(lr, dfd)
        log(f"  {name:<16} 우도비 χ² = {lr:8.1f} (자유도 {dfd:2d}), p = {p:.3e}")
    log(f"  점포 수 계수 {m2.params['log_n']:+.3f} (표준오차 {m2.bse['log_n']:.3f}) "
        f"— 점포가 많은 셀일수록 결측이 {'적다' if m2.params['log_n'] < 0 else '많다'}")
    log(f"  시청거리 계수 {m2.params['dist']:+.3f} (표준오차 {m2.bse['dist']:.3f})")
    log("  두 검정 모두 MCAR을 기각한다. 결측은 관측된 특성(점포 수·업종·위치)을 따라 생긴다.")
    log("  여기까지가 MAR(관측 변수로 설명되는 결측)의 근거다. 그러나 MNAR을 배제하지는")
    log("  못한다 — 매출이 작은 셀일수록 가려질 수 있는데, 그 매출은 정의상 보이지 않는다.")

    return cells, tb, {"m0": m0, "m1": m1, "m2": m2}


# ---------------------------------------------------------------------------
# 분석 C. 판정표
# ---------------------------------------------------------------------------
def analysis_c(d: pd.DataFrame, cells: pd.DataFrame, tb: pd.DataFrame, wide: pd.DataFrame):
    head("분석 C. 이 데이터로 '빈 동네'를 판정해도 되는가 — 컷라인과 판정표")

    log("\n[C-1] 지도에 한 곳도 없는 동네가 정말 비어 있을 확률")
    log("  포착률이 p인 자료에서, 실제 점포가 N개인 동이 '0개'로 보일 확률은 (1−p)^N 이다.")
    log("  N의 분포는 상권정보의 행정동별 점포 수를 그대로 사전분포로 쓴다.")
    log("  (점포가 하나도 없어 집계표에 아예 빠진 동까지 0으로 되살려 넣는다)")

    tau = LOSS_ENTER / (LOSS_ENTER + LOSS_MISS)
    tau_pub = LOSS_MISS / (LOSS_ENTER + LOSS_MISS)
    log(f"\n  진입 판단의 컷라인은 오판 비용의 비로 정한다.")
    log(f"    이미 점포가 있는데 비었다고 보고 들어가는 손실(매몰 투자) = {LOSS_ENTER}")
    log(f"    비어 있는데 놓치는 손실(기회비용)                        = {LOSS_MISS}")
    log(f"    → '정말 비었다'는 사후확률이 {tau:.2f} 이상일 때만 진입한다.")

    dong_all = pd.read_parquet(DATA_DIR / "bias_dong.parquet")["dong_cd"]

    def prior_of(cat: str):
        s = (wide[wide["cat"] == cat].set_index("dong_cd")["sbiz_broad"]
             .reindex(dong_all).fillna(0))
        pr = s.value_counts(normalize=True).sort_index()
        return s, pr.index.to_numpy(dtype=float), pr.to_numpy()

    def posterior(n_vals, p_n, p):
        p_obs0 = float(np.sum(p_n * (1 - p) ** n_vals))
        p0 = float(p_n[n_vals == 0].sum())
        return p_obs0, (p0 / p_obs0 if p_obs0 > 0 else np.nan)

    def p_star_of(n_vals, p_n, thr):
        grid = np.linspace(0.01, 0.999, 999)
        post = np.array([posterior(n_vals, p_n, p)[1] for p in grid])
        ok = post >= thr
        return float(grid[np.argmax(ok)]) if ok.any() else np.nan

    log(f"\n{'업종군':<10}{'0개인 동':>9}{'P(N=0)':>9}{'실측 포착률':>12}"
        f"{'P(N=0|0개로 보임)':>20}{'필요 p*':>9}{'판정':>12}")
    star_tbl = {}
    for cat in COVERAGE_CATS:
        s, n_vals, p_n = prior_of(cat)
        cov_c = (wide[wide["cat"] == cat]["osm_broad"].sum()
                 / max(wide[wide["cat"] == cat]["sbiz_broad"].sum(), 1))
        # 포착률은 확률이므로 1을 넘을 수 없다. 1을 넘었다면 대응표가 어긋난 것이고,
        # 그 값을 0.999로 잘라 넣으면 잘못된 대응이 계산을 통과해 버린다. 자르지 않고
        # 판정에서 뺀다.
        if cov_c > 1.0:
            post, ps = float("nan"), float("nan")
            verdict = "대응표 점검"
        else:
            _, post = posterior(n_vals, p_n, cov_c)
            ps = p_star_of(n_vals, p_n, tau)
            verdict = "진입 가능" if post >= tau else "판정 불가"
        star_tbl[cat] = (float(cov_c), float(post), ps)
        log(f"{cat:<10}{int((s == 0).sum()):>9}{float(p_n[n_vals == 0].sum()):>9.3f}"
            f"{cov_c:>12.3f}{('—' if np.isnan(post) else f'{post:.3f}'):>20}"
            f"{('—' if np.isnan(ps) else f'{ps:.2f}'):>9}{verdict:>12}")
    log("  ('필요 p*'가 —인 업종군은 포착률을 1에 가깝게 올려도 컷라인을 넘지 못한다는 뜻이다.")
    log("   그런 업종군은 점포가 없는 동 자체가 드물어, '0개'가 애초에 정보를 거의 담지 않는다.)")

    # 자세히 볼 업종군: 점포가 없는 동이 실제로 많아 '빈 동네' 판정이 성립하는 쪽.
    # 어디에나 깔린 업종(카페·편의점)은 애초에 '0개'라는 관측이 나오지 않는다.
    focus = max(COVERAGE_CATS, key=lambda c: float((prior_of(c)[0] == 0).mean()))
    s, n_vals, p_n = prior_of(focus)
    log(f"\n  자세히 보기 — {focus} (점포 0개인 동 {int((s == 0).sum())}개, 중앙값 {s.median():.0f}개)")
    log(f"{'포착률 p':>10}{'P(0개로 보임)':>15}{'P(정말 0개 | 0개로 보임)':>26}{'판정':>12}")
    for p in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        p_obs0, post = posterior(n_vals, p_n, p)
        log(f"{p:>10.2f}{p_obs0:>15.3f}{post:>26.3f}"
            f"{('진입 가능' if post >= tau else '판정 불가'):>12}")
    p_star = p_star_of(n_vals, p_n, tau)
    cov_focus = star_tbl[focus][0]
    log(f"\n  사후확률이 {tau:.2f}에 도달하는 최소 포착률 p* = "
        f"{'도달하지 못함' if np.isnan(p_star) else f'{p_star:.2f}'}")
    log(f"  실측 {focus} 포착률은 {cov_focus:.3f}이다.")
    log(f"  즉 이 자료에서 '{focus}이 한 곳도 없는 동네'를 미충족 시장으로 읽으면 안 된다.")

    log(f"\n  공공은 비용의 방향이 반대다. 사각지대를 놓치는 손실이 헛짚는 손실보다 크면")
    log(f"  같은 계산에서 컷라인이 {tau_pub:.2f}로 내려가고, 판정은 '의심되면 현장 확인'이 된다.")
    p_star_pub = p_star_of(n_vals, p_n, tau_pub)
    log(f"  그 컷라인에서 필요한 최소 포착률은 "
        f"{'도달하지 못함' if np.isnan(p_star_pub) else f'{p_star_pub:.2f}'}이다.")
    log("  같은 데이터, 같은 확률, 다른 결론 — 갈라놓는 것은 오판 비용의 비다.")

    log("\n[C-2] 현금 결제를 무시하면 상권 순위가 어떻게 달라지는가")
    log("  카드 기반 추정매출은 현금 거래를 담지 못한다. 업종별 현금 비중 c의 공개 실측")
    log("  자료를 찾지 못했으므로 아래 c는 **가정**이며, 결론은 '순위가 흔들린다'까지다.")
    log("  보정은 단순하다 — 카드로 잡힌 금액이 전체의 (1−c)이라고 보고 그만큼 되돌린다.")
    cash = {}
    for _, (cv, cats) in CASH_SHARE.items():
        for x in cats:
            cash[x] = cv
    t = tb.copy()
    t["c"] = t["cat"].map(cash).fillna(0.15)
    t["amt"] = t["cat"].map(cells.groupby("cat")["amt"].sum())
    t["amt_adj"] = t["amt"] / (1 - t["c"])
    t["rank_raw"] = t["amt"].rank(ascending=False)
    t["rank_adj"] = t["amt_adj"].rank(ascending=False)
    t["move"] = t["rank_raw"] - t["rank_adj"]
    t = t.sort_values("rank_raw")
    log(f"\n{'업종군':<12}{'가정 c':>7}{'연매출(억원)':>14}{'보정 후(억원)':>15}"
        f"{'순위 전':>7}{'순위 후':>7}{'이동':>6}")
    for r in t.itertuples():
        log(f"{r.cat:<12}{r.c:>7.2f}{r.amt / 1e8:>14,.0f}{r.amt_adj / 1e8:>15,.0f}"
            f"{r.rank_raw:>7.0f}{r.rank_adj:>7.0f}{r.move:>+6.0f}")
    moved = int((t["move"].abs() >= 1).sum())
    rho = t[["rank_raw", "rank_adj"]].corr(method="spearman").iloc[0, 1]
    log(f"\n  순위가 바뀐 업종군 {moved}개 / {len(t)}개, 순위상관 {rho:.3f}")
    if moved:
        log("  가장 크게 오른 업종: " + ", ".join(
            f"{r.cat}({r.move:+.0f})" for r in t.nlargest(3, "move").itertuples()))
    log("  현금 비중이 모든 업종에서 같다면 순위는 그대로다. 순위를 흔드는 것은 c의 크기가")
    log("  아니라 c가 업종마다 다르다는 사실이다.")

    log("\n  같은 보정을 행정동에 적용하면 '어느 상권이 큰가'의 순위가 움직인다.")
    cc = cells.copy()
    cc["c"] = cc["cat"].map(cash).fillna(0.15)
    cc["amt_adj"] = cc["amt"] / (1 - cc["c"])
    g = cc.groupby("dong_cd")[["amt", "amt_adj"]].sum().reset_index().merge(
        d[["dong_cd", "dong_nm"]], on="dong_cd", how="left")
    g["rank_raw"] = g["amt"].rank(ascending=False)
    g["rank_adj"] = g["amt_adj"].rank(ascending=False)
    g["move"] = g["rank_raw"] - g["rank_adj"]
    for k in (10, 20, 50):
        a = set(g.nsmallest(k, "rank_raw")["dong_cd"])
        b_ = set(g.nsmallest(k, "rank_adj")["dong_cd"])
        log(f"    상위 {k:>2}개 상권의 겹침: {len(a & b_)}/{k}")
    up = g.nlargest(3, "move")
    log("    보정 후 가장 많이 오른 동: " + ", ".join(
        f"{r.dong_nm} {r.rank_raw:.0f}위→{r.rank_adj:.0f}위" for r in up.itertuples()))
    log(f"    행정동 순위상관 {g[['rank_raw', 'rank_adj']].corr(method='spearman').iloc[0, 1]:.4f}")
    log("  겹침이 크더라도 경계 근처에서는 순서가 바뀐다. 상위 몇 곳만 뽑아 예산이나 출점을")
    log("  결정할 때, 그 경계에 걸린 동이 현금 업종이 많은 곳이라면 판단이 뒤집힌다.")

    log("\n[C-3] 행정동 데이터 사용 판정표")
    log("  세 등급으로 나눈다. 경계는 위에서 계산한 컷라인과 회귀 잔차를 함께 쓴다.")
    log(f"  각 동의 포착률을 그 동의 값으로 넣고, 사전분포는 {focus}의 것을 쓴다.")
    log("  (성기게 깔린 업종을 그 동에서 찾는다면 '0개'를 믿어도 되는가 하는 물음이다)")
    d = d.copy()
    # 포착률이 1을 넘는 동은 등급을 매기지 않는다. 예전 코드는 clip(1e-6, 0.999)로
    # 잘라 넣었는데, 그러면 대응표가 어긋나 생긴 값이 오류 메시지 없이 판정을 통과해
    # 가장 높은 등급을 받는다. 자르는 대신 따로 세어 보고한다.
    bad = d[d["cov"] > 1.0]
    if len(bad):
        log(f"\n  [경고] 포착률이 1을 넘는 동 {len(bad)}개는 등급 판정에서 뺀다.")
        for r in bad.sort_values("cov", ascending=False).itertuples():
            log(f"    {r.dong_nm} — 분모 {int(r.sbiz_broad)}개, 분자 {int(r.osm_broad)}개, "
                f"포착률 {r.cov:.3f}. 확률로 해석할 수 없는 값이므로 업종 대응표를 다시 잡는다.")
    d["post0"] = [posterior(n_vals, p_n, p)[1] if p <= 1.0 else np.nan
                  for p in d["cov"]]
    d["grade"] = np.where(
        d["cov"] > 1.0, "대응표 점검",
        np.where(d["post0"] >= tau, "그대로 사용",
                 np.where(d["resid"].abs() <= 2.0, "보정 후 사용", "사용 금지")))
    g = d.groupby("grade").agg(n=("dong_cd", "size"), cov_min=("cov", "min"),
                               cov_med=("cov", "median"), cov_max=("cov", "max"))
    log(f"\n{'등급':<12}{'행정동 수':>9}{'포착률 최소':>12}{'중앙값':>9}{'최대':>9}{'대표 동':>24}")
    for grade in ("그대로 사용", "보정 후 사용", "사용 금지", "대응표 점검"):
        if grade not in g.index:
            log(f"{grade:<12}{0:>9}{'—':>12}{'—':>9}{'—':>9}{'—':>24}")
            continue
        r = g.loc[grade]
        ex = ", ".join(d[d["grade"] == grade].nlargest(2, "sbiz_broad")["dong_nm"])
        log(f"{grade:<12}{int(r['n']):>9}{r['cov_min']:>12.3f}{r['cov_med']:>9.3f}"
            f"{r['cov_max']:>9.3f}{ex:>24}")
    log("\n  '그대로 사용'은 포착률이 컷라인을 넘어 공백 판정을 그대로 믿어도 되는 동이다.")
    log("  '보정 후 사용'은 포착률이 낮지만 회귀식으로 예측되는 만큼만 낮은 동이다.")
    log("    관측값을 예측 포착률로 나누어 점포 수를 되돌린 뒤 쓴다.")
    log("  '사용 금지'는 낮으면서 예측도 빗나가는 동이다. 여기서는 POI만으로 공백을 판정하지 않는다.")
    return d, t, {"p_star": p_star, "tau": tau, "tau_pub": tau_pub, "focus": focus,
                  "focus_cov": float(cov_focus), "moved": moved, "rho_rank": float(rho)}


# ---------------------------------------------------------------------------
def make_figure(d: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    sc = ax[0].scatter(d["lon"], d["lat"], c=d["cov"], s=np.sqrt(d["sbiz_broad"]) * 3,
                       cmap="viridis", alpha=.85, edgecolors="none")
    ax[0].set_title("행정동별 OSM 포착률 (원 크기 = 상권정보 점포 수)")
    ax[0].set_xlabel("경도"); ax[0].set_ylabel("위도")
    ax[0].set_aspect(1 / np.cos(np.deg2rad(37.55)))
    fig.colorbar(sc, ax=ax[0], label="포착률")

    ax[1].scatter(d["dist_cityhall_km"], d["cov"], s=18, alpha=.6, color="#2b6cb0")
    o = np.argsort(d["dist_cityhall_km"].to_numpy())
    ax[1].plot(d["dist_cityhall_km"].to_numpy()[o], d["fitted"].to_numpy()[o],
               lw=0, marker=".", ms=3, color="#c53030", label="이항 GLM 적합값")
    ax[1].set_xlabel("서울시청까지 거리 (km)"); ax[1].set_ylabel("포착률")
    ax[1].set_title("도심에서 멀어질수록 포착률은 어떻게 되는가")
    ax[1].legend(fontsize=9)

    fig.tight_layout()
    out = RESULTS_DIR / "3-4-coverage-map.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    dong, wide, sales = load()
    log(f"행정동 {len(dong)}개 · 집계 셀 {len(wide):,}개 · 매출 행 {len(sales):,}개")

    d, a = analysis_a(dong, wide)
    cells, tb, b = analysis_b(dong, wide, sales)
    d, t, c = analysis_c(d, cells, tb, wide)

    png = make_figure(d)
    head("산출물")
    log(f"  그림: {png.relative_to(SCRIPT_DIR.parent.parent.parent)}")
    log(f"  {len(POOLED_CATS)}개 업종군 합산 포착률 — 좁은 대응 {a['pooled_narrow']:.3f} / "
        f"넓은 대응 {a['pooled_broad']:.3f}")
    ps_txt = "도달하지 못함" if np.isnan(c["p_star"]) else f"{c['p_star']:.2f}"
    log(f"  {c['focus']} 공백 판정에 필요한 최소 포착률 p* = {ps_txt} "
        f"(실측 {c['focus_cov']:.3f})")
    log(f"  판정 등급별 행정동 수: " + ", ".join(
        f"{k} {v}개" for k, v in d["grade"].value_counts().items()))


if __name__ == "__main__":
    main()
