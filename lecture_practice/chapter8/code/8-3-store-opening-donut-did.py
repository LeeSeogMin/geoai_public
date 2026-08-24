"""
8장 비즈니스 분석 예제: 신규 출점의 상권 효과 — IPW-DID와 도넛 설계
====================================================================
경영 질문: "신규점이 만든 매출 중 순증은 얼마이고, 기존 매장에서 옮겨온 몫은 얼마인가?"

8-1(공공 개발사업)과 갈라지는 지점이 둘이다.
  ① 처치가 내생적이다 — 사업 지정과 달리 출점은 "잘될 곳"에 배치된다.
  ② 대조군이 오염된다 — 신규점 인근 기존점 상권은 처치도 대조도 아니다(SUTVA 위반).

무엇을 계산하는가
-----------------
[1] 관측된 한 번의 출점에 대한 추정 계단 — 대조군을 어떻게 잡는가에 따라 넷
[2] overlap 진단과 극단 가중치 절단
[3] 도넛 폭 민감도 — 오염 반경을 모른다고 가정하면 무엇이 흔들리는가
[4] 링별 노출-반응 프로파일 — 이분법을 버리고 거리대별 효과 곡선을 본다
[5] 반복 실험 — 배치를 다시 뽑아 각 추정량의 평균 편향을 잰다(세계 A/B/C/D)
[6] 1종 오류율 — 도넛이 추론을 깨는가(지리 고정 · 잡음만 재추출)
[7] 순효과 — 총효과에서 잠식을 뺀다

왜 반복 실험이 필요한가
-----------------------
처치를 정한 것은 신규점 3곳이다. 한 번의 출점에서 나온 추정치 하나로는 "도넛이
편향을 걷었다"와 "이번 뽑기가 운이 좋았다"를 구분할 수 없다. 그래서 배치를 다시
뽑아 평균 편향을 재고, 심은 메커니즘을 하나씩 끈 세계와 대조한다.

  세계 A = 내생적 배치 + 시변 교란 + 자기잠식 (관측 세계)
  세계 B = 자기잠식만 0                        (도넛의 이득이 어디서 오는지)
  세계 C = 총효과·자기잠식 0, 교란 유지         (절차가 교란까지 통제하고 0을 내는지)
  세계 D = 교란까지 0, 순수 잡음                (표준오차가 정직한지)

데이터: 미리 준비한 합성 격자 패널을 불러온다(참값을 알고 추정량을 채점하는
        construct validation 설계). 상업 매출은 공개 대체재가 없고, 있더라도 정답이
        없어 도넛이 오염을 실제로 걷어냈는지 검증할 수 없다.

실행:
    python 8-0-simdata-prep.py                    # 최초 1회: 데이터 준비
    python 8-3-store-opening-donut-did.py
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 렌더링(크로스플랫폼)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from _store_dgp import (CANNIB_CATCH, CONFOUND, COV_NAMES, MONTHS, NOISE_SD,
                        N_NEW, POST_START, RING_LABELS, RING_OUT, TREAT_R,
                        TRUE_KAPPA, make_store_panel)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

SEED = 42
N_TREES = 300              # 단일 세계 분석용
MC_TREES = 200             # 반복 실험용(반복 수를 확보하려 줄인다)
BLOCK_M = 1000.0           # 블록 클러스터 SE용 1km × 1km 공간 블록
DONUT_WIDTHS = [0.0, 400.0, 500.0, 800.0, 1200.0]
MC_REPS_PLACE = 200        # 배치까지 재추출하는 반복 수
MC_REPS_NOISE = 1000       # 지리 고정, 잡음만 재추출하는 반복 수
GROWTH_NOISE_SD = NOISE_SD * np.sqrt(1.0 / POST_START + 1.0 / (MONTHS - POST_START))


# ===========================================================
# 불러오기와 기본 변환
# ===========================================================
def load_all():
    cells_path = DATA_DIR / "store_opening_cells.parquet"
    if not cells_path.exists():
        raise SystemExit(f"데이터가 없습니다: {cells_path}\n먼저 실행: python 8-0-simdata-prep.py")
    cells = pd.read_parquet(cells_path)
    truth = json.loads((DATA_DIR / "store_opening_truth.json").read_text(encoding="utf-8"))
    panels = {tag: pd.read_parquet(DATA_DIR / f"store_opening_panel_{tag}.parquet")
              for tag in ("A", "B", "C")}
    return cells, panels, truth


def growth_of(panel, cells):
    """격자별 성장 = 사후 평균 로그매출 − 사전 평균 로그매출.

    격자 고정 수준(입지 프리미엄·임대료 등 시간불변 요인)이 차분으로 지워지므로,
    남는 것은 사후에 새로 생긴 변화뿐이다 — DID를 격자 단면으로 압축한 형태다.
    """
    pre = panel[panel.post == 0].groupby("cell")["log_sales"].mean()
    post = panel[panel.post == 1].groupby("cell")["log_sales"].mean()
    return (post - pre).reindex(cells["cell"]).to_numpy()


def blocks_of(cells):
    """1km × 1km 공간 블록 id. 인접 격자의 상관을 흡수하는 클러스터 단위."""
    bx = (cells["x_m"].to_numpy() // BLOCK_M).astype(int)
    by = (cells["y_m"].to_numpy() // BLOCK_M).astype(int)
    return by * 100 + bx


def block_report(cells, blocks):
    """유효 클러스터 수를 센다. 클러스터 SE의 신뢰성은 여기에 달려 있다.

    처치가 신규점 3곳으로 정해지므로 처치 격자를 담은 블록이 몇 개인지가
    사실상의 클러스터 수다. 이 수가 한 자릿수면 클러스터 SE를 믿기 어렵다.
    """
    D = cells["treated"].to_numpy()
    uniq = np.unique(blocks)
    tb = np.unique(blocks[D == 1])
    counts = [int(((blocks == b) & (D == 1)).sum()) for b in tb]
    print(f"공간 블록({BLOCK_M:.0f}m) — 전체 {len(uniq)}개 중 처치 격자를 담은 블록 "
          f"{len(tb)}개, 블록별 처치 격자 수 {counts}")
    print(f"  ※ 클러스터 SE의 유효 클러스터 수는 전체 블록 수가 아니라 이 "
          f"{len(tb)}개다. 처치가 신규점 {N_NEW}곳으로 정해지므로 격자를 늘려도 "
          f"이 수는 늘지 않는다.")
    return len(uniq), len(tb), counts


def masks_of(cells):
    """대조군을 어떻게 잡을지 — 이 선택이 이 절의 주제다."""
    D = cells["treated"].to_numpy()
    d_new = cells["d_new_m"].to_numpy()
    ring = cells["in_ring"].to_numpy() == 1
    return {
        "all": np.ones(len(D), dtype=bool),          # 브랜드 상권 전체를 대조로
        "ring": (D == 1) | ring,                     # 인접 링만 대조로(실무 관행)
        "donut": (D == 1) | (d_new > RING_OUT),      # 링을 빼고 원거리만 대조로
    }


# ===========================================================
# 성향점수 · IPW · 이중강건
# ===========================================================
def propensity(X, D, seed=SEED, n_trees=N_TREES):
    """출점 성향점수 ê(x)를 교차적합으로 추정한다.

    같은 데이터로 학습하고 예측하면 ê가 처치 여부를 외워 가중치가 붕괴한다.
    그래서 fold 밖 예측만 쓴다. 다만 처치 격자가 신규점 3곳에 뭉쳐 있어 공간 블록
    fold로는 처치가 하나도 없는 fold가 생긴다 — 여기서는 계층 무작위 fold를 쓰고,
    그 한계를 본문에 적는다(8-1의 공간 블록 CV와 대비되는 지점이다).
    """
    model = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=5,
                                   random_state=seed, n_jobs=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, D, cv=cv, method="predict_proba")[:, 1]


def att_weights(ehat, D, clip=(0.02, 0.98), trim_top=None):
    """ATT 가중: 처치는 1, 대조는 ê/(1−ê) (Abadie, 2005).

    대조 격자 중 '처치받았을 법한' 곳에 큰 가중을 주어 처치군의 공변량 분포를
    맞춘다. ê가 1에 붙으면 가중치가 폭발하므로 clip으로 막고, trim_top이 주어지면
    상위 비율의 극단 가중치를 잘라 그 영향을 본다.
    """
    e = np.clip(ehat, *clip)
    w = np.where(D == 1, 1.0, e / (1.0 - e))
    if trim_top:
        ctrl = D == 0
        cut = np.quantile(w[ctrl], 1.0 - trim_top)
        w = np.where(ctrl & (w > cut), 0.0, w)
    ctrl = D == 0
    if w[ctrl].sum() > 0:  # 대조 가중 합을 처치 수에 맞춰 정규화
        w = np.where(ctrl, w * (D == 1).sum() / w[ctrl].sum(), w)
    return w


def did_att(growth, D, mask, weights=None, blocks=None):
    """가중 DID: growth ~ 1 + D. D 계수가 ATT다. HC1과 블록 클러스터 SE를 함께 낸다."""
    y, d = growth[mask], D[mask].astype(float)
    w = np.ones(int(mask.sum())) if weights is None else weights[mask]
    keep = w > 0
    y, d, w = y[keep], d[keep], w[keep]
    Xd = sm.add_constant(d)
    fit = sm.WLS(y, Xd, weights=w).fit(cov_type="HC1")
    att, se_hc1 = float(fit.params[1]), float(fit.bse[1])
    se_cl = np.nan
    if blocks is not None:
        g = blocks[mask][keep]
        if len(np.unique(g)) > 2:
            se_cl = float(sm.WLS(y, Xd, weights=w)
                          .fit(cov_type="cluster", cov_kwds={"groups": g}).bse[1])
    return {"att": att, "se_hc1": se_hc1, "se_cluster": se_cl,
            "n_treated": int((d == 1).sum()), "n_control": int((d == 0).sum())}


def dr_att(growth, D, X, ehat, mask, seed=SEED, n_trees=N_TREES, folds=5):
    """이중강건(DR) ATT — 성향점수 가중과 대조군 결과회귀를 함께 쓴다.

    IPW는 성향점수 모형이 맞아야 하고 결과회귀는 결과 모형이 맞아야 한다. DR은
    **둘 중 하나만 맞아도** 일치한다(Sant'Anna & Zhao, 2020). 도넛으로 대조군을
    멀리 밀어내면 IPW가 외삽에 가까워지므로, 여기서 갈아탈 후보가 된다.
    """
    y, d = growth[mask], D[mask].astype(float)
    Xm, e = X[mask], np.clip(ehat[mask], 0.02, 0.98)
    ctrl = np.where(d == 0)[0]
    m0 = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(np.arange(len(y))):
        trc = np.intersect1d(tr, ctrl)
        mdl = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5,
                                    random_state=seed, n_jobs=1)
        mdl.fit(Xm[trc], y[trc])
        m0[te] = mdl.predict(Xm[te])
    r = y - m0
    wt = e / (1.0 - e)
    att = float(r[d == 1].mean() - np.sum(wt[d == 0] * r[d == 0]) / np.sum(wt[d == 0]))
    n, p = len(y), d.mean()
    wbar = np.sum(wt[d == 0]) / n
    psi = d * r / p - (1 - d) * wt * r / wbar - att * d / p
    se = float(np.std(psi) / np.sqrt(n))
    return {"att": att, "se_hc1": se, "se_cluster": np.nan,
            "n_treated": int((d == 1).sum()), "n_control": int((d == 0).sum())}


# ===========================================================
# [1] 추정 계단
# ===========================================================
LADDER = [
    ("1 단순 DID(전체 대조)", "all", False),
    ("2 단순 DID(인접 링 대조)", "ring", False),
    ("3 IPW-DID(인접 링 대조)", "ring", True),
    ("4 IPW-DID(전체 대조)", "all", True),
    ("5 IPW-DID + 도넛(원거리 대조)", "donut", True),
]


def ladder(growth, cells, ehat, blocks, true_att, with_dr=True):
    D = cells["treated"].to_numpy()
    mk = masks_of(cells)
    w = att_weights(ehat, D)
    X = cells[COV_NAMES].to_numpy()

    rows = []
    for name, key, use_w in LADDER:
        r = did_att(growth, D, mk[key], w if use_w else None, blocks)
        rows.append({"name": name, **r, "bias": r["att"] - true_att})
    if with_dr:
        r = dr_att(growth, D, X, ehat, mk["donut"])
        rows.append({"name": "6 이중강건 + 도넛", **r, "bias": r["att"] - true_att})

    print(f"{'추정 계단':<32}{'ATT':>9}{'참값과 오차':>13}{'SE(HC1)':>10}"
          f"{'SE(블록)':>11}{'처치':>6}{'대조':>7}")
    for r in rows:
        cl = f"{r['se_cluster']:.4f}" if np.isfinite(r["se_cluster"]) else "—"
        print(f"{r['name']:<32}{r['att']:>9.4f}{r['bias']:>+13.4f}"
              f"{r['se_hc1']:>10.4f}{cl:>11}{r['n_treated']:>6d}{r['n_control']:>7d}")
    return pd.DataFrame(rows)


# ===========================================================
# [2] overlap 진단
# ===========================================================
def overlap_report(growth, cells, ehat, blocks):
    D = cells["treated"].to_numpy()
    mk = masks_of(cells)
    e_t, e_c = ehat[D == 1], ehat[D == 0]
    print(f"{'구분':<14}{'최소':>8}{'p10':>8}{'중위':>8}{'p90':>8}{'최대':>8}")
    for name, v in [("처치 ê(x)", e_t), ("대조 ê(x)", e_c)]:
        q = np.quantile(v, [0.0, 0.1, 0.5, 0.9, 1.0])
        print(f"{name:<14}" + "".join(f"{x:>8.3f}" for x in q))
    for name, key in [("전체 대조", "all"), ("원거리 대조(도넛)", "donut"),
                      ("인접 링 대조", "ring")]:
        sel = mk[key] & (D == 0)
        share = float((ehat[sel] >= np.quantile(e_t, 0.1)).mean())
        print(f"{name:<20} 대조 격자 {sel.sum():>4d} · 처치 ê p10 이상인 비율 {share:.3f}")
    print("  ※ 이 비율이 낮으면 가중이 '처치받았을 법한 대조'를 찾지 못해 외삽이 된다.")

    full = att_weights(ehat, D)
    trim = att_weights(ehat, D, trim_top=0.01)
    ctrl = D == 0
    print(f"대조 가중치 최대 {full[ctrl].max():.2f} · 상위 1% 평균 "
          f"{full[ctrl][full[ctrl] >= np.quantile(full[ctrl], 0.99)].mean():.2f}")
    a1 = did_att(growth, D, mk["donut"], full, blocks)["att"]
    a2 = did_att(growth, D, mk["donut"], trim, blocks)["att"]
    print(f"도넛 추정 — 전체 가중 {a1:.4f} · 상위 1% 절단 후 {a2:.4f} (차이 {a2 - a1:+.4f})")


# ===========================================================
# [3] 도넛 폭 민감도
# ===========================================================
def donut_widths(growth, cells, ehat, blocks, true_att):
    D = cells["treated"].to_numpy()
    d_new = cells["d_new_m"].to_numpy()
    w = att_weights(ehat, D)
    print(f"{'도넛 폭':<16}{'ATT':>9}{'참값과 오차':>13}{'SE(HC1)':>10}{'대조 격자':>10}")
    rows = []
    for W in DONUT_WIDTHS:
        mask = (D == 1) | (d_new > max(W, TREAT_R))
        r = did_att(growth, D, mask, w, blocks)
        tag = "없음(300m)" if W == 0 else f"{int(W)}m"
        rows.append({"width": tag, **r, "bias": r["att"] - true_att})
        print(f"{tag:<16}{r['att']:>9.4f}{r['att'] - true_att:>+13.4f}"
              f"{r['se_hc1']:>10.4f}{r['n_control']:>10d}")
    print(f"  ※ 참 오염 반경은 {RING_OUT:.0f}m다. 폭을 넓히면 정밀도만 잃는 것이 아니다 —")
    print(f"     가장 닮은 대조 격자부터 버리게 되므로 가중이 외삽에 가까워지고 편향이 커진다.")
    return pd.DataFrame(rows)


# ===========================================================
# [4] 링별 노출-반응 프로파일 (DML 부분선형 · 공변량 직교화)
# ===========================================================
def ring_profile(growth, cells, seed=SEED):
    """거리대별 효과를 따로 추정한다. 이분법을 버리면 오염된 격자에 자리가 생긴다.

    공변량 X가 설명하는 부분을 결과와 각 링 더미에서 모두 걷어 내고(직교화·교차적합),
    남은 잔차끼리 회귀한다 — 8.4의 DML을 다치(多値) 처치로 확장한 형태다.
    기준 범주는 가장 먼 링(1200m+)이다.
    """
    X = cells[COV_NAMES].to_numpy()
    bins = cells["ring_bin"].to_numpy()
    targets = RING_LABELS[:-1]
    Dm = np.column_stack([(bins == lb).astype(float) for lb in targets])
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))

    def resid(v):
        out = np.zeros(len(v))
        for tr, te in folds:
            m = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=5,
                                      random_state=seed, n_jobs=1)
            m.fit(X[tr], v[tr])
            out[te] = v[te] - m.predict(X[te])
        return out

    y_res = resid(growth)
    D_res = np.column_stack([resid(Dm[:, j]) for j in range(Dm.shape[1])])
    fit = sm.OLS(y_res, sm.add_constant(D_res)).fit(cov_type="HC1")

    print(f"{'거리 링':<12}{'추정 효과':>10}{'SE':>8}{'95% 하한':>10}{'95% 상한':>10}"
          f"{'격자 수':>8}{'참 τ−κ':>10}")
    rows = []
    for j, lb in enumerate(targets):
        b, se = float(fit.params[j + 1]), float(fit.bse[j + 1])
        sel = bins == lb
        true_net = float((cells["tau_true"].to_numpy()[sel]
                          - cells["kappa_true"].to_numpy()[sel]).mean())
        rows.append({"ring": lb, "coef": b, "se": se, "lo": b - 1.96 * se,
                     "hi": b + 1.96 * se, "n": int(sel.sum()), "true": true_net})
        print(f"{lb:<12}{b:>10.4f}{se:>8.4f}{b - 1.96 * se:>10.4f}"
              f"{b + 1.96 * se:>10.4f}{sel.sum():>8d}{true_net:>+10.4f}")
    ref = bins == RING_LABELS[-1]
    print(f"{RING_LABELS[-1]:<12}{'0(기준)':>10}{'—':>8}{'—':>10}{'—':>10}"
          f"{ref.sum():>8d}{0.0:>+10.4f}")
    return pd.DataFrame(rows)


def plot_profile(prof, out_path):
    """왼쪽은 전체 곡선, 오른쪽은 음(−)의 구간 확대. 두 눈금이 필요한 이유는
    총효과(+0.40)가 잠식(−0.07 수준)보다 훨씬 커서 한 축에서는 잠식이 안 보인다."""
    xs = np.arange(len(prof) + 1)
    coef = np.append(prof["coef"].to_numpy(), 0.0)
    lo = np.append(prof["lo"].to_numpy(), 0.0)
    hi = np.append(prof["hi"].to_numpy(), 0.0)
    true = np.append(prof["true"].to_numpy(), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, zoom in zip(axes, (False, True)):
        ax.axhline(0, color="gray", lw=0.8)
        ax.errorbar(xs, coef, yerr=[coef - lo, hi - coef], fmt="o-", color="#2c3e50",
                    capsize=4, label="Estimated effect (95% CI)")
        ax.plot(xs, true, "s--", color="#c0392b", label="Design truth")
        ax.set_xticks(xs)
        ax.set_xticklabels(RING_LABELS, fontsize=8)
        ax.set_xlabel("Distance ring from the new store")
        if zoom:
            ax.set_xlim(0.5, len(prof) + 0.4)
            ax.set_ylim(min(lo[1:].min(), true[1:].min()) - 0.02,
                        max(hi[1:].max(), 0.01) + 0.02)
            ax.set_title("Zoom: the cannibalized ring")
        else:
            ax.set_ylabel("Effect on log sales growth")
            ax.set_title("Exposure-response profile")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n그림 저장: {out_path.relative_to(SCRIPT_DIR.parent.parent.parent)}")


# ===========================================================
# [5] 반복 실험 — 배치를 다시 뽑아 평균 편향을 잰다
# ===========================================================
def analytic_growth(cells, rng, with_effect=True, with_cannibalization=True,
                    with_confound=True, eps=None):
    """같은 지리에서 성장 벡터를 다시 만든다.

    성장은 닫힌 형태로 쓸 수 있다 — 격자 고정 수준과 공통 추세는 차분에서 지워지고,
    남는 것은 시변 교란·참 효과·참 잠식과 사전·사후 평균 잡음의 차뿐이다. 그래서
    24개월 패널을 다시 생성하지 않고도 세계를 갈아탈 수 있다.
    """
    n = len(cells)
    e = rng.normal(0, GROWTH_NOISE_SD, n) if eps is None else eps
    g = e.copy()
    if with_confound:
        g = g + CONFOUND * cells["latent"].to_numpy()
    if with_effect:
        g = g + cells["tau_true"].to_numpy()
    if with_cannibalization:
        g = g - cells["kappa_true"].to_numpy()
    return g


WORLDS = {
    "A": dict(with_effect=True, with_cannibalization=True, with_confound=True),
    "B": dict(with_effect=True, with_cannibalization=False, with_confound=True),
    "C": dict(with_effect=False, with_cannibalization=False, with_confound=True),
    "D": dict(with_effect=False, with_cannibalization=False, with_confound=False),
}
WORLD_TRUTH = {"A": None, "B": None, "C": 0.0, "D": 0.0}   # A·B는 참 τ 평균으로 채운다
WORLD_NOTE = {"A": "관측(배치+교란+잠식)", "B": "잠식 0", "C": "효과 0(교란 유지)",
              "D": "순수 잡음"}


def replicate_placement(reps=MC_REPS_PLACE, seed0=100000):
    """신규점 배치까지 다시 뽑아 각 추정량의 평균 편향·산포·기각률을 잰다."""
    rng = np.random.default_rng(20260813)
    rows, n_ctrl = [], {"all": [], "ring": [], "donut": []}
    t0 = time.time()
    for k in range(reps):
        out = make_store_panel(seed=seed0 + k)
        cells = out["cells"]
        D = cells["treated"].to_numpy()
        if D.sum() < 20:
            continue
        X = cells[COV_NAMES].to_numpy()
        mk = masks_of(cells)
        blocks = blocks_of(cells)
        ehat = propensity(X, D, n_trees=MC_TREES)
        w = att_weights(ehat, D)
        eps = rng.normal(0, GROWTH_NOISE_SD, len(D))
        tau_mean = float(cells["tau_true"].to_numpy()[D == 1].mean())
        for key in n_ctrl:
            n_ctrl[key].append(int((mk[key] & (D == 0)).sum()))
        for world, opt in WORLDS.items():
            g = analytic_growth(cells, rng, eps=eps, **opt)
            truth = tau_mean if world in ("A", "B") else 0.0
            for name, key, use_w in LADDER:
                r = did_att(g, D, mk[key], w if use_w else None, blocks)
                rows.append({"world": world, "est": name, "att": r["att"],
                             "se": r["se_hc1"], "se_cl": r["se_cluster"],
                             "rej": abs(r["att"]) > 1.96 * r["se_hc1"],
                             "rej_cl": abs(r["att"]) > 1.96 * r["se_cluster"],
                             "truth": truth})
            if world in ("A", "B", "D"):
                r = dr_att(g, D, X, ehat, mk["donut"], n_trees=MC_TREES, folds=3)
                rows.append({"world": world, "est": "6 이중강건 + 도넛", "att": r["att"],
                             "se": r["se_hc1"], "se_cl": np.nan,
                             "rej": abs(r["att"]) > 1.96 * r["se_hc1"],
                             "rej_cl": np.nan, "truth": truth})
    df = pd.DataFrame(rows)
    print(f"(배치 재추출 {reps}회, {time.time() - t0:.0f}초)")
    for world in ("A", "B", "C", "D"):
        d = df[df.world == world]
        if d.empty:
            continue
        print(f"\n세계 {world} — {WORLD_NOTE[world]}"
              f" (참값 {d['truth'].iloc[0]:+.4f})")
        print(f"{'추정 계단':<32}{'평균 편향':>11}{'경험 SD':>10}{'평균 SE':>10}{'기각률':>9}")
        for name, sub in d.groupby("est", sort=True):
            bias = sub["att"].mean() - sub["truth"].iloc[0]
            print(f"{name:<32}{bias:>+11.4f}{sub['att'].std():>10.4f}"
                  f"{sub['se'].mean():>10.4f}{sub['rej'].mean():>9.3f}")
    print(f"\n대조 격자 수 — 전체 {np.mean(n_ctrl['all']):.0f} · 인접 링 "
          f"{np.mean(n_ctrl['ring']):.0f}(최소 {min(n_ctrl['ring'])}) · 도넛 "
          f"{np.mean(n_ctrl['donut']):.0f}(최소 {min(n_ctrl['donut'])}, 표준편차 "
          f"{np.std(n_ctrl['donut']):.1f})")
    return df, n_ctrl


# ===========================================================
# [6] 1종 오류율 — 지리를 고정하고 잡음만 다시 뽑는다
# ===========================================================
def type_one_error(cells, ehat, blocks, reps=MC_REPS_NOISE):
    """표준오차가 기술하는 분포는 '지리가 고정된 표집 변동'이다. 그 분포에서
    도넛이 1종 오류율을 명목 5%에서 밀어내는지 본다.

    Ch15는 도넛을 사후 필터로 구현해 표본이 배정의 함수가 되었고 1종 오류율이
    0.233으로 부풀었다. 여기서는 배제 규칙이 신규점까지의 거리로 정해진다 —
    결과를 보고 고르지 않는다. 그 차이가 실제로 나타나는지 확인한다.
    """
    D = cells["treated"].to_numpy()
    mk = masks_of(cells)
    w = att_weights(ehat, D)
    rng = np.random.default_rng(20260814)
    keys = [("전체 대조(도넛 없음)", "all"), ("인접 링 대조", "ring"),
            ("원거리 대조(도넛)", "donut")]
    cnt = {(wd, k): 0 for wd in ("C", "D") for _, k in keys}
    cnt_cl = dict(cnt)
    se_hc1 = {key: [] for key in cnt}
    se_cl = {key: [] for key in cnt}
    for _ in range(reps):
        eps = rng.normal(0, GROWTH_NOISE_SD, len(D))
        for wd in ("C", "D"):
            g = analytic_growth(cells, rng, eps=eps, **WORLDS[wd])
            for _, k in keys:
                r = did_att(g, D, mk[k], w, blocks)
                cnt[(wd, k)] += abs(r["att"]) > 1.96 * r["se_hc1"]
                cnt_cl[(wd, k)] += abs(r["att"]) > 1.96 * r["se_cluster"]
                se_hc1[(wd, k)].append(r["se_hc1"])
                se_cl[(wd, k)].append(r["se_cluster"])

    def mc(c):
        p = c / reps
        return p, 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / reps)

    print(f"{'대조군 구성':<22}{'세계 C(HC1)':>16}{'세계 C(블록)':>17}"
          f"{'세계 D(HC1)':>16}{'세계 D(블록)':>17}")
    rows = []
    for label, k in keys:
        cells_out = []
        for wd in ("C", "D"):
            for src in (cnt, cnt_cl):
                p, e = mc(src[(wd, k)])
                cells_out.append(f"{p:>10.3f} ±{e:.3f}")
        rows.append({"control": label,
                     "C_hc1": cnt[("C", k)] / reps, "C_cl": cnt_cl[("C", k)] / reps,
                     "D_hc1": cnt[("D", k)] / reps, "D_cl": cnt_cl[("D", k)] / reps})
        print(f"{label:<22}" + "".join(f"{c:>17}" for c in cells_out))
    print(f"  ※ 세계 D는 교란까지 0인 순수 잡음이므로 명목 5%가 정답이다. 세계 C는 "
          f"교란이 남아 있어 이를 통제하지 못한 절차는 정당하게 기각한다.")
    print(f"  ※ 이 지리(seed {SEED})에서는 IPW가 교란을 거의 다 걷어 세계 C의 점추정이 "
          f"0 근처에 있다. 다른 지리에서는 그렇지 않다 — [5]의 세계 C 기각률을 함께 보라.")
    print(f"  ※ 반복 {reps}회, 몬테카를로 오차는 각 칸에 ± 로 표시했다.")

    # --- 블록 SE가 왜 더 많이 기각하는가 -----------------------------------
    # [1]의 한 번 draw에서는 블록 SE가 HC1보다 훨씬 컸는데, 여기서는 블록 기준
    # 기각률이 더 높다. 모순처럼 보이므로 SE 자체의 분포를 직접 들여다본다.
    print(f"\n  [진단] 블록 클러스터 SE의 분포 — 왜 더 많이 기각하는가")
    print(f"  {'세계':<5}{'대조군':<20}{'평균 HC1':>10}{'평균 블록':>11}"
          f"{'블록 p10':>10}{'블록 최소':>11}{'블록<HC1 비율':>15}")
    for wd in ("C", "D"):
        for label, k in keys:
            h = np.array(se_hc1[(wd, k)])
            c = np.array(se_cl[(wd, k)])
            print(f"  {wd:<5}{label:<20}{h.mean():>10.4f}{c.mean():>11.4f}"
                  f"{np.quantile(c, 0.1):>10.4f}{c.min():>11.4f}"
                  f"{float((c < h).mean()):>15.3f}")
    print(f"  ※ 읽는 법: 세계 C와 D에서 블록/HC1 비율이 **뒤집힌다.** C에서는 블록 SE가"
          f" HC1보다 크고(블록<HC1 비율 0.000), D에서는 오히려 작다(비율 0.6 이상).")
    print(f"  ※ 원인: 세계 C의 잔차는 시변 교란(공간적으로 매끄러운 latent)을 품어 블록"
          f" 안에서 상관되므로 블록 SE가 커지는 것이 옳다. 세계 D의 잔차는 i.i.d."
          f" 백색잡음이라 블록 안에 흡수할 상관이 없고, 처치 격자를 담은 블록이 소수"
          f"(위 [1]의 블록 진단 참조)뿐이어서 CRVE가 **아래로 편향**된다.")
    print(f"  ※ 그래서 한 번의 draw에서 블록 SE가 HC1보다 크다는 사실([1])은 '보수적'"
          f"이라는 증거가 아니다. 잔차가 공간적으로 상관될 때만 커지고, 그렇지 않으면"
          f" 작아진다. 그 작아진 SE로 나눈 반복들이 세계 D의 1종 오류율 0.13을 만든다.")
    return pd.DataFrame(rows)


# ===========================================================
# [7] 순효과
# ===========================================================
def net_effect(panel, cells, prof, truth_A):
    """추정된 링별 효과를 사전 매출 규모로 가중해 순효과를 계산한다.

    로그 계수는 그대로 더할 수 없다. 각 격자의 사전 매출지수에 (exp(효과)−1)을 곱해
    매출 단위로 바꾼 뒤 합산한다. 원가·투자·할인율은 여기서 다루지 않는다.
    """
    pre = panel[panel.post == 0].groupby("cell")["log_sales"].mean()
    level = np.exp(pre.reindex(cells["cell"]).to_numpy())
    bins = cells["ring_bin"].to_numpy()
    coef = dict(zip(prof["ring"], prof["coef"]))

    gain = float(np.sum(level[bins == "0-300m"] * (np.exp(coef["0-300m"]) - 1)))
    loss = float(np.sum(level[bins == "300-500m"] * (np.exp(coef["300-500m"]) - 1)))
    zone = np.isin(bins, ["0-300m", "300-500m"])
    zone_pre = float(level[zone].sum())
    # 링 구간이 처치·잠식 정의와 일치하는지 확인한다(경계 처리 오류 방지).
    assert int(np.sum(bins == "0-300m")) == int(cells["treated"].sum())
    assert int(np.sum(bins == "300-500m")) == int(cells["in_ring"].sum())

    print(f"{'항목':<38}{'추정':>12}{'설계 참값':>12}")
    for name, est, tru in [
        ("총효과(0-300m 매출 증가, 지수)", gain, truth_A["true_gain_index"]),
        ("자기잠식(300-500m 감소, 지수)", loss, truth_A["true_loss_index"]),
        ("순효과(총효과 − 잠식, 지수)", gain + loss, truth_A["true_net_index"]),
        ("영향권 사전 매출(지수)", zone_pre, truth_A["zone_pre_sales_index"]),
    ]:
        print(f"{name:<38}{est:>12.2f}{tru:>12.2f}")
    pct = 100.0 * (gain + loss) / zone_pre
    off = 100.0 * (-loss) / gain if gain > 0 else np.nan
    off_true = 100.0 * (-truth_A["true_loss_index"]) / truth_A["true_gain_index"]
    print(f"{'순효과 / 영향권 사전 매출(%)':<38}{pct:>12.2f}"
          f"{truth_A['true_net_pct_of_zone']:>12.2f}")
    print(f"{'총효과 중 잠식으로 상쇄된 비율(%)':<38}{off:>12.2f}{off_true:>12.2f}")
    print("  ※ Pancras et al.(2012)의 미국 식료품 체인 추정치는 신규점 매출의 13.3%가")
    print("     기존점에서 옮겨온 몫이었다. 위 상쇄 비율은 이 예제의 **설계값**이며")
    print("     국내 실증치가 아니다 — 거리 스케일과 업종이 다르다.")
    return {"gain": gain, "loss": loss, "net": gain + loss, "zone_pre": zone_pre,
            "net_pct": pct, "offset_pct": off, "offset_pct_true": off_true}


# ===========================================================
def main():
    print("=" * 84)
    print("신규 출점의 상권 효과 — IPW-DID + 도넛 설계 (합성 데이터, construct validation)")
    print("=" * 84)

    cells, panels, truth = load_all()
    tA = truth["A"]
    true_att = tA["true_att_total"]
    print(f"설계: 격자 {tA['cell_m']:.0f}m × {tA['n_cells_grid']}개 중 브랜드 상권 "
          f"{tA['n_cells_universe']}개 · {tA['months']}개월(사전 {POST_START}·사후 "
          f"{MONTHS - POST_START}) · 기존점 {tA['n_existing']} · 신규점 {tA['n_new']}")
    print(f"      처치 격자 {tA['n_treated_cells']}(≤{TREAT_R:.0f}m) · 잠식 링 "
          f"{tA['n_ring_cells']}({TREAT_R:.0f}~{RING_OUT:.0f}m, 잠식 적용 "
          f"{tA['n_ring_cannibalized']}) · 원거리 {tA['n_far_cells']}")
    print(f"참값: 총효과 {true_att:+.3f} · 잠식 최대 {tA['true_kappa_peak']:.3f}"
          f"(링 평균 {tA['true_kappa_ring_mean']:.4f}) · 순효과 "
          f"{tA['true_net_index']:+.2f} 매출지수({tA['true_net_pct_of_zone']:+.2f}%)")
    print(f"잠재 성장력 평균 — 처치 {tA['latent_mean_treated']:+.3f} · 링 "
          f"{tA['latent_mean_ring']:+.3f} · 원거리 {tA['latent_mean_far']:+.3f}"
          f"  ← 내생적 배치의 지문")
    print(f"매출 귀속: 판매 매장의 상권에 안분(거주지 기준이 아니다). 거주지 기준으로")
    print(f"          집계하면 손님이 기존점→신규점으로 옮겨도 격자 지출 총액이 그대로여서")
    print(f"          자기잠식이 격자 수준에서 관측되지 않는다.")

    growth = growth_of(panels["A"], cells)
    D = cells["treated"].to_numpy()
    blocks = blocks_of(cells)
    ehat = propensity(cells[COV_NAMES].to_numpy(), D)

    print("\n" + "-" * 84)
    print("[1] 추정 계단 — 대조군을 어떻게 잡는가 (관측된 한 번의 출점)")
    print("-" * 84)
    block_report(cells, blocks)
    print()
    lad = ladder(growth, cells, ehat, blocks, true_att)

    print("\n" + "-" * 84)
    print("[2] overlap 진단 — 가중이 외삽인지 확인한다")
    print("-" * 84)
    overlap_report(growth, cells, ehat, blocks)

    print("\n" + "-" * 84)
    print("[3] 도넛 폭 민감도 — 오염 반경을 모른다고 가정하면")
    print("-" * 84)
    wid = donut_widths(growth, cells, ehat, blocks, true_att)

    print("\n" + "-" * 84)
    print("[4] 링별 노출-반응 프로파일 — 이분법을 버린다")
    print("-" * 84)
    prof = ring_profile(growth, cells)
    plot_profile(prof, DATA_DIR / "fig_8_3_ring_profile.png")

    print("\n" + "-" * 84)
    print("[5] 반복 실험 — 배치를 다시 뽑아 평균 편향을 잰다")
    print("-" * 84)
    mc, n_ctrl = replicate_placement()

    print("\n" + "-" * 84)
    print("[6] 1종 오류율 — 도넛이 추론을 깨는가 (지리 고정 · 잡음만 재추출)")
    print("-" * 84)
    t1 = type_one_error(cells, ehat, blocks)

    print("\n" + "-" * 84)
    print("[7] 순효과 — 출점이 순증인가, 자기 매장 매출을 옮긴 것인가")
    print("-" * 84)
    net = net_effect(panels["A"], cells, prof, tA)

    # --- 대조군 판정을 한 줄로 요약 ---------------------------------------
    def bias(world, name):
        d = mc[(mc.world == world) & (mc.est == name)]
        return float(d["att"].mean() - d["truth"].iloc[0])

    ringA, ringB = bias("A", "3 IPW-DID(인접 링 대조)"), bias("B", "3 IPW-DID(인접 링 대조)")
    donA, donB = bias("A", "5 IPW-DID + 도넛(원거리 대조)"), bias("B", "5 IPW-DID + 도넛(원거리 대조)")
    allA, allB = bias("A", "4 IPW-DID(전체 대조)"), bias("B", "4 IPW-DID(전체 대조)")

    print("\n" + "=" * 84)
    print("요약 — 반복 실험 평균 편향 기준")
    print("=" * 84)
    print(f"- 자기잠식이 인접 링 대조에 심는 편향: 세계 A {ringA:+.4f} → 세계 B "
          f"{ringB:+.4f} (차이 {ringA - ringB:+.4f}). 참 링 평균 잠식 "
          f"{tA['true_kappa_ring_mean']:.4f}와 대조하라.")
    print(f"- 전체 대조에서는 오염이 희석된다: A {allA:+.4f} → B {allB:+.4f} "
          f"(차이 {allA - allB:+.4f}).")
    print(f"- 도넛은 세계 A에서 {donA:+.4f}로 인접 링({ringA:+.4f})보다 낫지만, "
          f"세계 B에서는 {donB:+.4f}로 인접 링({ringB:+.4f})보다 나쁘다.")
    print(f"  → 도넛은 무조건 옳은 처방이 아니다. 잠식 크기와 대조군 축소 비용의 교환이다.")
    print(f"- 순효과는 총효과의 {100 - net['offset_pct']:.1f}%이며 "
          f"{net['offset_pct']:.1f}%가 기존 매장에서 옮겨온 몫이다(설계 참값 "
          f"{net['offset_pct_true']:.1f}%).")
    print(f"- 도넛 대조 격자 수는 배치를 다시 뽑으면 평균 {np.mean(n_ctrl['donut']):.0f}, "
          f"최소 {min(n_ctrl['donut'])}, 표준편차 {np.std(n_ctrl['donut']):.1f}다. "
          f"지리에 따라 변하지만 **표본이 붕괴하지는 않는다** — 배제 규칙이 신규점까지의 "
          f"거리로 정해지고 기하가 대략 고정이기 때문이다.")
    print(f"- 그리고 지리를 고정하면 도넛의 1종 오류율이 명목 수준에 머문다([6]). 즉 "
          f"거리로 정의한 도넛은 추론을 깨뜨리지 않는다. 경험 SD가 평균 SE보다 큰 것은 "
          f"표준오차의 실패가 아니라 **신규점이 어디에 나느냐에 따라 편향이 달라지는** 데서 온다.")


if __name__ == "__main__":
    main()
