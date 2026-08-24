"""
15장 실습 1: 지역 실험의 설계와 검정력 — 시장 40개로 5% 효과를 잡을 수 있는가
================================================================================
결정할 문제
--------------------------------------------------------------------------------
어느 기업이 지역 한정 프로모션의 효과를 재려 한다. 개인 단위로 무작위 배정하고 싶지만
사람은 움직이고 광고는 경계를 모른다. 그래서 무작위화 단위를 **시장**으로 올린다.
시장 40개를 20/20으로 나눠 절반에만 처치한다.

그러면 두 가지를 묻게 된다.
  (1) 이 실험으로 5% 매출 변화를 잡을 수 있는가? (검정력)
  (2) 인접 시장으로 효과가 새는 상황에서 어떤 설계가 참값에 닿는가? (편향)

3층 모델에서의 위치:
  [1층 GIS]  시장 격자와 인접 관계(rook), 군집 블록 구성
  [2층 AI ]  **없다.** 무작위 배정이 식별을 보장하므로 예측 모델이 들어갈 자리가 없다.
             1층 게이트를 통과하지 못하는 문제다(14.2.4의 기준을 그대로 적용).
  [3층 결정] 실험을 할 것인가 말 것인가. 검정력이 없으면 실험은 돈만 쓴다.

이 스크립트는 **추정이 아니라 설계 평가**를 한다. 무작위 배정을 1,000회 반복해 설계별
편향·표준오차·검정력·최소 탐지 가능 효과(MDE)를 경험적으로 측정한다.

비교하는 설계 다섯
--------------------------------------------------------------------------------
  설계 1  단순 무작위 + 사후 비교      : 처치 후 두 집단 평균 차이
  설계 2  단순 무작위 + DID            : 사전 대비 변화량의 차이
  설계 3  짝지음 무작위 + DID          : 닮은 시장끼리 짝지어 쌍 안에서 하나씩 처치
  설계 4  짝지음 + DID + 오염 쌍 제외  : 대조 시장이 처치 시장에 인접한 쌍을 버린다
                                        (도넛을 **사후 필터**로 구현한 것)
  설계 5  군집 무작위 + DID            : 2×2 블록 10개를 군집으로 묶어 블록 단위 배정
                                        (파급을 **설계로 흡수**하려는 시도)

대조군(null control) — 심은 메커니즘마다 그것만 제거한 세계를 함께 돌린다
--------------------------------------------------------------------------------
  세계 A (기준)       : 효과 +0.049, 파급 −0.015
  세계 B (파급 0)     : 효과만. → 설계 1~3의 편향이 사라져야 한다. 남으면 편향의
                        원인이 파급이 아니다.
  세계 C (효과·파급 0): 순수 placebo. → 다섯 설계 모두 1종 오류율이 5% 근처여야 한다.
                        초과하면 절차 자체에 편향이 있다.

증명하지 못하는 것: 국내 실제 상권의 파급 거리, 실제 프로모션·출점 효과의 크기.
**이 예제는 방법의 성질을 보이지 시장의 사실을 보이지 않는다.**

실행:
    python 15-0-simdata-prep.py       # 최초 1회
    python 15-1-geo-experiment-power.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

SEED = 42
N_REPS = 1000          # 배정 반복 수. 500 미만으로 내리지 않는다(검정력 추정 오차 때문)
ALPHA = 0.05
POWER_TARGET = 0.80

DESIGNS = ["설계1 단순+사후비교", "설계2 단순+DID", "설계3 짝지음+DID",
           "설계4 짝지음+오염제외", "설계5 군집무작위+DID"]


def load_data():
    for p in ["geoexp_panel.parquet", "geoexp_markets.parquet",
              "geoexp_adjacency.npy", "geoexp_truth.json"]:
        if not (DATA_DIR / p).exists():
            raise SystemExit(f"데이터가 없습니다: {DATA_DIR / p}\n"
                             "먼저 실행: python 15-0-simdata-prep.py")
    panel = pd.read_parquet(DATA_DIR / "geoexp_panel.parquet")
    Y = panel.drop(columns=["market_id"]).to_numpy()
    markets = pd.read_parquet(DATA_DIR / "geoexp_markets.parquet")
    A = np.load(DATA_DIR / "geoexp_adjacency.npy").astype(float)
    truth = json.loads((DATA_DIR / "geoexp_truth.json").read_text(encoding="utf-8"))
    return Y, markets, A, truth


# ===========================================================
# 1. 짝지음 — 사전 궤적이 닮은 시장끼리 짝을 짓는다
# ===========================================================
def greedy_pairs(Y_pre):
    """사전기간 궤적의 유클리드 거리로 탐욕적 짝지음.

    왜 짝지음이 정밀도를 높이는가: 전국 공통 충격은 쌍 안의 두 시장에 함께 오므로 쌍
    안의 차이를 보면 상쇄된다. 남는 것은 처치의 효과와 국지적 잡음뿐이다
    (Imai, King & Nall, 2009의 논지를 시장 단위에 적용).

    짝지음은 **사전 데이터만** 쓴다. 처치 후 정보를 쓰면 설계가 결과를 보고 만들어진다.
    """
    n = Y_pre.shape[0]
    d = np.linalg.norm(Y_pre[:, None, :] - Y_pre[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    unused = set(range(n))
    pairs = []
    while len(unused) >= 2:
        idx = sorted(unused)
        sub = d[np.ix_(idx, idx)]
        i_loc, j_loc = np.unravel_index(np.argmin(sub), sub.shape)
        i, j = idx[i_loc], idx[j_loc]
        pairs.append((i, j))
        unused.discard(i)
        unused.discard(j)
    return np.array(pairs)


# ===========================================================
# 2. 배정 생성 (반복 × 시장)
# ===========================================================
def simple_assignments(rng, n, n_treat, reps):
    """단순 무작위: 매 반복 n개 중 n_treat개를 처치로."""
    Z = np.zeros((reps, n), dtype=bool)
    for r in range(reps):
        Z[r, rng.choice(n, n_treat, replace=False)] = True
    return Z


def paired_assignments(rng, pairs, n, reps):
    """짝지음 무작위: 각 쌍에서 하나를 처치로. 쌍별 독립 동전."""
    Z = np.zeros((reps, n), dtype=bool)
    coin = rng.random((reps, len(pairs))) < 0.5
    a, b = pairs[:, 0], pairs[:, 1]
    Z[:, a] = coin
    Z[:, b] = ~coin
    return Z


def cluster_assignments(rng, block_id, reps):
    """군집 무작위: 블록 단위로 절반을 처치. 같은 블록 시장은 운명을 공유한다."""
    blocks = np.unique(block_id)
    nb = len(blocks)
    Zb = np.zeros((reps, nb), dtype=bool)
    for r in range(reps):
        Zb[r, rng.choice(nb, nb // 2, replace=False)] = True
    return Zb[:, block_id], Zb


# ===========================================================
# 3. 처치·파급 적용 (시장 수준 사후 평균에 상수를 얹는다)
# ===========================================================
def apply_world(post_mean, Z, A, effect, spill):
    """배정 Z에 따라 사후 평균에 효과와 파급을 얹는다.

    처치효과가 사후 전 기간 일정하므로 시장 수준 사후 평균은 상수만큼 이동한다. 주별
    시계열을 다시 만들 필요가 없어 1,000회 반복이 가볍다.

    파급의 대상: 처치받지 않았는데 처치 시장에 인접한 시장. 이 시장의 매출이 내려가면
    대조군 평균이 내려가고, 처치−대조 차이는 그만큼 부풀려진다. **편향의 방향이 정해져
    있으므로 "보수적으로 읽으면 된다"는 변명이 통하지 않는다.**
    """
    exposed = (Z.astype(float) @ A.T) > 0          # (reps, n) 처치 시장에 인접한가
    contaminated = exposed & (~Z)                  # 대조군인데 노출된 시장
    return (post_mean[None, :]
            + effect * Z
            + spill * contaminated), contaminated


# ===========================================================
# 4. 설계별 추정과 검정
# ===========================================================
def two_sample(stat, Z):
    """두 집단 평균 차이와 Welch t검정. stat: (reps, n) 시장(또는 군집) 수준 통계량."""
    reps, n = stat.shape
    nt = Z.sum(axis=1)
    nc = n - nt
    mt = np.where(Z, stat, 0).sum(axis=1) / nt
    mc = np.where(~Z, stat, 0).sum(axis=1) / nc
    vt = (np.where(Z, (stat - mt[:, None]) ** 2, 0).sum(axis=1)) / (nt - 1)
    vc = (np.where(~Z, (stat - mc[:, None]) ** 2, 0).sum(axis=1)) / (nc - 1)
    se = np.sqrt(vt / nt + vc / nc)
    est = mt - mc
    df = (vt / nt + vc / nc) ** 2 / ((vt / nt) ** 2 / (nt - 1) + (vc / nc) ** 2 / (nc - 1))
    p = 2 * stats.t.sf(np.abs(est / se), df)
    return est, se, p, df


def paired_test(delta, Z, pairs, drop_mask=None):
    """쌍 내 차이의 일표본 t검정.

    delta    : (reps, n) 시장별 사전→사후 변화량
    drop_mask: (reps, n_pairs) True면 그 쌍을 버린다(설계 4의 오염 쌍 제외)
    """
    a, b = pairs[:, 0], pairs[:, 1]
    za = Z[:, a]
    d = np.where(za, delta[:, a] - delta[:, b], delta[:, b] - delta[:, a])
    keep = np.ones_like(d, dtype=bool) if drop_mask is None else ~drop_mask

    k = keep.sum(axis=1).astype(float)
    est = np.where(keep, d, 0).sum(axis=1) / np.maximum(k, 1)
    var = (np.where(keep, (d - est[:, None]) ** 2, 0).sum(axis=1)) / np.maximum(k - 1, 1)
    se = np.sqrt(var / np.maximum(k, 1))
    df = np.maximum(k - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(est / se), df)
    # 쌍이 2개 미만 남으면 검정 자체가 성립하지 않는다 → 결측으로 둔다
    est = np.where(k >= 2, est, np.nan)
    p = np.where(k >= 2, p, np.nan)
    return est, se, p, k


def mde(se_mean, df_mean):
    """최소 탐지 가능 효과 — 80% 검정력을 주는 가장 작은 참 효과.

    MDE = (t_{1−α/2, df} + t_{power, df}) × SE. 유의성 문턱을 넘으려면 효과가
    표준오차의 몇 배여야 하는지를 재는 값이다. 표본이 적으면 SE가 커지고 MDE가 커진다
    — "잴 수 있는 가장 작은 변화"가 커진다.
    """
    t_crit = stats.t.ppf(1 - ALPHA / 2, df_mean)
    t_pow = stats.t.ppf(POWER_TARGET, df_mean)
    return (t_crit + t_pow) * se_mean


def evaluate_world(Y, block_id, A, pairs, effect, spill, reps, seed):
    """한 세계(효과·파급 설정)에서 다섯 설계를 평가한다."""
    n = Y.shape[0]
    n_pre = Y.shape[1] // 2
    pre_mean = Y[:, :n_pre].mean(axis=1)
    post_mean = Y[:, n_pre:].mean(axis=1)

    rng = np.random.default_rng(seed)
    Z_simple = simple_assignments(rng, n, n // 2, reps)
    Z_pair = paired_assignments(rng, pairs, n, reps)
    Z_clu, Zb = cluster_assignments(rng, block_id, reps)

    post_s, contam_s = apply_world(post_mean, Z_simple, A, effect, spill)
    post_p, contam_p = apply_world(post_mean, Z_pair, A, effect, spill)
    post_c, contam_c = apply_world(post_mean, Z_clu, A, effect, spill)

    delta_s = post_s - pre_mean[None, :]
    delta_p = post_p - pre_mean[None, :]
    delta_c = post_c - pre_mean[None, :]

    def exposure_share(Z, contam):
        """대조 시장 중 노출된(오염된) 비율 — 설계가 오염을 얼마나 줄였는가."""
        return contam.sum(axis=1) / (~Z).sum(axis=1)

    out = {}

    e, s, p, df = two_sample(post_s, Z_simple)
    out[DESIGNS[0]] = (e, s, p, df, np.full(reps, float(n)), exposure_share(Z_simple, contam_s))

    e, s, p, df = two_sample(delta_s, Z_simple)
    out[DESIGNS[1]] = (e, s, p, df, np.full(reps, float(n)), exposure_share(Z_simple, contam_s))

    e, s, p, k = paired_test(delta_p, Z_pair, pairs)
    out[DESIGNS[2]] = (e, s, p, k - 1, 2 * k, exposure_share(Z_pair, contam_p))

    # 설계 4 — 쌍의 대조 쪽이 오염되었으면 그 쌍을 버린다
    a, b = pairs[:, 0], pairs[:, 1]
    ctrl_contam = np.where(Z_pair[:, a], contam_p[:, b], contam_p[:, a])
    e, s, p, k = paired_test(delta_p, Z_pair, pairs, drop_mask=ctrl_contam)
    # 남은 쌍의 대조군은 정의상 오염되지 않았으므로 노출 비율은 0이다
    out[DESIGNS[3]] = (e, s, p, k - 1, 2 * k, np.zeros(reps))

    # 설계 5 — 군집(블록) 수준으로 집계한 뒤 블록 간 비교
    nb = Zb.shape[1]
    B = np.zeros((nb, n))
    for bi in range(nb):
        m = (block_id == bi)
        B[bi, m] = 1.0 / m.sum()
    delta_block = delta_c @ B.T                     # (reps, nb) 블록 평균 변화량
    e, s, p, df = two_sample(delta_block, Zb)
    out[DESIGNS[4]] = (e, s, p, df, np.full(reps, float(nb)), exposure_share(Z_clu, contam_c))

    return out


def summarize(out, effect):
    """설계별 편향·표준오차·검정력·MDE·노출 비율 표."""
    rows = []
    for name, (est, se, p, df, n_used, expo) in out.items():
        ok = np.isfinite(est) & np.isfinite(p)
        rej = float(np.mean(p[ok] < ALPHA))
        rows.append({
            "설계": name,
            "평균 추정치": float(np.mean(est[ok])),
            "편향": float(np.mean(est[ok]) - effect),
            "경험 SD": float(np.std(est[ok], ddof=1)),
            "평균 SE": float(np.mean(se[ok])),
            "기각률": rej,
            "MC 오차": float(np.sqrt(rej * (1 - rej) / ok.sum())),
            "MDE": float(mde(float(np.mean(se[ok])), float(np.mean(df[ok])))),
            "분석 단위 수": float(np.mean(n_used[ok])),
            "대조군 노출률": float(np.mean(expo[ok])),
            "유효 반복": int(ok.sum()),
        })
    return pd.DataFrame(rows)


def mde_curve(Y, block_id, A, effect, spill, reps, seed):
    """시장 수와 MDE의 관계 — 파급 있음/없음 두 선(그림 15.2). 설계 3 기준."""
    n_pre = Y.shape[1] // 2
    sizes = [10, 20, 30, 40]      # 4×10 격자를 열 단위로 잘라 rook 구조를 보존
    curve = []
    for m in sizes:
        idx = np.arange(m)
        Ym, Am, bm = Y[idx], A[np.ix_(idx, idx)], block_id[idx]
        pairs_m = greedy_pairs(Ym[:, :n_pre])
        for label, sp in [("파급 있음", spill), ("파급 없음", 0.0)]:
            out = evaluate_world(Ym, bm, Am, pairs_m, effect, sp, reps, seed + m)
            est, se, p, df, n_used, _ = out[DESIGNS[2]]
            ok = np.isfinite(est) & np.isfinite(p)
            curve.append({
                "시장 수": m,
                "파급": label,
                "MDE": float(mde(float(np.mean(se[ok])), float(np.mean(df[ok])))),
                "검정력": float(np.mean(p[ok] < ALPHA)),
            })
    return pd.DataFrame(curve)


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    Y, markets, A, truth = load_data()
    block_id = markets["block_id"].to_numpy()
    n_pre = truth["n_weeks_pre"]
    effect = truth["true_effect_log"]
    spill = truth["spillover_log"]

    log("=" * 78)
    log("15-1 지역 실험의 설계와 검정력 — 시장 40개로 5% 효과를 잡을 수 있는가")
    log("=" * 78)
    log(f"시장 {Y.shape[0]}개({truth['grid'][0]}×{truth['grid'][1]} 격자) · "
        f"군집 {truth['n_blocks']}개 · 주 {Y.shape[1]}개(사전 {n_pre}) · 배정 반복 {N_REPS}회")
    log(f"참 효과 = 로그 {effect:+.4f} ({truth['true_effect_pct']:+.2f}%)")
    log(f"파급    = 로그 {spill:+.4f} ({truth['spillover_pct']:+.2f}%)  ← 처치 시장에 인접한 대조 시장")
    log()

    pairs = greedy_pairs(Y[:, :n_pre])
    adj_pairs = sum(1 for i, j in pairs if A[i, j] > 0)
    log(f"[짝지음] 사전 {n_pre}주 궤적 거리로 {len(pairs)}개 쌍 구성")
    log(f"  이 중 지리적으로 인접한 쌍 = {adj_pairs}개 / {len(pairs)}개")
    log("  → 공간 자기상관 때문에 '닮은 시장'과 '이웃 시장'이 겹친다. 짝지음은 좋은")
    log("    설계인데도 오염 위험을 스스로 끌어들이는 구조다.")
    log()

    worlds = [
        ("세계A 기준(효과+파급)", effect, spill),
        ("세계B 대조군(파급 0)", effect, 0.0),
        ("세계C 대조군(효과0·파급0)", 0.0, 0.0),
    ]

    tables = {}
    for wname, eff, sp in worlds:
        tab = summarize(evaluate_world(Y, block_id, A, pairs, eff, sp, N_REPS, SEED), eff)
        tables[wname] = tab
        log("-" * 78)
        log(wname)
        log("-" * 78)
        log(f"{'설계':<24}{'평균추정':>9}{'편향':>9}{'경험SD':>9}{'평균SE':>9}"
            f"{'기각률':>8}{'±MC':>7}{'MDE':>9}{'단위':>6}{'노출률':>8}")
        for _, r in tab.iterrows():
            log(f"{r['설계']:<24}{r['평균 추정치']:>9.4f}{r['편향']:>9.4f}"
                f"{r['경험 SD']:>9.4f}{r['평균 SE']:>9.4f}{r['기각률']:>8.3f}"
                f"{r['MC 오차']:>7.3f}{r['MDE']:>9.4f}{r['분석 단위 수']:>6.1f}"
                f"{r['대조군 노출률']:>8.3f}")
        log()

    A_tab, B_tab, C_tab = (tables[w[0]] for w in worlds)

    def val(tab, name, col):
        return float(tab.loc[tab["설계"] == name, col].iloc[0])

    # ---- 대조군 1: 파급이 편향의 원인인가 -------------------------------
    log("=" * 78)
    log("대조군 1 — 파급을 제거하면 편향이 사라지는가 (세계 A vs 세계 B)")
    log("=" * 78)
    log(f"{'설계':<24}{'A 편향':>10}{'B 편향':>10}{'차이':>10}  판정")
    pass1 = True
    for name in DESIGNS[:3] + [DESIGNS[4]]:   # 설계 4는 정의상 오염 대조를 버려 별도 취급
        ba, bb = val(A_tab, name, "편향"), val(B_tab, name, "편향")
        ok = abs(bb) < abs(ba) / 3 if abs(ba) > 1e-4 else abs(bb) < 1e-3
        pass1 &= ok
        log(f"{name:<24}{ba:>10.4f}{bb:>10.4f}{ba - bb:>10.4f}  "
            f"{'통과 — 편향의 원인은 파급' if ok else '주의 — 파급 외 원인 존재'}")
    log(f"판정: {'통과' if pass1 else '주의'}")
    log()
    log("[함께 볼 것] 설계별 대조군 노출률 — 어떤 설계가 오염을 줄이는가")
    for name in DESIGNS:
        log(f"  {name:<24} {val(A_tab, name, '대조군 노출률'):.3f}")
    log("  짝지음(설계 3)의 노출률이 단순 무작위보다 **높다.** 닮은 시장이 곧 이웃이라")
    log("  쌍마다 처치·대조가 붙어 앉기 때문이다. 정밀도를 높인 설계가 오염을 더 만든다는")
    log("  이 역설이 설계 4·5가 필요한 이유이며, 동시에 둘 다 대가를 치르는 이유다.")
    log()

    # ---- 대조군 2: 절차 자체의 편향 -------------------------------------
    log("=" * 78)
    log("대조군 2 — 효과·파급이 없는 세계에서 1종 오류율이 명목 5%인가")
    log("=" * 78)
    log("초과(부풀림)와 미달(보수적)을 구분해 읽는다. 부풀림은 없는 효과를 있다고 말하게")
    log("만들므로 위험하고, 보수적 검정은 있는 효과를 놓치게 만들어 검정력을 깎는다.")
    inflated = []
    for _, r in C_tab.iterrows():
        lo, hi = r["기각률"] - 2 * r["MC 오차"], r["기각률"] + 2 * r["MC 오차"]
        if lo > ALPHA:
            verdict = "**부풀림 — 없는 효과를 유의하다고 말한다**"
            inflated.append(r["설계"])
        elif hi < ALPHA:
            verdict = "보수적 — 기각을 덜 한다(검정력 손실)"
        else:
            verdict = "명목 수준 포함"
        log(f"  {r['설계']:<24} {r['기각률']:.3f} ± {2 * r['MC 오차']:.3f}  "
            f"가짜 효과 {r['평균 추정치']:+.4f}  → {verdict}")
    log(f"판정: 부풀림 설계 = {inflated if inflated else '없음'}")
    log()

    # ---- 설계 4 진단 ----------------------------------------------------
    log("=" * 78)
    log("설계 4 진단 — 도넛을 사후 필터로 구현하면 무슨 일이 일어나는가")
    log("=" * 78)
    log(f"평균 분석 단위 = {val(A_tab, DESIGNS[3], '분석 단위 수'):.1f}개 "
        f"(설계 3의 {val(A_tab, DESIGNS[2], '분석 단위 수'):.0f}개에서 감소)")
    log(f"유효 반복 = {int(val(A_tab, DESIGNS[3], '유효 반복'))}/{N_REPS}회 "
        "— 쌍이 2개 미만 남으면 검정 자체가 성립하지 않는다")
    log()
    log("① 점추정은 목적을 달성했다.")
    log(f"   세계 A 편향 {val(A_tab, DESIGNS[3], '편향'):+.4f} "
        f"(설계 3의 {val(A_tab, DESIGNS[2], '편향'):+.4f}에서 거의 사라졌다)")
    log("   오염된 대조를 뺀 것 자체는 파급 편향을 실제로 걷어 냈다.")
    log()
    log("② 그런데 추론이 깨진다.")
    log(f"   경험 SD {val(A_tab, DESIGNS[3], '경험 SD'):.4f} ≫ "
        f"보고된 평균 SE {val(A_tab, DESIGNS[3], '평균 SE'):.4f}")
    log(f"   세계 C 1종 오류율 {val(C_tab, DESIGNS[3], '기각률'):.3f} "
        f"(명목 {ALPHA}) — 네 배 이상 부풀었다")
    log("   표준오차가 실제 산포를 못 따라가므로 유의성 판정을 믿을 수 없다.")
    log()
    log("③ 원인: 표본이 배정 결과의 함수다.")
    log("   대조 시장이 오염되었는지는 그 반복에서 이웃이 처치를 받았는지에 달려 있다.")
    log("   즉 몇 쌍이 남는지가 반복마다 달라지는 확률변수이고, 배정을 보고 표본을 골랐으므로")
    log("   t분포의 자유도 가정이 성립하지 않는다. 같은 데이터로 설계와 추정을 함께 한 셈이다.")
    log()
    log("④ 처방 둘.")
    log("   (a) 도넛을 **설계**로 구현한다 — 처치 비중을 낮춰 순수 대조 시장을 미리 확보하고,")
    log("       배제 규칙을 배정 전에 확정한다.")
    log("   (b) 추론을 **순열 검정**으로 바꾼다 — 같은 필터를 귀무가설 아래 재배정에 적용해")
    log("       기준분포를 만든다. t분포를 빌려 쓰지 않는다.")
    log("   처치 비중이 50%이고 격자가 촘촘하면 사후 필터는 애초에 성립하지 않는다.")
    log()

    # ---- MDE 곡선 --------------------------------------------------------
    log("=" * 78)
    log("시장 수와 MDE — 몇 개 시장이 필요한가 (설계 3 기준)")
    log("=" * 78)
    curve = mde_curve(Y, block_id, A, effect, spill, N_REPS // 2, SEED + 7)
    log(f"{'시장 수':>8}{'파급':>12}{'MDE(로그)':>12}{'MDE(%)':>10}{'검정력':>10}")
    for _, r in curve.iterrows():
        log(f"{int(r['시장 수']):>8}{r['파급']:>12}{r['MDE']:>12.4f}"
            f"{np.expm1(r['MDE']) * 100:>10.2f}{r['검정력']:>10.3f}")
    log()
    log(f"참 효과 {np.expm1(effect) * 100:.2f}%와 비교해 읽는다. MDE가 참 효과보다 크면 그")
    log("규모로는 이 효과를 잡을 수 없다 — 실험을 설계 단계에서 접어야 한다.")
    log("(MDE는 반복 표본의 평균 SE로 계산한 값이라 자체 오차가 있다)")
    log()

    log("파급이 MDE는 거의 움직이지 않는데 기각률은 올린다는 점을 눈여겨본다. 파급은")
    log("표준오차가 아니라 추정치를 밀어 올리므로 **검정력을 가짜로 높인다** — 기각이 늘어난")
    log("것이 효과를 잘 잡았다는 뜻이 아니다.")
    log()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    sub = curve[curve["파급"] == "파급 없음"].sort_values("시장 수")
    ax.plot(sub["시장 수"], np.expm1(sub["MDE"]) * 100, marker="o", color="#1f6f8b")
    ax.axhline(np.expm1(effect) * 100, color="crimson", ls="--", lw=1.2,
               label=f"참 효과 {np.expm1(effect) * 100:.1f}%")
    ax.set_xlabel("무작위화에 쓰는 시장 수")
    ax.set_ylabel("최소 탐지 가능 효과 MDE (%)")
    ax.set_title("(a) 잴 수 있는 가장 작은 변화")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, mk, c in [("파급 없음", "o", "#1f6f8b"), ("파급 있음", "s", "#c0503d")]:
        sub = curve[curve["파급"] == label].sort_values("시장 수")
        ax.plot(sub["시장 수"], sub["검정력"], marker=mk, color=c, label=label)
    ax.axhline(POWER_TARGET, color="gray", ls=":", lw=1.0, label="검정력 0.80")
    ax.set_xlabel("무작위화에 쓰는 시장 수")
    ax.set_ylabel("기각률")
    ax.set_title("(b) 파급은 기각률을 가짜로 올린다")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "15-1-mde-curve.png", dpi=140)
    plt.close(fig)
    log("그림 저장: 15-1-mde-curve.png")

    for wname, tab in tables.items():
        tag = wname.split()[0]
        tab.to_csv(RESULTS_DIR / f"15-1-designs-{tag}.csv", index=False,
                   encoding="utf-8-sig")
    curve.to_csv(RESULTS_DIR / "15-1-mde-curve.csv", index=False, encoding="utf-8-sig")
    log("표 저장: 15-1-designs-세계A/B/C.csv, 15-1-mde-curve.csv")

    (RESULTS_DIR / "15-1-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
