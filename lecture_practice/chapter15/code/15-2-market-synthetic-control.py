"""
15장 실습 2: 실험이 불가능할 때 — 시장 단위 합성통제로 출점 효과 추정
================================================================================
결정할 문제
--------------------------------------------------------------------------------
출점은 무작위 배정이 안 된다. 되돌릴 수도 없다. 게다가 **어디에 열지는 기대 성과를 보고
정한다** — 성장하는 시장에 낸다. 실험 대신 준실험으로 가야 한다.

시장 하나에 출점했다. 그 시장의 매출이 올랐다. 얼마가 출점의 효과인가?
세 가지가 답을 방해한다.
  (1) 내생적 처치 — 성장하던 시장을 골랐으므로 사후 상승분에 원래 추세가 섞여 있다
  (2) 공간 파급   — 인접 시장이 손님을 잃었으므로 대조군이 아래로 눌려 있다
  (3) 단일 처치   — 처치 단위가 하나라 평균을 낼 수 없다. 반사실을 만들어야 한다

3층 모델에서의 위치:
  [1층 GIS]  시장 격자·인접 관계, 시장별 매출 시계열 집계
  [2층 AI ]  **없다.** 이 문제의 핵심은 예측이 아니라 반사실 구성이다. 볼록조합
             최적화로 여러 대조 시장을 가중결합해 합성 대조군을 만든다.
  [3층 결정] (실제 − 합성) 격차 = 출점의 효과. 이 값이 15-3의 NPV 계산에 들어간다.

방법 홈: SCM의 정식 설명은 8.2·8.4절. 공공 실증은 13장 분석 3. 이 절은 **비즈니스
조건에서 무엇이 달라지는가**만 다룬다.

이 예제의 핵심 긴장 — 좋은 기증 단위가 곧 오염된 시장이다
--------------------------------------------------------------------------------
15-0의 DGP는 시장의 성질을 공간적으로 매끄럽게 만들었다(공간 자기상관). 그래서 사전
궤적이 가장 닮은 시장이 바로 지리적 이웃이고, SCM이 높은 가중치를 주는 기증 단위가
바로 파급으로 오염된 시장이다. 오염을 피하려고 이웃을 빼면 좋은 합성이 어려워진다.
**교환이 설계에서 자동으로 발생한다.**

추정 계단 다섯
--------------------------------------------------------------------------------
  ① 사후 단순 비교          : 처치 시장 사후 평균 − 기증 풀 사후 평균
  ② DID (전체 기증 풀)      : 사전 대비 변화량의 차이
  ③ SCM (전체 기증 풀)      : 사전 궤적을 맞춘 합성 대조군. 오염이 남는다
  ④ SCM (인접 제외 기증 풀) : 도넛의 SCM 버전
  ⑤ placebo 순열추론        : 기증 단위를 차례로 가짜 처치로 두고 유의성 판정

대조군(null control)
--------------------------------------------------------------------------------
  세계 A (기준)      : 내생 선택 + 효과 + 파급
  세계 B (파급 0)    : → ③과 ④의 격차가 사라져야 한다
  세계 C (내생성 0)  : 진출 시장을 무작위 선택 → ①·②의 편향이 줄어야 한다.
                       안 줄면 격차의 원인이 내생성이 아니다

실행:
    python 15-0-simdata-prep.py       # 최초 1회
    python 15-2-market-synthetic-control.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def load_data():
    for p in ["geoexp_panel.parquet", "geoexp_markets.parquet",
              "geoexp_adjacency.npy", "geoexp_truth.json"]:
        if not (DATA_DIR / p).exists():
            raise SystemExit(f"데이터가 없습니다: {DATA_DIR / p}\n"
                             "먼저 실행: python 15-0-simdata-prep.py")
    panel = pd.read_parquet(DATA_DIR / "geoexp_panel.parquet")
    Y = panel.drop(columns=["market_id"]).to_numpy()
    markets = pd.read_parquet(DATA_DIR / "geoexp_markets.parquet")
    A = np.load(DATA_DIR / "geoexp_adjacency.npy").astype(int)
    truth = json.loads((DATA_DIR / "geoexp_truth.json").read_text(encoding="utf-8"))
    return Y, markets, A, truth


# ===========================================================
# 1. 처치 시장 선택 — 내생적 선택을 그대로 구현한다
# ===========================================================
def choose_treated(markets, endogenous, rng):
    """진출 시장을 고른다.

    endogenous=True : 진출 가능 후보(시장 규모 상위 절반) 중 **사전 관측 성장률 1위**.
        이것이 현실의 선택 규칙이다 — 규모가 받쳐 주고 성장하는 시장에 낸다. 성장률
        전체 1위를 고르지 않는 이유가 하나 더 있다: 전체 1위를 처치로 빼면 볼록조합이
        그 궤적을 재현할 수 없어(볼록껍질 밖) SCM이 원리적으로 실패한다. 규모 조건을
        두면 최상위 성장 시장 일부가 기증 풀에 남는다.
    endogenous=False: 같은 후보군에서 무작위 선택(대조군 세계 C).
    """
    level_med = markets["level"].median()
    cand = markets.index[markets["level"] > level_med].to_numpy()
    if endogenous:
        sub = markets.loc[cand]
        return int(sub["pre_growth_observed"].idxmax()), cand
    return int(rng.choice(cand)), cand


def apply_treatment(Y, A, treat, n_pre, effect, spill):
    """처치 시장의 사후 기간에 효과를, 인접 시장의 사후 기간에 파급을 얹는다."""
    Yt = Y.copy()
    Yt[treat, n_pre:] += effect
    neigh = np.where(A[treat] > 0)[0]
    Yt[np.ix_(neigh, np.arange(n_pre, Y.shape[1]))] += spill
    return Yt, neigh


# ===========================================================
# 2. SCM — 볼록조합 최적화 (Σw=1, w≥0)
# ===========================================================
def fit_weights(x_treat, X_donors):
    """minimize ‖x_treat − X_donorsᵀ w‖²  s.t.  Σw = 1, w ≥ 0  (SLSQP).

    교육용 축약: 예측자 가중행렬 V를 항등으로 둔다(표준화 예측자에 동일 가중).
    정식 SCM은 V를 외부 루프에서 최적화한다(중첩 최적화; 실무는 Synth/pysyncon).
    """
    n = X_donors.shape[0]

    def loss(w):
        r = x_treat - X_donors.T @ w
        return float(r @ r)

    def grad(w):
        r = x_treat - X_donors.T @ w
        return -2.0 * (X_donors @ r)

    w0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
             "jac": lambda w: np.ones_like(w)},)
    res = minimize(loss, w0, jac=grad, bounds=[(0.0, 1.0)] * n, constraints=cons,
                   method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def build_predictors(Y, n_pre):
    """예측자 = 사전기간 주별 결과 전체. 시점별로 단위 간 표준화.

    **예측자 선택을 짐작으로 하지 않고 재서 정했다.** 주별 원자료를 쓰면 최적화가 주별
    잡음(SD 0.035)을 쫓느라 추세 정합이 밀릴 것이라고 예상했고, 그래서 분기 평균 4개와
    사전 기울기를 예측자로 쓰는 안을 먼저 구현했다. 그런데 후보 20개 시장을 차례로 처치로
    두고 네 가지 예측자 조합을 비교하자 예상이 뒤집혔다.

      예측자                 전수 평균 절대편향(전체 풀)   사전 RMSPE 평균
      주별 52개                    0.0250                  0.0454
      분기 평균 4개                0.0322                  0.0496
      분기 평균 4개 + 기울기       0.0293                  0.0501
      월별 13개 + 기울기           0.0250                  0.0471

    주별 원자료가 가장 정확했다. 예측자를 줄이면 사전 적합도가 나빠지고(0.045 → 0.050)
    가중치가 소수 단위에 몰려(집중도 0.17 → 0.58) 오히려 편향이 커졌다. 잡음을 쫓는 손해가
    자유도를 잃는 손해보다 작았던 것이다. **직관으로 정하고 넘어갈 문제가 아니었다.**
    """
    pre = Y[:, :n_pre]
    mean, sd = pre.mean(0), pre.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (pre - mean) / sd


def scm(Y, treat, donors, n_pre):
    """treat를 donors의 볼록조합으로 합성한다.

    사전 궤적을 맞추는 것이 SCM의 신뢰 근거이므로 사전 적합도(pre_rmspe)를 항상 함께
    본다. 다만 아래 build_predictors의 주석이 말하듯, **사전 적합도가 좋다고 추세가
    맞았다고 볼 수는 없다.** 사후 격차의 기울기를 별도로 확인한다(post_gap_slope).
    """
    P = build_predictors(Y, n_pre)
    w = fit_weights(P[treat], P[donors])
    synth = w @ Y[donors]
    gap = Y[treat] - synth
    pre_r = float(np.sqrt(np.mean(gap[:n_pre] ** 2)))
    post_r = float(np.sqrt(np.mean(gap[n_pre:] ** 2)))
    # 사후 격차의 기울기 — 0에서 멀면 추세가 안 맞았다는 뜻이고, 효과 추정이 기간에
    # 따라 달라진다(참 효과가 일정하다는 설계에서는 경보 신호다).
    tp = np.arange(len(gap) - n_pre, dtype=float)
    tp -= tp.mean()
    gap_slope = float((gap[n_pre:] @ tp) / (tp @ tp))
    return {
        "weights": w, "donors": donors, "synth": synth, "gap": gap,
        "pre_rmspe": pre_r, "post_rmspe": post_r,
        "ratio": post_r / pre_r if pre_r > 0 else np.inf,
        "att": float(gap[n_pre:].mean()),
        "post_gap_slope": gap_slope,
    }


def herfindahl(w):
    """가중치 집중도. 1에 가까우면 한 단위에 몰렸고, 작으면 넓게 퍼졌다."""
    return float(np.sum(w ** 2))


# ===========================================================
# 3. placebo 순열추론
# ===========================================================
def placebo_inference(Y, treat, donors, n_pre, actual_ratio):
    """기증 단위를 차례로 가짜 처치로 두고 사후/사전 RMSPE 비율 분포를 만든다.

    참 처치가 없는 단위에서는 사후 격차가 사전 격차만큼만 커야 하므로 비율이 1 근처다.
    처치 단위의 비율이 분포의 최상위면 "우연히 이런 격차가 나올 확률이 낮다"고 말할 수 있다.
    유사 p값 = (비율이 처치 단위 이상인 단위 수) / (전체 단위 수).
    """
    ratios = []
    for d in donors:
        pool = np.array([u for u in donors if u != d])
        try:
            r = scm(Y, d, pool, n_pre)["ratio"]
        except Exception:
            continue
        ratios.append(r)
    ratios = np.array(ratios)
    p = float((np.sum(ratios >= actual_ratio) + 1) / (len(ratios) + 1))
    return ratios, p


# ===========================================================
# 4. 추정 계단
# ===========================================================
def ladder(Y, A, treat, n_pre, all_units):
    """다섯 계단 중 ①~④를 계산한다. 반환: dict."""
    post = slice(n_pre, Y.shape[1])
    donors_full = np.array([u for u in all_units if u != treat])
    neigh = np.where(A[treat] > 0)[0]
    donors_clean = np.array([u for u in donors_full if u not in set(neigh)])

    t_pre_m = Y[treat, :n_pre].mean()
    t_post_m = Y[treat, post].mean()

    out = {}
    out["① 사후 단순 비교"] = {
        "att": float(t_post_m - Y[donors_full][:, post].mean()),
        "pre_rmspe": np.nan, "n_donors": len(donors_full), "hhi": np.nan,
        "w_adjacent": np.nan,
    }
    d_treat = t_post_m - t_pre_m
    d_donor = (Y[donors_full][:, post].mean(axis=1)
               - Y[donors_full][:, :n_pre].mean(axis=1)).mean()
    out["② DID (전체 기증 풀)"] = {
        "att": float(d_treat - d_donor),
        "pre_rmspe": np.nan, "n_donors": len(donors_full), "hhi": np.nan,
        "w_adjacent": np.nan,
    }
    for label, pool in [("③ SCM (전체 기증 풀)", donors_full),
                        ("④ SCM (인접 제외 풀)", donors_clean)]:
        r = scm(Y, treat, pool, n_pre)
        w_adj = float(sum(w for u, w in zip(pool, r["weights"]) if u in set(neigh)))
        out[label] = {
            "att": r["att"], "pre_rmspe": r["pre_rmspe"], "n_donors": len(pool),
            "hhi": herfindahl(r["weights"]), "w_adjacent": w_adj,
            "gap_slope": r["post_gap_slope"], "_scm": r,
        }
    for k in ["① 사후 단순 비교", "② DID (전체 기증 풀)"]:
        out[k]["gap_slope"] = np.nan
    return out, donors_full, donors_clean, neigh


LADDER_KEYS = ["① 사후 단순 비교", "② DID (전체 기증 풀)",
               "③ SCM (전체 기증 풀)", "④ SCM (인접 제외 풀)"]


def sweep_random_selection(Y0, A, cand, n_pre, all_units, effect, spill):
    """대조군 세계 C — 진출 시장을 후보 전수로 돌려 계단별 평균 편향을 재다.

    왜 한 번 무작위로 고르는 것으로 부족한가: 처치 단위가 **하나**라 어느 시장을 골랐는지가
    추정치를 크게 흔든다. 무작위 선택 한 건의 편향은 선택 규칙의 성질이 아니라 그 한 번의
    뽑기 결과다. 후보 전체를 차례로 처치로 두고 평균을 봐야 "선택 규칙이 편향의 원인인가"에
    답할 수 있다. 이것 자체가 단일 처치 설계의 한계를 보여 준다.
    """
    recs = []
    for u in cand:
        Yt, _ = apply_treatment(Y0, A, int(u), n_pre, effect, spill)
        tab, _, _, _ = ladder(Yt, A, int(u), n_pre, all_units)
        rec = {k: tab[k]["att"] - effect for k in LADDER_KEYS}
        rec["시장"] = int(u)
        rec["사전RMSPE 전체풀"] = tab["③ SCM (전체 기증 풀)"]["pre_rmspe"]
        rec["사전RMSPE 인접제외"] = tab["④ SCM (인접 제외 풀)"]["pre_rmspe"]
        rec["인접 가중 합"] = tab["③ SCM (전체 기증 풀)"]["w_adjacent"]
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    Y0, markets, A, truth = load_data()
    n_pre = truth["n_weeks_pre"]
    effect = truth["true_effect_log"]
    spill = truth["spillover_log"]
    all_units = np.arange(Y0.shape[0])
    rng = np.random.default_rng(SEED)

    log("=" * 78)
    log("15-2 실험이 불가능할 때 — 시장 단위 합성통제로 출점 효과 추정")
    log("=" * 78)
    log(f"시장 {Y0.shape[0]}개 · 주 {Y0.shape[1]}개(사전 {n_pre}) · "
        f"참 효과 로그 {effect:+.4f} ({truth['true_effect_pct']:+.2f}%)")
    log(f"파급 로그 {spill:+.4f} ({truth['spillover_pct']:+.2f}%) ← 처치 시장에 인접한 기증 시장")
    log()

    worlds = [
        ("세계A 기준(내생선택+효과+파급)", effect, spill),
        ("세계B 대조군(파급 0)", effect, 0.0),
    ]

    results = {}
    for wname, eff, sp in worlds:
        treat, cand = choose_treated(markets, True, np.random.default_rng(SEED + 1))
        Yt, neigh = apply_treatment(Y0, A, treat, n_pre, eff, sp)
        tab, d_full, d_clean, neigh = ladder(Yt, A, treat, n_pre, all_units)
        results[wname] = {"treat": treat, "tab": tab, "Yt": Yt,
                          "d_full": d_full, "d_clean": d_clean, "neigh": neigh,
                          "cand": cand}

        log("-" * 78)
        log(wname)
        log("-" * 78)
        g = markets.loc[treat]
        rank = int((markets["pre_growth_observed"] > g["pre_growth_observed"]).sum()) + 1
        log(f"처치 시장 = {treat}번 (격자 {int(g['grid_row'])},{int(g['grid_col'])}) · "
            f"진출 후보 {len(cand)}개 중 선택 · 인접 시장 {[int(x) for x in neigh]}")
        log(f"사전 관측 성장률 = {g['pre_growth_observed']:+.5f} (전체 {rank}위)")
        log()
        log(f"{'추정 계단':<22}{'추정 ATT':>10}{'참값 대비':>10}{'사전RMSPE':>11}"
            f"{'격차기울기':>11}{'기증수':>7}{'가중집중':>9}{'인접가중':>9}")
        for k, v in tab.items():
            pr = f"{v['pre_rmspe']:.4f}" if np.isfinite(v["pre_rmspe"]) else "—"
            gs = f"{v['gap_slope']:+.5f}" if np.isfinite(v["gap_slope"]) else "—"
            hh = f"{v['hhi']:.3f}" if np.isfinite(v["hhi"]) else "—"
            wa = f"{v['w_adjacent']:.3f}" if np.isfinite(v["w_adjacent"]) else "—"
            log(f"{k:<22}{v['att']:>10.4f}{v['att'] - eff:>10.4f}{pr:>11}{gs:>11}"
                f"{v['n_donors']:>7}{hh:>9}{wa:>9}")
        log()

    # ---- 세계 C: 선택 규칙을 벗겨 가며 편향을 분해한다 ---------------------
    cand = results["세계A 기준(내생선택+효과+파급)"]["cand"]
    sweep_all = sweep_random_selection(Y0, A, all_units, n_pre, all_units, effect, spill)
    sweep = sweep_random_selection(Y0, A, cand, n_pre, all_units, effect, spill)

    log("-" * 78)
    log("세계C 대조군 — 선택 규칙을 벗겨 가며 편향을 분해한다")
    log("-" * 78)
    log("처치 단위가 하나이므로 '무작위 선택 한 번'의 편향은 규칙의 성질이 아니라 그 뽑기의")
    log("결과다. 그래서 처치가 될 수 있는 시장 전체를 차례로 돌려 평균 편향을 본다.")
    log()
    log("  C1 = 전체 40개 시장에서 무작위 선택 (선택 규칙을 완전히 제거)")
    log("  C2 = 진출 후보(시장 규모 상위 절반) 20개에서 무작위 선택 (규모 필터만 남김)")
    log("  A  = 후보 중 사전 성장률 1위 (실제 의사결정 규칙, 단일 사례)")
    log()
    log(f"{'추정 계단':<22}{'C1 평균':>10}{'C2 평균':>10}{'A 편향':>10}"
        f"{'C2 SD':>9}{'C2 절댓값':>11}")
    for k in LADDER_KEYS:
        bA = results["세계A 기준(내생선택+효과+파급)"]["tab"][k]["att"] - effect
        log(f"{k:<22}{sweep_all[k].mean():>10.4f}{sweep[k].mean():>10.4f}{bA:>10.4f}"
            f"{sweep[k].std(ddof=1):>9.4f}{sweep[k].abs().mean():>11.4f}")
    log()

    # ---- 세계 A 상세: placebo 추론 ---------------------------------------
    wA = results["세계A 기준(내생선택+효과+파급)"]
    log("=" * 78)
    log("⑤ placebo 순열추론 — 이 격차가 우연일 수 있는가 (세계 A, 인접 제외 풀)")
    log("=" * 78)
    scm4 = wA["tab"]["④ SCM (인접 제외 풀)"]["_scm"]
    ratios, pval = placebo_inference(wA["Yt"], wA["treat"], wA["d_clean"], n_pre,
                                     scm4["ratio"])
    log(f"처치 시장 사후/사전 RMSPE 비율 = {scm4['ratio']:.3f}")
    log(f"기증 {len(ratios)}개의 가짜 처치 비율: 중앙값 {np.median(ratios):.3f} · "
        f"최대 {ratios.max():.3f}")
    log(f"유사 p값 = {pval:.3f}  "
        f"({'유의' if pval < 0.05 else '유의하지 않음'}, 명목 0.05 기준)")
    log()

    # ---- 대조군 판정 -----------------------------------------------------
    log("=" * 78)
    log("대조군 판정 — 심은 메커니즘이 정말 원인인가")
    log("=" * 78)

    def att(w, k):
        return results[w]["tab"][k]["att"]

    log("[대조군 1] 파급이 오염시킨 양이 계산과 맞는가")
    a3 = att("세계A 기준(내생선택+효과+파급)", "③ SCM (전체 기증 풀)")
    b3 = att("세계B 대조군(파급 0)", "③ SCM (전체 기증 풀)")
    a4 = att("세계A 기준(내생선택+효과+파급)", "④ SCM (인접 제외 풀)")
    b4 = att("세계B 대조군(파급 0)", "④ SCM (인접 제외 풀)")
    w_adj = results["세계A 기준(내생선택+효과+파급)"]["tab"]["③ SCM (전체 기증 풀)"]["w_adjacent"]
    predicted = -w_adj * spill      # 합성 대조군이 눌린 만큼 격차가 올라간다
    log(f"  ③ 전체 풀 : 세계 A {a3:+.4f} − 세계 B {b3:+.4f} = {a3 - b3:+.4f}")
    log(f"  예측값     : 인접 가중 합 {w_adj:.3f} × 파급 {-spill:.4f} = {predicted:+.4f}")
    v1 = abs((a3 - b3) - predicted) < 1e-3
    log(f"  판정: {'통과' if v1 else '주의'} — 오염량이 "
        f"{'가중치×파급으로 정확히 설명된다' if v1 else '설명되지 않는다'}")
    log(f"  ④ 인접 제외: 세계 A {a4:+.4f} = 세계 B {b4:+.4f}  "
        f"(차이 {a4 - b4:+.4f} — 정의상 0이어야 한다)")
    log("  → 오염의 크기는 **인접 시장에 준 가중치**가 정한다. 가중치가 0이면 파급이 있어도")
    log("    추정에 들어오지 않는다. 즉 문제는 파급의 존재가 아니라 파급받은 시장을 얼마나")
    log("    쓰는가다. 이 진단은 실데이터에서도 계산할 수 있다 — 가중치는 관측된다.")
    log()
    log("[대조군 2] 선택 규칙을 완전히 제거하면 편향이 사라지는가 (C1 판정)")
    v2 = True
    for k in LADDER_KEYS[:2]:
        b1 = float(sweep_all[k].mean())
        v2 &= abs(b1) < 0.01
        log(f"  {k:<22} C1 평균 편향 {b1:+.4f}  "
            f"→ {'0에 수렴 — 편향의 원인은 선택이다' if abs(b1) < 0.01 else '남는다 — 선택 외 원인 존재'}")
    log(f"  판정: {'통과' if v2 else '주의'}")
    log()
    log("  선택 편향은 한 겹이 아니다 — C1 → C2 → A로 벗겨 보면 층이 드러난다.")
    for k in LADDER_KEYS[:2]:
        b1, b2 = float(sweep_all[k].mean()), float(sweep[k].mean())
        bA = att("세계A 기준(내생선택+효과+파급)", k) - effect
        log(f"    {k:<22} {b1:+.4f} → {b2:+.4f} → {bA:+.4f}")
    log("    ① 사후 단순 비교의 편향은 **규모 필터**만으로 이미 커진다. 규모가 큰 시장을")
    log("      후보로 삼는 것 자체가 수준 차이를 만들기 때문이다(C2 단계에서 발생).")
    log("    ② DID는 수준 차이를 차분으로 지우므로 규모 필터에 덜 흔들린다. 대신 **성장률")
    log("      선택**에 흔들린다 — A 단계에서 부호가 바뀌며 부풀려진다.")
    log("    즉 '어느 시장에 낼 수 있는가'(규모)와 '그중 어디에 낼까'(성장)가 서로 다른")
    log("    추정량을 서로 다른 방향으로 오염시킨다. 내생성을 한 단어로 다루면 놓친다.")
    log()
    log("  함께 읽을 것: 무작위로 골라도 편향의 **산포**가 크다.")
    for k in LADDER_KEYS:
        log(f"    {k:<22} C2 편향 SD {sweep[k].std(ddof=1):.4f} · "
            f"절댓값 평균 {sweep[k].abs().mean():.4f}")
    log("  처치 단위가 하나이므로 무작위 선택도 개별 추정의 편향을 없애 주지 않는다.")
    log("  무작위 배정의 이점은 표본이 커야 나타난다 — 단일 처치 설계의 근본 한계다.")
    log()

    # ---- 내생 선택이 왜 SCM을 어렵게 만드는가 -----------------------------
    log("=" * 78)
    log("내생 선택과 볼록껍질 — 왜 하필 이 시장에서 SCM이 실패하는가")
    log("=" * 78)
    treat = wA["treat"]
    for key in [LADDER_KEYS[2], LADDER_KEYS[3]]:
        b = wA["tab"][key]["att"] - effect
        pct = float((sweep[key].abs() < abs(b)).mean() * 100)
        log(f"  {key:<22} 이 시장 편향 {b:+.4f} · "
            f"후보 20개 중 {pct:.0f}%가 이보다 정확")
    pr = wA["tab"][LADDER_KEYS[2]]["pre_rmspe"]
    log(f"  사전 RMSPE {pr:.4f} · 후보 평균 {sweep['사전RMSPE 전체풀'].mean():.4f} "
        f"({'나쁘다' if pr > sweep['사전RMSPE 전체풀'].mean() else '좋다'})")
    log()
    # 볼록껍질 점검 — 처치 시장의 사전 기울기가 기증 단위 범위 안에 있는가
    t = np.arange(n_pre, dtype=float) - (n_pre - 1) / 2.0
    slope = (Y0[:, :n_pre] @ t) / (t @ t)
    d_full = wA["d_full"]
    log(f"  [볼록껍질 점검] 사전 추세 기울기")
    log(f"    처치 시장 {slope[treat]:+.6f} · 기증 풀 범위 "
        f"[{slope[d_full].min():+.6f}, {slope[d_full].max():+.6f}]")
    n_above = int((slope[d_full] > slope[treat]).sum())
    log(f"    처치 시장보다 기울기가 큰 기증 단위 = {n_above}개 / {len(d_full)}개")
    log(f"    → 볼록조합이 이 기울기를 재현할 여지는 {'있다' if n_above > 0 else '없다'}. "
        "그런데 여지가 있다는 것과")
    log("      실제로 맞춘다는 것은 다르다. 최적화는 사전 궤적 전체를 맞추지 추세만 맞추지 않는다.")
    log()
    log("  진단: 내생 선택은 두 가지를 동시에 한다. 첫째, 성장하던 시장을 고르므로 단순")
    log("  비교와 DID를 부풀린다(대조군 2가 확인). 둘째, **성장률 상위 시장은 볼록껍질의")
    log("  경계 쪽에 있어 좋은 합성을 만들기 어렵다.** 둘은 같은 뿌리에서 나온 문제다.")
    log("  그래서 SCM은 방법으로서 옳게 작동하는데(후보 전수 평균 편향 "
        f"{sweep[LADDER_KEYS[2]].mean():+.4f})")
    log("  하필 기업이 고르는 시장에서 가장 부정확하다. **평균적으로 잘 듣는 방법이 내가")
    log("  쓰려는 그 사례에서 잘 들으리라는 보장은 없다.**")
    log()

    # ---- 기증 풀 축소의 대가 ---------------------------------------------
    log("=" * 78)
    log("기증 풀을 줄이면 무엇을 내놓는가 — 15.4의 교환을 수치로")
    log("=" * 78)
    t3 = wA["tab"]["③ SCM (전체 기증 풀)"]
    t4 = wA["tab"]["④ SCM (인접 제외 풀)"]
    log(f"기증 풀      : {t3['n_donors']}개 → {t4['n_donors']}개 "
        f"(인접 {len(wA['neigh'])}개 제외)")
    log(f"사전 RMSPE   : {t3['pre_rmspe']:.4f} → {t4['pre_rmspe']:.4f} "
        f"({(t4['pre_rmspe'] / t3['pre_rmspe'] - 1) * 100:+.1f}%)")
    log(f"가중 집중도  : {t3['hhi']:.3f} → {t4['hhi']:.3f} "
        "(1에 가까우면 소수 단위에 몰렸다)")
    log(f"인접 시장 가중 합: {t3['w_adjacent']:.3f} → {t4['w_adjacent']:.3f}")
    log(f"추정 ATT     : {t3['att']:+.4f} → {t4['att']:+.4f} (참값 {effect:+.4f})")
    log()
    log("읽는 법: 인접 시장이 전체 풀에서 가중치를 크게 받는다면, 그것은 공간 자기상관")
    log("때문에 이웃이 사전 궤적을 가장 잘 맞추기 때문이다. 즉 **가장 좋은 기증 단위가")
    log("가장 오염된 단위**다. 빼면 사전 적합이 나빠지고, 안 빼면 효과가 부풀려진다.")
    log()
    log("[도넛이 평균적으로 이득인가] 후보 20개 전수 반복 결과")
    log(f"  전체 풀   평균 절대편향 {sweep[LADDER_KEYS[2]].abs().mean():.4f} · "
        f"사전 RMSPE 평균 {sweep['사전RMSPE 전체풀'].mean():.4f}")
    log(f"  인접 제외 평균 절대편향 {sweep[LADDER_KEYS[3]].abs().mean():.4f} · "
        f"사전 RMSPE 평균 {sweep['사전RMSPE 인접제외'].mean():.4f}")
    log(f"  인접 가중 합 평균 {sweep['인접 가중 합'].mean():.3f}  "
        f"→ 오염 기대량 {-sweep['인접 가중 합'].mean() * spill:+.4f}")
    better = sweep[LADDER_KEYS[3]].abs().mean() < sweep[LADDER_KEYS[2]].abs().mean()
    log(f"  판정: 인접 제외가 평균적으로 {'이득' if better else '**손해**'}다.")
    log("  파급으로 들어오는 오염량은 작은데(가중치×파급) 기증 풀을 줄여 잃는 적합도는 크다.")
    log("  **도넛은 무조건 옳은 처방이 아니다.** 파급의 크기와 기증 풀의 여유가 정하는")
    log("  교환이며, 둘을 재지 않고 관행으로 인접을 빼면 편향을 줄이는 대신 키울 수 있다.")
    log("  실무 판단: 인접 가중 합이 크고 파급이 클 때만 제외가 남는다. 두 값을 먼저 잰다.")
    log()

    # ---- 그림 15.3 ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    t = np.arange(Y0.shape[1])
    ax = axes[0]
    ax.plot(t, wA["Yt"][wA["treat"]], lw=1.6, color="#1a1a1a", label="실제(처치 시장)")
    ax.plot(t, scm4["synth"], lw=1.4, ls="--", color="#1f6f8b", label="합성 대조군(인접 제외)")
    ax.plot(t, wA["tab"]["③ SCM (전체 기증 풀)"]["_scm"]["synth"], lw=1.0, ls=":",
            color="#c0503d", label="합성 대조군(전체 풀)")
    ax.axvline(n_pre, color="gray", lw=1.0)
    ax.set_xlabel("주")
    ax.set_ylabel("로그 매출")
    ax.set_title("(a) 실제 대 합성 궤적")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(ratios, bins=14, color="#b9c8d0", edgecolor="white")
    ax.axvline(scm4["ratio"], color="crimson", lw=1.8,
               label=f"처치 시장 {scm4['ratio']:.2f} (p={pval:.3f})")
    ax.set_xlabel("사후/사전 RMSPE 비율")
    ax.set_ylabel("가짜 처치 단위 수")
    ax.set_title("(b) placebo 분포")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "15-2-scm-placebo.png", dpi=140)
    plt.close(fig)
    log("그림 저장: 15-2-scm-placebo.png")

    # ---- 산출물 저장 ------------------------------------------------------
    rows = []
    for wname, r in results.items():
        for k, v in r["tab"].items():
            rows.append({"세계": wname, "추정 계단": k, "ATT": v["att"],
                         "참값 대비": v["att"] - effect, "사전 RMSPE": v["pre_rmspe"],
                         "사후 격차 기울기": v["gap_slope"],
                         "기증 수": v["n_donors"], "가중 집중도": v["hhi"],
                         "인접 가중 합": v["w_adjacent"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "15-2-ladder.csv", index=False,
                              encoding="utf-8-sig")
    sweep.to_csv(RESULTS_DIR / "15-2-null-random-selection.csv", index=False,
                 encoding="utf-8-sig")

    # 15-3(의사결정)이 쓰는 추정 결과를 남긴다
    handoff = {
        "treated_market": int(wA["treat"]),
        "att_log_scm_clean": float(t4["att"]),
        "att_pct_scm_clean": float(np.expm1(t4["att"]) * 100),
        "att_log_scm_full": float(t3["att"]),
        "true_effect_log": float(effect),
        "placebo_p": float(pval),
        "placebo_ratio_treated": float(scm4["ratio"]),
        "post_gap_sd": float(np.std(scm4["gap"][n_pre:], ddof=1)),
        "pre_gap_sd": float(np.std(scm4["gap"][:n_pre], ddof=1)),
        "n_donors_clean": int(t4["n_donors"]),
    }
    (RESULTS_DIR / "15-2-estimate.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
    log("표 저장: 15-2-ladder.csv · 추정 결과 인계: 15-2-estimate.json")

    (RESULTS_DIR / "15-2-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
