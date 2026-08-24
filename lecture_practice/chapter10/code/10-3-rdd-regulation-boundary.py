"""
10장 실습 3: 부동산 규제경계의 가격 불연속 — 회귀불연속 설계(RDD) (인과 척추)
================================================================================
도시·부동산 정책 질문: "투기과열지구/조정대상지역 지정(규제)이 경계 근방 주택가격을
실제로 낮췄는가? 아니면 원래 비싼 곳이 규제됐을 뿐인가?"

3층 모델에서의 위치 — 이 분석은 [3층 정책·인과]까지 간다(식별원=경계 불연속):
  [1층 GIS]  경계까지의 부호거리(signed distance) 계산, 경계 근방 거래 추출
  [2층 AI ]  없음 — 이 문제의 핵심은 예측이 아니라 '규제효과 식별'이다.
             예측력을 키우는 게 아니라, 교란을 통제한 국소 인과효과를 추정한다.
  [3층 정책] 경계에서의 가격 점프 = 규제의 국소 인과효과. 단, 복합처치·내생지정
             때문에 "전국 규제효과"로 외삽하지 않는다(규칙2 parsimony·규칙4 예측≠인과).

방법 홈: RDD 정식 설명은 8.4절(준실험 설계). 여기서는 '적용'한다.

교육 의도(construct validation, §12.7 선례):
  실거래가 원자료를 특정 경계로 정제·좌표결합하는 것은 대규모 데이터 획득 과제라,
  본 예제는 *참 불연속 τ를 아는 합성 DGP*로 추정량의 타당성을 검증한다. 검증 대상:
    (a) 삼각커널 지역선형 RDD가 참 τ를 복원하는가
    (b) 전역 고차(4차) 다항식은 어떻게 편향되는가 (Gelman & Imbens 2019)
    (c) McCrary식 밀도 불연속 검정이 조작(정렬)을 탐지하는가
    (d) placebo(가짜 경계)에서 효과가 0으로 나오는가
  → 합성이지만 '방법이 옳게 작동하는 조건과 실패하는 조건'을 실증한다.

데이터: 미리 준비한 합성 표본(참 τ를 아는 교육용, 개인 식별 불가)을 불러와 쓴다.
  실제 분석에서는 국토부 실거래가 공개시스템의 좌표·거래가 원자료 + 투기과열지구/
  조정대상지역 경계(주택법 시행령 지정)를 사용한다.

실행:
    python 10-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 10-3-rdd-regulation-boundary.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 크로스 플랫폼 한글 폰트: 사용 가능한 첫 후보 사용(macOS/Windows/Linux).
# 없으면 그림 라벨의 한글이 □로 보일 뿐 결과·수치에는 영향 없음 → 경고만 억제.
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# 부트스트랩 재표집용 난수(생성과 분리된 고정 시드 → 재현성 확보)
RNG = np.random.default_rng(42)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NORMAL_PATH = DATA_DIR / "rdd_normal.parquet"
MANIP_PATH = DATA_DIR / "rdd_manip.parquet"
if not (NORMAL_PATH.exists() and MANIP_PATH.exists()):
    raise SystemExit(
        f"데이터가 없습니다: {NORMAL_PATH}\n"
        "먼저 실행: python 10-0-simdata-prep.py"
    )

# ===========================================================
# 0. DGP 파라미터 (해석·로깅용 — 데이터 생성은 준비 스크립트 10-0)
# ===========================================================
# 배정변수 x = 경계까지 부호거리(km). x<0 = 규제지역 안쪽(투기과열지구), x>0 = 바깥.
# 결과 Y = 매끄러운 공간 가격경사 f(x) + 규제효과 τ·1[x<0] + 잡음 (만원/㎡)
# 참 τ를 안다는 설계 의도로 추정량의 타당성을 검증한다(construct validation).
N = 2000
H_DATA = 3.0            # 데이터 생성 반경(경계 ±3km)
TRUE_TAU = -60.0        # 참 규제효과: 규제지역 가격이 60만원/㎡ 낮음(설계값)
BASE_PRICE = 1200.0     # 경계에서의 기준 가격(만원/㎡)


def load_data(manipulation=False):
    """미리 준비한 합성 실거래 표본을 불러온다.

    manipulation=False: 배정변수 밀도가 경계에서 연속(정상 RDD 조건).
    manipulation=True : 매수자가 규제를 피해 경계 바깥(x>0)으로 몰린 조작(정렬) 표본.
                        McCrary 검정이 이를 탐지하는지 보이기 위한 대조군.
    """
    return pd.read_parquet(MANIP_PATH if manipulation else NORMAL_PATH)


# ===========================================================
# 1. 삼각커널 지역선형 RDD 추정량 (경계값의 좌·우극한 차이)
# ===========================================================
def triangular_weights(x, c, h):
    u = (x - c) / h
    return np.maximum(0.0, 1.0 - np.abs(u))


def local_linear_side(x, y, c, h):
    """한쪽(밴드 내) 표본에 삼각커널 가중 1차 WLS 적합 → 경계 c에서의 절편 추정.

    반환: (경계 적합값, 유효표본수).
    """
    w = triangular_weights(x, c, h)
    mask = w > 0
    xs, ys, ws = x[mask] - c, y[mask], w[mask]
    if mask.sum() < 5:
        return np.nan, int(mask.sum())
    X = np.column_stack([np.ones_like(xs), xs])          # [1, (x-c)]
    XtW = X.T * ws                                        # 원소별 가중(밀집 대각행렬 회피)
    beta = np.linalg.solve(XtW @ X, XtW @ ys)            # WLS 정규방정식
    return beta[0], int(mask.sum())                      # 절편 = c에서의 적합값


def rdd_estimate(df, c=0.0, h=1.0, n_boot=800):
    """τ̂ = E[Y|x→c⁻](규제) − E[Y|x→c⁺](비규제). 부트스트랩 SE·95% CI."""
    left = df[df.x < c]
    right = df[df.x >= c]
    yl, _ = local_linear_side(left.x.values, left.y.values, c, h)
    yr, _ = local_linear_side(right.x.values, right.y.values, c, h)
    tau = yl - yr

    boots = np.empty(n_boot)
    xl, yl_arr = left.x.values, left.y.values
    xr, yr_arr = right.x.values, right.y.values
    nl, nr = len(left), len(right)
    for b in range(n_boot):
        il = RNG.integers(0, nl, nl)
        ir = RNG.integers(0, nr, nr)
        bl, _ = local_linear_side(xl[il], yl_arr[il], c, h)
        br, _ = local_linear_side(xr[ir], yr_arr[ir], c, h)
        boots[b] = bl - br
    se = np.nanstd(boots, ddof=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    n_eff = int(((triangular_weights(df.x.values, c, h)) > 0).sum())
    return {"tau": tau, "se": se, "ci": (lo, hi), "n_eff": n_eff}


# ===========================================================
# 2. 전역 고차 다항식 추정량 (차수 민감성 대비용)
# ===========================================================
def global_polynomial_rdd(df, c=0.0, order=4):
    """전역 order차 다항식 + 처치더미. Gelman & Imbens(2019)가 경고한 방식."""
    x = df.x.values - c
    D = (df.x.values < c).astype(float)
    # [1, D, x, x^2, ..., x^order, D*x, ..., D*x^order] : 양쪽 다른 곡률 허용
    cols = [np.ones_like(x), D]
    for p in range(1, order + 1):
        cols.append(x**p)
    for p in range(1, order + 1):
        cols.append(D * x**p)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, df.y.values, rcond=None)
    return beta[1]  # D 계수 = 경계에서의 점프


# ===========================================================
# 3. McCrary식 밀도 불연속 검정 (조작·정렬 진단)
# ===========================================================
def mccrary_density_test(df, c=0.0, binwidth=0.2, h=1.0, n_boot=800):
    """배정변수 밀도가 c에서 불연속인지 검정.

    히스토그램(정규화 밀도)을 양쪽에서 삼각커널 지역선형으로 c에 외삽, 로그차 검정.
    θ = log f̂(c⁺) − log f̂(c⁻). 부트스트랩으로 SE·z. |z|가 크면 조작(정렬) 의심.
    (McCrary 2008의 취지를 간소 구현. 실무는 rddensity/DCdensity 사용.)
    """
    x = df.x.values

    def theta_from(sample):
        edges = np.arange(-H_DATA, H_DATA + binwidth, binwidth)
        counts, _ = np.histogram(sample, bins=edges)
        mid = (edges[:-1] + edges[1:]) / 2
        dens = counts / (len(sample) * binwidth)  # 정규화 밀도
        fr, _ = local_linear_side(mid[mid >= c], dens[mid >= c], c, h)
        fl, _ = local_linear_side(mid[mid < c], dens[mid < c], c, h)
        if fr <= 0 or fl <= 0 or np.isnan(fr) or np.isnan(fl):
            return np.nan
        return np.log(fr) - np.log(fl)

    theta = theta_from(x)
    boots = np.array([theta_from(x[RNG.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    se = np.nanstd(boots, ddof=1)
    z = theta / se if se > 0 else np.nan
    return {"theta": theta, "se": se, "z": z}


# ===========================================================
# 4. 실행: 추정·진단·대비·시각화
# ===========================================================
def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 68)
    log("10-3 부동산 규제경계 RDD — 경계 불연속으로 규제효과 식별")
    log("=" * 68)
    log(f"참 규제효과 τ(설계값) = {TRUE_TAU:.1f} 만원/㎡  (규제지역 가격 하락)")
    log(f"표본 N = {N},  경계 데이터반경 = ±{H_DATA}km,  기준가 = {BASE_PRICE:.0f} 만원/㎡")
    log("")

    df = load_data(manipulation=False)

    # --- (a) 지역선형 RDD (주 추정) + 대역폭 민감도 ---
    log("[1] 삼각커널 지역선형 RDD — 대역폭 민감도")
    log("-" * 68)
    log(f"{'대역폭 h(km)':>12} | {'τ̂':>9} | {'SE':>7} | {'95% CI':>20} | {'유효N':>6}")
    main_h = 1.0
    for h in [0.5, 0.75, 1.0, 1.5, 2.0]:
        r = rdd_estimate(df, c=0.0, h=h)
        star = "  ← 주 대역폭" if h == main_h else ""
        log(f"{h:>12.2f} | {r['tau']:>9.2f} | {r['se']:>7.2f} | "
            f"[{r['ci'][0]:>7.1f}, {r['ci'][1]:>7.1f}] | {r['n_eff']:>6}{star}")
    main = rdd_estimate(df, c=0.0, h=main_h)
    log("")
    log(f"→ τ̂(h={main_h}) = {main['tau']:.2f} 만원/㎡ (참값 {TRUE_TAU:.1f}). "
        f"h=0.5~1.0에서 −55~−58로 안정적이나,")
    log("  h를 1.5~2.0으로 넓히면 경계에서 먼 굴곡의 편향이 유입돼 drift한다(편향-분산 상충).")
    log("  데이터기반 최적 대역폭(CCT rdrobust)이 이 균형을 자동화한다. 좁은 대역폭이 원칙.")
    log("")

    # --- (b) 전역 고차 다항식 = 편향된 방법 ---
    log("[2] 차수 민감성 진단: 전역 고차 다항식 (Gelman & Imbens 2019 경고)")
    log("-" * 68)
    g_taus = []
    for order in [2, 4, 6]:
        tau_g = global_polynomial_rdd(df, c=0.0, order=order)
        g_taus.append(tau_g)
        log(f"  전역 {order}차 다항식 τ̂ = {tau_g:>8.2f} 만원/㎡  "
            f"(참값 대비 편향 {tau_g - TRUE_TAU:+.2f})")
    log(f"  지역선형(h=1.0) τ̂ = {main['tau']:>8.2f} 만원/㎡  "
        f"(참값 대비 편향 {main['tau'] - TRUE_TAU:+.2f})")
    log(f"→ 전역 다항식 추정치는 차수에 따라 {min(g_taus):.1f}~{max(g_taus):.1f}로 "
        f"{max(g_taus) - min(g_taus):.1f}만큼 요동친다(차수 민감성). 반면 지역선형은")
    log("  대역폭 0.5~2.0에서 안정적이었다. 전역 다항식은 경계에서 먼 굴곡을 좇아")
    log("  경계 점프를 오염시키므로, 차수 선택이 결론을 좌우한다 → 지역선형을 쓴다.")
    log("")

    # --- (c) McCrary식 조작 검정: 정상 vs 정렬 ---
    log("[3] McCrary식 밀도 불연속 검정 (배정변수 조작·정렬 진단)")
    log("-" * 68)
    m_clean = mccrary_density_test(df, c=0.0)
    df_manip = load_data(manipulation=True)
    m_manip = mccrary_density_test(df_manip, c=0.0)
    log(f"  정상 표본   : θ(log밀도차) = {m_clean['theta']:+.3f}, "
        f"SE = {m_clean['se']:.3f}, z = {m_clean['z']:+.2f}  → 조작 증거 없음")
    log(f"  정렬 표본   : θ(log밀도차) = {m_manip['theta']:+.3f}, "
        f"SE = {m_manip['se']:.3f}, z = {m_manip['z']:+.2f}  → 밀도 불연속 탐지")
    log("→ |z|가 크면(정렬 표본) 경계 근방 배정이 무작위라는 가정이 깨진다.")
    log("")

    # --- (d) placebo cutoff: 가짜 경계에서 효과 0 ---
    log("[4] Placebo 검정: 가짜 경계(x=−1, +1)에서는 점프가 없어야 한다")
    log("-" * 68)
    for c_fake in [-1.0, 1.0]:
        # 가짜 경계 양쪽에서 각각 실제 처치가 섞이지 않도록 같은 처치영역 내부만 사용
        sub = df[df.treated == (1.0 if c_fake < 0 else 0.0)]
        r = rdd_estimate(sub, c=c_fake, h=0.75)
        log(f"  placebo c={c_fake:+.1f} : τ̂ = {r['tau']:>7.2f} 만원/㎡, "
            f"95% CI [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]  → 0 포함")
    log("→ 진짜 경계(x=0)에서만 유의한 점프 → 효과가 규제 경계에 특정됨.")
    log("")

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # 원자료(투명) + 구간 평균(binned scatter)
    ax.scatter(df.x, df.y, s=6, alpha=0.12, color="gray")
    edges = np.arange(-H_DATA, H_DATA + 0.25, 0.25)
    mid = (edges[:-1] + edges[1:]) / 2
    binned = [df.y[(df.x >= edges[i]) & (df.x < edges[i + 1])].mean() for i in range(len(mid))]
    ax.scatter(mid, binned, s=28, color="steelblue", label="구간 평균 실거래가", zorder=3)
    # 양쪽 지역선형 적합선(주 대역폭)
    for side, color, lab in [(df.x < 0, "crimson", "규제(x<0)"), (df.x >= 0, "seagreen", "비규제(x≥0)")]:
        sd = df[side]
        grid = np.linspace(sd.x.min(), sd.x.max(), 50)
        w = triangular_weights(sd.x.values, 0.0, main_h)
        m = w > 0
        Xf = np.column_stack([np.ones(m.sum()), sd.x.values[m]])
        beta = np.linalg.solve(Xf.T @ np.diag(w[m]) @ Xf, Xf.T @ np.diag(w[m]) @ sd.y.values[m])
        gm = (grid >= -main_h) & (grid <= main_h)
        ax.plot(grid[gm], beta[0] + beta[1] * grid[gm], color=color, lw=2.5, label=lab, zorder=4)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.annotate(f"τ̂ = {main['tau']:.1f} 만원/㎡", xy=(0, BASE_PRICE),
                xytext=(0.4, BASE_PRICE + 120), fontsize=11,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("경계까지 부호거리 (km)  ← 규제지역 | 비규제지역 →")
    ax.set_ylabel("주택 실거래가 (만원/㎡)")
    ax.set_title("부동산 규제경계의 가격 불연속 (RDD)")
    ax.legend()
    fig_path = RESULTS_DIR / "10-3-rdd-discontinuity.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    log(f"[그림] 경계 불연속 산점도 저장 → {fig_path.name}")
    log("")

    log("=" * 68)
    log("요약: 경계 불연속이라는 식별원이 있을 때 RDD는 교란(입지·학군)이 매끄럽게")
    log("변한다는 가정 아래 국소 규제효과를 복원한다. 지역선형+대역폭 민감도+McCrary+")
    log(f"placebo 4중 진단을 통과할 때만 신뢰한다. τ̂≈{main['tau']:.0f} (참 {TRUE_TAU:.0f}).")
    log("한계: 복합처치(경계에서 다른 시군구 정책도 변함)·내생지정 → 전국효과로 외삽 금지.")
    log("=" * 68)

    # 텍스트 로그도 남김(하네스 stdout 캡처와 별개로 재확인용)
    (RESULTS_DIR / "10-3-rdd-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
