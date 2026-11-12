"""ch14.md 설명 그림 여덟 장을 만든다. 모두 1~6절 가상 사례용이며 실습 출력이 아니다.

가상 사례: 새벌시의 베이커리 가맹 본부 '한올베이커리'(실재하지 않음).
그림 속 숫자는 모두 이 스크립트가 아래 가정값으로 계산한 값이다. 본문에 적은 가상 사례
숫자도 이 스크립트가 찍는 값을 옮긴 것이다(실행하면 화면에 함께 찍힌다).

그림 14.1  3절  "출점 효과"에 들어 있는 세 값(가상 숫자)
그림 14.2  3절  잘될 자리를 골랐을 때 전후 비교가 효과로 세는 폭(가상 숫자)
그림 14.3  4절  설계별 최소 탐지 가능 효과와 기대 효과(가상 숫자)
그림 14.4  4절  짝지은 상권이 이웃이 되고, 오염된 대조가 차이를 키운다(설명용 난수)
그림 14.5  5절  합성통제: 인접 상권에 준 가중치(설명용 난수)
그림 14.6  5절  두 경로 검산: 추정치가 말하는 자기잠식 비율(가상 숫자)
그림 14.7  6절  전이율에 따른 순현재가치와 손익분기(가상 숫자)
그림 14.8  6절  같은 예측구간, 다른 발주량(가상 숫자)

실행: python lecture/graphics/make_ch14_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT = HERE / "ch14"
OUT.mkdir(exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, RED, GREEN = "#1f77b4", "#e07b00", "#7f7f7f", "#d62728", "#2ca02c"
BLUE_L, ORANGE_L, RED_L, LIGHT = "#cfe3f4", "#fbe3c4", "#f6c9c9", "#f2f2f2"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", OUT / name)


def won(v):
    """만원 단위 정수를 천 단위 쉼표로. 음수 부호는 ASCII '-'."""
    return f"{v:,.0f}".replace("−", "-")


# ── 그림 14.1 세 값 ──
NEW, CANNIB, COMPET = 4000, 1000, 600          # 만원/월, 가상
brand = NEW - CANNIB
market = NEW - CANNIB - COMPET
print(f"[14.1] 신규점 {NEW} / 브랜드 순증 {brand} / 상권 순증 {market} / "
      f"신규점 매출을 쓰면 브랜드 순증보다 {NEW / brand - 1:.0%} 크게 셈")
fig, ax = plt.subplots(figsize=(9.5, 2.6))
labels = ["① 신규점 매출", "② 브랜드 순증", "③ 상권 순증"]
vals = [NEW, brand, market]
ax.barh([2, 1, 0], vals, color=[BLUE, ORANGE, GREEN], height=0.55)
ax.barh(1, CANNIB, left=brand, color="none", edgecolor=ORANGE, hatch="///", height=0.55)
ax.barh(0, COMPET, left=market, color="none", edgecolor=GREEN, hatch="///", height=0.55)
ax.text(brand + CANNIB / 2, 1, "자사 인근점 감소", ha="center", va="center", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.text(market + COMPET / 2, 0, "경쟁점 감소", ha="center", va="center", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
for y, v in zip([2, 1, 0], vals):
    ax.text(v / 2, y, f"{won(v)}만원", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
ax.set_yticks([2, 1, 0], labels, fontsize=11)
ax.set_xlim(0, 4700)
ax.set_xlabel("월 매출(만원)")
ax.text(4050, 2, "점포 손익", va="center", fontsize=10, color=BLUE)
ax.text(4050, 1, "출점 여부 판정", va="center", fontsize=10, color=ORANGE)
ax.text(4050, 0, "상권 확대 주장", va="center", fontsize=10, color=GREEN)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_1_three_values.png")

# ── 그림 14.2 잘될 자리를 고르면 ──
m = np.arange(0, 25)
open_m = 12
grow = 100 + 1.0 * m                          # 원래 오르던 상권(출점 없어도 계속 오름)
effect = np.where(m >= open_m, 4.0, 0.0)      # 참 효과 +4
actual = grow + effect
others = 100 + 0.15 * m
pre_mean = actual[:open_m].mean()
post_mean = actual[open_m:].mean()
naive = post_mean - pre_mean
print(f"[14.2] 전후 평균 차이 {naive:.1f} / 참 효과 4.0")
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.plot(m, actual, "-o", ms=3, color=RED, lw=2, label="출점한 상권(실제)")
ax.plot(m[open_m:], grow[open_m:], "--", color=RED, lw=1.5, alpha=0.7,
        label="출점하지 않았다면(보이지 않음)")
ax.plot(m, others, "-", color=GRAY, lw=1.5, label="다른 상권 평균")
ax.hlines(pre_mean, 0, open_m - 1, color=BLUE, lw=1.2, ls=":")
ax.hlines(post_mean, open_m, 24, color=BLUE, lw=1.2, ls=":")
ax.annotate("", (24.6, post_mean), (24.6, pre_mean),
            arrowprops=dict(arrowstyle="<->", color=BLUE))
ax.text(24.9, (pre_mean + post_mean) / 2, f"전후 비교\n+{naive:.1f}", va="center",
        fontsize=10, color=BLUE)
ax.annotate("", (23.5, actual[23]), (23.5, grow[23]),
            arrowprops=dict(arrowstyle="<->", color=RED))
ax.text(23.0, grow[23] - 4.0, "참 효과 +4", va="center", ha="right",
        fontsize=10, color=RED)
ax.axvline(open_m - 0.5, color=GRAY, lw=1)
ax.text(open_m - 0.7, 112, "출점", fontsize=10, color=GRAY, ha="right")
ax.set_xlim(-0.5, 27.5)
ax.set_xlabel("월")
ax.set_ylabel("매출 지수")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
save(fig, "fig_14_2_selection.png")

# ── 그림 14.3 설계별 MDE ──
designs = ["단순 무작위\n+ 사후 비교", "단순 무작위\n+ 전후 차이 비교", "짝지음\n+ 전후 차이 비교"]
se = np.array([0.080, 0.035, 0.012])          # 로그 스케일 표준오차, 가상
mde_log = (norm.ppf(0.975) + norm.ppf(0.80)) * se
mde_pct = np.expm1(mde_log) * 100
EXPECT = 4.0
print("[14.3] MDE(%):", ", ".join(f"{v:.1f}" for v in mde_pct))
fig, ax = plt.subplots(figsize=(9.5, 2.5))
cols = [GRAY if v > EXPECT else BLUE for v in mde_pct]
ax.barh([2, 1, 0], mde_pct, color=cols, height=0.55)
for y, v in zip([2, 1, 0], mde_pct):
    ax.text(max(v, EXPECT) + 0.5, y, f"{v:.1f}%" + ("  잡을 수 없음" if v > EXPECT else "  잡을 수 있음"),
            va="center", fontsize=10.5, color=RED if v > EXPECT else BLUE)
ax.axvline(EXPECT, color=RED, ls="--", lw=1.5)
ax.text(EXPECT + 0.3, 2.45, f"기대 효과 {EXPECT:.0f}%", color=RED, fontsize=10)
ax.set_yticks([2, 1, 0], designs, fontsize=10)
ax.set_xlim(0, 36)
ax.set_ylim(-0.5, 2.75)
ax.set_xlabel("최소 탐지 가능 효과(%) — 이보다 작은 효과는 실험해도 '효과 없음'으로 나옴")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_3_mde_designs.png")

# ── 그림 14.4 짝지음과 오염 ──
rng = np.random.default_rng(14)
R, C = 4, 9
n = R * C
rr, cc = np.divmod(np.arange(n), C)
adj = (np.abs(rr[:, None] - rr[None]) + np.abs(cc[:, None] - cc[None])) == 1
# 공간적으로 매끄러운 사전 궤적 특성(수준·성장률): 가까운 상권끼리 닮게 만든다
D2 = (rr[:, None] - rr[None]) ** 2 + (cc[:, None] - cc[None]) ** 2
F = np.exp(-D2 / (2 * 1.3 ** 2)) @ rng.normal(0, 1, (n, 2))
F = (F - F.mean(0)) / F.std(0) + rng.normal(0, 0.1, (n, 2))
# 사전 궤적이 가까운 순으로 탐욕적 짝짓기
left = set(range(n))
pairs = []
dist = np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))
np.fill_diagonal(dist, np.inf)
order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
for i, j in order:
    if i in left and j in left:
        pairs.append((i, j))
        left -= {i, j}
    if not left:
        break
treat = np.zeros(n, bool)
rng_t = np.random.default_rng(3)
for i, j in pairs:
    treat[i if rng_t.random() < 0.5 else j] = True
adj_pairs = sum(adj[i, j] for i, j in pairs)
ctrl = ~treat
exposed = ctrl & (adj & treat[None, :]).any(1)
expo_rate = exposed.sum() / ctrl.sum()
TRUE_FX, SPILL = 4.0, 1.5                      # %, 가상
gap = TRUE_FX + SPILL * expo_rate
print(f"[14.4] 쌍 {len(pairs)}개 중 인접 {adj_pairs}개 / 대조 노출률 {expo_rate:.2f} / "
      f"측정 차이 {gap:.1f}% (참 효과 {TRUE_FX}%)")
fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
ax = axes[0]
for k in range(n):
    face = ORANGE if treat[k] else (RED_L if exposed[k] else LIGHT)
    ax.add_patch(plt.Rectangle((cc[k], -rr[k]), 0.92, 0.92, facecolor=face, edgecolor="#999999"))
for i, j in pairs:
    x0, y0 = cc[i] + 0.46, -rr[i] + 0.46
    x1, y1 = cc[j] + 0.46, -rr[j] + 0.46
    ax.plot([x0, x1], [y0, y1], color=BLUE if adj[i, j] else GRAY,
            lw=2.2 if adj[i, j] else 0.9, alpha=0.9 if adj[i, j] else 0.6)
ax.set_xlim(-0.2, C + 0.1)
ax.set_ylim(-R + 0.75, 1.1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"(가) 36개 상권과 18개 짝 — 굵은 선 {adj_pairs}개는 서로 붙은 상권끼리의 짝", fontsize=10.5)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(facecolor=ORANGE, label="할인 상권(처치)"),
                   Patch(facecolor=RED_L, label="처치에 붙은 대조"),
                   Patch(facecolor=LIGHT, edgecolor="#999999", label="떨어진 대조")],
          frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=3)
ax = axes[1]
parts = [("참 효과", TRUE_FX, ORANGE), ("대조 하락분", SPILL * expo_rate, RED_L)]
ax.bar([0], [TRUE_FX], color=ORANGE, width=0.5)
ax.bar([1], [TRUE_FX], color=ORANGE, width=0.5)
ax.bar([1], [SPILL * expo_rate], bottom=TRUE_FX, color=RED_L, width=0.5, hatch="//",
       edgecolor=RED)
ax.text(0, TRUE_FX + 0.15, f"{TRUE_FX:.1f}%", ha="center", fontsize=10.5)
ax.text(1, gap + 0.15, f"{gap:.1f}%", ha="center", fontsize=10.5, color=RED)
ax.text(1.32, TRUE_FX + SPILL * expo_rate / 2, "옆 대조가\n내려간 몫", va="center",
        fontsize=9, color=RED)
ax.set_xticks([0, 1], ["실제 효과", "실험이 측정한 차이"])
ax.set_ylim(0, 6.6)
ax.set_xlim(-0.5, 2.1)
ax.set_ylabel("매출 증가(%)")
ax.set_title("(나) 오염된 대조와 비교하면", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_4_pairs_spillover.png")

# ── 그림 14.5 합성통제와 인접 상권 가중치 ──
rng = np.random.default_rng(7)
T, T0 = 36, 24
t = np.arange(T)
J = 12
neighbors = np.array([1, 5, 8])                 # 출점 상권에 붙은 기증 상권
slopes = np.linspace(-0.30, 0.62, J)
rng.shuffle(slopes)
slopes[neighbors] = [0.46, 0.52, 0.56]          # 이웃은 출점 상권과 닮음(공간 자기상관)
levels = rng.normal(100, 3, J)
levels[neighbors] = [100.5, 101.5, 102.0]
donors = levels[:, None] + slopes[:, None] * t[None] + rng.normal(0, 0.5, (J, T))
# 처치 상권: 잘될 자리를 고름 — 출점 뒤 아파트 입주로 원래 더 오를 자리였음
treated = 101 + 0.50 * t + rng.normal(0, 0.5, T)
treated[T0:] += 0.60 * (t[T0:] - T0 + 1)        # 출점과 무관한 추가 성장(보이지 않음)
treated[T0:] += 4.0                             # 실제 효과(현실에서는 보이지 않음)
donors[neighbors, T0:] -= 3.0                   # 인접 상권은 손님을 뺏겨 내려감

def synth(pool):
    Y = donors[pool][:, :T0].T
    k = len(pool)
    res = minimize(lambda w: ((treated[:T0] - Y @ w) ** 2).sum(), np.full(k, 1 / k),
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
    w = np.zeros(J)
    w[pool] = res.x
    path = w @ donors
    rmspe = np.sqrt(((treated[:T0] - path[:T0]) ** 2).mean())
    return w, path, rmspe


full = np.arange(J)
w_full, p_full, r_full = synth(full)
w_don, p_don, r_don = synth(np.setdiff1d(full, neighbors))
g_full = (treated[T0:] - p_full[T0:]).mean() / p_full[T0:].mean() * 100
g_don = (treated[T0:] - p_don[T0:]).mean() / p_don[T0:].mean() * 100
nb_w = w_full[neighbors].sum()
print(f"[14.5] 전체 풀: 인접 가중 합 {nb_w:.2f}, 사전 오차 {r_full:.2f}, 사후 격차 {g_full:.1f}% / "
      f"인접 제외: 사전 오차 {r_don:.2f}, 사후 격차 {g_don:.1f}%")
fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axes[0]
for j in range(J):
    ax.plot(t, donors[j], color="#cccccc", lw=0.8, zorder=1)
ax.plot(t, treated, color=RED, lw=2.2, label="출점 상권(실제)")
ax.plot(t, p_full, "--", color=BLUE, lw=1.8, label=f"합성(전체 풀) 격차 {g_full:.1f}%")
ax.plot(t, p_don, ":", color=GREEN, lw=2, label=f"합성(인접 제외) 격차 {g_don:.1f}%")
ax.axvline(T0 - 0.5, color=GRAY, lw=1)
ax.text(T0, ax.get_ylim()[0] + 1, " 출점", color=GRAY, fontsize=9.5)
ax.set_xlabel("월")
ax.set_ylabel("매출 지수")
ax.set_title("(가) 실제와 합성 궤적 (회색 = 기증 상권)", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
idx = np.argsort(-w_full)[:6]
ax.bar(range(len(idx)), w_full[idx], color=[RED if k in neighbors else BLUE for k in idx])
ax.set_xticks(range(len(idx)), [f"{'인접' if k in neighbors else '먼'}\n{k + 1}" for k in idx],
              fontsize=9)
ax.set_ylabel("가중치")
ax.set_title(f"(나) 전체 풀 가중치 상위 6곳 — 인접 합 {nb_w:.2f}", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_5_scm_neighbors.png")

# ── 그림 14.6 두 경로 검산 ──
ATT_EST = round(g_full, 1)                     # 14.5의 전체 풀 추정을 그대로 쓴다
N_STORE = 15
Ns = np.arange(5, 31)
imp = 1 - Ns * ATT_EST / 100
print(f"[14.6] 추정 {ATT_EST}% , N={N_STORE} -> 함의 자기잠식 비율 {1 - N_STORE * ATT_EST / 100:.0%}; "
      f"추정이 성립하는 N 상한 {100 / ATT_EST:.1f}")
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axhspan(0, 100, color=GREEN, alpha=0.10)
ax.text(29.8, 50, "말이 되는 범위\n0~100%", ha="right", va="center", fontsize=10, color=GREEN)
ax.plot(Ns, imp * 100, color=RED, lw=2.2, label=f"상권 매출 +{ATT_EST}%가 사실이라면")
v = (1 - N_STORE * ATT_EST / 100) * 100
ax.plot(N_STORE, v, "o", color=RED, ms=8)
ax.annotate(f"동종 점포 {N_STORE}곳이면 {v:.0f}%\n= 신규점이 자기 매출보다 더 큰 몫을\n   상권에 새로 만들었다는 뜻",
            (N_STORE, v), (5.6, -175), fontsize=9.5, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xlabel("상권 안 동종 점포 수(자사 + 경쟁)")
ax.set_ylabel("자기잠식 비율(%)")
ax.set_xlim(5, 30)
ax.set_ylim(-190, 115)
ax.legend(frameon=False, fontsize=10, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_6_reconcile.png")

# ── 그림 14.7 NPV와 손익분기 전이율 ──
SALES, MARGIN, INVEST, MONTHS, HURDLE = 4000, 0.14, 20000, 60, 0.10   # 만원, 가상


def npv(tr, margin=MARGIN, months=MONTHS, invest=INVEST):
    r = (1 + HURDLE) ** (1 / 12) - 1
    ann = (1 - (1 + r) ** -months) / r
    return SALES * margin * (1 - tr) * ann - invest


def breakeven(**kw):
    r = (1 + HURDLE) ** (1 / 12) - 1
    ann = (1 - (1 + r) ** -kw.get("months", MONTHS)) / r
    return 1 - kw.get("invest", INVEST) / (SALES * kw.get("margin", MARGIN) * ann)


cases = [("기준(이익률 14%, 계약 60개월)", {}, BLUE, "-"),
         ("이익률 12%", {"margin": 0.12}, ORANGE, "--"),
         ("계약 36개월", {"months": 36}, RED, ":")]
tr = np.linspace(0, 0.6, 61)
fig, ax = plt.subplots(figsize=(10, 2.5))
for lab, kw, c, ls in cases:
    be = breakeven(**kw)
    ax.plot(tr * 100, npv(tr, **kw), color=c, ls=ls, lw=2, label=f"{lab}: 손익분기 {be:.0%}")
    if be > 0:
        ax.plot(be * 100, 0, "o", color=c, ms=7)
    print(f"[14.7] {lab}: NPV(0%) {npv(0, **kw):,.0f}만원, 손익분기 {be:.1%}")
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xlabel("전이율(%) — 신규점 매출 가운데 자사 인근점에서 옮겨 온 비율")
ax.set_ylabel("순현재가치(만원)")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_14_7_npv_breakeven.png")

# ── 그림 14.8 임계비와 발주량 ──
MU, LO, HI = 200, 150, 250
sigma = (HI - LO) / (2 * norm.ppf(0.95))
items = [("당일 빵", 1000, 1800, ORANGE), ("냉동 생지", 3000, 300, BLUE)]
xs = np.linspace(120, 280, 400)
fig, ax = plt.subplots(figsize=(9.5, 2.6))
ax.plot(xs, norm.pdf(xs, MU, sigma), color="#333333", lw=1.5)
ax.fill_between(xs, 0, norm.pdf(xs, MU, sigma), where=(xs >= LO) & (xs <= HI), color=LIGHT)
ax.text(MU, norm.pdf(MU, MU, sigma) * 0.35, "90% 예측구간\n150~250개", ha="center", fontsize=9.5,
        color=GRAY)
for k, (lab, cu, co, c) in enumerate(items):
    cr = cu / (cu + co)
    q = norm.ppf(cr, MU, sigma)
    print(f"[14.8] {lab}: Cu {cu} Co {co} 임계비 {cr:.3f} 발주 {q:.0f}개")
    ax.axvline(q, color=c, lw=2.2)
    ax.text(q + (2 if k else -2), norm.pdf(MU, MU, sigma) * (1.02 - 0.2 * k),
            f"{lab}\n임계비 {cr:.2f} → {q:.0f}개", ha="left" if k else "right", fontsize=10, color=c)
ax.axvline(MU, color=GRAY, ls=":", lw=1)
ax.text(MU + 1, norm.pdf(MU, MU, sigma) * 1.12, "예측 200", ha="left", fontsize=9, color=GRAY)
ax.set_yticks([])
ax.set_xlabel("내일 수요(개)")
ax.set_ylim(-0.0018, norm.pdf(MU, MU, sigma) * 1.35)
ax.spines[["top", "right", "left"]].set_visible(False)
save(fig, "fig_14_8_critical_ratio.png")
