"""ch13.md 설명 그림을 만든다. 모두 1~11절 가상 사례(하람시 하늘구, 별다리커피)용이며 실습 출력이 아니다.

가상 세계
  - 하늘구: 가로 4km x 세로 3km, 1km 정사각형 행정동 12개
  - 낮 활동 인구: 동별 합계를 정하고, 동 안에서는 오피스·역세권 등 몇 곳에 몰리게 100m 격자로 나눔
  - 카페: 별다리커피(자사) 5곳, 경쟁 카페 55곳(낮 인구가 많은 곳에 몰리도록 난수 시드 13으로 배치)
  - 흡인 모형: 매력도는 모두 1, 거리 저항 2.0, 거리 하한 100m(실습 코드와 같은 계산 구조)

그림 13.1  문제 1   동별 주민 인구와 낮 활동 인구
그림 13.2  문제 2   한 수요 지점의 손님이 매장별로 나뉘는 비율: 새 매장 개점 전·후
그림 13.3  문제 2   하늘구 지도와 후보지 넷, 원형 버퍼 인구 vs 흡인 모형 예상 손님
그림 13.4  문제 3   후보지별 예상 손님 = 자사에서 옮겨 온 몫 + 경쟁사에서 옮겨 온 몫
그림 13.5  문제 4   같은 자기잠식을 두 분모로 나눈 전이율
그림 13.6  문제 5   자사 매장 비율과 전이율
그림 13.7  문제 6   수요를 100m 격자로 알 때 vs 동 대표점 하나로만 알 때
그림 13.8  문제 7   전이율 상한을 낮출 때 남는 후보 수와 최선 순증
그림 13.9  문제 8   거리 저항을 바꿨을 때 순증의 크기와 최선 후보 위치
그림 13.10 문제 10  매장을 닫을 때 그 매장 손님이 가는 곳

실행: python lecture/graphics/make_ch13_figures.py
그림에 적힌 숫자는 모두 이 파일이 계산한 값이며, 본문에 인용한 값은 실행 때 화면에 찍힌 값이다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE / "ch13"
OUT.mkdir(exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, RED, GREEN = "#1f77b4", "#e07b00", "#7f7f7f", "#d62728", "#2ca02c"
BLUE_L, ORANGE_L, LIGHT = "#cfe3f4", "#fbe3c4", "#f2f2f2"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", OUT / name)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ── 가상 세계 만들기 ──
# 행정동: 아래(남)쪽 줄부터 위(북)쪽 줄 순서, 각 줄은 서->동
DONG = ["남산동", "강변동", "새터동", "들말동",
        "강서동", "중앙동", "역전동", "신시가동",
        "북문동", "서원동", "향교동", "동문동"]
DAY = np.array([9, 8, 10, 7, 11, 38, 26, 14, 12, 10, 9, 8]) * 1000.0   # 낮 활동 인구
RES = np.array([17, 14, 15, 12, 15, 5, 10, 16, 14, 13, 11, 12]) * 1000.0  # 주민등록 인구
DCX = np.array([i % 4 + 0.5 for i in range(12)])   # 동 중심(km)
DCY = np.array([i // 4 + 0.5 for i in range(12)])

H = 0.1  # 100m 격자
gx, gy = np.meshgrid(np.arange(0, 4, H) + H / 2, np.arange(0, 3, H) + H / 2)
gx, gy = gx.ravel(), gy.ravel()
gdong = np.floor(gy).astype(int) * 4 + np.floor(gx).astype(int)
# 사람이 몰리는 곳: (x, y, 퍼짐 km, 세기) — 중앙동 오피스, 역전동 역 앞, 신시가동 상가, 그 밖의 생활권
HOT = [(1.8, 1.25, 0.18, 6), (2.15, 1.55, 0.15, 4), (3.3, 1.7, 0.2, 2),
       (0.4, 0.3, 0.25, 1.5), (2.7, 2.6, 0.25, 1.5), (0.7, 1.8, 0.25, 1.5)]
wgt = np.full(gx.size, 0.4)
for x0, y0, s, a in HOT:
    wgt += a * np.exp(-((gx - x0) ** 2 + (gy - y0) ** 2) / (2 * s * s))
fpop = np.zeros_like(wgt)
for d in range(12):
    m = gdong == d
    fpop[m] = wgt[m] / wgt[m].sum() * DAY[d]

rng = np.random.default_rng(13)
NCOMP = 55
pp = fpop ** 2
k = rng.choice(gx.size, NCOMP, p=pp / pp.sum())
compx = gx[k] + rng.uniform(-H / 2, H / 2, NCOMP)
compy = gy[k] + rng.uniform(-H / 2, H / 2, NCOMP)
OWNX = np.array([1.35, 2.25, 2.45, 0.6, 3.15])
OWNY = np.array([1.75, 1.45, 1.65, 2.4, 1.85])
SX, SY = np.r_[OWNX, compx], np.r_[OWNY, compy]
OWN = np.r_[np.ones(len(OWNX), bool), np.zeros(NCOMP, bool)]
MIN_D = 0.1

CAND = {"가": (1.8, 1.3), "나": (2.35, 1.55), "다": (3.35, 1.65), "라": (0.6, 0.5)}
CX = np.array([v[0] for v in CAND.values()])
CY = np.array([v[1] for v in CAND.values()])


def huff(candx, candy, beta=2.0, pop=fpop, px=gx, py=gy, own=OWN, sx=SX, sy=SY):
    """후보지마다 (예상 손님, 자사에서 옮겨 온 몫, 순증, 자사 기존점의 개점 전 손님)."""
    d = np.maximum(np.hypot(px[:, None] - sx[None], py[:, None] - sy[None]), MIN_D)
    w = 1 / d ** beta
    D = w.sum(1)
    S = w[:, own].sum(1)
    dc = np.maximum(np.hypot(px[:, None] - candx[None], py[:, None] - candy[None]), MIN_D)
    a = 1 / dc ** beta
    Dn = D[:, None] + a
    gross = (pop[:, None] * a / Dn).sum(0)
    loss = (pop[:, None] * S[:, None] * (1 / D[:, None] - 1 / Dn)).sum(0)
    return gross, loss, gross - loss, (pop * S / D).sum()


def buffer_pop(x0, y0, r=0.5):
    return fpop[np.hypot(gx - x0, gy - y0) <= r].sum()


gross4, loss4, net4, own_before = huff(CX, CY)
buf4 = np.array([buffer_pop(x, y) for x, y in zip(CX, CY)])
ncomp4 = np.array([(np.hypot(compx - x, compy - y) <= 0.5).sum() for x, y in zip(CX, CY)])
print(f"자사 {len(OWNX)}곳 / 경쟁 {NCOMP}곳 -> 자사 매장 비율 {len(OWNX) / (len(OWNX) + NCOMP):.1%}")
for i, c in enumerate(CAND):
    print(f"후보 {c}: 500m 인구 {buf4[i]:,.0f}  500m 안 경쟁 {ncomp4[i]}  예상 손님 {gross4[i]:,.0f}  "
          f"자기잠식 {loss4[i]:,.0f}  순증 {net4[i]:,.0f}  "
          f"전이율A {loss4[i] / gross4[i]:.1%}  전이율B {loss4[i] / own_before:.1%}")

# ── 그림 13.1 주민 인구와 낮 활동 인구 ──
fig, ax = plt.subplots(figsize=(10, 2.6))
x = np.arange(12)
ax.bar(x - 0.2, RES / 1000, 0.4, color=LIGHT, edgecolor=GRAY, label="주민등록 인구")
ax.bar(x + 0.2, DAY / 1000, 0.4, color=BLUE, label="평일 낮 활동 인구")
for i in (5, 6):
    ax.text(i + 0.2, DAY[i] / 1000 + 1, f"{DAY[i] / 1000:.0f}천", ha="center", fontsize=9.5,
            color=BLUE, fontweight="bold")
ax.annotate("주민은 꼴찌", (5 - 0.2, RES[5] / 1000), (3.3, 25), fontsize=9.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks(x, DONG, fontsize=9.5)
ax.set_ylabel("인구(천 명)")
ax.set_ylim(0, 44)
tidy(ax)
ax.legend(frameon=False, loc="upper right")
save(fig, "fig_13_1_resident_vs_daytime.png")


# ── 그림 13.3 지도와 버퍼 인구 vs 예상 손님 ──
def draw_map(ax, show_pop=True, cands=True, circle=None):
    if show_pop:
        ax.imshow(fpop.reshape(30, 40), origin="lower", extent=(0, 4, 0, 3), cmap="Greys",
                  alpha=0.4, vmax=np.quantile(fpop, 0.98))
    for v in range(1, 4):
        ax.axvline(v, color="#999999", lw=0.6)
    for h in range(1, 3):
        ax.axhline(h, color="#999999", lw=0.6)
    for i, nm in enumerate(DONG):
        ax.text(DCX[i] + 0.46, DCY[i] + 0.36, nm, fontsize=7.5, color="#555555", ha="right")
    ax.scatter(compx, compy, s=13, color="#333333", edgecolor="white", lw=0.4, label="경쟁 카페",
               zorder=3)
    ax.scatter(OWNX, OWNY, s=70, marker="*", color=ORANGE, edgecolor="k", lw=0.5,
               label="별다리커피", zorder=4)
    if cands:
        for c, (xx, yy) in CAND.items():
            ax.scatter(xx, yy, s=150, facecolor="white", edgecolor=BLUE, lw=1.8, zorder=5)
            ax.text(xx, yy, c, ha="center", va="center", fontsize=9, color=BLUE,
                    fontweight="bold", zorder=6)
    if circle:
        ax.add_patch(plt.Circle(circle, 0.5, fill=False, color=BLUE, ls="--", lw=1.2))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
draw_map(axes[0], circle=CAND["가"])
axes[0].legend(frameon=True, fontsize=8, loc="lower right", framealpha=0.9)
axes[0].set_title("하늘구(진할수록 낮 인구가 많음, 점선 = 가의 반경 500m)", fontsize=9.5)
ax = axes[1]
xx = np.arange(4)
ax.bar(xx - 0.2, buf4 / 1000, 0.4, color=LIGHT, edgecolor=GRAY, label="반경 500m 낮 인구")
ax.bar(xx + 0.2, gross4 / 1000, 0.4, color=BLUE, label="흡인 모형 예상 손님")
for i in range(4):
    ax.text(i - 0.2, buf4[i] / 1000 + 0.8, f"경쟁\n{ncomp4[i]}곳", ha="center", fontsize=8,
            color=GRAY)
    ax.text(i + 0.2, gross4[i] / 1000 + 0.8, f"{gross4[i] / 1000:.1f}", ha="center",
            fontsize=8.5, color=BLUE)
ax.set_xticks(xx, [f"후보 {c}" for c in CAND])
ax.set_ylabel("천 명")
ax.set_ylim(0, buf4.max() / 1000 * 1.25)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
tidy(ax)
save(fig, "fig_13_3_buffer_vs_huff.png")

# ── 그림 13.2 한 수요 지점의 손님이 나뉘는 비율: 개점 전·후 ──
# 신시가동 100m 칸 하나에서 가까운 매장 셋과 '그 밖의 매장 전부'로 나눠 봄
pi = int(np.argmin(np.hypot(gx - 3.45, gy - 1.45)))
newx, newy = CAND["다"]


def shares(extra):
    xs = np.r_[SX, [newx]] if extra else SX
    ys = np.r_[SY, [newy]] if extra else SY
    d = np.maximum(np.hypot(gx[pi] - xs, gy[pi] - ys), MIN_D)
    w = 1 / d ** 2
    return w / w.sum()


p0, p1 = shares(False), shares(True)
near = np.argsort(np.hypot(gx[pi] - SX, gy[pi] - SY))[:3]
fig, ax = plt.subplots(figsize=(10, 2.0))
cols = [ORANGE if OWN[j] else c for j, c in zip(near, ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7"])]
for row, p, extra in [(1, p0, False), (0, p1, True)]:
    left = 0
    parts = [(p[j], cols[n], f"{'별다리' if OWN[j] else '경쟁'}") for n, j in enumerate(near)]
    parts.append((1 - sum(p[j] for j in near) - (p[-1] if extra else 0), LIGHT, "그 밖의 매장"))
    if extra:
        parts.append((p[-1], RED, "새 매장"))
    for v, c, lab in parts:
        ax.barh(row, v, left=left, color=c, edgecolor="white")
        if v > 0.04:
            ax.text(left + v / 2, row, f"{lab}\n{v:.0%}", ha="center", va="center", fontsize=8.5)
        left += v
ax.set_yticks([1, 0], ["개점 전", "후보 다에\n개점 후"])
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("신시가동 한 칸(100m)의 손님이 각 매장을 고르는 확률 - 두 줄 모두 합이 100%")
tidy(ax)
save(fig, "fig_13_2_denominator.png")

# ── 그림 13.4 예상 손님 = 자사에서 + 경쟁사에서 ──
order = np.argsort(-gross4)
labels = list(CAND)
fig, ax = plt.subplots(figsize=(9, 2.4))
for r, i in enumerate(order):
    ax.barh(r, net4[i] / 1000, color=BLUE, label="경쟁사에서 옮겨 온 손님 = 순증" if r == 0 else None)
    ax.barh(r, loss4[i] / 1000, left=net4[i] / 1000, color=ORANGE,
            label="별다리 다른 매장에서 옮겨 온 손님 = 자기잠식" if r == 0 else None)
    nr = int((net4 > net4[i]).sum()) + 1
    ax.text(gross4[i] / 1000 + 0.05, r, f" 예상 손님 {r + 1}위 / 순증 {nr}위", va="center",
            fontsize=9.5)
ax.set_yticks(range(4), [f"후보 {labels[i]}" for i in order])
ax.invert_yaxis()
ax.set_xlabel("천 명")
ax.set_xlim(0, gross4.max() / 1000 * 1.45)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
tidy(ax)
save(fig, "fig_13_4_cannibalization.png")

# ── 그림 13.5 같은 자기잠식, 두 분모 ──
iD = labels.index("다")
rateA, rateB = loss4[iD] / gross4[iD], loss4[iD] / own_before
print(f"후보 다 전이율: 새 매장 손님 대비 {rateA:.1%} / 기존 자사 매장 손님 대비 {rateB:.1%}")
fig, ax = plt.subplots(figsize=(9, 1.9))
for r, (den, lab, rate, c) in enumerate([
        (gross4[iD], "출점팀: 자기잠식 / 새 매장 예상 손님", rateA, BLUE),
        (own_before, "재무팀: 자기잠식 / 기존 별다리 매장 손님 합", rateB, GREEN)]):
    ax.barh(r, den / 1000, color=LIGHT, edgecolor=GRAY)
    ax.barh(r, loss4[iD] / 1000, color=ORANGE)
    ax.text(den / 1000 + 0.2, r, f"{lab} = {rate:.0%}", va="center", fontsize=10, color=c)
ax.set_yticks([0, 1], ["분모 A", "분모 B"])
ax.invert_yaxis()
ax.set_xlim(0, own_before / 1000 * 2.0)
ax.set_xlabel("천 명 (주황 = 후보 다의 자기잠식, 두 줄에서 같은 크기)")
tidy(ax)
save(fig, "fig_13_5_two_denominators.png")

# ── 그림 13.6 자사 매장 비율과 전이율 ──
fracs = np.array([0.08, 0.15, 0.25, 0.35, 0.5, 0.65])
rng2 = np.random.default_rng(7)
rates = []
for f in fracs:
    n_own_extra = int(round(f * (len(OWNX) + NCOMP))) - len(OWNX)
    vals = []
    for _ in range(30):
        own = OWN.copy()
        if n_own_extra > 0:
            own[len(OWNX) + rng2.choice(NCOMP, n_own_extra, replace=False)] = True
        g, l, _, _ = huff(CX[[iD]], CY[[iD]], own=own)
        vals.append(l[0] / g[0])
    rates.append(np.mean(vals))
rates = np.array(rates)
print("자사 매장 비율별 후보 다 전이율:", ", ".join(f"{f:.0%}->{r:.0%}" for f, r in zip(fracs, rates)))
fig, ax = plt.subplots(figsize=(9, 2.3))
ax.plot(fracs * 100, rates * 100, "-o", color=ORANGE, lw=2)
ax.plot(fracs[0] * 100, rates[0] * 100, "o", ms=11, mfc="none", mec=RED, mew=2)
ax.annotate("하늘구의 별다리커피", (fracs[0] * 100, rates[0] * 100), (14, rates[0] * 100 + 22),
            fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("주변 카페 가운데 자사 매장 비율(%) - 경쟁 카페 일부를 자사로 바꿔 가며 계산, 30회 평균")
ax.set_ylabel("후보 다의 전이율(%)")
ax.set_ylim(0, 100)
tidy(ax)
save(fig, "fig_13_6_share_vs_rate.png")

# ── 후보 격자(250m) 전체 계산 ──
cgx, cgy = np.meshgrid(np.arange(0.125, 4, 0.25), np.arange(0.125, 3, 0.25))
cgx, cgy = cgx.ravel(), cgy.ravel()
gG, lG, nG, _ = huff(cgx, cgy)
tG = lG / gG

# ── 그림 13.7 100m 격자 수요 vs 동 대표점 수요 ──
gC, lC, nC, _ = huff(cgx, cgy, pop=DAY, px=DCX, py=DCY)
bF, bC = int(nG.argmax()), int(nC.argmax())
shift = np.hypot(cgx[bF] - cgx[bC], cgy[bF] - cgy[bC]) * 1000
print(f"최선 후보: 100m 격자 수요 ({cgx[bF]:.3f},{cgy[bF]:.3f}) / 동 대표점 수요 ({cgx[bC]:.3f},{cgy[bC]:.3f}) "
      f"-> {shift:.0f}m 이동")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
for ax, val, b, title in [(axes[0], nG, bF, "(가) 낮 인구를 100m 격자로 알 때"),
                          (axes[1], nC, bC, "(나) 동마다 대표점(x) 하나로만 알 때")]:
    im = ax.imshow(val.reshape(12, 16), origin="lower", extent=(0, 4, 0, 3), cmap="Blues")
    draw_map(ax, show_pop=False, cands=False)
    ax.scatter(cgx[b], cgy[b], s=220, facecolor="none", edgecolor=RED, lw=2.2, zorder=6)
    ax.set_title(title, fontsize=10)
axes[1].scatter(DCX, DCY, s=25, marker="x", color=RED, zorder=6)
axes[0].legend(frameon=True, fontsize=7.5, loc="lower right", framealpha=0.9)
fig.colorbar(im, ax=axes, shrink=0.8, label="순증(진할수록 큼), 빨간 원 = 1위", ticks=[])
fig.savefig(OUT / "fig_13_7_resolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("저장:", OUT / "fig_13_7_resolution.png")

# ── 그림 13.8 전이율 상한 ──
caps = np.linspace(0.0, tG.max() + 0.02, 200)
n_pass = np.array([(tG <= c).sum() for c in caps])
best = np.array([nG[tG <= c].max() if (tG <= c).any() else np.nan for c in caps])
first_bind = tG[nG.argmax()]
print(f"후보 {len(cgx)}곳 전이율: 중앙값 {np.median(tG):.1%}, 최대 {tG.max():.1%}; "
      f"순증 1위의 전이율 {first_bind:.1%}")
for c in (0.30, 0.10, 0.08, 0.06):
    m = tG <= c
    print(f"  상한 {c:.0%}: 통과 {m.sum()}/{len(cgx)}, 최선 순증 {nG[m].max():,.0f} "
          f"({nG[m].max() / nG.max() - 1:+.1%})")
fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
axes[0].plot(caps * 100, n_pass, color=BLUE, lw=2)
axes[0].set_ylabel("통과한 후보 수")
axes[1].plot(caps * 100, best / 1000, color=ORANGE, lw=2)
axes[1].set_ylabel("통과 후보 중\n최선 순증(천 명)")
axes[1].set_ylim(0, nG.max() / 1000 * 1.15)
for ax in axes:
    ax.axvline(first_bind * 100, color=RED, ls=":", lw=1.2)
    ax.set_xlabel("전이율 상한(%) - 이 값을 넘는 후보를 탈락시킴")
    ax.set_xlim(0, caps.max() * 100)
    tidy(ax)
axes[0].annotate("이보다 낮추면\n순증 1위가 탈락", (first_bind * 100, len(cgx) * 0.5),
                 (first_bind * 100 + 6, len(cgx) * 0.12), fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "fig_13_8_cap.png")

# ── 그림 13.9 거리 저항 민감도 ──
betas = [1.5, 2.0, 2.5]
res = {b: huff(cgx, cgy, beta=b) for b in betas}
for b in betas:
    n_ = res[b][2]
    i = int(n_.argmax())
    print(f"거리 저항 {b}: 최선 순증 {n_.max():,.0f}, 위치 ({cgx[i]:.3f},{cgy[i]:.3f}), "
          f"평균 전이율 {np.mean(res[b][1] / res[b][0]):.1%}")
ranks = {b: (-res[b][2]).argsort().argsort() + 1 for b in betas}
cells = [(int(np.argmin(np.hypot(cgx - 0.375, cgy - 0.375))), "남산동 칸", BLUE),
         (int(np.argmin(np.hypot(cgx - 3.375, cgy - 1.625))), "신시가동 칸", ORANGE)]
for i, nm, _ in cells:
    print(f"{nm} 순증 순위(거리 저항 1.5/2.0/2.5):", [int(ranks[b][i]) for b in betas])
fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
vals = [res[b][2].max() / 1000 for b in betas]
ax.bar(range(3), vals, color=[LIGHT, BLUE, LIGHT], edgecolor=GRAY)
for i, v in enumerate(vals):
    ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)
ax.set_xticks(range(3), [f"거리 저항\n{b}" for b in betas])
ax.set_ylabel("1위 후보의 순증(천 명)")
ax.set_ylim(0, max(vals) * 1.2)
ax.set_title("(가) 크기는 가정에 따라 크게 달라짐", fontsize=10)
tidy(ax)
ax = axes[1]
for (i, nm, c), up in zip(cells, (False, True)):
    rr = [ranks[b][i] for b in betas]
    ax.plot(range(3), rr, "-o", color=c, lw=2, label=nm)
    for k_, r_ in enumerate(rr):
        ax.text(k_, r_ + (-0.5 if up else 0.5), f"{r_}위", ha="center",
                va="bottom" if up else "top", fontsize=9.5, color=c)
ax.set_xticks(range(3), [f"거리 저항 {b}" for b in betas])
ax.set_xlim(-0.3, 2.5)
ax.invert_yaxis()
ax.set_ylabel("순증 순위")
ax.set_title(f"(나) 후보 {len(cgx)}곳 가운데 두 칸의 순위", fontsize=10)
ax.set_ylim(10.5, -1)
ax.set_yticks([1, 3, 5, 7, 9], ["1위", "3위", "5위", "7위", "9위"])
ax.legend(frameon=False, fontsize=9, loc="lower left")
tidy(ax)
save(fig, "fig_13_9_beta.png")

# ── 그림 13.10 매장을 닫을 때 손님이 가는 곳 ──


def close_store(j):
    d = np.maximum(np.hypot(gx[:, None] - SX[None], gy[:, None] - SY[None]), MIN_D)
    w = 1 / d ** 2
    D = w.sum(1)
    S = w[:, OWN].sum(1)
    mine = (fpop * w[:, j] / D).sum()
    Dr = D - w[:, j]
    S_rest = S - w[:, j]
    back = (fpop * S_rest * (1 / Dr - 1 / D)).sum()
    return mine, back


def nearest_own(j):
    d = np.hypot(OWNX - OWNX[j], OWNY - OWNY[j])
    d[j] = np.inf
    return d.min() * 1000


cases = [(j, f"{nm} 매장\n(가장 가까운 별다리 {nearest_own(j):,.0f}m)") for j, nm in [(2, "역전동"), (3, "북문동")]]
fig, ax = plt.subplots(figsize=(9, 2.0))
for r, (j, lab) in enumerate(cases):
    mine, back = close_store(j)
    print(f"{lab.splitlines()[0]} 폐점: 손님 {mine:,.0f}, 다른 별다리가 되찾는 몫 {back:,.0f} "
          f"({back / mine:.0%}), 회사 전체 손실 {mine - back:,.0f}")
    ax.barh(r, back / 1000, color=ORANGE, label="다른 별다리 매장으로 = 되찾는 몫" if r == 0 else None)
    ax.barh(r, (mine - back) / 1000, left=back / 1000, color=GRAY,
            label="경쟁 카페로 = 회사 전체 손실" if r == 0 else None)
    ax.text(mine / 1000 + 0.05, r, f" 되찾는 비율 {back / mine:.0%}", va="center", fontsize=10)
ax.set_yticks([0, 1], [c[1] for c in cases], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("닫는 매장의 손님(천 명)")
ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2)
tidy(ax)
save(fig, "fig_13_10_exit_recapture.png")
