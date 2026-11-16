"""ch12.md 10절 실습 D의 설명 그림 세 장을 만든다. 값은 모두 실습 D의 실행 결과에서 읽는다.

그림 12.12  실루엣 계수의 뜻(왼쪽, 설명용 점 배치)과 묶음 수별 실루엣(오른쪽, 실습 로그)
그림 12.13  상권 유형 세 개의 점포 수와 생활서비스 보유율(실습 CSV)
그림 12.14  편의점 상권 배분 지수의 분포와 두 철수 선(실습 CSV와 로그)

입력: lecture_practice/chapter12/results/ 의
      12-4-trade-area-exit-decision.log, trade_area_typology.csv, store_exit_decision.csv
실행: python lecture/graphics/make_ch12_practice_d_figures.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE / "ch12"
OUT.mkdir(exist_ok=True)
RES = HERE.parents[1] / "lecture_practice" / "chapter12" / "results"

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, RED, GREEN = "#1f77b4", "#e07b00", "#7f7f7f", "#d62728", "#2ca02c"

LOG = (RES / "12-4-trade-area-exit-decision.log").read_text(encoding="utf-8")


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", OUT / name)


# ── 그림 12.12 실루엣 ─────────────────────────────────────────────
sil = {int(k): float(v) for k, v in re.findall(r"k=(\d):(\d\.\d+)", LOG)}
assert sorted(sil) == list(range(2, 9)), sil
best = max(sil, key=sil.get)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
# 왼쪽: 설명용 점 배치(실습 자료 아님)
g1 = np.array([[1.0, 1.0], [1.4, 1.5], [0.7, 1.6], [1.3, 0.6]])
g2 = np.array([[4.0, 1.2], [4.5, 1.7], [4.3, 0.7], [3.8, 1.8]])
ax1.scatter(*g1.T, s=90, color=BLUE, zorder=3, label="묶음 가")
ax1.scatter(*g2.T, s=90, color=ORANGE, zorder=3, label="묶음 나")
me = g1[0]
ax1.scatter(*me, s=220, facecolor="none", edgecolor="black", lw=2, zorder=4)
ax1.text(me[0] - 0.15, me[1] - 0.35, "읍면동 A", ha="center", fontsize=10)
for p in g1[1:]:
    ax1.plot([me[0], p[0]], [me[1], p[1]], color=BLUE, lw=1.2, ls="--")
for p in g2:
    ax1.plot([me[0], p[0]], [me[1], p[1]], color=ORANGE, lw=0.8, ls=":")
ax1.text(1.1, 2.05, "a = 같은 묶음 점까지\n평균 거리 (짧음)", color=BLUE, fontsize=10, ha="center")
ax1.text(3.0, 0.15, "b = 가장 가까운 다른 묶음 점까지\n평균 거리 (긺)", color=ORANGE, fontsize=10, ha="center")
ax1.set_xlim(0, 5.2)
ax1.set_ylim(-0.2, 2.6)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("(가) 실루엣 = (b - a) / (a와 b 중 큰 값)\n1에 가까우면 잘 갈림, 0 근처면 경계가 흐림", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)

ks = sorted(sil)
cols = [RED if k == best else GRAY for k in ks]
ax2.bar([str(k) for k in ks], [sil[k] for k in ks], color=cols)
for k in ks:
    ax2.text(ks.index(k), sil[k] + 0.004, f"{sil[k]:.3f}", ha="center", fontsize=9.5)
ax2.set_ylim(0, 0.3)
ax2.set_xlabel("묶음 수 k")
ax2.set_ylabel("실루엣 계수")
ax2.set_title(f"(나) 실습 D: k = {best}에서 가장 크지만 {sil[best]:.3f}로 낮음", fontsize=11)
fig.tight_layout()
save(fig, "fig_12_12_silhouette.png")

# ── 그림 12.13 유형 프로필 ─────────────────────────────────────────
ty = pd.read_csv(RES / "trade_area_typology.csv", encoding="utf-8-sig")
prof = ty.groupby("cluster").agg(n=("n_stores", "size"), med=("n_stores", "median"),
                                 conv=("has_convenience", "mean"),
                                 phar=("has_pharmacy", "mean"),
                                 clin=("has_clinic", "mean")).sort_values("med")
labels = [f"유형{c}\n({int(r.n)}곳)" for c, r in prof.iterrows()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
ax1.bar(labels, prof["med"], color=[GREEN, ORANGE, BLUE])
for i, v in enumerate(prof["med"]):
    ax1.text(i, v + 12, f"{v:.0f}개", ha="center", fontsize=10)
ax1.set_ylabel("읍면동 점포 수 (중앙값)")
ax1.set_title("(가) 동네 상가 규모", fontsize=11)
ax1.set_ylim(0, prof["med"].max() * 1.15)

x = np.arange(len(prof))
w = 0.26
for j, (col, lab, c) in enumerate([("conv", "편의점", BLUE), ("phar", "약국", ORANGE), ("clin", "의원", GREEN)]):
    vals = prof[col].to_numpy() * 100
    ax2.bar(x + (j - 1) * w, vals, w, color=c, label=lab)
    for i, v in enumerate(vals):
        ax2.text(x[i] + (j - 1) * w, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
ax2.set_xticks(x, labels)
ax2.set_ylim(0, 112)
ax2.set_ylabel("하나 이상 있는 읍면동 비율 (%)")
ax2.set_title("(나) 생활서비스가 하나라도 있는 읍면동 비율", fontsize=11)
ax2.legend(fontsize=9, loc="upper left")
fig.tight_layout()
save(fig, "fig_12_13_type_profiles.png")

# ── 그림 12.14 상권 배분 지수 분포 ─────────────────────────────────
st = pd.read_csv(RES / "store_exit_decision.csv", encoding="utf-8-sig")
m = re.search(r"즉시 철수 s < ([\d.]+) \| 만료 시점에 적자 s < ([\d.]+)", LOG)
cut_now, cut_exp = float(m.group(1)), float(m.group(2))
med = float(re.search(r"중앙값 ([\d.]+), 사분위", LOG).group(1))
s = st["s_index"].to_numpy()
share = {k: (st["decision"] == k).mean() * 100 for k in ("즉시철수", "만료시철수", "유지")}

fig, ax = plt.subplots(figsize=(10, 4.4))
bins = np.unique(np.r_[np.arange(0, 150 + 2.5, 2.5), cut_now, cut_exp])  # 철수 선을 칸 경계에 넣음
clip = np.clip(s, 0, 149.9)
for lo, hi, c in ((0, cut_now, RED), (cut_now, cut_exp, ORANGE), (cut_exp, 151, BLUE)):
    ax.hist(clip[(clip >= lo) & (clip < hi)], bins=bins, color=c, alpha=0.85)
for v, lab in ((cut_now, f"즉시 철수 선 {cut_now:.2f}"), (cut_exp, f"만료 시 철수 선 {cut_exp:.2f}")):
    ax.axvline(v, color="black", lw=1.2, ls="--")
ax.axvline(med, color=GRAY, lw=1, ls=":")
ymax = ax.get_ylim()[1] * 1.12
ax.set_ylim(0, ymax)
ax.text(cut_now - 1, ymax * 0.93, f"즉시 철수 선\n{cut_now:.2f}", ha="right", fontsize=9.5)
ax.text(cut_exp + 1, ymax * 0.93, f"만료 시 철수 선\n{cut_exp:.2f}", ha="left", fontsize=9.5)
ax.text(med, ymax * 0.62, f"중앙값\n{med:.2f}\n(손익 0)", ha="center", fontsize=9, color="#333333",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
ax.text(20, ymax * 0.55, f"즉시 철수\n{share['즉시철수']:.1f}%", color=RED, fontsize=11, ha="center", fontweight="bold")
ax.text(95, ymax * 0.55, f"유지\n{share['유지']:.1f}%", color=BLUE, fontsize=11, ha="center", fontweight="bold")
ax.annotate(f"버티다 만료 때 철수\n{share['만료시철수']:.1f}%", xy=((cut_now + cut_exp) / 2, ymax * 0.3),
            xytext=(120, ymax * 0.3), color=ORANGE, fontsize=11, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set_xlabel("상권 배분 지수 s = 읍면동 전체 점포 수 / 읍면동 편의점 수 (150 이상은 맨 끝 칸에 합침)")
ax.set_ylabel("편의점 수")
ax.set_title(f"편의점 {len(st):,}개의 상권 배분 지수와 두 철수 선", fontsize=11.5)
fig.tight_layout()
save(fig, "fig_12_14_s_distribution.png")

print("유형 프로필:\n", prof.round(3))
print("판정 비율:", {k: round(v, 1) for k, v in share.items()}, "s>=150:", int((s >= 150).sum()))
