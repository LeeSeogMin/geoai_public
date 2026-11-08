"""ch11-2.md 그림 여덟 장을 만든다.

그림 11.1  4절  이웃을 정하는 다섯 가지 방법과 공간가중행렬(설명용 배치, 숫자 없음)
그림 11.2  4절  공간 시차와 행 표준화(본문 계산 예의 숫자, 실습 출력 아님)
그림 11.3  5절  그래디언트 부스팅: 11-0이 만든 보험 건물 500채의 표고 하나만 입력해 나무 수별 예측선을 그림
그림 11.4  6절  무작위 분할과 블록 분할의 배치도(설명용 배치, 숫자 없음)
그림 11.5  7절  예측구간 폭 비교(본문 계산 예의 숫자, 실습 출력 아님)
그림 11.6 10절  치안 시뮬레이션 R²와 순찰-실제 위험 상관(results/predictive_policing_bias.csv)
그림 11.7 11.2절 완전·관측·대조군 R²(results/11-3-insurance-risk-grading.log에서 읽음)
그림 11.8 11.3절 치안·보험 두 순환의 개념도(숫자 없음)

먼저 11-0, 11-2, 11-3 코드를 실행해 data/와 results/를 만든 뒤 이 파일을 실행한다.

실행: python lecture/graphics/make_ch11_figures.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PRACTICE = HERE.parent.parent / "lecture_practice" / "chapter11"
OUT = HERE / "ch11"
OUT.mkdir(exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, LIGHT, RED = "#1f77b4", "#e07b00", "#7f7f7f", "#f2f2f2", "#d62728"
BLUE_L, ORANGE_L = "#cfe3f4", "#fbe3c4"


def cell(ax, x, y, text, face, edge="#555555", bold=False, size=13):
    """(x, y)는 칸의 왼쪽 아래. 칸 크기는 1."""
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edge, lw=1.2))
    ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def cross(ax, ox, oy, center, up, right, down, left, center_face, nb_face):
    """가운데 칸과 상하좌우 이웃 네 칸을 십자로 그린다."""
    cell(ax, ox + 1, oy + 1, center, center_face, bold=True)
    cell(ax, ox + 1, oy + 2, up, nb_face)
    cell(ax, ox + 2, oy + 1, right, nb_face)
    cell(ax, ox + 1, oy + 0, down, nb_face)
    cell(ax, ox + 0, oy + 1, left, nb_face)


fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ── 왼쪽: 공간 시차 ──
axl.set_xlim(0, 9.2)
axl.set_ylim(-1.9, 4.2)
axl.axis("off")
axl.set_title("(가) 자기 기록이 같아도 주변이 다르다", fontsize=12, pad=8)

cross(axl, 0.0, 0.0, "3", "12", "14", "10", "8", ORANGE_L, LIGHT)
cross(axl, 5.2, 0.0, "3", "3", "2", "3", "4", BLUE_L, LIGHT)
axl.text(1.5, 3.35, "격자 A", ha="center", fontsize=12, fontweight="bold", color=ORANGE)
axl.text(6.7, 3.35, "격자 B", ha="center", fontsize=12, fontweight="bold", color=BLUE)
axl.text(1.5, -0.55, "(12+14+10+8) / 4", ha="center", fontsize=10.5, color=GRAY)
axl.text(1.5, -1.25, "공간 시차 = 11건", ha="center", fontsize=12.5,
         fontweight="bold", color=ORANGE)
axl.text(6.7, -0.55, "(3+2+3+4) / 4", ha="center", fontsize=10.5, color=GRAY)
axl.text(6.7, -1.25, "공간 시차 = 3건", ha="center", fontsize=12.5,
         fontweight="bold", color=BLUE)
axl.text(4.1, 1.5, "vs", ha="center", va="center", fontsize=11, color=GRAY)

# ── 오른쪽: 행 표준화 ──
axr.set_xlim(0, 9.4)
axr.set_ylim(-3.0, 5.4)
axr.axis("off")
axr.set_title("(나) 이웃 수가 달라도 평균하면 같다", fontsize=12, pad=14)

N = 4
gx, gy = 0.4, 0.2
inner, corner = (2, 2), (0, 3)           # 열, 행(아래에서부터)
inner_nb = {(2, 3), (3, 2), (2, 1), (1, 2)}
corner_nb = {(1, 3), (0, 2)}
for c in range(N):
    for r in range(N):
        face = "white"
        if (c, r) == inner:
            face = BLUE
        elif (c, r) == corner:
            face = ORANGE
        elif (c, r) in inner_nb:
            face = BLUE_L
        elif (c, r) in corner_nb:
            face = ORANGE_L
        cell(axr, gx + c, gy + r, "5", face, size=12,
             bold=face in (BLUE, ORANGE))
axr.text(gx + 2.5, gy + 4.3, "모든 칸의 기록 = 5건", ha="center", fontsize=10.5, color=GRAY)

axr.text(5.2, 3.0, "귀퉁이 칸 (이웃 2개)", fontsize=11, color=ORANGE, fontweight="bold")
axr.text(5.2, 2.4, "더하면 5+5 = 10건", fontsize=10.5)
axr.text(5.2, 1.85, "평균하면 10/2 = 5건", fontsize=10.5, fontweight="bold")
axr.text(5.2, 0.7, "한가운데 칸 (이웃 4개)", fontsize=11, color=BLUE, fontweight="bold")
axr.text(5.2, 0.1, "더하면 5×4 = 20건", fontsize=10.5)
axr.text(5.2, -0.45, "평균하면 20/4 = 5건", fontsize=10.5, fontweight="bold")
axr.text(4.7, -1.7,
         "주변 위험은 같은데 합은 두 배 차이 → 합 대신 평균(행 표준화)을 씀",
         ha="center", fontsize=10.5, color="#444444")

fig.tight_layout()
fig.savefig(OUT / "fig_11_2_spatial_lag.png", dpi=150)
plt.close(fig)
print("saved", OUT / "fig_11_2_spatial_lag.png")


# ── 그림 11.3: 그래디언트 부스팅(5절) ──
from sklearn.ensemble import GradientBoostingRegressor

bld = pd.read_parquet(PRACTICE / "data" / "insurance_buildings.parquet")
x = bld["elevation"].values.reshape(-1, 1)
y = bld["true_flood_prob"].values
grid_x = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
steps = [(0, "① 첫 예측: 전체 평균"), (1, "② 나무 1개를 더함"),
         (5, "③ 나무 5개"), (100, "④ 나무 100개")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
for ax, (k, title) in zip(axes, steps):
    ax.scatter(x, y, s=6, color="#bbbbbb", alpha=0.7, lw=0)
    if k == 0:
        yy = np.full(len(grid_x), y.mean())
    else:
        m = GradientBoostingRegressor(n_estimators=k, max_depth=2, learning_rate=0.1,
                                      random_state=42).fit(x, y)
        yy = m.predict(grid_x)
    ax.plot(grid_x, yy, color=BLUE, lw=2.2)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("표고(m)", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("침수 피해확률", fontsize=9.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.text(0.5, 0.015, "회색 점: 보험 건물 500채. 표고 하나만 입력한 설명용 모델(나무 깊이 2, 학습률 0.1)",
         ha="center", fontsize=9.5, color=GRAY)
fig.savefig(OUT / "fig_11_3_boosting_steps.png", dpi=150)
plt.close(fig)

# ── 그림 11.4: 무작위 분할과 블록 분할(6절) ──
G = 10
rng = np.random.default_rng(3)
rand_test = rng.random((G, G)) < 0.3
block_test = np.zeros((G, G), dtype=bool)
block_test[:, 7:] = True
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, test, title, note in (
    (axs[0], rand_test, "무작위 분할", "시험 칸 바로 옆 칸이 대부분 학습 칸\n→ 옆 값을 베껴 점수가 높아짐"),
    (axs[1], block_test, "공간 블록 분할", "시험 구역 전체를 학습에서 뺌\n→ 처음 보는 지역을 맞히는 능력"),
):
    for r in range(G):
        for c in range(G):
            ax.add_patch(Rectangle((c, G - 1 - r), 1, 1, lw=0.8, edgecolor="white",
                                   facecolor="#d62728" if test[r, c] else BLUE_L))
    ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(G / 2, -1.3, note, ha="center", va="top", fontsize=10.5, color="#444444")
axs[0].add_patch(Rectangle((0, 0), 0, 0))
fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE_L, label="학습 칸"),
                    Rectangle((0, 0), 1, 1, fc="#d62728", label="시험 칸")],
           loc="upper center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_11_4_random_vs_block.png", dpi=150)
plt.close(fig)

# ── 그림 11.5: 예측구간 폭(7절) ──
fig, ax = plt.subplots(figsize=(8.4, 2.5))
for yv, lo, hi, label, col in ((1, 0.225, 0.375, "건물 가  (σ̂ = 0.05)".replace("σ̂", "예상 오차"), BLUE),
                              (0, 0.12, 0.48, "건물 나  (예상 오차 = 0.12)", ORANGE)):
    ax.plot([lo, hi], [yv, yv], color=col, lw=7, solid_capstyle="butt", alpha=0.55)
    ax.plot([0.30], [yv], "o", color=col, ms=10)
    ax.text(lo - 0.015, yv, f"{lo:.3f}", ha="right", va="center", fontsize=10)
    ax.text(hi + 0.015, yv, f"{hi:.3f}", ha="left", va="center", fontsize=10)
    ax.text(0.0, yv + 0.30, label, fontsize=10.5, color=col, fontweight="bold")
ax.get_children()
ax.text(0.30, 1.62, "점예측 0.30(두 건물 같음)", ha="center", fontsize=10.5,
        bbox=dict(fc="white", ec="none", pad=2))
ax.axvline(0.30, color=GRAY, ls="--", lw=1)
ax.set_xlim(0, 0.62); ax.set_ylim(-0.5, 1.9)
ax.set_yticks([]); ax.set_xlabel("예측 침수확률", fontsize=10)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_11_5_interval_width.png", dpi=150)
plt.close(fig)

# ── 그림 11.6: 치안 시뮬레이션(10절) ──
h = pd.read_csv(PRACTICE / "results" / "predictive_policing_bias.csv")
fig, ax = plt.subplots(figsize=(7.4, 3.3))
ax.plot(h["round"], h["r2_recorded"], "-o", color=BLUE, lw=2.2, label="기록 기준 R² (모델이 기록을 맞힌 점수)")
ax.plot(h["round"], h["corr_patrol_true"], "-o", color=RED, lw=2.2, label="순찰과 실제 위험의 상관")
ax.axvline(0.5, color=GRAY, ls=":", lw=1)
ax.text(0.58, 0.05, "라운드 1부터 피드백 구간", fontsize=9.5, color=GRAY)
ax.annotate(f"{h['r2_recorded'].iloc[-1]:.3f}", (h['round'].iloc[-1], h['r2_recorded'].iloc[-1]),
            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=10, color=BLUE)
ax.annotate(f"{h['corr_patrol_true'].iloc[-1]:.3f}", (h['round'].iloc[-1], h['corr_patrol_true'].iloc[-1]),
            textcoords="offset points", xytext=(0, -15), ha="center", fontsize=10, color=RED)
ax.set_ylim(0, 1.08); ax.set_xlabel("라운드", fontsize=10)
ax.legend(loc="lower right", fontsize=9.5, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_11_6_policing_rounds.png", dpi=150)
plt.close(fig)

# ── 그림 11.7: 완전·관측·대조군 R²(11.2절) ──
log = (PRACTICE / "results" / "11-3-insurance-risk-grading.log").read_text(encoding="utf-8")
m1 = re.search(r"전체 R²\s+([0-9.]+)\s+([0-9.]+)", log)
m2 = re.search(r"무작위 표본 R²:\s*([0-9.]+)\s*\(SD\s*([0-9.]+)", log)
full, obs = float(m1.group(1)), float(m1.group(2))
ctrl, sd = float(m2.group(1)), float(m2.group(2))
fig, ax = plt.subplots(figsize=(6.8, 3.2))
labels = ["완전 모델\n(전체 500채,\n실제 피해확률)", "관측 모델\n(가입자 277채,\n청구 기록)", "대조군\n(무작위 277채,\n청구 기록)"]
vals = [full, obs, ctrl]
bars = ax.bar(labels, vals, color=[GRAY, ORANGE, BLUE], width=0.55)
ax.errorbar([2], [ctrl], yerr=[sd], color="black", capsize=6, lw=1.5, fmt="none")
for k, (b, v) in enumerate(zip(bars, vals)):
    gap = sd + 0.02 if k == 2 else 0.02
    ax.text(b.get_x() + b.get_width() / 2, v + gap, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.annotate("", xy=(2.5, ctrl), xytext=(2.5, full), arrowprops=dict(arrowstyle="<->", color="#444444"))
ax.text(2.58, (full + ctrl) / 2, f"{full - ctrl:.3f}\n표본 감소·\n청구 기록 라벨의 몫", va="center", fontsize=9.5, color="#444444")
ax.text(1.5, 0.80, f"관측 모델과 대조군의 차이 {obs - ctrl:.3f}\n= 가입 선택의 몫(대조군 SD {sd:.3f}보다 작음)",
        ha="center", fontsize=9.5, color="#444444")
ax.set_xlim(-0.5, 3.3)
ax.set_ylim(0, 1.02); ax.set_ylabel("전체 R²", fontsize=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_11_7_selection_control.png", dpi=150)
plt.close(fig)


# ── 그림 11.8: 두 순환 개념도(11.3절) ──
from matplotlib.patches import FancyBboxPatch


def loop(ax, title, labels, face, edge):
    ax.set_xlim(0, 10); ax.set_ylim(0.4, 5.6); ax.axis("off")
    ax.set_title(title, fontsize=11.5, pad=4, color=edge, fontweight="bold")
    pos = [(0.4, 3.5), (5.4, 3.5), (5.4, 0.7), (0.4, 0.7)]   # 왼쪽 아래 좌표, 시계 방향
    w, h = 4.2, 1.5
    for (x0, y0), text in zip(pos, labels):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    fc=face, ec=edge, lw=1.6))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=11)
    arrow = dict(arrowstyle="-|>", color=edge, lw=1.8, mutation_scale=16)
    ax.annotate("", xy=(5.4, 4.25), xytext=(4.6, 4.25), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 2.2), xytext=(7.5, 3.5), arrowprops=arrow)
    ax.annotate("", xy=(4.6, 1.45), xytext=(5.4, 1.45), arrowprops=arrow)
    ax.annotate("", xy=(2.5, 3.5), xytext=(2.5, 2.2), arrowprops=arrow)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.1))
loop(a1, "예측적 치안 — 기록이 늘어나는 순환",
     ["예측 위험 높음", "순찰 배정 증가", "적발 기록 증가", "학습자료에 반영"],
     "#fde0e0", RED)
loop(a2, "홍수보험 — 기록이 사라지는 순환",
     ["예측 위험 과소평가", "상품 설계 안 됨", "가입 감소", "청구 기록 결손"],
     BLUE_L, BLUE)
fig.tight_layout()
fig.savefig(OUT / "fig_11_8_two_loops.png", dpi=150)
plt.close(fig)
print("saved 8 figures")


# ── 그림 11.1: 이웃을 정하는 방법(4절) ──
def grid3(ax, nbr_cells, title):
    ax.set_xlim(-0.1, 3.1); ax.set_ylim(-0.1, 3.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, pad=5)
    for r in range(3):
        for c in range(3):
            if (r, c) == (1, 1):
                face = ORANGE
            elif (r, c) in nbr_cells:
                face = BLUE_L
            else:
                face = "white"
            ax.add_patch(Rectangle((c, 2 - r), 1, 1, fc=face, ec="#555555", lw=1.2))
    ax.text(1.5, 1.5, "나", ha="center", va="center", fontsize=12, color="white", fontweight="bold")


def scatter_panel(ax, seed, title):
    rng_p = np.random.default_rng(seed)
    pts = rng_p.random((22, 2)) * 0.9 + 0.05
    ctr = np.array([0.5, 0.5])
    pts[0] = ctr
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")
    ax.set_title(title, fontsize=11, pad=5)
    return pts, ctr


fig, axs = plt.subplots(2, 3, figsize=(10.5, 4.3))

grid3(axs[0, 0], {(0, 1), (1, 0), (1, 2), (2, 1)}, "① rook: 변을 맞댄 4칸")
grid3(axs[0, 1], {(r, c) for r in range(3) for c in range(3)} - {(1, 1)}, "② queen: 모서리까지 8칸")

pts, ctr = scatter_panel(axs[0, 2], 5, "③ 거리 임계: 반경 d 안의 모든 점")
dist = np.linalg.norm(pts - ctr, axis=1)
inside = (dist <= 0.27) & (dist > 0)
axs[0, 2].add_patch(plt.Circle(ctr, 0.27, fc="none", ec=BLUE, ls="--", lw=1.6))
axs[0, 2].scatter(pts[~inside, 0], pts[~inside, 1], s=30, color="#bbbbbb")
axs[0, 2].scatter(pts[inside, 0], pts[inside, 1], s=46, color=BLUE)
axs[0, 2].scatter(*ctr, s=90, color=ORANGE, zorder=5)

pts, ctr = scatter_panel(axs[1, 0], 11, "④ k-최근접: 가장 가까운 k개(k = 3)")
d2 = np.linalg.norm(pts - ctr, axis=1)
near = np.argsort(d2)[1:4]
mask = np.ones(len(pts), bool); mask[0] = False; mask[near] = False
for j in near:
    axs[1, 0].plot([ctr[0], pts[j, 0]], [ctr[1], pts[j, 1]], color=BLUE, lw=1.4)
axs[1, 0].scatter(pts[mask, 0], pts[mask, 1], s=30, color="#bbbbbb")
axs[1, 0].scatter(pts[near, 0], pts[near, 1], s=46, color=BLUE)
axs[1, 0].scatter(*ctr, s=90, color=ORANGE, zorder=5)

ax = axs[1, 1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color("#999999")
ax.set_title("⑤ 연결망: 도로로 이어진 곳만", fontsize=11, pad=5)
ax.add_patch(Rectangle((0.52, 0), 0.07, 1, fc="#d9d9d9", ec="none"))
ax.text(0.555, 0.04, "철도", ha="center", va="bottom", fontsize=9, color="#666666", rotation=90)
me = np.array([0.40, 0.50])
left = np.array([[0.15, 0.78], [0.18, 0.25], [0.38, 0.88], [0.36, 0.14]])
right = np.array([[0.68, 0.50], [0.82, 0.78], [0.80, 0.22]])
for q in (left[0], left[2]):
    ax.plot([me[0], q[0]], [me[1], q[1]], color=BLUE, lw=1.6)
ax.scatter(left[[0, 2], 0], left[[0, 2], 1], s=46, color=BLUE)
ax.scatter(left[[1, 3], 0], left[[1, 3], 1], s=30, color="#bbbbbb")
ax.scatter(right[:, 0], right[:, 1], s=30, color="#bbbbbb")
ax.annotate("", xy=(0.68, 0.50), xytext=(me[0], me[1]),
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", ls=":", lw=1.4))
ax.text(0.80, 0.585, "직선으론 가까워도\n철도 건너편은 제외", ha="center", fontsize=8.5, color="#777777")
ax.scatter(*me, s=90, color=ORANGE, zorder=5)

ax = axs[1, 2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("⑥ 이웃 정보를 표(W)로 적으면", fontsize=11, pad=5)
names = ["A", "B", "C"]
for k, nm in enumerate(names):
    ax.add_patch(Rectangle((0.3 + k * 1.15, 8.0), 1.0, 1.0, fc=BLUE_L if k != 1 else ORANGE_L, ec="#555555", lw=1.2))
    ax.text(0.8 + k * 1.15, 8.5, nm, ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(4.1, 8.5, "← 한 줄로 놓인 세 칸\n(rook: 맞닿은 칸이 이웃)", va="center", fontsize=8.5, color="#555555")
W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
x0, y0, cw = 2.2, 1.9, 1.35
for j, nm in enumerate(names):
    ax.text(x0 + j * cw + cw / 2, y0 + 3 * 1.3 + 0.35, nm, ha="center", fontsize=11, fontweight="bold")
for i, nm in enumerate(names):
    ax.text(x0 - 0.4, y0 + (2 - i) * 1.3 + 0.65, nm, ha="center", va="center", fontsize=11, fontweight="bold")
    for j in range(3):
        v = W[i][j]
        ax.add_patch(Rectangle((x0 + j * cw, y0 + (2 - i) * 1.3), cw, 1.3,
                               fc=BLUE_L if v else "white", ec="#555555", lw=1.2))
        ax.text(x0 + j * cw + cw / 2, y0 + (2 - i) * 1.3 + 0.65, str(v), ha="center", va="center", fontsize=12)
ax.text(5.0, 0.55, "i행 j열 = j가 i의 이웃이면 1, 아니면 0", ha="center", fontsize=8.5, color="#555555")

fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=9, label="나(기준 위치)"),
                    plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="이웃으로 정함"),
                    plt.Line2D([], [], marker="o", ls="", color="#bbbbbb", ms=7, label="이웃 아님")],
           loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "fig_11_1_neighbor_types.png", dpi=150)
plt.close(fig)
print("saved neighbor figure")
