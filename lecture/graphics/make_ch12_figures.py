"""ch12.md 설명 그림 아홉 장을 만든다. 모두 1~9절 가상 사례용이며 실습 출력이 아니다.

그림 12.2  3절  시 평균과 동별 인구 변화(가상 사례 숫자)
그림 12.3  4절  지방소멸위험지수 등급 구간(제도의 구간값)
그림 12.4  4절  복합 지수를 만드는 다섯 단계(개념도, 숫자 없음)
그림 12.5  4절  가중치를 바꿨을 때 네 군의 순위 변화(가상 사례 숫자)
그림 12.6  5절  서강군과 남포군의 표준화 지표 비교(가상 사례 숫자)
그림 12.7  6절  측정오차 편의: 잡음이 섞이면 기울기가 눕는다(설명용 난수, 실습 출력 아님)
그림 12.8  6절  점예측과 예측구간(가상 사례 숫자)
그림 12.9  7절  이웃 군 하나와 비교 vs 합성 대조군(가상 사례 숫자)
그림 12.10 8절  발표가 쇠퇴를 앞당기는 순환(개념도, 숫자 없음)
그림 12.11 9절  두 철수 선과 세 구간(개념도, 숫자 없음)

실행: python lecture/graphics/make_ch12_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "ch12"
OUT.mkdir(exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, RED, GREEN = "#1f77b4", "#e07b00", "#7f7f7f", "#d62728", "#2ca02c"
BLUE_L, ORANGE_L, LIGHT = "#cfe3f4", "#fbe3c4", "#f2f2f2"
GUNS = ["동산군", "서강군", "남포군", "북원군"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", OUT / name)


def box(ax, x, y, w, h, text, face, size=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor="#555555", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="#555555", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=1.4, connectionstyle=f"arc3,rad={rad}"))


# ── 그림 12.2 시 평균과 동별 변화 ──
fig, ax = plt.subplots(figsize=(9, 2.5))
names = ["신도심", "원도심·농촌", "그 밖의 동", "시 전체"]
before, after = [3, 6, 3, 12], [5, 4, 3, 12]
x = np.arange(len(names))
ax.bar(x - 0.18, before, 0.36, color=LIGHT, edgecolor=GRAY, label="10년 전")
ax.bar(x + 0.18, after, 0.36, color=[BLUE, RED, GRAY, GRAY], label="지금")
for i, (b, a) in enumerate(zip(before, after)):
    d = a - b
    ax.text(i + 0.18, a + 0.3, f"{'+' if d > 0 else ''}{d}만" if d else "변화 없음",
            ha="center", fontsize=10.5, color=RED if d < 0 else (BLUE if d > 0 else GRAY),
            fontweight="bold")
ax.axvline(2.5, color=GRAY, ls=":", lw=1)
ax.set_xticks(x, names, fontsize=11)
ax.set_ylabel("인구(만 명)")
ax.set_ylim(0, 14.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_12_2_city_average.png")

# ── 그림 12.3 소멸위험지수 등급 구간 ──
fig, ax = plt.subplots(figsize=(9, 1.9))
bands = [(0, 0.2, "소멸 고위험", RED), (0.2, 0.5, "소멸위험 진입", ORANGE),
         (0.5, 1.0, "주의", "#f2c94c"), (1.0, 1.5, "보통 이상", GREEN)]
for lo, hi, lab, c in bands:
    ax.add_patch(plt.Rectangle((lo, 0), hi - lo, 1, color=c, alpha=0.75))
    ax.text((lo + hi) / 2, 0.5, lab, ha="center", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([0, 0.2, 0.5, 1.0, 1.5], ["0", "0.2", "0.5", "1.0", "1.5 이상"])
ax.set_xlabel("지방소멸위험지수 = (20~39세 여성 인구) ÷ (65세 이상 인구)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
save(fig, "fig_12_3_risk_index_bands.png")

# ── 그림 12.4 복합 지수 다섯 단계 ──
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.3)
ax.axis("off")
steps = [("1 지표 고르기", "인구·생활·경제\n지표 넷"),
         ("2 방향 맞추기", "클수록 취약하도록\n부호를 맞춤"),
         ("3 표준화", "분·명·개를\n같은 척도로"),
         ("4 가중치", "지표마다 비중\n(합 = 1)"),
         ("5 합치기", "가중합 =\n복합 점수")]
for i, (t, d) in enumerate(steps):
    x0 = 0.1 + i * 2.0
    box(ax, x0, 1.25, 1.6, 0.75, t, BLUE_L, size=11, bold=True)
    ax.text(x0 + 0.8, 0.75, d, ha="center", va="center", fontsize=10, color="#333333")
    if i < 4:
        arrow(ax, x0 + 1.62, 1.62, x0 + 1.98, 1.62)
ax.text(5.0, 0.1, "방향 맞추기를 빼먹어도 계산은 오류 없이 끝남 → 1위와 꼴찌 지역의 원래 값을 직접 확인",
        ha="center", fontsize=10, color=RED)
save(fig, "fig_12_4_index_steps.png")

# ── 그림 12.5 가중치를 바꿨을 때 순위 ──
X = np.array([[0.42, -2.8, 38, 22], [0.75, -0.5, 41, 30],
              [0.50, -2.2, 22, 55], [1.20, 1.4, 16, 70]])
D = X * np.array([-1, -1, 1, -1])
Z = (D - D.mean(0)) / D.std(0)
w_a = np.array([0.35, 0.20, 0.25, 0.20])
w_b = np.array([0.45, 0.20, 0.15, 0.20])
s_a, s_b = Z @ w_a, Z @ w_b
r_a = (-s_a).argsort().argsort() + 1
r_b = (-s_b).argsort().argsort() + 1
fig, ax = plt.subplots(figsize=(8, 2.6))
colors = [GRAY, ORANGE, BLUE, GRAY]
for i, g in enumerate(GUNS):
    lw = 3 if r_a[i] != r_b[i] else 1.5
    ax.plot([0, 1], [r_a[i], r_b[i]], "-o", color=colors[i], lw=lw, ms=8)
    ax.text(-0.04, r_a[i], f"{g} {s_a[i]:.2f}", ha="right", va="center", fontsize=11)
    ax.text(1.04, r_b[i], f"{g} {s_b[i]:.2f}", ha="left", va="center", fontsize=11)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(4.5, 0.5)
ax.set_xticks([0, 1], ["원래 가중치\n소멸위험 0.35 · 서비스 0.25", "바꾼 가중치\n소멸위험 0.45 · 서비스 0.15"])
ax.set_yticks([1, 2, 3, 4], ["1위", "2위", "3위", "4위"])
ax.spines[["top", "right", "bottom"]].set_visible(False)
save(fig, "fig_12_5_weight_rank.png")

# ── 그림 12.6 서강군과 남포군의 결손 비교 ──
labels = ["소멸위험", "청년 유출", "서비스 거리", "사업체 부족"]
fig, ax = plt.subplots(figsize=(9, 2.4))
xx = np.arange(4)
ax.bar(xx - 0.2, Z[1], 0.4, color=ORANGE, label=f"서강군 (점수 {s_a[1]:.2f})")
ax.bar(xx + 0.2, Z[2], 0.4, color=BLUE, label=f"남포군 (점수 {s_a[2]:.2f})")
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(xx, labels, fontsize=11)
ax.set_ylabel("표준화 값\n(클수록 취약)")
ax.annotate("서비스가 가장 멂", (2 - 0.2, Z[1][2]), (1.2, 1.25), fontsize=10.5, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.annotate("청년이 빠져나감", (1 + 0.2, Z[2][1]), (-0.3, 1.25), fontsize=10.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.set_ylim(-1, 1.55)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=10)
save(fig, "fig_12_6_two_profiles.png")

# ── 그림 12.7 측정오차 편의(설명용 난수) ──
rng = np.random.default_rng(12)
n = 150
true_x = rng.normal(0, 1, n)
y = -1.0 * true_x + rng.normal(0, 0.5, n)
noisy_x = true_x + rng.normal(0, 0.5, n)
fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
for ax, xv, title, c in [(axes[0], true_x, "실제 경제활동(볼 수 없음)", GRAY),
                         (axes[1], noisy_x, "야간조명 = 실제 + 잡음", BLUE)]:
    ax.scatter(xv, y, s=12, color=c, alpha=0.55)
    b = np.polyfit(xv, y, 1)
    xs = np.linspace(-3, 3, 10)
    ax.plot(xs, np.polyval(b, xs), color=RED, lw=2.2, label=f"기울기 {b[0]:.2f}")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("경제활동(표준화)")
    ax.set_xlim(-3.2, 3.2)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("쇠퇴율")
save(fig, "fig_12_7_attenuation.png")

# ── 그림 12.8 점예측과 예측구간 ──
fig, ax = plt.subplots(figsize=(8.5, 2.2))
for i, (g, p, lo, hi, c) in enumerate([("동산군 (1순위)", 10.0, 7.0, 13.0, RED),
                                       ("서강군 (2순위)", 9.5, 8.5, 10.5, BLUE)]):
    yy = 1 - i
    ax.plot([lo, hi], [yy, yy], color=c, lw=6, alpha=0.35, solid_capstyle="butt")
    ax.plot(p, yy, "o", color=c, ms=10)
    ax.text(p, yy + 0.28, f"예측 {p}%", ha="center", fontsize=10.5, color=c)
    ax.text(hi + 0.15, yy, f"90% 구간 {lo}~{hi}%", va="center", fontsize=10.5, color=c)
ax.axvline(9.5, color=BLUE, ls=":", lw=1)
ax.annotate("동산군 구간 하한 7.0%가\n서강군 예측 9.5%보다 낮음", (7.0, 1), (5.2, 0.15),
            fontsize=10, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.set_yticks([1, 0], ["동산군 (1순위)", "서강군 (2순위)"], fontsize=11)
ax.set_xlim(5, 16)
ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("다음 5년 인구 감소율(%)")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_12_8_prediction_interval.png")

# ── 그림 12.9 이웃 군 하나 vs 합성 대조군 ──
years = [1, 2, 3, 4]
dong = [100, 101, 102, 108]
a = [90, 91, 92, 92]
b = [110, 111, 112, 112]
syn = [(p + q) / 2 for p, q in zip(a, b)]
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
ax = axes[0]
ax.plot(years, dong, "-o", color=RED, lw=2.2, label="동산군(기금 받음)")
ax.plot(years, a, "-o", color=GRAY, label="A군")
ax.plot(years, b, "-s", color=GRAY, label="B군")
ax.text(4.08, 92, "A와 비교: +16", va="center", fontsize=10, color=GRAY)
ax.text(4.08, 112, "B와 비교: -4", va="center", fontsize=10, color=GRAY)
ax.set_title("(가) 이웃 군 하나와 비교 — 고른 군에 따라 결론이 바뀜", fontsize=11)
ax2 = axes[1]
ax2.plot(years, dong, "-o", color=RED, lw=2.2, label="동산군(기금 받음)")
ax2.plot(years, syn, "--o", color=BLUE, lw=2, label="합성 동산군 = A 0.5 + B 0.5")
ax2.annotate("", (4, 108), (4, 102), arrowprops=dict(arrowstyle="<->", color="#333333"))
ax2.text(4.08, 105, "효과 +6", va="center", fontsize=11, fontweight="bold")
ax2.set_title("(나) 합성 대조군 — 투입 전 궤적을 따라가게 섞음", fontsize=11)
for axx in axes:
    axx.axvspan(3.5, 4.4, color=LIGHT)
    axx.text(3.95, 87.5, "기금 투입 후", ha="center", fontsize=9.5, color=GRAY)
    axx.set_xticks(years, ["1년차", "2년차", "3년차", "4년차"])
    axx.set_xlim(0.8, 5.1)
    axx.set_ylim(86, 115)
    axx.spines[["top", "right"]].set_visible(False)
    axx.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
axes[0].set_ylabel("인구 지수")
save(fig, "fig_12_9_synthetic_control.png")

# ── 그림 12.10 발표가 쇠퇴를 앞당기는 순환 ──
fig, ax = plt.subplots(figsize=(9, 2.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
nodes = [(0.2, 1.8, "\"소멸 1순위\"\n발표"), (2.7, 1.8, "대출 축소·\n투자 보류·이주"),
         (5.2, 1.8, "실제 쇠퇴가\n빨라짐"), (7.7, 1.8, "통계가 예측과\n일치")]
for x0, y0, t in nodes:
    box(ax, x0, y0, 2.0, 0.9, t, ORANGE_L, size=10.5)
for i in range(3):
    arrow(ax, nodes[i][0] + 2.02, 2.25, nodes[i + 1][0] - 0.02, 2.25)
arrow(ax, 8.7, 1.78, 1.2, 1.78, color=RED, rad=-0.25)
ax.text(5.0, 0.35, "\"예측이 맞았다\"로 읽히지만, 일치에는 발표가 앞당긴 몫이 섞여 있음",
        ha="center", fontsize=10.5, color=RED)
save(fig, "fig_12_10_publication_loop.png")

# ── 그림 12.11 두 철수 선과 세 구간 ──
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.3)
ax.axis("off")
zones = [(0.3, 3.5, "즉시 철수", "버티는 손실 > 지금 닫는 비용", RED),
         (3.5, 6.3, "만료 시 철수", "적자지만 위약금보다는 작음\n→ 계약 끝까지 버팀", ORANGE),
         (6.3, 9.7, "유지", "흑자", GREEN)]
for lo, hi, t, d, c in zones:
    ax.add_patch(plt.Rectangle((lo, 1.0), hi - lo, 0.7, color=c, alpha=0.35))
    ax.text((lo + hi) / 2, 1.35, t, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text((lo + hi) / 2, 0.5, d, ha="center", va="center", fontsize=10)
for xv, lab in [(3.5, "즉시 철수 선"), (6.3, "만료 시 철수 선(손익분기)")]:
    ax.plot([xv, xv], [0.95, 1.9], color="#333333", lw=2)
    ax.text(xv, 2.0, lab, ha="center", fontsize=10.5)
arrow(ax, 0.3, 0.05, 9.7, 0.05, color=GRAY)
ax.text(9.7, 0.18, "점포가 나눠 갖는 상권이 큼 →", ha="right", fontsize=9.5, color=GRAY)
save(fig, "fig_12_11_exit_zones.png")
