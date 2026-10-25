"""ch09-ex.md 설명용 그림 네 장을 만든다.

그림의 숫자는 모두 ch09-ex.md에 이미 적힌 값이다.
- 그림 1·2·4: 원고의 예시 숫자(보충수업·두 반·세 학교)
- 그림 3: lecture_practice/chapter9/results/9-1-nightlight-did-dml.log 에 찍힌 값과 진짜 ATT 0.50

실행: python lecture/graphics/make_ch09_ex_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent / "ch09"
OUT.mkdir(exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, RED, GRAY = "#1f77b4", "#d62728", "#7f7f7f"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# 그림 1: 학생 6명의 관측값과 관측되지 않는 값 (1.2절)
students = ["A", "B", "C", "D", "E", "F"]
with_class = [80, 75, 70, 85, 90, 60]      # 들었을 때 점수
without_class = [70, 67, 64, 83, 89, 57]   # 안 들었을 때 점수
took = [True, True, True, False, False, False]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, (w, wo, t) in enumerate(zip(with_class, without_class, took)):
    ax.plot([i, i], [wo, w], color=GRAY, lw=1, zorder=1)
    obs_y, hid_y = (w, wo) if t else (wo, w)
    obs_c, hid_c = (BLUE, RED) if t else (RED, BLUE)
    ax.scatter(i, obs_y, s=90, color=obs_c, zorder=3)
    ax.scatter(i, hid_y, s=90, facecolors="white", edgecolors=hid_c, lw=1.8, zorder=3)
ax.set_xticks(range(6))
ax.set_xticklabels([f"{s}\n({'들음' if t else '안 들음'})" for s, t in zip(students, took)])
ax.set_ylabel("시험 점수(예시 숫자)")
ax.set_ylim(50, 95)
ax.scatter([], [], s=90, color=BLUE, label="관측됨: 보충수업을 들은 쪽의 점수")
ax.scatter([], [], s=90, color=RED, label="관측됨: 안 들은 쪽의 점수")
ax.scatter([], [], s=90, facecolors="white", edgecolors=GRAY, lw=1.8, label="관측되지 않음(속이 빈 점)")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
ax.set_title("학생마다 두 점 중 한 점만 보인다")
save(fig, "fig_ex_1_observed_vs_hidden.png")

# 그림 2: 두 동네의 이중차분 (3.1절)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
x = [0, 1]
ax.plot(x, [100, 130], "o-", color=BLUE, lw=2.2, label="A동(지역화폐 도입)")
ax.plot(x, [90, 105], "o-", color=RED, lw=2.2, label="B동(미도입)")
ax.plot(x, [100, 115], "o--", color=BLUE, lw=1.6, alpha=0.7, label="A동이 도입하지 않았다면(B동처럼 15 증가)")
ax.annotate("", xy=(1.04, 130), xytext=(1.04, 115), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(1.07, 122.5, "이중차분\n15", va="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["도입 전", "도입 후"])
ax.set_xlim(-0.15, 1.45)
ax.set_ylim(80, 140)
ax.set_ylabel("점포당 월평균 매출(만원, 예시 숫자)")
ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("A동이 30 올랐지만 그중 15는 B동도 오른 몫")
save(fig, "fig_ex_2_did_two_towns.png")

# 그림 3: 도입 전 추세로 평행추세를 살펴봄 (3.2절)
GREEN = "green"
t4 = [0, 1, 2, 3]
B_path = [60, 75, 90, 105]
A_parallel = [70, 85, 100, 130]   # 도입 전 A동이 B동과 같은 폭(15)으로 오름
A_diverge = [50, 75, 100, 130]    # 도입 전 A동이 B동보다 빠르게(25) 오름
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, a_path, broken in zip(axes, (A_parallel, A_diverge), (False, True)):
    ax.plot(t4, a_path, "o-", color=BLUE, lw=2.2, label="A동(도입) 실제")
    ax.plot(t4, B_path, "o-", color=RED, lw=2.2, label="B동(미도입) 실제")
    ax.axvline(2.5, color=GRAY, ls="--", lw=1)
    ax.text(2.45, 143, "도입", ha="right", color=GRAY, fontsize=9)
    ax.plot([2, 3], [100, 115], "o--", color=BLUE, lw=1.6, alpha=0.6, label="B동의 변화폭(15)을 따랐다면")
    if broken:
        ax.plot([2, 3], [100, 125], "o:", color=GREEN, lw=2.2, label="도입 전 A동의 변화폭(25)을 이었다면")
        ax.set_title("도입 전 추세가 나란하지 않음: A동이 원래 더 빠르게 오름", fontsize=10)
    else:
        ax.set_title("도입 전 추세가 나란함: 두 동네의 간격이 10으로 일정", fontsize=10)
    ax.set_xticks(t4)
    ax.set_xticklabels(["3기 전", "2기 전", "1기 전\n(도입 직전)", "도입 후"])
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(40, 150)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.93))
axes[0].set_ylabel("점포당 월평균 매출(만원, 예시 숫자)")
save(fig, "fig_ex_2a_pre_trends.png")

# 그림 4: 평행추세 가정이 맞는 경우와 깨진 경우 (3.2절). 관측값은 두 칸이 같다
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, broken in zip(axes, (False, True)):
    ax.plot(x, [100, 130], "o-", color=BLUE, lw=2.2, label="A동(도입) 실제")
    ax.plot(x, [90, 105], "o-", color=RED, lw=2.2, label="B동(미도입) 실제")
    ax.plot(x, [100, 115], "o--", color=BLUE, lw=1.6, alpha=0.6, label="평행추세를 가정한 A동 경로")
    ax.annotate("", xy=(1.04, 130), xytext=(1.04, 115), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.07, 122.5, "이중차분\n15", va="center", fontsize=9.5, fontweight="bold")
    if broken:
        ax.plot(x, [100, 125], "o:", color=GREEN, lw=2.2, label="A동의 진짜 반사실(도입 없이 25 증가)")
        ax.annotate("", xy=(0.96, 130), xytext=(0.96, 125), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
        ax.text(0.96, 134.5, "진짜 효과 5", va="center", ha="right", fontsize=9.5, color=GREEN, fontweight="bold")
        ax.set_title("평행추세가 깨진 경우: 이중차분 15, 진짜 효과 5", fontsize=10)
    else:
        ax.set_title("평행추세가 맞는 경우: 이중차분 15 = 진짜 효과 15", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["도입 전", "도입 후"])
    ax.set_xlim(-0.15, 1.5)
    ax.set_ylim(80, 140)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("점포당 월평균 매출(만원, 예시 숫자)")
save(fig, "fig_ex_2b_parallel_trends.png")

# 그림 4: 실습 1의 이중차분과 진짜 효과 (5.1절)
pre_t, post_t = 3.229, 4.140
pre_c, post_c = 2.617, 2.509
parallel_cf = pre_t + (post_c - pre_c)   # 평행추세를 가정한 반사실
true_cf = post_t - 0.50                  # 진짜 ATT 0.50을 빼서 얻은 실제 반사실

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(x, [pre_t, post_t], "o-", color=BLUE, lw=2.2, label="처치(사업)")
ax.plot(x, [pre_c, post_c], "o-", color=RED, lw=2.2, label="대조(사업 없음)")
ax.plot(x, [pre_t, parallel_cf], "o--", color=BLUE, alpha=0.6, label="평행추세를 가정한 반사실")
ax.plot(x, [pre_t, true_cf], "o:", color="green", lw=2, label="진짜 반사실(진짜 ATT 0.50을 뺀 값)")
ax.annotate("", xy=(1.04, post_t), xytext=(1.04, parallel_cf), arrowprops=dict(arrowstyle="<->", lw=1.4))
ax.text(1.07, (post_t + parallel_cf) / 2, "이중차분\n+1.019", va="center", fontsize=9.5, fontweight="bold")
ax.annotate("", xy=(0.96, post_t), xytext=(0.96, true_cf), arrowprops=dict(arrowstyle="<->", color="green", lw=1.4))
ax.text(0.93, (post_t + true_cf) / 2, "진짜 효과\n+0.50", va="center", ha="right", fontsize=9.5, color="green", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["사전(2016-2019)", "사후(2020-2023)"])
ax.set_xlim(-0.15, 1.5)
ax.set_ylabel("야간조명(로그 radiance)")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("이중차분 +1.019가 진짜 효과 +0.50보다 큰 이유")
save(fig, "fig_ex_3_did_vs_truth.png")

# 그림 4: 세 학교의 변화량 구성과 비교 대상 (6.1절)
schools = ["가\n(먼저 시작)", "나\n(나중 시작)", "다\n(안 함)"]
common = [2, 2, 2]
effect = [2, 3, 0]
effect_label = ["효과 증가분 +2", "첫 학기 효과 +3", ""]

fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.bar(schools, common, color="#bbbbbb", label="공통 변화 +2")
ax.bar(schools, effect, bottom=common, color=BLUE, label="보충수업 효과")
for i, (c, e, t) in enumerate(zip(common, effect, effect_label)):
    ax.text(i, c / 2, "+2", ha="center", va="center")
    if e:
        ax.text(i, c + e / 2, t, ha="center", va="center", color="white", fontsize=9)
    ax.text(i, c + e + 0.12, f"+{c + e}", ha="center", fontweight="bold")
ax.set_ylim(0, 6.6)
ax.set_ylabel("이번 학기 점수 변화(예시 숫자)")
ax.set_title("나 학교의 진짜 효과는 +3\n다와 비교: 5-2=+3(정답), 가와 비교: 5-4=+1(오답)", fontsize=10.5)
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "fig_ex_4_comparison_group.png")

print("저장:", *sorted(p.name for p in OUT.glob("fig_ex_*.png")), sep="\n  ")
