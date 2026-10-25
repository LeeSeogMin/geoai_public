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

# 그림 2: 두 반의 이중차분 (3.1절)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
x = [0, 1]
ax.plot(x, [60, 75], "o-", color=BLUE, lw=2.2, label="1반(보충수업 함)")
ax.plot(x, [70, 75], "o-", color=RED, lw=2.2, label="2반(안 함)")
ax.plot(x, [60, 65], "o--", color=BLUE, lw=1.6, alpha=0.7, label="1반이 수업을 안 들었다면(2반처럼 +5)")
ax.annotate("", xy=(1.04, 75), xytext=(1.04, 65), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(1.07, 70, "이중차분\n+10", va="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["학기 초", "학기 말"])
ax.set_xlim(-0.15, 1.45)
ax.set_ylim(55, 80)
ax.set_ylabel("반 평균 점수(예시 숫자)")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("1반의 +15 중 +5는 2반도 오른 몫")
save(fig, "fig_ex_2_did_two_classes.png")

# 그림 3: 실습 1의 이중차분과 진짜 효과 (5.1절)
pre_t, post_t = 3.129, 4.040
pre_c, post_c = 2.717, 2.609
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
