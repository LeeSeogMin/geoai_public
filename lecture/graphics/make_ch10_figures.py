"""ch10.md 그림 세 장을 만든다.

그림 1은 6.2절의 시간 블록 검증 설명도(10-2 코드의 분할 규칙을 그대로 옮김)이고,
그림 2는 9.2절 표의 예시 숫자로 그린 경계 비교 설명도이고,
그림 3은 실습(10-2)이 저장한 results/complaint_response_priority_uncertainty.csv로 그린다.
먼저 10-2 코드를 실행해 results/를 만든 뒤 이 파일을 실행한다.

실행: python lecture/graphics/make_ch10_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
PRACTICE = HERE.parent.parent / "lecture_practice" / "chapter10"
RESULTS = PRACTICE / "results"
OUT = HERE / "ch10"
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


# 그림 1: 시간 블록 검증과 무작위 분할의 차이 (6.2절). 10-2 코드의 규칙:
# 예측 대상 주 13~39주(피처는 12~38주), 피처 주 t <= 30이 학습, t > 30이 평가
WEEKS, BURN, CUT = 40, 12, 30
target_weeks = np.arange(BURN + 1, WEEKS)          # 13..39
rng_split = np.random.default_rng(7)
is_test_random = rng_split.random(len(target_weeks)) < 0.3
GREEN_GRAY = "#d9d9d9"

fig, ax = plt.subplots(figsize=(9.5, 3.9))
def strip(y, colors, label):
    for w, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((w, y), 0.92, 0.8, color=c, lw=0))
    ax.text(-0.6, y + 0.4, label, ha="right", va="center", fontsize=10)
rows = []
for w in range(WEEKS):
    if w <= BURN:
        rows.append(GREEN_GRAY)
    else:
        rows.append(BLUE if (w - 1) <= CUT else RED)   # 대상 주 w의 피처 주는 w-1
strip(1.4, rows, "시간 블록 검증" + chr(10) + "(이 장의 방법)")
rows2 = []
for w in range(WEEKS):
    if w <= BURN:
        rows2.append(GREEN_GRAY)
    else:
        rows2.append(RED if is_test_random[w - BURN - 1] else BLUE)
strip(0.2, rows2, "무작위 분할" + chr(10) + "(잘못된 방법)")
ax.text(BURN + 0.46 + (CUT - BURN) / 2, 2.35, "학습: 13~31주", ha="center", color=BLUE, fontsize=10)
ax.text(32 + 3.5, 2.35, "평가: 32~39주", ha="center", color=RED, fontsize=10)
ax.text(6, 2.35, "분석에서 뺌(0~12주)", ha="center", color=GRAY, fontsize=10)
ax.text(26, -0.35, "평가 주(빨강)의 앞뒤에 학습 주(파랑)가 섞여 있음 → 미래 자료로 과거를 맞히게 됨",
        ha="center", fontsize=9, color=RED)
ax.set_xlim(-0.5, WEEKS + 0.3)
ax.set_ylim(-0.7, 2.8)
ax.set_xticks([0, 10, 20, 30, 39])
ax.set_xlabel("예측 대상 주(week)")
ax.set_yticks([])
for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)
save(fig, "fig_10_1_time_block_cv.png")

# 그림 2: 경계선 양쪽 비교 (9.2절 표의 예시 숫자). x < 0 규제지역, x > 0 비규제지역
reg_x, reg_y = [-1.0, -0.5, -0.02], [10.1, 9.9, 9.7]       # 규제지역 쪽(억)
non_x, non_y = [0.02, 0.5, 1.0], [10.0, 9.8, 9.6]          # 비규제지역 쪽(억)
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ax.axvspan(-1.25, 0, color=RED, alpha=0.06)
ax.axvline(0, color="black", ls="--", lw=1)
ax.plot(reg_x, reg_y, "o-", color=RED, lw=2, ms=8, label="규제지역 쪽 거래 평균")
ax.plot(non_x, non_y, "o-", color=BLUE, lw=2, ms=8, label="비규제지역 쪽 거래 평균")
ax.plot([0.02, -0.5, -1.0], [10.0, 10.2, 10.4], ":", color=BLUE, lw=1.6,
        label="규제가 없었다면(비규제 쪽 추세를 연장)")
ax.axhline(9.9, xmin=0.02, xmax=0.48, color=RED, lw=1, alpha=0.5)
ax.axhline(9.8, xmin=0.52, xmax=0.98, color=BLUE, lw=1, alpha=0.5)
ax.text(-1.2, 9.86, "규제지역 평균 9.9억", color=RED, fontsize=9, va="top")
ax.text(0.66, 9.83, "비규제지역 평균 9.8억", color=BLUE, fontsize=9, va="bottom")
ax.annotate("", xy=(0.0, 9.7), xytext=(0.0, 10.0),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(0.07, 9.6, "경계에서의 차이" + chr(10) + "-0.3억 = 규제 효과", fontsize=10, va="center")
ax.text(-0.62, 10.48, "규제지역", color=RED, fontsize=11, ha="center")
ax.text(0.62, 10.48, "비규제지역", color=BLUE, fontsize=11, ha="center")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(9.45, 10.6)
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels(["1km", "0.5km", "경계", "0.5km", "1km"])
ax.set_xlabel("경계까지 거리")
ax.set_ylabel("평균 거래가(억, 예시 숫자)")
ax.legend(loc="lower left", fontsize=8.5, frameon=False)
save(fig, "fig_10_2_boundary_comparison.png")

# 그림 3: 권역별 다음 주 예측과 예측구간 반폭 (11.3절)
unc = pd.read_csv(RESULTS / "complaint_response_priority_uncertainty.csv")
color = {"높음": BLUE, "중간": GRAY, "낮음": RED}
# 아래쪽에 몰린 여섯 권역의 이름이 겹치지 않도록 점마다 글자 위치를 따로 둔다
LABEL_OFFSET = {3: (10, -14), 5: (-18, -22), 2: (-34, 16), 6: (-4, -26), 4: (-6, 26), 7: (22, 14)}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for _, r in unc.iterrows():
    ax.scatter(r["pred_next_week_complaints"], r["mean_pi_half"], s=150,
               color=color[r["confidence"]], zorder=3)
    ax.annotate(f"권역 {int(r['region'])}", (r["pred_next_week_complaints"], r["mean_pi_half"]),
                textcoords="offset points", xytext=LABEL_OFFSET.get(int(r["region"]), (8, 5)),
                fontsize=9, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
for k, c in color.items():
    ax.scatter([], [], s=70, color=c, label=f"신뢰등급 {k}")
ax.set_xlabel("다음 주 예측 민원(권역 합계, 건)")
ax.set_ylabel("90% 예측구간 평균 반폭(격자당 건)")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.25)
save(fig, "fig_10_3_forecast_halfwidth.png")

print("저장:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
