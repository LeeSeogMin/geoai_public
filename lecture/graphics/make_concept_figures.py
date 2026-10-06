"""6장 개념 그림 다섯 장을 만든다.

만드는 파일 (이 스크립트와 같은 폴더)
  mask_to_decision.svg   3.2  마스크를 구역·객체·후보표로 옮기는 세 경로
  label_raster.svg       7.1  참·거짓 격자와 덩어리 번호 격자, 4-이웃과 8-이웃
  crack_perimeter.svg    7.3  크랙 둘레와 경계 칸 세기의 차이
  compactness.svg        7.3  면적이 같고 모양만 다른 네 도형의 조밀도
  min_width.svg          7.3  거리변환으로 최소폭을 재는 절차

다섯 장은 모두 **설명용 예시**이고 실습 자료를 그린 것이 아니다. 다만 그림에 적힌
수치는 손으로 적은 것이 아니라 이 스크립트가 계산한 값이다 — 조밀도는 4piA/P^2을
직접 계산하고, 최소폭은 `scipy.ndimage.distance_transform_edt`를 예시 블록에 걸어
얻는다. 그래서 수치를 의심할 때는 이 스크립트를 돌려 출력과 대조하면 된다.

예외가 하나 있다. mask_to_decision.svg의 상자에 적힌 값(17.3ha, 40건, 8건 등)은
이 스크립트가 계산한 값이 아니라 `lecture_practice/chapter6/results/`의 실행 로그에서
옮겨 적은 값이다. 로그가 바뀌면 아래 THREE_PATHS의 `val` 줄을 함께 고친다.

수치나 모양을 고칠 때는 SVG를 직접 손대지 말고 이 스크립트를 고쳐 다시 돌린다.

실행
    python lecture/graphics/make_concept_figures.py
"""

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

OUT_DIR = Path(__file__).parent

# 색은 unet.svg 계열을 따른다 (진한 색 = 선·글자, 연한 색 = 면)
BLUE, BLUE_L = "#2563eb", "#bfdbfe"
GREEN, GREEN_L = "#15803d", "#bbf7d0"
ORANGE, ORANGE_L = "#d97706", "#fde68a"
RED, RED_L = "#dc2626", "#fecaca"
GRID, GRAY_T = "#cbd5e1", "#667085"

CELL = 36          # crack_perimeter의 칸 크기
LABEL_CELL = 32    # label_raster의 칸 크기 (가운데 화살표 자리를 비우려고 작게 씀)


def write_svg(name: str, text: str) -> Path:
    path = OUT_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


# ──────────────────────────────── 3.2 세 경로 ────────────────────────────────

THREE_PATHS = [
    dict(x=15, head="구역별 면적", hf="#dbeafe", hc="#175cd3",
         how=["마스크를 행정구역 격자에 포개어", "구역마다 변화 픽셀을 합산"],
         val=["16구역(4×4) 중 15구역에서 변화", "합계 17.3ha · 1위 구역 9 = 2.93ha"],
         ask=["→ 어느 구역을 먼저 점검할까", "(2절)"]),
    dict(x=315, head="객체 목록", hf="#dcfce7", hc="#15803d",
         how=["맞닿은 픽셀을 연결요소로 묶고", "덩어리마다 면적·최소폭·접도를 잼"],
         val=["연결요소 949개 중 909개는 얼룩", "제외하고 분석 대상 40건"],
         ask=["→ 무엇을 하나로 셀까", "(7절)"]),
    dict(x=615, head="후보표", hf="#fef3c7", hc="#b45309",
         how=["객체 목록에 개발 요건 부등식을", "걸어 통과한 것만 남김"],
         val=["40건 → 면적 20건 → 접도 9건", "→ 최소폭 8건"],
         ask=["→ 현장 실사를 어디로 보낼까", "(10절)"]),
]


def build_mask_to_decision() -> Path:
    w, by, bh = 270, 165, 180
    parts = []
    for col in THREE_PATHS:
        cx = col["x"] + w / 2
        parts.append(f'<rect x="{col["x"]}" y="{by}" width="{w}" height="{bh}" rx="6" '
                     f'fill="#ffffff" stroke="{col["hc"]}" stroke-width="1.5"/>')
        parts.append(f'<path d="M {col["x"]} {by + 6} a 6 6 0 0 1 6 -6 h {w - 12} '
                     f'a 6 6 0 0 1 6 6 v 26 h -{w} z" fill="{col["hf"]}"/>')
        parts.append(f'<text x="{cx}" y="{by + 24}" font-size="15" font-weight="bold" '
                     f'text-anchor="middle" fill="{col["hc"]}">{col["head"]}</text>')
        y = by + 56
        for line in col["how"]:
            parts.append(f'<text x="{cx}" y="{y}" font-size="12" text-anchor="middle" fill="#334155">{line}</text>')
            y += 18
        y += 10
        for line in col["val"]:
            parts.append(f'<text x="{cx}" y="{y}" font-size="12" text-anchor="middle" fill="{GRAY_T}">{line}</text>')
            y += 18
        y += 12
        parts.append(f'<text x="{cx}" y="{y}" font-size="12.5" font-weight="bold" '
                     f'text-anchor="middle" fill="{col["hc"]}">{col["ask"][0]}</text>')
        parts.append(f'<text x="{cx}" y="{y + 18}" font-size="11.5" text-anchor="middle" '
                     f'fill="{GRAY_T}">{col["ask"][1]}</text>')
        parts.append(f'<path d="M 450 113 V 138 H {cx} V {by - 4}" fill="none" '
                     f'stroke="{GRAY_T}" stroke-width="1.5"/>')
        parts.append(f'<path d="M {cx} {by} l -5 -9 l 10 0 z" fill="{GRAY_T}"/>')

    body = "\n".join("  " + p for p in parts)
    return write_svg("mask_to_decision.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400" font-family="sans-serif">
  <text x="450" y="28" font-size="18" font-weight="bold" text-anchor="middle">마스크를 결정 단위로 옮기는 세 경로</text>

  <rect x="350" y="52" width="200" height="61" rx="6" fill="#f8fafc" stroke="{GRAY_T}" stroke-width="1.5"/>
  <text x="450" y="76" font-size="14" font-weight="bold" text-anchor="middle">마스크</text>
  <text x="450" y="98" font-size="11.5" text-anchor="middle" fill="{GRAY_T}">픽셀마다 대상인지 아닌지를 적은 격자</text>

{body}

  <text x="450" y="378" font-size="12" text-anchor="middle" fill="{GRAY_T}">셋 모두 픽셀을 세어 사람이 세는 단위로 바꾸는 계산임. 어느 경로를 쓸지 착수 시점에 정함(3.2)</text>
</svg>
''')


# ─────────────────────────────── 7.1 덩어리 번호 ───────────────────────────────

BLOBS = {1: [(1, 0), (2, 0), (1, 1), (2, 1)],
         2: [(4, 1), (5, 1), (5, 2)],
         3: [(1, 3), (2, 3), (3, 3), (2, 4)]}
CELL2BLOB = {c: b for b, cs in BLOBS.items() for c in cs}
BLOB_FILL = {1: BLUE_L, 2: GREEN_L, 3: ORANGE_L}
BLOB_EDGE = {1: BLUE, 2: GREEN, 3: ORANGE}

# 모서리 하나로만 이어진 두 덩어리 (4-이웃이면 둘, 8-이웃이면 하나)
CORNER = {(0, 0): "P", (1, 0): "P", (2, 1): "Q", (3, 1): "Q"}


def _label_grid(ox, oy, cols, rows, mode):
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = ox + c * LABEL_CELL, oy + r * LABEL_CELL
            blob = CELL2BLOB.get((c, r))
            if mode == "tf":
                fill = BLUE_L if blob else "#ffffff"
                edge = BLUE if blob else GRID
                text = "T" if blob else "F"
                color = BLUE if blob else "#aab4c0"
            else:
                fill = BLOB_FILL[blob] if blob else "#ffffff"
                edge = BLOB_EDGE[blob] if blob else GRID
                text = str(blob) if blob else "0"
                color = BLOB_EDGE[blob] if blob else "#aab4c0"
            weight = "bold" if blob else "normal"
            out.append(f'<rect x="{x}" y="{y}" width="{LABEL_CELL}" height="{LABEL_CELL}" fill="{fill}" stroke="{edge}" stroke-width="1"/>')
            out.append(f'<text x="{x + LABEL_CELL / 2}" y="{y + LABEL_CELL / 2 + 4}" font-size="13" font-weight="{weight}" '
                       f'text-anchor="middle" fill="{color}">{text}</text>')
    return "\n    ".join(out)


def _corner_grid(ox, oy, neighbourhood):
    out = []
    for r in range(2):
        for c in range(4):
            x, y = ox + c * LABEL_CELL, oy + r * LABEL_CELL
            tag = CORNER.get((c, r))
            if tag is None:
                out.append(f'<rect x="{x}" y="{y}" width="{LABEL_CELL}" height="{LABEL_CELL}" fill="#ffffff" stroke="{GRID}" stroke-width="1"/>')
                continue
            num = 1 if neighbourhood == 8 else (1 if tag == "P" else 2)
            fill, edge = (BLUE_L, BLUE) if num == 1 else (GREEN_L, GREEN)
            out.append(f'<rect x="{x}" y="{y}" width="{LABEL_CELL}" height="{LABEL_CELL}" fill="{fill}" stroke="{edge}" stroke-width="1"/>')
            out.append(f'<text x="{x + LABEL_CELL / 2}" y="{y + LABEL_CELL / 2 + 4}" font-size="13" font-weight="bold" '
                       f'text-anchor="middle" fill="{edge}">{num}</text>')
    return "\n    ".join(out)


def build_label_raster():
    counts = {b: len(cs) for b, cs in BLOBS.items()}
    total = sum(counts.values())
    lx, rx, gy = 112, 424, 92
    tally = " · ".join(f"{b}번 {n}픽셀" for b, n in sorted(counts.items()))
    path = write_svg("label_raster.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 520" font-family="sans-serif">
  <text x="380" y="30" font-size="18" font-weight="bold" text-anchor="middle">같은 격자에 참·거짓을 적을 때와 덩어리 번호를 적을 때</text>

  <text x="{lx + 112}" y="78" font-size="15" font-weight="bold" text-anchor="middle">① 참·거짓 마스크</text>
  <g>
    {_label_grid(lx, gy, 7, 5, "tf")}
  </g>
  <text x="{lx + 112}" y="276" font-size="12" text-anchor="middle" fill="{GRAY_T}">픽셀마다 답이 하나 — 대상 픽셀 {total}개</text>
  <text x="{lx + 112}" y="294" font-size="12" text-anchor="middle" fill="{RED}">어느 픽셀이 어느 덩어리인지 알 수 없음</text>

  <line x1="346" y1="172" x2="406" y2="172" stroke="{GRAY_T}" stroke-width="2"/>
  <path d="M 406 172 l -9 -5 l 0 10 z" fill="{GRAY_T}"/>
  <text x="376" y="160" font-size="12" text-anchor="middle" fill="{GRAY_T}">연결요소</text>
  <text x="376" y="196" font-size="11" text-anchor="middle" fill="{GRAY_T}">label()</text>

  <text x="{rx + 112}" y="78" font-size="15" font-weight="bold" text-anchor="middle">② 덩어리 번호 격자</text>
  <g>
    {_label_grid(rx, gy, 7, 5, "num")}
  </g>
  <text x="{rx + 112}" y="276" font-size="12" text-anchor="middle" fill="{GRAY_T}">덩어리마다 다른 번호, 배경은 0</text>
  <text x="{rx + 112}" y="294" font-size="12" text-anchor="middle" fill="{GREEN}">번호별로 세면 {tally}</text>

  <line x1="60" y1="318" x2="700" y2="318" stroke="{GRID}" stroke-width="1"/>
  <text x="380" y="344" font-size="15" font-weight="bold" text-anchor="middle">③ 무엇을 "맞닿았다"고 볼지에 따라 번호가 갈린다</text>
  <text x="234" y="372" font-size="13" font-weight="bold" text-anchor="middle">4-이웃 (상하좌우만)</text>
  <g>
    {_corner_grid(170, 384, 4)}
  </g>
  <text x="234" y="472" font-size="12" text-anchor="middle" fill="{GRAY_T}">모서리만 닿은 두 덩어리 → 번호 1과 2</text>
  <text x="234" y="490" font-size="12" text-anchor="middle" fill="{BLUE}">두 객체로 셈</text>
  <text x="514" y="372" font-size="13" font-weight="bold" text-anchor="middle">8-이웃 (대각선도 포함)</text>
  <g>
    {_corner_grid(450, 384, 8)}
  </g>
  <text x="514" y="472" font-size="12" text-anchor="middle" fill="{GRAY_T}">같은 그림 → 모두 번호 1</text>
  <text x="514" y="490" font-size="12" text-anchor="middle" fill="{BLUE}">한 객체로 셈</text>
</svg>
''')
    return path, counts


# ──────────────────────────────── 7.3 크랙 둘레 ────────────────────────────────

CRACK_W, CRACK_H = 4, 3      # 예시 블록 (픽셀 한 변 1m)
GAP = 4                      # 토막 양끝을 줄여 하나씩 떨어져 보이게


def _block_grid(ox, oy, cols, rows, fill_in, fill_bd, edge):
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = ox + c * CELL, oy + r * CELL
            inside = 1 <= c <= CRACK_W and 1 <= r <= CRACK_H
            if not inside:
                out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="#ffffff" stroke="{GRID}"/>')
                continue
            edge_cell = (c in (1, CRACK_W)) or (r in (1, CRACK_H))
            out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                       f'fill="{fill_bd if edge_cell else fill_in}" stroke="{edge}"/>')
    return "\n    ".join(out)


def build_crack_perimeter():
    block = np.zeros((CRACK_H + 2, CRACK_W + 2), bool)
    block[1:CRACK_H + 1, 1:CRACK_W + 1] = True
    boundary = block & ~ndimage.binary_erosion(block)
    n_boundary = int(boundary.sum())
    n_inner = int((block & ~boundary).sum())
    perim = 2 * (CRACK_W + CRACK_H)

    lx, rx, gy = 60, 420, 95
    seg = []
    for c in range(1, CRACK_W + 1):                       # 위·아래 변
        for yy in (1, CRACK_H + 1):
            x0, y0 = rx + c * CELL, gy + yy * CELL
            seg.append(f'<line x1="{x0 + GAP}" y1="{y0}" x2="{x0 + CELL - GAP}" y2="{y0}" '
                       f'stroke="{RED}" stroke-width="6" stroke-linecap="round"/>')
    for r in range(1, CRACK_H + 1):                       # 좌·우 변
        for xx in (1, CRACK_W + 1):
            x0, y0 = rx + xx * CELL, gy + r * CELL
            seg.append(f'<line x1="{x0}" y1="{y0 + GAP}" x2="{x0}" y2="{y0 + CELL - GAP}" '
                       f'stroke="{RED}" stroke-width="6" stroke-linecap="round"/>')
    assert len(seg) == perim, (len(seg), perim)

    path = write_svg("crack_perimeter.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 410" font-family="sans-serif">
  <text x="380" y="30" font-size="18" font-weight="bold" text-anchor="middle">크랙 둘레는 경계 칸의 개수가 아니라 경계선 토막의 길이다</text>
  <text x="380" y="52" font-size="12" text-anchor="middle" fill="{GRAY_T}">가로 {CRACK_W}m × 세로 {CRACK_H}m 블록, 픽셀 한 변 1m</text>

  <text x="{lx + 3 * CELL}" y="80" font-size="15" font-weight="bold" text-anchor="middle">① 경계에 놓인 칸을 세면</text>
  {_block_grid(lx, gy, 6, 5, BLUE_L, ORANGE_L, ORANGE)}
  <text x="{lx + 3 * CELL}" y="{gy + 5 * CELL + 28}" font-size="13" font-weight="bold" text-anchor="middle" fill="{ORANGE}">경계 칸 {n_boundary}개 · 안쪽 칸 {n_inner}개</text>
  <text x="{lx + 3 * CELL}" y="{gy + 5 * CELL + 48}" font-size="12" text-anchor="middle" fill="{GRAY_T}">모퉁이 칸도 한 칸으로 셈</text>

  <text x="{rx + 3 * CELL}" y="80" font-size="15" font-weight="bold" text-anchor="middle">② 칸 사이 경계선 토막을 세면</text>
  {_block_grid(rx, gy, 6, 5, BLUE_L, BLUE_L, BLUE)}
  {chr(10).join("  " + s for s in seg)}
  <text x="{rx + 3 * CELL}" y="{gy + 5 * CELL + 28}" font-size="13" font-weight="bold" text-anchor="middle" fill="{RED}">토막 {perim}개 = 2 × ({CRACK_W} + {CRACK_H}) = {perim}m</text>
  <text x="{rx + 3 * CELL}" y="{gy + 5 * CELL + 48}" font-size="12" text-anchor="middle" fill="{GRAY_T}">모퉁이 칸은 두 변이 드러나 토막이 둘</text>

  <line x1="60" y1="352" x2="700" y2="352" stroke="{GRID}"/>
  <text x="380" y="376" font-size="12.5" text-anchor="middle" fill="{GRAY_T}">같은 도형을 두 방식으로 세면 {n_boundary}과 {perim}로 갈림</text>
  <text x="380" y="396" font-size="12.5" text-anchor="middle" fill="{GRAY_T}">40m × 25m 필지라면 크랙 둘레는 2 × (40 + 25) = 130m</text>
</svg>
''')
    return path, n_boundary, n_inner, perim


# ───────────────────────────────── 7.3 조밀도 ─────────────────────────────────

COMPACT_AREA = 100.0        # 네 도형의 면적을 이 값으로 고정한다
COMPACT_SHAPES = [("원", None, BLUE, BLUE_L),
                  ("정사각형", (10, 10), GREEN, GREEN_L),
                  ("직사각형", (25, 4), ORANGE, ORANGE_L),
                  ("직사각형", (50, 2), RED, RED_L)]
COMPACT_SLOTS = [110, 305, 500, 695]
COMPACT_SCALE = 3.6         # 1m를 몇 픽셀로 그릴지


def build_compactness():
    parts, result = [], []
    for (name, wh, col, fill), cx in zip(COMPACT_SHAPES, COMPACT_SLOTS):
        if wh is None:
            r = math.sqrt(COMPACT_AREA / math.pi)
            perim = 2 * math.pi * r
            parts.append(f'<circle cx="{cx}" cy="150" r="{r * COMPACT_SCALE:.1f}" '
                         f'fill="{fill}" stroke="{col}" stroke-width="2"/>')
            size = f"지름 {2 * r:.1f}m"
        else:
            w, h = wh
            perim = 2 * (w + h)
            parts.append(f'<rect x="{cx - w * COMPACT_SCALE / 2:.1f}" y="{150 - h * COMPACT_SCALE / 2:.1f}" '
                         f'width="{w * COMPACT_SCALE:.1f}" height="{h * COMPACT_SCALE:.1f}" '
                         f'fill="{fill}" stroke="{col}" stroke-width="2"/>')
            size = f"{w}m × {h}m"
        comp = 4 * math.pi * COMPACT_AREA / perim ** 2
        result.append((name, size, perim, comp))
        parts.append(f'<text x="{cx}" y="78" font-size="14" font-weight="bold" text-anchor="middle">{name}</text>')
        parts.append(f'<text x="{cx}" y="96" font-size="11.5" text-anchor="middle" fill="{GRAY_T}">{size}</text>')
        parts.append(f'<text x="{cx}" y="228" font-size="12" text-anchor="middle" fill="{GRAY_T}">둘레 {perim:.1f}m</text>')
        parts.append(f'<text x="{cx}" y="252" font-size="17" font-weight="bold" text-anchor="middle" fill="{col}">{comp:.3f}</text>')

    body = "\n".join("  " + p for p in parts)
    path = write_svg("compactness.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 320" font-family="sans-serif">
  <text x="410" y="30" font-size="18" font-weight="bold" text-anchor="middle">조밀도 4πA/P² — 면적이 같아도 길쭉할수록 값이 0에 가까워진다</text>
  <text x="410" y="52" font-size="12" text-anchor="middle" fill="{GRAY_T}">넷 모두 면적 {COMPACT_AREA:.0f}㎡로 같고 모양만 다름. 그림은 실제 비율대로 그림</text>
{body}
  <line x1="60" y1="272" x2="760" y2="272" stroke="{GRID}"/>
  <text x="410" y="296" font-size="13" text-anchor="middle" fill="{GRAY_T}">면적이 분자에 한 번, 둘레가 분모에 제곱으로 들어가므로 둘레만 길어지면 값이 빠르게 떨어짐</text>
</svg>
''')
    return path, result


# ───────────────────────────────── 7.3 최소폭 ─────────────────────────────────

MW_CELL = 32
MW_W, MW_H = 7, 5           # 예시 블록 (픽셀 한 변 1m)


def build_min_width():
    block = np.zeros((MW_H + 2, MW_W + 2), bool)
    block[1:MW_H + 1, 1:MW_W + 1] = True
    edt = ndimage.distance_transform_edt(block)[1:MW_H + 1, 1:MW_W + 1]
    radius = float(edt.max())          # 내접원 반지름 (칸 단위 = m)
    width = 2 * radius                 # 격자가 낸 최소폭
    true_width = float(min(MW_W, MW_H))  # 실제로 들어가는 가장 큰 원의 지름

    def grid(ox, oy, show_values):
        out = []
        for r in range(MW_H):
            for c in range(MW_W):
                x, y = ox + c * MW_CELL, oy + r * MW_CELL
                deepest = edt[r, c] == radius
                fill = RED_L if (deepest and show_values) else BLUE_L
                edge = RED if (deepest and show_values) else BLUE
                out.append(f'<rect x="{x}" y="{y}" width="{MW_CELL}" height="{MW_CELL}" fill="{fill}" stroke="{edge}"/>')
                if show_values:
                    out.append(f'<text x="{x + MW_CELL / 2}" y="{y + MW_CELL / 2 + 5}" font-size="12.5" '
                               f'font-weight="{"bold" if deepest else "normal"}" text-anchor="middle" '
                               f'fill="{RED if deepest else GRAY_T}">{edt[r, c]:.0f}</text>')
        return "\n    ".join(out)

    lx, rx, gy = 55, 440, 100
    ccx, ccy = rx + 3.5 * MW_CELL, gy + 2.5 * MW_CELL
    path = write_svg("min_width.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 430" font-family="sans-serif">
  <text x="380" y="30" font-size="18" font-weight="bold" text-anchor="middle">최소폭은 객체 안에 들어가는 가장 큰 원의 지름으로 잰다</text>
  <text x="380" y="52" font-size="12" text-anchor="middle" fill="{GRAY_T}">가로 {MW_W}m × 세로 {MW_H}m 블록, 픽셀 한 변 1m</text>

  <text x="{lx + 3.5 * MW_CELL}" y="86" font-size="14.5" font-weight="bold" text-anchor="middle">① 거리변환 — 가장 가까운 배경까지 몇 칸인가</text>
  {grid(lx, gy, True)}
  <text x="{lx + 3.5 * MW_CELL}" y="{gy + MW_H * MW_CELL + 26}" font-size="12" text-anchor="middle" fill="{GRAY_T}">가장자리는 1, 안으로 갈수록 커짐</text>
  <text x="{lx + 3.5 * MW_CELL}" y="{gy + MW_H * MW_CELL + 45}" font-size="12.5" font-weight="bold" text-anchor="middle" fill="{RED}">최댓값 {radius:.0f} — 가장 깊은 칸</text>

  <text x="{rx + 3.5 * MW_CELL}" y="86" font-size="14.5" font-weight="bold" text-anchor="middle">② 그 값을 반지름으로 삼은 원</text>
  {grid(rx, gy, False)}
  <ellipse cx="{ccx}" cy="{ccy}" rx="{true_width / 2 * MW_CELL}" ry="{true_width / 2 * MW_CELL}" fill="none" stroke="{GRAY_T}" stroke-width="2" stroke-dasharray="6 4"/>
  <circle cx="{ccx}" cy="{ccy}" r="{radius * MW_CELL}" fill="none" stroke="{RED}" stroke-width="3"/>
  <circle cx="{ccx}" cy="{ccy}" r="4" fill="{RED}"/>
  <text x="{rx + 3.5 * MW_CELL}" y="{gy + MW_H * MW_CELL + 26}" font-size="12.5" font-weight="bold" text-anchor="middle" fill="{RED}">격자가 낸 원 — 지름 2 × {radius:.0f} = {width:.0f}m</text>
  <text x="{rx + 3.5 * MW_CELL}" y="{gy + MW_H * MW_CELL + 45}" font-size="12" text-anchor="middle" fill="{GRAY_T}">점선은 실제로 들어가는 가장 큰 원 — 지름 {true_width:.0f}m</text>

  <line x1="55" y1="356" x2="705" y2="356" stroke="{GRID}"/>
  <text x="380" y="380" font-size="12.5" text-anchor="middle" fill="{GRAY_T}">블록의 좁은 변이 {true_width:.0f}m인데 이렇게 재면 {width:.0f}m가 나옴. 거리를 칸 단위로만 셀 수 있어 생기는 차이임</text>
  <text x="380" y="402" font-size="12.5" text-anchor="middle" fill="{GRAY_T}">코드 주석의 "격자 이산화로 약 1m 과대"가 이 차이를 가리킴</text>
</svg>
''')
    return path, radius, width, true_width


def main():
    path = build_mask_to_decision()
    print(f"{path.name}  — 3.2 세 경로 (상자 안 수치는 실행 로그에서 옮겨 적은 값)")

    path, counts = build_label_raster()
    tally = " / ".join(f"{b}번 {n}픽셀" for b, n in sorted(counts.items()))
    print(f"{path.name}  — 7.1 덩어리 번호: 대상 {sum(counts.values())}픽셀 = {tally}")

    path, n_bd, n_in, perim = build_crack_perimeter()
    print(f"{path.name}  — 7.3 크랙 둘레: 경계 칸 {n_bd}개 / 안쪽 칸 {n_in}개 / 토막 {perim}개")

    path, rows = build_compactness()
    print(f"{path.name}  — 7.3 조밀도 (면적 {COMPACT_AREA:.0f}㎡ 고정)")
    for name, size, perim, comp in rows:
        print(f"    {name:5s} {size:12s} 둘레 {perim:6.2f}m  4πA/P² = {comp:.3f}")

    path, radius, width, true_width = build_min_width()
    print(f"{path.name}  — 7.3 최소폭: 거리변환 최댓값 {radius:.2f}m → 지름 {width:.2f}m "
          f"(실제 좁은 변 {true_width:.0f}m, {width - true_width:+.2f}m 과대)")


if __name__ == "__main__":
    main()
