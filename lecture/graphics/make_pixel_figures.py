"""6장 4.1절 그림 두 장을 만든다.

만드는 파일 (이 스크립트와 같은 폴더)
  house_pixel_error.svg  격자가 정하는 오차의 하한 — 경계선이 가로지른 칸은 어떤 규칙으로도 갈림
  pixel_error.svg        그 칸이 전체에서 차지하는 비율, 그리고 그 수가 둘레와 같다는 것

두 그림이 말하는 것은 **격자 때문에 생기는 오차**이지 이 장의 경계 침식 전체가 아니다.
경계선이 가로지르는 칸은 폴리곤을 따라 한 겹이므로, 양자화가 만드는 침식은 경계에
수직으로는 한 겹이 상한이다. 다만 그것이 잃는 면적의 상한은 아니다 — 폭이 2픽셀
미만인 부분은 안쪽 칸이 없어 그 한 겹이 전부이고, 통째로 사라진다(그림의 ① 패널이
그 경우다). 한편 모델이 풀링으로 위치 정보를 잃으면 경계를 더 안쪽에 그어, 경계선이
지나가지도 않은 칸까지 깎는다(5절·8.2절). 두 그림에 이 단서를 두 줄로 달아 두며,
수치는 urban_block.npz에서 직접 읽는다.

두 그림은 `lecture/ch06.md` 4.1절이 쓴다. 수치나 모양을 고칠 때는 SVG를 직접
손대지 말고 이 스크립트를 고쳐 다시 돌린다.

무엇을 계산하는가
  픽셀 하나를 16×16으로 잘게 나눠 주택이 그 칸의 몇 할을 덮는지(f)를 구한다.
    f = 1        칸이 통째로 주택 안  → 어떤 규칙으로도 대상으로 판정됨
    0 < f < 1    주택 경계선이 칸을 가로지름 → 판정이 갈리는 칸
    f = 0        칸이 통째로 주택 밖
  침식은 가운데 칸들을 전부 배경으로 본 결과이고, 팽창은 전부 대상으로 본 결과다.
  즉 침식과 팽창의 폭은 곧 '경계선이 가로지르는 칸'의 개수다.

  주택은 격자에 반 칸 어긋나게 놓는다(GRID_OFFSET). 변의 길이가 격자에 딱
  맞아떨어지면 경계선이 격자선 위에 포개져 칸을 가로지르지 않는데, 실제 지형이
  격자에 맞아떨어지는 일은 없기 때문이다.

그림에 찍히는 숫자는 전부 여기서 센 값이다. 손으로 적은 숫자가 없으므로
HOUSE나 해상도를 바꾸면 그림의 숫자도 따라 바뀐다.

실행
    python lecture/graphics/make_pixel_figures.py

필요한 패키지는 `lecture_practice/requirements-student.txt`에 들어 있다
(numpy, matplotlib).
"""

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

# 폭 12m 주택: 몸통 12m×7m(84㎡) + 지붕 12m×4m(24㎡) = 108㎡
HOUSE = [(0, 0), (12, 0), (12, 7), (6, 11), (0, 7)]
HOUSE_W, HOUSE_H = 12.0, 11.0
HOUSE_AREA = 108.0
HOUSE_PERIM = 12 + 7 + 2 * (6 ** 2 + 4 ** 2) ** 0.5 + 7   # 40.42m

GRID_OFFSET = 0.5      # 주택을 격자에서 몇 칸 어긋나게 놓을지
SUBSAMPLE = 16         # 칸 하나를 몇 등분해 덮임 비율을 재는지

BLUE = "#4a90e2"    # 칸이 통째로 주택 안
PALE = "#cfe0f5"    # 안쪽을 흐리게 깔 때
AMBER = "#f0a93c"   # 주택 경계선이 가로지르는 칸
GREY = "#eaeaea"    # 빈 칸
LINE = "#1f1f1f"    # 주택 경계선
RED = "#c0271b"
DARK = "#333"

OUT_DIR = Path(__file__).parent


# ── 계산 ──────────────────────────────────────────────────────────────────

def coverage(pixel_m, pad):
    """칸마다 주택이 덮은 비율 f를 구한다.

    주택을 GRID_OFFSET칸 + pad칸만큼 옮겨 놓고 잰다. 돌려주는 배열은
    row 0이 아래쪽이다(지도와 같은 방향).
    """
    shift = (GRID_OFFSET + pad) * pixel_m
    poly = MplPath([(x + shift, y + shift) for x, y in HOUSE])
    nx = int(np.ceil((HOUSE_W + shift) / pixel_m)) + pad
    ny = int(np.ceil((HOUSE_H + shift) / pixel_m)) + pad
    step = (np.arange(SUBSAMPLE) + 0.5) / SUBSAMPLE
    frac = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            gx, gy = np.meshgrid((c + step) * pixel_m, (r + step) * pixel_m)
            frac[r, c] = poly.contains_points(np.c_[gx.ravel(), gy.ravel()]).mean()
    return frac


def classify(frac):
    """f를 세 갈래로 나눈다 — (통째로 안, 경계선이 지남, 둘을 합친 것)."""
    inside = frac >= 0.999
    edge = (frac > 0.001) & (frac < 0.999)
    return inside, edge, inside | edge


def caveat_lines():
    """그림이 말하는 범위를 좁히는 단서 두 줄.

    1행 — 격자가 만드는 오차는 경계에 수직으로 한 겹이지만, 폭이 2픽셀 미만인
           부분은 그 한 겹이 전부라 통째로 사라진다. "한 겹"은 깊이의 상한이지
           잃는 면적의 상한이 아니다.
    2행 — 모델의 경계 침식은 이와 별개이고 더 깊이 들어간다. 수치는 실습 자료
           urban_block.npz에서 직접 읽는다(없으면 수치 없이 문장만).
    """
    npz = OUT_DIR.parent.parent / "lecture_practice" / "chapter6" / "data" / "urban_block.npz"
    tail = ""
    if npz.exists():
        data = np.load(npz)
        n_deep = len(data["deep_erosion_ids"])
        n_parcel = int(data["parcel_id"].max())
        tail = f" — 이 장의 실습 자료는 {n_parcel}필지 중 {n_deep}필지가 두 겹 침식"
    return [
        "격자가 애매하게 만드는 것은 경계에 수직으로 한 겹뿐임. "
        "다만 ①처럼 폭이 2픽셀 미만이면 그 한 겹이 전부라 통째로 사라짐",
        "모델이 경계를 흐리게 그리면 큰 대상에서도 여러 겹이 깎임" + tail + " (5절·8.2)",
    ]


# ── SVG 조각 ──────────────────────────────────────────────────────────────

def grid_bg(nx, ny, cell, stroke):
    parts = [f'<rect x="0" y="0" width="{nx * cell:g}" height="{ny * cell:g}" fill="{GREY}"/>']
    for i in range(nx + 1):
        parts.append(f'<line x1="{i * cell:g}" y1="0" x2="{i * cell:g}" '
                     f'y2="{ny * cell:g}" stroke="#fff" stroke-width="{stroke}"/>')
    for j in range(ny + 1):
        parts.append(f'<line x1="0" y1="{j * cell:g}" x2="{nx * cell:g}" '
                     f'y2="{j * cell:g}" stroke="#fff" stroke-width="{stroke}"/>')
    return "\n        ".join(parts)


def cells(mask, cell, color, stroke, rows):
    parts = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                parts.append(
                    f'<rect x="{c * cell:g}" y="{(rows - 1 - r) * cell:g}" '
                    f'width="{cell:g}" height="{cell:g}" '
                    f'fill="{color}" stroke="#fff" stroke-width="{stroke}"/>')
    return "\n        ".join(parts)


def boundary_line(pixel_m, cell, pad, rows, width=2.6):
    """주택의 실제 경계선. 침식도 팽창도 이 선이 지나간 칸에서 생긴다."""
    shift = (GRID_OFFSET + pad) * cell
    px_per_m = cell / pixel_m
    pts = " ".join(f"{shift + x * px_per_m:.1f},{rows * cell - shift - y * px_per_m:.1f}"
                   for x, y in HOUSE)
    return (f'<polygon points="{pts}" fill="none" stroke="{LINE}" '
            f'stroke-width="{width}" stroke-linejoin="round"/>')


def minus(value, fmt="+.1f"):
    return format(value, fmt).replace("-", "−")


def write_svg(name, width, height, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {width} {height:g}" font-family="sans-serif">\n'
           f'  <rect width="{width}" height="{height:g}" fill="#fff"/>\n'
           + "\n".join(body) + "\n</svg>\n")
    path = OUT_DIR / name
    path.write_text(svg, encoding="utf-8")
    return path


def legend(x, y, items, box=15, gap=210, size=13):
    """색 설명 한 줄."""
    parts = []
    for i, (color, label) in enumerate(items):
        cx = x + i * gap
        parts.append(f'<rect x="{cx}" y="{y}" width="{box}" height="{box}" '
                     f'fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<text x="{cx + box + 7}" y="{y + box - 2}" font-size="{size}" '
                     f'fill="{DARK}">{label}</text>')
    return "\n    ".join(parts)


# ── 그림 1: 판정이 갈리는 폭 (house_pixel_error.svg) ──────────────────────

def build_house_pixel_error():
    width = 1000
    out = [
        f'  <text x="{width // 2}" y="34" font-size="21" font-weight="bold" '
        f'text-anchor="middle">격자가 정하는 오차의 하한 — 경계선이 가로지른 칸은 '
        f'어떤 규칙으로도 갈린다</text>',
        f'  <text x="{width // 2}" y="57" font-size="13.5" fill="#666" '
        f'text-anchor="middle">같은 주택(폭 12m, 108㎡)을 두 해상도로 관측함 &#183; '
        f'검은 선이 주택의 실제 경계선, 주황은 그 선이 가로지른 칸</text>',
    ]
    result = {}

    def panel(y0, title, sub, pixel_m, cell, pad, stroke, note):
        frac = coverage(pixel_m, pad)
        inside, edge, grown = classify(frac)
        rows, cols = frac.shape
        n_in, n_edge, n_out = int(inside.sum()), int(edge.sum()), int(grown.sum())
        gw, gh = cols * cell, rows * cell
        panel_h = 108 + gh + 96

        out.append(f'  <g transform="translate(26,{y0})">')
        out.append(f'    <rect x="0" y="0" width="948" height="{panel_h:g}" '
                   f'fill="#fafafa" stroke="#e0e0e0" rx="10"/>')
        out.append(f'    <text x="22" y="32" font-size="17.5" font-weight="bold" '
                   f'fill="{DARK}">{title}</text>')
        out.append(f'    <text x="22" y="54" font-size="13.5" fill="#666">{sub}</text>')

        stages = [
            ("경계선이 지난 칸을 배경으로 봄", [(inside, BLUE)],
             f"{n_in}칸", "침식 — 가장 작게 잡은 값", RED if n_in == 0 else DARK),
            ("주택 경계선과 격자", [(inside, BLUE), (edge, AMBER)],
             f"안쪽 {n_in}칸 + 경계 {n_edge}칸", "경계선이 가로지른 칸이 주황", DARK),
            ("경계선이 지난 칸을 대상으로 봄", [(grown, BLUE)],
             f"{n_out}칸", "팽창 — 가장 크게 잡은 값", DARK),
        ]
        xs = [36, 336, 636]
        for i, (label, layers, big, small, color) in enumerate(stages):
            out.append(f'    <g transform="translate({xs[i]},108)">')
            out.append(f'      <text x="{gw / 2:g}" y="-16" font-size="14.5" '
                       f'font-weight="bold" text-anchor="middle">{label}</text>')
            out.append(f'      {grid_bg(cols, rows, cell, stroke)}')
            for mask, color_ in layers:
                out.append(f'      {cells(mask, cell, color_, stroke, rows)}')
            out.append(f'      {boundary_line(pixel_m, cell, pad, rows)}')
            out.append(f'      <text x="{gw / 2:g}" y="{gh + 25:g}" font-size="15" '
                       f'font-weight="bold" fill="{color}" text-anchor="middle">{big}</text>')
            out.append(f'      <text x="{gw / 2:g}" y="{gh + 46:g}" font-size="13" '
                       f'fill="#666" text-anchor="middle">{small}</text>')
            out.append(f'    </g>')

        mid_y = 108 + gh / 2 + 8
        out.append(f'    <text x="{xs[0] + gw + 18}" y="{mid_y:g}" font-size="26" '
                   f'fill="#bbb" text-anchor="middle">&#8592;</text>')
        out.append(f'    <text x="{xs[1] + gw + 18}" y="{mid_y:g}" font-size="26" '
                   f'fill="#bbb" text-anchor="middle">&#8594;</text>')
        out.append(f'    <text x="474" y="{108 + gh + 76:g}" font-size="14.5" '
                   f'fill="{DARK}" text-anchor="middle">{note}</text>')
        out.append(f'  </g>')
        return y0 + panel_h + 22, (n_in, n_edge, n_out)

    y, result["10m"] = panel(
        78, "① 10m 격자 — 주택 폭이 1.2픽셀",
        "통째로 주택 안에 들어가는 칸이 하나도 없음. 모든 칸을 경계선이 가로지름",
        pixel_m=10.0, cell=52, pad=1, stroke=1.6,
        note="판정 규칙 하나로 0칸과 4칸 사이를 오감. 대상이 통째로 사라지거나 몇 배로 커짐")
    y, result["0.5m"] = panel(
        y, "② 0.5m 격자 — 주택 폭이 24픽셀",
        "안쪽은 어떤 규칙으로도 대상이고, 경계선이 지난 테두리 한 줄만 갈림",
        pixel_m=0.5, cell=6.0, pad=1, stroke=0.7,
        note="판정이 갈려도 387칸과 479칸 사이이고, 지붕과 몸통의 모양은 그대로 남음")

    i1, e1, o1 = result["10m"]
    i2, e2, o2 = result["0.5m"]
    out.append(f'  <text x="{width // 2}" y="{y + 4:g}" font-size="15" fill="{DARK}" '
               f'text-anchor="middle">같은 주택인데 1.2픽셀에서는 {i1}칸~{o1}칸으로 갈리고, '
               f'24픽셀에서는 {i2}칸~{o2}칸에 머무름</text>')
    for k, line in enumerate(caveat_lines()):
        out.append(f'  <text x="{width // 2}" y="{y + 26 + k * 20:g}" font-size="13" '
                   f'fill="{RED}" text-anchor="middle">{line}</text>')
    out.append(f'  <g>{legend(300, y + 60, [(BLUE, "통째로 주택 안"), (AMBER, "경계선이 가로지름")])}</g>')
    return write_svg("house_pixel_error.svg", width, y + 94, out), result


# ── 그림 2: 경계 칸의 비율 (pixel_error.svg) ──────────────────────────────

def build_pixel_error():
    width = 1000
    out = [
        f'  <text x="{width // 2}" y="34" font-size="21" font-weight="bold" '
        f'text-anchor="middle">경계선이 가로지른 칸은 둘레만큼 생기므로, '
        f'작은 대상일수록 그 몫이 커진다</text>',
        f'  <text x="{width // 2}" y="57" font-size="13.5" fill="#666" '
        f'text-anchor="middle">주택 = 몸통 12m×7m + 지붕 12m×4m = 108㎡, 둘레 '
        f'{HOUSE_PERIM:.1f}m &#183; 검은 선이 주택의 실제 경계선</text>',
    ]
    result = {}

    def panel(y0, title, sub, pixel_m, cell, pad, stroke):
        frac = coverage(pixel_m, pad)
        inside, edge, grown = classify(frac)
        rows, cols = frac.shape
        n_in, n_edge, n_all = int(inside.sum()), int(edge.sum()), int(grown.sum())
        pct = n_edge / n_all * 100
        perim_cells = HOUSE_PERIM / pixel_m
        area_cells = HOUSE_AREA / (pixel_m ** 2)
        tail_note = ("  (면적 108㎡는 10m 칸으로 1칸 값어치뿐인데 4칸에 걸침)"
                     if area_cells < 2 else "")
        gw, gh = cols * cell, rows * cell
        panel_h = 104 + gh + 104

        out.append(f'  <g transform="translate(26,{y0})">')
        out.append(f'    <rect x="0" y="0" width="948" height="{panel_h:g}" '
                   f'fill="#fafafa" stroke="#e0e0e0" rx="10"/>')
        out.append(f'    <text x="22" y="32" font-size="17.5" font-weight="bold" '
                   f'fill="{DARK}">{title}</text>')
        out.append(f'    <text x="22" y="54" font-size="13.5" fill="#666">{sub}</text>')

        stages = [
            ("1. 경계선을 격자 위에 얹음", [],
             "선이 칸을 가로지름"),
            ("2. 그 선이 지난 칸을 표시", [(inside, BLUE), (edge, AMBER)],
             f"경계 {n_edge}칸"),
            ("3. 경계 칸이 차지하는 몫", [(inside, PALE), (edge, AMBER)],
             f"{n_edge} ÷ {n_all} = {pct:.1f}%"),
        ]
        xs = [36, 336, 636]
        for i, (label, layers, caption) in enumerate(stages):
            out.append(f'    <g transform="translate({xs[i]},104)">')
            out.append(f'      <text x="{gw / 2:g}" y="-16" font-size="14.5" '
                       f'font-weight="bold" text-anchor="middle">{label}</text>')
            out.append(f'      {grid_bg(cols, rows, cell, stroke)}')
            for mask, color_ in layers:
                out.append(f'      {cells(mask, cell, color_, stroke, rows)}')
            out.append(f'      {boundary_line(pixel_m, cell, pad, rows)}')
            weight = "bold" if i == 2 else "normal"
            size = 15 if i == 2 else 13.5
            out.append(f'      <text x="{gw / 2:g}" y="{gh + 25:g}" font-size="{size}" '
                       f'font-weight="{weight}" fill="{DARK}" '
                       f'text-anchor="middle">{caption}</text>')
            out.append(f'    </g>')

        out.append(f'    <text x="474" y="{104 + gh + 56:g}" font-size="14.5" '
                   f'fill="{DARK}" text-anchor="middle">'
                   f'경계 칸 {n_edge}칸 &#8776; 둘레 {HOUSE_PERIM:.1f}m ÷ {pixel_m:g}m = '
                   f'{perim_cells:.0f}칸 &#8212; 경계 칸 수는 대략 둘레만큼임</text>')
        out.append(f'    <text x="474" y="{104 + gh + 80:g}" font-size="15" '
                   f'font-weight="bold" fill="{RED if pct > 50 else DARK}" '
                   f'text-anchor="middle">판정에 따라 흔들리는 몫 = {n_edge} ÷ {n_all} '
                   f'= {pct:.1f}%{tail_note}</text>')
        out.append(f'  </g>')
        return y0 + panel_h + 22, (n_in, n_edge, n_all, pct)

    y, result["10m"] = panel(
        78, "① 10m 격자 — 주택 폭이 1.2픽셀",
        "경계선이 모든 칸을 가로지르므로 경계 칸이 곧 전체임",
        pixel_m=10.0, cell=52, pad=1, stroke=1.6)
    y, result["0.5m"] = panel(
        y, "② 0.5m 격자 — 주택 폭이 24픽셀",
        "경계선이 테두리 한 줄만 가로지르므로 안쪽은 흔들리지 않음",
        pixel_m=0.5, cell=6.0, pad=1, stroke=0.7)

    out.append(f'  <text x="{width // 2}" y="{y + 4:g}" font-size="15" fill="{DARK}" '
               f'text-anchor="middle">한 변이 10배 길어지면 면적은 100배가 되지만 둘레는 '
               f'10배에 그치므로, 경계가 차지하는 몫은 1/10로 줄어듦</text>')
    for k, line in enumerate(caveat_lines()):
        out.append(f'  <text x="{width // 2}" y="{y + 26 + k * 20:g}" font-size="13" '
                   f'fill="{RED}" text-anchor="middle">{line}</text>')
    out.append(f'  <g>{legend(300, y + 60, [(BLUE, "통째로 주택 안"), (AMBER, "경계선이 가로지름")])}</g>')
    return write_svg("pixel_error.svg", width, y + 94, out), result


def main():
    path1, r1 = build_house_pixel_error()
    print(f"{path1.name}  — 판정이 갈리는 폭")
    for key, (n_in, n_edge, n_all) in r1.items():
        print(f"    {key:>5} 격자: 침식 {n_in}칸 / 경계선이 지난 칸 {n_edge} / 팽창 {n_all}칸")

    path2, r2 = build_pixel_error()
    print(f"{path2.name}  — 경계 칸이 차지하는 비율")
    for key, (n_in, n_edge, n_all, pct) in r2.items():
        print(f"    {key:>5} 격자: 경계 {n_edge}칸 ÷ 전체 {n_all}칸 = {pct:.1f}%")


if __name__ == "__main__":
    main()
