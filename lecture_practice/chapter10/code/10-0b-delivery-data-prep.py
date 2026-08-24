"""
10장 준비(비즈니스 예제): 배달 권역 분석용 실데이터 스냅샷 만들기
=====================================================================
두 개의 국내 공개 원자료를 읽어, 500m 격자 단위 **수요·후보지 스냅샷** 하나로
줄인다. 분석 코드(10-4)는 이 스냅샷만 읽으므로 네트워크 없이 재현된다.

  ① 소상공인시장진흥공단 상가(상권)정보 — 점포 위치·업종(경도·위도 포함)
     https://www.data.go.kr/data/15083033/fileData.do
  ② 서울 생활인구(행정동 단위) — 시간대별 생활인구
     https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do

원자료는 용량이 커서(합계 수백 MB) 저장소에 넣지 않는다(.gitignore).
저장소 안 어느 장의 `data/raw/` 아래에 두어도 이 스크립트가 찾는다.
아직 없다면 위 두 주소에서 한 번 직접 내려받아 `../data/raw/`에 둔다.

무엇을 만드는가
  - 대상 자치구의 점포 좌표를 500m 격자로 묶어 **활동 격자**를 정의한다.
  - 행정동 생활인구를 그 격자들에 **POI 밀도 가중**으로 내린다(면적 균등 배분은
    대조군으로 함께 저장한다 — 배분 규칙이 결론을 만들었는지 확인하기 위해).
  - 음식점 밀집 격자(픽업 지점)와 보건의료·체육 업종 POI(생활 인프라 대리지표)를
    함께 남긴다.

정직 고지
  - 생활인구는 '그 시간에 그곳에 있는 사람'이지 '배달 주문'이 아니다.
    인구 → 주문 환산은 분석 코드의 **가정값**이다.
  - POI 밀도 가중 배분은 상업 밀도를 주거 수요의 대리로 쓰는 것이다.
    이 선택의 영향은 균등 배분 대조군으로 측정한다.

실행:
    python 10-0b-delivery-data-prep.py      # 이 준비 단계가 10-4보다 먼저 돌아야 한다
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PRACTICE_DIR = SCRIPT_DIR.parent.parent          # lecture_practice/

# ---------------------------------------------------------------- 설정
TARGET_SIGUNGU = ("종로구", "서대문구", "은평구")   # 도심 → 외곽으로 이어지는 인접 3구
CELL_M = 500.0                                    # 격자 한 변(m) — 10.7 분석 1과 같은 크기
MIN_POI_PER_CELL = 3                              # 점포가 너무 적은 셀은 활동 격자로 보지 않는다
OFFPEAK_HOURS = (14, 16)                          # 평시(오후)
PEAK_HOURS = (18, 20)                             # 저녁 배달 피크
FOOD_KEYWORDS = ("음식",)                          # 픽업 지점 = 음식점
INFRA_KEYWORDS = ("보건", "의료", "스포츠", "체육")  # 생활 인프라 대리지표(민간 시설)
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")
EPSG_WGS84, EPSG_KOREA = 4326, 5179               # 5179 = Korea 2000 / Unified CS(m)


def nfc(s: str) -> str:
    """윈도우·macOS의 한글 파일명 정규화 차이를 흡수한다(NFD ↔ NFC)."""
    return unicodedata.normalize("NFC", s)


def find_raw(patterns: tuple[str, ...], label: str, prefer: str | None = None) -> Path:
    """저장소 안 모든 `data/raw/`를 훑어 파일명에 패턴이 든 CSV를 찾는다."""
    cands = [p for p in PRACTICE_DIR.glob("*/data/raw/**/*.csv")
             if any(k in nfc(p.name) for k in patterns)]
    if not cands:
        sys.exit(
            f"[중단] {label} 원자료를 찾지 못했다.\n"
            f"  찾는 이름: {' 또는 '.join(patterns)} 가 들어간 .csv\n"
            f"  찾는 위치: lecture_practice/*/data/raw/ (하위 폴더 포함)\n"
            f"  → 이 파일 머리말의 주소에서 내려받아 "
            f"{(DATA_DIR / 'raw').relative_to(PRACTICE_DIR.parent)} 에 둔다.")
    if prefer:
        pref = [p for p in cands if prefer in nfc(p.name)]
        if pref:
            return max(pref, key=lambda p: p.stat().st_size)
    return max(cands, key=lambda p: p.stat().st_size)


def sniff_encoding(path: Path) -> str:
    """국내 공공 CSV는 cp949와 utf-8이 섞여 있어 헤더로 판별한다."""
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, encoding=enc, nrows=1, low_memory=False)
            return enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했다: {path.name}")


def resolve_columns(cols, needed: dict[str, tuple[str, ...]], label: str) -> dict[str, str]:
    """필요한 열을 후보 이름으로 찾는다. 못 찾으면 실제 열 목록을 보여 주고 멈춘다.

    공공 데이터는 갱신하면서 열 이름이 바뀐다. 조용히 잘못된 열을 쓰는 것보다
    멈추는 편이 낫다.
    """
    resolved, missing = {}, []
    for key, cands in needed.items():
        hit = next((c for c in cands if c in cols), None)
        if hit is None:
            missing.append(f"{key}: {cands}")
        else:
            resolved[key] = hit
    if missing:
        sys.exit(f"[중단] {label}에서 필요한 열을 찾지 못했다.\n  못 찾은 것:\n    "
                 + "\n    ".join(missing) + f"\n  실제 열:\n    {list(cols)}")
    return resolved


def load_stores() -> tuple[pd.DataFrame, str]:
    """상가정보에서 대상 자치구의 점포를 뽑고 격자 좌표를 붙인다."""
    src = find_raw(("상가업소", "소상공인", "상권정보"), "상가(상권)정보", prefer="서울")
    enc = sniff_encoding(src)
    print(f"  원본: {nfc(src.name)} ({src.stat().st_size / 1e6:.0f} MB, {enc})")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = resolve_columns(head.columns, {
        "sigungu": ("시군구명",),
        "dong_code": ("행정동코드",),
        "dong": ("행정동명",),
        "major": ("상권업종대분류명",),
        "middle": ("상권업종중분류명",),
        "lon": ("경도",),
        "lat": ("위도",),
    }, "상가정보")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    df = df.rename(columns={v: k for k, v in col.items()})
    sel = df[df["sigungu"].isin(TARGET_SIGUNGU)].copy()
    if sel.empty:
        sys.exit(f"[중단] 대상 자치구가 없다: {TARGET_SIGUNGU}\n"
                 f"  예시: {list(df['sigungu'].dropna().unique()[:10])}")

    sel["lon"] = pd.to_numeric(sel["lon"], errors="coerce")
    sel["lat"] = pd.to_numeric(sel["lat"], errors="coerce")
    sel["dong_code"] = pd.to_numeric(sel["dong_code"], errors="coerce")
    sel = sel.dropna(subset=["lon", "lat", "dong_code"])
    sel["dong_code"] = sel["dong_code"].astype(int)

    # 업종 분류: 대분류가 갱신으로 바뀔 수 있어 중분류까지 함께 본다.
    label = (sel["major"].fillna("") + "|" + sel["middle"].fillna(""))
    sel["is_food"] = label.str.contains("|".join(FOOD_KEYWORDS))
    sel["is_infra"] = label.str.contains("|".join(INFRA_KEYWORDS))

    print(f"  대상 {len(TARGET_SIGUNGU)}개 구 점포 {len(sel):,}개 "
          f"(음식 {int(sel['is_food'].sum()):,} / 생활 인프라 {int(sel['is_infra'].sum()):,})")
    print("  업종 대분류 상위:")
    for name, cnt in sel["major"].value_counts().head(6).items():
        print(f"    {name}: {cnt:,}")

    # 좌표 투영: 미터 단위로 바꿔야 격자·거리 계산이 성립한다.
    tf = Transformer.from_crs(EPSG_WGS84, EPSG_KOREA, always_xy=True)
    sel["x_m"], sel["y_m"] = tf.transform(sel["lon"].values, sel["lat"].values)
    return sel.reset_index(drop=True), nfc(src.name)


def load_living_population() -> tuple[pd.DataFrame, str, str]:
    """생활인구에서 행정동별 평시·피크 평균 인구를 만든다(평일만)."""
    src = find_raw(("LOCAL_PEOPLE", "생활인구"), "생활인구(행정동)")
    enc = sniff_encoding(src)
    print(f"  원본: {nfc(src.name)} ({src.stat().st_size / 1e6:.0f} MB, {enc})")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = resolve_columns(head.columns, {
        "date": ("기준일ID", "기준일자"),
        "hour": ("시간대구분",),
        "dong_code": ("행정동코드",),
        "pop": ("총생활인구수",),
    }, "생활인구")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    df = df.rename(columns={v: k for k, v in col.items()})
    for c in ("hour", "dong_code", "pop"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["hour", "dong_code", "pop"])

    # 평일만 쓴다. 주말은 저녁 피크의 모양이 달라 두 시나리오의 대비가 흐려진다.
    dates = pd.to_datetime(df["date"].astype("int64").astype(str), format="%Y%m%d")
    df = df[dates.dt.dayofweek < 5]

    def avg(hours: tuple[int, int], name: str) -> pd.DataFrame:
        sub = df[df["hour"].between(*hours)]
        return (sub.groupby("dong_code", as_index=False)["pop"].mean()
                   .rename(columns={"pop": name}))

    out = avg(OFFPEAK_HOURS, "pop_offpeak").merge(avg(PEAK_HOURS, "pop_peak"),
                                                 on="dong_code", how="inner")
    out["dong_code"] = out["dong_code"].astype(int)
    period = f"{int(df['date'].min())}~{int(df['date'].max())}"
    print(f"  행정동 {len(out):,}개 | 평일 기간 {period} | "
          f"평시 {OFFPEAK_HOURS[0]}~{OFFPEAK_HOURS[1]}시, 피크 {PEAK_HOURS[0]}~{PEAK_HOURS[1]}시")
    return out, nfc(src.name), period


def build_grid(stores: pd.DataFrame) -> pd.DataFrame:
    """점포 좌표를 500m 격자로 묶어 활동 격자를 만든다."""
    x0 = np.floor(stores["x_m"].min() / CELL_M) * CELL_M
    y0 = np.floor(stores["y_m"].min() / CELL_M) * CELL_M
    stores = stores.assign(
        gx=((stores["x_m"] - x0) // CELL_M).astype(int),
        gy=((stores["y_m"] - y0) // CELL_M).astype(int),
    )

    grouped = stores.groupby(["gx", "gy"])
    grid = grouped.agg(n_poi=("is_food", "size"),
                       n_food=("is_food", "sum"),
                       n_infra=("is_infra", "sum")).reset_index()
    # 셀의 행정동: 그 셀 점포들의 최빈 행정동(정책 단위 집계에 쓴다)
    dom = grouped["dong_code"].agg(lambda s: s.mode().iat[0]).rename("dong_code").reset_index()
    name = (stores.groupby("dong_code")["dong"].agg(lambda s: s.mode().iat[0]).rename("dong"))
    sig = (stores.groupby("dong_code")["sigungu"].agg(lambda s: s.mode().iat[0]).rename("sigungu"))
    grid = grid.merge(dom, on=["gx", "gy"]).merge(name, on="dong_code").merge(sig, on="dong_code")

    dropped = grid[grid["n_poi"] < MIN_POI_PER_CELL]
    print(f"  격자 후보 {len(grid):,}개 → 점포 {MIN_POI_PER_CELL}개 미만 셀 "
          f"{len(dropped):,}개 제외(제외 셀의 점포 {int(dropped['n_poi'].sum()):,}개)")
    grid = grid[grid["n_poi"] >= MIN_POI_PER_CELL].copy()

    grid["x_m"] = x0 + (grid["gx"] + 0.5) * CELL_M          # 셀 중심 좌표
    grid["y_m"] = y0 + (grid["gy"] + 0.5) * CELL_M
    grid = grid.sort_values(["gy", "gx"]).reset_index(drop=True)
    grid.insert(0, "cell_id", range(len(grid)))
    return grid


def allocate_population(grid: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """행정동 생활인구를 격자로 내린다. POI 밀도 가중(본안)과 균등(대조군) 둘 다.

    두 배분을 같이 만드는 이유: '컷라인 밖 인구 비율'이 배분 규칙의 산물인지
    확인해야 한다. 대조군이 없으면 자기 확인에 그친다.
    """
    g = grid.merge(pop, on="dong_code", how="left")
    missing = g["pop_offpeak"].isna()
    if missing.any():
        lost = sorted(set(g.loc[missing, "dong"]))
        print(f"  ⚠ 생활인구와 결합 실패 행정동 {len(lost)}개 → 해당 격자 제외: {lost[:6]}")
        g = g[~missing].copy()

    for base in ("pop_offpeak", "pop_peak"):
        w = g.groupby("dong_code")["n_poi"].transform("sum")
        g[f"{base}_cell"] = g[base] * g["n_poi"] / w
        n = g.groupby("dong_code")["cell_id"].transform("size")
        g[f"{base}_cell_uniform"] = g[base] / n

    keep = ["cell_id", "gx", "gy", "x_m", "y_m", "dong_code", "dong", "sigungu",
            "n_poi", "n_food", "n_infra",
            "pop_offpeak_cell", "pop_peak_cell",
            "pop_offpeak_cell_uniform", "pop_peak_cell_uniform"]
    out = g[keep].rename(columns={"pop_offpeak_cell": "pop_offpeak",
                                  "pop_peak_cell": "pop_peak",
                                  "pop_offpeak_cell_uniform": "pop_offpeak_uniform",
                                  "pop_peak_cell_uniform": "pop_peak_uniform"})
    return out.reset_index(drop=True)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    print("=" * 68)
    print("10-0b 준비: 배달 권역 분석용 실데이터 스냅샷")
    print("=" * 68)
    print(f"대상: 서울 {' · '.join(TARGET_SIGUNGU)} | 격자 {CELL_M:.0f}m")

    print("\n[1/3] 상가(상권)정보")
    stores, stores_src = load_stores()

    print("\n[2/3] 생활인구(행정동)")
    pop, pop_src, period = load_living_population()

    print("\n[3/3] 격자 구성과 인구 배분")
    grid = build_grid(stores)
    matched = grid["dong_code"].isin(set(pop["dong_code"]))
    print(f"  행정동코드 결합률: {matched.mean():.1%} ({int(matched.sum())}/{len(grid)})")
    cells = allocate_population(grid, pop)

    infra = stores.loc[stores["is_infra"], ["x_m", "y_m"]].reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    grid_path = DATA_DIR / "delivery_grid.parquet"
    infra_path = DATA_DIR / "delivery_infra.parquet"
    cells.to_parquet(grid_path, index=False)
    infra.to_parquet(infra_path, index=False)

    meta = {
        "생성일": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "대상_자치구": list(TARGET_SIGUNGU),
        "격자_m": CELL_M,
        "셀_최소_점포수": MIN_POI_PER_CELL,
        "평시_시간대": list(OFFPEAK_HOURS),
        "피크_시간대": list(PEAK_HOURS),
        "원본_상가정보": stores_src,
        "원본_생활인구": f"{pop_src} (평일 {period})",
        "격자_수": int(len(cells)),
        "행정동_수": int(cells["dong_code"].nunique()),
        "총_평시인구": float(cells["pop_offpeak"].sum()),
        "총_피크인구": float(cells["pop_peak"].sum()),
        "생활인프라_POI": int(len(infra)),
        "스냅샷_sha256": sha256_of_file(grid_path),
    }
    (DATA_DIR / "delivery_snapshot_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 68)
    print(f"활동 격자 {len(cells):,}개 | 행정동 {cells['dong_code'].nunique()}개 | "
          f"생활 인프라 POI {len(infra):,}개")
    print(f"평시 총 생활인구 {cells['pop_offpeak'].sum():,.0f}명 → "
          f"피크 {cells['pop_peak'].sum():,.0f}명 "
          f"({cells['pop_peak'].sum() / cells['pop_offpeak'].sum() - 1:+.1%})")
    print(f"셀 평시인구: 중앙값 {cells['pop_offpeak'].median():,.0f} / "
          f"최대 {cells['pop_offpeak'].max():,.0f}")
    print(f"저장: {grid_path.name}, {infra_path.name}, delivery_snapshot_meta.json")
    print(f"스냅샷 sha256: {meta['스냅샷_sha256'][:16]}…")
    print("공개 데이터가 갱신되면 수치가 달라진다. 재현의 기준은 이 스냅샷이다.")
    print("=" * 68)
    print("[완료] 다음: python 10-4-delivery-zone-optimization.py")


if __name__ == "__main__":
    main()
