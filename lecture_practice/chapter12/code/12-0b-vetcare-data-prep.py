"""
12-0b. 비즈니스 분석 데이터 준비: 서울 동물병원 입지 분석용 실데이터 정리
=========================================================================
12-5(미충족 수요와 제약 하 순차 입지 선택)가 쓰는 분석용 파일을 만든다.
12-0(합성 데이터)과 달리 이 스크립트는 **국내 공개 실데이터**를 읽는다.

두 원자료를 쓴다.

① 소상공인시장진흥공단 상가(상권)정보 — 서울 파일
   기존 동물병원의 위치와 업종 분류, 그리고 후보지 격자를 만들 때 쓰는
   '상업 공간이 있는 칸'의 근거가 된다.

② 서울 열린데이터광장 행정동 단위 생활인구(내국인)
   배후 수요의 근거다. 동물병원 수요는 카페처럼 낮에 오는 사람이 아니라
   **그 동네에 사는 사람**에서 나오므로, 심야(0~5시) 평균 생활인구를 쓴다.
   그 시간에 그곳에 있는 사람은 대체로 그곳에 사는 사람이다.

원자료는 저장소에 포함하지 않는다. `../data/raw/README.md`의 절차를 한 번 밟으면
되고, 같은 원자료를 쓰는 14장 폴더에 이미 있으면 그쪽을 자동으로 찾아 쓴다.

만드는 파일 세 개:
  - vet_clinics.parquet       기존 동물병원 (좌표, 자치구, 행정동)
  - vet_demand_dong.parquet   행정동별 심야 생활인구와 대표점
  - vet_candidates.parquet    250m 후보지 격자 (상업 공간이 있는 칸만)

재현성: 원본 파일명·기준월·처리 일자를 SOURCES_vet.txt에 남긴다. 공개 데이터가
갱신되면 결과가 달라지므로 본문 수치는 이 기준과 함께 인용한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    python lecture_practice/chapter12/code/12-0b-vetcare-data-prep.py
"""

from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# 원자료 탐색 순서: 12장 폴더 → 14장 폴더(같은 원자료를 쓰는 장).
# 480MB 원자료를 저장소에 두 벌 두지 않기 위한 장치다. 어느 쪽을 썼는지 로그에 남긴다.
RAW_ROOTS = (DATA_DIR / "raw",
             SCRIPT_DIR.parents[1] / "chapter14" / "data" / "raw")

TARGET_SIDO_KEYS = ("서울",)      # 원본 파일명으로 서울 파일을 고른다
NIGHT_HOURS = (0, 5)              # 심야 재실 = 상주 인구의 대리
GRID_M = 250                      # 후보지 격자 크기(m)
MIN_STORES_PER_CELL = 5           # '상업 공간이 있는 칸'의 기준
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")
CRS_TO = "EPSG:5179"              # 한국 중부원점(m). 거리 계산은 투영좌표에서 한다


def sniff_encoding(path: Path) -> str:
    """국내 공공 CSV의 인코딩이 제각각이라 헤더로 판별한다."""
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, encoding=enc, nrows=1, low_memory=False)
            return enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했다: {path.name}")


def find_raw(patterns: tuple[str, ...], label: str, prefer: str | None = None) -> Path:
    """파일명에 패턴이 들어간 CSV를 RAW_ROOTS에서 차례로 찾는다."""
    for root in RAW_ROOTS:
        if not root.exists():
            continue
        cands = [p for p in root.rglob("*.csv")
                 if any(k in unicodedata.normalize("NFC", p.name) for k in patterns)]
        if not cands:
            continue
        if prefer:
            pref = [p for p in cands if prefer in unicodedata.normalize("NFC", p.name)]
            if pref:
                return max(pref, key=lambda p: p.stat().st_size)
        return max(cands, key=lambda p: p.stat().st_size)

    searched = "\n".join(f"    - {r}" for r in RAW_ROOTS)
    sys.exit(f"[중단] {label} 파일을 찾지 못했다.\n"
             f"  찾는 이름: {' 또는 '.join(patterns)} 가 들어간 .csv\n"
             f"  찾아본 폴더:\n{searched}\n"
             f"  → lecture_practice/chapter12/data/raw/README.md 의 내려받기 절차를 먼저 밟는다.")


def require_columns(cols, needed: dict[str, tuple[str, ...]], label: str) -> dict[str, str]:
    """필요한 열을 후보 이름으로 찾는다.

    공공 데이터는 갱신하면서 열 이름이 바뀐다. 못 찾으면 실제 열 목록을 보여 주고
    멈춘다 — 조용히 잘못된 열을 쓰는 것보다 낫다.
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
                 + "\n    ".join(missing) + f"\n  실제 열({len(cols)}개):\n    {list(cols)}")
    return resolved


def resolve_vet_category(sub_series: pd.Series) -> str:
    """상가정보의 업종 소분류에서 동물병원에 해당하는 분류명을 확정한다.

    분류 체계가 갱신될 수 있으므로 이름을 하드코딩하지 않는다. 정확히 '동물병원'이
    있으면 그것을 쓰고, 없으면 '동물병원'을 포함하는 분류를 찾고, 그래도 없으면
    동물·수의 관련 분류 목록을 보여 주고 멈춘다.
    """
    values = sub_series.dropna().astype(str)
    uniq = set(values.unique())
    if "동물병원" in uniq:
        return "동물병원"
    contains = sorted(v for v in uniq if "동물병원" in v)
    if len(contains) == 1:
        print(f"  ! 소분류명이 '동물병원'이 아니라 '{contains[0]}'이다. 그대로 쓴다.")
        return contains[0]
    related = sorted(v for v in uniq if any(k in v for k in ("동물", "수의", "애견", "반려")))
    sys.exit("[중단] 업종 소분류에서 동물병원을 특정하지 못했다.\n"
             f"  '동물병원' 포함 분류: {contains}\n"
             f"  동물/수의/애견/반려 포함 분류: {related}\n"
             "  → 분류 체계가 바뀌었을 수 있다. 위 목록을 보고 대상 분류를 확정한다.")


def prepare_stores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """상가정보에서 동물병원·행정동 대표점·후보지 격자를 만든다."""
    src = find_raw(("상가업소", "소상공인", "상권정보"), "상가(상권)정보",
                   prefer=TARGET_SIDO_KEYS[0])
    enc = sniff_encoding(src)
    print(f"  원본: {src.name} ({src.stat().st_size / 1e6:.0f} MB, {enc})")
    print(f"  경로: {src.parent}")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = require_columns(head.columns, {
        "sido": ("시도명",),
        "sigungu": ("시군구명",),
        "major": ("상권업종대분류명",),
        "middle": ("상권업종중분류명",),
        "sub": ("상권업종소분류명",),
        "name": ("상호명",),
        "dong_code": ("행정동코드",),
        "dong": ("행정동명",),
        "lon": ("경도",),
        "lat": ("위도",),
    }, "상가정보")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    df = df.rename(columns={v: k for k, v in col.items()})
    df = df[df["sido"].astype(str).str.contains(TARGET_SIDO_KEYS[0], na=False)]
    print(f"  서울 전체 점포 {len(df):,}개, 자치구 {df['sigungu'].nunique()}개")

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["dong_code"] = pd.to_numeric(df["dong_code"], errors="coerce")
    coord_missing = df[["lon", "lat"]].isna().any(axis=1).mean()
    print(f"  좌표 결측률(전 업종) {coord_missing:.3%}")
    df = df.dropna(subset=["lon", "lat", "dong_code"]).copy()
    df["dong_code"] = df["dong_code"].astype(np.int64)

    # 투영좌표(m). 거리 계산과 격자 생성은 여기서 한다
    tf = Transformer.from_crs("EPSG:4326", CRS_TO, always_xy=True)
    df["x"], df["y"] = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())

    # --- ① 동물병원 ---
    vet_label = resolve_vet_category(df["sub"])
    vet = df[df["sub"].astype(str) == vet_label].copy()
    if vet.empty:
        sys.exit(f"[중단] 소분류 '{vet_label}' 점포가 서울에 없다.")
    clinics = (vet[["name", "sigungu", "dong_code", "dong", "lon", "lat", "x", "y"]]
               .reset_index(drop=True))
    clinics.insert(0, "clinic_id", range(len(clinics)))
    print(f"  동물병원(소분류 '{vet_label}') {len(clinics)}개 "
          f"| 자치구 {clinics['sigungu'].nunique()}개 | 행정동 {clinics['dong'].nunique()}개")
    print(f"    대분류/중분류: {sorted(set(vet['major'].astype(str)))} / "
          f"{sorted(set(vet['middle'].astype(str)))}")
    top5 = clinics["sigungu"].value_counts().head(5)
    bottom3 = clinics["sigungu"].value_counts().tail(3)
    print(f"    자치구 상위5: {dict(top5)}")
    print(f"    자치구 하위3: {dict(bottom3)}")

    # --- ② 행정동 대표점 ---
    # 동물병원 좌표로 대표점을 잡으면 '공급이 수요를 정의하는' 순환이 된다.
    # 전 업종 점포의 중심을 쓰면 그 동의 활동 중심에 대한 중립적 대리가 된다.
    rep = (df.groupby(["dong_code", "dong", "sigungu"], as_index=False)
             .agg(rep_lon=("lon", "mean"), rep_lat=("lat", "mean"),
                  rep_x=("x", "mean"), rep_y=("y", "mean"), n_stores=("name", "size")))
    print(f"  행정동 대표점 {len(rep)}개 (전 업종 점포 중심)")

    # --- ③ 후보지 격자 ---
    # 상업 공간이 없는 칸에는 개설할 수 없으므로, 점포가 일정 수 이상인 칸만 남긴다
    gx = np.floor(df["x"].to_numpy() / GRID_M).astype(np.int64)
    gy = np.floor(df["y"].to_numpy() / GRID_M).astype(np.int64)
    cell = pd.DataFrame({"gx": gx, "gy": gy})
    counts = cell.groupby(["gx", "gy"], as_index=False).size()
    print(f"  점포가 1개 이상인 {GRID_M}m 격자 {len(counts):,}칸")
    cand = counts[counts["size"] >= MIN_STORES_PER_CELL].reset_index(drop=True)
    cand["x"] = cand["gx"] * GRID_M + GRID_M / 2
    cand["y"] = cand["gy"] * GRID_M + GRID_M / 2
    inv = Transformer.from_crs(CRS_TO, "EPSG:4326", always_xy=True)
    cand["lon"], cand["lat"] = inv.transform(cand["x"].to_numpy(), cand["y"].to_numpy())
    cand = cand.rename(columns={"size": "n_stores"})
    cand.insert(0, "cand_id", range(len(cand)))
    print(f"  후보지: 점포 {MIN_STORES_PER_CELL}개 이상인 칸 {len(cand):,}개")

    return clinics, rep, cand, src.name, f"{src.name} (좌표 결측 {coord_missing:.3%})"


def prepare_night_population() -> tuple[pd.DataFrame, str]:
    """생활인구에서 행정동별 심야 평균 인구를 만든다(상주 인구의 대리)."""
    src = find_raw(("LOCAL_PEOPLE", "생활인구"), "생활인구(행정동)")
    enc = sniff_encoding(src)
    print(f"  원본: {src.name} ({src.stat().st_size / 1e6:.0f} MB, {enc})")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = require_columns(head.columns, {
        "date": ("기준일ID", "기준일자"),
        "hour": ("시간대구분",),
        "dong_code": ("행정동코드",),
        "pop": ("총생활인구수",),
    }, "생활인구")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    df = df.rename(columns={v: k for k, v in col.items()})
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["pop", "hour"])

    night = df[df["hour"].between(*NIGHT_HOURS)]
    out = (night.groupby("dong_code", as_index=False)["pop"].mean()
                .rename(columns={"pop": "night_pop"}))
    out["dong_code"] = out["dong_code"].astype(np.int64)

    period = f"{int(df['date'].min())}~{int(df['date'].max())}"
    print(f"  행정동 {len(out)}개 | 기간 {period} | {NIGHT_HOURS[0]}~{NIGHT_HOURS[1]}시 평균")
    print(f"  서울 심야 생활인구 합계 {out['night_pop'].sum():,.0f}명")
    return out, f"{src.name} ({period}, {NIGHT_HOURS[0]}~{NIGHT_HOURS[1]}시)"


def main() -> None:
    print("=" * 70)
    print("12-0b 준비: 서울 동물병원 입지 분석용 실데이터")
    print("=" * 70)

    print("\n[1/2] 상가(상권)정보 — 동물병원·행정동 대표점·후보지")
    clinics, rep, cand, stores_src, stores_note = prepare_stores()

    print("\n[2/2] 생활인구(행정동) — 심야 배후 인구")
    night, night_src = prepare_night_population()

    # 두 자료의 행정동코드는 8자리로 체계가 같다. 결합 성공률을 반드시 확인한다
    demand = rep.merge(night, on="dong_code", how="inner")
    rate = len(demand) / len(rep)
    print(f"\n결합 점검: 대표점 행정동 중 생활인구와 매칭 {rate:.1%} ({len(demand)}/{len(rep)})")
    if rate < 0.9:
        miss = sorted(set(rep.loc[~rep["dong_code"].isin(night["dong_code"]), "dong"]))[:8]
        print(f"  ⚠ 매칭 실패 행정동 예: {miss}")
    clinic_match = clinics["dong_code"].isin(set(demand["dong_code"])).mean()
    print(f"  동물병원의 행정동 매칭 {clinic_match:.1%}")

    DATA_DIR.mkdir(exist_ok=True)
    clinics.to_parquet(DATA_DIR / "vet_clinics.parquet", index=False)
    demand.to_parquet(DATA_DIR / "vet_demand_dong.parquet", index=False)
    cand.to_parquet(DATA_DIR / "vet_candidates.parquet", index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    (DATA_DIR / "SOURCES_vet.txt").write_text(
        "12장 비즈니스 분석(동물병원 입지) 원자료 출처\n"
        f"- 상가(상권)정보: {stores_note}\n"
        f"- 생활인구(행정동): {night_src}\n"
        f"- 처리 일자: {stamp}\n"
        f"- 대상: 서울 전역, 업종 소분류 '동물병원'\n"
        f"- 수요 정의: 심야 {NIGHT_HOURS[0]}~{NIGHT_HOURS[1]}시 평균 생활인구(상주 인구 대리)\n"
        f"- 후보지: {GRID_M}m 격자 중 점포 {MIN_STORES_PER_CELL}개 이상인 칸\n"
        "공개 데이터가 갱신되면 결과가 달라진다. 본문 수치는 이 기준으로 인용한다.\n",
        encoding="utf-8")

    print("-" * 70)
    print(f"저장: vet_clinics.parquet({len(clinics)}행), "
          f"vet_demand_dong.parquet({len(demand)}행), "
          f"vet_candidates.parquet({len(cand)}행), SOURCES_vet.txt")
    print("=" * 70)
    print("[완료] 다음: python 12-5-unmet-demand-siting.py")


if __name__ == "__main__":
    main()
