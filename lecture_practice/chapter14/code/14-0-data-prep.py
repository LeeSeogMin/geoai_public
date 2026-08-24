"""
14장 준비: 국내 공개 데이터를 입지 분석용 파일로 정리
=====================================================
원자료 두 개(소상공인 상가정보, 서울 생활인구 행정동)를 읽어 한 자치구·한 업종으로
좁힌 분석용 파일을 만든다. 분석 코드(14-1)는 이 결과만 읽는다.

내려받는 방법은 ../data/raw/README.md 참조. 두 포털 모두 자동 다운로드가 막혀
있어 한 번은 직접 받아야 한다.

원자료가 크므로(서울 상가정보 약 300MB, 생활인구 약 180MB) 헤더를 먼저 읽어
필요한 열만 가져온다.

재현성: 원본 파일명과 처리 일자를 SOURCES.txt에 남긴다. 공개 데이터가 갱신되면
결과가 달라지므로, 본문 수치는 이 기준과 함께 인용한다.

실행:
    python 14-0-data-prep.py
"""

from pathlib import Path
import sys
import unicodedata

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# 분석 대상
TARGET_SIGUNGU = "강남구"
CAFE_SUBCATEGORY = "카페"          # 소분류. '독서실/스터디 카페'는 성격이 달라 제외한다
DEMAND_HOURS = (11, 15)            # 카페 수요의 대리: 평일 낮 활동 인구
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")


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
    """파일명에 패턴이 들어간 CSV를 하위 폴더까지 찾는다."""
    if not RAW_DIR.exists():
        sys.exit(f"[중단] {RAW_DIR} 폴더가 없다. data/raw/README.md의 절차를 먼저 밟는다.")
    cands = [p for p in RAW_DIR.rglob("*.csv")
             if any(k in unicodedata.normalize("NFC", p.name) for k in patterns)]
    if not cands:
        listing = "\n".join(f"    - {p.relative_to(RAW_DIR)}" for p in RAW_DIR.rglob("*")
                            if p.is_file()) or "    (비어 있음)"
        sys.exit(f"[중단] {label} 파일을 찾지 못했다.\n"
                 f"  찾는 이름: {' 또는 '.join(patterns)} 가 들어간 .csv\n"
                 f"  raw 폴더 내용:\n{listing}\n  → data/raw/README.md 참조")
    if prefer:
        pref = [p for p in cands if prefer in unicodedata.normalize("NFC", p.name)]
        if pref:
            return max(pref, key=lambda p: p.stat().st_size)
    return max(cands, key=lambda p: p.stat().st_size)


def require_columns(cols, needed: dict[str, tuple[str, ...]], label: str) -> dict[str, str]:
    """필요한 열을 후보 이름으로 찾는다.

    공공 데이터는 갱신하면서 열 이름이 바뀐다. 못 찾으면 실제 열 목록을 보여 주고
    멈춘다 — 조용히 잘못된 열을 쓰는 것보다 낫다.
    """
    resolved, missing = {}, []
    for key, cands in needed.items():
        hit = next((c for c in cands if c in cols), None)
        (missing.append(f"{key}: {cands}") if hit is None else resolved.update({key: hit}))
    if missing:
        sys.exit(f"[중단] {label}에서 필요한 열을 찾지 못했다.\n  못 찾은 것:\n    "
                 + "\n    ".join(missing) + f"\n  실제 열({len(cols)}개):\n    {list(cols)}")
    return resolved


def prepare_stores() -> tuple[pd.DataFrame, str]:
    """상가정보에서 대상 자치구의 카페를 뽑는다."""
    src = find_raw(("상가업소", "소상공인", "상권정보"), "상가(상권)정보", prefer="서울")
    enc = sniff_encoding(src)
    print(f"  원본: {src.name} ({src.stat().st_size / 1e6:.0f} MB, {enc})")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = require_columns(head.columns, {
        "sigungu": ("시군구명",),
        "sub": ("상권업종소분류명",),
        "name": ("상호명",),
        "dong_code": ("행정동코드",),
        "dong": ("행정동명",),
        "lon": ("경도",),
        "lat": ("위도",),
    }, "상가정보")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    sel = df[df[col["sigungu"]] == TARGET_SIGUNGU]
    if sel.empty:
        sys.exit(f"[중단] '{TARGET_SIGUNGU}'가 없다. 예시: {list(df[col['sigungu']].dropna().unique()[:10])}")

    cafes = sel[sel[col["sub"]] == CAFE_SUBCATEGORY]
    out = pd.DataFrame({
        "name": cafes[col["name"]].astype(str).values,
        "dong_code": pd.to_numeric(cafes[col["dong_code"]], errors="coerce").values,
        "dong": cafes[col["dong"]].astype(str).values,
        "lon": pd.to_numeric(cafes[col["lon"]], errors="coerce").values,
        "lat": pd.to_numeric(cafes[col["lat"]], errors="coerce").values,
    }).dropna().reset_index(drop=True)
    out["dong_code"] = out["dong_code"].astype(int)
    out.insert(0, "store_id", range(len(out)))

    print(f"  {TARGET_SIGUNGU} '{CAFE_SUBCATEGORY}' 점포 {len(out)}개, 행정동 {out['dong'].nunique()}개")
    top = out["name"].value_counts().head(3)
    print(f"  상호명 상위: {dict(top)}")
    return out, src.name


def prepare_demand() -> tuple[pd.DataFrame, str]:
    """생활인구에서 행정동별 평일 낮 활동 인구를 만든다."""
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

    day = df[df["hour"].between(*DEMAND_HOURS)]
    out = (day.groupby("dong_code", as_index=False)["pop"].mean()
              .rename(columns={"pop": "day_pop"}))
    out["dong_code"] = out["dong_code"].astype(int)

    period = f"{int(df['date'].min())}~{int(df['date'].max())}"
    print(f"  행정동 {len(out)}개 | 기간 {period} | {DEMAND_HOURS[0]}~{DEMAND_HOURS[1]}시 평균")
    return out, f"{src.name} ({period})"


def main() -> None:
    print("=" * 64)
    print("14장 준비: 국내 공개 데이터 → 입지 분석용 파일")
    print("=" * 64)

    print("\n[1/2] 상가(상권)정보")
    stores, stores_src = prepare_stores()

    print("\n[2/2] 생활인구(행정동)")
    demand, demand_src = prepare_demand()

    # 두 자료의 행정동코드는 8자리로 체계가 같다. 결합 성공률을 반드시 확인한다.
    matched = stores["dong_code"].isin(set(demand["dong_code"]))
    print(f"\n결합 점검: 점포의 행정동코드 중 생활인구와 매칭 {matched.mean():.1%} "
          f"({matched.sum()}/{len(stores)})")
    if matched.mean() < 0.9:
        miss = sorted(set(stores.loc[~matched, "dong"]))[:8]
        print(f"  ⚠ 매칭 실패 행정동 예: {miss}")

    DATA_DIR.mkdir(exist_ok=True)
    stores.to_parquet(DATA_DIR / "cafes.parquet", index=False)
    demand.to_parquet(DATA_DIR / "demand_dong.parquet", index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    (DATA_DIR / "SOURCES.txt").write_text(
        "14장 분석용 파일의 출처\n"
        f"- 상가(상권)정보: {stores_src}\n"
        f"- 생활인구(행정동): {demand_src}\n"
        f"- 처리 일자: {stamp}\n"
        f"- 대상: 서울 {TARGET_SIGUNGU}, 소분류 '{CAFE_SUBCATEGORY}'\n"
        f"- 수요 정의: 평일 낮 {DEMAND_HOURS[0]}~{DEMAND_HOURS[1]}시 평균 생활인구\n"
        "공개 데이터가 갱신되면 결과가 달라진다. 본문 수치는 이 기준으로 인용한다.\n",
        encoding="utf-8")

    print("-" * 64)
    print(f"저장: cafes.parquet({len(stores)}행), demand_dong.parquet({len(demand)}행), SOURCES.txt")
    print("=" * 64)
    print("[완료] 다음: python 14-1-huff-location-cannibalization.py")


if __name__ == "__main__":
    main()
