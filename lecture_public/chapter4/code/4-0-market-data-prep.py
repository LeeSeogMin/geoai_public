"""
4장 준비: 국내 공개 데이터를 행정동 단위 입지 분석용 파일로 정리
================================================================
원자료 두 개(소상공인 상가정보, 서울 생활인구 행정동)를 읽어 **서울 전역 행정동**
단위로 집계한다. 분석 코드(4-6)는 이 결과만 읽는다.

14장도 같은 원자료를 쓴다. 다만 대상이 다르다 — 14장은 강남구 한 구의 점포 하나하나를
다루고, 이 장은 서울 전역을 행정동으로 묶어 425개 안팎의 공간단위 표를 만든다.
14장 원자료를 이미 받아 두었다면 그대로 재사용하므로 다시 내려받지 않아도 된다.

내려받는 방법은 ../data/raw/README.md 참조.

원자료가 크므로(서울 상가정보 약 300MB, 생활인구 약 180MB) 필요한 열만 가져온다.

재현성: 원본 파일명과 처리 일자를 SOURCES.txt에 남긴다. 공개 데이터가 갱신되면
결과가 달라지므로, 본문 수치는 이 기준과 함께 인용한다.

실행:
    python 4-0-market-data-prep.py
"""

from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
# 14장이 같은 원자료를 쓴다. 이미 받아 두었으면 재사용한다.
RAW_DIRS = (DATA_DIR / "raw",
            SCRIPT_DIR.parent.parent / "chapter14" / "data" / "raw")

TARGET_SIDO = "서울"
CAFE_SUBCATEGORY = "카페"      # 소분류. '독서실/스터디 카페'는 성격이 달라 제외된다
DAY_HOURS = (11, 15)           # 낮 활동 인구
NIGHT_HOURS = (0, 6)           # 심야 인구 ≈ 거주 인구의 대리
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

# 서울시청 (EPSG:4326). 도심으로부터의 거리를 재는 기준점으로만 쓴다.
CITY_HALL = (126.9780, 37.5665)


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
    """파일명에 패턴이 들어간 CSV를 raw 후보 폴더에서 찾는다."""
    cands: list[Path] = []
    for root in RAW_DIRS:
        if root.exists():
            cands += [p for p in root.rglob("*.csv")
                      if any(k in unicodedata.normalize("NFC", p.name) for k in patterns)]
    if not cands:
        looked = "\n".join(f"    - {r}" for r in RAW_DIRS)
        sys.exit(f"[중단] {label} 파일을 찾지 못했다.\n"
                 f"  찾는 이름: {' 또는 '.join(patterns)} 가 들어간 .csv\n"
                 f"  찾아본 곳:\n{looked}\n  → practice/chapter4/data/raw/README.md 참조")
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
    """상가정보를 행정동 단위로 집계한다.

    집계 결과는 둘이다.
    - n_cafe: 이 분석의 결과변수. 행정동별 카페 점포 수
    - 중심 좌표: 행정동의 위치를 나타내는 기하 정보. 전 업종 점포 좌표의 평균이다.
      행정동 경계 파일 없이 위치를 잡기 위한 근사이며, 피처로 쓰는 것은 좌표 자체이지
      점포 수가 아니다(공급량을 피처로 넣으면 '가게 많은 곳에 가게 많다'가 된다).
    """
    src = find_raw(("상가업소", "소상공인", "상권정보"), "상가(상권)정보", prefer=TARGET_SIDO)
    enc = sniff_encoding(src)
    print(f"  원본: {src.name} ({src.stat().st_size / 1e6:.0f} MB, {enc})")

    head = pd.read_csv(src, encoding=enc, nrows=1, low_memory=False)
    col = require_columns(head.columns, {
        "sigungu": ("시군구명",),
        "sub": ("상권업종소분류명",),
        "dong_code": ("행정동코드",),
        "dong": ("행정동명",),
        "lon": ("경도",),
        "lat": ("위도",),
    }, "상가정보")

    df = pd.read_csv(src, encoding=enc, low_memory=False, usecols=list(col.values()))
    df = df.rename(columns={v: k for k, v in col.items()})
    df["dong_code"] = pd.to_numeric(df["dong_code"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["dong_code", "lon", "lat"])
    df["dong_code"] = df["dong_code"].astype(int)
    print(f"  전체 점포 {len(df):,}개 | 자치구 {df['sigungu'].nunique()}개")

    is_cafe = df["sub"] == CAFE_SUBCATEGORY
    print(f"  소분류 '{CAFE_SUBCATEGORY}' {int(is_cafe.sum()):,}개 ({is_cafe.mean():.1%})")

    agg = df.groupby("dong_code").agg(
        sigungu=("sigungu", lambda s: s.mode().iat[0]),
        dong=("dong", lambda s: s.mode().iat[0]),
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        n_store=("sub", "size"),
    ).reset_index()
    agg["n_cafe"] = df[is_cafe].groupby("dong_code").size().reindex(agg["dong_code"]).fillna(0).values
    agg["n_cafe"] = agg["n_cafe"].astype(int)

    print(f"  행정동 {len(agg)}개로 집계 | 카페 수 중앙값 {agg['n_cafe'].median():.0f}, "
          f"최대 {agg['n_cafe'].max()}")
    return agg, src.name


def prepare_demand() -> tuple[pd.DataFrame, str]:
    """생활인구에서 행정동별 수요 피처를 만든다.

    상주 인구가 아니라 '그 시간에 그곳에 있는 사람'이다. 카페 수요는 거주자보다
    낮 시간 유동 인구에 달려 있으므로, 낮과 심야를 나누어 성격을 구분한다.
    """
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
    for c in ("hour", "pop", "dong_code"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["hour", "pop", "dong_code"])
    df["dong_code"] = df["dong_code"].astype(int)

    dt = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df["is_weekend"] = dt.dt.dayofweek >= 5

    day = df[df["hour"].between(*DAY_HOURS)]
    night = df[df["hour"].between(*NIGHT_HOURS)]

    out = day.groupby("dong_code", as_index=False)["pop"].mean().rename(columns={"pop": "day_pop"})
    out = out.merge(night.groupby("dong_code", as_index=False)["pop"].mean()
                        .rename(columns={"pop": "night_pop"}), on="dong_code", how="inner")

    # 주간/심야 비율: 1보다 크면 낮에 사람이 들어오는 업무·상업 지역,
    # 1보다 작으면 낮에 빠져나가는 주거 지역이다.
    out["day_night_ratio"] = out["day_pop"] / out["night_pop"].replace(0, np.nan)

    # 주말/평일 낮 비율: 주말에도 사람이 오는 곳인지 구분한다.
    wk = (day.groupby(["dong_code", "is_weekend"])["pop"].mean().unstack("is_weekend"))
    wk.columns = ["weekday_day", "weekend_day"]
    out = out.merge((wk["weekend_day"] / wk["weekday_day"].replace(0, np.nan))
                    .rename("weekend_ratio").reset_index(), on="dong_code", how="left")

    # 시간대 변동: 하루 안에서 인구가 얼마나 출렁이는지(체류 vs 통과)
    hourly = df.groupby(["dong_code", "hour"])["pop"].mean()
    swing = (hourly.groupby("dong_code").max() / hourly.groupby("dong_code").mean()).rename("peak_ratio")
    out = out.merge(swing.reset_index(), on="dong_code", how="left")

    period = f"{int(df['date'].min())}~{int(df['date'].max())}"
    print(f"  행정동 {len(out)}개 | 기간 {period}")
    print(f"  낮 {DAY_HOURS[0]}~{DAY_HOURS[1]}시 / 심야 {NIGHT_HOURS[0]}~{NIGHT_HOURS[1]}시 평균")
    return out, f"{src.name} ({period})"


def add_spatial_features(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """공간 피처를 더한다 — 도심 거리와 이웃 동의 공간 시차.

    행정동 경계 파일이 없으므로 인접성을 중심점 사이의 k-최근접으로 근사한다.
    경계 인접(rook/queen)과 완전히 같지는 않으나, 4.2절이 말한 '이웃의 상태가
    추가 정보를 갖는다'는 성질을 확인하는 데는 충분하다.
    """
    # 경위도를 미터로 근사. 서울 위도(약 37.5°)에서 경도 1° ≈ 88.4km, 위도 1° ≈ 111.0km
    lat0 = np.deg2rad(df["lat"].mean())
    x = (df["lon"].values - CITY_HALL[0]) * 111_320 * np.cos(lat0)
    y = (df["lat"].values - CITY_HALL[1]) * 110_570
    df = df.copy()
    df["x_m"], df["y_m"] = x, y
    df["dist_center_km"] = np.hypot(x, y) / 1000

    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    df["nbr_cafe_lag"] = df["n_cafe"].values[nn].mean(axis=1)
    df["nbr_daypop_lag"] = df["day_pop"].values[nn].mean(axis=1)
    print(f"  공간 시차: 최근접 {k}개 동 평균 (경계 파일 없이 중심점 거리로 근사)")
    return df


def main() -> None:
    print("=" * 64)
    print("4장 준비: 국내 공개 데이터 → 행정동 단위 입지 분석용 파일")
    print("=" * 64)

    print("\n[1/3] 상가(상권)정보 → 행정동 집계")
    stores, stores_src = prepare_stores()

    print("\n[2/3] 생활인구(행정동) → 수요 피처")
    demand, demand_src = prepare_demand()

    print("\n[3/3] 결합과 공간 피처")
    matched = stores["dong_code"].isin(set(demand["dong_code"]))
    print(f"  결합 점검: 행정동코드 매칭 {matched.mean():.1%} ({matched.sum()}/{len(stores)})")
    if matched.mean() < 0.9:
        miss = sorted(set(stores.loc[~matched, "dong"]))[:8]
        print(f"  ⚠ 매칭 실패 행정동 예: {miss}")

    df = stores.merge(demand, on="dong_code", how="inner")
    df = df.dropna(subset=["day_pop", "night_pop", "day_night_ratio",
                           "weekend_ratio", "peak_ratio"]).reset_index(drop=True)
    df = add_spatial_features(df)

    print(f"  최종 {len(df)}개 행정동 | 자치구 {df['sigungu'].nunique()}개")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(DATA_DIR / "seoul_dong_market.parquet", index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    (DATA_DIR / "SOURCES_market.txt").write_text(
        "4장 입지 분석용 파일의 출처\n"
        f"- 상가(상권)정보: {stores_src}\n"
        f"- 생활인구(행정동): {demand_src}\n"
        f"- 처리 일자: {stamp}\n"
        f"- 대상: 서울 전역, 결과변수 = 행정동별 소분류 '{CAFE_SUBCATEGORY}' 점포 수\n"
        f"- 수요 정의: 낮 {DAY_HOURS[0]}~{DAY_HOURS[1]}시 / 심야 {NIGHT_HOURS[0]}~{NIGHT_HOURS[1]}시 평균 생활인구\n"
        "- 행정동 중심 좌표: 전 업종 점포 좌표의 평균(경계 파일 미사용 근사)\n"
        "공개 데이터가 갱신되면 결과가 달라진다. 본문 수치는 이 기준으로 인용한다.\n",
        encoding="utf-8")

    print("-" * 64)
    print(f"저장: seoul_dong_market.parquet({len(df)}행 × {df.shape[1]}열), SOURCES_market.txt")
    print("=" * 64)
    print("[완료] 다음: python 4-6-store-location-supply.py")


if __name__ == "__main__":
    main()
