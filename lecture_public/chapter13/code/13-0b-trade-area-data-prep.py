"""
13장 준비(비즈니스): 상가(상권)정보 → 읍면동 상권 지표와 편의점 점포 목록
=========================================================================
13-0-simdata-prep.py가 공공 분석 1~3의 교육용 시뮬레이션을 만드는 것과 달리,
이 스크립트는 **국내 공개 실데이터**를 읽어 비즈니스 분석(13-4)의 입력을 만든다.

만드는 것 두 개:
  1) trade_area_units.parquet — 인구감소지역 + 비지정 군 지역의 읍면동별 상권 지표
     (점포 수, 업종 대분류 구성비, 업종 다양성, 체인 비율, 생활서비스 보유 여부)
  2) trade_area_stores.parquet — 같은 지역의 편의점 점포 목록 + 소속 읍면동의 상권 규모

원자료: 소상공인시장진흥공단 상가(상권)정보(공공데이터포털 15083033, 분기 갱신).
  포털이 스크립트로 다운로드를 처리해 자동 내려받기가 막히므로 한 번은 직접 받아야
  한다. 내려받는 절차는 practice/chapter14/data/raw/README.md와 같다.
  14장이 이미 받아 둔 폴더가 있으면 그대로 읽는다(1.6GB를 두 벌 두지 않는다).

주의: 이 데이터에는 **매출도 인구도 없다.** 13-4는 그 사실을 전제로 설계되어 있다.

실행:
    python 13-0b-trade-area-data-prep.py
"""

from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PRACTICE_DIR = SCRIPT_DIR.parent.parent

# 원자료를 찾을 폴더 후보. 13장 아래에 두어도 되고, 14장이 받아 둔 것을 그대로 써도 된다.
RAW_CANDIDATES = (
    DATA_DIR / "raw",
    PRACTICE_DIR / "chapter14" / "data" / "raw",
)
RAW_PATTERNS = ("상가", "소상공인", "상권정보")
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

# 행정안전부 인구감소지역 89곳 (「지방자치분권 및 지역균형발전에 관한 특별법」 제2조·
# 시행령 제3조, 2021년 10월 최초 지정, 5년 주기).
# 출처: 행정안전부 인구감소지역 지정 현황 페이지(2026-08-13 열람).
# 같은 이름의 시군구가 여러 시도에 있으므로(고성군, 동구, 서구) 반드시 시도와 짝으로 쓴다.
DECLINE_AREAS: dict[str, tuple[str, ...]] = {
    "부산": ("동구", "서구", "영도구"),
    "대구": ("남구", "서구", "군위군"),
    "인천": ("강화군", "옹진군"),
    "경기": ("가평군", "연천군"),
    "강원": ("고성군", "삼척시", "양구군", "양양군", "영월군", "정선군",
             "철원군", "태백시", "평창군", "홍천군", "화천군", "횡성군"),
    "충북": ("괴산군", "단양군", "보은군", "영동군", "옥천군", "제천시"),
    "충남": ("공주시", "금산군", "논산시", "보령시", "부여군", "서천군",
             "예산군", "청양군", "태안군"),
    "전북": ("고창군", "김제시", "남원시", "무주군", "부안군", "순창군",
             "임실군", "장수군", "정읍시", "진안군"),
    "전남": ("강진군", "고흥군", "곡성군", "구례군", "담양군", "보성군",
             "신안군", "영광군", "영암군", "완도군", "장성군", "장흥군",
             "진도군", "함평군", "해남군", "화순군"),
    "경북": ("고령군", "문경시", "봉화군", "상주시", "성주군", "안동시",
             "영덕군", "영양군", "영주시", "영천시", "울릉군", "울진군",
             "의성군", "청도군", "청송군"),
    "경남": ("거창군", "고성군", "남해군", "밀양시", "산청군", "의령군",
             "창녕군", "하동군", "함안군", "함양군", "합천군"),
}

# 시도명 표기가 자료마다 다르다(강원도/강원특별자치도, 전라북도/전북특별자치도).
# 짧은 별칭으로 정규화한다.
SIDO_ALIASES: tuple[tuple[str, str], ...] = (
    ("서울", "서울"), ("부산", "부산"), ("대구", "대구"), ("인천", "인천"),
    ("광주", "광주"), ("대전", "대전"), ("울산", "울산"), ("세종", "세종"),
    ("경기", "경기"), ("강원", "강원"), ("충청북", "충북"), ("충북", "충북"),
    ("충청남", "충남"), ("충남", "충남"), ("전라북", "전북"), ("전북", "전북"),
    ("전라남", "전남"), ("전남", "전남"), ("경상북", "경북"), ("경북", "경북"),
    ("경상남", "경남"), ("경남", "경남"), ("제주", "제주"),
)

# 생활서비스 판정에 쓰는 업종 소분류 키워드. 실제로 무엇이 걸렸는지 로그에 남긴다.
SERVICE_KEYWORDS = {
    "convenience": ("편의점",),
    "pharmacy": ("약국",),
    "clinic": ("의원",),      # 일반의원·치과의원·한의원 등
    "gas": ("주유소",),
}
TARGET_SERVICE = "convenience"   # 철수 판정 대상 업종
LIFE_MAJOR = "수리·개인"          # 배후 인구 대리로 쓸 업종 대분류(미용·세탁·수리 등)


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def short_sido(name: str) -> str:
    """'강원특별자치도' → '강원'. 못 맞추면 앞 두 글자."""
    s = norm(name)
    for key, short in SIDO_ALIASES:
        if s.startswith(key):
            return short
    return s[:2]


def find_raw_files() -> tuple[list[Path], Path]:
    """상가정보 CSV를 담은 폴더를 찾는다."""
    for raw_dir in RAW_CANDIDATES:
        if not raw_dir.exists():
            continue
        files = sorted(p for p in raw_dir.rglob("*.csv")
                       if any(k in norm(p.name) for k in RAW_PATTERNS))
        if files:
            return files, raw_dir
    tried = "\n".join(f"    - {p}" for p in RAW_CANDIDATES)
    sys.exit(
        "[중단] 상가(상권)정보 CSV를 찾지 못했다.\n"
        f"  찾은 폴더:\n{tried}\n"
        f"  찾는 이름: {' 또는 '.join(RAW_PATTERNS)} 가 들어간 .csv\n"
        "  → 공공데이터포털 15083033에서 내려받아 위 폴더 중 하나에 둔다.\n"
        "    절차는 practice/chapter14/data/raw/README.md 참조."
    )


def sniff_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, encoding=enc, nrows=1, low_memory=False)
            return enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했다: {path.name}")


def resolve_columns(cols) -> dict[str, str]:
    """필요한 열을 후보 이름으로 찾는다. 못 찾으면 실제 열 목록을 보여 주고 멈춘다.

    공공 데이터는 갱신하면서 열 이름이 바뀐다. 조용히 잘못된 열을 쓰는 것보다
    멈추는 편이 낫다.
    """
    needed = {
        "sido": ("시도명",),
        "sigungu": ("시군구명",),
        "dong": ("행정동명",),
        "dong_code": ("행정동코드",),
        "major": ("상권업종대분류명",),
        "middle": ("상권업종중분류명",),
        "sub": ("상권업종소분류명",),
        "branch": ("지점명",),
        "lon": ("경도",),
        "lat": ("위도",),
    }
    resolved, missing = {}, []
    for key, cands in needed.items():
        hit = next((c for c in cands if c in cols), None)
        if hit is None:
            missing.append(f"{key}: {cands}")
        else:
            resolved[key] = hit
    if missing:
        sys.exit("[중단] 상가정보에서 필요한 열을 찾지 못했다.\n  못 찾은 것:\n    "
                 + "\n    ".join(missing)
                 + f"\n  실제 열({len(cols)}개):\n    {list(cols)}")
    return resolved


def load_stores(files: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """분할 CSV를 모두 읽어 필요한 열만 남긴다."""
    frames, used = [], []
    for i, path in enumerate(files, 1):
        enc = sniff_encoding(path)
        head = pd.read_csv(path, encoding=enc, nrows=1, low_memory=False)
        col = resolve_columns(head.columns)
        df = pd.read_csv(path, encoding=enc, low_memory=False,
                         usecols=list(col.values()))
        df = df.rename(columns={v: k for k, v in col.items()})
        frames.append(df)
        used.append(path.name)
        print(f"  [{i:2d}/{len(files)}] {path.name[:52]:52s} {len(df):>8,}행 ({enc})")
    out = pd.concat(frames, ignore_index=True)
    print(f"  전국 합계 {len(out):,}행")
    return out, used


def tag_regions(df: pd.DataFrame) -> pd.DataFrame:
    """시도를 정규화하고 인구감소지역 지정 여부를 붙인다."""
    df["sido_short"] = df["sido"].map(short_sido)
    df["sigungu"] = df["sigungu"].map(norm)

    designated = {(s, g) for s, gs in DECLINE_AREAS.items() for g in gs}
    keys = list(zip(df["sido_short"], df["sigungu"]))
    df["is_decline_area"] = [k in designated for k in keys]
    df["is_gun"] = df["sigungu"].str.endswith("군")
    return df


def build_units(df: pd.DataFrame) -> pd.DataFrame:
    """읍면동 단위 상권 지표를 만든다.

    쓸 수 있는 것이 점포 목록뿐이므로 지표도 점포 구성에서만 나온다.
    인구·매출·면적이 없으므로 '밀도'는 만들지 못한다(점포 수로 규모를 본다).
    """
    unit_keys = ["sido_short", "sigungu", "dong", "dong_code",
                 "is_decline_area", "is_gun"]

    # 업종 대분류 구성비
    major = (df.pivot_table(index=unit_keys, columns="major", aggfunc="size",
                            fill_value=0, observed=True))
    n_stores = major.sum(axis=1)
    share = major.div(n_stores, axis=0)
    share.columns = [f"share_{c}" for c in share.columns]

    # 업종 다양성: 중분류 분포의 정규화 섀넌 엔트로피(0=한 업종, 1=고르게 분산)
    mid = df.pivot_table(index=unit_keys, columns="middle", aggfunc="size",
                         fill_value=0, observed=True)
    p = mid.div(mid.sum(axis=1), axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    k_eff = (mid > 0).sum(axis=1).to_numpy()
    diversity = np.where(k_eff > 1, ent / np.log(np.maximum(k_eff, 2)), 0.0)

    units = pd.DataFrame({"n_stores": n_stores, "diversity": diversity},
                         index=major.index)
    units = units.join(share)

    # 생활밀착 업종 점포 수: 배후 인구의 또 다른 대리(미용·세탁·수리 등).
    # 소매 업종에는 편의점 자신이 들어 있어 배후 대리로 쓰면 순환이 된다.
    if LIFE_MAJOR in major.columns:
        units["n_personal"] = major[LIFE_MAJOR].astype(int)
    else:
        units["n_personal"] = 0

    # 체인 비율: 지점명이 있는 점포의 비율(프랜차이즈·다점포 운영의 대리)
    has_branch = df["branch"].notna() & (df["branch"].astype(str).str.strip() != "")
    units["chain_share"] = (df.assign(_b=has_branch)
                              .groupby(unit_keys, observed=True)["_b"].mean())

    # 읍면동 대표 좌표: 소속 점포 좌표의 중앙값. **그림 표시용 근사**이며 행정동
    # 경계의 중심이 아니다(경계 파일을 쓰지 않았다). 분석 계산에는 쓰지 않는다.
    for axis in ("lon", "lat"):
        units[axis] = df.groupby(unit_keys, observed=True)[axis].median()

    # 생활서비스 보유 여부
    for name, kws in SERVICE_KEYWORDS.items():
        hit = df["sub"].fillna("").map(lambda s: any(k in s for k in kws))
        cnt = (df.assign(_h=hit).groupby(unit_keys, observed=True)["_h"].sum())
        units[f"n_{name}"] = cnt.astype(int)
        units[f"has_{name}"] = units[f"n_{name}"] > 0

    return units.reset_index()


def build_target_stores(df: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """철수 판정 대상 업종(편의점) 점포 목록 + 소속 읍면동의 상권 규모."""
    kws = SERVICE_KEYWORDS[TARGET_SERVICE]
    sel = df[df["sub"].fillna("").map(lambda s: any(k in s for k in kws))].copy()

    unit_keys = ["sido_short", "sigungu", "dong"]
    size = units.set_index(unit_keys)[["n_stores", "n_personal", f"n_{TARGET_SERVICE}"]]
    sel = sel.join(size, on=unit_keys, rsuffix="_unit")

    out = sel[unit_keys + ["dong_code", "lon", "lat", "sub",
                           "is_decline_area", "is_gun",
                           "n_stores", "n_personal", f"n_{TARGET_SERVICE}"]].copy()
    out = out.rename(columns={f"n_{TARGET_SERVICE}": "n_same_kind"})
    out = out.dropna(subset=["n_stores", "n_same_kind"]).reset_index(drop=True)
    out.insert(0, "store_id", range(len(out)))
    return out


def main() -> None:
    print("=" * 72)
    print("13장 준비(비즈니스): 상가(상권)정보 → 읍면동 상권 지표·편의점 점포 목록")
    print("=" * 72)

    files, raw_dir = find_raw_files()
    print(f"\n[1/4] 원자료 {len(files)}개 읽기  (폴더: {raw_dir})")
    df, used = load_stores(files)

    print("\n[2/4] 지역 구분 — 인구감소지역 고시 여부")
    df = tag_regions(df)
    n_designated = sum(len(v) for v in DECLINE_AREAS.values())
    found = (df.loc[df["is_decline_area"], ["sido_short", "sigungu"]]
               .drop_duplicates())
    print(f"  고시 목록 {n_designated}곳 중 자료에서 확인된 시군구 {len(found)}곳")
    if len(found) < n_designated:
        listed = {(s, g) for s, gs in DECLINE_AREAS.items() for g in gs}
        miss = sorted(listed - set(map(tuple, found.to_numpy())))
        print(f"  ⚠ 자료에 없는 시군구: {miss}")

    # 분석 우주: 인구감소지역 89곳 + 비지정 군 지역(대조군)
    universe = df[df["is_decline_area"] | df["is_gun"]].copy()
    universe = universe.dropna(subset=["dong", "dong_code", "major", "middle"])
    universe = universe[universe["dong"].astype(str).str.strip() != ""]
    n_ctrl = (universe.loc[~universe["is_decline_area"], ["sido_short", "sigungu"]]
                      .drop_duplicates())
    print(f"  분석 우주: 점포 {len(universe):,}개 "
          f"(인구감소지역 {int(universe['is_decline_area'].sum()):,} / "
          f"비지정 군 {int((~universe['is_decline_area']).sum()):,})")
    # 대조군의 정체를 로그에 남긴다. 인구감소지역 지정이 군 지역을 거의 다 덮었으므로
    # 남은 비지정 군이 어떤 성격인지가 대조의 해석을 좌우한다.
    ctrl_list = sorted(f"{s} {g}" for s, g in n_ctrl.to_numpy())
    print(f"  대조군 시군구(비지정 군) {len(n_ctrl)}곳: {ctrl_list}")

    print("\n[3/4] 읍면동 상권 지표")
    units = build_units(universe)
    print(f"  읍면동 {len(units):,}개  "
          f"(인구감소지역 {int(units['is_decline_area'].sum()):,} / "
          f"비지정 군 {int((~units['is_decline_area']).sum()):,})")
    print(f"  점포 수 분포: 중앙값 {units['n_stores'].median():.0f}, "
          f"[{units['n_stores'].min():.0f}, {units['n_stores'].max():.0f}]")
    share_cols = [c for c in units.columns if c.startswith("share_")]
    print(f"  업종 대분류 {len(share_cols)}개: "
          f"{[c.replace('share_', '') for c in share_cols]}")
    for name, kws in SERVICE_KEYWORDS.items():
        matched = sorted(universe.loc[
            universe["sub"].fillna("").map(lambda s: any(k in s for k in kws)),
            "sub"].dropna().unique())
        print(f"  [{name}] 키워드 {kws} → 소분류 {len(matched)}종, "
              f"보유 읍면동 {int(units[f'has_{name}'].sum()):,}개 "
              f"({units[f'has_{name}'].mean():.1%})")
        print(f"      매칭 소분류 예: {matched[:6]}")

    print("\n[4/4] 철수 판정 대상 점포")
    stores = build_target_stores(universe, units)
    print(f"  {TARGET_SERVICE} 점포 {len(stores):,}개 "
          f"(인구감소지역 {int(stores['is_decline_area'].sum()):,} / "
          f"비지정 군 {int((~stores['is_decline_area']).sum()):,})")

    DATA_DIR.mkdir(exist_ok=True)
    units.to_parquet(DATA_DIR / "trade_area_units.parquet", index=False)
    stores.to_parquet(DATA_DIR / "trade_area_stores.parquet", index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    (DATA_DIR / "SOURCES_trade_area.txt").write_text(
        "13장 비즈니스 분석(13-4) 입력 파일의 출처\n"
        "- 소상공인시장진흥공단 상가(상권)정보 (공공데이터포털 15083033)\n"
        + "".join(f"  · {n}\n" for n in used)
        + f"- 원자료 폴더: {raw_dir}\n"
        f"- 처리 일자: {stamp}\n"
        "- 인구감소지역 목록: 행정안전부 인구감소지역 지정 현황(89곳, 2021.10 최초 지정)\n"
        "  https://www.mois.go.kr/frt/sub/a06/b06/populationDecline/screen.do\n"
        "- 이 데이터에는 매출·인구·면적이 없다. 13-4는 그 제약 위에서 설계되었다.\n"
        "공개 데이터가 갱신되면 결과가 달라진다. 본문 수치는 이 기준으로 인용한다.\n",
        encoding="utf-8")

    print("-" * 72)
    print(f"저장: trade_area_units.parquet({len(units):,}행), "
          f"trade_area_stores.parquet({len(stores):,}행), SOURCES_trade_area.txt")
    print("=" * 72)
    print("[완료] 다음: python 13-4-trade-area-exit-decision.py")


if __name__ == "__main__":
    main()
