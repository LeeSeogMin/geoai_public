"""
3장 준비(비즈니스): 상업 공간 데이터의 편향 진단용 집계 자료 만들기
====================================================================
세 출처를 서울 행정동 단위로 맞대어 놓은 집계표를 만든다. 분석 코드(3-4)는 이
집계표만 읽으므로 인터넷 없이 재현된다.

만드는 것 네 개:
  1) bias_dong.parquet    — 행정동 425개의 면적·중심좌표·시청 거리
  2) bias_counts.parquet  — 행정동 × 업종군 × 출처별 점포 수(좁은 대응·넓은 대응)
  3) bias_sales.parquet   — 행정동 × 서비스업종 × 분기별 카드 기반 추정매출
  4) data/README.md       — 출처·취득일·라이선스·재취득 방법

원자료 세 갈래:
  (A) 소상공인시장진흥공단 상가(상권)정보 — 점포별 좌표·업종. 포털이 스크립트로
      다운로드를 처리해 자동 내려받기가 막히므로 한 번은 직접 받아야 한다.
      절차는 ../data/raw/README.md. 14장이 이미 받아 둔 폴더가 있으면 그대로
      읽는다(1.6GB를 두 벌 두지 않는다).
  (B) 서울 열린데이터광장 상권분석서비스 — 추정매출(OA-22175)과 행정동 경계
      (OA-22160). 이 둘은 자동 내려받기가 된다.
  (C) OpenStreetMap POI — Overpass API로 서울 영역을 조회한다.

라이선스: (A)(B)는 공공누리, (C)는 ODbL. 원자료는 저장소에 넣지 않고 집계값만
남긴다. 근거는 만들어지는 data/README.md에 적는다.

실행:
    python 3-0b-commercial-bias-data-prep.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import unicodedata
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PRACTICE_DIR = SCRIPT_DIR.parent.parent

# 상가(상권)정보 원자료를 찾을 폴더 후보. 3장 아래에 두어도 되고, 14장이 받아 둔
# 것을 그대로 써도 된다.
SBIZ_RAW_CANDIDATES = (RAW_DIR, PRACTICE_DIR / "chapter14" / "data" / "raw")
SBIZ_PATTERNS = ("상가", "소상공인", "상권정보")
ENCODINGS = ("utf-8-sig", "utf-8", "cp949")

UA = {"User-Agent": "geoai-textbook/1.0 (educational use)"}

# 서울 열린데이터광장 파일 내려받기: 데이터셋 페이지에서 세션을 얻은 뒤
# datafile 서버에 POST 한다.
SEOUL_VIEW = "https://data.seoul.go.kr/dataList/{oid}/S/1/datasetView.do"
SEOUL_FILE = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
SALES_OID, SALES_SEQ, SALES_YEAR = "OA-22175", "7", 2025   # seq 7 = 2025년 파일
AREA_OID, AREA_SEQ = "OA-22160", "1"

# Overpass 공개 엔드포인트. 앞의 것이 막히면 다음으로 넘어간다.
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
SEOUL_BBOX = (37.41, 126.75, 37.72, 127.20)   # (south, west, north, east)

# 서울시청 좌표(WGS84). 도심에서 멀어질수록 편향이 커지는지 보려고 쓴다.
CITYHALL_LONLAT = (126.9784, 37.5666)

# ---------------------------------------------------------------------------
# 업종 대응표
# ---------------------------------------------------------------------------
# 세 출처의 업종 분류가 서로 다르므로 대응을 명시적으로 적어 둔다. 이 대응을 어떻게
# 잡느냐가 포착률을 바꾸므로, 코드 안에 숨기지 않고 표로 본문에 싣는다.
#
# narrow = 이름이 정확히 겹치는 것만. broad = 성격이 가까운 것을 더 넣은 판본.
# 두 판본을 나란히 만드는 이유는, 결론이 대응 방식에 좌우되지 않는 부분만 본문
# 주장으로 삼기 위해서다. 대응은 상권정보 쪽과 OSM 쪽 모두에서 넓혔다 좁혔다 한다.
CATEGORY_MAP: dict[str, dict] = {
    "카페": {
        "svc_cd": "CS100010",           # 서울 상권분석 '커피-음료'
        "narrow": ("카페",),
        "broad_extra": ("아이스크림/빙수", "토스트/샌드위치/샐러드"),
        "osm_narrow": ("amenity=cafe",),
        "osm_extra": ("shop=coffee", "shop=tea"),
    },
    "제과점": {
        "svc_cd": "CS100005",
        "narrow": ("빵/도넛",),
        "broad_extra": ("떡/한과",),
        "osm_narrow": ("shop=bakery",),
        "osm_extra": ("shop=pastry", "shop=confectionery"),
    },
    "편의점": {
        "svc_cd": "CS300002",
        "narrow": ("편의점",),
        "broad_extra": (),
        "osm_narrow": ("shop=convenience",),
        "osm_extra": (),
    },
    "미용·뷰티": {
        # OSM의 shop=hairdresser에는 네일숍·피부관리실이 섞여 들어온다(태그 하나에
        # 여러 업종이 담긴다). 좁은 대응으로 재면 포착률이 1을 넘어 물리적으로
        # 불가능한 값이 나오므로, 넓은 대응이 필요하다는 사실 자체가 발견 사항이다.
        "svc_cd": "CS200028",
        "narrow": ("미용실",),
        "broad_extra": ("네일숍", "피부 관리실", "마사지/안마", "체형/비만 관리"),
        "osm_narrow": ("shop=hairdresser",),
        "osm_extra": ("shop=beauty", "shop=nails", "shop=massage"),
    },
    "약국": {
        "svc_cd": "CS300018",           # 서울 상권분석 '의약품'
        "narrow": ("약국",),
        "broad_extra": (),
        "osm_narrow": ("amenity=pharmacy",),
        "osm_extra": ("shop=chemist",),
    },
    "안경": {
        "svc_cd": "CS300016",
        "narrow": ("안경렌즈 소매업",),
        "broad_extra": (),
        "osm_narrow": ("shop=optician",),
        "osm_extra": (),
    },
    "화장품": {
        "svc_cd": "CS300022",
        "narrow": ("화장품 소매업",),
        "broad_extra": (),
        "osm_narrow": ("shop=cosmetics",),
        "osm_extra": ("shop=perfumery",),
    },
    "서점": {
        "svc_cd": "CS300020",
        "narrow": ("서점",),
        "broad_extra": (),
        # shop=stationery(문구점)는 넣지 않는다. 상권정보에서 문구는 별도 업종이므로
        # OSM 쪽만 넓히면 포착률이 실제보다 부풀려진다 — 대응은 양쪽을 같이 넓혀야 한다.
        "osm_narrow": ("shop=books",),
        "osm_extra": (),
    },
    "세탁소": {
        "svc_cd": "CS200031",
        "narrow": ("세탁소",),
        "broad_extra": ("셀프 빨래방",),
        "osm_narrow": ("shop=laundry",),
        "osm_extra": ("shop=dry_cleaning",),
    },
}

# 카드 매출 쪽에서 함께 보는 업종. OSM 태그가 마땅치 않아 포착률 분석에는 넣지
# 않지만, 결측 구조 분석에는 점포 수가 필요하다. 현금 결제가 많다고 알려진
# 재래 업종(청과·정육·수산·반찬·미곡·철물)과 카드 결제가 사실상 전부인 업종
# (편의점·카페·화장품)이 함께 들어가도록 골랐다.
SALES_ONLY_MAP: dict[str, dict] = {
    "한식음식점": {"svc_cd": "CS100001",
                "narrow": ("백반/한정식", "국/탕/찌개류", "돼지고기 구이/찜",
                           "소고기 구이/찜", "닭/오리고기 구이/찜", "곱창 전골/구이",
                           "족발/보쌈", "해산물 구이/찜", "국수/칼국수", "냉면/밀면",
                           "전/부침개", "기타 한식 음식점"),
                "broad_extra": ()},
    "치킨전문점": {"svc_cd": "CS100007", "narrow": ("치킨",), "broad_extra": ()},
    "분식전문점": {"svc_cd": "CS100008", "narrow": ("김밥/만두/분식",), "broad_extra": ()},
    "호프-간이주점": {"svc_cd": "CS100009",
                  "narrow": ("생맥주 전문", "요리 주점"), "broad_extra": ("일반 유흥 주점",)},
    "노래방": {"svc_cd": "CS200037", "narrow": ("노래방",), "broad_extra": ()},
    "슈퍼마켓": {"svc_cd": "CS300001", "narrow": ("슈퍼마켓",), "broad_extra": ()},
    "청과상": {"svc_cd": "CS300009", "narrow": ("채소/과일 소매업",), "broad_extra": ()},
    "육류판매": {"svc_cd": "CS300007", "narrow": ("정육점",), "broad_extra": ()},
    "수산물판매": {"svc_cd": "CS300008", "narrow": ("수산물 소매업",), "broad_extra": ()},
    "반찬가게": {"svc_cd": "CS300010", "narrow": ("반찬/식료품 소매업",), "broad_extra": ()},
    "미곡판매": {"svc_cd": "CS300006", "narrow": ("곡물/곡분 소매업",), "broad_extra": ()},
    "철물점": {"svc_cd": "CS300033", "narrow": ("철물/공구 소매업",), "broad_extra": ()},
    "신발": {"svc_cd": "CS300014", "narrow": ("신발 소매업",), "broad_extra": ()},
    "문구": {"svc_cd": "CS300021", "narrow": ("문구/회화용품 소매업",), "broad_extra": ()},
    "시계및귀금속": {"svc_cd": "CS300017", "narrow": ("시계/귀금속 소매업",), "broad_extra": ()},
    "가구": {"svc_cd": "CS300031", "narrow": ("가구 소매업",), "broad_extra": ()},
    "자전거": {"svc_cd": "CS300025", "narrow": ("자전거 소매업",), "broad_extra": ()},
    "애완동물": {"svc_cd": "CS300029", "narrow": ("애완동물/애완용품 소매업",), "broad_extra": ()},
}

ALL_MAP = {**CATEGORY_MAP, **SALES_ONLY_MAP}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# (A) 상가(상권)정보
# ---------------------------------------------------------------------------
def find_sbiz_csv() -> Path:
    """서울 상가정보 CSV를 후보 폴더에서 찾는다."""
    cands: list[Path] = []
    for root in SBIZ_RAW_CANDIDATES:
        if not root.exists():
            continue
        cands += [
            p for p in root.rglob("*.csv")
            if any(k in unicodedata.normalize("NFC", p.name) for k in SBIZ_PATTERNS)
            and "서울" in unicodedata.normalize("NFC", p.name)
        ]
    if not cands:
        sys.exit(
            "[중단] 서울 상가(상권)정보 CSV를 찾지 못했다.\n"
            f"  찾은 곳: {[str(p) for p in SBIZ_RAW_CANDIDATES]}\n"
            "  → practice/chapter3/data/raw/README.md의 절차를 먼저 밟는다."
        )
    return max(cands, key=lambda p: p.stat().st_size)


def read_sbiz(path: Path) -> pd.DataFrame:
    cols = ["상권업종대분류명", "상권업종중분류명", "상권업종소분류명",
            "행정동코드", "행정동명", "시군구명", "경도", "위도"]
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, usecols=cols, low_memory=False)
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했다: {path.name}")


# ---------------------------------------------------------------------------
# (B) 서울 열린데이터광장
# ---------------------------------------------------------------------------
def seoul_download(oid: str, seq: str) -> bytes:
    """데이터셋 페이지에서 세션을 얻은 뒤 파일 서버에 POST 한다."""
    s = requests.Session()
    view = SEOUL_VIEW.format(oid=oid)
    s.get(view, headers=UA, timeout=60)
    r = s.post(SEOUL_FILE, data={"infId": oid, "seqNo": seq, "seq": seq, "infSeq": "3"},
               headers={**UA, "Referer": view}, timeout=600)
    r.raise_for_status()
    if len(r.content) < 100_000:
        sys.exit(f"[중단] {oid} 응답이 너무 작다({len(r.content)}바이트). 포털 구조가 바뀌었을 수 있다.")
    return r.content


def cached_download(oid: str, seq: str, fname: str) -> bytes:
    """이미 받아 둔 파일이 있으면 다시 받지 않는다."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / fname
    if p.exists() and p.stat().st_size > 100_000:
        log(f"  (이미 받아 둔 파일 사용) {p.name} — {p.stat().st_size / 1e6:.1f} MB")
        return p.read_bytes()
    log(f"  내려받는 중: {oid} → {p.name}")
    blob = seoul_download(oid, seq)
    p.write_bytes(blob)
    log(f"  저장 — {len(blob) / 1e6:.1f} MB")
    return blob


def read_zip_csv(blob: bytes) -> tuple[pd.DataFrame, str]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    try:
        shown = name.encode("cp437").decode("cp949")
    except Exception:
        shown = name
    raw = z.read(name)
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False), shown
        except UnicodeDecodeError:
            continue
    sys.exit("[중단] 매출 CSV 인코딩을 판별하지 못했다.")


def read_zip_shapefile(blob: bytes, workdir: Path) -> gpd.GeoDataFrame:
    """압축 안의 shp 일습을 ASCII 이름으로 풀어 읽는다(한글 파일명 인코딩 회피)."""
    workdir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(blob))
    for n in z.namelist():
        ext = n.rsplit(".", 1)[-1].lower()
        (workdir / f"area_dong.{ext}").write_bytes(z.read(n))
    return gpd.read_file(workdir / "area_dong.shp")


# ---------------------------------------------------------------------------
# (C) OpenStreetMap POI
# ---------------------------------------------------------------------------
def overpass_query(body: str, retries: int = 3) -> dict:
    """엔드포인트를 돌아가며 재시도한다. 공개 서버는 혼잡할 때 504를 돌려준다."""
    last = ""
    for attempt in range(retries):
        for ep in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": body}, headers=UA, timeout=600)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    return r.json()
                last = f"{ep} → HTTP {r.status_code}"
            except Exception as e:                       # noqa: BLE001
                last = f"{ep} → {type(e).__name__}"
        wait = 20 * (attempt + 1)
        log(f"    Overpass 재시도 {attempt + 1}/{retries} ({last}) — {wait}초 대기")
        time.sleep(wait)
    sys.exit(f"[중단] Overpass 조회에 실패했다: {last}")


def fetch_osm_pois() -> pd.DataFrame:
    """업종군별 OSM POI 좌표를 가져온다. 점(node)뿐 아니라 면(way·relation)으로
    그려진 상점도 있으므로 `out center`로 대표점을 함께 받는다.
    태그마다 따로 받아 두어야 좁은 대응·넓은 대응을 나중에 갈라 셀 수 있다."""
    s, w, n, e = SEOUL_BBOX
    bbox = f"({s},{w},{n},{e})"
    rows: list[dict] = []
    for cat, spec in CATEGORY_MAP.items():
        for scope in ("osm_narrow", "osm_extra"):
            for tag in spec[scope]:
                parts = "".join(f"{kind}[{tag}]{bbox};" for kind in ("node", "way", "relation"))
                js = overpass_query(f"[out:json][timeout:300];({parts});out center;")
                got = 0
                for el in js.get("elements", []):
                    lon = el.get("lon") or (el.get("center") or {}).get("lon")
                    lat = el.get("lat") or (el.get("center") or {}).get("lat")
                    if lon is None or lat is None:
                        continue
                    rows.append({"cat": cat, "tag": tag,
                                 "scope": "narrow" if scope == "osm_narrow" else "extra",
                                 "lon": lon, "lat": lat})
                    got += 1
                log(f"    {cat} [{tag}]: {got}개")
                time.sleep(3)          # 공개 서버 예의
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log("3장 준비: 상업 공간 데이터 편향 진단용 집계 자료")
    log("=" * 72)

    # --- 1) 행정동 경계 -----------------------------------------------------
    log("\n[1] 서울 행정동 경계 (서울 열린데이터광장 OA-22160)")
    area_blob = cached_download(AREA_OID, AREA_SEQ, "seoul_sangkwon_area_dong.zip")
    gdf = read_zip_shapefile(area_blob, RAW_DIR / "_shp_tmp")
    log(f"  행정동 {len(gdf)}개, 좌표계 {gdf.crs}")

    cent = gdf.geometry.centroid
    cent_wgs = gpd.GeoSeries(cent, crs=gdf.crs).to_crs(4326)
    hall = gpd.GeoSeries([Point(*CITYHALL_LONLAT)], crs=4326).to_crs(gdf.crs).iloc[0]

    dong = pd.DataFrame({
        "dong_cd": gdf["ADSTRD_CD"].astype(int),
        "dong_nm": gdf["ADSTRD_NM"].astype(str),
        "area_km2": gdf["RELM_AR"].astype(float) / 1e6,
        "lon": cent_wgs.x.to_numpy(),
        "lat": cent_wgs.y.to_numpy(),
        "dist_cityhall_km": cent.distance(hall).to_numpy() / 1000.0,
    })
    log(f"  면적 중앙값 {dong['area_km2'].median():.2f} km², "
        f"시청 거리 {dong['dist_cityhall_km'].min():.1f}~{dong['dist_cityhall_km'].max():.1f} km")

    # --- 2) 상가(상권)정보 --------------------------------------------------
    log("\n[2] 소상공인시장진흥공단 상가(상권)정보")
    sbiz_path = find_sbiz_csv()
    log(f"  원본: {sbiz_path.name} ({sbiz_path.stat().st_size / 1e6:.0f} MB)")
    sbiz = read_sbiz(sbiz_path)
    log(f"  서울 점포 {len(sbiz):,}개, 행정동 {sbiz['행정동코드'].nunique()}개")

    counts: list[dict] = []
    for cat, spec in ALL_MAP.items():
        narrow = set(spec["narrow"])
        broad = narrow | set(spec["broad_extra"])
        miss = [n for n in broad if n not in set(sbiz["상권업종소분류명"].unique())]
        if miss:
            log(f"  [경고] {cat}: 상권정보에 없는 소분류명 {miss}")
        for scope, names in (("narrow", narrow), ("broad", broad)):
            g = (sbiz[sbiz["상권업종소분류명"].isin(names)]
                 .groupby("행정동코드").size().rename("n").reset_index())
            for _, r in g.iterrows():
                counts.append({"dong_cd": int(r["행정동코드"]), "cat": cat,
                               "source": f"sbiz_{scope}", "n": int(r["n"])})
    sbiz_tot = (sbiz[sbiz["상권업종소분류명"].isin(
        {n for s in ALL_MAP.values() for n in s["narrow"]})].shape[0])
    log(f"  대응표에 잡힌 점포(좁은 대응 합계): {sbiz_tot:,}개")

    # --- 3) OSM POI ---------------------------------------------------------
    log("\n[3] OpenStreetMap POI (Overpass API)")
    osm_cache = RAW_DIR / "osm_seoul_poi_v2.parquet"
    if osm_cache.exists():
        osm = pd.read_parquet(osm_cache)
        log(f"  (이미 받아 둔 파일 사용) {len(osm):,}개")
    else:
        osm = fetch_osm_pois()
        osm.to_parquet(osm_cache, index=False)
        log(f"  합계 {len(osm):,}개 — {osm_cache.name}에 보관")
    # 보관된 응답에는 지금 대응표에서 빠진 태그가 남아 있을 수 있다. 대응표를 고쳤을 때
    # 전부 다시 받지 않아도 되도록, 현재 선언된 태그만 남기고 좁은/넓은 구분도 다시 붙인다.
    scope_of = {(cat, tag): ("narrow" if s == "osm_narrow" else "extra")
                for cat, spec in CATEGORY_MAP.items()
                for s in ("osm_narrow", "osm_extra") for tag in spec[s]}
    before = len(osm)
    osm = osm[[(c, t) in scope_of for c, t in zip(osm["cat"], osm["tag"])]].copy()
    osm["scope"] = [scope_of[(c, t)] for c, t in zip(osm["cat"], osm["tag"])]
    if before != len(osm):
        log(f"  대응표에서 빠진 태그 {before - len(osm):,}개 제외 → {len(osm):,}개")
    for (cat, tag), n in osm.groupby(["cat", "tag"]).size().items():
        log(f"    {cat} [{tag}]: {n:,}개")

    osm_gdf = gpd.GeoDataFrame(
        osm, geometry=gpd.points_from_xy(osm["lon"], osm["lat"]), crs=4326
    ).to_crs(gdf.crs)
    joined = gpd.sjoin(osm_gdf, gdf[["ADSTRD_CD", "geometry"]], how="inner", predicate="within")
    log(f"  서울 행정동 안에 떨어진 것 {len(joined):,}개 "
        f"(경계 밖 {len(osm) - len(joined):,}개는 bbox가 서울보다 넓어 생긴다)")
    joined["dong_cd"] = joined["ADSTRD_CD"].astype(int)
    for scope, sub in (("osm_narrow", joined[joined["scope"] == "narrow"]),
                       ("osm_broad", joined)):
        for (d, c), n in sub.groupby(["dong_cd", "cat"]).size().items():
            counts.append({"dong_cd": int(d), "cat": c, "source": scope, "n": int(n)})

    counts_df = pd.DataFrame(counts)

    # --- 4) 카드 기반 추정매출 ----------------------------------------------
    log(f"\n[4] 서울 상권분석서비스 추정매출-행정동 {SALES_YEAR}년 (OA-22175)")
    sales_blob = cached_download(SALES_OID, SALES_SEQ, f"seoul_sales_dong_{SALES_YEAR}.zip")
    sales_raw, sales_name = read_zip_csv(sales_blob)
    log(f"  원본: {sales_name} — {len(sales_raw):,}행")
    keep = {
        "기준_년분기_코드": "yq", "행정동_코드": "dong_cd", "서비스_업종_코드": "svc_cd",
        "서비스_업종_코드_명": "svc_nm", "당월_매출_금액": "amt", "당월_매출_건수": "cnt",
        "연령대_20_매출_금액": "amt_20s", "연령대_60_이상_매출_금액": "amt_60p",
        "연령대_20_매출_건수": "cnt_20s", "연령대_60_이상_매출_건수": "cnt_60p",
    }
    sales = sales_raw[list(keep)].rename(columns=keep)
    sales["dong_cd"] = sales["dong_cd"].astype(int)
    log(f"  행정동 {sales['dong_cd'].nunique()}개 × 업종 {sales['svc_cd'].nunique()}개 "
        f"× 분기 {sorted(sales['yq'].unique())}")

    # --- 5) 저장 ------------------------------------------------------------
    log("\n[5] 저장")
    dong.to_parquet(DATA_DIR / "bias_dong.parquet", index=False)
    counts_df.to_parquet(DATA_DIR / "bias_counts.parquet", index=False)
    sales.to_parquet(DATA_DIR / "bias_sales.parquet", index=False)
    for f in ("bias_dong", "bias_counts", "bias_sales"):
        p = DATA_DIR / f"{f}.parquet"
        log(f"  {p.name} — {p.stat().st_size / 1024:.0f} KB")

    meta = {
        "처리_일자": date.today().isoformat(),
        "상가정보_원본": sbiz_path.name,
        "추정매출_원본": sales_name,
        "행정동_경계": "서울시 상권분석서비스(영역-행정동), EPSG:5181",
        "OSM_조회_bbox": list(SEOUL_BBOX),
        "업종군_코드": {k: v["svc_cd"] for k, v in ALL_MAP.items()},
        "업종_대응표": {
            k: {"상권정보_좁은": list(v["narrow"]),
                "상권정보_넓은추가": list(v["broad_extra"]),
                "OSM_좁은": list(v.get("osm_narrow", ())),
                "OSM_넓은추가": list(v.get("osm_extra", ()))}
            for k, v in ALL_MAP.items()
        },
    }
    (DATA_DIR / "SOURCES_bias.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log("  SOURCES_bias.json")
    log("\n완료. 다음: python 3-4-commercial-data-bias.py")


if __name__ == "__main__":
    main()
