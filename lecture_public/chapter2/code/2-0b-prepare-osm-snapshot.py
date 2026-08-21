"""
2장 준비(비즈니스 예제): 배달 권역 분석용 OpenStreetMap 스냅샷 만들기
=====================================================================
이 스크립트는 **실습 본편이 아니다.** 분석 코드(2-3)가 읽을 원자료를 한 번
내려받아 `practice/chapter2/data/`에 고정해 두는 준비 단계다. 스냅샷이 이미
있으면 네트워크에 접속하지 않고 요약만 출력하고 끝낸다. 그래서 하네스가 이
파일을 매번 실행해도 결과가 흔들리지 않는다.

왜 스냅샷을 고정하는가
  OpenStreetMap은 누구나 계속 편집하는 지도다. 같은 코드를 오늘과 다음 달에
  돌리면 도로 한 줄, 건물 몇 채가 달라져 본문 수치가 바뀐다. 교재의 수치가
  재현되려면 원자료를 시점으로 묶어 두어야 한다.

무엇을 받는가 (출처: OpenStreetMap, 라이선스 ODbL — © OpenStreetMap contributors)
  ① 주행 도로망(network_type="drive") — 자동차·오토바이가 다닐 수 있는 도로만
  ② 건물 풋프린트(building=*) — 권역 안에 사람이 얼마나 있는지 세는 대리지표
  세 지점을 함께 받는다. 한 지점만 보고 "직선 반경은 틀렸다"고 말하면 그 지점을
  고른 사람의 결론이 된다(§3.3 지점 선택의 자의성).

지점 선택은 **결과를 보기 전에** 지형 근거로 정했다.
  - 옥수(성동구): 남쪽이 한강으로 막힌다        → 차이가 클 것으로 예상
  - 신림(관악구): 남쪽이 관악산으로 막힌다      → 차이가 클 것으로 예상
  - 화곡(강서구): 큰 자연 장애물이 없는 평지 격자 → 차이가 작을 것으로 예상
  예상이 맞았는지는 분석 코드(2-3)가 실측으로 채점한다.

재취득(스냅샷을 새로 받고 싶을 때):
    python 2-0b-prepare-osm-snapshot.py --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# ---------------------------------------------------------------- 설정
EPSG_WGS84 = 4326      # 경위도(도) — OSM이 주는 좌표
EPSG_KOREA = 5179      # Korea 2000 / Unified CS(미터) — 거리·면적 계산은 여기서

# 도로망을 받을 반경(m). 10분 등시권이 최대 가정속도 25km/h에서 4,167m까지
# 뻗을 수 있으므로 그보다 넉넉하게 잡는다.
GRAPH_DIST_M = 5000
# 건물을 받을 반경(m). 가장 큰 권역(등시권)을 덮을 만큼만.
BUILDING_DIST_M = 4300

SITES = [
    {
        "site": "옥수",
        "label": "성동구 옥수동 일대",
        "lat": 37.5405,
        "lon": 127.0180,
        "barrier": "한강(남쪽) + 남산 자락(북쪽)",
        "expectation": "큰 차이",
    },
    {
        "site": "신림",
        "label": "관악구 신림동 일대",
        "lat": 37.4840,
        "lon": 126.9295,
        "barrier": "관악산(남쪽)",
        "expectation": "큰 차이",
    },
    {
        "site": "화곡",
        "label": "강서구 화곡동 일대",
        "lat": 37.5410,
        "lon": 126.8400,
        "barrier": "없음(평지 격자)",
        "expectation": "작은 차이",
    },
]

NODES_PATH = DATA_DIR / "osm_nodes.parquet"
EDGES_PATH = DATA_DIR / "osm_edges.parquet"
BLDG_PATH = DATA_DIR / "osm_buildings.parquet"
SITES_PATH = DATA_DIR / "store_points.geojson"
META_PATH = DATA_DIR / "osm_snapshot_meta.json"

ARTIFACTS = [NODES_PATH, EDGES_PATH, BLDG_PATH, SITES_PATH]


def sha256_of_file(path: Path) -> str:
    """파일 내용의 SHA-256. 스냅샷이 바뀌지 않았음을 증명하는 데 쓴다."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def summarize(reason: str) -> int:
    """이미 있는 스냅샷을 요약만 출력한다(네트워크 접속 없음)."""
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    print("=" * 72)
    print("2장 준비: OpenStreetMap 스냅샷")
    print("=" * 72)
    print(f"\n[건너뜀] {reason}")
    print(f"  취득일          : {meta['취득일']}")
    print(f"  출처            : {meta['출처']}")
    print(f"  라이선스        : {meta['라이선스']}")
    print(f"  도로망 반경(m)  : {meta['도로망_반경_m']}")
    print(f"  건물 반경(m)    : {meta['건물_반경_m']}")
    print("\n지점별 스냅샷 규모")
    print(f"  {'지점':<6} {'노드':>8} {'엣지':>8} {'건물':>8}  차단 지형")
    for row in meta["지점"]:
        print(
            f"  {row['site']:<6} {row['노드수']:>8,} {row['엣지수']:>8,} "
            f"{row['건물수']:>8,}  {row['barrier']}"
        )
    print("\n파일 해시(SHA-256 앞 16자리)")
    for name, digest in meta["파일해시"].items():
        print(f"  {name:<24} {digest[:16]}…")
    print("\n스냅샷을 새로 받으려면: python 2-0b-prepare-osm-snapshot.py --refresh")
    return 0


def build_snapshot() -> int:
    """OSM에서 도로망·건물을 받아 스냅샷 파일로 저장한다."""
    try:
        import osmnx as ox
    except ImportError:
        print("[오류] osmnx가 필요합니다.  pip install -r practice/requirements.txt")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # osmnx는 Overpass 응답을 캐시에 쌓는다. 지정하지 않으면 현재 작업 디렉터리
    # 아래 ./cache 에 수십 MB가 쌓여 저장소가 지저분해진다. 자기 장 안으로 못박는다.
    # (폴더 이름을 cache로 두면 저장소의 .gitignore 규칙에 그대로 걸린다)
    ox.settings.cache_folder = str(DATA_DIR / "cache")
    print("=" * 72)
    print("2장 준비: OpenStreetMap 스냅샷 내려받기")
    print("=" * 72)
    print("  © OpenStreetMap contributors — ODbL 1.0\n")

    node_frames, edge_frames, bldg_frames, meta_sites = [], [], [], []

    for cfg in SITES:
        site, lat, lon = cfg["site"], cfg["lat"], cfg["lon"]
        print(f"▶ {site} ({cfg['label']}) — ({lat}, {lon})")

        # --- 도로망 -------------------------------------------------
        # simplify=True: 교차로가 아닌 중간 노드를 접어 그래프를 줄인다.
        #   접힌 구간의 실제 길이는 엣지의 length 속성에 그대로 남으므로
        #   네트워크 거리 계산에는 손해가 없다.
        G = ox.graph_from_point(
            (lat, lon), dist=GRAPH_DIST_M, network_type="drive", simplify=True
        )
        nodes, edges = ox.graph_to_gdfs(G)
        print(f"    도로망 노드 {len(nodes):,} · 엣지 {len(edges):,}")

        nodes_m = nodes.to_crs(EPSG_KOREA)
        nd = pd.DataFrame(
            {
                "site": site,
                "osmid": nodes.index.to_numpy(),
                "lon": nodes.geometry.x.to_numpy(),
                "lat": nodes.geometry.y.to_numpy(),
                "x": nodes_m.geometry.x.to_numpy(),   # EPSG:5179 미터 좌표
                "y": nodes_m.geometry.y.to_numpy(),
            }
        )
        node_frames.append(nd)

        ed = edges.reset_index()[["u", "v", "length", "highway", "oneway"]].copy()
        ed["site"] = site
        # highway 태그는 리스트로 오는 경우가 있어(한 도로에 두 등급) 문자열로 눌러 둔다
        ed["highway"] = ed["highway"].astype(str)
        ed["oneway"] = ed["oneway"].astype(str)
        edge_frames.append(ed[["site", "u", "v", "length", "highway", "oneway"]])

        # --- 건물 ---------------------------------------------------
        bl = ox.features_from_point(
            (lat, lon), tags={"building": True}, dist=BUILDING_DIST_M
        )
        bl = bl[bl.geometry.notna()].copy()
        bl_m = bl.to_crs(EPSG_KOREA)
        # 폴리곤은 대표점(중심)으로, 점 객체는 그대로. 권역 포함 판정에 쓴다.
        cent = bl_m.geometry.representative_point()
        area = bl_m.geometry.area  # 점 객체는 0이 된다
        levels = bl["building:levels"] if "building:levels" in bl.columns else None
        bd = pd.DataFrame(
            {
                "site": site,
                "x": cent.x.to_numpy(),
                "y": cent.y.to_numpy(),
                "footprint_m2": area.to_numpy(),
                "building": bl["building"].astype(str).to_numpy(),
                "levels": (
                    pd.to_numeric(levels, errors="coerce").to_numpy()
                    if levels is not None
                    else float("nan")
                ),
            }
        )
        print(
            f"    건물 {len(bd):,}채 · 층수 태그 보유 "
            f"{int(bd['levels'].notna().sum()):,}채 "
            f"({bd['levels'].notna().mean():.1%})"
        )
        bldg_frames.append(bd)

        meta_sites.append(
            {
                "site": site,
                "label": cfg["label"],
                "lat": lat,
                "lon": lon,
                "barrier": cfg["barrier"],
                "사전예상": cfg["expectation"],
                "노드수": int(len(nd)),
                "엣지수": int(len(ed)),
                "건물수": int(len(bd)),
            }
        )

    # ---------------------------------------------------------------- 저장
    pd.concat(node_frames, ignore_index=True).to_parquet(NODES_PATH, index=False)
    pd.concat(edge_frames, ignore_index=True).to_parquet(EDGES_PATH, index=False)
    pd.concat(bldg_frames, ignore_index=True).to_parquet(BLDG_PATH, index=False)

    sites_gdf = gpd.GeoDataFrame(
        [
            {
                "site": c["site"],
                "label": c["label"],
                "barrier": c["barrier"],
                "사전예상": c["expectation"],
            }
            for c in SITES
        ],
        geometry=[Point(c["lon"], c["lat"]) for c in SITES],
        crs=f"EPSG:{EPSG_WGS84}",
    )
    sites_gdf.to_file(SITES_PATH, driver="GeoJSON")

    meta = {
        "취득일": date.today().isoformat(),
        "출처": "OpenStreetMap (Overpass API, osmnx)",
        "라이선스": "ODbL 1.0 — © OpenStreetMap contributors",
        "도로망_반경_m": GRAPH_DIST_M,
        "건물_반경_m": BUILDING_DIST_M,
        "좌표계": {"원본": EPSG_WGS84, "계산": EPSG_KOREA},
        "지점": meta_sites,
        "파일해시": {p.name: sha256_of_file(p) for p in ARTIFACTS},
    }
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n저장 완료")
    for p in ARTIFACTS + [META_PATH]:
        print(f"  {p.relative_to(DATA_DIR.parent)}  ({p.stat().st_size / 1024:,.0f} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="2장 OSM 스냅샷 준비")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="스냅샷이 있어도 OSM에서 다시 내려받는다(수치가 바뀔 수 있음)",
    )
    args = parser.parse_args()

    have_all = all(p.exists() for p in ARTIFACTS) and META_PATH.exists()
    if have_all and not args.refresh:
        return summarize("스냅샷이 이미 있어 내려받지 않았다(재현성 유지).")
    return build_snapshot()


if __name__ == "__main__":
    sys.exit(main())
