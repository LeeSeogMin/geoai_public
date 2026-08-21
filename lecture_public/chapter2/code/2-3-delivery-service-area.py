"""
2장 분석 예제 3: 배달 권역 — 원형 반경과 도로망 도달권의 차이
=====================================================================
"우리 매장은 반경 3km까지 배달합니다." 이 한 줄에는 세 가지 선택이 숨어 있다.
어느 좌표계에서 재는가, 직선으로 재는가 도로로 재는가, 거리로 재는가
시간으로 재는가. 셋 중 하나만 바꿔도 권역에 들어오는 집이 달라진다.

이 코드는 같은 "3km"를 다섯 방식으로 계산해 그 차이를 면적·건물 수·주행
거리로 잰다. 세 지점(옥수·신림·화곡)에서 모두 계산해, 결론이 특정 지점을
고른 결과가 아닌지 확인한다.

다섯 권역
  A1  경위도 버퍼      EPSG:4326에서 buffer(3000/111320°)  — 3km를 도로 환산해 버퍼
  A2  웹 메르카토르 버퍼 EPSG:3857에서 buffer(3000)          — 미터 단위지만 축척이 틀린 좌표계
  B   투영 버퍼        EPSG:5179에서 buffer(3000)          — 올바른 직선 3km(실무 기본값)
  C   도로망 도달권     주행 네트워크 거리 3,000m 이내
  D   시간 도달권       주행 10분(평균 20km/h 가정 → 3,333m)

읽는 데이터 (2-0b-prepare-osm-snapshot.py가 만든 스냅샷)
  ../data/osm_nodes.parquet      도로망 교차점(EPSG:5179 미터 좌표 포함)
  ../data/osm_edges.parquet      도로 구간과 실제 길이(m)
  ../data/osm_buildings.parquet  건물 대표점과 바닥면적
  ../data/store_points.geojson   매장 후보 지점 3곳

출처: OpenStreetMap — © OpenStreetMap contributors (ODbL 1.0)

가정과 그 표시
  - "건물 1채 = 수요 1단위"가 아니다. 건물 수는 권역 안에 사람이 얼마나 있는지
    보는 대리지표다. 층수 태그 보유율이 낮아(스냅샷 14~29%) 연면적으로 올리지
    않고 채수와 바닥면적만 쓴다.
  - 평균 주행속도, 월 주문 건수, 거리 할증 단가는 모두 **가정값**이다.
    출력에서 [가정]으로 표시하고 민감도를 함께 낸다.

실행:
    python 2-3-delivery-service-area.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버·CI)에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point

# 크로스 플랫폼 한글 폰트: 사용 가능한 첫 후보 사용(macOS/Windows/Linux).
# 없으면 그림 라벨의 한글이 □로 보일 뿐 결과·수치에는 영향 없음 → 경고만 억제.
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPSG_KOREA = 5179  # Korea 2000 / Unified CS — 미터 단위, 국내 통계·격자 표준

# ---------------------------------------------------------------- 설계 상수
RADIUS_M = 3000.0                 # 실무 기본값 "반경 3km"
DEG_PER_M = 1.0 / 111_320.0       # 위도 1도 ≈ 111.32km로 환산한 값(적도 기준 근사)
PROMISE_MIN = 10.0                # 주행 약속시간(분) — 배달 약속 30분 중 주행 몫 [가정]
SPEED_BASE_KMH = 20.0             # 도심 배달 평균 주행속도 [가정]
SPEED_SENS_KMH = (15.0, 20.0, 25.0)   # 속도 민감도 3구간
HULL_RATIOS = (0.1, 0.3, 0.5)     # concave hull 파라미터 민감도
HULL_MAIN = 0.3                   # 본문 표에 쓰는 기준값

# 비용 가정 — 전부 가정값이다. 실제 배달 대행 단가는 계약마다 다르다.
MONTHLY_ORDERS = 1000             # 매장 월 주문 건수 [가정]
PER_KM_FEE_SENS = (500, 1000, 1500)   # 거리 할증 단가(원/km) [가정]

RNG = np.random.default_rng(42)
N_RELOCATE = 200                  # 매장 위치 재배치 반복 횟수
RELOCATE_RADIUS_M = 2000.0        # 재배치 후보를 중심에서 이 거리 안으로 제한

# 완전한 격자(맨해튼) 도로망에서 방향이 균등할 때 우회비의 이론 평균 = 4/π.
# 실측 우회비를 이 값과 견주면 "지형 때문에 특별히 돌아가는지"를 판별할 수 있다.
GRID_BENCHMARK = 4.0 / np.pi


def build_graph(edges: pd.DataFrame, osmid_to_idx: dict[int, int]) -> csr_matrix:
    """도로 구간 표를 최단거리 계산용 희소 행렬로 바꾼다.

    같은 두 교차점을 잇는 구간이 여러 개면(평행 엣지) 가장 짧은 것만 남긴다.
    방향은 살린다 — 일방통행 도로는 반대 방향 구간이 애초에 없다.
    """
    u = edges["u"].map(osmid_to_idx).to_numpy()
    v = edges["v"].map(osmid_to_idx).to_numpy()
    w = edges["length"].to_numpy(dtype=float)
    n = len(osmid_to_idx)
    m = csr_matrix((w, (u, v)), shape=(n, n))
    # csr_matrix는 중복 (u,v)를 더해 버리므로, 중복을 미리 제거해 최솟값만 남긴다
    key = u.astype(np.int64) * n + v
    order = np.lexsort((w, key))
    keep = np.ones(len(key), dtype=bool)
    keep[1:] = key[order][1:] != key[order][:-1]
    sel = order[keep]
    return csr_matrix((w[sel], (u[sel], v[sel])), shape=(n, n))


def zone_area_km2(poly) -> float:
    """EPSG:5179 폴리곤의 면적(km²)."""
    return float(poly.area) / 1e6


def concave_zone(xy: np.ndarray, ratio: float):
    """도달 노드 집합을 감싸는 오목 껍질. 볼록 껍질을 쓰면 도달 불가 구역까지 삼킨다."""
    if len(xy) < 4:
        return MultiPoint([Point(p) for p in xy]).convex_hull
    return shapely.concave_hull(MultiPoint([Point(p) for p in xy]), ratio=ratio)


SNAPSHOT_FILES = [
    "osm_snapshot_meta.json",
    "osm_nodes.parquet",
    "osm_edges.parquet",
    "osm_buildings.parquet",
    "store_points.geojson",
]


def require_snapshot() -> int:
    """스냅샷이 없으면 무엇을 해야 하는지 알려 주고 멈춘다."""
    missing = [n for n in SNAPSHOT_FILES if not (DATA_DIR / n).exists()]
    if not missing:
        return 0
    print("[중단] 배달 권역 분석에 필요한 OpenStreetMap 스냅샷이 없습니다.")
    print(f"  없는 파일: {', '.join(missing)}")
    print(f"  있어야 할 위치: {DATA_DIR}")
    print("\n  스냅샷은 저장소에 함께 배포됩니다. 내려받은 사본에 없다면 준비 스크립트로")
    print("  다시 만들 수 있습니다(인터넷 연결과 osmnx가 필요합니다).")
    print("      pip install -r practice/requirements.txt")
    print("      python practice/chapter2/code/2-0b-prepare-osm-snapshot.py --refresh")
    print("\n  주의: 다시 만들면 그 사이 갱신된 OpenStreetMap을 받으므로 교재 본문의")
    print("  표·그림 수치와 값이 조금 달라집니다. 절차와 출처는 data/README.md 참고.")
    return 1


def main() -> int:
    print("=" * 78)
    print("2장 분석 예제 3: 배달 권역 — 원형 반경과 도로망 도달권의 차이")
    print("=" * 78)

    if require_snapshot() != 0:
        return 1

    meta = json.loads((DATA_DIR / "osm_snapshot_meta.json").read_text(encoding="utf-8"))
    print(f"\n데이터: OpenStreetMap 스냅샷 (취득 {meta['취득일']}, {meta['라이선스']})")
    print(f"도로망 반경 {meta['도로망_반경_m']:,}m · 건물 반경 {meta['건물_반경_m']:,}m")

    nodes_all = pd.read_parquet(DATA_DIR / "osm_nodes.parquet")
    edges_all = pd.read_parquet(DATA_DIR / "osm_edges.parquet")
    bldg_all = pd.read_parquet(DATA_DIR / "osm_buildings.parquet")
    sites = gpd.read_file(DATA_DIR / "store_points.geojson").to_crs(EPSG_KOREA)
    sites_wgs = gpd.read_file(DATA_DIR / "store_points.geojson")

    print("\n[가정] 다음 값은 실측이 아니라 가정이다.")
    print(f"  주행 약속시간 {PROMISE_MIN:.0f}분 · 평균 주행속도 {SPEED_BASE_KMH:.0f}km/h "
          f"(민감도 {SPEED_SENS_KMH[0]:.0f}/{SPEED_SENS_KMH[1]:.0f}/{SPEED_SENS_KMH[2]:.0f})")
    print(f"  월 주문 {MONTHLY_ORDERS:,}건 · 거리 할증 "
          f"{PER_KM_FEE_SENS[0]:,}/{PER_KM_FEE_SENS[1]:,}/{PER_KM_FEE_SENS[2]:,}원/km")

    zone_rows, diag_rows, hull_rows, reloc_rows = [], [], [], []
    panels = {}

    for _, site_row in sites.iterrows():
        site = site_row["site"]
        nodes = nodes_all[nodes_all["site"] == site].reset_index(drop=True)
        edges = edges_all[edges_all["site"] == site]
        bldg = bldg_all[bldg_all["site"] == site].reset_index(drop=True)

        osmid_to_idx = {int(o): i for i, o in enumerate(nodes["osmid"])}
        node_xy = nodes[["x", "y"]].to_numpy(dtype=float)
        graph = build_graph(edges, osmid_to_idx)

        store_pt = site_row.geometry           # EPSG:5179
        store_xy = np.array([store_pt.x, store_pt.y])
        store_wgs = sites_wgs.loc[sites_wgs["site"] == site, "geometry"].iloc[0]

        print("\n" + "─" * 78)
        print(f"[{site}] {site_row['label']} — 차단 지형: {site_row['barrier']} "
              f"(사전예상: {site_row['사전예상']})")
        print("─" * 78)

        # ---------------------------------------------------------- 매장 스냅
        tree_nodes = cKDTree(node_xy)
        snap_dist, snap_idx = tree_nodes.query(store_xy)
        print(f"  매장 지점을 가장 가까운 도로 교차점에 붙였다: {snap_dist:,.0f}m 이동")
        print(f"  도로망 규모: 교차점 {len(nodes):,}개 · 구간 {len(edges):,}개 "
              f"· 건물 {len(bldg):,}채")

        # ---------------------------------------------------------- 거리 계산
        # 건물마다 (ㄱ) 매장에서의 직선거리 (ㄴ) 가장 가까운 도로 교차점까지의
        # 주행거리를 구한다. 건물은 도로에 붙어 있으므로 교차점 거리로 근사한다.
        b_xy = bldg[["x", "y"]].to_numpy(dtype=float)
        straight = np.hypot(b_xy[:, 0] - store_xy[0], b_xy[:, 1] - store_xy[1])
        _, b_node = tree_nodes.query(b_xy)

        # 상한을 두지 않고 끝까지 푼다. 상한을 두면 "멀다"와 "못 간다"가 둘 다
        # 무한대로 뭉개져, 초과 주행거리를 계산할 수 없다.
        net_node = dijkstra(graph, indices=snap_idx)
        network = net_node[b_node]
        unreachable = ~np.isfinite(network)
        if unreachable.any():
            print(f"  스냅샷 도로망(반경 {meta['도로망_반경_m']:,}m) 안에서 도달 경로를 "
                  f"찾지 못한 건물 {int(unreachable.sum()):,}채는 거리 통계에서 뺀다")

        # 반경 안에 도로가 아예 없는 구역이 얼마나 되는지 — 차단 지형의 직접 지표.
        # 원 안을 50m 격자로 훑어, 가장 가까운 도로 교차점이 200m를 넘는 칸을 센다.
        gx = np.arange(store_xy[0] - RADIUS_M, store_xy[0] + RADIUS_M, 50.0)
        gy = np.arange(store_xy[1] - RADIUS_M, store_xy[1] + RADIUS_M, 50.0)
        GX, GY = np.meshgrid(gx, gy)
        inside = np.hypot(GX - store_xy[0], GY - store_xy[1]) <= RADIUS_M
        gpts = np.column_stack([GX[inside], GY[inside]])
        gd, _ = tree_nodes.query(gpts)
        void_share = 100.0 * float((gd > 200.0).mean())
        print(f"  반경 3km 원 안에서 도로가 200m 안에 없는 구역: {void_share:.1f}%")

        # ---------------------------------------------------------- 다섯 권역
        # A1: 경위도(도) 좌표계에서 만든 버퍼. 위도 1도와 경도 1도의 실제 거리가
        #     달라 지표에서는 남북으로 긴 타원이 된다.
        # GeoPandas는 이 실수를 경고로 알려 준다. 경고를 끄지 않고 붙잡아 출력한다 —
        # 이 예제에서 가장 값싼 방어선이 바로 이 한 줄이기 때문이다.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a1 = (
                gpd.GeoSeries([store_wgs], crs="EPSG:4326")
                .buffer(RADIUS_M * DEG_PER_M)
                .to_crs(EPSG_KOREA)
                .iloc[0]
            )
        for w in caught:
            print(f"  [경고] A1을 만들 때 GeoPandas가 알려 준 말: "
                  f"{str(w.message).splitlines()[0]}")
        # A2: 웹 메르카토르. 미터 단위지만 위도 φ에서 축척이 1/cos φ배로 부풀어 있어
        #     buffer(3000)이 지표에서는 3000×cos φ 만큼만 덮는다.
        a2 = (
            gpd.GeoSeries([store_wgs], crs="EPSG:4326")
            .to_crs(3857)
            .buffer(RADIUS_M)
            .to_crs(EPSG_KOREA)
            .iloc[0]
        )
        b_zone = store_pt.buffer(RADIUS_M)

        promise_m = SPEED_BASE_KMH * 1000 / 60 * PROMISE_MIN
        reach_c = node_xy[net_node <= RADIUS_M]
        reach_d = node_xy[net_node <= promise_m]
        c_zone = concave_zone(reach_c, HULL_MAIN)
        d_zone = concave_zone(reach_d, HULL_MAIN)

        in_a1 = shapely.contains_xy(a1, b_xy[:, 0], b_xy[:, 1])
        in_a2 = shapely.contains_xy(a2, b_xy[:, 0], b_xy[:, 1])
        in_b = straight <= RADIUS_M
        in_c = network <= RADIUS_M
        in_d = network <= promise_m

        base_n = int(in_b.sum())
        base_area = zone_area_km2(b_zone)
        for name, desc, poly, mask in [
            ("A1", f"경위도 버퍼 {RADIUS_M * DEG_PER_M:.5f}°", a1, in_a1),
            ("A2", "웹 메르카토르 buffer(3000)", a2, in_a2),
            ("B", "투영(5179) buffer(3000)", b_zone, in_b),
            ("C", "도로망 3,000m 도달권", c_zone, in_c),
            ("D", f"주행 {PROMISE_MIN:.0f}분 등시권({promise_m:,.0f}m)", d_zone, in_d),
        ]:
            area = zone_area_km2(poly)
            n_b = int(mask.sum())
            zone_rows.append(
                {
                    "site": site, "zone": name, "설명": desc,
                    "면적_km2": area, "면적_B대비_%": 100 * area / base_area - 100,
                    "건물수": n_b, "건물_B대비_%": 100 * n_b / base_n - 100,
                    "바닥면적_ha": float(bldg.loc[mask, "footprint_m2"].sum()) / 1e4,
                }
            )

        print(f"\n  ▶ 권역 비교 (B = 올바른 직선 3km 기준)")
        print(f"    {'권역':<4} {'면적 km²':>9} {'B대비':>8} {'건물수':>8} {'B대비':>8}  설명")
        for r in zone_rows[-5:]:
            print(f"    {r['zone']:<4} {r['면적_km2']:>9.2f} {r['면적_B대비_%']:>+7.1f}% "
                  f"{r['건물수']:>8,} {r['건물_B대비_%']:>+7.1f}%  {r['설명']}")

        # ------------------------------------------------- 오목 껍질 파라미터
        for ratio in HULL_RATIOS:
            hull_rows.append(
                {
                    "site": site, "ratio": ratio,
                    "C_면적_km2": zone_area_km2(concave_zone(reach_c, ratio)),
                    "D_면적_km2": zone_area_km2(concave_zone(reach_d, ratio)),
                }
            )
        hr = [h for h in hull_rows if h["site"] == site]
        print("\n    오목 껍질 ratio 민감도 — 면적은 파라미터에 따라 움직인다")
        print("      ratio " + "  ".join(f"{h['ratio']:.1f}: C {h['C_면적_km2']:.2f}km²" for h in hr))

        # ---------------------------------------------------------- 오배정 진단
        # B∖C: 직선으로는 3km 안인데 실제로는 3km 넘게 달려야 하는 건물
        # D∖B: 직선 3km 밖인데 10분 안에 갈 수 있는 건물
        b_not_c = in_b & ~in_c
        d_not_b = in_d & ~in_b
        detour = np.divide(network, straight, out=np.full_like(straight, np.nan),
                           where=(straight > 50) & np.isfinite(network))
        det_in_b = detour[in_b & np.isfinite(detour)]

        share_bnc = 100 * b_not_c.sum() / base_n
        extra_km = (network[b_not_c & np.isfinite(network)] - RADIUS_M) / 1000.0

        # 컷라인: 직선 반경을 얼마로 줄이면 오배정이 사라지는가
        order = np.argsort(straight)
        s_sorted, bad_sorted = straight[order], (~in_c)[order]
        cum_bad = np.cumsum(bad_sorted)
        cum_n = np.arange(1, len(s_sorted) + 1)
        within = s_sorted <= RADIUS_M
        bad_rate = np.where(within, cum_bad / cum_n, np.nan)
        first_bad = s_sorted[bad_sorted & within]
        cut_strict = float(first_bad.min()) if first_bad.size else RADIUS_M
        ok95 = np.where(within & (bad_rate <= 0.05))[0]
        cut_95 = float(s_sorted[ok95.max()]) if ok95.size else 0.0

        print(f"\n  ▶ 오배정 진단")
        print(f"    B∖C (직선 3km 안 · 주행 3km 초과): {int(b_not_c.sum()):,}채 "
              f"({share_bnc:.1f}%)")
        print(f"    D∖B (직선 3km 밖 · 10분 안 도달)  : {int(d_not_b.sum()):,}채 "
              f"(B 건물수의 {100 * d_not_b.sum() / base_n:.1f}%)")
        print(f"    우회비(주행/직선) 중앙값 {np.median(det_in_b):.3f} · "
              f"90분위 {np.percentile(det_in_b, 90):.3f} "
              f"(완전 격자 이론값 {GRID_BENCHMARK:.3f})")
        if extra_km.size:
            print(f"    B∖C 건물의 추가 주행거리 중앙값 {np.median(extra_km) * 1000:,.0f}m · "
                  f"최대 {extra_km.max() * 1000:,.0f}m")
        print(f"    반경 컷라인: 오배정 0%까지 줄이면 {cut_strict:,.0f}m, "
              f"오배정 5% 이하면 {cut_95:,.0f}m")
        # 우회비가 어디서나 같다면 컷라인은 3,000m ÷ 우회비다. 이 근사가 어긋나는
        # 곳이 곧 지형이 도로를 끊어 우회비가 한쪽으로 쏠린 곳이다.
        naive_cut = RADIUS_M / float(np.median(det_in_b))
        print(f"    우회비 중앙값만으로 낸 근사 컷라인 {naive_cut:,.0f}m "
              f"(실측 5% 컷라인과 {cut_95 - naive_cut:+,.0f}m 차이)")

        diag_rows.append(
            {
                "site": site,
                "B건물수": base_n,
                "BnotC_건물수": int(b_not_c.sum()),
                "BnotC_%": share_bnc,
                "DnotB_건물수": int(d_not_b.sum()),
                "DnotB_%": 100 * d_not_b.sum() / base_n,
                "우회비_중앙값": float(np.median(det_in_b)),
                "우회비_p90": float(np.percentile(det_in_b, 90)),
                "추가주행_중앙값_m": float(np.median(extra_km) * 1000) if extra_km.size else 0.0,
                "컷라인_0%_m": cut_strict,
                "컷라인_5%_m": cut_95,
                "근사컷라인_m": naive_cut,
                "도로공백_%": void_share,
                "평균추가주행_km": float(extra_km.mean()) if extra_km.size else 0.0,
                "도달불가_건물수": int(unreachable.sum()),
            }
        )

        # ---------------------------------------------------------- 속도 민감도
        print("\n    주행 10분 등시권의 속도 가정 민감도")
        for spd in SPEED_SENS_KMH:
            lim = spd * 1000 / 60 * PROMISE_MIN
            n_in = int((network <= lim).sum())
            n_out = int(((network <= lim) & ~in_b).sum())
            print(f"      {spd:>4.0f}km/h → {lim:,.0f}m: 건물 {n_in:,}채 "
                  f"(B 대비 {100 * n_in / base_n - 100:+.1f}%), "
                  f"그중 직선 3km 밖 {n_out:,}채")
        # "반경 3km"라는 약속은 사실 속도를 하나 정해 둔 것과 같다. 10분 등시권이
        # 원과 같은 수의 건물을 담으려면 평균 몇 km/h로 달려야 하는지 역산한다.
        net_ok = np.sort(network[np.isfinite(network)])
        equal_m = float(net_ok[min(base_n, len(net_ok)) - 1])
        equal_kmh = equal_m / 1000.0 / (PROMISE_MIN / 60.0)
        print(f"      원과 같은 {base_n:,}채를 10분에 담으려면 주행 {equal_m:,.0f}m "
              f"→ 평균 {equal_kmh:.1f}km/h가 필요하다")
        diag_rows[-1]["동일건물수_필요속도_kmh"] = equal_kmh

        # ---------------------------------------------------------- 재배치 검사
        # 이 지점을 고른 것이 결론을 만들었는지 본다. 같은 동네 안에서 매장을
        # 200번 다시 놓아 보고, 실제 지점의 값이 그 분포의 어디에 있는지 확인한다.
        near = np.where(
            np.hypot(node_xy[:, 0] - store_xy[0], node_xy[:, 1] - store_xy[1])
            <= RELOCATE_RADIUS_M
        )[0]
        pick = RNG.choice(near, size=min(N_RELOCATE, len(near)), replace=False)
        nd_multi = dijkstra(graph, indices=pick, limit=RADIUS_M * 1.05)
        shares = []
        for k in range(len(pick)):
            sxy = node_xy[pick[k]]
            st = np.hypot(b_xy[:, 0] - sxy[0], b_xy[:, 1] - sxy[1])
            nb = st <= RADIUS_M
            if nb.sum() < 100:
                continue
            nc = nd_multi[k][b_node] <= RADIUS_M
            shares.append(100 * (nb & ~nc).sum() / nb.sum())
        shares = np.array(shares)
        pctl = float((shares < share_bnc).mean() * 100)
        print(f"\n  ▶ 매장 위치 재배치 {len(shares)}회 (반경 {RELOCATE_RADIUS_M:,.0f}m 안 교차점)")
        print(f"    B∖C 비율 분포: 중앙값 {np.median(shares):.1f}% · "
              f"10~90분위 {np.percentile(shares, 10):.1f}~{np.percentile(shares, 90):.1f}%")
        print(f"    실제 지점 {share_bnc:.1f}% → 분포의 {pctl:.0f}분위")
        reloc_rows.append(
            {
                "site": site, "n": len(shares), "실제_%": share_bnc,
                "중앙값_%": float(np.median(shares)),
                "p10_%": float(np.percentile(shares, 10)),
                "p90_%": float(np.percentile(shares, 90)),
                "분위": pctl,
            }
        )

        panels[site] = {
            "label": site_row["label"], "store": store_xy, "node_xy": node_xy,
            "b": b_zone, "a1": a1, "a2": a2, "c": c_zone, "d": d_zone,
            "bad_xy": b_xy[b_not_c], "out_xy": b_xy[d_not_b],
        }

    # ================================================================ 종합
    zdf = pd.DataFrame(zone_rows)
    ddf = pd.DataFrame(diag_rows)

    print("\n" + "=" * 78)
    print("종합 1 — 다섯 방식의 권역 면적과 건물 수 (표 2.12)")
    print("=" * 78)
    print(f"  {'권역':<4} " + " ".join(f"{s:>18}" for s in ddf['site']))
    for zone in ["A1", "A2", "B", "C", "D"]:
        sub = zdf[zdf["zone"] == zone].set_index("site")
        cells = " ".join(
            f"{sub.loc[s, '면적_km2']:>7.2f}km²{sub.loc[s, '건물수']:>7,}채"
            for s in ddf["site"]
        )
        print(f"  {zone:<4} {cells}")
    print("\n  B(올바른 직선 3km) 대비 건물 수 증감률")
    for zone in ["A1", "A2", "C", "D"]:
        sub = zdf[zdf["zone"] == zone].set_index("site")
        cells = " ".join(f"{s}: {sub.loc[s, '건물_B대비_%']:>+6.1f}%" for s in ddf["site"])
        print(f"  {zone:<4} {cells}")

    print("\n" + "=" * 78)
    print("종합 2 — 오배정과 반경 컷라인 (표 2.13)")
    print("=" * 78)
    print(f"  {'지점':<6} {'B건물':>8} {'B∖C':>8} {'B∖C%':>7} {'D∖B%':>7} "
          f"{'우회비중앙':>10} {'컷라인5%':>9} {'근사컷라인':>10} {'도로공백':>8}")
    for _, r in ddf.iterrows():
        print(f"  {r['site']:<6} {r['B건물수']:>8,} {r['BnotC_건물수']:>8,} "
              f"{r['BnotC_%']:>6.1f}% {r['DnotB_%']:>6.1f}% "
              f"{r['우회비_중앙값']:>10.3f} {r['컷라인_5%_m']:>8,.0f}m "
              f"{r['근사컷라인_m']:>9,.0f}m {r['도로공백_%']:>7.1f}%")
    print(f"\n  참고: 완전한 격자 도로망의 이론 우회비 = 4/π = {GRID_BENCHMARK:.3f}")
    print("  근사 컷라인 = 3,000m ÷ 우회비 중앙값. 실측 컷라인이 이보다 많이 낮으면")
    print("  우회가 고르게 퍼진 것이 아니라 한쪽이 막혀 있다는 뜻이다.")

    print("\n종합 3 — 오배정을 돈으로 옮기면 [전부 가정값]")
    print(f"  가정: 매장 월 주문 {MONTHLY_ORDERS:,}건, 주문은 권역 안 건물에 비례,")
    print(f"        B∖C 주문은 초과 거리만큼 거리 할증을 더 문다(편도 기준)")
    print(f"  {'지점':<6} {'B∖C주문/월':>10} {'평균초과':>9} " +
          " ".join(f"{f:>10,}원/km" for f in PER_KM_FEE_SENS))
    for _, r in ddf.iterrows():
        orders = MONTHLY_ORDERS * r["BnotC_%"] / 100
        cells = " ".join(
            f"{orders * r['평균추가주행_km'] * f:>13,.0f}원" for f in PER_KM_FEE_SENS
        )
        print(f"  {r['site']:<6} {orders:>9.0f}건 {r['평균추가주행_km'] * 1000:>8,.0f}m {cells}")

    print("\n종합 4 — 사전 예상은 맞았는가")
    for _, r in ddf.iterrows():
        exp = sites.loc[sites["site"] == r["site"], "사전예상"].iloc[0]
        bar = sites.loc[sites["site"] == r["site"], "barrier"].iloc[0]
        print(f"  {r['site']:<6} 예상 '{exp}' / 실측 B∖C {r['BnotC_%']:.1f}%, "
              f"우회비 {r['우회비_중앙값']:.3f}, 도로공백 {r['도로공백_%']:.1f}% (차단: {bar})")
    print("\n종합 5 — '반경 3km'가 암묵적으로 전제하는 주행속도")
    for _, r in ddf.iterrows():
        print(f"  {r['site']:<6} 원과 같은 건물 수를 10분에 담으려면 "
              f"평균 {r['동일건물수_필요속도_kmh']:.1f}km/h "
              f"(가정한 {SPEED_BASE_KMH:.0f}km/h의 {r['동일건물수_필요속도_kmh'] / SPEED_BASE_KMH:.2f}배)")

    print("\n  이 예제가 보이는 것과 보이지 않는 것")
    print("  - 보인다: 같은 3km라도 좌표계·거리 정의를 바꾸면 권역 안 건물 수가 달라진다.")
    print("            B∖C 건물 수와 우회비, 컷라인은 오목 껍질 파라미터와 무관하게")
    print("            건물별 주행거리로 직접 세므로 파라미터의 산물이 아니다.")
    print("  - 보이지 않는다: 권역 면적(C·D)은 오목 껍질 ratio에 따라 움직인다.")
    print("            건물 수는 주문 수가 아니고, 월 손실액은 가정값의 곱이다.")
    print("            도로망은 반경 5km에서 잘려 있어 경계 부근 우회 경로가 빠질 수 있다.")
    print("            네트워크 거리는 도로 길이이며 신호·정체·이륜차 통행 규칙은 넣지 않았다.")

    # ---------------------------------------------------------------- 지도
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    for ax, (site, p) in zip(axes, panels.items()):
        ax.scatter(p["node_xy"][:, 0], p["node_xy"][:, 1], s=0.4, c="#d9d9d9",
                   linewidths=0, zorder=1)
        ax.scatter(p["bad_xy"][:, 0], p["bad_xy"][:, 1], s=1.6, c="#d62728",
                   linewidths=0, zorder=2, label="직선 3km 안 · 주행 3km 밖 건물")
        ax.scatter(p["out_xy"][:, 0], p["out_xy"][:, 1], s=1.6, c="#2ca02c",
                   linewidths=0, zorder=2, label="직선 3km 밖 · 주행 10분 안 건물")
        for geom, color, ls, lab in [
            (p["a1"], "#7f7f7f", ":", "A1 경위도 버퍼"),
            (p["a2"], "#9467bd", (0, (1, 3)), "A2 웹 메르카토르 버퍼"),
            (p["b"], "#1f77b4", "--", "B 직선 3km"),
            (p["c"], "#ff7f0e", "-", "C 도로망 3km 도달권"),
            (p["d"], "#2ca02c", "-.", "D 주행 10분 등시권"),
        ]:
            first = True
            for g in getattr(geom, "geoms", [geom]):
                if g.geom_type != "Polygon":
                    continue
                x, y = g.exterior.xy
                ax.plot(x, y, color=color, ls=ls, lw=1.6, zorder=3,
                        label=lab if first else None)
                first = False
        ax.plot(*p["store"], marker="*", ms=16, c="black", zorder=4, label="매장")
        ax.set_title(f"{site} — {p['label']}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("같은 '반경 3km'가 계산 방식에 따라 달라지는 범위 (EPSG:5179)", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out_png = RESULTS_DIR / "2-3-service-areas.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\n지도 저장: results/{out_png.name}")

    zdf.to_csv(RESULTS_DIR / "2-3-zone-summary.csv", index=False, encoding="utf-8-sig")
    ddf.to_csv(RESULTS_DIR / "2-3-misassignment.csv", index=False, encoding="utf-8-sig")
    print("표 저장: results/2-3-zone-summary.csv, results/2-3-misassignment.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
