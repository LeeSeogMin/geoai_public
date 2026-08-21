"""
3-1. STAC API 검색 실습
======================
pystac-client로 Microsoft Planetary Computer STAC API에 접속하여
서울 지역 Sentinel-2 영상을 검색하고, 메타데이터를 분석한다.

실행 방법:
    cd practice/chapter3
    python -m venv venv
    source venv/bin/activate
    pip install -r code/requirements.txt
    python code/3-1-stac-search.py
"""

import time

from pystac_client import Client
from pystac_client.exceptions import APIError


def fetch_items(search, retries=3, backoff=5):
    """STAC 검색 결과를 가져온다.
    Planetary Computer API는 간헐적으로 타임아웃을 반환하므로,
    일시적 APIError에 대해 지수 백오프로 재시도한다(근본 원인=서버 지연).
    """
    for attempt in range(1, retries + 1):
        try:
            return list(search.item_collection())
        except APIError as e:
            if attempt == retries:
                raise
            print(f"    (재시도 {attempt}/{retries - 1}: {type(e).__name__}) "
                  f"{backoff * attempt}초 대기 후 재요청")
            time.sleep(backoff * attempt)


# ============================================================
# 1. STAC 카탈로그 접속
# ============================================================
print("=" * 60)
print("STAC API 검색 실습 — Microsoft Planetary Computer")
print("=" * 60)

# Planetary Computer STAC API (인증 불필요 — 검색만 수행)
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ============================================================
# 2. 컬렉션 탐색
# ============================================================
print("\n--- 1. Sentinel 관련 컬렉션 ---")
sentinel_collections = []
for collection in catalog.get_collections():
    if "sentinel" in collection.id.lower():
        sentinel_collections.append(collection)
        print(f"  {collection.id}: {collection.title}")

print(f"\n  총 {len(sentinel_collections)}개 Sentinel 컬렉션 발견")

# ============================================================
# 3. 서울 지역 Sentinel-2 검색
# ============================================================
print("\n--- 2. 서울 지역 Sentinel-2 검색 ---")

# 서울 bbox (경도min, 위도min, 경도max, 위도max)
seoul_bbox = [126.8, 37.4, 127.2, 37.7]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=seoul_bbox,
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=20,
)

items = fetch_items(search)
print(f"  검색 조건: 2024년 6-8월, 구름 < 10%")
print(f"  검색 결과: {len(items)}개 영상")

# ============================================================
# 4. 검색 결과 메타데이터 분석
# ============================================================
print("\n--- 3. 검색 결과 메타데이터 ---")
for i, item in enumerate(items[:5]):
    props = item.properties
    cloud = props.get("eo:cloud_cover", "N/A")
    dt = item.datetime.strftime("%Y-%m-%d %H:%M") if item.datetime else "N/A"
    platform = props.get("platform", "N/A")
    print(f"  [{i+1}] {dt} | 구름: {cloud:.1f}% | 플랫폼: {platform}")

if len(items) > 5:
    print(f"  ... 외 {len(items) - 5}개")

# ============================================================
# 5. STAC Item 구조 확인
# ============================================================
print("\n--- 4. STAC Item 구조 (첫 번째 Item) ---")
if items:
    item = items[0]
    print(f"  ID: {item.id}")
    print(f"  날짜: {item.datetime}")
    print(f"  좌표계: EPSG:{item.properties.get('proj:epsg', 'N/A')}")
    print(f"  구름비율: {item.properties.get('eo:cloud_cover', 'N/A'):.1f}%")

    # Asset 목록
    print(f"\n  Asset 목록 ({len(item.assets)}개):")
    for key, asset in sorted(item.assets.items()):
        media = asset.media_type or "N/A"
        # 간략히 표시
        print(f"    {key:15s} | {media}")

# ============================================================
# 6. 구름 비율 통계
# ============================================================
print("\n--- 5. 구름 비율 통계 ---")
cloud_covers = [
    item.properties.get("eo:cloud_cover", 0) for item in items
]
if cloud_covers:
    avg_cloud = sum(cloud_covers) / len(cloud_covers)
    min_cloud = min(cloud_covers)
    max_cloud = max(cloud_covers)
    print(f"  평균 구름 비율: {avg_cloud:.1f}%")
    print(f"  최소 구름 비율: {min_cloud:.1f}%")
    print(f"  최대 구름 비율: {max_cloud:.1f}%")
    print(f"  총 {len(items)}개 영상 중 구름 5% 미만: "
          f"{sum(1 for c in cloud_covers if c < 5)}개")

# ============================================================
# 7. 다른 컬렉션 검색 비교 (Landsat)
# ============================================================
print("\n--- 6. Landsat C2 L2 검색 비교 ---")
search_landsat = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=seoul_bbox,
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=20,
)
landsat_items = fetch_items(search_landsat)
print(f"  동일 조건 Landsat 검색 결과: {len(landsat_items)}개 영상")
for item in landsat_items[:3]:
    dt = item.datetime.strftime("%Y-%m-%d") if item.datetime else "N/A"
    cloud = item.properties.get("eo:cloud_cover", "N/A")
    print(f"    {dt}: 구름 {cloud:.1f}%")

print(f"\n  Sentinel-2: {len(items)}개 vs Landsat: {len(landsat_items)}개")
print("  → Sentinel-2의 재방문 주기(5일)가 Landsat(16일)보다 짧아 더 많은 영상 확보")

print("\n[완료] STAC API 검색 실습을 마쳤다.")
