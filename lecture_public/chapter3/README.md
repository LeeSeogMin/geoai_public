# 3장 실습: STAC 검색·클라우드 네이티브 포맷과 상업 데이터 편향 진단

이 실습은 `docs/ch03.md`와 `lecture/chapter3.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `practice/chapter3/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `practice/README.md`의 설치 지침을 먼저 따르세요.
- `3-1-stac-search.py`는 Microsoft Planetary Computer STAC API에 접속하므로 인터넷 연결이 필요합니다.
- `3-0b-commercial-bias-data-prep.py`도 인터넷이 필요합니다(서울 열린데이터광장 zip 두 개와 Overpass 조회). **본편인 `3-4-commercial-data-bias.py`는 인터넷 없이 돌아갑니다** — 준비 스크립트가 만든 집계 파일만 읽습니다.

실습 파일

- `practice/chapter3/code/3-0b-commercial-bias-data-prep.py` — **준비 스크립트.** 세 출처(상가정보·서울 상권분석·OSM)를 행정동 단위 집계표로 만듭니다. 데이터를 처음 만들 때 또는 최신 자료로 다시 받고 싶을 때만 실행합니다
- `practice/chapter3/code/3-1-stac-search.py` — STAC API로 Sentinel-2·Landsat 영상 검색
- `practice/chapter3/code/3-2-cloud-formats.py` — GeoJSON·Shapefile·GeoParquet 저장/읽기 성능 비교
- `practice/chapter3/code/3-3-cog-partial-read.py` — COG 부분 읽기(HTTP Range Request) 실측: 전체 파일을 내려받지 않고 관심 영역만 읽어 시간·픽셀 비율을 비교하고, 받은 소구역만 GeoTIFF로 저장
- `practice/chapter3/code/3-4-commercial-data-bias.py` — 상업 공간 데이터의 편향 진단: 포착률, 카드 매출의 결측 구조, 데이터 사용 판정표

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python practice/chapter3/code/3-0b-commercial-bias-data-prep.py   # 최초 1회 (원자료 준비는 data/raw/README.md 참조)
python practice/chapter3/code/3-1-stac-search.py
python practice/chapter3/code/3-2-cloud-formats.py
python practice/chapter3/code/3-3-cog-partial-read.py
python practice/chapter3/code/3-4-commercial-data-bias.py
```

예상 결과(검증 포인트)

- Sentinel 관련 컬렉션: `16개`
- 서울 지역 Sentinel-2 검색(2024년 6-8월, 구름 10% 미만): `5개` 영상, 평균 구름 비율 `6.8%`
- 동일 조건 Landsat C2 L2 검색: `6개` 영상
- 50,000개 포인트 저장 크기: GeoJSON `11.89 MB` / Shapefile `21.05 MB` / GeoParquet `2.18 MB`
- 읽기 속도: GeoParquet가 GeoJSON 대비 약 `13.6배`, Shapefile 대비 약 `9.3배` 빠름
- COG 부분 읽기: 전체 파일 `268.5 MB`(120,560,400픽셀) 중 강남 소구역 `58,320`픽셀(전체의 `0.048%`)만 `1.60초`에 읽음
- 편향 진단(`3-4`, 상가정보 2026-06 · OSM 2026-08 조회 · 추정매출 2025년 기준):
  - 8개 업종군 합산 OSM 포착률 `0.217`(넓은 대응). 업종군별로는 편의점 `0.713`, 카페 `0.178`, 화장품 `0.039`
  - 미용·뷰티는 좁은 대응에서 포착률 `1.249` — 1을 넘으므로 대응이 어긋났다는 신호입니다
  - 포착률 이항 GLM: 시청거리 계수 `−0.3075`(p = 4.95e−13), 잔차 Moran's I `0.261`(p = 0.001)
  - 점포가 실재하는 셀 `10,186개` 중 카드 매출이 한 분기도 없는 셀 `1,895개`(`18.6%`). MCAR 기각
  - 판정 등급: 그대로 사용 `1개` / 보정 후 사용 `249개` / 사용 금지 `159개`
  - 정확한 수치는 `results/3-4-commercial-data-bias.log`를 기준으로 봅니다

검증 팁

- `3-1`은 원격 카탈로그를 실시간으로 조회합니다. 카탈로그에 영상이 추가되면 검색 결과 개수가 위 값과 달라질 수 있습니다. 개수보다 **검색 조건(기간·구름 비율·영역)이 결과를 어떻게 좁히는지**를 확인하세요.
- `3-2`의 시간 값은 디스크와 CPU에 따라 달라집니다. 절대 시간이 아니라 **포맷 사이의 배수 관계**가 유지되는지 보세요. 파일 크기는 환경과 무관하게 거의 같게 나옵니다.
- `3-3`은 Planetary Computer의 서명된(SAS) COG URL에 접근하므로 인터넷 연결이 필요합니다. 전체 파일 크기·소요 시간은 그날 선택된 장면과 네트워크 상태에 따라 달라질 수 있습니다. 절대 숫자보다 **전체 파일 대비 실제로 받은 픽셀 비율이 매우 작다는 점**과 **소구역 읽기가 몇 초 안에 끝난다는 점**을 확인하세요. 받은 소구역 크롭은 `practice/chapter3/data/gangnam_b04_window.tif`(약 91 KB, 270×216픽셀)로 저장됩니다 — 전체 268.5 MB 원본 파일은 여전히 받지 않습니다.
- `3-3`은 마지막 단계에서 VSCode 등 일반 이미지 뷰어로 바로 볼 수 있는 PNG 미리보기도 2장 만듭니다: `gangnam_b04_raw_preview.png`(B04 단일 밴드, 대비 스트레치 없음)와 `gangnam_rgb_stretched.png`(B02·B03·B04 RGB 합성 + 2~98% percentile 스트레치). 두 파일을 나란히 열어 보면 단일 밴드 그레이스케일과 RGB 합성의 선명도·색상 차이를 직접 비교할 수 있습니다.
- `3-4`는 준비 스크립트가 만든 집계 파일을 읽으므로, 같은 파일에서는 값이 그대로 재현됩니다. 반대로 `3-0b`를 다시 돌리면 세 출처가 모두 갱신되어 값이 달라집니다 — 상가정보는 분기 갱신, 서울 추정매출은 분기 갱신, OSM은 실시간입니다. 본문 수치를 인용할 때는 `data/SOURCES_bias.json`의 기준일을 함께 보세요.
- `3-4`의 업종 대응표는 완전하지 않습니다. 세 출처의 업종 분류가 서로 일대일로 맞지 않기 때문입니다. 그래서 코드는 좁은 대응과 넓은 대응 두 판본을 함께 계산합니다. **절대 수보다 동 사이의 순서와 방향**을 보세요 — 대응을 바꾸면 포착률의 수준은 변하지만 순서는 대체로 유지됩니다.
- `3-0b`가 Overpass에서 504를 받으면 세 엔드포인트를 돌아가며 재시도합니다. 공개 서버가 혼잡한 시간대에는 몇 분씩 걸릴 수 있습니다. 한 번 받은 결과는 `data/raw/osm_seoul_poi_v2.parquet`에 남아 다음 실행에서 재사용됩니다.

결과 파일

- `practice/chapter3/results/3-1-stac-search.log` — 실행 로그
- `practice/chapter3/results/3-2-cloud-formats.log` — 실행 로그
- `practice/chapter3/results/3-3-cog-partial-read.log` — 실행 로그
- `practice/chapter3/results/3-4-commercial-data-bias.log` — 실행 로그
- `practice/chapter3/results/3-4-coverage-map.png` — 행정동별 포착률 지도와 시청 거리 산점도

데이터 출처와 라이선스

이 장은 자기가 다루는 주제를 스스로 지켜야 합니다. 3.5절이 "가장 제한적인 라이선스가 결과물에 적용된다"고 가르치는 장에서 원자료를 무단 재배포하면 앞뒤가 맞지 않습니다. 그래서 **원자료도 집계 파일도 저장소에 넣지 않고, 그것을 만드는 코드만 관리합니다.**

- 소상공인시장진흥공단 상가(상권)정보 — 공공누리. 직접 내려받아야 합니다(절차는 `data/raw/README.md`). 14장이 이미 받아 둔 폴더가 있으면 그대로 씁니다.
- 서울시 상권분석서비스 추정매출·영역(행정동) — 서울특별시·서울신용보증재단, 공공누리 제1유형. 준비 스크립트가 자동으로 받습니다.
- OpenStreetMap POI — © OpenStreetMap contributors, ODbL. Overpass API로 조회합니다.

자세한 내용은 `data/README.md`와 `data/raw/README.md`를 보세요.

연관 자료

- 교재: `docs/ch03.md` — 데이터 생태계·표준·거버넌스
- 강의: `lecture/chapter3.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `practice/chapter3/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
