# 2장 실습: 좌표계 변환, 공간 연산과 서비스 권역 계산

이 실습은 `docs/ch02.md`와 `lecture/chapter2.md`의 내용을 코드로 확인하는 목적입니다. 실습은 로컬에서 실행되며, 결과 로그는 `lecture_practice/chapter2/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `lecture_practice/README.md`의 설치 지침을 먼저 따르세요.
- 세 번째 실습(`2-3`)은 `networkx`가, 준비 스크립트(`2-0b`)는 `osmnx`가 필요합니다. 두 패키지는 루트 `lecture_practice/requirements.txt`에 들어 있으므로, 예전에 환경을 만들었다면 한 번 다시 설치하세요.

```bash
pip install -r lecture_practice/requirements.txt
```

실습 파일

- `lecture_practice/chapter2/code/2-1-coordinate-transform.py` — 경위도 ↔ UTM 변환 예제
- `lecture_practice/chapter2/code/2-2-spatial-operations.py` — 버퍼·오버레이·면적 계산 예제
- `lecture_practice/chapter2/code/2-3-delivery-service-area.py` — 배달 권역: 원형 반경과 도로망 도달권 비교

준비 스크립트(평소에는 실행하지 않습니다)

- `lecture_practice/chapter2/code/2-0b-prepare-osm-snapshot.py` — OpenStreetMap에서 도로망·건물을 내려받아 `data/`에 스냅샷으로 저장합니다. **스냅샷이 이미 저장소에 들어 있으므로 실행할 필요가 없습니다.** 그냥 실행하면 스냅샷 요약만 출력하고 끝나고, `--refresh`를 붙여야 다시 내려받습니다. 다시 받으면 OSM이 그동안 갱신된 만큼 아래 기대값과 어긋날 수 있습니다.

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python lecture_practice/chapter2/code/2-1-coordinate-transform.py
python lecture_practice/chapter2/code/2-2-spatial-operations.py
python lecture_practice/chapter2/code/2-3-delivery-service-area.py
```

예상 결과(검증 포인트)

`2-1`, `2-2`

- 서울 UTM 좌표: 약 `(323,210.47m E, 4,152,220.15m N)`
- 서울-부산 거리: UTM ≈ `321.13km`, Haversine ≈ `321.45km`
- 서울 500km 버퍼 면적: 약 `784,137 km²`
- 한국 면적(등면적 투영): 약 `99,206 km²`

`2-3` (직선 3km 원 안 건물 수 대비)

| 지점 | 직선 3km(B) | 도로망 3km(C) | B∖C 비율 | 우회비 중앙값 |
| --- | ---: | ---: | ---: | ---: |
| 옥수 | 10,343채 | 2,418채 | 76.6% | 1.493 |
| 신림 | 11,699채 | 9,233채 | 21.2% | 1.270 |
| 화곡 | 11,755채 | 8,379채 | 28.7% | 1.327 |

- 다섯 권역의 면적: A1 `22.33km²`, A2 `17.70km²`, B `28.23km²` (세 지점 공통, 소수 둘째 자리)
- 매장 위치 재배치 200회 B∖C 중앙값: 옥수 `58.8%`, 신림 `45.5%`, 화곡 `25.8%`
- 완전 격자 도로망의 이론 우회비 `4/π = 1.273`이 출력에 함께 찍힙니다

결과 파일

- `results/2-1-coordinate-transform.log`, `results/2-2-spatial-operations.log` — 실행 로그
- `results/2-3-delivery-service-area.log` — 실행 로그 (본문 표 2.13~2.16의 모든 수치 출처)
- `results/2-3-service-areas.png` — 세 지점의 권역 지도
- `results/2-3-zone-summary.csv`, `results/2-3-misassignment.csv` — 표로 다시 쓴 결과
- `results/2-0b-prepare-osm-snapshot.log` — 스냅샷 요약

검증 팁

- `2-1`, `2-2`의 출력이 조금 다르면 좌표계나 라이브러리 버전에 의한 소수점 차이일 수 있습니다. 큰 오차가 나면 `crs` 확인(`EPSG:4326` vs `EPSG:32652`)과 `to_crs()` 적용 여부를 점검하세요.
- `2-3`은 저장소에 커밋된 스냅샷을 읽으므로 값이 그대로 재현됩니다. 값이 달라졌다면 누군가 `2-0b`를 `--refresh`로 다시 돌렸다는 뜻입니다(`data/osm_snapshot_meta.json`의 취득일과 파일 해시를 확인하세요).
- `2-3` 출력 맨 위에 `[경고] A1을 만들 때 GeoPandas가 알려 준 말: Geometry is in a geographic CRS…`가 세 번 찍힙니다. 이것은 버그가 아니라 **일부러 붙잡아 출력한 것**입니다. 경위도 좌표계에서 버퍼를 만드는 실수를 GeoPandas가 어떻게 알려 주는지 눈으로 보라는 뜻입니다.
- `results/2-3-service-areas.png`를 열어 옥수 패널의 빨간 점(직선 3km 안 · 주행 3km 밖 건물)이 한강 남쪽에 띠를 이루는지 확인하세요. 숫자만 보고 넘어가면 이 예제의 요점을 놓칩니다.
- 권역 **면적**(C·D)은 도달 노드를 감싸는 오목 껍질의 `ratio`에 따라 움직입니다. 출력의 "오목 껍질 ratio 민감도" 줄이 그 폭을 보여 줍니다. 반면 건물 수·우회비·컷라인은 건물마다 직접 잰 거리로 세므로 이 파라미터와 무관합니다.

데이터

- `data/` 폴더의 출처·라이선스·재취득 방법은 `data/README.md`를 보세요. OpenStreetMap 자료이며 사용 시 `© OpenStreetMap contributors` 표기가 필요합니다.

연관 자료

- 교재: `docs/ch02.md` — 좌표계·투영·전처리 이론과 배달 권역 분석 예제
- 강의: `lecture/chapter2.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `lecture_practice/chapter2/results/*.evidence.json`를 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
