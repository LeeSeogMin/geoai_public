# context.md — 프로젝트 컨텍스트

> 세션 시작 시 이 파일과 [AGENTS.md](AGENTS.md)를 먼저 읽는다. 실습 작업 시 [lecture_practice/AGENTS.md](lecture_practice/AGENTS.md)도 읽는다.

## 프로젝트 개요

- 2026-2학기 GeoAI 강의 공개 저장소. 강의 원고, 학생용 실습 코드, 과제 제출물을 관리한다.
- 규칙 정본은 `AGENTS.md`다. 이 파일에 규칙을 복제하지 않는다.

## 디렉토리 구조

| 경로 | 내용 |
|------|------|
| `lecture/` | 강의 원고 (chapter1~15, 부록). 학생이 수정하지 않음 |
| `lecture_practice/` | 장별 실습 코드(chapter1~15), 환경 점검(`check_env.py`), requirements |
| `docs/` | 배포용 PDF (ch01, 부록) |
| `scripts/` | 과제 제출 스크립트 (`submit.py`, `_seeded_run.py`) |
| `submissions/` | 학생 과제 제출물 |

## 현재 상태 (2026-11-16 기준)

- 강의 원고는 13장 체제다: `ch01~07`, `ch09~14`(8주차 중간고사로 8번 없음). 환경·기후, 복지·보건, 미래 GeoAI는 선택 읽기 보론(`bonus-*.md`)이다
- PDF는 `ch05`를 뺀 12개 장에 있다
- 12~14장은 가상 사례를 따라가며 문제를 푸는 방식으로 다시 썼다. 그림은 `lecture/graphics/make_ch12_figures.py` 등 장별 스크립트로 만든다
- 실습 결과 폴더(`lecture_practice/**/results/`)는 png만 git에 올리고 로그·CSV는 올리지 않는다. 원고가 이 폴더의 그림을 직접 링크하기 때문이다
- 과제 제출 파이프라인 동작 중(`scripts/submit.py`)

## 작업 시 주의

- `lecture_practice/data/`는 저장소에 포함하지 않는다. 데이터는 장별 준비 코드로 생성한다.
- `lecture_practice/results/` 기존 결과는 덮어쓰기 전에 사용자에게 알린다.
- 경로는 `pathlib.Path` + 상대 경로, 절대 경로·API 키 금지.

## 다음 작업

[todo.md](todo.md) 참조.
