# GeoAI 공개 저장소 에이전트 규칙

이 파일이 공개 저장소의 공통 정본이다. Claude, Codex, GitHub Copilot을 사용할 때 먼저 이 파일을 읽는다.

## 학생 지원 원칙

- 답변은 한국어로 작성한다.
- 실행하지 않은 결과·수치·표를 만들지 않는다. 데이터가 없으면 임의의 데이터를 만들지 말고 `lecture_public`의 데이터 준비 코드를 안내한다.
- 설명은 “무엇을 왜 하는가 → 실행 방법 → 성공 여부 확인” 순서로 제시한다.
- 한 번에 여러 설정을 바꾸지 않는다. 하나를 바꾸고 확인한 뒤 다음 단계로 간다.
- 오류를 숨기거나 예외 처리로 덮지 않는다. 원인을 확인하고 재현 가능한 해결 방법을 제시한다.

## 저장소 범위

- `lecture/`: 강의 원고다. 학생이 임의로 수정하지 않는다.
- `lecture_public/`: 학생용 실습 코드와 실행 결과다.
- `lecture_public/data/`는 공개 저장소에 포함하지 않는다. 필요한 데이터는 장별 데이터 준비 코드로 생성한다.
- `lecture_public/results/`의 기존 실행 결과는 근거 자료이므로 덮어쓰기 전에 사용자에게 알린다.

## 환경·코드 규칙

- Windows와 macOS 모두에서 실행되도록 `pathlib.Path`와 상대 경로를 사용한다.
- 절대 경로, API 키, 토큰, 비밀번호를 코드와 문서에 넣지 않는다.
- Python 파일을 수정할 때 UTF-8 인코딩과 운영체제별 가상환경 명령을 고려한다.
- 패키지는 `lecture_public/requirements-student.txt` 또는 장별 `requirements.txt`를 우선 사용한다.
- 환경 문제는 먼저 `python lecture_public/check_env.py`로 확인한다.

상세한 GeoAI 실습 규칙은 `lecture_public/AGENTS.md`를 추가로 읽는다.
