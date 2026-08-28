"""
과제 제출 파일 만들기
==============================
명령 한 줄로 그 장의 실습을 돌리고, LMS에 올릴 **파일 하나**를 만든다.

    python scripts/submit.py 11 --id 201912345 --name 홍길동

이 명령이 대신 해 주는 일:

1. 그 장의 **데이터 준비 스크립트를 먼저 실행**한다. 저장소를 내려받으면 데이터
   파일은 들어 있지 않으므로(용량·저작권 때문에 제외돼 있다) 이 단계가 없으면
   실습이 돌지 않는다.
2. 그 장의 **대표 실습 하나**를 학번 시드로 실행한다. 학번마다 난수가 달라지므로
   결과 숫자도 실행자마다 다르다.
3. 실행 기록과 로그를 붙인 **제출 파일 한 개**를 만든다.

실행자가 할 일은 만들어진 파일을 열어 맨 아래 빈칸 세 개를 채우고 그 파일을
업로드하는 것뿐이다.

실행이 실패해도 제출 파일은 만들어진다. 그때는 오류 메시지가 자동으로 담기고,
질문이 "무엇을 얻었나" 대신 "어디서 막혔나"로 바뀐다. **실패해도 제출한다.**

제출 산출물(로그·증거·제출 파일)은 `submissions/` 아래에 쌓인다.

⚠ 다만 실습 스크립트 자신이 `lecture_practice/chapter*/data/`와 `results/`에 CSV·그림을
직접 쓴다. 학번 시드로 돌리면 그 파일들도 실행자 기준으로 바뀐다. 실행자 컴퓨터에서는
문제가 되지 않지만, **검토하는 쪽에서 돌렸다면**
`python scripts/run_and_capture.py <장번호>` 로 기준 산출물을 되돌린다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRACTICE_DIR = PROJECT_ROOT / "lecture_practice"
SEEDED_RUN = Path(__file__).resolve().parent / "_seeded_run.py"
LOG_LINES_IN_SUBMISSION = 150

# ---------------------------------------------------------------------------
# 장별 과제 정의
#   prep   : 대표 실습보다 먼저 돌려야 하는 준비 스크립트 ("장번호/파일명")
#   target : 대표 실습 (제출 대상)
#   ask    : 3번 문항 — 이 장에서 내려야 하는 결정
# ---------------------------------------------------------------------------
ASSIGNMENTS: dict[int, dict] = {
    1: {
        "title": "공간 데이터를 읽는 습관",
        "prep": [],
        "target": "1/1-1-geoai-tools-preview.py",
        "watch": "국가 수, 좌표계, 기하 유형, 그리고 마지막에 뜨는 경고",
        "ask": "이 결과는 컴퓨터가 달라도 똑같이 나와야 한다. 왜 그런가? "
               "만약 달랐다면 무엇을 먼저 의심하겠는가?",
        "fixed_seed": True,
    },
    2: {
        "title": "배달 권역 — 원형 반경과 도로망 도달권",
        "prep": ["2/2-0b-prepare-osm-snapshot.py"],
        "target": "2/2-3-delivery-service-area.py",
        "watch": "우회비(주행거리 ÷ 직선거리) 중앙값과 지점별 차이",
        "ask": "매장 중심 반경 3km로 배달 권역을 그은 가게가 있다. "
               "이 결과를 근거로 권역을 어떻게 바꾸라고 하겠는가?",
    },
    3: {
        "title": "이 데이터를 어디까지 믿을 것인가",
        "prep": ["3/3-0b-commercial-bias-data-prep.py"],
        "target": "3/3-4-commercial-data-bias.py",
        "watch": "업종별·동별 포착률, 그리고 1을 넘는 값이 있는지",
        "fixed_seed": True,
        "ask": "지도에 점포가 하나도 없는 동네가 있다. "
               "'여기는 비어 있으니 창업하자'고 말해도 되는가? 근거를 숫자로 대라.",
    },
    4: {
        "title": "공간 교차검증 — 시험이 쉬웠던 이유",
        "prep": ["4/4-0-simdata-prep.py"],
        "target": "4/4-2-spatial-cv.py",
        "watch": "무작위 교차검증 점수와 공간 블록 교차검증 점수의 차이",
        "ask": "무작위 교차검증 점수를 그대로 보고서에 실었다면, "
               "그 보고서는 무엇을 잘못 말하게 되는가?",
    },
    5: {
        "title": "이 과업에 딥러닝이 필요한가",
        "prep": ["4/4-0-data-download.py"],
        "target": "5/5-2-baseline-vs-cnn.py",
        "watch": "값싼 기준선과 CNN의 정확도 차이, 그리고 그 차이의 흔들림 폭",
        "ask": "이 정확도 차이만 보고 딥러닝 도입을 결정할 수 있는가? "
               "결정을 뒤집을 수 있는 값을 하나 들고, 왜 그것이 정확도보다 센지 말하라.",
    },
    6: {
        "title": "마스크에서 결정으로 — 후보지 걸러 내기",
        "prep": ["6/6-0b-site-simdata-prep.py"],
        "target": "6/6-2-site-sourcing.py",
        "watch": "면적 임계별 후보 수와 오탈락 수",
        "ask": "면적 요건이 1,000㎡다. 임계를 그대로 1,000㎡에 두겠는가, "
               "낮추겠는가? 그 판단을 정하는 것은 무엇인가?",
    },
    7: {
        "title": "구간에서 결정으로 — 몇 개를 발주할 것인가",
        "prep": ["7/7-0b-demand-simdata.py"],
        "target": "7/7-3-demand-newsvendor.py",
        "watch": "두 품목의 임계비, 그리고 예측값과 실제 발주량의 위아래 관계",
        "ask": "점장에게 '내일 90% 확률로 70~130개'라고만 전하면 무엇이 남는가? "
               "숫자 하나를 정해 준다면 몇 개이고, 그 근거는 무엇인가?",
    },
    8: {
        "title": "그 사업, 정말 효과가 있었나",
        "prep": ["8/8-0-simdata-prep.py"],
        "target": "8/8-1-nightlight-did-dml.py",
        "watch": "단순 비교 추정치와 이중차분·이중강건 추정치의 격차",
        "ask": "단순 비교 값을 사업 성과로 보고하면 무엇이 부풀려지는가? "
               "추정치와 함께 반드시 적어야 할 것을 하나 들어라.",
    },
    9: {
        "title": "어디를 검사할 것인가",
        "prep": ["9/9-0-simdata-prep.py"],
        "target": "9/9-3-supply-chain-due-diligence.py",
        "watch": "비용비에 따라 달라지는 컷오프와 실사 대상 구역 수",
        "ask": "데이터도 예측도 그대로인데 검사 대상 수가 크게 움직인다. "
               "무엇이 그것을 움직이는가? 당신이라면 몇 곳을 검사하겠는가?",
    },
    10: {
        "title": "경계선의 마법 — 규제가 집값을 낮췄나",
        "prep": ["10/10-0-simdata-prep.py"],
        "target": "10/10-3-rdd-regulation-boundary.py",
        "watch": "경계에서의 낙차 추정치와 대역폭을 바꿨을 때의 변화",
        "ask": "이 낙차를 '규제의 효과'라고 부를 때, "
               "그 말이 적용되는 범위는 어디까지인가? 전국에 일반화할 수 있는가?",
    },
    11: {
        "title": "우선순위표를 그대로 집행할 것인가",
        "prep": ["11/11-0-simdata-prep.py"],
        "target": "11/11-1-flood-risk-priority.py",
        "watch": "상위 순위 격자의 예측값과 그 예측의 신뢰등급",
        "ask": "1순위 지역의 예측이 가장 불확실하게 나왔다면, "
               "그 지역에 대한 처방은 '즉시 집행'인가 다른 무엇인가?",
    },
    12: {
        "title": "부족한 곳을 찾은 다음의 질문",
        "prep": ["12/12-0b-vetcare-data-prep.py"],
        "target": "12/12-5-unmet-demand-siting.py",
        "watch": "미충족 수요 상위 목록과 순차 선택 결과가 어긋나는 지점",
        "fixed_seed": True,
        "ask": "'가장 부족한 곳 10군데'와 '순서대로 고른 10군데'가 다르다. "
               "예산을 어느 쪽에 쓰겠는가? 왜 두 목록이 갈리는가?",
    },
    13: {
        "title": "남을 것인가 철수할 것인가",
        "prep": ["13/13-0b-trade-area-data-prep.py"],
        "target": "13/13-4-trade-area-exit-decision.py",
        "watch": "상권 유형별 단위 경제와 철수 판정이 갈리는 경계",
        "ask": "철수로 판정된 지역 하나를 골라라. "
               "그 판정을 뒤집으려면 어떤 값이 얼마나 달라져야 하는가?",
    },
    14: {
        "title": "어디에 낼 것인가 — 손님을 나눠 갖는 계산",
        "prep": ["14/14-0-data-prep.py"],
        "target": "14/14-1-huff-location-cannibalization.py",
        "watch": "후보지별 배후 수요와 기존 점포에서 옮겨 오는 몫(전이율)",
        "fixed_seed": True,
        "ask": "순증이 가장 큰 후보가 곧 최선인가? "
               "이 계산에 학습 모델이 하나도 쓰이지 않았다는 사실은 무엇을 말하는가?",
    },
    15: {
        "title": "통계가 통과시킨 것을 산수가 잡는다",
        "prep": ["15/15-0-simdata-prep.py"],
        "target": "15/15-2-market-synthetic-control.py",
        "watch": "합성통제 추정치와 참값의 격차, 그리고 통계 진단을 통과했는지",
        "ask": "이 추정치는 통계 진단을 통과했다. "
               "그대로 결정에 넣겠는가, 한 칸을 더 두겠는가? 그 한 칸은 무엇인가?",
    },
}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_code_file(spec: str) -> Path:
    """'11/11-0-simdata-prep.py' → 실제 경로. chapter11 / chapter011 둘 다 지원."""
    chapter, filename = spec.split("/", 1)
    for name in (f"chapter{chapter}", f"chapter{int(chapter):02d}"):
        candidate = PRACTICE_DIR / name / "code" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"실습 파일을 찾을 수 없습니다: {spec}\n"
        f"  저장소를 통째로 내려받았는지 확인하세요."
    )


def run_one(py_file: Path, student_id: str, timeout: int) -> dict:
    """실습 파일 하나를 학번 시드로 실행하고 결과를 딕셔너리로 돌려준다."""
    env = dict(os.environ)
    env["GEOAI_STUDENT_ID"] = student_id
    # 윈도우에서 한글 출력이 깨지지 않도록 자식 프로세스 인코딩을 고정한다
    env["PYTHONIOENCODING"] = "utf-8"

    started = time.perf_counter()
    started_iso = datetime.now(timezone.utc).astimezone().isoformat()
    try:
        proc = subprocess.run(
            [sys.executable, str(SEEDED_RUN), py_file.name],
            cwd=py_file.parent,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        exit_code, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        stderr += f"\n[시간 초과] {timeout}초를 넘겨 중단했습니다."
        exit_code, timed_out = None, True

    duration = round(time.perf_counter() - started, 2)
    combined = (
        f"$ python {py_file.name}\n\n"
        f"=== 표준 출력 ===\n{stdout}\n"
        f"=== 오류 출력 ===\n{stderr}"
    )
    return {
        "file": py_file.name,
        "relpath": str(py_file.relative_to(PROJECT_ROOT)),
        "started_at": started_iso,
        "duration_sec": duration,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": exit_code == 0,
        "stdout": stdout,
        "stderr": stderr,
        "combined": combined,
        "source_sha256": sha256_of_text(py_file.read_text(encoding="utf-8")),
        "output_sha256": sha256_of_text(combined),
    }


def tail_lines(text: str, limit: int) -> tuple[str, bool]:
    lines = text.rstrip("\n").split("\n")
    if len(lines) <= limit:
        return "\n".join(lines), False
    return "\n".join(lines[-limit:]), True


# ---------------------------------------------------------------------------
# 제출 파일 만들기
# ---------------------------------------------------------------------------
def build_submission(chapter: int, spec: dict, student_id: str, name: str,
                     offset: int, prep_runs: list[dict], main_run: dict) -> str:
    ok = main_run["success"]
    body, truncated = tail_lines(main_run["stdout"] or "(출력 없음)",
                                 LOG_LINES_IN_SUBMISSION)

    prep_lines = []
    for r in prep_runs:
        mark = "성공" if r["success"] else f"실패(종료코드 {r['exit_code']})"
        prep_lines.append(f"- {r['file']} — {mark}, {r['duration_sec']}초")
    prep_block = "\n".join(prep_lines) if prep_lines else "- (준비 스크립트 없음)"

    seed_note = (
        "고정 시드(이 장은 난수를 쓰지 않는다. 교재와 같은 값이 나와야 정상이다)"
        if spec.get("fixed_seed") else f"학번 시드 오프셋 {offset}"
    )

    head = f"""# {chapter}장 과제 — {spec['title']}

- **제출자**: {name or "(이름을 적으세요)"} / 학번 {student_id}
- **제출일**: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}

---

## 1. 자동 기록

| 항목 | 값 |
|---|---|
| 대표 실습 | `{main_run['relpath']}` |
| 실행 결과 | {"성공" if ok else f"실패 (종료코드 {main_run['exit_code']})"} |
| 소요 시간 | {main_run['duration_sec']}초 |
| 난수 | {seed_note} |
| 실행 환경 | Python {sys.version.split()[0]} / {platform.platform()} |
| 코드 해시 | `{main_run['source_sha256'][:16]}…` |
| 출력 해시 | `{main_run['output_sha256'][:16]}…` |

**준비 스크립트**

{prep_block}

---

## 2. 실행 결과 — 자동으로 붙습니다
"""

    if ok:
        note = "  (앞부분은 줄여 붙였습니다)" if truncated else ""
        result_block = f"""
아래에서 볼 것: **{spec['watch']}**{note}

```
{body}
```
"""
        # 경고는 표준 출력이 아니라 오류 출력으로 나온다. 실행에 성공했더라도
        # 경고가 있으면 함께 붙인다 — 경고를 읽는 것이 이 수업의 훈련이다.
        warn, _ = tail_lines(main_run["stderr"].strip(), 40)
        if warn:
            result_block += f"""
**실행 중 나온 경고** — 오류가 아니라서 코드는 끝까지 돌았습니다.
그래도 결과가 맞다는 뜻은 아닙니다.

```
{warn}
```
"""
        answer_block = f"""
---

## 3. 여기만 채우세요

**① 눈에 띈 숫자 하나를 그대로 옮겨 적고, 왜 눈에 띄었는지 한 문장으로.**

→

**② 그 숫자가 왜 그렇게 나왔는가? (한두 문장)**

→

**③ {spec['ask']} (한두 문장)**

→

---

_③번이 이 과제의 핵심입니다. ①②는 로그를 보면 답이 나오지만 ③은 그렇지 않습니다._
"""
    else:
        err_block, _ = tail_lines(main_run["stderr"] or "(오류 출력 없음)", 60)
        result_block = f"""
실행이 끝까지 가지 못했습니다. **그래도 이 파일을 그대로 제출하세요.**
어디서 막혔는지가 이 과제의 절반입니다.

```
{err_block}
```
"""
        answer_block = """
---

## 3. 여기만 채우세요 (실행이 실패한 경우)

**① 오류 메시지에서 문제의 원인으로 보이는 줄을 하나 골라 옮겨 적으세요.**

→

**② 그 줄이 무슨 뜻이라고 이해했습니까? 무엇을 시도해 봤습니까?**

→

**③ 다음에 무엇을 시도하겠습니까? 어디에 도움을 요청하겠습니까?**

→

---

_막힌 과정을 적은 제출도 온전한 제출입니다. 실행 성공 여부로 점수를 매기지 않습니다._
"""

    return head + result_block + answer_block


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="과제 제출 파일을 만든다 (실습 실행 + 제출용 파일 1개 생성)",
    )
    parser.add_argument("chapter", type=int, help="장 번호 (1~15)")
    parser.add_argument("--id", required=True, help="학번")
    parser.add_argument("--name", default="", help="이름 (선택)")
    parser.add_argument("--timeout", type=int, default=900, help="스크립트별 제한 시간(초)")
    parser.add_argument("--dry-run", action="store_true",
                        help="무엇을 실행할지 보기만 한다")
    args = parser.parse_args()

    spec = ASSIGNMENTS.get(args.chapter)
    if spec is None:
        print(f"[오류] {args.chapter}장 과제가 정의돼 있지 않습니다. "
              f"가능한 장: {sorted(ASSIGNMENTS)}", file=sys.stderr)
        return 2

    student_id = args.id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,20}", student_id):
        print("[오류] --id 는 영문·숫자 4~20자로 적어 주세요 (예: --id 201912345)",
              file=sys.stderr)
        return 2

    try:
        prep_files = [resolve_code_file(s) for s in spec["prep"]]
        target_file = resolve_code_file(spec["target"])
    except FileNotFoundError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PRACTICE_DIR))
    from student import seed_offset
    offset = 0 if spec.get("fixed_seed") else seed_offset(student_id)

    print("=" * 62)
    print(f" {args.chapter}장 과제 — {spec['title']}")
    print("=" * 62)
    print(f" 학번 {student_id}" + (f" / {args.name}" if args.name else ""))
    print(f" 준비 {len(prep_files)}개 → 대표 실습 {target_file.name}")
    if args.dry_run:
        for p in prep_files:
            print(f"   준비  {p.relative_to(PROJECT_ROOT)}")
        print(f"   대표  {target_file.relative_to(PROJECT_ROOT)}")
        return 0
    print(" 처음 실행하는 장은 데이터를 만드느라 조금 더 걸립니다.\n")

    # 준비 스크립트는 학번 시드를 그대로 받아야 한다(데이터가 실행자마다 달라진다)
    seed_for_run = student_id if not spec.get("fixed_seed") else ""

    prep_runs = []
    for p in prep_files:
        print(f"▶ 준비: {p.name} …", end=" ", flush=True)
        r = run_one(p, seed_for_run, args.timeout)
        prep_runs.append(r)
        print("성공" if r["success"] else f"실패(종료코드 {r['exit_code']})",
              f"{r['duration_sec']}초")
        if not r["success"]:
            print("  준비 단계에서 막혔습니다. 그래도 제출 파일은 만들어집니다.")

    print(f"▶ 실습: {target_file.name} …", end=" ", flush=True)
    main_run = run_one(target_file, seed_for_run, args.timeout)
    print("성공" if main_run["success"] else f"실패(종료코드 {main_run['exit_code']})",
          f"{main_run['duration_sec']}초")

    # 산출물 저장 — 교재의 기준 결과(lecture_practice/*/results/)는 건드리지 않는다
    out_dir = PROJECT_ROOT / "submissions" / student_id / f"chapter{args.chapter}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in prep_runs + [main_run]:
        (out_dir / f"{Path(r['file']).stem}.log").write_text(
            r["combined"], encoding="utf-8")
    evidence = {
        "student_id": student_id,
        "student_name": args.name,
        "chapter": args.chapter,
        "seed_offset": offset,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "prep": [{k: v for k, v in r.items()
                  if k not in ("stdout", "stderr", "combined")} for r in prep_runs],
        "target": {k: v for k, v in main_run.items()
                   if k not in ("stdout", "stderr", "combined")},
    }
    (out_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    submission = build_submission(args.chapter, spec, student_id, args.name,
                                  offset, prep_runs, main_run)
    out_path = PROJECT_ROOT / "submissions" / f"제출_{args.chapter}장_{student_id}.md"
    out_path.write_text(submission, encoding="utf-8")

    print("\n" + "-" * 62)
    print(f"제출 파일을 만들었습니다:\n  {out_path.relative_to(PROJECT_ROOT)}")
    print("\n다음 순서로 하세요.")
    print("  1) 위 파일을 열어 맨 아래 '3. 여기만 채우세요'의 빈칸 세 개를 채운다")
    print("  2) 그 파일 하나를 수업 게시판(LMS)에 올린다")
    print("-" * 62)
    if not spec.get("fixed_seed"):
        chapter_dir = target_file.parent.parent
        print(f"\n[참고] 이번 실행으로 {chapter_dir.name}/ 아래 data/ 와 results/ 의 산출물이")
        print("       학번 시드 기준으로 다시 만들어졌습니다(제출 로그는 submissions/에 따로")
        print("       보관되므로 안전합니다). 교재의 기준 수치로 되돌리려면")
        print(f"       python scripts/run_and_capture.py {args.chapter}  를 돌리세요.")
    return 0 if main_run["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
