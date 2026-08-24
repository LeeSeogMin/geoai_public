"""
AI 에이전트 규칙 파일 배치
==========================
규칙 원본은 `lecture_practice/AGENTS.md` 하나다. 그런데 도구마다 읽는 파일 이름이 달라서,
저장소 루트에 도구별 파일을 만들어 줘야 같은 규칙이 모두에게 적용된다.

이 스크립트가 만드는 것 (저장소 루트):

    AGENTS.md                        <- lecture_practice/AGENTS.md 복사본.
                                        Codex, Antigravity, Copilot이 직접 읽는다.
    CLAUDE.md                        <- 첫 줄에 @AGENTS.md 임포트.
                                        Claude Code는 AGENTS.md를 직접 읽지 않는다.
    .github/copilot-instructions.md  <- Copilot이 확실히 잡도록 두는 안전장치.
    context.md, todo.md              <- 학생이 채워 쓰는 작업 기록 서식.

사용법:

    python lecture_practice/agent-setup/install.py            # 없는 파일만 만든다
    python lecture_practice/agent-setup/install.py --force    # 있어도 덮어쓴다

이미 있는 파일은 기본적으로 건드리지 않는다. 직접 쓴 context.md, todo.md가
날아가지 않게 하기 위해서다.
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRACTICE_DIR = HERE.parent
ROOT = PRACTICE_DIR.parent
SOURCE_RULES = PRACTICE_DIR / "AGENTS.md"

CLAUDE_MD = """@AGENTS.md

<!--
Claude Code는 AGENTS.md를 직접 읽지 않는다. 위 한 줄이 AGENTS.md를 그대로
불러오므로, 규칙을 고칠 때는 lecture_practice/AGENTS.md만 고치면 된다.
이 파일은 lecture_practice/agent-setup/install.py가 만든다. 직접 고치지 않는다.
-->
"""

COPILOT_MD = """# GitHub Copilot 지시사항 (GeoAI 실습)

이 저장소의 규칙 원본은 저장소 루트의 `AGENTS.md`다. **작업을 시작하기 전에 `AGENTS.md`를
먼저 읽고 그 규칙을 따른다.** 아래는 특히 자주 어기는 항목만 추린 것이다.

- 코드를 돌리지 않은 채 실행 결과나 숫자를 지어내지 않는다. 실행은 학생에게 요청하고, 나온 출력을 해석한다.
- 답변은 한국어로 한다. 코드를 내놓기 전에 무엇을 왜 하는지 먼저 설명한다.
- Windows와 macOS 양쪽에서 도는 코드를 쓴다. 경로는 `pathlib.Path`, 텍스트 파일은 `encoding="utf-8"`,
  엑셀로 열 CSV는 `encoding="utf-8-sig"`를 쓴다. 화면에 찍는 문자열에 `→`, `—`, 이모지를 넣지 않는다.
- 실행은 프로젝트 루트의 가상환경 `.venv`에서 한다.
- 세션을 시작할 때 `context.md`와 `todo.md`를 읽고, 작업이 끝나면 갱신한다.

<!-- lecture_practice/agent-setup/install.py가 만든 파일이다. 규칙 수정은 lecture_practice/AGENTS.md에서 한다. -->
"""


def place(target: Path, content: str, force: bool, made, kept):
    """파일을 만든다. 이미 있으면 건드리지 않는다(--force 제외)."""
    rel = target.relative_to(ROOT)
    if target.exists() and not force:
        kept.append(str(rel))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    made.append(str(rel))


def main():
    parser = argparse.ArgumentParser(description="에이전트 규칙 파일을 저장소 루트에 배치한다")
    parser.add_argument("--force", action="store_true", help="이미 있는 파일도 덮어쓴다")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("AI 에이전트 규칙 파일 배치")
    print("=" * 62)
    print(f"  저장소 루트: {ROOT}")

    if not SOURCE_RULES.exists():
        print(f"  [오류] 규칙 원본을 찾지 못했다: {SOURCE_RULES}")
        return 1

    made, kept = [], []

    place(ROOT / "AGENTS.md", SOURCE_RULES.read_text(encoding="utf-8"), args.force, made, kept)
    place(ROOT / "CLAUDE.md", CLAUDE_MD, args.force, made, kept)
    place(ROOT / ".github" / "copilot-instructions.md", COPILOT_MD, args.force, made, kept)

    for name in ("context.md", "todo.md"):
        template = HERE / "templates" / name
        if template.exists():
            place(ROOT / name, template.read_text(encoding="utf-8"), args.force, made, kept)

    print()
    if made:
        print("  만든 파일:")
        for f in made:
            print(f"    + {f}")
    if kept:
        print("  이미 있어서 그대로 둔 파일:")
        for f in kept:
            print(f"    = {f}")
        print("    (덮어쓰려면 --force 를 붙인다)")

    print()
    print("  규칙을 고칠 때는 lecture_practice/AGENTS.md 를 고친 뒤 이 스크립트를 --force 로 다시 실행한다.")
    print("  (context.md 와 todo.md 는 직접 쓴 내용이 날아가므로 --force 를 조심해서 쓴다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
