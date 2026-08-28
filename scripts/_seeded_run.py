"""
학번 시드를 씌워 실습 스크립트를 실행하는 런처 (내부용)
=======================================================
`scripts/submit.py`가 이 파일을 통해 실습 코드를 실행한다. 학생이 직접 부를 일은
없다.

하는 일은 셋뿐이다.

1. `lecture_practice/student.py`의 `apply_student_seed()`로 난수 시드에 학번 오프셋을 씌운다.
2. 실습 파일이 원래대로(`python 11-1-....py`) 실행된 것처럼 환경을 맞춘다 —
   `sys.path[0]`을 그 파일이 있는 폴더로 두어 `from _s2_data import ...` 같은
   같은 폴더 임포트가 그대로 동작하게 한다.
3. `runpy`로 실행한다. 실습 파일은 한 줄도 고치지 않는다.

사용법:
    python scripts/_seeded_run.py <실습파일.py>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: python scripts/_seeded_run.py <실습파일.py>", file=sys.stderr)
        return 2

    target = Path(sys.argv[1]).resolve()
    if not target.exists():
        print(f"파일을 찾을 수 없습니다: {target}", file=sys.stderr)
        return 2

    # lecture_practice/student.py 를 임포트할 수 있게 한다
    sys.path.insert(0, str(PROJECT_ROOT / "lecture_practice"))
    from student import apply_student_seed

    source = target.read_text(encoding="utf-8")
    offset = apply_student_seed(patch_torch=("import torch" in source))
    if offset:
        print(f"[학번 시드] 난수 오프셋 {offset} 적용 — 이 결과는 제출자 고유값이다.\n",
              flush=True)

    # 실습 파일이 직접 실행됐을 때와 같은 임포트 환경을 만든다
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target)] + sys.argv[2:]

    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
