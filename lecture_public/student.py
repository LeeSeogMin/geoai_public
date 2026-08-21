"""
학번 시드 (Student Seed)
========================
같은 실습을 학생마다 **다른 숫자**로 끝나게 만드는 장치다.

왜 필요한가. 실습 코드는 재현성을 위해 난수 시드를 42 같은 고정값으로 박아 둔다.
그래서 누가 돌리든 결과가 똑같고, 한 사람의 제출물이 그대로 돌아다녀도 구별할 수
없다. 이 모듈은 환경변수 `GEOAI_STUDENT_ID`가 설정돼 있을 때만 모든 난수 시드를
**학번에서 계산한 오프셋만큼 옮긴다.**

핵심 성질 세 가지:

1. **실습 파일을 하나도 고치지 않는다.** `np.random.default_rng(42)` 같은 호출을
   실행 시점에 가로채 `default_rng(42 + 오프셋)`으로 바꾼다.
2. **환경변수가 없으면 아무 일도 하지 않는다.** 따라서 교재에 인용된 수치는
   `scripts/run_and_capture.py`로 그대로 재현된다. 집필 쪽 재현성이 깨지지 않는다.
3. **시드 스트림의 독립성이 보존된다.** 같은 파일 안에서 42와 2024를 따로 쓰던
   코드는 오프셋을 더한 뒤에도 여전히 서로 다른 스트림을 쓴다.

사용법 (보통은 `scripts/submit.py`가 대신 호출한다):

    import os
    os.environ["GEOAI_STUDENT_ID"] = "201912345"
    from student import apply_student_seed
    apply_student_seed()

한계: sklearn 추정기에 리터럴로 박힌 `random_state=42`는 바꾸지 않는다. 모델의
초기화 시드는 고정된 채 **입력 데이터가 학생마다 달라지므로** 결과는 갈린다.
"""

from __future__ import annotations

import hashlib
import os
import random as _py_random

ENV_VAR = "GEOAI_STUDENT_ID"

# 같은 프로세스에서 두 번 적용해 오프셋이 겹쳐 더해지는 것을 막는다.
_applied = False


def student_id() -> str:
    """환경변수에 담긴 학번. 없으면 빈 문자열."""
    return (os.environ.get(ENV_VAR) or "").strip()


def seed_offset(sid: str | None = None) -> int:
    """학번 → 0보다 큰 정수 오프셋.

    파이썬 내장 hash()는 실행할 때마다 값이 달라지므로 쓰지 않는다. SHA-256을
    쓰면 어느 컴퓨터에서 몇 번을 돌려도 같은 학번은 항상 같은 오프셋을 얻는다
    (= 학생이 다시 돌려도 자기 숫자가 유지된다).
    """
    sid = student_id() if sid is None else sid.strip()
    if not sid:
        return 0
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
    # 1 이상 1,000,000 이하. 0이 되면 "적용 안 됨"과 구별되지 않으므로 피한다.
    return int(digest[:8], 16) % 1_000_000 + 1


def _shift(seed, offset: int):
    """정수 시드에만 오프셋을 더한다. None·Generator·SeedSequence는 그대로 통과."""
    if isinstance(seed, bool):          # bool은 int의 하위형이라 먼저 걸러낸다
        return seed
    if isinstance(seed, int):
        return seed + offset
    return seed


def _patch_numpy(offset: int) -> None:
    import numpy as np

    _default_rng = np.random.default_rng
    _seed = np.random.seed
    _RandomState = np.random.RandomState

    def default_rng(seed=None):
        return _default_rng(_shift(seed, offset))

    def seed(s=None):
        return _seed(_shift(s, offset))

    # RandomState는 반드시 '클래스'로 남겨야 한다. 함수로 바꾸면 넘파이 내부의
    # isinstance(seed, RandomState) 검사가 TypeError로 깨진다.
    class RandomState(_RandomState):    # noqa: N801 (넘파이 이름을 그대로 따른다)
        def __init__(self, seed=None):
            super().__init__(_shift(seed, offset))

    np.random.default_rng = default_rng
    np.random.seed = seed
    np.random.RandomState = RandomState


def _patch_pyrandom(offset: int) -> None:
    _seed = _py_random.seed

    def seed(a=None, version=2):
        return _seed(_shift(a, offset), version)

    _py_random.seed = seed


def _patch_torch(offset: int) -> None:
    """torch는 임포트가 무거우므로 실제로 쓰는 스크립트에서만 부른다."""
    try:
        import torch
    except ImportError:
        return

    _manual_seed = torch.manual_seed

    def manual_seed(s):
        return _manual_seed(_shift(s, offset))

    torch.manual_seed = manual_seed

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
        _cuda_all = torch.cuda.manual_seed_all

        def manual_seed_all(s):
            return _cuda_all(_shift(s, offset))

        torch.cuda.manual_seed_all = manual_seed_all


def apply_student_seed(patch_torch: bool = False) -> int:
    """난수 시드에 학번 오프셋을 씌운다. 적용한 오프셋을 돌려준다(0이면 미적용).

    환경변수 `GEOAI_STUDENT_ID`가 비어 있으면 아무것도 하지 않고 0을 돌려준다.
    """
    global _applied
    if _applied:
        return seed_offset()

    offset = seed_offset()
    if offset == 0:
        return 0

    _patch_numpy(offset)
    _patch_pyrandom(offset)
    if patch_torch:
        _patch_torch(offset)

    _applied = True
    return offset
