"""중간고사 공통 도구: 학번 시드, 자료 경로, 채점용 키 출력."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PRACTICE = REPO / "lecture_practice"


def parse_id() -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="학번")
    return p.parse_args().id


def seed_of(student_id: str) -> int:
    return int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)


def rng_of(student_id: str, salt: int) -> np.random.Generator:
    return np.random.default_rng(seed_of(student_id) + salt)


def key(name: str, value) -> None:
    """채점기가 읽는 줄. 이 줄의 이름을 바꾸거나 지우지 않는다."""
    if isinstance(value, float):
        value = round(value, 4)
    print(f"@@KEY {name}={value}")
