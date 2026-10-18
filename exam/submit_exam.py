"""
중간고사 제출 파일 만들기
=========================
    python exam/submit_exam.py --id 학번 --name 이름

하는 일
1. 문제 1~6의 원본(결함 있는 코드) q1.py ~ q6.py 를 학번으로 실행해 "수정 전 출력"을 모은다.
2. 학생이 고친 q1_fixed.py ~ q6_fixed.py 가 있으면 실행해 "수정 후 출력"과 코드 전문을 모은다.
3. 답 칸이 붙은 제출 파일 하나를 submissions/중간고사_학번.md 로 만든다.

학생이 할 일: 만들어진 파일을 열어 답 칸(">> 답:" 아래)을 채우고 그 파일 하나만 올린다.
수정본이 없거나 실행이 실패해도 파일은 만들어진다. 실패한 채로 제출해도 된다.
원본 q1.py ~ q6.py 는 고치지 않는다. 고치면 "수정 전 출력"이 달라져 제출에서 드러난다.
"""

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXAM = Path(__file__).resolve().parent
REPO = EXAM.parent
OUT_DIR = REPO / "submissions"
TIMEOUT = 600

QUESTIONS = {
    1: ("1장 + 2장. 역 3km 안 시설 수", 10, [
        ("1-1", 2, "결함이 있는 줄과 원인 (한 문장)"),
        ("1-2", 4, "수정 전·후 표: 3km 안 시설 수, 실제 3km 안인데 빠진 시설 수"),
        ("1-3", 2, "권역 면적은 수정 전후가 거의 같은데 시설 수는 왜 달라졌는가"),
        ("1-4", 2, "1장의 '새 자료를 받으면 확인하는 네 가지' 가운데 이 결함을 미리 잡을 항목과, 수정 전 출력에서 그것이 보이는 줄"),
    ]),
    2: ("3장. 지도에 0개로 나온 업종", 15, [
        ("2-1", 3, "결함이 있는 줄과, 그 계산이 사후확률이 아닌 이유"),
        ("2-2", 6, "수정 후 표: 업종별 포착률, 사후확률, 판정"),
        ("2-3", 3, "판정이 수정 전과 달라진 업종 하나를 골라 숫자로 설명"),
        ("2-4", 3, "포착률이 1을 넘은 업종을 어떻게 처리했고, 그 업종에 사후확률을 계산하면 안 되는 이유"),
    ]),
    3: ("4장. 새 지역에서도 이 점수가 나오는가", 15, [
        ("3-1", 4, "결함 두 개: 위치, 이름(두 누수 중 무엇), 고치는 방법"),
        ("3-2", 5, "수정 후 표: 무작위 R2, 블록 R2, 차이, 블록 폴드 표준편차"),
        ("3-3", 3, "무작위와 블록의 차이를 결과로 인용해도 되는가 (4장의 규칙을 내 수치에 적용)"),
        ("3-4", 3, "수정 전 보고서 문장을 수정 후 수치에 맞게 다시 쓴 문장"),
    ]),
    4: ("5장. CNN을 도입해도 되는가", 15, [
        ("4-1", 2, "'평균이 가장 높은 방법을 도입한다'가 결함인 이유"),
        ("4-2", 6, "수정 후 표: CNN-L 폴드별 차이 5개, 차이가 양수인 폴드 수, 평균 차이, 손익분기 차이, 연 순이익"),
        ("4-3", 4, "도입 여부 결론과 근거 (5개 폴드 방향 + 손익분기)"),
        ("4-4", 3, "R1을 기준선에 함께 두는 이유를 내 수치로 한 문장"),
    ]),
    5: ("6장. 거짓변화 필터", 15, [
        ("5-1", 3, "결함이 있는 줄과 원인, 정확도를 대표 지표로 쓴 문제"),
        ("5-2", 6, "수정 후 표: 필터 전·후 IoU, 정밀도, 재현율"),
        ("5-3", 3, "정확도로 이 모델을 판단하면 안 되는 이유를 내 수치(정확도 대 IoU)로 설명"),
        ("5-4", 3, "필터 전후 재현율을 비교하고, 같거나 내려간 이유를 6장의 규칙으로 설명"),
    ]),
    6: ("7장. 발주량 정하기", 15, [
        ("6-1", 3, "결함 두 개: 위치와 고치는 방법"),
        ("6-2", 6, "수정 후 표: 임계비, 점추정 손실, newsvendor 손실, 절감률"),
        ("6-3", 3, "임계비가 0.5의 어느 쪽에 있고, 발주량 보정값(offset)의 부호가 왜 그런가"),
        ("6-4", 3, "수정 전 손실(무작위 분할)과 수정 후 점추정 손실(시간순 분할)을 비교하고, 다르다면 이유"),
    ]),
}
SYNTH = ("종합", 15, [
    ("S-1", 5, "같은 예측값에 공공기관과 기업이 서로 다른 결정을 내리는 이유를 문제 2의 컷라인(0.80, 0.20) 또는 문제 6의 임계비로 설명"),
    ("S-2", 5, "오류 없이 실행되어 그럴듯한 숫자를 낸 결함 두 개를 골라, 7장의 검증 관문 관점에서 실행 전에 있었어야 할 점검"),
    ("S-3", 5, "6문제 가운데 '모델을 안 쓰거나 더 단순한 방법을 쓰라'는 결론이 나온 사례 하나와 근거 수치 (없으면 '없음'과 이유)"),
])


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(py: Path, student_id: str) -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [sys.executable, py.name, "--id", student_id],
            cwd=str(EXAM), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=TIMEOUT, env=env,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr,
                "code": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"[시간 초과] {TIMEOUT}초", "code": -1}


def block(res: dict) -> str:
    text = res["stdout"].rstrip()
    if not res["ok"]:
        text += "\n[오류 종료 코드 %s]\n%s" % (res["code"], res["stderr"].rstrip())
    return "```text\n" + text + "\n```\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="학번")
    ap.add_argument("--name", required=True, help="이름")
    a = ap.parse_args()
    if not a.id.isalnum():
        print("학번은 숫자나 영문자만 써야 합니다.")
        return 2

    parts = [
        f"# 중간고사 제출 -- {a.id} {a.name}\n",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 실행 환경: Python {sys.version.split()[0]} / {platform.platform()}",
        "- 이 파일의 `>> 답:` 아래만 채워서 이 파일 하나를 올린다.\n",
    ]
    status = []
    for n, (title, pts, items) in QUESTIONS.items():
        orig = EXAM / f"q{n}.py"
        fixed = EXAM / f"q{n}_fixed.py"
        parts.append(f"\n---\n\n## 문제 {n} ({pts}점). {title}\n")
        parts.append(f"원본 코드 지문: `q{n}.py` (해시 {sha256_file(orig)})\n")
        parts.append("### 수정 전 출력\n")
        before = run(orig, a.id)
        parts.append(block(before))
        parts.append("### 수정 후 출력\n")
        if fixed.exists():
            after = run(fixed, a.id)
            parts.append(f"<!-- AFTER q{n} -->\n")
            parts.append(block(after))
            parts.append(f"### 수정한 코드 전문 (`q{n}_fixed.py`)\n")
            parts.append("```python\n" + fixed.read_text(encoding="utf-8").rstrip() + "\n```\n")
            note = ""
            if after["ok"] and "@@KEY" not in after["stdout"]:
                note = " -- @@KEY 줄이 출력되지 않음 (원본 q%d.py의 key(...) 줄을 확인)" % n
            status.append(f"문제 {n}: 수정본 실행 {'성공' if after['ok'] else '실패'}{note}")
        else:
            parts.append(f"(q{n}_fixed.py 가 없어 수정 후 출력이 없음. 막힌 곳을 아래 답에 적는다.)\n")
            status.append(f"문제 {n}: 수정본 없음")
        parts.append("### 답안\n")
        for code, p, desc in items:
            parts.append(f"**{code}** ({p}점) {desc}\n\n>> 답:\n\n")

    parts.append("\n---\n\n## 종합 (15점)\n")
    for code, p, desc in SYNTH[2]:
        parts.append(f"**{code}** ({p}점) {desc}\n\n>> 답:\n\n")
    parts.append("\n---\n\n## 선택: 사용한 AI 도구와 도움을 받은 부분 (감점 없음)\n\n>> 답:\n")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"중간고사_{a.id}.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(status))
    print(f"제출 파일: {out}")
    print("열어서 '>> 답:' 아래를 채운 뒤 이 파일 하나만 올리세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
