"""
실습 환경 자가진단
==================
실습을 시작하기 전에 이 파일을 한 번 실행한다. 내 컴퓨터에서 무엇이 준비됐고
무엇이 빠졌는지 항목별로 알려주고, 빠진 항목은 어떻게 채우는지까지 알려준다.

    python lecture_practice/check_env.py

Windows와 macOS 모두에서 같은 방식으로 동작한다.

이 파일 자체는 어떤 외부 패키지도 필요로 하지 않는다. 파이썬만 깔려 있으면
실행되므로, 설치가 실패한 상태에서도 진단 결과를 볼 수 있다.

주의: 출력에는 cp949(윈도우 한국어 기본 인코딩)에서 표현할 수 없는 기호를
쓰지 않는다. 화살표나 이모지를 넣으면 진단 스크립트 자체가 깨질 수 있다.
"""

import importlib
import locale
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# 진단 결과를 파일로도 남긴다. 온라인 수업이라 화면을 옆에서 봐줄 사람이 없으므로,
# 막혔을 때 이 파일을 그대로 첨부해 물어보면 상태를 설명하지 않아도 된다.
REPORT_PATH = Path(__file__).with_name("env-report.txt")


class _Tee:
    """화면에 찍으면서 동시에 내용을 모아 둔다."""

    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        # encoding, isatty 처럼 여기 없는 속성은 원래 출력 통로에 그대로 물어본다.
        return getattr(self.stream, name)

    def text(self):
        return "".join(self.parts)

# ----------------------------------------------------------------------
# 점검 대상 패키지
#   core: 1-4장, 8-13장에서 쓴다. 학기 대부분을 이것만으로 진행한다.
#   dl  : 5-7장 딥러닝에서만 쓴다. 용량이 크므로 필요할 때 설치해도 된다.
# ----------------------------------------------------------------------
CORE_PACKAGES = [
    ("geopandas", "벡터 공간데이터"),
    ("shapely", "도형 연산"),
    ("pyproj", "좌표계 변환"),
    ("rasterio", "위성영상 래스터"),
    ("xarray", "다차원 배열"),
    ("pystac_client", "위성영상 카탈로그 검색"),
    ("pyarrow", "Parquet 입출력"),
    ("libpysal", "공간 가중행렬"),
    ("esda", "공간 자기상관 통계"),
    ("numpy", "수치 배열"),
    ("pandas", "표 데이터"),
    ("sklearn", "머신러닝"),
    ("statsmodels", "통계 모형"),
    ("xgboost", "부스팅 모델"),
    ("lightgbm", "부스팅 모델"),
    ("matplotlib", "그래프"),
    ("shap", "모델 해석"),
]

DL_PACKAGES = [
    ("torch", "딥러닝(5-7장)"),
    ("torchvision", "영상 딥러닝(5-7장)"),
]

# 한글 라벨이 들어간 그림을 그리려면 아래 중 하나는 있어야 한다.
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "AppleSDGothicNeo", "NanumGothic"]

MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = "3.12"


def head(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def check_python():
    """파이썬 버전과 가상환경 사용 여부를 확인한다."""
    head("1. 파이썬")
    v = sys.version_info
    print(f"  버전   : {v.major}.{v.minor}.{v.micro}")
    print(f"  실행파일: {sys.executable}")
    print(f"  운영체제: {platform.system()} {platform.release()} ({platform.machine()})")

    problems = []
    if (v.major, v.minor) < MIN_PYTHON:
        problems.append(
            f"파이썬 {v.major}.{v.minor}은 너무 낮다. "
            f"{RECOMMENDED_PYTHON} 이상을 설치한다."
        )

    # 가상환경 안에서 도는지 판정한다. venv를 쓰면 base_prefix가 prefix와 달라진다.
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"  가상환경: 사용 중 ({Path(sys.prefix).name})")
    else:
        print("  가상환경: 사용 안 함")
        problems.append(
            "가상환경 밖에서 돌고 있다. README의 2단계대로 .venv를 만들고 "
            "활성화한 다음 다시 실행한다."
        )
    return problems


def check_encoding():
    """윈도우 한국어 환경에서 한글이 깨지는 문제를 미리 잡는다."""
    head("2. 문자 인코딩")
    preferred = locale.getpreferredencoding(False)
    print(f"  기본 인코딩 : {preferred}")
    print(f"  표준출력    : {sys.stdout.encoding}")
    print(f"  PYTHONUTF8  : {os.environ.get('PYTHONUTF8', '(설정 안 됨)')}")

    problems = []
    is_utf8 = preferred.lower().replace("-", "") in ("utf8", "cp65001")
    if not is_utf8:
        problems.append(
            "기본 인코딩이 UTF-8이 아니다. 실행 결과를 파일로 저장할 때 "
            "한글과 특수기호가 깨진다. 환경변수 PYTHONUTF8=1을 설정한다. "
            "(README의 '윈도우 인코딩' 항목 참고)"
        )
    else:
        print("  판정        : UTF-8 정상")
    return problems


def check_packages(packages, label, required):
    """패키지를 실제로 임포트해 보고 버전을 찍는다."""
    head(f"3. 패키지 - {label}")
    missing = []
    for module, purpose in packages:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "버전정보 없음")
            print(f"  [OK] {module:<16} {version:<12} {purpose}")
        except ImportError:
            print(f"  [--] {module:<16} {'없음':<12} {purpose}")
            missing.append(module)

    problems = []
    if missing and required:
        problems.append(
            f"{label} 패키지가 빠졌다: {', '.join(missing)}. "
            "README의 3단계 설치 명령을 다시 실행한다."
        )
    elif missing:
        print()
        print(f"  참고: {label}은 아직 없어도 된다. 5장에 들어갈 때 설치한다.")
    return problems


def detect_nvidia():
    """NVIDIA 그래픽카드가 달려 있는지 본다. 없으면 None을 돌려준다."""
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    import re

    m = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
    return m.group(1) if m else "확인 불가"


def check_gpu():
    """딥러닝을 어느 장치로 돌리게 되는지 미리 알려준다."""
    head("4. 딥러닝 장치 (5-7장)")
    problems = []

    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )
    cuda_driver = detect_nvidia() if platform.system() != "Darwin" else None

    if cuda_driver:
        print(f"  하드웨어  : NVIDIA 그래픽카드 있음 (드라이버 CUDA {cuda_driver})")
    elif is_apple_silicon:
        print("  하드웨어  : 애플 실리콘 (내장 GPU 사용 가능)")
    else:
        print("  하드웨어  : 쓸 수 있는 GPU 없음 (CPU로 진행한다)")

    try:
        import torch
    except ImportError:
        print("  PyTorch   : 아직 설치 안 함 (5장에서 설치한다)")
        print("  설치 방법 : python lecture_practice/setup_torch.py")
        return problems

    device = "cpu"
    detail = ""
    try:
        if torch.cuda.is_available():
            x = torch.randn(64, 64, device="cuda")
            (x @ x).sum().item()  # 인식만 되고 계산이 안 되는 경우를 걸러낸다
            device, detail = "cuda", torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            x = torch.randn(64, 64, device="mps")
            (x @ x).sum().item()
            device, detail = "mps", "Apple GPU"
    except Exception as exc:
        problems.append(
            f"GPU를 인식했지만 계산에서 실패했다({type(exc).__name__}). "
            "python lecture_practice/setup_torch.py 를 다시 실행해 CPU 빌드로 바꾼다."
        )

    label = {"cuda": "그래픽카드(CUDA)", "mps": "맥 내장 GPU(MPS)", "cpu": "CPU"}
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  실제 사용 : {label[device]} {detail}".rstrip())

    if device == "cpu" and (cuda_driver or is_apple_silicon):
        problems.append(
            "GPU가 있는데 CPU로 돌게 되어 있다. 학습이 몇 배 느려진다. "
            "python lecture_practice/setup_torch.py 를 실행하면 맞는 버전으로 바꿔준다."
        )
    return problems


def check_korean_font():
    """한글이 들어간 그림이 두부(네모)로 깨지는지 미리 확인한다."""
    head("5. 한글 폰트")
    try:
        from matplotlib import font_manager
    except ImportError:
        print("  matplotlib이 없어 건너뛴다.")
        return []

    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in KOREAN_FONTS if f in installed]
    if found:
        print(f"  사용 가능: {', '.join(found)}")
        return []

    print("  사용 가능한 한글 폰트를 찾지 못했다.")
    return [
        "한글 폰트가 없어 그림의 한글이 네모로 깨진다. "
        "나눔고딕을 설치하거나(무료), 윈도우면 '맑은 고딕'이 있는지 확인한다."
    ]


def check_network():
    """1장 실습은 인터넷에서 경계 데이터를 내려받는다."""
    head("6. 네트워크")
    import urllib.request

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  Natural Earth 접속: 정상 (HTTP {resp.status})")
        return []
    except Exception as exc:  # 사내망/방화벽 등 원인이 다양해 넓게 받는다
        print(f"  Natural Earth 접속 실패: {type(exc).__name__}")
        return [
            "실습 데이터 서버에 접속하지 못했다. 학교 와이파이나 방화벽이 "
            "막고 있을 수 있다. 다른 네트워크에서 다시 시도한다."
        ]


def main():
    print()
    print("GeoAI 실습 환경 자가진단")
    print("(문제가 있으면 아래 '해야 할 일'에 무엇을 하면 되는지 나온다)")

    problems = []
    problems += check_python()
    problems += check_encoding()
    problems += check_packages(CORE_PACKAGES, "기본(1-4장, 8-13장)", required=True)
    problems += check_packages(DL_PACKAGES, "딥러닝(5-7장)", required=False)
    problems += check_gpu()
    problems += check_korean_font()
    problems += check_network()

    head("진단 결과")
    if not problems:
        print("  모든 항목 통과. 실습을 시작해도 된다.")
        print()
        print("  다음 단계: python lecture_practice/chapter1/code/1-1-geoai-tools-preview.py")
        return 0

    print(f"  해결할 항목 {len(problems)}개")
    print()
    for i, p in enumerate(problems, 1):
        print(f"  {i}) {p}")
    print()
    print("  [혼자 해결하는 순서]")
    print("  1) 위 문장을 AI 코딩 에이전트에 그대로 붙여넣고")
    print("     '내 운영체제에 맞게 해결 방법을 단계로 알려줘'라고 물어본다.")
    print("  2) 시킨 대로 한 뒤 이 진단을 다시 돌려 항목이 사라졌는지 확인한다.")
    print("  3) 두세 번 시도해도 같은 항목이 남으면, 아래 진단 파일을 첨부해")
    print("     수업에서 안내한 문의 창구에 올린다. 화면을 설명할 필요가 없다.")
    return 1


def run():
    """진단을 돌리고, 화면에 나온 내용을 그대로 파일로도 남긴다."""
    original = sys.stdout
    tee = _Tee(original)
    sys.stdout = tee
    try:
        code = main()
    finally:
        sys.stdout = original

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"GeoAI 실습 환경 진단 결과\n생성 시각: {stamp}\n"
    try:
        REPORT_PATH.write_text(header + tee.text(), encoding="utf-8")
        print()
        print(f"  진단 결과를 파일로도 저장했다: {REPORT_PATH}")
        print("  질문할 때 이 파일을 첨부하면 된다.")
    except OSError as exc:
        print()
        print(f"  (진단 파일을 저장하지 못했다: {type(exc).__name__})")
    return code


if __name__ == "__main__":
    sys.exit(run())
