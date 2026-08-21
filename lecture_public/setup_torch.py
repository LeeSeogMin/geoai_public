"""
딥러닝(5-7장) 설치 도우미 - 내 컴퓨터에 맞는 PyTorch를 알아서 고른다
======================================================================
PyTorch는 컴퓨터마다 깔아야 할 파일이 다르다.

  - NVIDIA 그래픽카드가 있는 Windows/Linux : 그래픽카드용(CUDA) 빌드
  - 애플 실리콘 맥(M1 이상)                : 기본 빌드 (내장 GPU를 MPS로 쓴다)
  - 그 외                                   : CPU 빌드

어느 것을 깔아야 하는지 직접 판단하지 않아도 되도록, 이 파일이 컴퓨터를 살펴보고
맞는 것을 골라 설치한다. 설치가 끝나면 실제로 GPU 연산을 한 번 돌려서 정말
동작하는지까지 확인한다.

사용법 (가상환경을 켠 상태에서):

    python practice/setup_torch.py              # 진단 후 설치
    python practice/setup_torch.py --dry-run    # 무엇을 깔지 보기만 한다
    python practice/setup_torch.py --check      # 이미 깔린 PyTorch 상태만 확인

그래픽카드가 없어도 괜찮다. 실습 코드는 CPU로도 돌아가도록 크기를 줄여 두었다.
GPU가 있으면 더 빨리 끝날 뿐이다.
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 고정 버전은 5-7장 요구사항 파일에서 읽는다(한 곳에서만 관리하기 위해).
DL_REQUIREMENTS = Path(__file__).with_name("requirements-student-dl.txt")

PYTORCH_INDEX = "https://download.pytorch.org/whl"

# PyTorch가 제공하는 CUDA 빌드 후보. 높은 것부터 시도한다.
# (2026-08 기준 실제 제공 목록에서 최근 것만 추렸다. 실제 존재 여부는 접속해서 확인한다.)
CUDA_CANDIDATES = [
    (13, 2, "cu132"),
    (13, 0, "cu130"),
    (12, 9, "cu129"),
    (12, 8, "cu128"),
    (12, 6, "cu126"),
]


def say(msg=""):
    print(msg, flush=True)


def read_pinned_versions():
    """requirements-student-dl.txt에서 고정 버전을 읽는다."""
    pins = {}
    if not DL_REQUIREMENTS.exists():
        return pins
    for line in DL_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def detect_nvidia_driver():
    """nvidia-smi로 그래픽카드와 드라이버가 지원하는 CUDA 버전을 알아낸다.

    반환: (major, minor, gpu이름) 또는 None
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None

    out = proc.stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2))

    name = "NVIDIA GPU"
    for line in out.splitlines():
        gm = re.search(r"\|\s+\d+\s+(NVIDIA[^|]*?)\s{2,}", line)
        if gm:
            name = gm.group(1).strip()
            break
    return major, minor, name


def index_exists(tag):
    """해당 빌드 저장소가 실제로 있는지 접속해서 확인한다."""
    url = f"{PYTORCH_INDEX}/{tag}/"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def decide_target():
    """이 컴퓨터에 맞는 설치 방식을 정한다.

    반환: dict(kind, index_url, reason, device)
      kind    : cuda / mps / cpu
      index_url: pip에 넘길 저장소 주소 (None이면 기본 저장소)
      device  : 설치 후 확인할 장치 이름
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return {
                "kind": "mps",
                "index_url": None,
                "reason": "애플 실리콘 맥이다. 기본 빌드가 내장 GPU(MPS)를 지원한다.",
                "device": "mps",
            }
        return {
            "kind": "cpu",
            "index_url": None,
            "reason": "인텔 맥이다. CPU 빌드를 쓴다.",
            "device": "cpu",
        }

    driver = detect_nvidia_driver()
    if driver is None:
        return {
            "kind": "cpu",
            "index_url": f"{PYTORCH_INDEX}/cpu",
            "reason": "NVIDIA 그래픽카드를 찾지 못했다(nvidia-smi 없음). CPU 빌드를 쓴다.",
            "device": "cpu",
        }

    major, minor, gpu_name = driver
    say(f"  그래픽카드   : {gpu_name}")
    say(f"  드라이버 CUDA: {major}.{minor}")

    for c_major, c_minor, tag in CUDA_CANDIDATES:
        if (c_major, c_minor) > (major, minor):
            continue  # 드라이버보다 높은 버전은 건너뛴다
        if index_exists(tag):
            return {
                "kind": "cuda",
                "index_url": f"{PYTORCH_INDEX}/{tag}",
                "reason": f"드라이버가 CUDA {major}.{minor}까지 지원한다. {tag} 빌드를 쓴다.",
                "device": "cuda",
            }

    return {
        "kind": "cpu",
        "index_url": f"{PYTORCH_INDEX}/cpu",
        "reason": (
            f"드라이버 CUDA {major}.{minor}에 맞는 PyTorch 빌드를 찾지 못했다. "
            "그래픽카드 드라이버를 최신으로 올리면 GPU를 쓸 수 있다. 일단 CPU 빌드로 진행한다."
        ),
        "device": "cpu",
    }


def installed_build_tag():
    """이미 깔린 PyTorch가 어떤 빌드인지 본다.

    버전 번호만 보면 안 된다. CPU 빌드와 CUDA 빌드는 버전이 '2.13.0'으로 같고
    뒤에 붙는 꼬리표(+cpu, +cu130)만 다르다. pip은 이 꼬리표를 무시하고
    "이미 설치됨"으로 판단해 버려서, 종류를 바꾸려면 먼저 지워야 한다.

    반환: '+cu130' 같은 꼬리표, 꼬리표가 없으면 '', 설치 안 됐으면 None
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        return None
    version = proc.stdout.strip()
    return "+" + version.split("+", 1)[1] if "+" in version else ""


def expected_build_tag(target):
    """설치하려는 빌드에 붙어야 할 꼬리표."""
    if target["index_url"] is None:
        return ""  # 맥용 기본 빌드에는 꼬리표가 없다
    return "+" + target["index_url"].rsplit("/", 1)[-1]


def pip_uninstall(packages):
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + packages
    say("  실행: " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def pip_install(packages, index_url):
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if index_url:
        cmd += ["--index-url", index_url]
    say()
    say("  실행: " + " ".join(cmd))
    say("  (내려받는 용량이 커서 몇 분 걸린다)")
    say()
    return subprocess.run(cmd).returncode == 0


def verify(expected_device):
    """설치된 PyTorch가 실제로 계산을 하는지 확인한다.

    is_available()만 보면 안 된다. 오래된 그래픽카드는 인식은 되지만
    실제 연산에서 실패하는 경우가 있어, 작은 행렬곱을 직접 돌려 본다.
    """
    code = """
import json, torch
info = {"torch": torch.__version__, "device": "cpu", "detail": ""}
try:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        x = torch.randn(64, 64, device=d)
        (x @ x).sum().item()
        info["device"] = "cuda"
        info["detail"] = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        x = torch.randn(64, 64, device=d)
        (x @ x).sum().item()
        info["device"] = "mps"
        info["detail"] = "Apple GPU"
    else:
        x = torch.randn(64, 64)
        (x @ x).sum().item()
except Exception as exc:
    info["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(info))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        say("  확인 실패: PyTorch를 불러오지 못했다.")
        say(proc.stderr.strip()[-500:])
        return False

    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        say("  확인 결과를 읽지 못했다.")
        say(proc.stdout.strip()[-500:])
        return False

    say(f"  PyTorch 버전 : {info['torch']}")
    if "error" in info:
        say(f"  연산 확인    : 실패 - {info['error']}")
        say("  GPU를 인식했지만 실제 계산에서 막혔다. CPU 빌드로 다시 설치한다:")
        say(f"    {sys.executable} -m pip install torch torchvision "
            f"--index-url {PYTORCH_INDEX}/cpu --force-reinstall")
        return False

    label = {"cuda": "그래픽카드(CUDA)", "mps": "맥 내장 GPU(MPS)", "cpu": "CPU"}
    say(f"  실제 사용 장치: {label[info['device']]} {info['detail']}".rstrip())

    if info["device"] == "cpu" and expected_device != "cpu":
        say("  예상과 다르다. GPU를 쓰려 했으나 CPU로 잡혔다.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="내 컴퓨터에 맞는 PyTorch 설치")
    parser.add_argument("--dry-run", action="store_true", help="무엇을 깔지 보기만 한다")
    parser.add_argument("--check", action="store_true", help="이미 깔린 상태만 확인한다")
    args = parser.parse_args()

    say()
    say("=" * 62)
    say("PyTorch 설치 도우미 (5-7장 딥러닝)")
    say("=" * 62)

    if args.check:
        say()
        say("[설치 상태 확인]")
        return 0 if verify("cpu") else 1

    if sys.prefix == sys.base_prefix:
        say()
        say("  가상환경이 켜져 있지 않다. 먼저 활성화한다:")
        say("    Windows : .venv\\Scripts\\activate")
        say("    macOS   : source .venv/bin/activate")
        return 1

    say()
    say(f"[1] 컴퓨터 확인  ({platform.system()} {platform.machine()})")
    target = decide_target()
    say(f"  판단        : {target['reason']}")

    pins = read_pinned_versions()
    pinned = [f"{name}=={ver}" for name, ver in pins.items()]
    say()
    say("[2] 설치할 것")
    say(f"  패키지      : {', '.join(pinned) if pinned else 'torch, torchvision'}")
    say(f"  저장소      : {target['index_url'] or 'PyPI 기본'}")

    if args.dry_run:
        say()
        say("  --dry-run 이므로 실제 설치는 하지 않는다.")
        return 0

    say()
    say("[3] 설치")
    current = installed_build_tag()
    wanted = expected_build_tag(target)
    if current is not None:
        say(f"  이미 깔린 빌드: {current or '(꼬리표 없음)'}  /  필요한 빌드: {wanted or '(꼬리표 없음)'}")
        if current == wanted:
            say("  같은 빌드가 이미 깔려 있다. 설치를 건너뛰고 동작만 확인한다.")
            say()
            say("[4] 실제로 도는지 확인")
            return 0 if verify(target["device"]) else 1
        say("  빌드 종류가 다르다. 먼저 지운 뒤 새로 깐다.")
        pip_uninstall(["torch", "torchvision"])

    ok = False
    if pinned:
        ok = pip_install(pinned, target["index_url"])
        if not ok:
            say()
            say("  고정 버전 설치가 실패했다. 이 저장소에 해당 버전이 없을 수 있다.")
            say("  버전을 풀고 다시 시도한다.")
    if not ok:
        ok = pip_install(["torch", "torchvision"], target["index_url"])
    if not ok:
        say()
        say("  설치에 실패했다. 위 오류 메시지 전체를 AI 에이전트나 담당 교수에게 보여준다.")
        return 1

    say()
    say("[4] 실제로 도는지 확인")
    if not verify(target["device"]):
        return 1

    say()
    say("  준비 끝. 5장 실습을 시작해도 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
