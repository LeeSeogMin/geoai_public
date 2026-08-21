"""
_s2_data.py (Ch5 shim) — Ch4의 공통 로더를 재사용한다.
실제 Sentinel-2 클립·라벨은 practice/chapter4/data/에 한 벌만 두고,
Ch5는 명확한 상대경로(../../chapter4/code)로 같은 로더를 불러 쓴다.
파일명이 '_'로 시작하므로 실행 증거 게이트의 실행 대상에서 제외된다.
"""

import importlib.util
from pathlib import Path

_CH4 = (Path(__file__).resolve().parent.parent.parent
        / "chapter4" / "code" / "_s2_data.py")
_spec = importlib.util.spec_from_file_location("_s2_data_ch4", _CH4)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BAND_NAMES = _mod.BAND_NAMES
WC_CLASSES = _mod.WC_CLASSES
build_pixel_table = _mod.build_pixel_table
extract_patches = _mod.extract_patches
load_reflectance = _mod.load_reflectance
spectral_indices = _mod.spectral_indices
