"""
7장 실습 1: 자율 GIS 에이전트 — 자연어 공간 쿼리와 실패 양상
=================================================================
질문: "자연어로 공간 분석을 시키면 비전문가도 GIS를 쓸 수 있다. 그런데 믿어도 되는가?"

자율 GIS(Li & Ning 2023, LLM-Geo)의 5목표 중 하나가 self-verifying(자기검증)이다.
이 실습은 자율 GIS 에이전트의 아키텍처와 **실패 양상**을 결정론적으로 시연한다.

아키텍처: 자연어 질의 → 의도 파싱 → GIS 도구 계획 → [검증] → 실행 → 결과

재현성 원칙: 실제 시스템은 LLM(GPT-4 등)으로 파싱하지만, 비용·비재현성 때문에
  여기서는 **결정론적 규칙기반 파서**로 개념을 시연한다(가짜 LLM 출력 금지).
  핵심은 파서가 아니라, '검증 단계가 없으면 무엇이 잘못되는가'를 보이는 것이다.

실패 양상(GeoAnalystBench 맥락):
  ① 좌표계(CRS) 불일치 — 위경도(EPSG:4326)를 미터좌표(EPSG:5179) 데이터에 그대로 사용
  ② hallucinated layer — 존재하지 않는 레이어("지하철역") 참조
  ③ 모호한 질의 — 의도를 파싱할 수 없음

데이터: 미리 준비한 교육용 합성 카탈로그를 불러와 쓴다(개인 식별 불가,
  격자·시설 좌표). 계산은 실제값.
실행:
    python 7-0-simdata-prep.py   # 최초 1회: 데이터 준비
    python 7-1-autonomous-gis-query.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_CRS = "EPSG:5179"          # 데이터 좌표계(한국 미터좌표)


def load_catalog():
    """미리 준비한 레이어 카탈로그를 불러온다.

    미터좌표(EPSG:5179) 가상의 도시(0~10000m 범위)의 세 레이어 —
    인구격자·도서관·학교 — 를 저장 파일에서 읽어 카탈로그 딕셔너리로 만든다.
    """
    paths = {
        "인구격자": (DATA_DIR / "pop_grid.parquet", "grid"),
        "도서관": (DATA_DIR / "libraries.parquet", "points"),
        "학교": (DATA_DIR / "schools.parquet", "points"),
    }
    missing = [str(p) for p, _ in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "데이터가 없습니다: " + ", ".join(missing)
            + "\n먼저 실행: python 7-0-simdata-prep.py"
        )
    return {
        name: {"crs": DATA_CRS, "data": pd.read_parquet(path), "kind": kind}
        for name, (path, kind) in paths.items()
    }


# --------- 도구 레지스트리(GIS 연산) ---------
def tool_buffer_count(catalog, layer, radius_m, target="인구격자"):
    pts = catalog[layer]["data"]
    grid = catalog[target]["data"]
    total = 0.0
    for _, p in pts.iterrows():
        d = np.sqrt((grid["x"] - p["x"]) ** 2 + (grid["y"] - p["y"]) ** 2)
        total += grid.loc[d <= radius_m, "pop"].sum()
    return total


# --------- 의도 파서(결정론적 규칙기반) ---------
def parse_query(q):
    """자연어 → 계획(plan dict). 파싱 불가면 error."""
    m = re.search(r"(\S+?)(?:에서|로부터)\s*([\d.]+)\s*(km|m)\s*(?:이내|반경)", q)
    if m and ("인구" in q or "사람" in q):
        layer, val, unit = m.group(1), float(m.group(2)), m.group(3)
        radius_m = val * 1000 if unit == "km" else val
        coord_crs = "EPSG:4326" if re.search(r"위도|경도|lat|lon", q) else DATA_CRS
        return {"op": "buffer_count", "layer": layer, "radius_m": radius_m,
                "coord_crs": coord_crs}
    return {"op": None, "error": "의도를 파싱할 수 없음(모호한 질의)"}


# --------- 검증 단계(self-verifying) ---------
def verify(plan, catalog):
    """실행 전 계획을 검증. 실패 양상을 여기서 잡아낸다."""
    if plan["op"] is None:
        return False, plan["error"]
    if plan["layer"] not in catalog:                       # ② hallucinated layer
        avail = ", ".join(catalog.keys())
        return False, f"존재하지 않는 레이어 '{plan['layer']}' (사용가능: {avail})"
    if plan.get("coord_crs") != DATA_CRS:                  # ① CRS 불일치
        return False, (f"좌표계 불일치: 질의 좌표 {plan['coord_crs']} vs "
                       f"데이터 {DATA_CRS} — 변환(transform) 없이 실행 불가")
    return True, "검증 통과"


def run_agent(q, catalog):
    plan = parse_query(q)
    ok, msg = verify(plan, catalog)
    if not ok:
        return {"query": q, "plan": plan, "status": "거부", "detail": msg, "result": None}
    result = tool_buffer_count(catalog, plan["layer"], plan["radius_m"])
    return {"query": q, "plan": plan, "status": "성공",
            "detail": "검증 통과", "result": result}


def main():
    print("=" * 64)
    print("자율 GIS 에이전트 — 자연어 공간 쿼리와 실패 양상 (7-1)")
    print("=" * 64)
    catalog = load_catalog()
    print(f"\n레이어 카탈로그: {', '.join(catalog.keys())} (좌표계 {DATA_CRS})")
    print("아키텍처: 질의 → 파싱 → 계획 → [검증] → 실행 → 결과\n")

    queries = [
        "도서관에서 1km 이내 인구는?",                 # 성공
        "지하철역에서 500m 이내 인구는?",              # ② hallucinated layer
        "위도 37.5 경도 127.0 학교에서 1km 이내 인구는?",  # ① CRS 불일치
        "그 동네 살기 좋은 곳 알려줘",                  # ③ 모호
    ]
    rows = []
    for q in queries:
        out = run_agent(q, catalog)
        print(f"Q: {q}")
        if out["status"] == "성공":
            print(f"   계획: buffer_count(layer={out['plan']['layer']}, "
                  f"radius={out['plan']['radius_m']:.0f}m)")
            print(f"   ✅ {out['status']}: 약 {out['result']:,.0f}명")
        else:
            print(f"   ❌ {out['status']}: {out['detail']}")
        print()
        rows.append({"query": q, "status": out["status"], "detail": out["detail"],
                     "result": out["result"]})

    df = pd.DataFrame(rows)
    n_ok = (df["status"] == "성공").sum()
    n_block = (df["status"] == "거부").sum()
    print(f"[요약] 질의 {len(df)}건 중 성공 {n_ok}건, 검증 단계가 차단 {n_block}건")
    print("  → 검증(self-verifying) 단계가 없었다면 CRS 오류·없는 레이어가 그대로")
    print("    '그럴듯한 숫자'로 출력됐을 것이다. 자율 GIS의 신뢰성은 검증에 달렸다.")
    print("  ※ 정책 함의: 자연어 질의는 비전문가의 GeoAI 접근성을 높이지만,")
    print("    출력 검증·책임성 없이 의사결정에 직결하면 hallucinated 결과의 위험이 크다.")

    csv_path = RESULTS_DIR / "autonomous_gis_query_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  질의 처리 로그 저장 → {csv_path.name}")
    print("\n[완료] 자율 GIS 에이전트 시연을 마쳤다.")


if __name__ == "__main__":
    main()
