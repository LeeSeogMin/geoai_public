# GeoAI 강의·실습 공개자료

이 저장소는 GeoAI 강의 원고와 학생용 실습 코드를 제공한다.

## 폴더

- `lecture/`: GeoAI 강의 원고
- `lecture_public/`: 학생용 실습 코드와 실행 결과

`lecture_public/data/`는 용량이 큰 입력자료를 포함하지 않는다. 각 장의 데이터 준비 코드나 README에 적힌 절차로 필요한 자료를 생성한다.

## 시작하기

```bash
git clone https://github.com/LeeSeogMin/geoai_public.git
cd geoai_public/lecture_public
python -m venv .venv
# Windows PowerShell: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-student.txt
python check_env.py
```

운영체제별 상세 안내는 `lecture_public/README.md`를 읽는다. 딥러닝 장은 `lecture_public/setup_torch.py`와 장별 README의 추가 안내를 따른다.
