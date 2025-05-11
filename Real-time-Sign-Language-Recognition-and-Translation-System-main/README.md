# 실시간 수화 인식 및 번역 시스템

이 프로젝트는 컴퓨터 비전과 머신 러닝을 활용하여 미국 수화(ASL)의 문자와 구문을 나타내는 손짓을 인식하고 텍스트로 번역합니다. 또한, 힌디어, 텔루구어, 타밀어, 칸나다어, 말라얄람어 등 인도어를 포함한 여러 언어로 번역 기능을 제공합니다.

## 필수 조건 
이 프로젝트를 실행하기 전에 다음이 설치되어 있는지 확인하세요.

- Python(버전 3.6 이상) 
- OpenCV ( opencv-python) 
- 미디어파이프 ( mediapipe) 
- 구글 번역 API ( googletrans==4.0.0-rc1) 
- 맞춤법 검사기 ( spellchecker) 
Python의 패키지 관리자( )를 사용하여 이러한 종속성을 설치할 수 있습니다 pip.

You can install these dependencies using Python's package manager (`pip`):

```
pip install opencv-python mediapipe googletrans==4.0.0-rc1 spellchecker
```

##프로젝트 구조

### 1. 데이터 수집 (`collect_images.py`)
- 알파벳 각 글자와 "감사합니다", "만나서 반갑습니다"와 같은 추가 메시지의 이미지를 카메라로 캡처합니다.
- 이미지는 해당 레이블 이름의 디렉토리에 저장됩니다 (`./data`).

### 2. 데이터 준비 (`create_dataset.py`)
- 수집된 이미지를 Mediapipe로 처리하여 손 랜드마크를 추출합니다.
- 랜드마크 데이터와 레이블을 포함한 데이터셋(`data.pickle`)을 생성합니다.

### 3. 모델 학습 (`train_model.py`)
- `data.pickle`에서 데이터셋을 로드합니다.
- 추출된 랜드마크를 기반으로 손동작을 분류하는 `RandomForestClassifier`를 학습합니다.
- 학습된 모델을 평가하고, 모델 파일(`model.p`)로 저장합니다.

### 4. 실시간 번역 및 GUI (`translate_gui.py`)
- 훈련된 모델과 Mediapipe를 사용하여 실시간 손동작 인식을 구현합니다.
- 인식된 제스처를 영어 텍스트로 표시합니다.
- Google Translate API를 통해 인식된 텍스트를 선택한 언어로 번역합니다.

---

## 사용법 🚀

1. **데이터 수집**
    - 훈련용 이미지를 촬영하려면 `collect_images.py`를 실행하세요.
    - 화면의 지시에 따라 제스처와 메시지를 촬영합니다.

2. **데이터 준비**
    - 캡처한 이미지를 처리하려면 `create_dataset.py`를 실행하여 `data.pickle`을 생성하세요.

3. **모델 훈련**
    - `train_model.py`를 실행하여 준비된 데이터셋으로 `RandomForestClassifier`를 학습하세요.

4. **실시간 번역 및 GUI**
    - `translate_gui.py`를 실행하면 그래픽 사용자 인터페이스(GUI)가 열립니다.
    - 인식된 제스처를 번역하려면 드롭다운 메뉴에서 언어를 선택하세요.

---

## 노트 ℹ️

- 더 나은 인식 정확도를 위해 이미지를 수집할 때 적절한 조명과 손 위치를 확보하세요.
- GUI는 카메라 입력에 따라 실시간으로 업데이트됩니다. GUI 창을 닫으면 프로그램이 종료됩니다.
- 특정 애플리케이션에 맞게 매개변수를 조정하고 필요한 제스처를 추가할 수 있습니다.

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
