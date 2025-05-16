# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
#
# model = load_model("model/asl_model.h5")
# # labels = np.load("model/labels.npy")
# labels = np.load("model/labels.npy", allow_pickle=True)
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     image = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)
#
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             coords = []
#             for lm in hand_landmarks.landmark:
#                 coords.extend([lm.x, lm.y])
#             coords = np.array(coords).reshape(1, -1)
#             pred = model.predict(coords)
#             char = labels[np.argmax(pred)]
#
#             cv2.putText(image, f"{char}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
#
#     cv2.imshow("Predict", image)
#     if cv2.waitKey(10) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# =======================================
#
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # Pillow 라이브러리 임포트
import os  # 폰트 경로 확인을 위해 추가

# --- 설정 값 ---
MODEL_PATH = "model/asl_model.h5"
LABELS_PATH = "model/labels.npy"
# 사용할 폰트 파일 경로 (스크립트와 같은 디렉토리에 있다고 가정)
# 실제 폰트 파일명으로 변경하거나, 전체 경로를 지정하세요.
# 예: "C:/Windows/Fonts/malgun.ttf" (Windows 맑은 고딕)
# 예: "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" (Linux 나눔고딕)
FONT_FILENAME = "NanumGothic.ttf"  # 나눔고딕 폰트 파일명을 예시로 사용
DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)  # 스크립트 폴더 기준
FONT_SIZE = 60
TEXT_POSITION = (50, 30)  # 텍스트 표시 위치 (x, y)
TEXT_COLOR_RGB = (0, 0, 255)  # 텍스트 색상 (Pillow는 RGB 순서, 파란색)
BOX_COLOR_BGR = (255, 0, 0)  # 랜드마크 박스 색상 (OpenCV는 BGR 순서, 파란색)
# --- ---

# 모델과 라벨 로드
try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)  # allow_pickle=True 추가
except Exception as e:
    print(f"Error loading model or labels: {e}")
    print(f"모델 파일 경로: {os.path.abspath(MODEL_PATH)}")
    print(f"라벨 파일 경로: {os.path.abspath(LABELS_PATH)}")
    exit()

# MediaPipe Hands 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 웹캠 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 폰트 로드 시도
font_path_to_use = DEFAULT_FONT_PATH
if not os.path.exists(font_path_to_use):
    print(f"Warning: Font file '{FONT_FILENAME}' not found at '{DEFAULT_FONT_PATH}'.")
    # Windows 시스템 폰트 경로 시도 (맑은 고딕)
    font_path_to_use_windows = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path_to_use_windows):
        font_path_to_use = font_path_to_use_windows
        print(f"Using system font: {font_path_to_use}")
    else:
        print("Using default OpenCV font (Korean may not display correctly).")
        font_path_to_use = None  # Pillow 폰트 사용 불가

pil_font = None
if font_path_to_use:
    try:
        pil_font = ImageFont.truetype(font_path_to_use, FONT_SIZE)
    except IOError:
        print(f"Error: Could not load font at '{font_path_to_use}'. Using default OpenCV font.")
        pil_font = None  # 폰트 로드 실패 시 None으로 설정

print("Starting real-time prediction. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # 이미지 좌우 반전 및 BGR에서 RGB로 변환
    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe Hands를 사용하여 손 랜드마크 처리
    result = hands.process(rgb_image)

    predicted_char = ""  # 예측된 글자 초기화

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 랜드마크 좌표 추출 및 정규화 (hand_capture.py와 동일한 방식)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])  # x, y 좌표만 사용

            # 입력 데이터의 feature 개수가 모델과 맞는지 확인
            # hand_capture.py에서는 21개 랜드마크 * 2 (x,y) = 42개의 feature
            # train_model.py에서 X.shape[1]과 일치해야 함
            if len(coords) == model.input_shape[1]:
                coords_array = np.array(coords).reshape(1, -1)  # 모델 입력을 위해 2D 배열로 변환

                # 모델 예측
                prediction = model.predict(coords_array)
                char_index = np.argmax(prediction)

                if 0 <= char_index < len(labels):
                    predicted_char = labels[char_index]
                else:
                    predicted_char = "ERR:IDX"  # 라벨 인덱스 오류
            else:
                # 입력 데이터의 feature 개수가 모델과 다를 경우 에러 메시지 표시
                error_msg = f"Data-Model Dim Mismatch: Data({len(coords)}) vs Model({model.input_shape[1]})"
                cv2.putText(image, error_msg, (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(error_msg)  # 콘솔에도 출력
                predicted_char = "ERR:DIM"

    # Pillow를 사용하여 한글 텍스트 오버레이
    if pil_font and predicted_char:
        # OpenCV 이미지를 Pillow 이미지로 변환 (BGR -> RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        # 텍스트 그리기
        draw.text(TEXT_POSITION, predicted_char, font=pil_font, fill=TEXT_COLOR_RGB)
        # Pillow 이미지를 다시 OpenCV 이미지로 변환 (RGB -> BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif predicted_char:  # Pillow 폰트 로드 실패 시 OpenCV의 putText 사용 (한글 깨짐 가능성)
        cv2.putText(image, f"{predicted_char} (Font N/A)", TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 2, BOX_COLOR_BGR, 3)

    # 결과 이미지 보여주기
    cv2.imshow("Korean Sign Language Prediction", image)

    # 'ESC' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Prediction stopped.")
