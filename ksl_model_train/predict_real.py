import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import os

# 기본 설정
MODEL_PATH = "model/asl_model.h5"
LABELS_PATH = "model/labels.npy"
FONT_FILENAME = "GmarketSansTTFMedium.ttf"
DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)
FONT_SIZE = 60
TEXT_POSITION = (50, 30)
TEXT_COLOR_RGB = (0, 0, 255)
BOX_COLOR_BGR = (255, 0, 0)

# 모델과 라벨 로딩
try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
except Exception as e:
    print(f"모델 또는 라벨 로딩 중 오류 발생: {e}")
    exit()

# Mediapipe 세팅
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 카메라
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 폰트 설정
font_path_to_use = DEFAULT_FONT_PATH
if not os.path.exists(font_path_to_use):
    font_path_to_use_windows = "C:/Windows/Fonts/GmarketSansTTFMedium.ttf"
    if os.path.exists(font_path_to_use_windows):
        font_path_to_use = font_path_to_use_windows
    else:
        font_path_to_use = None

pil_font = None
if font_path_to_use:
    try:
        pil_font = ImageFont.truetype(font_path_to_use, FONT_SIZE)
    except:
        pil_font = None

print("수화 인식을 시작하려면 SPACE를, 종료하려면 ESC를 누르세요.")

latest_char = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("오류: 이미지 캡처에 실패했습니다.")
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 글자 출력 (직전 인식 결과)
    if pil_font and latest_char:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text(TEXT_POSITION, latest_char, font=pil_font, fill=TEXT_COLOR_RGB)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif latest_char:
        cv2.putText(image, latest_char, TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 2, BOX_COLOR_BGR, 3)

    cv2.imshow("수화 인식", image)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y])

                if len(coords) == model.input_shape[1]:
                    coords_array = np.array(coords).reshape(1, -1)
                    prediction = model.predict(coords_array)
                    char_index = np.argmax(prediction)

                    if 0 <= char_index < len(labels):
                        latest_char = labels[char_index]
                        print(f"🔤 인식 결과: {latest_char}")
                    else:
                        latest_char = "ERR:IDX"
                        print("⚠️ 예측 결과 인덱스 오류")
                else:
                    latest_char = "ERR:DIM"
                    print(f"⚠️ 좌표 수 불일치: {len(coords)} → 필요: {model.input_shape[1]}")
        else:
            print("🖐️ 손이 인식되지 않았습니다. 다시 시도하세요.")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("인식이 종료되었습니다.")
