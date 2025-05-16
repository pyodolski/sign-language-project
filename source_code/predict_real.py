import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import os

MODEL_PATH = "model/asl_model.h5"
LABELS_PATH = "model/labels.npy"
FONT_FILENAME = "NanumGothic.ttf"
DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)
FONT_SIZE = 60
TEXT_POSITION = (50, 30)
TEXT_COLOR_RGB = (0, 0, 255)
BOX_COLOR_BGR = (255, 0, 0)

try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
except Exception as e:
    print(f"Error loading model or labels: {e}")
    print(f"모델 파일 경로: {os.path.abspath(MODEL_PATH)}")
    print(f"라벨 파일 경로: {os.path.abspath(LABELS_PATH)}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

font_path_to_use = DEFAULT_FONT_PATH
if not os.path.exists(font_path_to_use):
    print(f"Warning: Font file '{FONT_FILENAME}' not found at '{DEFAULT_FONT_PATH}'.")
    font_path_to_use_windows = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path_to_use_windows):
        font_path_to_use = font_path_to_use_windows
        print(f"Using system font: {font_path_to_use}")
    else:
        print("Using default OpenCV font (Korean may not display correctly).")
        font_path_to_use = None

pil_font = None
if font_path_to_use:
    try:
        pil_font = ImageFont.truetype(font_path_to_use, FONT_SIZE)
    except IOError:
        print(f"Error: Could not load font at '{font_path_to_use}'. Using default OpenCV font.")
        pil_font = None

print("Starting real-time prediction. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_image)

    predicted_char = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])

            if len(coords) == model.input_shape[1]:
                coords_array = np.array(coords).reshape(1, -1)

                prediction = model.predict(coords_array)
                char_index = np.argmax(prediction)

                if 0 <= char_index < len(labels):
                    predicted_char = labels[char_index]
                else:
                    predicted_char = "ERR:IDX"
            else:
                error_msg = f"Data-Model Dim Mismatch: Data({len(coords)}) vs Model({model.input_shape[1]})"
                cv2.putText(image, error_msg, (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(error_msg)
                predicted_char = "ERR:DIM"

    if pil_font and predicted_char:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text(TEXT_POSITION, predicted_char, font=pil_font, fill=TEXT_COLOR_RGB)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif predicted_char:
        cv2.putText(image, f"{predicted_char} (Font N/A)", TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 2, BOX_COLOR_BGR, 3)

    cv2.imshow("Korean Sign Language Prediction", image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Prediction stopped.")
