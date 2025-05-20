from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from deep_translator import GoogleTranslator

app = Flask(__name__)

# ==== 경로 설정 ====
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.npy")

# ==== 모델 로딩 ====
try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    print("✅ 모델 및 라벨 로딩 성공")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# ==== Mediapipe 설정 ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ==== 인식 결과 저장 ====
recognized_string = ""
latest_char = ""

# ==== 영상 스트리밍 ====
def generate_frames():
    global recognized_string, latest_char
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y)]

                if len(coords) == model.input_shape[1]:
                    coords_array = np.array(coords).reshape(1, -1)
                    prediction = model.predict(coords_array)
                    idx = np.argmax(prediction)

                    if 0 <= idx < len(labels):
                        latest_char = labels[idx]
                    else:
                        latest_char = "ERR:IDX"
                else:
                    latest_char = "ERR:DIM"

        # 텍스트 출력
        display_text = f"현재: {latest_char} | 누적: {recognized_string}"
        cv2.putText(image, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==== 라우팅 ====
@app.route('/')
def index():
    return render_template('index.html')  # 메인 홈 화면

@app.route('/asl')
def asl_page():
    return render_template('asl.html')  # ASL 인식 화면

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_string')
def get_string():
    return {'string': recognized_string, 'current': latest_char}

@app.route('/add_char')
def add_char():
    global recognized_string, latest_char
    if latest_char and latest_char not in ["ERR:IDX", "ERR:DIM"]:
        recognized_string += latest_char
    return jsonify({'success': True})

@app.route('/remove_char')
def remove_char():
    global recognized_string
    if recognized_string:
        recognized_string = recognized_string[:-1]
    return jsonify({'success': True})

@app.route('/clear_string')
def clear_string():
    global recognized_string
    recognized_string = ""
    return jsonify({'success': True})

@app.route('/translate')
def translate():
    global recognized_string
    original = recognized_string.strip() or "Hello"

    try:
        # deep-translator 사용 (동기 처리)
        en = GoogleTranslator(source='auto', target='en').translate(original)
        ko = GoogleTranslator(source='auto', target='ko').translate(original)
        zh = GoogleTranslator(source='auto', target='zh-CN').translate(original)
        ja = GoogleTranslator(source='auto', target='ja').translate(original)
    except Exception as e:
        print("❌ 번역 실패:", e)
        en = ko = zh = ja = "(번역 오류)"

    return render_template('translate.html', ko=ko, en=en, zh=zh, ja=ja)

# ==== 실행 ====
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
