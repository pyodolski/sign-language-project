from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import os
from deep_translator import GoogleTranslator

app = Flask(__name__)

# ==== 경로 설정 ====
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

ASL_MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.tflite")
ASL_LABELS_PATH = os.path.join(MODEL_DIR, "asl_labels.npy")

KSL_MODEL_PATH = os.path.join(MODEL_DIR, "ksl_model.tflite")
KSL_LABELS_PATH = os.path.join(MODEL_DIR, "ksl_labels.npy")

# ==== 모델 로딩 ====
try:
    asl_interpreter = tf.lite.Interpreter(model_path=ASL_MODEL_PATH)
    asl_interpreter.allocate_tensors()
    asl_input_details = asl_interpreter.get_input_details()
    asl_output_details = asl_interpreter.get_output_details()
    labels_asl = np.load(ASL_LABELS_PATH, allow_pickle=True)

    ksl_interpreter = tf.lite.Interpreter(model_path=KSL_MODEL_PATH)
    ksl_interpreter.allocate_tensors()
    ksl_input_details = ksl_interpreter.get_input_details()
    ksl_output_details = ksl_interpreter.get_output_details()
    labels_ksl = np.load(KSL_LABELS_PATH, allow_pickle=True)

    print("✅ ASL, KSL 모델 및 라벨 로딩 성공")
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
recognized_string = {"asl": "", "ksl": ""}
latest_char = {"asl": "", "ksl": ""}

# ==== 공통 영상 스트리밍 ====
def generate_frames(interpreter, input_details, output_details, labels, lang_key):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    last_prediction_time = 0
    prediction_interval = 5  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            image = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_image)

            current_time = time.time()

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coords = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y)]

                    if current_time - last_prediction_time >= prediction_interval:
                        input_data = np.array(coords, dtype=np.float32).reshape(1, -1)
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                        idx = np.argmax(prediction)

                        if 0 <= idx < len(labels):
                            latest_char[lang_key] = labels[idx]
                        else:
                            latest_char[lang_key] = "ERR:IDX"

                        last_prediction_time = current_time

            display_text = f"현재: {latest_char[lang_key]} | 누적: {recognized_string[lang_key]}"
            cv2.putText(image, display_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        print("🛑 스트리밍 중단 감지: 클라이언트 연결 종료됨")
    finally:
        cap.release()
        print("✅ 카메라 자원 해제 완료")

# ==== 라우팅 ====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asl')
def asl_page():
    return render_template('asl.html')

@app.route('/ksl')
def ksl_page():
    return render_template('ksl.html')

@app.route('/video_feed_asl')
def video_feed_asl():
    return Response(generate_frames(asl_interpreter, asl_input_details, asl_output_details, labels_asl, "asl"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ksl')
def video_feed_ksl():
    return Response(generate_frames(ksl_interpreter, ksl_input_details, ksl_output_details, labels_ksl, "ksl"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_string/<lang>')
def get_string(lang):
    return {'string': recognized_string[lang], 'current': latest_char[lang]}

@app.route('/add_char/<lang>')
def add_char(lang):
    if latest_char[lang] and latest_char[lang] not in ["ERR:IDX", "ERR:DIM"]:
        recognized_string[lang] += latest_char[lang]
    return jsonify({'success': True})

@app.route('/remove_char/<lang>')
def remove_char(lang):
    if recognized_string[lang]:
        recognized_string[lang] = recognized_string[lang][:-1]
    return jsonify({'success': True})

@app.route('/clear_string/<lang>')
def clear_string(lang):
    recognized_string[lang] = ""
    return jsonify({'success': True})

@app.route('/translate/<lang>')
def translate(lang):
    original = recognized_string[lang].strip() or "Hello"
    try:
        en = GoogleTranslator(source='auto', target='en').translate(original)
        ko = GoogleTranslator(source='auto', target='ko').translate(original)
        zh = GoogleTranslator(source='auto', target='zh-CN').translate(original)
        ja = GoogleTranslator(source='auto', target='ja').translate(original)
    except Exception as e:
        print("❌ 번역 실패:", e)
        en = ko = zh = ja = "(번역 오류)"

    return render_template('translate.html', ko=ko, en=en, zh=zh, ja=ja)

@app.route('/edu/<lang>')
def edu_page(lang):
    string = recognized_string.get(lang, "")
    chars = list(string)
    return render_template("edu.html", chars=chars, lang=lang)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
