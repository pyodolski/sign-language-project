import cv2
import mediapipe as mp
from collections import deque
from wordsegment import load, segment
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import os
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# 설정
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['space', 'end']
SAVE_DIR = './project_data'  # 프로젝트 루트 내 저장 경로
os.makedirs(SAVE_DIR, exist_ok=True)
MODE = 'ASL'  # 추후 확장 가능 (예: 'KSL')

# 예측기 클래스
class SignPredictor:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sequence):
        input_data = np.array(sequence, dtype=np.float32).reshape(1, 30, 63)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return LABELS[np.argmax(output)]

# 손 관절 추적
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 시퀀스 처리
SEQ_LENGTH = 30
sequence = deque(maxlen=SEQ_LENGTH)

def preprocess_landmarks(landmarks):
    return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]

def update_sequence(landmarks):
    sequence.append(preprocess_landmarks(landmarks))
    return list(sequence)

# 번역 및 음성
translator = Translator()
load()  # wordsegment 초기화

def translate(text, dest='ko'):
    return translator.translate(text, src='en', dest=dest).text

def speak_korean(text):
    tts = gTTS(text=text, lang='ko')
    mp3_path = os.path.join(SAVE_DIR, "speech.mp3")
    tts.save(mp3_path)
    playsound(mp3_path)

# 문자 버퍼
buffer = ""
prev_char = ""
last_time = time.time()

def add_char(char):
    global buffer, prev_char, last_time
    if char == prev_char and time.time() - last_time < 1.0:
        return None
    prev_char = char
    last_time = time.time()
    buffer += ' ' if char == 'space' else char
    return buffer

# 모델 로딩
model_path = "asl_model.tflite"
predictor = SignPredictor(model_path)

# 메인 실행 루프
cap = cv2.VideoCapture(0)
print("▶ 실시간 수어 인식 시작 (q 키로 종료)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            seq = update_sequence(hand_landmarks.landmark)
            if len(seq) == 30:
                predicted = predictor.predict(seq)
                print(f"예측: {predicted}")
                buf = add_char(predicted)
                if buf:
                    print(f"버퍼: {buf}")
                if predicted == 'space':
                    sentence = " ".join(segment(buffer.strip()))
                    print("문장 생성:", sentence)
                    translated = translate(sentence)
                    print("번역 결과:", translated)
                    speak_korean(translated)
                    buffer = ""

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
