# 예시 main함수 코드
import cv2
import mediapipe as mp
from landmark_sequence_collector import update_sequence
from tflite_predictor import ASLPredictor
from buffer_manager import add_char
from word_segmenter import build_english_sentence
from translator import translate_to_korean
from tts_speaker import speak_korean

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 수어 예측 모델 로드
predictor = ASLPredictor()

# 웹캠 시작
cap = cv2.VideoCapture(0)
print("실행 중... 수어를 입력하고 'q'를 눌러 종료하세요.")

# 알파벳 누적 버퍼
result_text = ""

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

            # 시퀀스 누적
            seq = update_sequence(hand_landmarks.landmark)
            if len(seq) == 30:
                try:
                    pred = predictor.predict(seq)

                    if pred == ' ':  # SPACE 수어가 인식된 경우
                        if result_text.strip():  # 이전 알파벳이 있다면
                            sentence = build_english_sentence(result_text)
                            print(f"[영어 문장] {sentence}")

                            korean = translate_to_korean(sentence)
                            print(f"[한국어 번역] {korean}")

                            speak_korean(korean)
                            result_text = ""  # 누적 초기화
                    else:
                        print(f"예측된 알파벳: {pred}")
                        temp = add_char(pred)
                        if temp:
                            result_text = temp
                            print(f"누적된 알파벳: {result_text}")
                except Exception as e:
                    print("예측 오류:", e)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
