import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 설정
LABELS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㅏ', 'ㅑ', 'space', 'end']  # 필요한 자모 추가 가능
SAVE_DIR = '../data'  # 프로젝트 루트 기준으로 data 폴더

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 디렉토리 준비
os.makedirs(SAVE_DIR, exist_ok=True)
for label in LABELS:
    open(os.path.join(SAVE_DIR, f'{label}.csv'), 'a').close()

print("수어 데이터를 수집하려면 손 모양을 만든 후 라벨 키를 누르세요.")
print("사용 가능한 라벨:", LABELS)
print("저장 중: s / 종료: q")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 21개 좌표 (x, y, z) 추출
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # 총 63차원

            # 현재 프레임에서 키 입력 기다림
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord('s'):
                print("저장할 라벨 키를 입력하세요:")
                label_key = cv2.waitKey(0) & 0xFF
                label = chr(label_key)
                if label in LABELS:
                    print(f"[저장됨] 라벨: {label}")
                    with open(os.path.join(SAVE_DIR, f'{label}.csv'), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(coords)
                else:
                    print(f"❌ 지원되지 않는 라벨 키: {label}")

    cv2.imshow("KSL 좌표 수집기", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
