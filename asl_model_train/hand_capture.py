import cv2
import mediapipe as mp
import csv
import os

label = input("저장할 글자 (예: ㄱ, ㄴ): ")

# 데이터 저장 경로
os.makedirs("data", exist_ok=True)
csv_file = f"data/{label}.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with open(csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    print("스페이스바 누를 때마다 데이터 저장됩니다. ESC로 종료.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand", image)

        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == 32:  # 스페이스바
            if result.multi_hand_landmarks:
                coords = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y])
                coords.append(label)
                writer.writerow(coords)
                print("저장 완료")

cap.release()
cv2.destroyAllWindows()

