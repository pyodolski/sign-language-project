import cv2  # OpenCV 라이브러리: 영상 처리 및 웹캠 캡처용
import mediapipe as mp  # MediaPipe 라이브러리: 손 인식 모듈 사용

mp_hands = mp.solutions.hands  # MediaPipe의 손 인식 기능을 불러옴
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
# 손 인식 모델 생성: 실시간 모드, 최대 1개의 손 추적, 최소 신뢰도 설정

mp_draw = mp.solutions.drawing_utils  # 손 관절을 화면에 시각화하기 위한 도구

cap = cv2.VideoCapture(0)  # 기본 웹캠(카메라 0번)을 열어서 영상 스트림 시작

while True:  # 무한 루프: 실시간 영상 프레임 반복 처리
    ret, frame = cap.read()  # 프레임을 읽어오기 (ret은 성공 여부, frame은 현재 이미지 프레임)
    if not ret:
        break  # 프레임을 못 받았으면 종료

    frame = cv2.flip(frame, 1)  # 영상 좌우 반전 (거울처럼 보이도록)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    # 프레임 색상 포맷 변환: OpenCV는 기본 BGR, MediaPipe는 RGB 요구 → BGR → RGB로 변환

    result = hands.process(rgb)  
     # 손 인식 모델에 RGB 영상 입력하여 손 관절 추출 수행 (result에 결과 저장)

    if result.multi_hand_landmarks:  # 손 랜드마크가 하나라도 감지되면
        for hand_landmarks in result.multi_hand_landmarks:  # 감지된 손들에 대해 반복, hand_landmarks: 21개 관절의 위치 정보 포함
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 각 손의 관절과 뼈 연결선을 영상 위에 그림, (HAND_CONNECTIONS는 관절 간 연결 관계)

    cv2.imshow("Hand Tracking", frame)  # 인식 결과를 화면에 출력
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 키보드 'q'를 누르면 종료
        break

cap.release()  # 카메라 자원 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

# 이 코드는 웹캠을 통해 실시간 영상 스트림을 받아오고,
# MediaPipe를 이용해 손의 21개 관절 좌표를 인식한 뒤,
# 화면에 손 구조를 시각화하여 실시간으로 보여준다.
# 사용자가 'q'를 누르면 프로그램이 종료된다.
