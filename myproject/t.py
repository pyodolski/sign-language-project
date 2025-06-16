import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ /dev/video{i} 열림 성공")
        ret, frame = cap.read()
        if ret:
            print(f"✅ 프레임 읽기 성공 from /dev/video{i}")
            cv2.imwrite(f"test_capture_{i}.jpg", frame)
            print(f"✅ test_capture_{i}.jpg 저장 완료")
        else:
            print(f"❌ 프레임 읽기 실패 from /dev/video{i}")
        cap.release()
    else:
        print(f"❌ /dev/video{i} 열기 실패")

