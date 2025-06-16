from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def generate_frames():
    # 카메라 열기 시도
    for cam_index in range(5):  # 0~4까지 시도
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"✅ /dev/video{cam_index} 열기 성공")
            break
        else:
            print(f"❌ /dev/video{cam_index} 열기 실패")
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        print("❌ 카메라 전혀 열리지 않음. 서버 중단")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ 프레임 읽기 실패 또는 None")
                continue

            print(f"✅ 프레임 읽기 성공, shape: {frame.shape}, dtype: {frame.dtype}")

            # 강제 사이즈 조정
            frame = cv2.resize(frame, (640, 480))

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ 프레임 인코딩 실패")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except GeneratorExit:
        print("🛑 스트리밍 중단 (클라이언트 연결 종료)")
    finally:
        cap.release()
        print("✅ 카메라 자원 해제 완료")

@app.route('/')
def index():
    return "<h1>디버깅용 웹캠 스트리밍</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

