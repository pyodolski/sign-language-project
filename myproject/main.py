from flask import Flask, render_template_string, request, redirect, url_for
import RPi.GPIO as GPIO
import time
import atexit

app = Flask(__name__)

SERVO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# 현재 각도 상태
current_angle = 0

def move_servo_to(angle):
    global current_angle
    angle = max(0, min(180, angle))  # 범위 제한
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo.ChangeDutyCycle(0)
    current_angle = angle

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_angle
    if request.method == 'POST':
        try:
            target = int(request.form.get('angle', "0"))
            move_servo_to(target)
        except ValueError:
            pass

    return render_template_string('''
        <h2>서보 모터 제어 (절대 각도 + 초기화)</h2>
        <form method="post">
            <label>목표 각도 (0~180):</label>
            <input type="range" name="angle" min="0" max="180" value="{{ current_angle }}" oninput="this.nextElementSibling.value = this.value">
            <output>{{ current_angle }}</output>도
            <button type="submit">이동</button>
        </form>
        <form action="/reset" method="post" style="margin-top:1em;">
            <button type="submit">초기화 (0도)</button>
        </form>
        <p>현재 각도: {{ current_angle }}도</p>
        <script>
            const slider = document.querySelector("input[type='range']");
            const output = document.querySelector("output");
            slider.addEventListener("input", function() {
                output.textContent = this.value;
            });
        </script>
    ''', current_angle=current_angle)

@app.route('/reset', methods=['POST'])
def reset_servo():
    move_servo_to(0)
    return redirect(url_for('index'))

atexit.register(servo.stop)
atexit.register(GPIO.cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)

