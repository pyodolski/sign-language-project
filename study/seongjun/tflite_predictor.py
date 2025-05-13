import numpy as np # NumPy 라이브러리 불러오기: 모델 입력 데이터를 배열로 만들고 형변환할 때 사용
import tflite_runtime.interpreter as tflite # TensorFlow Lite 인터프리터 불러오기: .tflite 모델을 로딩하고 실행하기 위한 핵심 도구

class ASLPredictor:
    def __init__(self, model_path="asl_model.tflite"):
        self.interpreter = tflite.Interpreter(model_path=model_path) # TFLite 인터프리터 생성 (model_path: 사용할 수어 예측 모델 파일 경로)
        self.interpreter.allocate_tensors() # 모델을 실행하기 위한 텐서 메모리 할당
        self.input_details = self.interpreter.get_input_details() # 모델 입력 정보 가져오기 (입력 텐서의 모양, dtype 등)
        self.output_details = self.interpreter.get_output_details() # 모델 출력 정보 가져오기 (출력 텐서의 모양, 위치 등)

    def predict(self, sequence): 
        input_data = np.array(sequence, dtype=np.float32).reshape(1, 30, 63)
        # 시퀀스 데이터를 float32형 NumPy 배열로 변환하고, (1, 30, 63) 모양으로 reshape
        # 1개 샘플, 30프레임, 프레임당 63개 값(x,y,z * 21개 관절)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data) # 변환된 데이터를 모델의 입력 텐서에 설정
        self.interpreter.invoke() # 모델을 실행하여 추론 수행
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # 추론 결과(출력 텐서 값)를 가져와서 첫 번째 결과만 추출 (알파벳 26개 확률 벡터)
        return chr(np.argmax(output) + ord('A')) 
        # 확률이 가장 높은 인덱스를 찾아 해당 인덱스를 알파벳으로 변환 ('A'부터 시작)
        # 예: argmax(output)=0 → 'A', argmax=2 → 'C'


# 이 코드는 사전 학습된 수어 인식 모델(.tflite)을 로딩하고,
# 입력된 손 관절 시퀀스(30프레임 x 63값)를 기반으로 가장 가능성 높은 알파벳(A~Z)을 예측한다.
# NumPy를 사용해 입력 데이터를 준비하고, TFLite 인터프리터를 통해 추론을 수행하며,
# 예측 결과는 알파벳 문자 하나로 반환된다.
