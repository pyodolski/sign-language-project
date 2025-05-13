from collections import deque # deque 자료구조를 불러옴

SEQ_LENGTH = 30 # 시퀀스 길이 설정: 30프레임 (즉, 30장의 연속 손 모양 데이터를 저장)
sequence = deque(maxlen=SEQ_LENGTH) # 고정 길이 큐 생성: 최대 30개까지의 데이터가 저장되며, 초과 시 자동으로 오래된 데이터부터 제거됨

def preprocess_landmarks(landmarks):
    data = [] # 변환된 결과를 저장할 배열 
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z]) # 각 관절의 x, y, z 좌표를 data 리스트에 순서대로 추가
    return data # 1차원 리스트(x, y, z 순으로 21개의 관절 데이터 총 63개)를 반환

def update_sequence(landmarks): # 새로운 랜드마크 데이터를 받아 시퀀스에 누적하고, 현재 시퀀스를 리스트로 반환하는 함수
    sequence.append(preprocess_landmarks(landmarks))  # 새 랜드마크를 전처리 후 큐(sequence)에 추가 (길이가 30을 넘으면 자동으로 앞에서 제거됨)
    return list(sequence) # 현재까지 누적된 30프레임 이하의 데이터를 리스트로 반환 (모델 입력에 사용)

# 이 코드는 실시간으로 인식된 손 관절(landmark) 데이터를
# 30프레임 길이의 시퀀스로 누적하여 예측 모델에 입력할 수 있도록 준비하는 역할을 한다.
# 각 프레임에서 21개 관절의 x, y, z 좌표를 뽑아 63차원 벡터로 만들고,
# 최신 30개의 벡터를 순서대로 유지하며 반환한다.
