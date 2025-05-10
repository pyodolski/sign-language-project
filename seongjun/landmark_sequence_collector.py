from collections import deque

SEQ_LENGTH = 30
sequence = deque(maxlen=SEQ_LENGTH)

def preprocess_landmarks(landmarks):
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])
    return data

def update_sequence(landmarks):
    sequence.append(preprocess_landmarks(landmarks))
    return list(sequence)
