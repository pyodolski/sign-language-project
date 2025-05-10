import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

X, y = [], []

# 모든 CSV 파일 읽어서 데이터로 변환
for file in os.listdir("data"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/{file}", header=None)
        X.extend(df.iloc[:, :-1].values.tolist())
        y.extend(df.iloc[:, -1].values.tolist())

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)
y_cat = to_categorical(y)

# 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=50, batch_size=16)

# 모델과 라벨 저장
os.makedirs("model", exist_ok=True)
model.save("model/ksl_model.h5")
np.save("model/labels.npy", le.classes_)
print("모델 저장 완료")

