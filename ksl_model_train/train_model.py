import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

X, y = [], []

# 현재 파일 위치 기준
base_dir = os.path.dirname(__file__)
data_directory = os.path.join(base_dir, "data")

print(f"CSV 파일을 읽는 경로: {os.path.abspath(data_directory)}")

if not os.path.isdir(data_directory):
    print(f"오류: '{data_directory}' 디렉토리를 찾을 수 없습니다.")
    exit()

csv_files_found = False
for file in os.listdir(data_directory):
    if file.endswith(".csv"):
        csv_files_found = True
        file_path = os.path.join(data_directory, file)
        print(f"파일 처리 중: {file_path}")
        try:
            df = pd.read_csv(file_path, header=None, encoding='utf-8')
            if df.shape[1] > 1:
                X.extend(df.iloc[:, :-1].values.tolist())
                y.extend(df.iloc[:, -1].values.tolist())
                print(f"  {file}의 라벨 샘플: {df.iloc[:3, -1].unique()}")
            else:
                print(f"  경고: {file} 파일은 하나의 열만 가지고 있습니다. 건너뜁니다.")
        except Exception as e:
            print(f"파일 {file_path} 읽기 오류: {e}")
            continue

if not csv_files_found or not X or not y:
    print("사용 가능한 데이터를 찾을 수 없습니다. 종료합니다.")
    exit()

print(f"\n로드된 총 샘플 수: {len(X)}")
print(f"인코딩 전 고유 라벨: {np.unique(y)}")

X = np.array(X, dtype=np.float32)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

labels_original_order = le.classes_
print(f"라벨 클래스: {labels_original_order}")
print(f"클래스 수: {len(labels_original_order)}")

if len(X) > 1:
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
    )
    print(f"\n훈련셋 크기: {X_train.shape[0]}")
    print(f"검증셋 크기: {X_val.shape[0]}")
else:
    X_train, y_train_cat = X, y_cat
    X_val, y_val_cat = None, None

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

if X_val is not None and y_val_cat is not None:
    history = model.fit(X_train, y_train_cat, epochs=50, batch_size=16,
                        validation_data=(X_val, y_val_cat),
                        callbacks=[early_stopping])
else:
    history = model.fit(X_train, y_train_cat, epochs=50, batch_size=16)

# === 모델 저장 ===
output_model_dir = os.path.join(base_dir, "model")  # 현재 디렉토리의 model/ 폴더
os.makedirs(output_model_dir, exist_ok=True)

model.save(os.path.join(output_model_dir, "ksl_model.h5"))
np.save(os.path.join(output_model_dir, "ksl_labels.npy"), labels_original_order)

print(f"\n모델과 라벨이 '{output_model_dir}' 디렉토리에 저장되었습니다.")
print("ksl_labels.npy 라벨 목록:", labels_original_order)
