import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split  # 검증셋 분리를 위해 추가
from tensorflow.keras.callbacks import EarlyStopping  # 조기 종료를 위해 추가

X, y = [], []

# 모든 CSV 파일 읽어서 데이터로 변환
# data 디렉토리가 현재 스크립트와 같은 위치에 있다고 가정합니다.
# 만약 다른 경로라면 "data" 부분을 적절히 수정해야 합니다.
base_dir = os.path.dirname(__file__)
data_directory = os.path.join(base_dir, "data")


print(f"Reading CSV files from: {os.path.abspath(data_directory)}")

if not os.path.isdir(data_directory):
    print(f"Error: Directory '{data_directory}' not found. Please make sure it exists and contains your CSV files.")
    exit()

csv_files_found = False
for file in os.listdir(data_directory):
    if file.endswith(".csv"):
        csv_files_found = True
        file_path = os.path.join(data_directory, file)
        print(f"Processing file: {file_path}")
        try:
            # UTF-16 인코딩으로 CSV 파일 읽기
            df = pd.read_csv(file_path, header=None, encoding='utf-8')

            # 데이터와 라벨 분리
            # 마지막 열을 라벨로 사용, 나머지를 데이터로 사용
            if df.shape[1] > 1:  # 열이 최소 2개 이상 있어야 데이터와 라벨 분리 가능
                X.extend(df.iloc[:, :-1].values.tolist())
                y.extend(df.iloc[:, -1].values.tolist())
                # 라벨이 제대로 읽혔는지 샘플 출력 (디버깅용)
                print(f"  Labels sample from {file}: {df.iloc[:3, -1].unique()}")
            else:
                print(
                    f"  Warning: File {file} has only one column. Skipping this file as it cannot be split into data and label.")

        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")
            print("  Please ensure the file is a valid CSV and encoded in UTF-16.")
            print("  If it's UTF-16LE or UTF-16BE, you might need to specify 'utf-16-le' or 'utf-16-be'.")
            continue  # 문제가 있는 파일은 건너뛰고 계속 진행

if not csv_files_found:
    print(f"No CSV files found in '{data_directory}'. Please check the directory and file extensions.")
    exit()

if not X or not y:
    print("No data was successfully loaded. Exiting.")
    exit()

print(f"\nTotal samples loaded: {len(X)}")
print(f"Unique labels found before encoding: {np.unique(y)}")

X = np.array(X, dtype=np.float32)  # 데이터 타입을 float32로 명시 (Keras에서 종종 권장)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 라벨 인코더 클래스 저장 (le.classes_는 원본 라벨 순서를 가짐)
labels_original_order = le.classes_
print(f"LabelEncoder classes (original labels): {labels_original_order}")
print(f"Number of unique classes: {len(labels_original_order)}")

# 데이터셋을 훈련셋과 검증셋으로 분리
# stratify=y_cat을 사용하여 각 클래스의 비율을 훈련셋과 검증셋에서 유사하게 유지
if len(X) > 1:  # 데이터가 하나 이상 있을 때만 분리 시도
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
else:
    print("Not enough data to create a validation set. Using all data for training.")
    X_train, y_train_cat = X, y_cat
    X_val, y_val_cat = None, None  # 검증셋 없음

# 모델 구성
# 입력층의 크기는 X_train.shape[1] (특징의 개수)
# 출력층의 뉴런 수는 y_cat.shape[1] (클래스의 개수)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')  # 출력층 뉴런 수를 y_cat.shape[1]로 변경
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 모델 학습
# 검증 데이터가 있을 경우에만 callbacks와 validation_data를 사용
if X_val is not None and y_val_cat is not None:
    history = model.fit(X_train, y_train_cat, epochs=50, batch_size=16,
                        validation_data=(X_val, y_val_cat),
                        callbacks=[early_stopping])
else:
    history = model.fit(X_train, y_train_cat, epochs=50, batch_size=16)

# 모델과 라벨 저장
output_model_dir = os.path.join(base_dir, "model")  # <-- 현재 위치 기준 model/ 폴더

os.makedirs(output_model_dir, exist_ok=True)

model.save(os.path.join(output_model_dir, "asl_model.h5"))
np.save(os.path.join(output_model_dir, "asl_labels.npy"), labels_original_order)

print(f"\n모델과 라벨이 '{output_model_dir}' 디렉토리에 저장되었습니다.")
print("labels.npy 에는 다음 라벨들이 저장되었습니다:", labels_original_order)

