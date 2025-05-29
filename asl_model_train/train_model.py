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


print(f"CSV 파일을 읽는 경로: {os.path.abspath(data_directory)}")

if not os.path.isdir(data_directory):
    print(f"오류: '{data_directory}' 디렉토리를 찾을 수 없습니다. 디렉토리가 존재하고 CSV 파일이 포함되어 있는지 확인해주세요.")
    exit()

csv_files_found = False
for file in os.listdir(data_directory):
    if file.endswith(".csv"):
        csv_files_found = True
        file_path = os.path.join(data_directory, file)
        print(f"파일 처리 중: {file_path}")
        try:
            # UTF-16 인코딩으로 CSV 파일 읽기
            df = pd.read_csv(file_path, header=None, encoding='utf-8')

            # 데이터와 라벨 분리
            # 마지막 열을 라벨로 사용, 나머지를 데이터로 사용
            if df.shape[1] > 1:  # 열이 최소 2개 이상 있어야 데이터와 라벨 분리 가능
                X.extend(df.iloc[:, :-1].values.tolist())
                y.extend(df.iloc[:, -1].values.tolist())
                # 라벨이 제대로 읽혔는지 샘플 출력 (디버깅용)
                print(f"  {file}의 라벨 샘플: {df.iloc[:3, -1].unique()}")
            else:
                print(
                    f"  경고: {file} 파일은 하나의 열만 가지고 있습니다. 데이터와 라벨로 분리할 수 없어 건너뜁니다.")

        except Exception as e:
            print(f"파일 {file_path} 읽기 또는 처리 중 오류 발생: {e}")
            print("  파일이 유효한 CSV이고 UTF-16으로 인코딩되어 있는지 확인해주세요.")
            print("  UTF-16LE나 UTF-16BE인 경우 'utf-16-le' 또는 'utf-16-be'를 지정해야 할 수 있습니다.")
            continue  # 문제가 있는 파일은 건너뛰고 계속 진행

if not csv_files_found:
    print(f"'{data_directory}'에서 CSV 파일을 찾을 수 없습니다. 디렉토리와 파일 확장자를 확인해주세요.")
    exit()

if not X or not y:
    print("데이터를 성공적으로 로드하지 못했습니다. 종료합니다.")
    exit()

print(f"\n로드된 총 샘플 수: {len(X)}")
print(f"인코딩 전 발견된 고유 라벨: {np.unique(y)}")

X = np.array(X, dtype=np.float32)  # 데이터 타입을 float32로 명시 (Keras에서 종종 권장)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 라벨 인코더 클래스 저장 (le.classes_는 원본 라벨 순서를 가짐)
labels_original_order = le.classes_
print(f"LabelEncoder 클래스 (원본 라벨): {labels_original_order}")
print(f"고유 클래스 수: {len(labels_original_order)}")

# 데이터셋을 훈련셋과 검증셋으로 분리
# stratify=y_cat을 사용하여 각 클래스의 비율을 훈련셋과 검증셋에서 유사하게 유지
if len(X) > 1:  # 데이터가 하나 이상 있을 때만 분리 시도
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
    )
    print(f"\n훈련셋 크기: {X_train.shape[0]}")
    print(f"검증셋 크기: {X_val.shape[0]}")
else:
    print("검증셋을 만들기에 데이터가 충분하지 않습니다. 모든 데이터를 훈련에 사용합니다.")
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

