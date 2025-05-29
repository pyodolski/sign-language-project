import tensorflow as tf

# 원래 Keras 모델(.h5) 경로
keras_model_path = 'model/ksl_model.h5'

# 변환될 .tflite 파일 경로
tflite_model_path = 'model/ksl_model.tflite'

# Keras 모델 로드
model = tf.keras.models.load_model(keras_model_path)
print("✅ Keras 모델 로드 성공")

# TFLiteConverter 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (선택) 최적화 옵션 추가 → 용량 축소 및 속도 향상
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# (선택) float16 양자화 → 라즈베리파이처럼 자원이 적은 장치에서 빠름
converter.target_spec.supported_types = [tf.float16]

# 변환 수행
tflite_model = converter.convert()
print("✅ TFLite 모델 변환 성공")

# .tflite 파일로 저장
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ .tflite 모델 저장 완료 → {tflite_model_path}")
