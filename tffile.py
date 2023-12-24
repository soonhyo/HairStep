import tensorflow as tf

# TensorFlow Lite 모델 파일 로드
tflite_model_path = 'selfie_multiclass_256x256.tflite'

# TensorFlow Lite GPU 대리자 생성
gpu_delegate_options = {
    'is_precision_loss_allowed': True,  # 정밀도 손실을 허용할지 여부 (성능 향상을 위해)
    'inference_preference': 0,          # 높은 성능 선호
    'inference_priority1': 0,           # 성능 우선
    'inference_priority2': 1,           # 정밀도 다음
    'inference_priority3': 2            # 레이턴시 마지막
}
gpu_delegate = tf.lite.experimental.load_delegate('./libtensorflowlite_gpu_delegate.so', gpu_delegate_options)

# TensorFlow Lite 모델 로드 및 GPU 대리자 설정
interpreter = tf.lite.Interpreter(model_path=tflite_model_path, experimental_delegates=[gpu_delegate])
interpreter.allocate_tensors()


# interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# Interpreter 초기화
# interpreter.allocate_tensors()

# 모델의 입력 및 출력 정보 출력
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:")
for input in input_details:
    print(input)

print("\nOutput Details:")
for output in output_details:
    print(output)

# 모델의 각 레이어 정보 출력
# for i in range(len(interpreter.get_tensor_details())):
#     print("Layer {}: {}".format(i, interpreter.get_tensor_details()[i]))
