import tensorflow as tf

# TensorFlow Lite 모델 파일 로드
tflite_model_path = 'selfie_multiclass_256x256.tflite'

import tensorflow as tf

# TensorFlow Lite 모델 파일 로드
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
operations = interpreter.get_tensor_details()


# 특정 노드 번호 확인
node_number = 175
if node_number < len(operations):
    node_details = operations[node_number]
    print(f"Details of node #{node_number}:")
    print(node_details)
else:
    print(f"Node number {node_number} is out of range. The model has {len(operations)} nodes.")
