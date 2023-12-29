# import tensorflow as tf

# # TensorFlow Lite 모델 로드
# interpreter = tf.lite.Interpreter(model_path="selfie_multiclass_256x256.tflite")
import tflite2onnx

tflite_path = 'hair_segmenter.tflite'
onnx_path = 'hair_segmenter.onnx'

tflite2onnx.convert(tflite_path, onnx_path)
