#!/usr/bin/env python3

import cv2
import numpy as np
import onnxruntime as ort

def preprocess(frame):
    # 이미지 전처리 로직 (모델에 맞게 조정 필요)
    frame = cv2.resize(frame, (256,256))
    frame = frame.reshape(1,256,256,3)
    frame = frame.astype(np.float32) / 255.0  # 정규화
    # frame = np.transpose(frame, (2, 0, 1))  # 채널 변경 (HWC to CHW)
    # frame = np.expand_dims(frame, axis=0)  # 배치 차원 추가
    return frame
def overlay_mask(image, mask, colors):
    # 마스크의 크기를 이미지의 크기와 일치하도록 조정
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # 마스크를 이미지 위에 오버레이
    for category in range(len(colors)):
        category_mask = mask_resized == category
        image[category_mask] = colors[category]
    return image

def main():
     # 카테고리별 색상 정의 (0부터 5까지의 카테고리)
    colors = [
        (0, 255, 0),   # 카테고리 0: 녹색
        (0, 0, 255),   # 카테고리 1: 빨간색
        (255, 0, 0),   # 카테고리 2: 파란색
        (255, 255, 0), # 카테고리 3: 청록색
        (255, 0, 255), # 카테고리 4: 자홍색
        (0, 255, 255)  # 카테고리 5: 노란색
    ]

    # ONNX 모델 로드
    # session = ort.InferenceSession('model.onnx')
    session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    while True:
        # 웹캠으로부터 이미지 캡처
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        input_data = preprocess(frame)

        # 모델 추론
        inputs = {session.get_inputs()[0].name: input_data}
        outputs = session.run(None, inputs)[0]

        # 추론 결과 처리 (카테고리 마스크를 사용하여 오버레이)
        mask = np.argmax(outputs, axis=-1).squeeze()
        overlayed_image = overlay_mask(frame.copy(), mask, colors)

        cv2.imshow('Webcam', overlayed_image)

        # 'q'를 누르면 반복문 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
