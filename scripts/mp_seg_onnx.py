import cv2
import numpy as np
import time
from typing import List

import onnxruntime as ort

from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

BG_COLOR = (192, 192, 192) # gray
BLACK_COLOR = (0, 0, 0) # black
MASK_COLOR = (255, 255, 255) # white
BODY_COLOR = (0, 255, 0) # green
FACE_COLOR = (255, 0, 0) # red
CLOTHES_COLOR = (255, 0, 255) # purple

# 0 - background
# 1 - hair
# 2 - body-skin
# 3 - face-skin
# 4 - clothes
# 5 - others (accessories)


class App:
    def __init__(self):
        self.output_image = None

        # 카테고리별 색상 정의 (0부터 5까지의 카테고리)
        self.colors = [
            (0, 255, 0),   # 카테고리 0: 녹색
            (0, 0, 255),   # 카테고리 1: 빨간색
            (255, 0, 0),   # 카테고리 2: 파란색
            (255, 255, 0), # 카테고리 3: 청록색
            (255, 0, 255), # 카테고리 4: 자홍색
            (0, 255, 255)  # 카테고리 5: 노란색
        ]

        # ONNX 모델 로드
        # self.session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
        self.session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])

    def preprocess(self, frame):
        frame = cv2.resize(frame, (256,256))
        frame = frame.reshape(1,256,256,3)
        frame = frame.astype(np.float32) / 255.0  # 정규화
        return frame

    def overlay_mask(self, image, mask, colors):
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # 마스크를 이미지 위에 오버레이
        for category in range(len(colors)):
            category_mask = mask_resized == category
            image[category_mask] = colors[category]
        return image

    def get_target_mask(self, image, mask):
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        fg_image = np.zeros(mask.shape[:2], dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(mask.shape[:2], dtype=np.uint8)
        bg_image[:] = BLACK_COLOR[0]

        condition = mask == 1 # hair
        self.output_image = np.where(condition, fg_image, bg_image)

    def update(self, image):
        # 이미지 전처리
        input_data = self.preprocess(image)

        # 모델 추론
        inputs = {self.session.get_inputs()[0].name: input_data}
        outputs = self.session.run(None, inputs)[0]

        # 추론 결과 처리 (카테고리 마스크를 사용하여 오버레이)
        mask = np.argmax(outputs, axis=-1).squeeze()
        self.overlay_masks = self.overlay_mask(image.copy(), mask, self.colors)
        self.get_target_mask(image, mask)

if __name__=="__main__":
    app = App()
    app.main()
