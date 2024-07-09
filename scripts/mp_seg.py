import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import time
from typing import List

from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


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
        # Create an FaceLandmarker object.
        # self.base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite',
        #                                        delegate=python.BaseOptions.Delegate.GPU)

        self.base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
                                               delegate=python.BaseOptions.Delegate.CPU)
        # self.base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite',
        #                                        delegate=python.BaseOptions.Delegate.GPU)


        self.options = ImageSegmenterOptions(base_options=self.base_options,
                                             running_mode=VisionRunningMode.LIVE_STREAM,
                                             output_category_mask=True,
                                             output_confidence_masks=False,
                                             result_callback=self.mp_callback)

        self.segmenter = ImageSegmenter.create_from_options(self.options)

        self.latest_time_ms = 0

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            print("no update")
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.segmenter.segment_async(mp_image, t_ms)
        self.latest_time_ms = t_ms

    def mp_callback(self, segmentation_result: List[mp.Image], rgb_image: mp.Image, timestamp_ms: int):
        category_mask = segmentation_result.category_mask
        # confidence_mask = segmentation_result.confidence_mask

        image_data = rgb_image.numpy_view()
        fg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        bg_image[:] = BLACK_COLOR[0]

        condition1 = category_mask.numpy_view() == 1 # hair
        # print(np.unique(category_mask.numpy_view()))
        # confidence = confidence_mask.numpy_view()
        # print(confidence)
        self.output_image = np.where(condition1, fg_image, bg_image)

        # self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2RGB)

    def main(self):
        cap = cv2.VideoCapture(0)
        opt = MyBaseOptions().parse()

        rate = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # frame = cv2.resize(frame, (512, 512)) # ( X, Y)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            # rgb_img.flags.writeable = False

            # Detect face landmarks from the input image.
            self.update(rgb_img)

            time.sleep(1/rate)

            if self.output_image is None:
                continue

            strand_map = img2strand(opt, rgb_img, self.output_image)
            strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)
            # print(strand_rgb)
            # self.output_image = None
            result = np.vstack((frame, strand_rgb))
            cv2.imshow('MediaPipe FaceMesh', result)
            #Exit if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
11;rgb:3030/0a0a/2424cap.release()
        cv2.destroyAllWindows()
if __name__=="__main__":
    app = App()
    app.main()
