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

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

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
        self.output_image_face = None
        self.output_image_human = None
        self.output_image_human_color = None

        # Create an FaceLandmarker object.
        # self.base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite',
        #                                        delegate=python.BaseOptions.Delegate.GPU)

        self.base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
                                               delegate=python.BaseOptions.Delegate.CPU)

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
        condition2 = category_mask.numpy_view() == 3
        #condition3 = (mask == 1) | (mask == 2) | (mask == 3)
        condition3 = category_mask.numpy_view() != 0

        if np.sum(condition1) == 0:
            self.output_image = bg_image
        else:
            self.output_image = np.where(condition1, fg_image, bg_image)
        if np.sum(condition2) == 0:
            self.output_image_face = bg_image
        else:
            self.output_image_face = np.where(condition2, fg_image, bg_image)
        if np.sum(condition3) == 0:
            self.output_image_human = bg_image
            self.output_image_human_color = image_data[:,:,::-1]
        else:
            self.output_image_human = np.where(condition3, np.ones(image_data.shape[:2], dtype=np.uint8), bg_image)

    def main(self):
        cap = cv2.VideoCapture(0)
        opt = MyBaseOptions().parse()

        rate = 30

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
        cap.release()
        cv2.destroyAllWindows()

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("segmented_image", Image, queue_size=1)
        self.opt = MyBaseOptions().parse()
        self.rate = rospy.Rate(60)
        self.cv_image = None
        rospy.Subscriber("/camera/color/image_rect_color", Image, self.image_callback)

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def main(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                self.update(self.cv_image)
            else:
                continue

            self.rate.sleep()

            if self.output_image is not None:
                try:
                    strand_map, angle_map = img2strand(self.opt, self.cv_image, self.output_image)
                    strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

                    ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                    ros_image.header = Header(stamp=rospy.Time.now())
                    self.image_pub.publish(ros_image)
                except CvBridgeError as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
