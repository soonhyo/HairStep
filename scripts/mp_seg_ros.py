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
from scripts.utils import HairAngleCalculator

from scipy.special import comb

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

        # self.options = ImageSegmenterOptions(base_options=self.base_options,
        #                                      running_mode=VisionRunningMode.LIVE_STREAM,
        #                                      output_category_mask=True,
        #                                      output_confidence_masks=False,
        #                                      result_callback=self.mp_callback)
        self.options = ImageSegmenterOptions(base_options=self.base_options,
                                             output_category_mask=True)

        self.segmenter = ImageSegmenter.create_from_options(self.options)

        self.latest_time_ms = 0

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            print("no update")
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        segmentation_result = self.segmenter.segment(mp_image)
        self.mp_callback(segmentation_result, mp_image)
        # self.segmenter.segment_async(mp_image, t_ms)
        # self.latest_time_ms = t_ms

    # def mp_callback(self, segmentation_result: List[mp.Image], rgb_image: mp.Image, timestamp_ms: int):
    def mp_callback(self, segmentation_result, rgb_image):

        category_mask = segmentation_result.category_mask
        # confidence_mask = segmentation_result.confidence_mask

        image_data = rgb_image.numpy_view()

        fg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        bg_image[:] = BLACK_COLOR[0]

        condition1 = category_mask.numpy_view() == 1 # hair
        condition2 = category_mask.numpy_view() == 3
        condition3 = (category_mask.numpy_view() == 1) | (category_mask.numpy_view() == 2) | (category_mask.numpy_view() == 3)
        # condition3 = category_mask.numpy_view() != 0

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
            self.output_image_human_color = self.output_image_human[:,:,np.newaxis] * image_data[:,:,::-1]

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
        cap.release()
        cv2.destroyAllWindows()

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("segmented_image", Image, queue_size=1)
        self.opt = MyBaseOptions().parse()
        self.rate = rospy.Rate(30)
        self.cv_image = None
        self.mode = "nn"
        self.size = 15

        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)

        rospy.Subscriber("/camera/color/image_rect_color/rotated_image", Image, self.image_callback)
        # rospy.Subscriber("/usb_cam/image_raw/rotated_image", Image, self.image_callback)

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def bezier_curve(self, points, n_points=10):
        """
        베지어 곡선을 생성하는 함수.
        :param points: np.array 형태의 제어점들.
        :param n_points: 생성할 곡선의 점 개수.
        :return: 베지어 곡선을 이루는 점들의 배열.
        """
        n = len(points) - 1
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, points.shape[1]))
        for i in range(n + 1):
            binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(binom, points[i])  # 변경된 부분
        return curve.astype(np.int16)

    def approximate_bezier(self, strands, n_points=10):
        """
        스트랜드를 베지어 곡선으로 근사화하는 함수.
        :param strands: np.array 형태의 점들을 포함하는 스트랜드.
        :param n_points: 생성할 곡선의 점 개수.
        :return: 근사화된 베지어 곡선.
        """
        bezier_strands = []
        for strand in strands:
            bezier_strands.append(self.bezier_curve(strand, n_points))
        return np.asarray(bezier_strands)


    def create_hair_strands(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3, gradiation=3):
        """가상의 헤어 스트랜드를 생성합니다."""
        strands = []
        start_x = np.linspace(0, image.shape[1]-1, n_strands)
        start_y = np.linspace(0, image.shape[0]-1, n_strands)

        for x in start_x:
            for y in start_y:
                if hair_mask[int(y), int(x)] > 0:
                    _path = self.hair_angle_calculator.cal_flow_path([int(y), int(x)], hair_mask, image, distance, W, strand_length, angle_map)
                    if len(_path) > 0:
                        strands.append(np.asarray(_path))

        # strands = np.asarray(strands)
        # strands = strands.astype(np.int16)
        img_edge = image.astype(np.uint8) * hair_mask[:,:,np.newaxis] * 255

        if len(strands) > 0:
            strands = self.approximate_bezier(strands, strand_length)
            np.random.seed(42)
            color_list = np.random.randint(255, size=(len(strands), 3))
            for i, path in enumerate(strands):
                # color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))

                for j, point in enumerate(path):

                    color = tuple((color_list[i]-j*gradiation).tolist()) # -j*n is for gradiation

                    cv2.circle(img_edge, (point[0], point[1]), 1, (color), -1)

        return img_edge, strands


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

                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB) #
                    strand_rgb, strands = self.create_hair_strands(np.zeros_like(self.cv_image), self.output_image, angle_map, W=1, n_strands=20, strand_length=50, distance=5)
                    strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)
                    # strand_rgb = cv2.rotate(strand_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "rgb8")
                    ros_image.header = Header(stamp=rospy.Time.now())
                    self.image_pub.publish(ros_image)
                except CvBridgeError as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
