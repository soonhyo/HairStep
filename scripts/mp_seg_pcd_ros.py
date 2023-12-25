import time
from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np
import cv2

from scripts.mp_seg import App

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("segmented_image", Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("segmented_cloud", PointCloud2, queue_size=1)
        self.opt = MyBaseOptions().parse()
        self.rate = rospy.Rate(5)
        self.cv_image = None
        self.cv_depth = None
        self.camera_info = None
        rospy.Subscriber("/camera/color/image_rect_color", Image, self.image_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def camera_info_callback(self, data):
        self.camera_info = data

    def create_point_cloud(self, color_image, depth_image, mask, time_now):
        if self.camera_info is None:
            return None

        # 카메라 내부 파라미터 사용
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        points = []
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                if mask[v, u]:
                    z = depth_image[v, u]/1000.0  # 깊이
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    # 이미지의 RGB 값을 가져옴
                    color = color_image[v, u]
                    r, g, b = color[2], color[1], color[0]  # OpenCV는 BGR 순서로 색상을 사용합니다
                    rgba = (0xFF << 24) | (r << 16) | (g << 8) | b  # RGBA 형태로 포맷
                    points.append([x, y, z, rgba])

        # PointField 구조 정의
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1)]

        # PointCloud2 메시지 생성
        header = Header(frame_id='camera_color_optical_frame', stamp=time_now)
        cloud = pc2.create_cloud(header, fields, points)
        return cloud

    def main(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None and self.cv_depth is not None:
                self.update(self.cv_image)
                time_now = rospy.Time.now()
                if self.output_image is not None:
                    strand_map = img2strand(self.opt, self.cv_image, self.output_image)
                    strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                        ros_image.header = Header(stamp=time_now)
                        self.image_pub.publish(ros_image)

                        cloud = self.create_point_cloud(strand_rgb, self.cv_depth, self.output_image, time_now)
                        if cloud is not None:
                            self.cloud_pub.publish(cloud)
                    except CvBridgeError as e:
                        rospy.logerr(e)
            self.rate.sleep()

if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
