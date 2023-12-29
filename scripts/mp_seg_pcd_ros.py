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
import open3d as o3d


# from scripts.mp_segmenter import App
from scripts.mp_seg_onnx import App
from scripts.create_pcd import CreatePointCloud
from scripts.crf import CRFSegmentationRefiner
from scripts.utils import HairAngleCalculator

BLACK_COLOR = (0, 0, 0) # black


class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.strand_pub = rospy.Publisher("segmented_image", Image, queue_size=1)
        self.hair_pub = rospy.Publisher("segmented_hair_image", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("masked_depth_image", Image, queue_size=1)

        self.cloud_pub = rospy.Publisher("segmented_cloud", PointCloud2, queue_size=1)
        self.opt = MyBaseOptions().parse()
        self.rate = rospy.Rate(30)
        self.cv_image = None
        self.cv_depth = None
        self.camera_info = None
        self.points = None

        self.distance = 1.5
        self.create_pcd = None
        self.refiner = None

        self.hair_angle_calculator = HairAngleCalculator(15)
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

    def apply_depth_mask(self, depth_image, mask):
        # 마스크에 해당하는 깊이 정보만을 포함하는 깊이 이미지 생성
        masked_depth = np.where(mask == 255, depth_image, 0)
        return masked_depth

    def make_depth_msg(self, depth_image, time_now):
        # ROS 메시지로 변환
        ros_depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")
        ros_depth_msg.header = Header(stamp=time_now)
        return ros_depth_msg

    def apply_hair_mask(self, image, mask, time_now):
        bg_image = np.zeros(image.shape, dtype=np.uint8)

        condition = mask == 255 # hair
        condition_3d = condition[:,:,np.newaxis]
        ros_image = np.where(condition_3d, image, bg_image)

        ros_msg = self.bridge.cv2_to_imgmsg(ros_image, "bgr8")
        ros_msg.header = Header(stamp=time_now)
        return ros_msg

    def camera_info_callback(self, data):
        self.camera_info = data

    def main(self):
        while not self.camera_info:
            continue
        print("received camera_info... ")
        self.create_pcd = CreatePointCloud(self.camera_info)
        self.refiner = CRFSegmentationRefiner()
        print("start masked pcd publishing")
        while not rospy.is_shutdown():
            if self.cv_image is not None and self.cv_depth is not None:
                self.update(self.cv_image)
                time_now = rospy.Time.now()
                if self.output_image is not None:
                    # self.output_image = self.refiner.refine(self.cv_image, self.output_image)
                    masked_depth = self.apply_depth_mask(self.cv_depth, self.output_image)
                    self.output_image = self.refiner.refine_mask_with_depth(self.output_image, masked_depth, self.distance)

                    # strand_map = img2strand(self.opt, self.cv_image, self.output_image)
                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

                    strand_rgb = self.hair_angle_calculator.process_image(self.cv_image, self.output_image)
                    masked_depth_msg= self.make_depth_msg(masked_depth, time_now)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                        ros_image.header = Header(stamp=time_now)
                        self.depth_pub.publish(masked_depth_msg)
                        self.strand_pub.publish(ros_image)
                        self.hair_pub.publish(self.apply_hair_mask(self.cv_image, self.output_image, time_now))

                        points, header, fields= self.create_pcd.create_point_cloud(strand_rgb, self.cv_depth, self.output_image, time_now)
                        if len(points) == 0:
                            continue
                        indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
                        closest_cloud = points[indices]

                        largest_cloud_msg = pc2.create_cloud(header, fields, closest_cloud)
                        if largest_cloud_msg is not None:
                            self.cloud_pub.publish(largest_cloud_msg)
                    except CvBridgeError as e:
                        rospy.logerr(e)
            self.rate.sleep()
        print("shutdown the program")
if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
