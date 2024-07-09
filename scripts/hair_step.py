import time

import rospy
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import tf.transformations as tf_trans
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from scipy.special import comb

from my_img2strand import img2strand, compute_3d_orientation_map, visualize_orientation_map
from create_pcd import CreatePointCloud
from crf import CRFSegmentationRefiner
from utils import HairAngleCalculator
from ros_utils import *
from normal_map import compute_normal_map


class RosApp():
    def __init__(self):
        rospy.init_node('img2strand_node', anonymous=True)
        self.bridge = CvBridge()

        self.camera_ns = rospy.get_param("camera_ns", "camera")

        rospy.loginfo("camera_ns: "+self.camera_ns)

        self.strand_pub = rospy.Publisher("/strand_image", Image, queue_size=1)
        self.normal_pub = rospy.Publisher("/normal_image", Image, queue_size=1)
        self.ori3d_pub = rospy.Publisher("/orientation_image", Image, queue_size=1)

        self.rate = rospy.Rate(30)

        self.cv_image = None
        self.cv_depth = None
        self.cv_mask = None
        self.points = None
        self.camera_info = None

        self.mode = "nn"
        self.size = 15
        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)

        self.frame_id = self.camera_ns+"_color_optical_frame"

        rospy.Subscriber("/segmented_image", Image, self.image_callback)
        rospy.Subscriber("/segmented_depth", Image, self.depth_callback)
        rospy.Subscriber("/segmented_mask", Image, self.mask_callback)

        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

        self.output_image = None

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.stamp = data.header.stamp
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def mask_callback(self, data):
        try:
            self.cv_mask = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def camera_info_callback(self, data):
        self.camera_info = data

    def create_hair_strands(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3, gradiation=3):
        strands = []
        start_x = np.linspace(0, image.shape[1]-1, n_strands)
        start_y = np.linspace(0, image.shape[0]-1, n_strands)

        for x in start_x:
            for y in start_y:
                if hair_mask[int(y), int(x)] > 0:
                    _path = self.hair_angle_calculator.cal_flow_path([int(y), int(x)], hair_mask, image, distance, W, strand_length, angle_map)
                    if len(_path) > 0:
                        strands.append(np.asarray(_path))

        img_edge = image.astype(np.uint8) * hair_mask[:,:,np.newaxis] * 255

        if len(strands) > 0:
            strands = self.approximate_bezier(strands, strand_length)
            np.random.seed(42)
            color_list = np.random.randint(255, size=(len(strands), 3))
            for i, strand in enumerate(strands):
                # color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
                color = tuple((color_list[i]).tolist()) # -j*n is for gradiation
                for j in range(len(strand) - 1):
                    cv2.line(img_edge, tuple(strand[j].astype(int)), tuple(strand[j+1].astype(int)), color, 3)

        return img_edge, strands

    def visualize_hair_strands(self, visual_map, strands):
        for strand in strands:
            for point in strand:
                x, y = point
                cv2.circle(visual_map, (x, y), 3, (255, 255, 255), -1)
        return visual_map

    def bezier_curve(self, points, n_points=10):
        n = len(points) - 1
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, points.shape[1]))
        for i in range(n + 1):
            binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(binom, points[i])
        return curve.astype(np.int16)

    def approximate_bezier(self, strands, n_points=10):
        bezier_strands = []
        for strand in strands:
            bezier_strands.append(self.bezier_curve(strand, n_points))
        return np.asarray(bezier_strands)

    def main(self):
        while not rospy.is_shutdown():
            if self.cv_depth is None or self.cv_image is None or self.cv_mask is None:
                continue
            normal_vis, normal_map = compute_normal_map(self.cv_depth, 7)
            normal_image = self.bridge.cv2_to_imgmsg(normal_vis, "bgr8")
            normal_image.header = Header(stamp=self.stamp)
            self.normal_pub.publish(normal_image)

            if self.mode == "3d_color":
                strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.cv_mask, self.cv_depth, self.camera_info)
            if self.mode == "nn":
                strand_map, angle_map = img2strand("./img2strand.pth",self.cv_image, self.cv_mask)
                strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

                strand_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                strand_image.header = Header(stamp=self.stamp)
                self.strand_pub.publish(strand_image)

                # strand_rgb, strands = self.create_hair_strands(np.zeros_like(self.cv_image), self.cv_mask, angle_map, W=1, n_strands=50, strand_length=20, distance=5)
                # hair_image = self.cv_image * (self.cv_mask[:,:,np.newaxis]/255)
                # strand_rgb = cv2.addWeighted(strand_rgb, 0.5, strand_rgb_large, 0.5, 2.2)
                # strand_rgb = cv2.addWeighted(hair_image.astype(np.uint8), 0.5, strand_rgb, 0.5, 2.2)

                # strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)

                orientation_3d_map =  compute_3d_orientation_map(normal_map, angle_map, self.cv_mask)
                orientation_3d_vis = visualize_orientation_map(orientation_3d_map.to("cpu").numpy())

                orientation_3d_image = self.bridge.cv2_to_imgmsg(orientation_3d_vis, "bgr8")
                orientation_3d_image.header = Header(stamp=self.stamp)
                self.ori3d_pub.publish(orientation_3d_image)

            self.rate.sleep()
if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
