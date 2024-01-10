import time
from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import tf.transformations as tf_trans
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np
import cv2
import open3d as o3d
import pyransac3d as pyrsc

# from scripts.mp_segmenter import App
from scripts.mp_seg_onnx import App
from scripts.create_pcd import CreatePointCloud
#from scripts.crf import CRFSegmentationRefiner
from scripts.utils import HairAngleCalculator
from scripts.ros_utils import *
from scipy.special import comb

BLACK_COLOR = (0, 0, 0) # black

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.opt = MyBaseOptions().parse()

        self.camera_num = self.opt.camera_id
        if self.camera_num == 0:
            self.camera_num = ""
        self.camera_ns = "camera" + str(self.camera_num)

        rospy.loginfo("camera_ns: "+self.camera_ns)

        self.strand_pub = rospy.Publisher("segmented_image"+"/"+str(self.camera_num), Image, queue_size=1)
        self.hair_pub = rospy.Publisher("segmented_hair_image"+"/"+str(self.camera_num), Image, queue_size=1)
        self.depth_pub = rospy.Publisher("masked_depth_image"+"/"+str(self.camera_num), Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("segmented_cloud"+"/"+str(self.camera_num), PointCloud2, queue_size=1)
        self.plane_pub = rospy.Publisher('estimated_plane'+"/"+str(self.camera_num), Marker, queue_size=10)
        self.sphere_pub = rospy.Publisher('estimated_sphere'+"/"+str(self.camera_num), Marker, queue_size=10)
        self.strips_pub = rospy.Publisher('strips'+"/"+str(self.camera_num), MarkerArray, queue_size=1)

        self.rate = rospy.Rate(30)

        self.cv_image = None
        self.cv_depth = None
        self.camera_info = None
        self.points = None

        self.distance = 0.8
        self.create_pcd = None
        # self.refiner = None
        self.sph = pyrsc.Sphere()
        self.mode = "3d_color"
        self.size = 15
        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)
        self.frame_id = self.camera_ns+"_color_optical_frame"

        rospy.Subscriber("/"+self.camera_ns+"/color/image_rect_color", Image, self.image_callback)
        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

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

    def refine_mask_with_depth(self, mask, depth_image, threshold=0.8):
        outliers = depth_image/1000.0 > threshold

        refined_mask = np.copy(mask)
        refined_mask[outliers] = 0

        depth_image[outliers] = 0
        return refined_mask, depth_image

    def ransac(self, points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        # Convert the list of points to a PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        try:
            plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
        except:
            return None, None
        return plane_model, inliers

    def create_hair_strands(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3):
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
            strands = self.approximate_bezier(strands, 50)
            np.random.seed(42)
            color_list = np.random.randint(255, size=(len(strands), 3))
            for i, path in enumerate(strands):
                # color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
                color = tuple(color_list[i].tolist())

                for point in path:
                    cv2.circle(img_edge, (point[0], point[1]), 3, (color), -1)

        return img_edge, strands

    def create_hair_strands_gabor(self, orientation_map, n_strands=100, strand_length=10, distance=5):
        """가상의 헤어 스트랜드를 생성합니다."""
        strands = []
        for _ in range(n_strands):
            # 시작점을 임의로 선택합니다.
            start_x = np.random.randint(0, orientation_map.shape[1])
            start_y = np.random.randint(0, orientation_map.shape[0])
            strand = [(start_x, start_y)]

            for _ in range(strand_length - 1):
                x, y = strand[-1]
                direction = orientation_map[y % orientation_map.shape[0], x % orientation_map.shape[1]]
                dx, dy = distance*np.cos(direction), distance*np.sin(direction)
                next_x, next_y = (x + dx).astype(np.uint8), (y + dy).astype(np.uint8)

                if next_x >= orientation_map.shape[1] or next_y >= orientation_map.shape[0]:
                    continue
                strand.append((next_x, next_y))

            strands.append(strand)
        return strands

    def visualize_hair_strands(self, visual_map, strands):
        for strand in strands:
            for point in strand:
                x, y = point
                cv2.circle(visual_map, (x, y), 3, (255, 255, 255), -1)
        return visual_map

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

    def main(self):
        while not self.camera_info:
            continue
        print("received camera_info... ")
        self.create_pcd = CreatePointCloud(self.camera_info)
        # self.refiner = CRFSegmentationRefiner()
        print("start masked pcd publishing")
        while not rospy.is_shutdown():
            if self.cv_image is not None and self.cv_depth is not None:
                self.update(self.cv_image)
                time_now = rospy.Time.now()
                if self.output_image is not None:
                    # self.output_image = self.refiner.refine(self.cv_image, self.output_image)
                    masked_depth = self.apply_depth_mask(self.cv_depth, self.output_image)
                    self.output_image, masked_depth = self.refine_mask_with_depth(self.output_image, masked_depth, self.distance)

                    if self.mode == "strip":
                        strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                        # create_and_publish_strips_markers(self.strips_pub, self.frame_id, xyz_strips)
                        # strand_rgb, strands = self.create_hair_strands(strand_rgb, self.output_image, angle_map.to("cpu").numpy().copy(), W=self.size, n_strands=50, strand_length=50, distance=10)
                        # strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)
                    if self.mode == "gabor":
                        strand_rgb, orientation = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                        strands = self.create_hair_strands_gabor(orientation)
                        strand_rgb = self.visualize_hair_strands(strand_rgb, strands)
                    if self.mode == "color":
                        strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                    if self.mode == "3d_color":
                        strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)

                    # strand_map = img2strand(self.opt, self.cv_image, self.output_image)
                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

                    masked_depth_msg= self.make_depth_msg(masked_depth, time_now)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                        ros_image.header = Header(stamp=time_now)
                        points, header, fields= self.create_pcd.create_point_cloud(strand_rgb, self.cv_depth, self.output_image, time_now)
                        if len(points) == 0:
                            continue
                        indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
                        closest_cloud = points[indices]

                        # if len(closest_cloud[:,:3]) > 4:
                        #     center, radius, sph_inliers = self.sph.fit(closest_cloud[:,:3].astype(np.float32),thresh=0.05, maxIteration=100)
                        # if (center is not None) and (radius is not None):
                        #     sph_msg = create_sphere_marker(center, radius, self.frame_id)
                        #     self.sphere_pub.publish(sph_msg)
                        estimated_plane, plane_inliers = self.ransac(closest_cloud[:,:3][::4], 0.3, 3, 50)
                        if (estimated_plane is not None) and (plane_inliers is not None):
                            plane_msg, plane_inliers_msg = create_plane_and_inliers_markers(estimated_plane, plane_inliers, closest_cloud[:,:3], (0.5, 0.5, 0.001), frame_id=self.frame_id)
                            self.plane_pub.publish(plane_msg)

                        largest_cloud_msg = pc2.create_cloud(header, fields, closest_cloud)

                        if largest_cloud_msg is not None:
                            self.cloud_pub.publish(largest_cloud_msg)
                        self.depth_pub.publish(masked_depth_msg)
                        self.strand_pub.publish(ros_image)
                        self.hair_pub.publish(self.apply_hair_mask(self.cv_image, self.output_image, time_now))

                    except CvBridgeError as e:
                        rospy.logerr(e)
            self.rate.sleep()
        print("shutdown the program")
if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
