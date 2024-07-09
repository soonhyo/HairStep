import time
from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import *
from scripts.my_depth import img2depth

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
# from scripts.mp_seg_onnx import App
from scripts.mp_seg_ros import App

from scripts.create_pcd import CreatePointCloud
from scripts.crf import CRFSegmentationRefiner
from scripts.utils import HairAngleCalculator
from scripts.ros_utils import *
from scipy.special import comb

from scripts.normal_map import *

import segmentation_refinement as refine

from sklearn.cluster import KMeans
from scripts.detectors_ros import MP

from std_msgs.msg import Int32
from geometry_msgs.msg import Pose, PoseArray, Quaternion
import tf

BLACK_COLOR = (0, 0, 0) # black

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.opt = MyBaseOptions().parse()

        self.camera_ns = self.opt.camera_name

        rospy.loginfo("camera_ns: "+self.camera_ns)

        self.strand_pub = rospy.Publisher("segmented_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.hair_pub = rospy.Publisher("segmented_hair_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.hair_mask_pub = rospy.Publisher("segmented_hair_mask_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.human_pub = rospy.Publisher("segmented_human_image"+"/"+str(self.camera_ns), Image, queue_size=1)

        self.depth_pub = rospy.Publisher("masked_depth_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.depth_human_pub = rospy.Publisher("masked_human_depth_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.depth_2d_pub = rospy.Publisher("2d_masked_depth_image"+"/"+str(self.camera_ns), Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("segmented_cloud"+"/"+str(self.camera_ns), PointCloud2, queue_size=1)
        
        self.plane_pub = rospy.Publisher('estimated_plane'+"/"+str(self.camera_ns), Marker, queue_size=1)
        self.sphere_pub = rospy.Publisher('estimated_sphere'+"/"+str(self.camera_ns), Marker, queue_size=1)
        self.strips_pub = rospy.Publisher('strips'+"/"+str(self.camera_ns), MarkerArray, queue_size=1)
        self.strand_path_pub = rospy.Publisher('comb/path', PoseArray, queue_size=1)

        self.rate = rospy.Rate(60)

        self.cv_image = None
        self.cv_depth = None
        self.camera_info = None
        self.points = None

        self.distance = 0.9
        self.create_pcd = None
        # self.refiner = None
        self.sph = pyrsc.Sphere()
        self.mode = "strip"
        self.size = 15
        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)
        self.frame_id = self.camera_ns+"_color_optical_frame"

        self.decompressed = self.opt.decompressed
        self.face_detector = MP()

        if self.decompressed:
            rospy.Subscriber("/"+self.camera_ns+"/color/image_rect_color/decompressed", Image, self.image_callback)
            rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/image_raw/decompressed", Image, self.depth_callback)
        else:
            rospy.Subscriber("/"+self.camera_ns+"/color/image_rect_color", Image, self.image_callback)
            rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)
         # task_status 구독
        self.task_status = 0
        rospy.Subscriber("/task_status", Int32, self.task_status_callback)

        # 클러스터 관련 변수
        self.current_cluster_index = 0
        self.clusters_completed = False
        self.waiting_for_task_completion = False

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

    def task_status_callback(self, msg):
        self.task_status = msg.data
        if self.task_status == 0 and self.waiting_for_task_completion:
            self.waiting_for_task_completion = False
            self.current_cluster_index += 1
            if self.current_cluster_index >= len(self.cluster_paths):
                self.current_cluster_index = 0
                self.clusters_completed = True

    def send_next_cluster_path(self):
        if not self.waiting_for_task_completion and not self.clusters_completed:
            strand_posearray = self.cluster_paths[self.current_cluster_index]
            self.strand_path_pub.publish(strand_posearray)
            self.waiting_for_task_completion = True

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

    # def bezier_curve(self, points, n_points=10):
    #     """
    #     베지어 곡선을 생성하는 함수.
    #     :param points: np.array 형태의 제어점들.
    #     :param n_points: 생성할 곡선의 점 개수.
    #     :return: 베지어 곡선을 이루는 점들의 배열.
    #     """
    #     n = len(points) - 1
    #     t = np.linspace(0, 1, n_points)
    #     curve = np.zeros((n_points, points.shape[1]))
    #     for i in range(n + 1):
    #         binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    #         curve += np.outer(binom, points[i])  # 변경된 부분
    #     return curve.astype(np.int16)

    # def approximate_bezier(self, strands, n_points=10):
    #     """
    #     스트랜드를 베지어 곡선으로 근사화하는 함수.
    #     :param strands: np.array 형태의 점들을 포함하는 스트랜드.
    #     :param n_points: 생성할 곡선의 점 개수.
    #     :return: 근사화된 베지어 곡선.
    #     """
    #     bezier_strands = []
    #     for strand in strands:
    #         bezier_strands.append(self.bezier_curve(strand, n_points))
    #     return np.asarray(bezier_strands)

    def extract_points_from_hull(self, depth_image, hull):
        """
        깊이 이미지와 Convex Hull을 사용하여 Hull 영역 내의 3D 포인트를 실제 거리(미터)로 반환합니다.

        :param depth_image: 깊이 이미지 (2D numpy array, 단위: mm)
        :param hull: Convex Hull의 포인트 리스트 (numpy array로 표현된 포인트의 리스트)
        :param camera_info: 카메라 내부 파라미터 (fx, fy, cx, cy, depth_scale)
        :return: Hull 영역 내의 3D 포인트 리스트 [(X1, Y1, Z1), (X2, Y2, Z2), ...] (단위: 미터)
        """
        if self.camera_info is None:
            return []
        # 카메라 내부 파라미터 추출
        K = np.array(self.camera_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        depth_scale = 0.001  # 대부분의 깊이 카메라는 밀리미터 단위로 깊이를 제공합니다.
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        ys, xs = np.where(mask == 255)
        points_3d_meters = []
        for x, y in zip(xs, ys):
            depth = depth_image[y, x] * depth_scale  # 깊이 스케일 변환
            if depth > 0:  # 유효한 깊이 값만 고려
                # 픽셀 좌표계에서 카메라 좌표계로 변환
                Z = depth
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                points_3d_meters.append([X, Y, Z])

        return np.asarray(points_3d_meters)

    def project_strand_onto_plane(self, pixel_strand, depth_image, plane_coefficients):
        """
        이미지 상의 픽셀 좌표로 주어진 스트랜드를 3차원 평면 위에 사영합니다.

        :param pixel_strand: 이미지 상의 픽셀 좌표로 주어진 스트랜드 [(x1, y1), (x2, y2), ...]
        :param depth_image: 해당 픽셀의 깊이 정보를 담고 있는 깊이 이미지
        :param camera_intrinsics: 카메라 내부 파라미터 (fx, fy, cx, cy)
        :param plane_coefficients: 3차원 평면의 파라미터 (a, b, c, d) for ax + by + cz + d = 0
        :return: 평면 위의 3차원 좌표 리스트 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
        """
        if self.camera_info is None:
            return []
        # 카메라 내부 파라미터 추출
        K = np.array(self.camera_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        depth_scale = 0.001  # 대부분의 깊이 카메라는 밀리미터 단위로 깊이를 제공합니다.

        a, b, c, d = plane_coefficients

        projected_points = []

        for x, y in pixel_strand:
            if depth_image[y,x] == 0:
                continue
            # 깊이 이미지에서 해당 픽셀의 깊이 값 가져오기
            depth = depth_image[y, x] * depth_scale

            # 이미지 좌표계에서 카메라 좌표계로 변환
            X_cam = (x - cx) * depth / fx
            Y_cam = (y - cy) * depth / fy
            Z_cam = depth

            # 카메라 좌표계의 포인트를 평면에 사영
            t = -(a * X_cam + b * Y_cam + c * Z_cam + d) / (a * a + b * b + c * c)
            X_plane = X_cam + a * t
            Y_plane = Y_cam + b * t
            Z_plane = Z_cam + c * t

            projected_points.append((X_plane, Y_plane, Z_plane))

        return np.asarray(projected_points)

    def generate_pose_path(self, projected_points):
        """
        사영된 포인트들을 기반으로 연속적인 포즈를 가진 경로를 생성합니다.

        :param projected_points: 사영된 3D 포인트 리스트
        :return: 포즈 경로 (x, y, z 축 벡터들의 리스트)
        """
        result_x, result_y, result_z = [], [], []

        for pt_b, pt, pt_a in zip(projected_points[:-2], projected_points[1:-1], projected_points[2:]):
            # z축: 포인트의 노멀 벡터 (여기서는 단순화를 위해 pt_a - pt 벡터 사용)
            axis_z = np.array(pt_a) - np.array(pt_b)
            axis_z = axis_z / np.linalg.norm(axis_z)

            # y축: pt_b와 pt_a를 연결하는 벡터에서 z축 벡터의 성분을 제거한 후 정규화
            v = np.array(pt_a) - np.array(pt_b)
            axis_y = v - np.dot(axis_z, v) * axis_z
            axis_y = axis_y / np.linalg.norm(axis_y)

            # x축: y축과 z축의 외적
            axis_x = np.cross(axis_y, axis_z)

            result_x.append(axis_x)
            result_y.append(axis_y)
            result_z.append(axis_z)

            return result_x, result_y, result_z
    def quaternion_angle_diff(self, q1, q2):
        # 두 쿼터니언 사이의 내적 계산
        dot_product = np.dot(q1, q2)

        # 내적 값이 1보다 크면 1로, -1보다 작으면 -1로 제한
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # 각도 차이 계산
        angle_diff = 2 * np.arccos(dot_product)

        # 라디안에서 각도로 변환 (선택적)
        # angle_diff = np.degrees(angle_diff)

        return angle_diff

    def generate_combed_path_with_orientation_on_plane(self, projected_points, plane_coefficients, time_now, max_angle_diff_degree=10.0, frame_id="base"):
        """
        사영된 3D 포인트들의 경로와 평면 계수를 기반으로 PoseArray 메시지를 생성합니다.

        :param projected_points: 사영된 3D 포인트 리스트 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
        :param plane_coefficients: 평면의 계수 (a, b, c, d) for ax + by + cz + d = 0
        :param frame_id: PoseArray 메시지의 frame_id
        :return: PoseArray 메시지
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = time_now

        # 평면의 법선 벡터 계산
        a, b, c, _ = plane_coefficients
        normal_vector = np.array([a, b, c])
        normal_vector /= np.linalg.norm(normal_vector)  # 정규화

        camera_direction = np.array([0, 0, 1])  # 카메라 방향 벡터

        max_angle_diff_degree = float(max_angle_diff_degree)
        max_angle_diff=np.radians(max_angle_diff_degree)

        # 법선 벡터와 카메라 방향 벡터 사이의 내적
        dot_product = np.dot(normal_vector, camera_direction)

        # 내적이 음수인 경우 법선 벡터 반전
        if dot_product < 0:
            normal_vector = -normal_vector

        normal_vector /= np.linalg.norm(normal_vector)  # 정규화

        last_quaternion = None

        for i in range(len(projected_points) - 1):
            pose = Pose()

            # 현재 포인트와 다음 포인트 설정
            current_point = np.array(projected_points[i])
            next_point = np.array(projected_points[i + 1])

            # 위치 설정
            pose.position.x = current_point[0]
            pose.position.y = current_point[1]
            pose.position.z = current_point[2]

            # y축 방향 설정: 현재 포인트에서 다음 포인트로의 벡터
            y_axis = next_point - current_point
            y_axis /= -np.linalg.norm(y_axis)  # 정규화

            # z축 방향 설정: 평면의 법선 벡터
            z_axis = normal_vector

            # x축 방향 설정: y축과 z축의 외적
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)  # 정규화

            # 방향 벡터들로부터 회전 행렬 생성
            rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
            # 회전 행렬을 쿼터니언으로 변환
            quaternion = tf.transformations.quaternion_from_matrix(
                np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
            )

            # 각도 차이가 max_angle_diff보다 큰 경우 조정
            if last_quaternion is not None:
                angle_diff = self.quaternion_angle_diff(last_quaternion, quaternion)
                if angle_diff > max_angle_diff:
                    # 각도 차이를 줄이기 위한 조정 로직 (예: 선형 보간, Slerp 등)
                    t = max_angle_diff / angle_diff
                    quaternion = tf.transformations.quaternion_slerp(last_quaternion, quaternion, t)

            pose.orientation = Quaternion(*quaternion)

            # Pose 추가
            pose_array.poses.append(pose)
            last_quaternion = quaternion

        return pose_array

    def main(self):             #
        while not self.camera_info:
            continue
        print("received camera_info... ")
        self.create_pcd = CreatePointCloud(self.camera_info)
        # self.refiner = CRFSegmentationRefiner(0.9, 1.0)
        self.refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

        print("start masked pcd publishing")
        while not rospy.is_shutdown():
            if self.cv_image is not None and self.cv_depth is not None:
                self.update(self.cv_image)
                time_now = rospy.Time.now()
                # if self.output_image_human is None:
                #     continue
                # self.output_image = self.refiner.refine(self.cv_image, self.output_image)
                self.output_image = self.refiner.refine(self.cv_image, self.output_image, fast=True, L=200)
                self.output_image = np.where(self.output_image > 255*0.9, 255, 0).astype(np.uint8)
                # self.output_image_face = self.refiner.refine(self.cv_image, self.output_image_face, fast=True, L=200)
                # self.output_image_face = np.where(self.output_image_face > 255*0.9, 255, 0).astype(np.uint8)

                # self.output_image_human = self.refiner.refine(self.cv_image, self.output_image_human, fast=True, L=100)
                # self.output_image_human = np.where(self.output_image_human > 255*0.9, 255, 0).astype(np.uint8)
                # self.output_image_human = self.output_image | self.output_image_face

                # self.output_image_human_color = self.output_image_human[:,:,np.newaxis] # /255
                # self.output_image_human_color = self.output_image_human_color.astype(np.uint8) *  self.cv_image[:,:,::-1]

                # hair masekd depth image
                masked_depth = self.apply_depth_mask(self.cv_depth, self.output_image)
                self.output_image, masked_depth = self.refine_mask_with_depth(self.output_image, masked_depth, self.distance)

                # for human tracking
                # masked_human_depth = self.apply_depth_mask(self.cv_depth, self.output_image_human)
                # self.output_image_human, masked_human_depth = self.refine_mask_with_depth(self.output_image_human, masked_human_depth, self.distance)
                # masked_human_depth_msg= self.make_depth_msg(masked_human_depth, time_now)

                # normal_map_vis, normal_map = compute_normal_map(masked_depth)

                if self.mode == "strip":
                    strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                    # create_and_publish_strips_markers(self.strips_pub, self.frame_id, xyz_strips)
                    angle_map = angle_map.to("cpu").numpy().copy()
                    strand_rgb, strands = self.create_hair_strands(strand_rgb, self.output_image, angle_map, W=self.size, n_strands=50, strand_length=50, distance=5)
                    strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)
                if self.mode == "gabor":
                    strand_rgb, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                    # strands = self.create_hair_strands_gabor(angle_map)
                    strand_rgb, strands = self.create_hair_strands(strand_rgb, self.output_image, angle_map, W=1, n_strands=50, strand_length=50, distance=5)
                    # strand_rgb = self.visualize_hair_strands(strand_rgb, strands)
                if self.mode == "color":
                    strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                if self.mode == "3d_color":
                    strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.output_image, masked_depth, self.camera_info)
                if self.mode == "nn":
                    strand_map, angle_map = img2strand(self.opt, self.cv_image, self.output_image)
                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)
                    strand_rgb, strands = self.create_hair_strands(np.zeros_like(self.cv_image), self.output_image, angle_map, W=1, n_strands=50, strand_length=100, distance=20)
                    strands = self.remove_strands_with_start_point_in_combined_points(strands)

                    strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)
                    # self.cv_image= cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

                strand_rgb_reversed, strands_reversed= self.create_strands_from_mask_boundary_with_rotated_angle_map(np.zeros_like(self.cv_image), self.output_image, angle_map, W=self.size, n_strands=50, strand_length=50, distance=5, gradiation=3)
                visualized_image, labels, cluster_areas = self.cluster_and_visualize_strands(self.cv_image, strands, n_clusters=self.n_clusters)
                out = self.face_detector.run(self.cv_image)
                visualized_image = self.face_detector.draw_landmarks_on_image(visualized_image, out)
                direction_vectors_reversed = self.calculate_direction_vectors_reversed(strands_reversed)
                # checked_start_points, checked_strands = self.find_strands_with_large_direction_difference_reversed(strands_reversed, direction_vectors_reversed, 20)
                visualized_image, hulls = self.visualize_convex_hulls(strands, labels, visualized_image)
                # if checked_start_points is not None:
                #     mean_parting_point = np.mean(checked_start_points, axis=0)
                #     print("mean:", mean_parting_point)
                # visualized_image = self.visualize_checked_start_points(visualized_image, checked_start_points)
                # face_landmarks_list = self.face_detector.get_landmarks()
                # face_landmarks_np = self.face_detector.face_landmarks_np
                visualized_image, representative_strands = self.visualize_representative_strands(visualized_image, strands, labels, hulls, self.output_image, angle_map)
                # if len(face_landmarks_list) > 0 and not np.isnan(mean_parting_point).any():
                #     eye1 = face_landmarks_list[0][self.eye1_ind]
                #     eye2 = face_landmarks_list[0][self.eye2_ind]
                #     angle = self.calculate_angle(eye1, eye2, mean_parting_point)
                #     print(angle)
                #     visualized_image = self.visualize_line_between_eyes_and_parting_point(visualized_image, eye1, eye2, mean_parting_point)

                if len(hulls) > 0:
                    for i, hull in enumerate(hulls):
                        hull_3d = self.extract_points_from_hull(self.cv_depth, hull)
                        # print(hull_3d)
                        if len(hull_3d) > 0:
                            estimated_plane, plane_inliers = self.ransac(hull_3d[:,:3], 0.01, 3, 100)
                            if (estimated_plane is not None) and (plane_inliers is not None):
                                plane_msg, plane_inliers_msg = create_plane_and_inliers_markers(estimated_plane, plane_inliers, hull_3d[:,:3], (0.5, 0.5, 0.001), frame_id=self.frame_id)
                                self.plane_pub.publish(plane_msg)
                                projected_strand = self.project_strand_onto_plane(representative_strands[i], self.cv_depth, estimated_plane)
                                strand_posearray = self.generate_combed_path_with_orientation_on_plane(projected_strand, estimated_plane, time_now, 5 ,self.frame_id)
                                # print(strand_posearray)
                                self.strand_path_pub.publish(strand_posearray)
                        time.sleep(1)
                    # depth_2d_map = img2depth(self.opt, self.cv_image, self.output_image)
                    # depth_map_2d_msg= self.bridge.cv2_to_imgmsg(depth_2d_map, "bgr8")
                    # depth_map_2d_msg.header = Header(stamp=time_now)
                    # self.depth_2d_pub.publish(depth_map_2d_msg)

                    # orientation_map_3d = compute_3d_orientation_map(normal_map, angle_map, self.output_image)
                    # strand_rgb = visualize_orientation_map(orientation_map_3d.to("cpu").numpy())

                masked_depth_msg= self.make_depth_msg(masked_depth, time_now)
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(visualized_image, "bgr8")
                    ros_image.header = Header(stamp=time_now)
                    # points, header, fields= self.create_pcd.create_point_cloud(strand_rgb, self.cv_depth, self.output_image, time_now)
                    # if len(points) != 0:
                    #     indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
                    #     closest_cloud = points[indices]
                    #     largest_cloud_msg = pc2.create_cloud(header, fields, closest_cloud)
                    #     if largest_cloud_msg is not None:
                    #         self.cloud_pub.publish(largest_cloud_msg)

                    # ransac
                    # if len(closest_cloud[:,:3]) > 4:
                    #     center, radius, sph_inliers = self.sph.fit(closest_cloud[:,:3].astype(np.float32),thresh=0.1, maxIteration=100)
                    # if (center is not None) and (radius is not None):
                    #     sph_msg = create_sphere_marker(center, radius, self.frame_id)
                    #     self.sphere_pub.publish(sph_msg)
                    # estimated_plane, plane_inliers = self.ransac(closest_cloud[:,:3][::4], 0.3, 3, 50)

                    # if (estimated_plane is not None) and (plane_inliers is not None):
                    #     plane_msg, plane_inliers_msg = create_plane_and_inliers_markers(estimated_plane, plane_inliers, closest_cloud[:,:3], (0.5, 0.5, 0.001), frame_id=self.frame_id)
                    #     self.plane_pub.publish(plane_msg)

                    hair_mask_msg = self.bridge.cv2_to_imgmsg(self.output_image, "passthrough")
                    hair_mask_msg.header = Header(stamp=time_now)

                    # human_msg = self.bridge.cv2_to_imgmsg(self.output_image_human_color, "rgb8")
                    # human_msg.header = Header(stamp=time_now)

                    # self.human_pub.publish(human_msg)
                    self.hair_mask_pub.publish(hair_mask_msg)
                    # self.depth_human_pub.publish(masked_human_depth_msg)
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
