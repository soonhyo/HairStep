import time
from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker
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
# from scripts.crf import CRFSegmentationRefiner
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
        self.plane_pub = rospy.Publisher('estimated_plane', Marker, queue_size=10)
        self.sphere_pub = rospy.Publisher('estimated_sphere', Marker, queue_size=10)

        self.opt = MyBaseOptions().parse()

        self.rate = rospy.Rate(30)
        self.frame_id = "camera_color_optical_frame"

        self.cv_image = None
        self.cv_depth = None
        self.camera_info = None
        self.points = None

        self.distance = 1.5
        self.create_pcd = None
        # self.refiner = None
        self.sph = pyrsc.Sphere()

        self.hair_angle_calculator = HairAngleCalculator(7)
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

    def refine_mask_with_depth(self, mask, depth_image, threshold=0.8):
        outliers = depth_image/1000.0 > threshold

        refined_mask = np.copy(mask)
        refined_mask[outliers] = 0

        return refined_mask

    def ransac(self, points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        # Convert the list of points to a PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        try:
            plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
        except:
            return None, None
        return plane_model, inliers

    def create_sphere_marker(self, center, radius, frame_id="base_link", marker_id=0):
        """
        Create a ROS marker for visualizing a sphere.

        :param center: Center of the sphere (x, y, z).
        :param radius: Radius of the sphere.
        :param frame_id: The frame ID in which the marker is displayed.
        :param marker_id: Unique ID of the marker.
        :return: sphere_marker
        """
        # Create a sphere marker
        sphere_marker = Marker()
        sphere_marker.header.frame_id = frame_id
        sphere_marker.id = marker_id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position = Point(*center)
        sphere_marker.scale.x = radius * 2  # Diameter in x
        sphere_marker.scale.y = radius * 2  # Diameter in y
        sphere_marker.scale.z = radius * 2  # Diameter in z
        sphere_marker.color.a = 0.8  # Transparency
        sphere_marker.color.r = 0.0
        sphere_marker.color.g = 1.0
        sphere_marker.color.b = 0.0

        return sphere_marker

    def create_plane_and_inliers_markers(self, plane, inliers, points, plane_scale=(0.5, 0.5, 0.001), frame_id="base"):
        # Plane normal vector
        a, b, c, d = plane
        normal_vector = np.array([a, b, c])
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

        # Quaternion for plane orientation
        up_vector = np.array([0, 0, 1])  # Z-axis as the up vector
        quaternion = tf_trans.quaternion_about_axis(
            np.arccos(np.dot(up_vector, normal_vector_normalized)),
            np.cross(up_vector, normal_vector_normalized)
        )
        # Plane Marker
        plane_marker = Marker()
        plane_marker.header.frame_id = frame_id
        plane_marker.type = Marker.CUBE
        plane_marker.action = Marker.ADD
        plane_marker.pose.position.x = -d * normal_vector_normalized[0]
        plane_marker.pose.position.y = -d * normal_vector_normalized[1]
        plane_marker.pose.position.z = -d * normal_vector_normalized[2]
        plane_marker.pose.orientation = Quaternion(*quaternion)
        plane_marker.scale.x, plane_marker.scale.y, plane_marker.scale.z = plane_scale
        plane_marker.color.a = 1.0  # Transparency
        plane_marker.color.r = 0.0
        plane_marker.color.g = 1.0
        plane_marker.color.b = 0.0

        # Inliers Marker
        inliers_marker = Marker()
        inliers_marker.header.frame_id = frame_id
        inliers_marker.type = Marker.POINTS
        inliers_marker.action = Marker.ADD
        inliers_marker.scale.x = 0.02  # Size of the points
        inliers_marker.scale.y = 0.02
        inliers_marker.color.a = 1.0  # Transparency
        inliers_marker.color.r = 1.0
        inliers_marker.color.g = 0.0
        inliers_marker.color.b = 0.0

        # Add inlier points to the marker
        for idx in inliers:
            p = Point(*points[idx])
            inliers_marker.points.append(p)

        return plane_marker, inliers_marker

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
                    self.output_image = self.refine_mask_with_depth(self.output_image, masked_depth, self.distance)

                    # strand_map = img2strand(self.opt, self.cv_image, self.output_image)
                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

                    strand_rgb = self.hair_angle_calculator.process_image(self.cv_image, self.output_image)
                    masked_depth_msg= self.make_depth_msg(masked_depth, time_now)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
                        ros_image.header = Header(stamp=time_now)

                        points, header, fields= self.create_pcd.create_point_cloud(strand_rgb, self.cv_depth, self.output_image, time_now)

                        if len(points) == 0:
                            continue

                        indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
                        closest_cloud = points[indices]

                        estimated_plane, plane_inliers = self.ransac(closest_cloud[:,:3][::2], 0.3, 3, 100)
                        # if len(closest_cloud[:,:3]) > 4:
                        #     center, radius, sph_inliers = self.sph.fit(closest_cloud[:,:3].astype(np.float32), thresh=0.3, maxIteration=100)
                        # if (center is not None) and (radius is not None):
                        #     sph_msg = self.create_sphere_marker(center, radius, self.frame_id)
                        #     self.sphere_pub.publish(sph_msg)

                        if (estimated_plane is not None) and (plane_inliers is not None):
                            plane_msg, plane_inliers_msg = self.create_plane_and_inliers_markers(estimated_plane, plane_inliers, closest_cloud[:,:3], (0.5, 0.5, 0.001), frame_id=self.frame_id)
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
