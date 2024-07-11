import time

import rospy
import numpy as np
import torch
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray
import tf.transformations as tf_trans
import tf
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

class OrientationFieldGenerator():
    def __init__(self):
        self.cv_image = None
        self.cv_depth = None
        self.cv_mask = None
        self.points = None
        self.camera_info = None

        self.distance = 1.0 # m

        self.hair_angle_calculator = HairAngleCalculator(mode="nn")
        self.create_pcd = CreatePointCloud()

        self.normal_bgr = None
        self.normal_map = None

        self.strand_bgr = None
        self.angle_map = None

        self.orientation_3d_bgr = None

    def get_normal_image(self, cv_depth):
        # normal image
        normal_bgr, normal_map = compute_normal_map(cv_depth, 3)
        return normal_bgr, normal_map

    def get_strand_map(self, cv_image, cv_mask):
        # strand map
        strand_rgb, angle_map = img2strand("./img2strand.pth", cv_image, cv_mask)
        strand_bgr = cv2.cvtColor(strand_rgb, cv2.COLOR_BGR2RGB)
        return strand_bgr, angle_map

    def transform_points(self, matrix, points):
        # points: m x n x 3
        m, n, _ = points.shape

        # Convert points to homogeneous coordinates: m x n x 4
        ones = np.ones((m, n, 1))
        homogeneous_points = np.concatenate((points, ones), axis=-1)

        # Reshape points to (m*n, 4) for matrix multiplication
        homogeneous_points = homogeneous_points.reshape(-1, 4).T

        # Apply transformation
        transformed_points = np.dot(matrix, homogeneous_points)

        # Reshape back to (m, n, 4) and drop homogeneous coordinate
        transformed_points = transformed_points.T.reshape(m, n, 4)
        transformed_points = transformed_points[:, :, :3] / transformed_points[:, :, 3][:, :, np.newaxis]

        return transformed_points

    def set_transform(self, points, trans, rot):
        M = tf_trans.concatenate_matrices(tf_trans.translation_matrix(trans)
                                          ,tf_trans.quaternion_matrix(rot))
        transformed_map = self.transform_points(M, points)
        return transformed_map

    def get_orientation_3d_map(self, normal_map, angle_map, cv_mask, trans, rot):
        # orientation 3d map
        orientation_3d_map = compute_3d_orientation_map(normal_map, angle_map, cv_mask)
        orientation_3d_map_transformed= self.set_transform(orientation_3d_map, trans, rot)
        orientation_3d_rgb = visualize_orientation_map(orientation_3d_map_transformed)
        orientation_3d_bgr = cv2.cvtColor(orientation_3d_rgb, cv2.COLOR_RGB2BGR)
        return orientation_3d_bgr # changed with camera?

    def get_pointcloud_with_orientation(self, orientation_3d_bgr, cv_depth, cv_mask, camera_info, trans, rot):
        # pointcloud with orientation 3d map
        points = self.create_pcd.create_point_cloud_o3d(orientation_3d_bgr, cv_depth, cv_mask, camera_info, trans, rot, 0.01)
        if len(points) != 0:
            indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
            closest_cloud = points[indices]

            points = closest_cloud
        else:
            points = points
        return points

    def main(self, cv_depth, cv_image, cv_mask, camera_info, trans, rot):
        normal_bgr, normal_map = self.get_normal_image(cv_depth)
        strand_bgr, angle_map = self.get_strand_map(cv_image, cv_mask)
        orientation_3d_bgr =  self.get_orientation_3d_map(normal_map, angle_map, cv_mask, trans, rot)
        points = self.get_pointcloud_with_orientation(orientation_3d_bgr, cv_depth, cv_mask, camera_info, trans, rot)
        return normal_bgr, strand_bgr, orientation_3d_bgr, points

class RosApp():
    def __init__(self):
        rospy.init_node('orientation_field_node', anonymous=True)
        self.bridge = CvBridge()
        self.camera_ns = rospy.get_param("camera_ns", "camera")

        rospy.loginfo("camera_ns: "+self.camera_ns)

        # TODO: repeated variables
        self.cv_image = None
        self.cv_depth = None
        self.cv_mask = None
        self.points = None
        self.camera_info = None

        self.strand_pub = rospy.Publisher("/strand_image", Image, queue_size=1)
        self.normal_pub = rospy.Publisher("/normal_image", Image, queue_size=1)
        self.ori3d_pub = rospy.Publisher("/orientation_image", Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("/hair_pointcloud", PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher("/orientation_field", PoseArray, queue_size=1)

        self.rate = rospy.Rate(30)

        self.frame_id = self.camera_ns+"_color_optical_frame"

        self.orientation_field_generator = OrientationFieldGenerator()

        self.target_frame = "camera_color_optical_frame"

        self.trans = None
        self.rot = None
        self.listener = tf.TransformListener()

        rospy.Subscriber("/segmented_image", Image, self.image_callback)
        rospy.Subscriber("/segmented_depth", Image, self.depth_callback)
        rospy.Subscriber("/segmented_mask", Image, self.mask_callback)
        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

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
        self.get_transform(self.target_frame, data.header.frame_id)

    def get_transform(self, target, source):
        while self.trans is None or self.rot is None:
            try:
                (self.trans, self.rot) = self.listener.lookupTransform(target, source, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

    def revert_rgba(self, data):
        # Convert uint32 values back to RGBA arrays
        rgba = np.zeros((data.shape[0], 4), dtype=np.uint8)
        rgba[:, 0] = np.right_shift(data, 16) & 0xFF  # Red
        rgba[:, 1] = np.right_shift(data, 8) & 0xFF   # Green
        rgba[:, 2] = data & 0xFF                      # Blue
        rgba[:, 3] = np.right_shift(data, 24) & 0xFF  # Alpha
        return rgba

    def publish_pose_array(self, point_cloud, header):
        pose_array = PoseArray()
        pose_array.header = header

        for point in point_cloud[::2]:
            x, y, z = point[0], point[1], point[2]
            rgba = self.revert_rgba(np.asarray([point[3]]))

            # Normalize the direction vector
            direction = rgba[:, :3]/255*2 -1
            print(direction)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm

            # Convert the direction vector to a quaternion
            quat = self.vector_to_quaternion(direction)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            pose_array.poses.append(pose)

        self.pose_pub.publish(pose_array)

    def calculate_angles(self, vector):
        # Normalize the input vector
        vector = vector / np.linalg.norm(vector)

        # Define the unit vectors for the x, y, z axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Calculate the angles (in radians)
        angle_x = np.arccos(np.dot(vector, x_axis))
        angle_y = np.arccos(np.dot(vector, y_axis))
        angle_z = np.arccos(np.dot(vector, z_axis))

        return angle_x, angle_y, angle_z

    def vector_to_quaternion(self, vector):
        # Calculate angles with respect to x, y, z axes
        angle_x, angle_y, angle_z = self.calculate_angles(vector)

        # Convert Euler angles to quaternion
        quaternion = tf_trans.quaternion_from_euler(angle_x, angle_y, angle_z)

        return quaternion


    def main(self):
        while not rospy.is_shutdown():
            if self.cv_depth is None or self.cv_image is None or self.cv_mask is None or self.camera_info is None:
                continue
            result = self.orientation_field_generator.main(self.cv_depth, self.cv_image, self.cv_mask, self.camera_info, self.trans, self.rot)
            if result is None:
                continue
            else:
                normal_bgr, strand_bgr, orientation_3d_bgr, points = result
            normal_msg = self.bridge.cv2_to_imgmsg(normal_bgr, "bgr8")
            normal_msg.header = Header(stamp=self.stamp, frame_id=self.target_frame)
            self.normal_pub.publish(normal_msg)

            strand_msg = self.bridge.cv2_to_imgmsg(strand_bgr, "bgr8")
            strand_msg.header = Header(stamp=self.stamp, frame_id=self.target_frame)
            self.strand_pub.publish(strand_msg)

            orientation_3d_img_msg = self.bridge.cv2_to_imgmsg(orientation_3d_bgr, "bgr8")
            orientation_3d_img_msg.header = Header(stamp=self.stamp, frame_id=self.target_frame)
            self.ori3d_pub.publish(orientation_3d_img_msg)

            # self.publish_pose_array(points, Header(stamp=self.stamp, frame_id=self.target_frame))

            # PointField
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('rgba', 12, PointField.UINT32, 1)]

            # PointCloud2 header
            header = Header(frame_id=self.target_frame, stamp=self.stamp)

            cloud_msg = pc2.create_cloud(header, fields, points)
            if cloud_msg is not None:
                self.cloud_pub.publish(cloud_msg)
            self.rate.sleep()

if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
