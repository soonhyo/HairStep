#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose
import tf.transformations as tf_trans
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from create_pcd import CreatePointCloud

class NormalImageProcessor:
    def __init__(self):
        rospy.init_node('normal_image_processor', anonymous=True)

        self.bridge = CvBridge()
        # Publishers
        self.cloud_pub = rospy.Publisher("/hair_pointcloud", PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher("/orientation_field", PoseArray, queue_size=1)

        # Subscribers
        rospy.Subscriber("/normal_image", Image, self.normal_image_callback)
        rospy.Subscriber("/segmented_depth", Image, self.depth_image_callback)
        rospy.Subscriber("/segmented_mask", Image, self.mask_callback)
        rospy.Subscriber("/" + rospy.get_param("camera_ns", "camera") + "/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

        self.normal_image = None
        self.depth_image = None
        self.create_pcd = None
        self.camera_info = None

        self.stamp = None

        self.rate = rospy.Rate(30)

    def normal_image_callback(self, data):
        try:
            self.normal_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def mask_callback(self, data):
        try:
            self.cv_mask = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_image_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.stamp = data.header.stamp
        except CvBridgeError as e:
            rospy.logerr(e)

    def camera_info_callback(self, data):
        self.camera_info = data
        self.create_pcd = CreatePointCloud(self.camera_info)

    def process_data(self):
        while not rospy.is_shutdown():
            if self.normal_image is None or self.cv_mask is None or self.cv_depth is None or self.camera_info is None:
                continue

            # Generate PointCloud and PoseArray
            points, header, fields = self.create_pcd.create_point_cloud(self.normal_image, self.cv_depth, self.cv_mask, self.stamp)
            if len(points) != 0:
                point_cloud_msg = pc2.create_cloud(header, fields, points)
                if point_cloud_msg is not None:
                    self.cloud_pub.publish(point_cloud_msg)

                self.publish_pose_array(points, header)
            self.rate.sleep()

    def publish_pose_array(self, points, header):
        pose_array = PoseArray()
        pose_array.header = header

        for point in points[::16]:
            x, y, z, rgba = point
            nx, ny, nz = self.rgba_to_normal(rgba)
            # Normalize the normal vector
            normal = np.array([nx, ny, nz])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm

            # Convert the normal vector to a quaternion
            quat = self.vector_to_quaternion(normal)

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

    def rgba_to_normal(self, rgba):
        r = (rgba >> 16) & 0xFF
        g = (rgba >> 8) & 0xFF
        b = rgba & 0xFF
        normal = np.array([r, g, b]) / 255.0 * 2 - 1
        return normal

    def vector_to_quaternion(self, vector):
        angle = np.arccos(vector[2])  # Angle between vector and z-axis
        axis = np.cross([0, 0, 1], vector)
        if np.linalg.norm(axis) == 0:
            axis = [1, 0, 0]
        axis = axis / np.linalg.norm(axis)

        quat = tf_trans.quaternion_about_axis(angle, axis)
        return quat

if __name__ == "__main__":
    processor = NormalImageProcessor()
    processor.process_data()
