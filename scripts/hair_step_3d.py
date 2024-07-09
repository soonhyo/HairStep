import time

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError

from create_pcd import CreatePointCloud
from utils import HairAngleCalculator
from ros_utils import *

from tf import transformations as tf_trans

class RosApp():
    def __init__(self):
        rospy.init_node('orientation_field_node', anonymous=True)
        self.bridge = CvBridge()

        self.camera_ns = rospy.get_param("camera_ns", "camera")

        rospy.loginfo("camera_ns: "+self.camera_ns)

        self.pose_pub = rospy.Publisher("/orientation_field", PoseArray, queue_size=1)
        self.cloud_pub = rospy.Publisher("/hair_pointcloud", PointCloud2, queue_size=1)

        self.rate = rospy.Rate(30)

        self.orientation_3d_vis = None
        self.cv_depth = None
        self.cv_mask = None
        self.points = None
        self.camera_info = None

        self.stamp = None

        self.mode = "nn"
        self.size = 15
        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)
        self.create_pcd = None

        self.distance = 0.8

        self.frame_id = self.camera_ns+"_color_optical_frame"

        rospy.Subscriber("/segmented_depth", Image, self.depth_callback)
        rospy.Subscriber("/segmented_mask", Image, self.mask_callback)
        rospy.Subscriber("/orientation_image", Image, self.orientation_callback)

        rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

        self.output_image = None

    def depth_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.stamp = data.header.stamp
        except CvBridgeError as e:
            rospy.logerr(e)

    def mask_callback(self, data):
        try:
            self.cv_mask = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def orientation_callback(self, data):
        try:
            self.orientation_3d_vis = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

    def camera_info_callback(self, data):
        self.camera_info = data
        self.create_pcd = CreatePointCloud(self.camera_info)

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

    def publish_pose_array(self, point_cloud, orientation_map, header):
        pose_array = PoseArray()
        pose_array.header = header

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for point in point_cloud[::16]:
            x, y, z = point[0], point[1], point[2]

            if z > 0:
                u = int(fx * x / z + cx)
                v = int(fy * y / z + cy)

                if u >= 0 and u < orientation_map.shape[1] and v >= 0 and v < orientation_map.shape[0]:
                    direction = orientation_map[v, u]

                    # Normalize the direction vector
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

    def vector_to_quaternion(self, vector):
        # Assumes the vector is normalized
        angle = np.arccos(vector[2])  # Angle between vector and z-axis
        axis = np.cross([0, 0, 1], vector)
        if np.linalg.norm(axis) == 0:
            # If axis is zero vector, vector is aligned with z-axis
            axis = [1, 0, 0]
        axis = axis / np.linalg.norm(axis)

        quat = tf_trans.quaternion_about_axis(angle, axis)
        return quat

    def main(self):
        while not rospy.is_shutdown():
            if self.cv_depth is None or self.orientation_3d_vis is None or self.cv_mask is None:
                continue
            points, header, fields = self.create_pcd.create_point_cloud(self.orientation_3d_vis, self.cv_depth, self.cv_mask, self.stamp)
            if len(points) != 0:
                indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
                closest_cloud = points[indices]
                largest_cloud_msg = pc2.create_cloud(header, fields, closest_cloud)
                if largest_cloud_msg is not None:
                    self.cloud_pub.publish(largest_cloud_msg)

                self.orientation_3d_map = self.orientation_3d_vis/255*2-1 # range : 0-1
                # Create and publish PoseArray
                self.publish_pose_array(closest_cloud, self.orientation_3d_map, header)

            self.rate.sleep()
if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
