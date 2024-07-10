import time

import rospy
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray
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

    def get_orientation_3d_map(self, normal_map, angle_map, cv_mask):
        # orientation 3d map
        orientation_3d_map = compute_3d_orientation_map(normal_map, angle_map, cv_mask)
        orientation_3d_rgb = visualize_orientation_map(orientation_3d_map.to("cpu").numpy())
        orientation_3d_bgr = cv2.cvtColor(orientation_3d_rgb, cv2.COLOR_RGB2BGR)
        return orientation_3d_bgr

    def get_pointcloud_with_orientation(self, orientation_3d_bgr, cv_depth, cv_mask, camera_info):
        # pointcloud with orientation 3d map
        points = self.create_pcd.create_point_cloud(orientation_3d_bgr, cv_depth, cv_mask, camera_info)
        if len(points) != 0:
            indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
            closest_cloud = points[indices]

            points = closest_cloud
        else:
            points = points
        return points

    def main(self, cv_depth, cv_image, cv_mask, camera_info):
        normal_bgr, normal_map = self.get_normal_image(cv_depth)
        strand_bgr, angle_map = self.get_strand_map(cv_image, cv_mask)
        orientation_3d_bgr =  self.get_orientation_3d_map(normal_map, angle_map, cv_mask)
        points = self.get_pointcloud_with_orientation(orientation_3d_bgr, cv_depth, cv_mask, camera_info)
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

    def publish_pose_array(self, point_cloud, orientation_3d_bgr, header):
        pose_array = PoseArray()
        pose_array.header = header

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        orientation_map = orientation_3d_bgr/255*2-1 # range : 0-1

        for point in point_cloud[::128]:
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
            if self.cv_depth is None or self.cv_image is None or self.cv_mask is None:
                continue
            result = self.orientation_field_generator.main(self.cv_depth, self.cv_image, self.cv_mask, self.camera_info)
            if result is None:
                continue
            else:
                normal_bgr, strand_bgr, orientation_3d_bgr, points = result
            normal_msg = self.bridge.cv2_to_imgmsg(normal_bgr, "bgr8")
            normal_msg.header = Header(stamp=self.stamp)
            self.normal_pub.publish(normal_msg)

            strand_msg = self.bridge.cv2_to_imgmsg(strand_bgr, "bgr8")
            strand_msg.header = Header(stamp=self.stamp)
            self.strand_pub.publish(strand_msg)

            orientation_3d_img_msg = self.bridge.cv2_to_imgmsg(orientation_3d_bgr, "bgr8")
            orientation_3d_img_msg.header = Header(stamp=self.stamp)
            self.ori3d_pub.publish(orientation_3d_img_msg)

            self.publish_pose_array(points, orientation_3d_bgr, Header(stamp=self.stamp, frame_id=self.frame_id))

            # PointField
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('rgba', 12, PointField.UINT32, 1)]

            # PointCloud2 header
            header = Header(frame_id=self.camera_info.header.frame_id, stamp=self.stamp)

            cloud_msg = pc2.create_cloud(header, fields, points)
            if cloud_msg is not None:
                self.cloud_pub.publish(cloud_msg)
            self.rate.sleep()

if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()

# lass RosApp():
#     def __init__(self):
#         super.__init__()
#         rospy.init_node('img2strand_node', anonymous=True)
#         self.bridge = CvBridge()

#         self.camera_ns = rospy.get_param("camera_ns", "camera")

#         rospy.loginfo("camera_ns: "+self.camera_ns)

#         self.strand_pub = rospy.Publisher("/strand_image", Image, queue_size=1)
#         self.normal_pub = rospy.Publisher("/normal_image", Image, queue_size=1)
#         self.ori3d_pub = rospy.Publisher("/orientation_image", Image, queue_size=1)
#         self.cloud_pub = rospy.Publisher("/hair_pointcloud", PointCloud2, queue_size=1)

#         self.rate = rospy.Rate(30)

#         self.cv_image = None
#         self.cv_depth = None
#         self.cv_mask = None
#         self.points = None
#         self.camera_info = None
#         self.distance = 1.0 # m

#         self.mode = "nn"
#         self.size = 15
#         self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)
#         self.create_pcd = None

#         self.frame_id = self.camera_ns+"_color_optical_frame"

#         rospy.Subscriber("/segmented_image", Image, self.image_callback)
#         rospy.Subscriber("/segmented_depth", Image, self.depth_callback)
#         rospy.Subscriber("/segmented_mask", Image, self.mask_callback)

#         rospy.Subscriber("/"+self.camera_ns+"/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)

#         self.output_image = None

#     def image_callback(self, data):
#         try:
#             self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#             self.stamp = data.header.stamp
#         except CvBridgeError as e:
#             rospy.logerr(e)

#     def depth_callback(self, data):
#         try:
#             self.cv_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
#         except CvBridgeError as e:
#             rospy.logerr(e)

#     def mask_callback(self, data):
#         try:
#             self.cv_mask = self.bridge.imgmsg_to_cv2(data, "passthrough")
#         except CvBridgeError as e:
#             rospy.logerr(e)

#     def camera_info_callback(self, data):
#         self.camera_info = data
#         self.create_pcd = CreatePointCloud(self.camera_info)

#     def create_hair_strands(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3, gradiation=3):
#         strands = []
#         start_x = np.linspace(0, image.shape[1]-1, n_strands)
#         start_y = np.linspace(0, image.shape[0]-1, n_strands)

#         for x in start_x:
#             for y in start_y:
#                 if hair_mask[int(y), int(x)] > 0:
#                     _path = self.hair_angle_calculator.cal_flow_path([int(y), int(x)], hair_mask, image, distance, W, strand_length, angle_map)
#                     if len(_path) > 0:
#                         strands.append(np.asarray(_path))

#         img_edge = image.astype(np.uint8) * hair_mask[:,:,np.newaxis] * 255

#         if len(strands) > 0:
#             strands = self.approximate_bezier(strands, strand_length)
#             np.random.seed(42)
#             color_list = np.random.randint(255, size=(len(strands), 3))
#             for i, strand in enumerate(strands):
#                 # color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
#                 color = tuple((color_list[i]).tolist()) # -j*n is for gradiation
#                 for j in range(len(strand) - 1):
#                     cv2.line(img_edge, tuple(strand[j].astype(int)), tuple(strand[j+1].astype(int)), color, 3)

#         return img_edge, strands

#     def visualize_hair_strands(self, visual_map, strands):
#         for strand in strands:
#             for point in strand:
#                 x, y = point
#                 cv2.circle(visual_map, (x, y), 3, (255, 255, 255), -1)
#         return visual_map

#     def bezier_curve(self, points, n_points=10):
#         n = len(points) - 1
#         t = np.linspace(0, 1, n_points)
#         curve = np.zeros((n_points, points.shape[1]))
#         for i in range(n + 1):
#             binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
#             curve += np.outer(binom, points[i])
#         return curve.astype(np.int16)

#     def approximate_bezier(self, strands, n_points=10):
#         bezier_strands = []
#         for strand in strands:
#             bezier_strands.append(self.bezier_curve(strand, n_points))
#         return np.asarray(bezier_strands)

#     def main(self):
#         while not rospy.is_shutdown():
#             if self.cv_depth is None or self.cv_image is None or self.cv_mask is None:
#                 continue
#             normal_vis, normal_map = compute_normal_map(self.cv_depth, 3)
#             normal_image = self.bridge.cv2_to_imgmsg(normal_vis, "bgr8")
#             normal_image.header = Header(stamp=self.stamp)
#             self.normal_pub.publish(normal_image)

#             if self.mode == "3d_color":
#                 strand_rgb, xyz_strips, angle_map = self.hair_angle_calculator.process_image(self.cv_image, self.cv_mask, self.cv_depth, self.camera_info)
#             if self.mode == "nn":
#                 strand_map, angle_map = img2strand("./img2strand.pth",self.cv_image, self.cv_mask)
#                 strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)

#                 strand_image = self.bridge.cv2_to_imgmsg(strand_rgb, "bgr8")
#                 strand_image.header = Header(stamp=self.stamp)
#                 self.strand_pub.publish(strand_image)

#                 # strand_rgb, strands = self.create_hair_strands(np.zeros_like(self.cv_image), self.cv_mask, angle_map, W=1, n_strands=50, strand_length=20, distance=5)
#                 # hair_image = self.cv_image * (self.cv_mask[:,:,np.newaxis]/255)
#                 # strand_rgb = cv2.addWeighted(strand_rgb, 0.5, strand_rgb_large, 0.5, 2.2)
#                 # strand_rgb = cv2.addWeighted(hair_image.astype(np.uint8), 0.5, strand_rgb, 0.5, 2.2)

#                 # strand_rgb = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)

#                 orientation_3d_map =  compute_3d_orientation_map(normal_map, angle_map, self.cv_mask)
#                 orientation_3d_vis = visualize_orientation_map(orientation_3d_map.to("cpu").numpy())
#                 orientation_3d_vis = cv2.cvtColor(orientation_3d_vis, cv2.COLOR_RGB2BGR)
#                 orientation_3d_image = self.bridge.cv2_to_imgmsg(orientation_3d_vis, "bgr8")
#                 orientation_3d_image.header = Header(stamp=self.stamp)
#                 self.ori3d_pub.publish(orientation_3d_image)

#                 points, header, fields = self.create_pcd.create_point_cloud(orientation_3d_vis, self.cv_depth, self.cv_mask, self.stamp)
#                 if len(points) != 0:
#                     indices= self.create_pcd.filter_points_in_distance_range(points[:,:3], 0.01, self.distance)
#                     closest_cloud = points[indices]
#                     largest_cloud_msg = pc2.create_cloud(header, fields, closest_cloud)
#                     if largest_cloud_msg is not None:
#                         self.cloud_pub.publish(largest_cloud_msg)

#             self.rate.sleep()
# if __name__ == "__main__":
#     ros_app = RosApp()
#     ros_app.main()
