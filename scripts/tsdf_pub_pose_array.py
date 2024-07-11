import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import tf
import tf.transformations as tf_trans

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('point_cloud_processor', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber('/open3d_pointcloud', PointCloud2, self.point_cloud_callback)
        self.pose_pub = rospy.Publisher('/orientation_field', PoseArray, queue_size=1)

    def revert_rgba(self, data):
        # Convert uint32 values back to RGBA arrays
        rgba = np.zeros((data.shape[0], 4), dtype=np.uint8)
        rgba[:, 0] = np.right_shift(data, 16) & 0xFF  # Red
        rgba[:, 1] = np.right_shift(data, 8) & 0xFF   # Green
        rgba[:, 2] = data & 0xFF                      # Blue
        rgba[:, 3] = np.right_shift(data, 24) & 0xFF  # Alpha
        return rgba

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

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        point_cloud = []
        for point in pc2.read_points(msg, skip_nans=True):
            point_cloud.append(point)
        point_cloud = np.array(point_cloud)

        # Get the header from the point cloud message
        header = msg.header

        # Publish the pose array
        self.publish_pose_array(point_cloud, header)

    def publish_pose_array(self, point_cloud, header):
        pose_array = PoseArray()
        pose_array.header = header

        for point in point_cloud[::16]:
            x, y, z = point[0], point[1], point[2]
            rgba = self.revert_rgba(np.asarray([point[3]], dtype=np.uint32))

            # Normalize the direction vector
            direction = rgba[:, :3]/255.0*2.0 - 1.0
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

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
