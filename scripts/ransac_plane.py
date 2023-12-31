# Importing necessary libraries for ROS and Point Cloud processing
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
import pcl
import numpy as np
from geometry_msgs.msg import Point, Quaternion
import tf.transformations as tf_trans

# Function to process the incoming PointCloud data
def cloud_cb(cloud_msg):
    # Convert ROS PointCloud2 message to PCL PointCloud
    cloud = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))
    cloud_array = np.array(list(cloud), dtype=np.float32)
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(cloud_array)

    # Perform Plane Segmentation using RANSAC
    seg = pcl_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    inliers, plane = seg.segment()

    plane_scale=(0.5, 0.5, 0.001)

    # Extract the inliers
    extracted_inliers = pcl_cloud.extract(inliers, negative=False)

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

    # Create a Marker for visualization in RViz
    plane_marker = Marker()
    plane_marker.header.frame_id = cloud_msg.header.frame_id
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

    plane_pub.publish(plane_marker)

    # Publish the Marker

# ROS Node Initialization
rospy.init_node('point_cloud_processor', anonymous=True)
rospy.Subscriber("/segmented_cloud", PointCloud2, cloud_cb)
plane_pub = rospy.Publisher('estimated_plane', Marker, queue_size=10)


# Spin until node is shut down
rospy.spin()
