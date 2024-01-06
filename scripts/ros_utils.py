import numpy as np
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion

def create_sphere_marker(center, radius, frame_id="base_link", marker_id=0):
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
    sphere_marker.pose.orientation = Quaternion(0, 0, 0, 1)

    sphere_marker.scale.x = radius * 2  # Diameter in x
    sphere_marker.scale.y = radius * 2  # Diameter in y
    sphere_marker.scale.z = radius * 2  # Diameter in z
    sphere_marker.color.a = 0.8  # Transparency
    sphere_marker.color.r = 0.0
    sphere_marker.color.g = 1.0
    sphere_marker.color.b = 0.0

    return sphere_marker

def create_plane_and_inliers_markers(plane, inliers, points, plane_scale=(0.5, 0.5, 0.001), frame_id="base"):
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
