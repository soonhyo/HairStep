import rospy
import numpy as np
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import ColorRGBA

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

    # 점군의 중심 계산
    inliers_points = np.array([points[i] for i in inliers])
    centroid = np.mean(inliers_points, axis=0)

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
    plane_marker.pose.position.x = centroid[0]
    plane_marker.pose.position.y = centroid[1]
    plane_marker.pose.position.z = centroid[2]
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
def create_and_publish_strips_markers(marker_pub, frame_id, xyz_strips, line_width=0.001, r=1.0, g=0.0, b=0.0, a=1.0):
    """
    여러 3D 좌표 쌍을 사용하여 MarkerArray를 생성하고 발행합니다.

    :param marker_array_pub: ROS 마커 배열 발행자(publisher)
    :param frame_id: 마커의 프레임 ID
    :param xyz_strips: 시작점과 끝점을 포함하는 3D 좌표 리스트
    :param line_width: 선의 두께
    :param r, g, b, a: 선의 색상과 투명도 (RGBA)
    """
    marker_array = MarkerArray()

    # 각 좌표 쌍에 대한 마커 생성
    for i, (begin, end) in enumerate(xyz_strips):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.id = i

        # 마커 크기 및 색상 설정
        marker.scale.x = line_width
        marker.color = ColorRGBA(r, g, b, a)

        # 마커 좌표 설정
        marker.points.append(Point(*begin))
        marker.points.append(Point(*end))

        # MarkerArray에 마커 추가
        marker_array.markers.append(marker)

    # MarkerArray 발행
    marker_pub.publish(marker_array)
