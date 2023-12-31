import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header

# Assuming 'sdf_utils.py' contains the provided functions: create_grid, eval_grid, etc.
from lib.sdf import create_grid, eval_grid

def publish_sdf_as_pointcloud(sdf, resolution, origin, threshold):
    # PointCloud 데이터 준비
    points = []
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            for k in range(resolution[2]):
                if sdf[i, j, k] < threshold:  # 임계값 이하인 포인트만 추가
                    x = origin[0] + i * grid_size
                    y = origin[1] + j * grid_size
                    z = origin[2] + k * grid_size
                    points.append([x, y, z])

    # PointCloud2 메시지 생성
    header = Header(frame_id="camera_color_optical_frame")  # 혹은 적절한 좌표계
    pc2_msg = pc2.create_cloud_xyz32(header, points)

    # PointCloud2 메시지 발행
    point_cloud_publisher.publish(pc2_msg)

# Callback function for point cloud subscriber
def point_cloud_callback(cloud_msg):
    # Convert ROS PointCloud2 to numpy array
    cloud_points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))[:5000]

    # Calculate bounding box of the point cloud
    b_min = np.amin(cloud_points, axis=0)
    b_max = np.amax(cloud_points, axis=0)

    # Create a grid for SDF computation
    grid_resolution = [50, 50, 50]  # Adjust the resolution as needed
    coords, _ = create_grid(grid_resolution[0], grid_resolution[1], grid_resolution[2], b_min, b_max)

    # Define the evaluation function for SDF computation
    def eval_func(points):
        # Reshape points for broadcasting
        # points shape: (3, N), where N is the number of points in the grid
        # cloud_points shape: (M, 3), where M is the number of points in the point cloud
        points_reshaped = np.transpose(points)  # Reshape to (N, 1, 3)
        print(points_reshaped.shape)
        print(cloud_points.shape)
        distances = np.sqrt(((points_reshaped - cloud_points) ** 2).sum(axis=2))
        return np.min(distances, axis=1)
    grid_origin = b_min
    # Compute SDF
    sdf = eval_grid(coords, eval_func)
    publish_sdf_as_pointcloud(sdf, grid_resolution, grid_origin, 0.5)

# Initialize ROS node
rospy.init_node('point_cloud_to_sdf_node')

# Create a subscriber for the point cloud data
point_cloud_subscriber = rospy.Subscriber("/segmented_cloud", PointCloud2, point_cloud_callback)

# Create a publisher for the SDF data
point_cloud_publisher = rospy.Publisher("/sdf_pointcloud", PointCloud2, queue_size=10)

# Keep the node running
rospy.spin()
