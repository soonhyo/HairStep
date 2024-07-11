from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np
import cv2
import open3d as o3d
import tf.transformations as tf_trans

class CreatePointCloud(object):
    def __init__(self):
        pass

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

    def apply_transform_to_pcd(self, pcd, trans, rot):
        M = tf_trans.concatenate_matrices(tf_trans.translation_matrix(trans),
                                          tf_trans.quaternion_matrix(rot))
        pcd.transform(M)
        return pcd

    def set_transform(self, points):
        M = tf_trans.concatenate_matrices(tf_trans.translation_matrix(self.trans)
                                          ,tf_trans.quaternion_matrix(self.rot))
        transformed_map = self.transform_points(M, points)
        return transformed_map

    def filter_points_in_distance_range(self, points, min_distance=0.01, max_distance=1.0):
        # 카메라(원점)로부터의 거리 계산
        points = points.astype(np.float32)
        distances = np.linalg.norm(points, axis=1)

        # 거리 범위에 있는 포인트 필터링
        filtered_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]
        return filtered_indices

    def downsample_and_cluster(self, points, voxel_size=0.005, eps=0.05, min_points=10):
        if len(points) == 0:
            return []

        # Voxel 다운샘플링
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Voxel 다운샘플링
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # DBSCAN 클러스터링
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(downsampled_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        # 가장 큰 클러스터의 인덱스 찾기
        unique_labels, counts = np.unique(labels, return_counts=True)
        ind = np.argmax(counts)

        max_label = unique_labels[ind]
        if max_label < 0:
            return []
            # raise ValueError("유효한 클러스터를 찾을 수 없습니다.")

        largest_cluster_idx = np.argmax(np.bincount(labels[labels == max_label]))

        # 가장 큰 클러스터에 해당하는 포인트들의 원래 인덱스 찾기
        largest_cluster_points = downsampled_points[labels == largest_cluster_idx]


        # 가장 큰 클러스터에 해당하는 원래 포인트들의 인덱스 찾기
        original_indices = []
        for i, original_point in enumerate(points):
            if any(np.linalg.norm(original_point - cluster_point) < voxel_size for cluster_point in largest_cluster_points):
                original_indices.append(i)

        return original_indices

    def create_point_cloud(self, color_image, depth_image, mask, camera_info):

        # camera intrinsic parameters
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        mask_3d = mask[:,:,np.newaxis].astype(np.bool_)

        v, u = np.indices((depth_image.shape[0], depth_image.shape[1]))
        z = depth_image / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        colors = np.where(mask_3d, color_image, [0,0,0])
        r, g, b = colors[:,:, 2], colors[:,:, 1], colors[:,:, 0]

        # rgba = (0xFF << 24) | (r << 16) | (g << 8) | b  # RGBA 형태로 변환
        rgba = (0xFF << 24) | (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        points = np.concatenate((x[:,:,np.newaxis][mask_3d][:,np.newaxis], y[:,:,np.newaxis][mask_3d][:,np.newaxis], z[:,:,np.newaxis][mask_3d][:,np.newaxis], rgba[:,:,np.newaxis][mask_3d][:,np.newaxis].astype(np.uint32)), axis=1, dtype=object)

        return points

    def create_point_cloud_o3d(self, color_image, depth_image, mask_image, camera_info, trans, rot, voxel_size=0.02):

        # 카메라 내부 파라미터 사용
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        mask_3d = mask_image[:, :, np.newaxis].astype(np.bool_)

        v, u = np.indices((depth_image.shape[0], depth_image.shape[1]))
        z = depth_image / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.0

        # Mask 적용
        points = points[mask_3d.flatten()]
        colors = colors[mask_3d.flatten()]

        # Open3D PointCloud 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Voxel 다운샘플링
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Open3D PointCloud에 변환 적용
        transformed_pcd = self.apply_transform_to_pcd(downsampled_pcd, trans, rot)

        transformed_points = np.asarray(transformed_pcd.points)
        transformed_colors = np.asarray(transformed_pcd.colors)

        # Create RGB values in uint32 format
        rgb_values = np.left_shift(0xFF, 24) + np.left_shift((transformed_colors[:, 0] * 255).astype(np.uint32), 16) + \
                     np.left_shift((transformed_colors[:, 1] * 255).astype(np.uint32), 8) + \
                     (transformed_colors[:, 2] * 255).astype(np.uint32)

        # Stack xyz and rgb values
        cloud_data = np.concatenate((transformed_points, rgb_values[:, np.newaxis]), axis=1, dtype=object)
        return cloud_data
