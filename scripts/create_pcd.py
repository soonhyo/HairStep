from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np
import cv2
import open3d as o3d


class CreatePointCloud(object):
    def __init__(self, camera_info):
        self.camera_info = camera_info
        print("initialized create point cloud instance...")
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

    def create_point_cloud(self, color_image, depth_image, mask, time_now):

        # 카메라 내부 파라미터 사용
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        mask_3d = mask[:,:,np.newaxis].astype(np.bool_)

        v, u = np.indices((depth_image.shape[0], depth_image.shape[1]))
        z = depth_image / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # 마스크 적용
        colors = np.where(mask_3d, color_image, [0,0,0])
        r, g, b = colors[:,:, 2], colors[:,:, 1], colors[:,:, 0]

        # rgba = (0xFF << 24) | (r << 16) | (g << 8) | b  # RGBA 형태로 변환
        rgba = (0xFF << 24) | (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        points = np.concatenate((x[:,:,np.newaxis][mask_3d][:,np.newaxis], y[:,:,np.newaxis][mask_3d][:,np.newaxis], z[:,:,np.newaxis][mask_3d][:,np.newaxis], rgba[:,:,np.newaxis][mask_3d][:,np.newaxis].astype(np.uint32)), axis=1, dtype=object)

        # PointField 구조 정의
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1)]

        # PointCloud2 메시지 생성
        header = Header(frame_id='camera_color_optical_frame', stamp=time_now)
        return points, header, fields
