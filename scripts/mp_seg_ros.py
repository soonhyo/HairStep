import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import time
from typing import List

from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
from scripts.mycam_strand import img2strand
from scripts.my_depth import img2depth

from scripts.utils import HairAngleCalculator

from scipy.special import comb

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from scripts.detectors_ros import MP

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


BG_COLOR = (192, 192, 192) # gray
BLACK_COLOR = (0, 0, 0) # black
MASK_COLOR = (255, 255, 255) # white
BODY_COLOR = (0, 255, 0) # green
FACE_COLOR = (255, 0, 0) # red
CLOTHES_COLOR = (255, 0, 255) # purple

# 0 - background
# 1 - hair
# 2 - body-skin
# 3 - face-skin
# 4 - clothes
# 5 - others (accessories)

class App:
    def __init__(self):
        self.output_image = None
        self.output_image_face = None
        self.output_image_human = None
        self.output_image_human_color = None

        # Create an FaceLandmarker object.
        # self.base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite',
        #                                        delegate=python.BaseOptions.Delegate.GPU)

        self.base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
                                               delegate=python.BaseOptions.Delegate.CPU)

        # self.options = ImageSegmenterOptions(base_options=self.base_options,
        #                                      running_mode=VisionRunningMode.LIVE_STREAM,
        #                                      output_category_mask=True,
        #                                      output_confidence_masks=False,
        #                                      result_callback=self.mp_callback)
        self.options = ImageSegmenterOptions(base_options=self.base_options,
                                             output_category_mask=True)

        self.segmenter = ImageSegmenter.create_from_options(self.options)

        self.latest_time_ms = 0
        self.n_clusters = 3
        self.colors = np.random.randint(255, size=(self.n_clusters, 3))
        self.eye1_ind = 243
        self.eye2_ind = 463

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            print("no update")
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        segmentation_result = self.segmenter.segment(mp_image)
        self.mp_callback(segmentation_result, mp_image)
        # self.segmenter.segment_async(mp_image, t_ms)
        # self.latest_time_ms = t_ms
    def bezier_curve(self, points, n_points=10):
        """
        베지어 곡선을 생성하는 함수.
        :param points: np.array 형태의 제어점들.
        :param n_points: 생성할 곡선의 점 개수.
        :return: 베지어 곡선을 이루는 점들의 배열.
        """
        n = len(points) - 1
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, points.shape[1]))
        for i in range(n + 1):
            binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(binom, points[i])  # 변경된 부분
        return curve.astype(np.int16)

    def approximate_bezier(self, strands, n_points=10, extend_length=0):
        """
        스트랜드를 베지어 곡선으로 근사화하는 함수.
        :param strands: np.array 형태의 점들을 포함하는 스트랜드.
        :param n_points: 생성할 곡선의 점 개수.
        :return: 근사화된 베지어 곡선.
        """
        bezier_strands = []
        for strand in strands:
            curve =self.bezier_curve(strand, n_points)
            extended_curve = curve
            if extend_length > 0:
                # 시작점에서의 연장
                tangent_start = curve[1] - curve[0]  # 첫 번째 제어점을 통해 시작점에서의 접선 벡터 계산
                tangent_start = tangent_start / np.linalg.norm(tangent_start)  # 정규화
                extended_start = curve[0] - tangent_start * extend_length  # 연장된 시작점

                # 끝점에서의 연장
                tangent_end = curve[-1] - curve[-2]  # 마지막 제어점을 통해 끝점에서의 접선 벡터 계산
                tangent_end = tangent_end / np.linalg.norm(tangent_end)  # 정규화
                extended_end = curve[-1] + tangent_end * extend_length  # 연장된 끝점

                # 연장된 포인트를 포함한 새로운 베지어 곡선 근사화
                extended_curve = np.vstack([extended_start.astype(np.int16), curve, extended_end.astype(np.int16)])

            bezier_strands.append(extended_curve)
        return np.asarray(bezier_strands)

    def calculate_strand_angles(self, strands):
        angles = []
        for i in range(len(strands) - 1):
            for j in range(i + 1, len(strands)):
                # 두 스트랜드의 방향 벡터 계산
                v1 = strands[i][-1] - strands[i][0]
                v2 = strands[j][-1] - strands[j][0]
                # 두 벡터 간의 각도 계산
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:  # 0으로 나누기 방지
                    continue
                unit_v1 = v1 / norm_v1
                unit_v2 = v2 / norm_v2
                angle = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))
                angles.append(np.degrees(angle))
        return angles

    def calculate_strand_density(self, strands, image_shape):
        density_map = np.zeros(image_shape[:2])
        for strand in strands:
            for point in strand:
                density_map[int(point[1]), int(point[0])] += 1
        return density_map

    def calculate_strand_curvature(self, strands):
        curvature_values = []
        for strand in strands:
            curvatures = []
            for i in range(1, len(strand) - 1):
                p1, p2, p3 = strand[i - 1], strand[i], strand[i + 1]
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                # 0으로 나누기 방지
                if a == 0 or b == 0 or c == 0:
                    continue
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                curvature = 4 * area / (a * b * c)
                if not np.isnan(curvature):  # NaN 확인
                    curvatures.append(curvature)
            if curvatures:
                curvature_values.append(np.mean(curvatures))
        return np.array(curvature_values)

    def calculate_entropy(self, strands, image_shape):
        orientation_map = np.zeros(image_shape[:2])
        for strand in strands:
            for i in range(1, len(strand)):  # 첫 번째 포인트를 제외하고 순회
                p1 = strand[i - 1]  # 이전 포인트
                p2 = strand[i]  # 현재 포인트
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:  # 동일한 점 방지
                    continue
                orientation = np.arctan2(dy, dx) % (2 * np.pi)  # 두 포인트 사이의 각도 계산
                orientation_map[int(p2[1]), int(p2[0])] = orientation  # 현재 포인트의 위치에 각도 저장

        hist, _ = np.histogram(orientation_map, bins=36, range=(0, 2 * np.pi))
        hist = hist / hist.sum()  # 정규화
        entropy = -np.sum(hist * np.log2(hist + 1e-9))  # 엔트로피 계산
        return entropy

    def integrate_analysis_results(self, angle_results, density_map, curvature_values, entropy_value):
        # 각도 결과의 평균
        avg_angle_deviation = np.mean(angle_results)
        # 밀도 맵의 표준 편차
        density_std = np.std(density_map)
        # 곡률 값의 평균
        avg_curvature = np.mean(curvature_values)
        # 엔트로피 값은 이미 스칼라 값임

        # 결과 통합 (여기서는 단순 평균을 사용함)
        integrated_score = (avg_angle_deviation + density_std + avg_curvature + entropy_value) / 4
        return integrated_score

    def visualize_tangled_areas(self, image, strands, integrated_score, threshold=0.1):
        # 통합 점수가 임계값을 초과하는 경우 해당 스트랜드의 중심점에 원을 그림
        if integrated_score > threshold:
            for strand in strands:
                center_point = np.mean(strand, axis=0).astype(int)
                cv2.circle(image, tuple(center_point), 10, (0, 0, 255), 2)

        return image

    def find_parting_line(self, strands):
        # 스트랜드 방향 벡터 저장
        direction_vectors = []

        for strand in strands:
            start_point = strand[0]
            end_point = strand[-1]
            # 방향 벡터 계산
            direction_vector = end_point - start_point
            direction_vectors.append(direction_vector)

        direction_vectors = np.array(direction_vectors)

        # 방향 벡터의 x 성분을 기준으로 평균 벡터 계산
        avg_vector = np.mean(direction_vectors, axis=0)

        # 평균 벡터의 방향을 기준으로 스트랜드 그룹 분류
        left_group = []
        right_group = []
        for i, vec in enumerate(direction_vectors):
            if np.dot(vec, avg_vector) > 0:
                right_group.append(strands[i])
            else:
                left_group.append(strands[i])

        # 각 그룹의 중심점 계산
        left_center = np.mean([np.mean(strand, axis=0) for strand in left_group], axis=0)
        right_center = np.mean([np.mean(strand, axis=0) for strand in right_group], axis=0)

        # 가르마 위치는 두 그룹의 중심점을 연결하는 선
        parting_line = (left_center, right_center)

        return parting_line, right_group, left_group

    def visualize_parting_line(self, image, parting_line):
        cv2.line(image, tuple(parting_line[0].astype(int)), tuple(parting_line[1].astype(int)), (255, 0, 0), 2)
        return image

    def create_hair_strands(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3, gradiation=3):
        """가상의 헤어 스트랜드를 생성합니다."""
        strands = []
        start_x = np.linspace(0, image.shape[1]-1, n_strands)
        start_y = np.linspace(0, image.shape[0]-1, n_strands)

        for x in start_x:
            for y in start_y:
                if hair_mask[int(y), int(x)] > 0:
                    _path = self.hair_angle_calculator.cal_flow_path([int(y), int(x)], hair_mask, image, distance, W, strand_length, angle_map)
                    if len(_path) > 0:
                        strands.append(np.asarray(_path))

        # strands = np.asarray(strands)
        # strands = strands.astype(np.int16)
        img_edge = image.astype(np.uint8) * hair_mask[:,:,np.newaxis] * 255

        if len(strands) > 0:
            strands = self.approximate_bezier(strands, strand_length)
            # np.random.seed(42)
            # color_list = np.random.randint(255, size=(len(strands), 3))
            # for i, strand in enumerate(strands):
            #     # color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))

            #     # for j, point in enumerate(path):

            #     #     color = tuple((color_list[i]-j*gradiation).tolist()) # -j*n is for gradiation

            #     #     cv2.circle(img_edge, (point[0], point[1]), 1, (color), -1)
            #     color = tuple(color_list[i].tolist()) # -j*n is for gradiation
            #     for j in range(len(strand) - 1):
            #         cv2.line(img_edge, tuple(strand[j].astype(int)), tuple(strand[j+1].astype(int)), color, 2)

        return img_edge, strands

    def cluster_and_visualize_strands(self, image, strands, n_clusters=2):
        if len(strands) < n_clusters:
            sorted_labels = []
            cluster_areas = []
            return image, sorted_labels, cluster_areas

        # 스트랜드의 시작점을 추출하여 클러스터링에 사용
        # K-Means 클러스터링 수행
        strands_ = strands[:,int(len(strands[0])/2)].reshape(len(strands), -1)
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0, algorithm='elkan').fit(strands_)
        # eps = 0.001
        # min_samples=5
        # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # dbscan.fit(strands_)
        # 클러스터 중심을 가져와 x 좌표에 따라 정렬
        centroids = kmeans.cluster_centers_
        # centroids = dbscan.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0])  # x 좌표 기준으로 정렬

        labels = kmeans.labels_
        # labels = dbscan.labels_
        # sorted_labels = labels

        sorted_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(sorted_indices):
            sorted_labels[labels == old_label] = new_label

        # 정렬된 클러스터 중심 반환
        #sorted_centroids = centroids[sorted_indices]

        # 클러스터링 결과를 시각화하기 위한 색상 지정
         # -j*n is for gradiation
        colors = self.colors
        # 클러스터링 결과 시각화
        cluster_colors_areas = []

        for label, strand in zip(sorted_labels, strands):
            color = tuple(colors[label].tolist())
            # color = colors[label]
            # cv2.circle(image, tuple(strands.astype(int)), 5, color, -1)  # 시작점에 색상으로 표시
            for j in range(len(strand) - 1):
                cv2.line(image, tuple(strand[j].astype(int)), tuple(strand[j+1].astype(int)), color, 1)
            mask = cv2.inRange(image, color, color)
            area = cv2.countNonZero(mask)
            cluster_colors_areas.append(area)

        return image, sorted_labels, cluster_colors_areas

    def calculate_direction_vectors(self, strands):
        direction_vectors = []
        for strand in strands:
            if len(strand) > 1:
                direction = strand[-1] - strand[0]
                direction = direction.astype(np.float32)
                if np.linalg.norm(direction) > 0:
                    direction /= np.linalg.norm(direction)
                    direction_vectors.append(direction)
                else:
                    direction_vectors.append(np.array([0, 0]))  # 길이가 1인 스트랜드 처리
            else:
                direction_vectors.append(np.array([0, 0]))  # 길이가 1인 스트랜드 처리
        return direction_vectors

    def calculate_direction_vectors_reversed(self, strands):
        direction_vectors = []
        for strand in strands:
            if len(strand) > 1:
                direction = strand[-3] - strand[-1]
                direction = direction.astype(np.float32)
                if np.linalg.norm(direction) > 0:
                    direction /= np.linalg.norm(direction)
                    direction_vectors.append(direction)
                else:
                    direction_vectors.append(np.array([0, 0]))  # 길이가 1인 스트랜드 처리
            else:
                direction_vectors.append(np.array([0, 0]))  # 길이가 1인 스트랜드 처리
        return direction_vectors

    def find_strands_with_large_direction_difference_reversed(self, strands, direction_vectors, radius=10):
        checked_start_points = []
        checked_strands = []
        for i, (strand_i, vec_i) in enumerate(zip(strands, direction_vectors)):
            for j, (strand_j, vec_j) in enumerate(zip(strands, direction_vectors)):
                if i != j:
                    distance = np.linalg.norm(strand_i[-1] - strand_j[-1])
                    if distance < radius:  # 반경 내에 있는 스트랜드
                        angle_diff = np.arccos(np.clip(np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-9), -1.0, 1.0))
                        if np.degrees(angle_diff) > 100:  # 방향 차이가 90도보다 큰 경우
                            checked_start_points.append(strand_i[-1])
                            checked_strands.append(strand_i)
                            break
        return checked_start_points, checked_strands

    def find_strands_with_large_direction_difference(self, strands, direction_vectors, radius=10):
        checked_start_points = []
        checked_strands = []
        for i, (strand_i, vec_i) in enumerate(zip(strands, direction_vectors)):
            for j, (strand_j, vec_j) in enumerate(zip(strands, direction_vectors)):
                if i != j:
                    distance = np.linalg.norm(strand_i[0] - strand_j[0])
                    if distance < radius:  # 반경 내에 있는 스트랜드
                        angle_diff = np.arccos(np.clip(np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-9), -1.0, 1.0))
                        if np.degrees(angle_diff) > 100:  # 방향 차이가 90도보다 큰 경우
                            checked_start_points.append(strand_i[0])
                            checked_strands.append(strand_i)
                            break
        return checked_start_points, checked_strands

    def visualize_checked_start_points(self, image, checked_start_points):
        for point in checked_start_points:
            cv2.circle(image, tuple(point.astype(int)), 2, (255, 0, 0), -1)  # 빨간색으로 시작점 표시
        return image

    def visualize_strand_groups(self, image, left_group, right_group):
        # 왼쪽 그룹 스트랜드를 파란색으로 시각화
        for strand in left_group:
            for point in strand:
                cv2.circle(image, tuple(point.astype(int)), 2, (255, 0, 0), -1)  # BGR 색상 코드

        # 오른쪽 그룹 스트랜드를 초록색으로 시각화
        for strand in right_group:
            for point in strand:
                cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)  # BGR 색상 코드

        return image

    def estimate_line_segments(self, checked_start_points):
        if len(checked_start_points) < 2:
            return []

        line_segments = []
        points = np.array(checked_start_points)

        # PCA 수행
        pca = PCA(n_components=1)
        pca.fit(points)

        # 데이터 포인트의 중심 계산
        center = np.mean(points, axis=0)

        # PCA로부터 얻은 주성분 방향
        direction = pca.components_[0]

        # points 내 모든 점 쌍 사이의 최대 거리 계산
        max_distance = max([np.linalg.norm(p1-p2) for i, p1 in enumerate(points) for p2 in points[i+1:]])

        # 선분의 시작점과 끝점 설정
        line_start = center - direction * (max_distance / 2)  # 선분 길이를 points의 최대 거리에 맞춤
        line_end = center + direction * (max_distance / 2)  # 선분 길이를 points의 최대 거리에 맞춤

        # 계산된 선분 추가
        line_segments.append((line_start, line_end))
        return line_segments

    def visualize_strand_endpoints(self, image, strands, color=(0, 255, 0), radius=3, thickness=-1):
        """
        스트랜드들의 끝점을 시각화합니다.

        :param image: 시각화를 수행할 원본 이미지
        :param strands: 스트랜드 데이터, 각 스트랜드는 점의 리스트임
        :param color: 원의 색상, 기본값은 초록색 (BGR)
        :param radius: 원의 반지름, 기본값은 3
        :param thickness: 원의 두께, 기본값은 -1로 원 내부를 채움
        :return: 시각화된 이미지
        """
        for strand in strands:
            if len(strand) > 0:  # 스트랜드가 비어있지 않은 경우
                endpoint = strand[-1]  # 스트랜드의 마지막 점
                cv2.circle(image, tuple(endpoint.astype(int)), radius, color, thickness)
        return image

    def visualize_strands(self, image, strands, color=(0, 255, 0), radius=3, thickness=-1):
        for strand in strands:
            if len(strand) > 0:  # 스트랜드가 비어있지 않은 경우
                for p in strand:
                    cv2.circle(image, tuple(p.astype(int)), radius, color, thickness)
        return image

    def detect_parting_line_or_not(self, checked_start_points, variance_ratio_threshold=0.7):
        # 스트랜드 시작점 추출
        if len(checked_start_points) < 2:
            return False, None

        points = checked_start_points
        line_segments = []

        # PCA 수행
        pca = PCA(n_components=2)
        pca.fit(points)

        # 첫 번째 주성분의 분산 비율 확인
        variance_ratio = pca.explained_variance_ratio_[0]

        # 첫 번째 주성분의 분산 비율이 임계값보다 높은 경우, 가르마 존재 가능성 판별
        if variance_ratio > variance_ratio_threshold:
            center = np.mean(points, axis=0)
            direction = pca.components_[0]
            line_start = center - direction * 50  # 선분 길이 조절
            line_end = center + direction * 50  # 선분 길이 조절
            line_segments.append((line_start, line_end))

            return True, line_segments  # 가르마 존재, 주성분 방향 반환
        else:
            return False, None  # 가르마 없음

    def visualize_strands_on_mask_boundary(self, image, strands, mask, color=(255, 0, 0), thickness=1):
        # 마스크에서 경계 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 모든 스트랜드의 끝점을 검사하여 경계 내에 존재하는지 확인
        for strand in strands:
            end_point = strand[-1]
            if cv2.pointPolygonTest(contours[0], tuple(end_point), False) == 0:
                # 스트랜드 시각화
                for i in range(len(strand) - 1):
                    cv2.line(image, tuple(strand[i].astype(int)), tuple(strand[i+1].astype(int)), color, thickness)

        return image

    def rotate_orientation_map(self, angle_map):
        """오리엔테이션 맵의 각도를 180도 회전시킵니다."""
        rotated_angle_map = np.mod(angle_map + np.pi, 2 * np.pi)
        return rotated_angle_map

    def find_largest_contour(self, contours):
        """
        주어진 윤곽선 리스트에서 가장 큰 윤곽선을 찾습니다.

        :param contours: `findContours` 함수로부터 얻은 윤곽선 리스트
        :return: 가장 큰 윤곽선, 해당 윤곽선의 면적
        """
        if not contours:
            return None, 0  # 윤곽선이 없는 경우

        # 각 윤곽선의 면적을 계산하고, 가장 큰 면적과 해당 윤곽선을 찾습니다.
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        return largest_contour, max_area

    def fill_gaps_in_contour(self, contour, max_gap_length=5):
        """
        윤곽선의 선분에서 주어진 최대 간격보다 큰 빈 점들을 채웁니다.

        :param contour: 채울 윤곽선, numpy 배열 형태
        :param max_gap_length: 두 점 사이의 최대 허용 간격, 이 값보다 큰 간격에 대해 점을 삽입
        :return: 보간된 윤곽선
        """
        if contour is None:
            return contour
        filled_contour = [contour[0]]  # 시작점 추가
        for i in range(1, len(contour)):
            current_point = contour[i]
            previous_point = contour[i - 1]

            # 두 점 사이의 거리 계산
            distance = np.linalg.norm(current_point - previous_point)

            if distance > max_gap_length:
                # 두 점 사이에 새로운 점들을 선형적으로 추가
                num_points_to_add = int(distance // max_gap_length)
                for j in range(1, num_points_to_add + 1):
                    new_point = previous_point + (current_point - previous_point) * (j / (num_points_to_add + 1))
                    filled_contour.append(new_point.astype(int))

            filled_contour.append(current_point)

        return np.array(filled_contour)

    def summarize_cluster_strands(self, strands):
        all_points = np.concatenate(strands, axis=0)
        mean_strand = np.mean(all_points, axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_points)
        components = pca.components_

        var_strand_1 = mean_strand + components[0] * np.max(pca.transform(all_points)[:, 0])
        var_strand_2 = mean_strand - components[0] * np.max(pca.transform(all_points)[:, 0])

        return mean_strand, var_strand_1, var_strand_2

    def visualize_strands_by_label(self, strands, labels, image ):
        # 고유 라벨 값과 색상 설정
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # 현재 라벨에 해당하는 스트랜드 선택
            label_strands = [strand for strand, l in zip(strands, labels) if l == label]
            mean_strand, var_strand_1, var_strand_2 = self.summarize_cluster_strands(label_strands)

            # # 클러스터 내 모든 스트랜드 시각화
            # for strand in label_strands:
            #     for i in range(len(strand) - 1):
            #         cv2.line(image, tuple(strand[i].astype(int)), tuple(strand[i + 1].astype(int)), colors[label % len(colors)], 1)

            # 대표 스트랜드 시각화
            color = tuple(self.colors[label].tolist())
            for pt in [mean_strand, var_strand_1, var_strand_2]:
                cv2.circle(image, tuple(pt.astype(int)), 3, color, -1)

        return image

    def create_strands_from_mask_boundary_with_rotated_angle_map(self, image, hair_mask, angle_map, W=15, n_strands=100, strand_length=20, distance=3, gradiation=3):
        """마스크의 경계에서 시작하여 오리엔테이션 맵의 값을 180도 회전시킨 방향으로 스트랜드를 생성합니다."""
        # 마스크 erode
        # kernel = np.ones((5, 5), np.uint8)  # Erode를 위한 커널 설정
        # eroded_mask = cv2.erode(hair_mask, kernel, iterations=1)  # 마스크에 erode 적용

        # 오리엔테이션 맵 180도 회전
        rotated_angle_map = self.rotate_orientation_map(angle_map)

        strands = []
        contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, strands

        contours_largest, max_area = self.find_largest_contour(contours)
        contours_largest_interp = self.fill_gaps_in_contour(contours_largest)
        if contours_largest_interp is None:
            return image, strands
        boundary_points = contours_largest_interp.reshape(-1, 2)
        selected_indices = np.linspace(0, len(boundary_points) - 1, n_strands, dtype=int)
        selected_start_points = boundary_points[selected_indices][:,::-1]

        for point in selected_start_points:
            # 회전된 오리엔테이션 맵을 사용하여 스트랜드 생성
            _path = self.hair_angle_calculator.cal_flow_path(point, hair_mask, image, distance, W, strand_length, rotated_angle_map)
            if len(_path) > 0:
                strands.append(np.asarray(_path))

        img_edge = image.astype(np.uint8) * hair_mask[:,:,np.newaxis] * 255
        if strands:
            strands = self.approximate_bezier(strands, strand_length)
            np.random.seed(42)
            color_list = np.random.randint(255, size=(len(strands), 3))
            for i, strand in enumerate(strands):
                color = tuple(color_list[i].tolist())
                for j in range(len(strand) - 1):
                    cv2.line(img_edge, tuple(strand[j].astype(int)), tuple(strand[j+1].astype(int)), color, 2)


        # 모든 윤곽선을 이미지에 그립니다.
        cv2.drawContours(img_edge, contours_largest_interp, -1, (0,255,0), 2)
        return img_edge, strands


    def extend_strands_with_orientation_map(self, strand, orientation_map, image, hair_mask, distance=1, W=15, max_length=10):
        """
        대표 스트랜드들의 양끝점에서 오리엔테이션 맵에 따라 성장시킵니다.

        :param strands: 대표 스트랜드 리스트
        :param orientation_map: 오리엔테이션 맵
        :param distance: 성장시킬 때 한 단계의 거리
        :param W: 오리엔테이션 맵의 윈도우 크기
        :param max_length: 성장시킬 최대 길이
        :return: 연장된 스트랜드 리스트
        """
        start_point = strand[0][::-1]
        end_point = strand[-1][::-1]

        strand = strand.reshape(-1, 2)

        # 시작점에서 역방향으로 성장
        rotated_map = self.rotate_orientation_map(orientation_map)
        backward_path = self.hair_angle_calculator.cal_flow_path(start_point, hair_mask, image, distance, W, max_length, rotated_map)
        print("b:", backward_path[::-1].shape)

        if len(backward_path) > 0:
            backward_path = np.array(backward_path[::-1])
            strand = np.vstack((backward_path, strand))


        # 끝점에서 순방향으로 성장
        forward_path =  self.hair_angle_calculator.cal_flow_path(end_point, hair_mask, image, distance, W, max_length, orientation_map)
        print("f:", forward_path.shape)

        if len(forward_path) > 0:
            # 전체 경로 결합
            strand = np.vstack((strand, forward_path[1:]))

        return strand


    def calculate_orientation_change(self, orientation_map):
        """
        오리엔테이션 맵에서 각 픽셀 주변의 각도 변화를 계산합니다.

        :param orientation_map: 각 픽셀의 방향성을 나타내는 오리엔테이션 맵
        :return: 각도 변화 맵
        """
        # 그래디언트 계산을 위해 Sobel 연산자 사용
        grad_x = cv2.Sobel(orientation_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(orientation_map, cv2.CV_64F, 0, 1, ksize=3)

        # 각도 변화량 계산
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return magnitude

    def visualize_high_orientation_change(self, image, orientation_change_map, threshold=1.0):
        """
        각도 변화가 특정 임계값 이상인 부분을 시각화합니다.

        :param image: 원본 이미지
        :param orientation_change_map: 각도 변화 맵
        :param threshold: 각도 변화의 임계값
        :return: 시각화된 이미지
        """
        # 임계값 이상의 각도 변화를 가진 픽셀 마스크 생성
        mask = orientation_change_map > threshold

        # 시각화를 위한 이미지 복사
        vis_image = image.copy()

        # 임계값 이상인 부분을 빨간색으로 표시
        vis_image[mask] = [0, 0, 255]

        return vis_image

    def calculate_angle(self, eye1, eye2, parting_point):
        """
        눈의 끝점들과 가르마의 중점 사이의 각도를 계산합니다.

        :param eye1: 첫 번째 눈의 끝점 좌표 (x1, y1)
        :param eye2: 두 번째 눈의 끝점 좌표 (x2, y2)
        :param parting_point: 가르마의 중점 좌표 (px, py)
        :return: 계산된 각도 (도 단위)
        """
        # 눈의 중앙점 계산
        eye_center = (640*(eye1.x + eye2.x) / 2, 480*(eye1.y + eye2.y) / 2)

        # 가르마 중점과 눈의 중앙점 사이의 벡터 계산
        dx = parting_point[0] - eye_center[0]
        dy = parting_point[1] - eye_center[1]

        # 아크탄젠트 함수를 사용하여 각도 계산 (라디안)
        angle_radians = np.arctan2(dy, dx)

        # 라디안을 도로 변환
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def visualize_line_between_eyes_and_parting_point(self, image, eye1, eye2, parting_point):
        """
        눈의 중앙점과 가르마의 중점 사이에 직선을 그립니다.

        :param image: 시각화할 이미지
        :param eye1: 첫 번째 눈의 끝점 좌표 (x1, y1)
        :param eye2: 두 번째 눈의 끝점 좌표 (x2, y2)
        :param parting_point: 가르마의 중점 좌표 (px, py)
        """
        # 눈의 중앙점 계산
        eye_center = (int(640*(eye1.x + eye2.x) / 2), int(480*(eye1.y + eye2.y) / 2))

        # 가르마의 중점 좌표를 이미지 스케일에 맞게 조정
        parting_point_scaled = (int(parting_point[0]), int(parting_point[1]))

        # 직선 그리기
        cv2.line(image, eye_center, parting_point_scaled, (255, 0, 0), 2)

        return image


    def visualize_convex_hulls(self, strands, labels, image):
        """
        주어진 스트랜드 클러스터에 대해 Convex Hull을 계산하고 시각화합니다.

        :param strands: 모든 스트랜드의 배열
        :param labels: 각 스트랜드에 해당하는 클러스터 라벨
        """
                # 고유 라벨 추출
        unique_labels = np.unique(labels)

        hulls = []

        for label in unique_labels:
            # 현재 라벨에 속하는 모든 스트랜드 포인트 선택
            label_strands = [strand for strand, l in zip(strands, labels) if l == label]
            points = np.concatenate(label_strands, axis=0).reshape(-1, 1, 2).astype(np.int32)

            # Convex Hull 계산
            hull = cv2.convexHull(points)
            hulls.append(hull)
            # Convex Hull 시각화
            cv2.polylines(image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

        return image, hulls

    def find_longest_distance_in_convex_hull(self, hull_points):
        """
        Convex Hull 내 포인트들 중 가장 긴 거리를 구합니다.

        :param hull_points: Convex Hull을 구성하는 포인트의 배열
        :return: 가장 긴 거리
        """
        max_distance = 0  # 최대 거리를 저장할 변수

        # 모든 포인트 쌍에 대해 반복
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                # 포인트 쌍 사이의 유클리드 거리 계산
                distance = np.linalg.norm(hull_points[i] - hull_points[j])

                # 최대 거리 업데이트
                if distance > max_distance:
                    max_distance = distance

        return max_distance

    def calculate_representative_strand(self, strands):
        """각 스트랜드 그룹의 평균 좌표를 사용하여 대표 스트랜드를 계산합니다."""
        # all_points = np.concatenate(strands, axis=0)
        # print(all_points)
        return np.mean(strands, axis=0).reshape(-1, 2)

    def visualize_representative_strands(self, image, strands, labels, hulls, hair_mask, angle_map, out, mean_parting_point):
        """라벨별 대표 스트랜드를 이미지에 시각화하고 이미지를 반환합니다."""
        # 라벨별로 스트랜드 그룹화
        unique_labels = np.unique(labels)
        representative_strands = {}

        for label, hull in zip(unique_labels, hulls):
            # 현재 라벨에 속하는 스트랜드 선택
            current_strands = [strand for strand, l in zip(strands, labels) if l == label]
            # 대표 스트랜드 계산
            representative_strand = self.calculate_representative_strand(current_strands)
            representative_strand = self.extend_strands_with_orientation_map(representative_strand, angle_map, image, hair_mask, 5, 15, 100)# strands, orientation_map, image, hair_mask, distance=1, W=15, max_length=10)
            representative_strand = self.approximate_bezier([representative_strand], 20, 0)[0]
            # if not self.switch:
            #     self.parameters = self.parameterize_strand(representative_strand, out, mean_parting_point)
            #     self.switch = True
            # if self.parameters is not None:
            #     representative_strand = self.reconstruct_strand(self.parameters, out, mean_parting_point)

            # representative_strand =self.approximate_polynomial_curve(representative_strand, extend_length=0, degree=1)
            # self.extend_curve_to_match_distance(, self.find_longest_distance_in_convex_hull(hull))
            representative_strands[label] = representative_strand
            # 대표 스트랜드 시각화
            for i in range(len(representative_strand) - 1):
                pt1 = tuple(representative_strand[i].astype(int))
                pt2 = tuple(representative_strand[i+1].astype(int))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)  # 선으로 그림

        return image, representative_strands

    def cluster_orientation_map(self, orientation_map, n_clusters=3):
        # 오리엔테이션 맵의 각 픽셀에 대해 (x, y, 방향) 형태의 특성 데이터 생성
        coords = np.dstack(np.meshgrid(np.arange(orientation_map.shape[1]), np.arange(orientation_map.shape[0]))).reshape(-1, 2)
        features = np.hstack((coords, orientation_map.reshape(-1, 1)))

        # K-Means 클러스터링 수행
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_.reshape(orientation_map.shape)

        return labels

    def visualize_clusters(self, image, labels):
        # 각 클러스터에 대해 다른 색상 할당
        cluster_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # RGB
        colored_image = np.zeros((*labels.shape, 3), dtype=np.uint8)

        for i in range(cluster_colors.shape[0]):
            colored_image[labels == i] = cluster_colors[i]

        return colored_image

    def approximate_polynomial_curve(self, points, extend_length, degree=3):
        """주어진 포인트를 사용하여 다차 방정식으로 곡선을 근사화합니다."""
        x = points[:, 0]
        y = points[:, 1]
        # 다차 방정식 계수 계산
        coefficients = np.polyfit(x, y, degree)
        # 다차 방정식 모델 생성
        polynomial = np.poly1d(coefficients)
        # 다차 방정식으로 새로운 포인트 생성
        x_new = np.linspace(x.min() - extend_length, x.max() + extend_length, 30)
        y_new = polynomial(x_new)
        return np.column_stack((x_new, y_new))

    def extend_curve_to_match_distance(self, curve_points, target_distance):
        # 커브의 시작점과 끝점
        start_point = curve_points[0]
        end_point = curve_points[-1]

        # 시작점과 끝점 사이의 거리 계산
        current_distance = np.linalg.norm(end_point - start_point)

        # 거리 비율 계산
        distance_ratio = target_distance / current_distance

        # 커브 포인트들을 조정하여 거리 비율에 맞춤
        adjusted_curve_points = np.array([(start_point + (point - start_point) * distance_ratio) for point in curve_points])

        return adjusted_curve_points


    def is_flyaway_hair(self,clusters_areas, hulls, image, threshold_ratio=0.8):
        """
        Convex Hull 내부에 머리카락 이외의 이미지가 많은 경우 Flyaway로 판별합니다.

        :param hair_mask: 머리카락 마스크 이미지 (binary image)
        :param convex_hull: Convex Hull을 정의하는 포인트의 배열
        :param threshold: Flyaway 판별을 위한 임계값
        :return: Flyaway 여부 (True/False)
        """
        flyaway_results = []

        for hull, cluster_area in zip(hulls, clusters_areas):
            # Convex Hull 내부를 채워 마스크 생성
            hull_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 255)

            # Convex Hull 영역의 크기 계산
            hull_area = cv2.countNonZero(hull_mask)
            print("hull:", hull_area)
            # 클러스터 영역의 총합 계산
            total_cluster_area = cluster_area
            print("cluster:", total_cluster_area)
            # 클러스터 영역과 Convex Hull 영역의 비율 계산
            area_ratio = total_cluster_area / hull_area if hull_area > 0 else 0

            # 비율이 임계값보다 낮으면 Flyaway로 판별
            is_flyaway = area_ratio < threshold_ratio
            flyaway_results.append(is_flyaway)

        return flyaway_results

    def parameterize_strand(self, strand, landmarks, parting_point):
        """
        스트랜드를 얼굴 랜드마크와 가르마 위치를 기준으로 파라미터화합니다.

        :param strand: 스트랜드 포인트의 리스트 [(x1, y1), (x2, y2), ...]
        :param landmarks: 얼굴 랜드마크 포인트의 리스트 [(lx1, ly1), (lx2, ly2), ...]
        :param parting_point: 가르마의 위치 (px, py)
        :return: 파라미터화된 스트랜드 정보
        """
        # 얼굴 랜드마크의 중심점 계산
        center_point = np.mean(landmarks, axis=0)

        # 스트랜드 파라미터화
        parameters = []
        for point in strand:
            # 가르마 위치로부터의 상대적 거리와 각도 계산
            relative_pos = np.array(point) - parting_point
            distance = np.linalg.norm(relative_pos)
            angle = np.arctan2(relative_pos[1], relative_pos[0])

            # 얼굴 중심점으로부터의 상대적 거리 계산 (옵션)
            center_distance = np.linalg.norm(np.array(point) - center_point)

            parameters.append((distance, angle, center_distance))

        return parameters

    def reconstruct_strand(self, parameters, landmarks, parting_point):
        """
        파라미터화된 정보를 기반으로 스트랜드를 재구성합니다.

        :param parameters: 파라미터화된 스트랜드 정보
        :param landmarks: 얼굴 랜드마크 포인트의 리스트
        :param parting_point: 가르마의 위치
        :return: 재구성된 스트랜드 포인트의 리스트
        """
        reconstructed_strand = []
        for distance, angle, _ in parameters:
            # 가르마 위치를 기준으로 절대 위치 계산
            x = parting_point[0] + distance * np.cos(angle)
            y = parting_point[1] + distance * np.sin(angle)
            reconstructed_strand.append([int(x), int(y)])

        return np.asarray(reconstructed_strand)

    # def clustering(self):

    # def mp_callback(self, segmentation_result: List[mp.Image], rgb_image: mp.Image, timestamp_ms: int):
    def mp_callback(self, segmentation_result, rgb_image):

        category_mask = segmentation_result.category_mask
        # confidence_mask = segmentation_result.confidence_mask

        image_data = rgb_image.numpy_view()

        fg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        bg_image[:] = BLACK_COLOR[0]

        condition1 = category_mask.numpy_view() == 1 # hair
        condition2 = category_mask.numpy_view() == 3
        condition3 = (category_mask.numpy_view() == 1) | (category_mask.numpy_view() == 2) | (category_mask.numpy_view() == 3)
        # condition3 = category_mask.numpy_view() != 0

        if np.sum(condition1) == 0:
            self.output_image = bg_image
        else:
            self.output_image = np.where(condition1, fg_image, bg_image)
        if np.sum(condition2) == 0:
            self.output_image_face = bg_image
        else:
            self.output_image_face = np.where(condition2, fg_image, bg_image)
        if np.sum(condition3) == 0:
            self.output_image_human = bg_image
            self.output_image_human_color = image_data[:,:,::-1]
        else:
            self.output_image_human = np.where(condition3, np.ones(image_data.shape[:2], dtype=np.uint8), bg_image)
            self.output_image_human_color = self.output_image_human[:,:,np.newaxis] * image_data[:,:,::-1]

    def main(self):
        cap = cv2.VideoCapture(0)
        opt = MyBaseOptions().parse()

        rate = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # frame = cv2.resize(frame, (512, 512)) # ( X, Y)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            # rgb_img.flags.writeable = False

            # Detect face landmarks from the input image.
            self.update(rgb_img)

            time.sleep(1/rate)

            if self.output_image is None:
                continue

            strand_map = img2strand(opt, rgb_img, self.output_image)
            strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB)
            # print(strand_rgb)
            # self.output_image = None
            result = np.vstack((frame, strand_rgb))
            cv2.imshow('MediaPipe FaceMesh', result)
            #Exit if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("segmented_image", Image, queue_size=1)
        self.depth_2d_pub = rospy.Publisher("estimated_depth", Image, queue_size=1)
        self.opt = MyBaseOptions().parse()
        self.rate = rospy.Rate(30)
        self.cv_image = None
        self.mode = "nn"
        self.size = 15
        self.switch = False
        self.parameters = None

        self.hair_angle_calculator = HairAngleCalculator(size=self.size, mode=self.mode)
        self.face_detector = MP()

        # rospy.Subscriber("/camera/color/image_rect_color/rotated_image", Image, self.image_callback)
        rospy.Subscriber("/camera/color/image_rect_color", Image, self.image_callback)
        # rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        # rospy.Subscriber("/blender/camera/image, Image, self.image_callback)

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)


    def main(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                self.update(self.cv_image)
            else:
                continue

            self.rate.sleep()

            if self.output_image is not None:
                try:
                    time_now=rospy.Time.now()

                    strand_map, angle_map = img2strand(self.opt, self.cv_image, self.output_image)
                    # print(angle_map)
                    # strand_rgb = cv2.cvtColor(strand_map, cv2.COLOR_BGR2RGB) #
                    strand_rgb, strands = self.create_hair_strands(np.zeros_like(self.cv_image), self.output_image, angle_map, W=1, n_strands=50, strand_length=100, distance=20)
                    #print(visualized_image)
                    strand_rgb_reversed, strands_reversed= self.create_strands_from_mask_boundary_with_rotated_angle_map(np.zeros_like(self.cv_image), self.output_image, angle_map, W=1, n_strands=100, strand_length=50, distance=5, gradiation=3)
                    # visualized_image = cv2.addWeighted(self.cv_image, 0.5, strand_rgb, 0.5, 2.2)
                    # visualized_image = self.cv_image
                    # visualized_image = strand_rgb_reversed
                    # visualized_image = cv2.addWeighted(self.cv_image, 0.5, strand_rgb_reversed, 0.5, 2.2)
                    # visualized_image = cv2.addWeighted(visualized_image, 0.5, strand_rgb_reversed, 0.5, 2.2)

                    # depth_2d_map = img2depth(self.opt, self.cv_image, self.output_image)
                    # depth_map_2d_msg= self.bridge.cv2_to_imgmsg(depth_2d_map, "bgr8")
                    # depth_map_2d_msg.header = Header(stamp=time_now)
                    # self.depth_2d_pub.publish(depth_map_2d_msg)

                    # orientation_change_map = self.calculate_orientation_change(angle_map)
                    # visualized_image = self.visualize_high_orientation_change(visualized_image, orientation_change_map, threshold=0.1)

                    # strand_rgb = cv2.rotate(strand_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # 분석 결과 계산 (예시 데이터 및 함수 호출, 실제 구현 필요)
                    # angle_results = self.calculate_strand_angles(strands)
                    # density_map = self.calculate_strand_density(strands, self.cv_image.shape)
                    # curvature_values = self.calculate_strand_curvature(strands)
                    # entropy_value = self.calculate_entropy(strands, self.cv_image.shape)
                    # print(angle_results)
                    #print(density_map)
                    # print(curvature_values)
                    # print(entropy_value)
                    # 결과 통합
                    # integrated_score = self.integrate_analysis_results(angle_results, density_map, curvature_values, entropy_value)
                    # print(integrated_score)
                    # 시각화
                    # visualized_image = self.visualize_tangled_areas(self.cv_image, strands, integrated_score)

                    # 가르마 위치 찾기
                    # parting_line, right_group, left_group = self.find_parting_line(strands)
                    # visualized_image = self.visualize_strand_groups(self.cv_image, left_group, right_group)

                    visualized_image, labels, cluster_areas = self.cluster_and_visualize_strands(self.cv_image, strands, n_clusters=self.n_clusters)
                    out = self.face_detector.run(self.cv_image)
                    visualized_image = self.face_detector.draw_landmarks_on_image(visualized_image, out)

                    # print(out.facial_transformation_matrixes)

                    # visualized_image = self.visualize_strands_by_label(strands, labels, visualized_image)
                    # 가르마 위치 시각화
                    # visualized_image = self.visualize_parting_line(self.cv_image, parting_line)

                    # 각 스트랜드의 방향 벡터 계산
                    #direction_vectors = self.calculate_direction_vectors(strands)

                    # print(direction_vectors)
                    direction_vectors_reversed = self.calculate_direction_vectors_reversed(strands_reversed)

                    # 방향 차이가 큰 스트랜드의 시작점 식별
                    checked_start_points, checked_strands = self.find_strands_with_large_direction_difference_reversed(strands_reversed, direction_vectors_reversed, 20)
                    visualized_image, hulls = self.visualize_convex_hulls(strands, labels, visualized_image)

                    hair_ratio = self.is_flyaway_hair(cluster_areas, hulls, self.cv_image)
                    print(hair_ratio)
                    # print(checked_start_points)
                    if checked_start_points is not None:
                        mean_parting_point = np.mean(checked_start_points, axis=0)
                        print("mean:", mean_parting_point)
                    # checked_start_points, checked_strands = self.find_strands_with_large_direction_difference(strands, direction_vectors, 30)

                    # print(checked_start_points)
                    # visualized_image = self.visualize_strands(visualized_image, checked_strands)
                    # visualized_image = self.visualize_strands_on_mask_boundary(visualized_image, strands, self.output_image)

                    # 시작점 시각화
                    # 클러스터 대표 선분 추정
                    # line_segments = self.estimate_line_segments(np.array(checked_start_points))
                    # if len(line_segments) > 0:
                    #     optimal_line_segment = line_segments[0]
                    #     cv2.line(visualized_image, tuple(optimal_line_segment[0].astype(int)), tuple(optimal_line_segment[1].astype(int)), (255, 0, 0), 2)

                    # has_parting_line, direction = self.detect_parting_line_or_not(checked_start_points, 0.6)
                    # if has_parting_line:
                    #     print("가르마가 존재합니다. 방향:", direction)

                    #     # 최적의 선분 선택 (여기서는 단순히 첫 번째 선분을 선택)
                    #     optimal_line_segment = direction[0]
                    #     cv2.line(visualized_image, tuple(optimal_line_segment[0].astype(int)), tuple(optimal_line_segment[1].astype(int)), (255, 0, 0), 2)
                    # else:
                    #     print("가르마가 존재하지 않습니다.")

                    # 스트랜드 끝점 시각화
                    # visualized_image = self.visualize_strand_endpoints(visualized_image, strands_reversed)

                    visualized_image = self.visualize_checked_start_points(visualized_image, checked_start_points)
                    face_landmarks_list = self.face_detector.get_landmarks()
                    face_landmarks_np = self.face_detector.face_landmarks_np
                    # print(face_landmarks_list)
                    # def visualize_representative_strands(self, image, strands, labels, hulls, image, hair_mask, angle_map):

                    visualized_image = self.visualize_representative_strands(visualized_image, strands, labels, hulls, self.output_image, angle_map, face_landmarks_np, mean_parting_point)

                    if len(face_landmarks_list) > 0 and not np.isnan(mean_parting_point).any():
                        eye1 = face_landmarks_list[0][self.eye1_ind]
                        eye2 = face_landmarks_list[0][self.eye2_ind]
                        angle = self.calculate_angle(eye1, eye2, mean_parting_point)
                        print(angle)
                        visualized_image = self.visualize_line_between_eyes_and_parting_point(visualized_image, eye1, eye2, mean_parting_point)

                    ros_image = self.bridge.cv2_to_imgmsg(visualized_image, "rgb8")
                    #ros_image = self.bridge.cv2_to_imgmsg(strand_rgb, "rgb8)"
                    ros_image.header = Header(stamp=time_now)
                    self.image_pub.publish(ros_image)
                except CvBridgeError as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
