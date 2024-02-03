import cv2
import onnxruntime
import numpy as np
import math
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

import scripts.sample_cef_gpu as coh

from scipy.interpolate import griddata

import matplotlib.pyplot as plt

import colorsys

class HairAngleCalculator:
    def __init__(self, size=15, mode="strip"):
        self.size = size
        self.mode = mode

        self.padding = self.size//2
        self.num_angles = 180 # for gabor filter
        self.psi = 0 # for gabor filter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_gabor_batch(self, size, sigma, thetas, lambd, gamma, psi):
        x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="xy")
        x = x.view(1, 1, size, size).repeat(len(thetas), 1, 1, 1).to(thetas.device)
        y = y.view(1, 1, size, size).repeat(len(thetas), 1, 1, 1).to(thetas.device)
        x_thetas = x * torch.cos(thetas.view(-1, 1, 1, 1)) + y * torch.sin(thetas.view(-1, 1, 1, 1))
        y_thetas = -x * torch.sin(thetas.view(-1, 1, 1, 1)) + y * torch.cos(thetas.view(-1, 1, 1, 1))
        gabor = torch.exp(-0.5 * (x_thetas ** 2 + gamma ** 2 * y_thetas ** 2) / sigma ** 2) * torch.cos(2 * np.pi * x_thetas / lambd + psi)
        return gabor

    def seg_hair(self, image, color_mask=[]):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        image = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device)
        num_angles = self.num_angles
        orientations = torch.linspace(0, np.pi, num_angles).to(self.device)
        gabors = self.create_gabor_batch(self.size, 10, orientations, 20, 1, self.psi).to(self.device) # 15, 0.3, 180, 0.8, 1, 0 # 15 5 180 10 1

        filtered_images = F.conv2d(image, gabors, padding=self.padding).squeeze(0)
        orientation_map = torch.argmax(filtered_images, axis=0)

        # theta_xy = torch.argmax(filtered_images, axis=0)
        # f_xy = torch.max(torch.pow(torch.abs(filtered_images), 0.5))

        # psi_xy = torch.exp(1j * 2 * theta_xy)
        # product = f_xy * psi_xy
        # orientation_map = torch.where(f_xy > 0, product, 0)

        orientation_map = orientation_map.cpu().numpy()
        # orientation_map = np.angle(orientation_map)

        if color_mask.any():
            color_mask = color_mask / 255
            orientation_map = orientation_map * color_mask

        float_map = orientation_map / self.num_angles
        # float_map = np.angle(orientation_map) / np.pi
        print("max:", np.max(float_map*np.pi))
        print("min:", np.min(float_map*np.pi))
        color_map = cv2.applyColorMap(np.uint8(float_map*255), cv2.COLORMAP_JET) # blue colors is smaller than red colors

        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # color_map = cv2.addWeighted(image, 0.7, color_map, 0.3, 0)
        return color_map, float_map*np.pi

    def equalize_image(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return image

    def find_hair_strands(self, image, gabor_filters, start_point, num_steps, step_size=10):
        # Initialize the path with the start point
        path = [start_point]
        # Convert image to grayscale if not already
        for i in range(num_steps):
            current_point = path[-1]

            responses = gabor_filters[current_point[0], current_point[1]]
            # Find the direction with the highest response
            if responses > int(self.num_angles/2):
                direction = np.pi/self.num_angles * responses + self.psi + np.pi
            else:
                direction = np.pi/self.num_angles * responses + self.psi

            # Calculate the next point in the direction of the highest response
            next_point = (current_point[0] + int(step_size*np.cos(direction))), current_point[1] + int(step_size*(np.sin(direction)))
            # Add the next point to the path
            path.append(next_point)

        # Draw the path on the image
        image_path = image.copy()
        for point in path:
            cv2.circle(image_path, point, 5, (0, 0, 255), -1)

        return path, image_path

    def calculate_angles(self, im, W):
        # Make sure the image is on the GPU and is a float32
        im = im.to('cuda:0').float()

        # Calculate the gradients
        Gx_, Gy_ = torch.gradient(im, axis=(1, 0))

        # reshape tensors to allow vectorized computation
        Gx_ = Gx_.unfold(0, W, W).unfold(1, W, W)
        Gy_ = Gy_.unfold(0, W, W).unfold(1, W, W)

        # Calculate the local nominator and denominator
        nominator = 2 * Gx_ * Gy_
        denominator = Gx_.pow_(2) - Gy_.pow_(2)
        # Sum over the local windows

        nominator = nominator.sum(dim=(2,3))
        denominator = denominator.sum(dim=(2,3))

        # Calculate the angles
        angles = torch.zeros_like(nominator)
        non_zero_mask = (nominator != 0) | (denominator != 0)
        try:
            angles[non_zero_mask] = (torch.pi + torch.atan2(nominator[non_zero_mask], denominator[non_zero_mask])) / 2
        except:
            angles[non_zero_mask] = (math.pi + torch.atan2(nominator[non_zero_mask], denominator[non_zero_mask])) / 2

        return angles

    def get_line_ends(self, i, j, W, tang, im):
        if -1 <= tang and tang <= 1:
            begin = (i, int((-W/2) * tang + j + W/2))
            end = (i + W, int((W/2) * tang + j + W/2))
        else:
            begin = (int(i + W/2 + W/(2 * tang)), int(j + W//2))
            end = (int(i + W/2 - W/(2 * tang)), int(j - W//2))
            # begin = (int(i + W/2 + W/(2 * tang)), int(j + W//2)) # original code
            # end = (int(i + W/2 - W/(2 * tang)), int(j - W//2))
        return (begin, end)

    def get_3d_point(self, point ,depth_image, camera_info):
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        y = np.clip(point[1], 0, depth_image.shape[0]-1)
        x = np.clip(point[0], 0, depth_image.shape[1]-1)

        depth = depth_image[y, x]

        z = depth / 1000.0  # 뎁스 값을 미터 단위로 변환
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        return np.array([x_3d, y_3d, z])

    def visualize_angles(self, im, mask, angles, W):
        (y, x) = im.shape[:2]
        xy_list = []
        result = cv2.cvtColor(np.zeros(im.shape[:2], np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (W-1)**2
        for i in range(1, x, W):
            for j in range(1, y, W):
                radian = torch.sum(mask[j - 1:j + W, i-1:i+W])
                if radian > mask_threshold:
                    idx_i = (i - 1) // W
                    idx_j = (j - 1) // W
                    if idx_j < angles.shape[0] and idx_i < angles.shape[1]:  # Safety check
                        tang = torch.tan(angles[idx_j, idx_i])
                        (begin, end) = self.get_line_ends(i, j, W, tang.item(), im)  # .item() is used to convert the tensor to a Python number
                        xy_list.append((begin, end))
                        cv2.line(result, begin, end, color=(255,255,255))
                        #cv2.resize(result, im.shape[:2], result)
        return xy_list, result

    def visualize_angle_differences(self, im, mask, angles, W, angle_diff_threshold=np.pi/6, boundary_margin=10):
        (y, x) = im.shape[:2]
        xy_list = []
        result = cv2.cvtColor(np.zeros(im.shape[:2], np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (W-1)**2

        # PyTorch 텐서를 Numpy 배열로 변환
        if mask.is_cuda:
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.numpy().astype(np.uint8)
        # 마스크의 경계를 찾기
        edges = cv2.Canny(mask_np, 100, 200)

        for i in range(0, x, W):
            for j in range(0, y, W):
                radian = torch.sum(mask[j:j + W, i:i + W])

                if radian > mask_threshold:
                    idx_i = i // W
                    idx_j = j // W

                    # 마스크 경계로부터의 거리가 boundary_margin 이상인지 확인
                    if np.any(edges[j:j+W, i:i+W]) or i < boundary_margin or j < boundary_margin or i > x - W - boundary_margin or j > y - W - boundary_margin:
                        continue  # 마스크 경계 부분이거나, 이미지 경계에 가까운 그리드는 처리하지 않음

                    if idx_j < angles.shape[0] and idx_i < angles.shape[1]:
                        angle = angles[idx_j, idx_i].item()

                        surrounding_angles = angles[max(idx_j-1, 0):min(idx_j+2, angles.shape[0]), max(idx_i-1, 0):min(idx_i+2, angles.shape[1])]
                        mean_surrounding_angle = torch.mean(surrounding_angles).item()

                        angle_difference = abs(angle - mean_surrounding_angle)

                        color = (0, 255, 0) if angle_difference <= angle_diff_threshold else (0, 0, 255)

                        center = (i + W//2, j + W//2)
                        length = W // 2
                        end = (int(center[0] + length * np.cos(angle)), int(center[1] + length * np.sin(angle)))

                        xy_list.append((center, end))
                        cv2.line(result, center, end, color, 2)
        return xy_list, result

    def visualize_angle_differences_(self, im, mask, angles, W, angle_diff_threshold=np.pi/6, boundary_margin=10):
        (y, x) = im.shape[:2]
        result = cv2.cvtColor(np.zeros(im.shape[:2], np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (W-1)**2

        tangled_mask = np.zeros_like(mask.cpu().numpy(), dtype=np.uint8)

        # 마스크의 경계 찾기
        contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(0, x, W):
            for j in range(0, y, W):
                radian = torch.sum(mask[j:j + W, i:i + W])

                if radian > mask_threshold:
                    idx_i = i // W
                    idx_j = j // W

                    center = (i + W//2, j + W//2)

                    # 마스크 경계로부터의 거리가 boundary_margin 이상인지 확인
                    # distance_to_edge = min([cv2.pointPolygonTest(contour, center, True) for contour in contours])
                    # if distance_to_edge < boundary_margin:
                    #     continue  # 마스크 경계 근처는 제외

                    if idx_j < angles.shape[0] and idx_i < angles.shape[1]:
                        angle = angles[idx_j, idx_i].item()
                        surrounding_angles = angles[max(idx_j-1, 0):min(idx_j+2, angles.shape[0]), max(idx_i-1, 0):min(idx_i+2, angles.shape[1])]
                        mean_surrounding_angle = torch.mean(surrounding_angles).item()

                        angle_difference = abs(angle - mean_surrounding_angle)
                        if angle_difference > angle_diff_threshold:
                            tangled_mask[j:j+W, i:i+W] = 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tangled_mask, connectivity=8)

        for i in range(1, num_labels):  # 0번 라벨(배경)은 제외
            x, y, w, h, area = stats[i]
            if area < 1000:  # 너무 작은 영역은 무시
                continue
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, f'Area {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

    def visualize_color_angles(self, im, mask, angles, W):
        (y, x) = im.shape[:2]
        xy_list = []
        result = np.zeros(im.shape[:2], np.float32)
        mask_threshold = (W-1)**2
        for i in range(1, x, W):
            for j in range(1, y, W):
                radian = torch.sum(mask[j - 1:j + W, i-1:i+W])
                if radian > mask_threshold:
                    idx_i = (i - 1) // W
                    idx_j = (j - 1) // W
                    if idx_j < angles.shape[0] and idx_i < angles.shape[1]:
                        tang = angles[idx_j, idx_i].cpu().numpy()
                        result[j-1:j+W, i-1:i+W] = tang
        # print(result.shape)

        return xy_list, result

    def visualize_3d_angles(self, im, mask, angles, depth_image, camera_info, W):
        (y, x) = im.shape[:2]
        result = cv2.cvtColor(np.zeros(im.shape[:2], np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (W-1)**2
        for i in range(1, x, W):
            for j in range(1, y, W):
                radian = torch.sum(mask[j - 1:j + W, i-1:i+W])
                if radian > mask_threshold:
                    idx_i = (i - 1) // W
                    idx_j = (j - 1) // W
                    if idx_j < angles.shape[0] and idx_i < angles.shape[1]:  # Safety check
                        tang = torch.tan(angles[idx_j, idx_i])
                        (begin, end) = self.get_line_ends(i, j, W, tang.item(), im)  # .item() is used to convert the tensor to a Python number
                        begin_3d = self.get_3d_point(begin, depth_image, camera_info)
                        end_3d = self.get_3d_point(end, depth_image, camera_info)

                        direction_vector = end_3d - begin_3d
                        if np.linalg.norm(direction_vector) != 0:
                            direction_vector /= np.linalg.norm(direction_vector)
                        color = self.vector_to_hsv_color(direction_vector)
                        result[j-1:j+W, i-1:i+W] = color

                        #cv2.resize(result, im.shape[:2], result)
        return result

    def vector_to_hsv_color(self,vec3d):
        # 벡터의 방향을 HSV 색상으로 변환
        azimuth = np.arctan2(vec3d[1], vec3d[0]) / (2 * np.pi) + 0.5
        elevation = np.arctan2(vec3d[2], np.sqrt(vec3d[0]**2 + vec3d[1]**2)) / np.pi + 0.5
        if np.linalg.norm(vec3d) != 0:
            magnitude = np.linalg.norm(vec3d)
        else:
            magnitude = 0
        hue = azimuth  # 방향에 따른 색상
        saturation = elevation  # 높이에 따른 채도
        value = magnitude

        color = self.hsv_to_rgb(hue, saturation, value)
        return color

    def hsv_to_rgb(self, h, s, v):
        # HSV를 RGB로 변환
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def vector_to_rgb_color(self, vec3d):
        # 벡터의 각 성분을 RGB 색상으로 변환
        r = np.clip(vec3d[0], 0, 1)
        g = np.clip(vec3d[1], 0, 1)
        b = np.clip(vec3d[2], 0, 1)

        return (int(r * 255), int(g * 255), int(b * 255))


    def create_xyz_strips(self, xy_list, depth_image, camera_info):
        """
        xy_list와 depth_image를 기반으로 하는 3D 스트립 배열을 생성합니다.

        :param xy_list: 2D 좌표 목록 (x, y 쌍의 리스트)
        :param depth_image: 깊이 정보를 담고 있는 이미지 (각 픽셀의 깊이 값)
        :return: 시작점과 끝점을 담은 3D 좌표 리스트
        """
        xyz_strips = []
        # 카메라 내부 파라미터 사용
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        v, u = np.indices((depth_image.shape[0], depth_image.shape[1]))
        z = depth_image / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = np.concatenate((x[:,:, np.newaxis], y[:,:, np.newaxis], z[:,:,np.newaxis]), axis=2)

        for begin, end in xy_list:
            # 깊이 이미지에서 해당 xy 좌표의 깊이 값 추출
            # print("xyz:", xyz)
            # begin_x = np.clip(begin[1], 0, depth_image.shape[1]-1)
            # begin_y = np.clip(begin[0], 0, depth_image.shape[0]-1)
            # end_x = np.clip(end[1], 0, depth_image.shape[1]-1)
            # end_y = np.clip(end[0], 0, depth_image.shape[0]-1)
            begin_x = begin[1]
            begin_y = begin[0]
            end_x = end[1]
            end_y = end[0]

            # 시작점 (x, y, z)
            try:
                xyz_begin = np.array(xyz[begin_x, begin_y])
                # 끝점 (x, y, z)
                xyz_end = np.array(xyz[end_x, end_y])
            except Exception as e:
                continue

            if np.abs(xyz_end[2] - xyz_begin[2]) > 0.2:
                # print(xyz_end[2] - xyz_begin[2])
                continue

            # 시작점과 끝점 추가
            xyz_strips.append((xyz_begin, xyz_end))

        return np.asarray(xyz_strips)
    #cal xy to xy in angle map
    def cal_angle_map_xy(self, p_y, p_x, W):
        a_y = int(p_y // W)
        a_x = int(p_x // W)
        return a_x, a_y

    def cal_flow_path(self, p_0, mask, img, k, W, max_iter, angles, max_angle_change=np.pi/4):
        (height, width) = img.shape[:2]
        p_t = np.array(p_0, dtype=int)  # 안전한 타입 변환
        path_ = []
        prev_angle = None  # 이전 각도 초기화

        for _ in range(max_iter):
            if not (0 <= p_t[0] < height and 0 <= p_t[1] < width and mask[p_t[0], p_t[1]] > 0):
                break  # 범위를 벗어나거나 마스크 조건을 만족하지 않으면 종료

            path_.append(p_t[::-1].copy())  # 현재 위치를 경로에 추가

            a_x, a_y = self.cal_angle_map_xy(p_t[0], p_t[1], W)
            a_x, a_y = np.clip([a_x, a_y], [0, 0], [angles.shape[1]-1, angles.shape[0]-1])

            angle_t = angles[a_y, a_x]

            # if prev_angle is not None and abs(angle_t - prev_angle) > max_angle_change:
            #     break
            # prev_angle = angle_t  # 현재 각도를 이전 각도로 업데이트

            # 다음 위치 계산
            delta_x = int(k * np.cos(angle_t))
            delta_y = int(k * np.sin(angle_t))
            p_t += [delta_y, delta_x]  # y, x 순서 주의

        return np.asarray(path_)

    # calculate path
    # def cal_flow_path(self, p_0, mask, img, k, W, max_iter, angles):
    #     (y, x) = img.shape[:2]
    #     p_t = p_0
    #     path_ = []
    #     a_x, a_y = self.cal_angle_map_xy(p_t[0], p_t[1], W)
    #     a_x = np.clip(a_x, 0, angles.shape[1]-1)
    #     a_y = np.clip(a_y, 0, angles.shape[0]-1)
    #     angle_t = angles[a_y, a_x]
    #     i = 0
    #     while (mask[p_t[0],p_t[1]] > 0 and p_t[1] < x-1 and p_t[0] < y-1 and i < max_iter):
    #         path_.append(p_t[::-1])

    #         a_x, a_y = self.cal_angle_map_xy(p_t[0], p_t[1], W)
    #         if a_x == angles.shape[1]:
    #             a_x -= 1
    #         if a_y == angles.shape[0]:
    #             a_y -= 1

    #         angle_t = angles[a_y, a_x]

    #         p_t[1] = int(p_t[1] + k*np.cos(angle_t))
    #         p_t[0] = int(p_t[0] + k*np.sin(angle_t))

    #         if p_t[0]  < 0:
    #             p_t[0] = 0
    #         if p_t[1]  < 0:
    #             p_t[1] = 0
    #         if p_t[1]  > x-1:
    #             p_t[1] = x-1
    #         if p_t[0]  > y-1:
    #             p_t[0] = y-1
    #         i += 1

    #     return path_
    def find_most_tangled_area(self, angles, mask, W, angle_diff_threshold=np.pi/6, search_radius=2,boundary_margin=10):
        """
        마스크 영역 안에서 가장 흐트러진 영역을 찾되, 마스크 경계 부분은 제외합니다.
        """
        y, x = angles.shape
        tangled_scores = torch.zeros_like(angles)

        # 마스크의 경계를 찾기
        contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask.cpu().numpy(), dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, color=255, thickness=boundary_margin)

        for i in range(y):
            for j in range(x):
                if mask[i * W, j * W] == 0 or contour_mask[i * W, j * W] == 255:  # 마스크 영역 밖이거나 경계 부분은 무시
                    continue

                current_angle = angles[i, j]
                score = 0

                # 주변 영역 탐색
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        ny, nx = i + dy, j + dx
                        if 0 <= ny < y and 0 <= nx < x and mask[ny * W, nx * W] != 0 and contour_mask[ny * W, nx * W] != 255:  # 마스크 영역 내부 그리고 경계가 아닌 그리드만 고려
                            angle_diff = abs(current_angle - angles[ny, nx])
                            if angle_diff > angle_diff_threshold:
                                score += 1

                tangled_scores[i, j] = score

        max_score, max_index = torch.max(tangled_scores.view(-1), 0)
        max_index = max_index.item()
        max_y, max_x = divmod(max_index, x)

        return max_y * W + W // 2, max_x * W + W // 2

    def visualize_most_tangled_area(self, image, mask, angles, W, angle_diff_threshold=np.pi/6, search_radius=2, box_size=50):
        """
        가장 흐트러진 영역을 이미지 위에 시각화합니다.

        :param image: 원본 이미지입니다.
        :param angles: 각 그리드의 평균 각도를 포함하는 2D 텐서입니다.
        :param W: 각 그리드의 크기입니다.
        :param angle_diff_threshold: 각도 차이의 임계값입니다.
        :param search_radius: 주변 영역을 탐색할 반경입니다.
        :param box_size: 시각화할 영역의 크기입니다.
        :return: 가장 흐트러진 영역이 시각화된 이미지입니다.
        """
        # 가장 흐트러진 영역의 중심 그리드 위치 찾기
        center_y, center_x = self.find_most_tangled_area(angles, mask, W, angle_diff_threshold, search_radius)

        # 시각화할 이미지 복사
        vis_image = image.copy()

        # 가장 흐트러진 영역을 나타내는 사각형 그리기
        top_left = (int(center_x - box_size / 2), int(center_y - box_size / 2))
        bottom_right = (int(center_x + box_size / 2), int(center_y + box_size / 2))
        cv2.rectangle(vis_image, top_left, bottom_right, (0, 0, 255), 2)  # 빨간색 사각형

        return vis_image


    def find_tangled_areas(self, angle_differences, angle_diff_threshold=np.pi/6, min_area=50):
        # 각도 차이가 임계값보다 큰 부분을 1로, 작거나 같은 부분을 0으로 하는 바이너리 이미지 생성
        tangled_mask = (angle_differences > angle_diff_threshold).astype(np.uint8)

        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tangled_mask, connectivity=8)

        tangled_areas = []
        for i in range(1, num_labels):  # 0번 라벨(배경)은 제외
            x, y, w, h, area = stats[i]
            if area >= min_area:  # 최소 영역 크기보다 큰 컴포넌트만 선택
                tangled_areas.append((x, y, w, h))

        return tangled_areas

    def visualize_tangled_areas(self, image, tangled_areas):
        # 이미지의 복사본을 생성하여 원본 이미지를 보존
        vis_image = image.copy()

        # 흐트러진 부분이 밀집된 영역을 사각형으로 그리기
        for i, (x, y, w, h) in enumerate(tangled_areas):
            # 사각형 그리기
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 영역 번호 텍스트 추가
            cv2.putText(vis_image, f'Area {i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis_image

    def process_image(self, frame, hair_mask, depth_image, camera_info):
        # resize_cef = 1
        # frame = cv2.resize(frame, None, fx=resize_cef, fy=resize_cef)
        # hair_mask = cv2.resize(hair_mask, None, fx=resize_cef, fy=resize_cef)

        # frame = self.equalize_image(frame)
        gray_map=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, otsu_map = cv2.threshold(gray_map, 0, 128, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #cof_map = coh.coherence_filter(gray_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)
        # cof_map = gray_map
        xyz_strips = None
        debug_map = None
        angle_map = self.calculate_angles(torch.Tensor(gray_map), self.size)
        if self.mode == "strip":
            # xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, self.size)
            # angle_visual= self.visualize_angle_differences_(gray_map, torch.Tensor(hair_mask), angle_map, self.size, np.pi/6, 0)
            xy_list, angle_visual= self.visualize_angle_differences(gray_map, torch.Tensor(hair_mask), angle_map, self.size, np.pi/6, 0)

            # angle_visual = self.visualize_most_tangled_area(frame, torch.Tensor(hair_mask), angle_map, self.size, np.pi/3, 10, 80)
            # xyz_strips = self.create_xyz_strips(xy_list, depth_image, camera_info)
            # debug_map = cv2.addWeighted(frame, 0.5, angle_visual, 0.5, 2.2)
            debug_map = angle_visual

            # debug_map = angle_most_visual
            return debug_map, xyz_strips, angle_map
        elif self.mode == "gabor":
            color_map, orientation_map = self.seg_hair(gray_map, hair_mask)
            # return cv2.cvtColor(cof_map, cv2.COLOR_GRAY2RGB), None
            return color_map, orientation_map
        elif self.mode == "color":
            _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, self.size)

            angle_color= angle_color / np.pi
            angle_color_map = cv2.applyColorMap(np.uint8(angle_color*255), cv2.COLORMAP_JET)
            # angle_color_map v= cv2.GaussianBlur(angle_color_map, (self.size, self.size), 0)
            return angle_color_map, xyz_strips, angle_map
        elif self.mode == "3d_color":
            angle_color= self.visualize_3d_angles(gray_map, torch.Tensor(hair_mask), angle_map, depth_image, camera_info, self.size)
            # angle_color_map = cv2.GaussianBlur(angle_color_map, (self.size, self.size), 0)
            return angle_color, xyz_strips, angle_map
        # angle_color_map = cv2.resize(angle_color_map, None, fx=1/resize_cef, fy=1/resize_cef)

        # debug_map = cv2.resize(debug_map, None, fx=1/resize_cef, fy=1/resize_cef)
        # cv2.cvtColor(cof_map, cv2.COLOR_GRAY2BGR)
