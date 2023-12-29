import cv2
import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

import scripts.sample_cef_gpu as coh

from scipy.interpolate import griddata

class HairAngleCalculator:
    def __init__(self, size=15):
        self.size = size

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
        angles[non_zero_mask] = (torch.pi + torch.atan2(nominator[non_zero_mask], denominator[non_zero_mask])) / 2

        return angles

    def get_line_ends(self, i, j, W, tang):
        if -1 <= tang and tang <= 1:
            begin = (i, int((-W/2) * tang + j + W/2))
            end = (i + W, int((W/2) * tang + j + W/2))
        else:
            begin = (int(i + W/2 + W/(2 * tang)), j + W//2)
            end = (int(i + W/2 - W/(2 * tang)), j - W//2)
        return (begin, end)

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
                        (begin, end) = self.get_line_ends(i, j, W, tang.item())  # .item() is used to convert the tensor to a Python number
                        xy_list.append((begin, end))
                        cv2.line(result, begin, end, color=(255,255,255))
                        #cv2.resize(result, im.shape[:2], result)
        return xy_list, result

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
        print(result.shape)

        return xy_list, result

    def process_image(self, frame, hair_mask):
        resize_cef = 1
        frame = cv2.resize(frame, None, fx=resize_cef, fy=resize_cef)
        hair_mask = cv2.resize(hair_mask, None, fx=resize_cef, fy=resize_cef)
        # frame = self.equalize_image(frame)
        gray_map=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cof_map = coh.coherence_filter(gray_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)
        # cof_map = gray_map

        angle_map = self.calculate_angles(torch.Tensor(cof_map), self.size)
        xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, self.size)
        debug_map = cv2.addWeighted(frame, 0.5, angle_visual,0.5, 2.2)

        # _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, self.size)
        # if np.max(angle_color) != 0:
        #     angle_color= img_as_ubyte(angle_color / np.max(angle_color))
        # else:
        #     angle_color= img_as_ubyte(angle_color)
        # angle_color_map = cv2.applyColorMap(angle_color, cv2.COLORMAP_JET)
        # angle_color_map = cv2.resize(angle_color_map, None, fx=1/resize_cef, fy=1/resize_cef)

        debug_map = cv2.resize(debug_map, None, fx=1/resize_cef, fy=1/resize_cef)

        return debug_map
