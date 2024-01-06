import cv2
import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

# from cuml import KMeans
# from cuml.cluster import DBSCAN
from sklearn.cluster import DBSCAN, KMeans

import sample_cef_gpu as coh

import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib as mpl


from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter

import skimage

import torch.fft

from .mp_seg import App
class AngleCalculator:
     def __init__(self):
          self.prev_angles = None
     def calculate_angles(self, im, W=15, diff_threshold=0.5):
          # Ensure the image is on the GPU and is a float32
          im = im.to('cuda:0').float()

          # Compute the gradients
          Gx_, Gy_ = torch.gradient(im, axis=(1, 0))

          # reshape tensors to allow vectorized computation
          Gx_small = Gx_.unfold(0, W, W).unfold(1, W, W)
          Gy_small = Gy_.unfold(0, W, W).unfold(1, W, W)

          # Compute the local nominator and denominator
          nominator_small = 2 * Gx_small * Gy_small
          denominator_small = Gx_small ** 2 - Gy_small ** 2

          # Sum over the local windows
          nominator_small = nominator_small.sum(dim=(2,3))
          denominator_small = denominator_small.sum(dim=(2,3))

          # Compute the angles
          angles_small = torch.where(
               (nominator_small != 0) | (denominator_small != 0),
               (torch.pi + torch.atan2(nominator_small, denominator_small)) / 2,
               torch.zeros_like(nominator_small)
          )

          # Compute the difference with previous angles if they exist
          if self.prev_angles is not None:
               angle_diff = torch.abs(angles_small - self.prev_angles)
               unstable_angles = angle_diff > diff_threshold
          else:
               unstable_angles = torch.zeros_like(angles_small, dtype=torch.bool)

          # Save the current angles for the next frame
          self.prev_angles = angles_small.clone()

          # Compute angles with a larger window
          Gx_large = Gx_.unfold(0, W*2+1, W*2+1).unfold(1, W*2+1, W*2+1).unfold(2, W*2+1, W*2+1)
          Gy_large = Gy_.unfold(0, W*2+1, W*2+1).unfold(1, W*2+1, W*2+1).unfold(2, W*2+1, W*2+1)
          nominator_large = 2 * Gx_large * Gy_large
          denominator_large = Gx_large ** 2 - Gy_large ** 2
          nominator_large = nominator_large.sum(dim=(2,3))
          denominator_large = denominator_large.sum(dim=(2,3))
          angles_large = torch.where(
               (nominator_large != 0) | (denominator_large != 0),
               (torch.pi + torch.atan2(nominator_large, denominator_large)) / 2,
               torch.zeros_like(nominator_large)
          )

          # Trim angles_large to match the shape of angles_small and unstable_angles
          # Get the start and end indices for trimming
          # Compute the start indices for trimming
          # Interpolate angles_W2 to match the shape of angles_W1
          print(angles_large.shape)
          angles_large_resized = F.interpolate(angles_large[None, None, :, :], size=angles_small.shape, mode='bilinear', align_corners=False)[0, 0]

          # Correct unstable angles
          corrected_angles = torch.where(unstable_angles, angles_large_resized, angles_small)

          return corrected_angles
class HairSegmenter:
     def __init__(self, onnx_session, input_size, score_thresh):
          self.onnx_session = onnx_session
          self.input_size = input_size
          self.score_thresh = score_thresh
          self.mp_face_mesh = mp.solutions.face_mesh
          self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
          self.size = 15 # gabor filter size
          self.padding = self.size//2
          self.num_angles = 72 # for gabor filter
          self.psi = np.pi/2 # for gabor filter
          self.angle_calculator = AngleCalculator()

          self.avg_color = np.asarray([])
          self.std_color = np.asarray([])

          self.before_hair_mask=[]

          self.app = App()
     def create_gabor_batch(self, size, sigma, thetas, lambd, gamma, psi):
          x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="xy")
          x = x.view(1, 1, size, size).repeat(len(thetas), 1, 1, 1).to(thetas.device)
          y = y.view(1, 1, size, size).repeat(len(thetas), 1, 1, 1).to(thetas.device)
          x_thetas = x * torch.cos(thetas.view(-1, 1, 1, 1)) + y * torch.sin(thetas.view(-1, 1, 1, 1))
          y_thetas = -x * torch.sin(thetas.view(-1, 1, 1, 1)) + y * torch.cos(thetas.view(-1, 1, 1, 1))
          gabor = torch.exp(-0.5 * (x_thetas ** 2 + gamma ** 2 * y_thetas ** 2) / sigma ** 2) * torch.cos(2 * np.pi * x_thetas / lambd + psi)
          return gabor

     def run_inference(self, image):
          input_image = cv2.resize(image, dsize=(self.input_size, self.input_size))
          input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
          mean = [0.485, 0.456, 0.406]
          std = [0.229, 0.224, 0.225]
          x = (input_image / 255 - mean) / std
          x = x.transpose(2, 0, 1).astype('float32')
          x = x.reshape(-1, 3, self.input_size, self.input_size)
          input_name = self.onnx_session.get_inputs()[0].name
          output_name = self.onnx_session.get_outputs()[0].name
          onnx_result = self.onnx_session.run([output_name], {input_name: x})
          onnx_result = np.array(onnx_result).squeeze()
          return onnx_result, input_image

     def calculate_average_direction(self, direction_tensor, kernel_size=3):
          direction_tensor = direction_tensor.float()

          # Expand dimensions to fit the requirement of conv2d
          direction_tensor = direction_tensor[None, None, :, :]
          # Create a kernel for convolution. Here we use a uniform kernel, you can use different weights.
          kernel = torch.ones((1, 1, kernel_size, kernel_size), device=direction_tensor.device) / (kernel_size * kernel_size)

          # Pad the tensor in order to keep the size
          pad = kernel_size // 2
          direction_tensor = F.pad(direction_tensor, (pad, pad, pad, pad), mode='reflect')

          # Use 2D convolution to calculate the average direction in the local neighborhood
          average_direction = F.conv2d(direction_tensor, kernel)

          # Remove the added dimensions
          average_direction = average_direction.squeeze(0).squeeze(0)

          return average_direction

     def seg_hair(self, image, color_mask=[]):
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          image = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device)
          num_angles = self.num_angles
          orientations = torch.linspace(0, 2*np.pi, num_angles).to(device)
          gabors = self.create_gabor_batch(self.size, 1, orientations, 1, 1, self.psi).to(device)  # psi is now pi/2 (90 degrees)
          filtered_images = F.conv2d(image, gabors, padding=self.padding).squeeze(0)
          orientation_map = torch.argmax(filtered_images, axis=0)
          threshold = 1
          max_val, _ = torch.max(filtered_images, dim=0)
          min_val, _ = torch.min(filtered_images, dim=0)
          diff = max_val - min_val
          #orientation_map= self.calculate_average_direction(orientation_map)
          orientation_map = orientation_map.cpu().numpy()
          orientation_map[diff.cpu().numpy() < threshold] = 0

          if color_mask.any():
               color_mask = color_mask / 255
               orientation_map = orientation_map * color_mask
          if np.max(orientation_map) != 0:
               float_map = img_as_ubyte(orientation_map / np.max(orientation_map))
          else:
               float_map = img_as_ubyte(orientation_map)
          color_map = cv2.applyColorMap(float_map, cv2.COLORMAP_JET)

          return color_map, orientation_map

     
     def equalize_image(self, image):
          img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
          clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
          img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
          image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
          return image

     def equalize_histogram_color_masked(self, frame, mask_partial):
          # Convert the image to YCrCb
          ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

          # Apply the mask to the Y channel
          y_channel = cv2.bitwise_and(ycrcb[:,:,0], ycrcb[:,:,0], mask=mask_partial)

          # Apply histogram equalization to the masked Y channel
          y_channel_eq = cv2.equalizeHist(y_channel)

          # Apply the inverse mask to the equalized Y channel
          y_channel_eq = cv2.bitwise_and(y_channel_eq, y_channel_eq, mask=cv2.bitwise_not(mask_partial))

          # Add the equalized Y channel back to the original Y channel
          ycrcb[:,:,0] = cv2.add(ycrcb[:,:,0], y_channel_eq)

          # Convert the image back to BGR
          equalized_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

          return equalized_frame

     def extract_hair_color(self, image, mask):
          hair_pixels = image[mask==255].astype('float32')
          num_clusters = 2
          kmeans = KMeans(n_clusters=num_clusters)
          labels = kmeans.fit_predict(hair_pixels)
          unique_labels, counts_labels = np.unique(labels, return_counts=True)
          # Sort the labels by their counts in descending order.
          sorted_labels = unique_labels[np.argsort(-counts_labels)]

          dark_hair_label = sorted_labels[0]
          if len(sorted_labels) > 1 :
               light_hair_label = sorted_labels[1]
          else:
               light_hair_label = sorted_labels[0]


          #minor_cluster_label = unique_labels[np.argmin(counts_labels)]

          dominant_pixels = hair_pixels[(labels == dark_hair_label) | (labels == light_hair_label)]

          avg_color = np.mean(dominant_pixels, axis=0)
          std_color = np.std(dominant_pixels, axis=0)
          return avg_color, std_color

    # def extract_hair_color(self, image, mask):
    #     # Convert image to HSV
    #     h_channel = image[:, :, 0]
    #     print(h_channel.shape)
    #     hair_pixels = h_channel[mask==255].astype('float32').reshape(-1, 1)
    #     num_clusters = 3
    #     kmeans = KMeans(n_clusters=num_clusters)
    #     labels = kmeans.fit_predict(hair_pixels)
    #     unique_labels, counts_labels = np.unique(labels, return_counts=True)
    #     sorted_labels = unique_labels[np.argsort(-counts_labels)]
    #     dark_hair_label = sorted_labels[0]
    #     light_hair_label = sorted_labels[1]
    #     dominant_pixels = hair_pixels[(labels.ravel() == dark_hair_label) | (labels.ravel() == light_hair_label)]
    #     avg_color = np.mean(dominant_pixels, axis=0)
    #     std_color = np.std(dominant_pixels, axis=0)
    #     return avg_color, std_color


     def create_color_mask(self, image, avg_color, std_color, thresh=1.5):
          lower_bound = avg_color - (std_color * thresh)
          upper_bound = avg_color + (std_color * thresh)
          mask = cv2.inRange(image, lower_bound, upper_bound)
          return mask

     def create_partial_mask(self, image):
          masks, resized_image = self.run_inference(image)
          masks = np.where(masks > self.score_thresh, 0, 1)
          new_masks = []
          for mask in masks:
               mask_partial = np.where(mask==0, 255, 0).astype('uint8')
               mask_partial = cv2.resize(mask_partial, dsize=image.shape[:2][::-1])
               contours, _ = cv2.findContours(mask_partial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               max_contour = max(contours, key=cv2.contourArea)
               # Create a new mask to draw the largest contour
               new_mask = np.zeros_like(mask_partial)
               cv2.drawContours(new_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
               new_masks.append(new_mask)
               # 0 : skin, 1 : cloth, 2: hair

          return new_masks

     def create_hair_mask(self, partial_mask, color_mask):
          hair_mask = cv2.bitwise_and(partial_mask, color_mask)
          # hair_mask = cv2.medianBlur(hair_mask, 5)
          # hair_mask = cv2.GaussianBlur(hair_mask, (5,5), 0)
          return hair_mask

     def separate_hair_clothes(self, mask, eps=15.0, min_samples=100):
          # Find the indices of the pixels that are part of the mask.
          indices = np.array(np.where(mask != 0)).T

          # Convert the indices to float for DBSCAN
          indices_float = indices.astype(np.float32)

          # Apply DBSCAN clustering.
          labels_ = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(indices_float)
          labels, counts = np.unique(labels_, return_counts=True)

          # Sort the labels by their counts in descending order.
          sorted_labels = labels[np.argsort(-counts)]

          # Create separate masks for the hair and clothes.
          hair_mask = np.zeros_like(mask)
          clothes_mask = np.zeros_like(mask)
          print("labels",sorted_labels)

          # Get the labels for the hair and clothes.
          clothes_label = sorted_labels[1]  if len(sorted_labels) > 1 else sorted_labels[0]
          hair_label = sorted_labels[0]

          # Assign the pixels of the hair and clothes masks.
          hair_mask[indices[labels_ == hair_label, 0], indices[labels_ == hair_label, 1]] = 255
          clothes_mask[indices[labels_ == clothes_label, 0], indices[labels_ == clothes_label, 1]] = 255

          return hair_mask, clothes_mask

     def apply_closing(self, mask, kernel_size=10):
          # Define the structuring element
          kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

          # Apply the closing operation
          closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

          return closed_mask
     def apply_dilate(self, mask, kernel_size=10):
         # Define the structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Apply the closing operation
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        return closed_mask

     def apply_erode(self, mask, kernel_size=10):
         # Define the structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Apply the closing operation
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

        return closed_mask

     def equalize_histogram_color_masked(self, frame, mask_partial):
          # Convert the image to YCrCb
          ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

          # Apply the mask to the Y channel
          y_channel = cv2.bitwise_and(ycrcb[:,:,0], ycrcb[:,:,0], mask=mask_partial)

          # Apply histogram equalization to the masked Y channel
          y_channel_eq = cv2.equalizeHist(y_channel)

          # Apply the inverse mask to the equalized Y channel
          y_channel_eq = cv2.bitwise_and(y_channel_eq, y_channel_eq, mask=cv2.bitwise_not(mask_partial))

          # Add the equalized Y channel back to the original Y channel
          ycrcb[:,:,0] = cv2.add(ycrcb[:,:,0], y_channel_eq)

          # Convert the image back to BGR
          equalized_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

          return equalized_frame

     def restore_image(self, image, M_inv):
          restored_image = cv2.warpAffine(image, M_inv, (image.shape[1], image.shape[0]), borderValue=(0, 0, 0))
          return restored_image

     def align_face(self, image):
          # Convert the image from BGR to RGB
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Process the image and find faces
          results = self.face_mesh.process(image_rgb)

          M_inv = np.array([])
          # If a face is found
          if results.multi_face_landmarks:
               for face_landmarks in results.multi_face_landmarks:
                    # Convert face landmarks from normalized to pixel coordinates
                    landmarks = [[int(p.x * image.shape[1]), int(p.y * image.shape[0])] for p in face_landmarks.landmark]

                    # Calculate the center of gravity for each eye
                    left_eye = np.mean(np.array([landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153], landmarks[154], landmarks[155]]), axis=0)
                    right_eye = np.mean(np.array([landmarks[362], landmarks[398], landmarks[384], landmarks[385], landmarks[386], landmarks[387], landmarks[263]]), axis=0)

                    # Calculate the angle between the eyes
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    angle = np.arctan2(dy, dx) * 180. / np.pi
                    mp.solutions.drawing_utils.draw_landmarks(
                         image=image_rgb,
                         landmark_list=face_landmarks,
                         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                         landmark_drawing_spec=None,
                         connection_drawing_spec=mp.solutions.drawing_styles
                         .get_default_face_mesh_contours_style())
                    # Create rotation matrix
                    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
                    M_inv = cv2.invertAffineTransform(M)
                    # Apply affine warp
                    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(0, 255, 0))
                    return aligned_face, image_rgb, M_inv

          return image, image_rgb, M_inv

     def align_face_scale(self, image, scale_factor=100):
          # Convert the image from BGR to RGB
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Process the image and find faces
          results = self.face_mesh.process(image_rgb)

          M_inv = np.array([])
          # If a face is found
          if results.multi_face_landmarks:
               for face_landmarks in results.multi_face_landmarks:
                    # Convert face landmarks from normalized to pixel coordinates
                    landmarks = [[int(p.x * image.shape[1]), int(p.y * image.shape[0])] for p in face_landmarks.landmark]

                    # Calculate the center of gravity for each eye
                    left_eye = np.mean(np.array([landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153], landmarks[154], landmarks[155]]), axis=0)
                    right_eye = np.mean(np.array([landmarks[362], landmarks[398], landmarks[384], landmarks[385], landmarks[386], landmarks[387], landmarks[263]]), axis=0)
                    nose = np.array(landmarks[4])  # Nose tip
                    mouth = np.array(landmarks[57])  # Lower lip

                    # Calculate the angle between the eyes
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    angle = np.arctan2(dy, dx) * 180. / np.pi

                    # Create rotation matrix
                    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
                    M_inv = cv2.invertAffineTransform(M)
                    # Apply affine warp
                    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(0, 255, 0))

                    # Calculate distances between the points (eyes, nose, mouth)
                    d_eyes_nose = np.linalg.norm(nose - ((left_eye + right_eye) / 2.0))
                    d_nose_mouth = np.linalg.norm(nose - mouth)

                    # Determine scaling factor
                    avg_dist = np.mean([d_eyes_nose, d_nose_mouth])
                    scaling_factor = scale_factor / avg_dist

                    # Resize the image
                    aligned_face = cv2.resize(aligned_face, None, fx=scaling_factor, fy=scaling_factor)

                    return aligned_face, M_inv

          return image, M_inv

     def find_hair_strands(self, image, gabor_filters, start_point, num_steps, step_size=10):
          # Initialize the path with the start point
          path = [start_point]
          # Convert image to grayscale if not already
          for i in range(num_steps):
               current_point = path[-1]

               # Calculate the response of the gabor filters at the current point

               responses = gabor_filters[current_point[0], current_point[1]]
               # Find the direction with the highest response
               if responses > int(self.num_angles/2):
                    direction = np.pi/self.num_angles * responses + self.psi + np.pi
               else:
                    direction = np.pi/self.num_angles * responses + self.psi

               print(np.rad2deg(direction))
               # Calculate the next point in the direction of the highest response
               next_point = (current_point[0] + int(step_size*np.cos(direction))), current_point[1] + int(step_size*(np.sin(direction)))
               # Add the next point to the path
               path.append(next_point)

          # Draw the path on the image
          image_path = image.copy()
          for point in path:
               cv2.circle(image_path, point, 5, (0, 0, 255), -1)

          return path, image_path

     def segment_hairs(self, orientation_map, direction_threshold=10):
          # Ensure the image is 2D by taking the average over the color channels
          if len(orientation_map.shape) > 2:
               orientation_map = np.mean(orientation_map, axis=-1)

               # Calculate gradients along each axis
          gradient_y, gradient_x = np.gradient(orientation_map)

          # Compute the magnitude of the gradient vector at each pixel
          magnitude = np.sqrt(gradient_y**2 + gradient_x**2)

          mask = magnitude < direction_threshold

          # Use gaussian_filter to smooth the mask
          mask = gaussian_filter(mask.astype(float), sigma=1)


          # Use watershed algorithm to segment the image into different regions based on the mask
          segmented_image = watershed(mask)


          # Label each region
          labels = label(segmented_image)

          labels = skimage.color.label2rgb(labels, image=orientation_map, bg_label=0)

          return labels

     def mp_hair_mask(self, image):
          self.app.update(image)
          if self.app.output_image is None:
               return None
          return self.app.output_image

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

     def calculate_local_angles(self, W, Gx_, Gy_):
          # reshape tensors to allow vectorized computation
          Gx_ = Gx_.unfold(0, W, W).unfold(1, W, W)
          Gy_ = Gy_.unfold(0, W, W).unfold(1, W, W)

          # Calculate the local nominator and denominator
          nominator = 2 * Gx_ * Gy_
          denominator = Gx_ ** 2 - Gy_ ** 2

          # Sum over the local windows
          nominator = nominator.sum(dim=(2,3))
          denominator = denominator.sum(dim=(2,3))

          # Calculate the angles
          angles = torch.where(
               (nominator != 0) | (denominator != 0),
               (torch.pi + torch.atan2(nominator, denominator)) / 2,
               torch.zeros_like(nominator)
          )

          return angles, nominator, denominator

     def calculate_angles_W1W2(self, im, W1=15, W2=31):
          # Make sure the image is on the GPU and is a float32
          im = im.to('cuda:0').float()

          # Calculate the gradients
          Gx_, Gy_ = torch.gradient(im, axis=(1, 0))

          n1, d1, angles_W1 = self.calculate_local_angles(W1, Gx_, Gy_)
          n2, d2, angles_W2 = self.calculate_local_angles(W2, Gx_, Gy_)

          # Interpolate angles_W2 to match the shape of angles_W1
          angles_W2_resized = F.interpolate(angles_W2[None, None, :, :], size=angles_W1.shape, mode='bilinear', align_corners=False)[0, 0]


          # Determine unstable angles where denominator and nominator are close to zero
          unstable_angles = (d1.abs() < 1) & (n1.abs() < 1)

          # Replace unstable angles with the resized W2 angles
          stable_angles = torch.where(unstable_angles, angles_W2_resized, angles_W1)
          return stable_angles

     def calculate_angles_denoise(self, im, W, kernel_size=5, sigma=1.0):
          # Make sure the image is on the GPU and is a float32
          im = im.to('cuda:0').float()

          # Apply Gaussian blur
          # Define a Gaussian kernel
          kernel_1D = torch.exp(-(torch.arange(-kernel_size // 2, kernel_size // 2, dtype=im.dtype, device=im.device) ** 2) / (2 * sigma ** 2))
          kernel_1D = kernel_1D / kernel_1D.sum()

          # Create 2D kernel
          kernel = kernel_1D[:, None] * kernel_1D[None, :]
          kernel = kernel[None, None, :, :]
          
          # Convolve the image with the kernel
          im = F.conv2d(im[None, None, :], kernel, padding=kernel_size // 2)[0, 0, :]

          # Calculate the gradients
          Gx_, Gy_ = torch.gradient(im, axis=(1, 0))

          # reshape tensors to allow vectorized computation
          Gx_ = Gx_.unfold(0, W, W).unfold(1, W, W)
          Gy_ = Gy_.unfold(0, W, W).unfold(1, W, W)

          # Calculate the local nominator and denominator
          nominator = 2 * Gx_ * Gy_
          denominator = Gx_ ** 2 - Gy_ ** 2

          # Sum over the local windows
          nominator = nominator.sum(dim=(2,3))
          denominator = denominator.sum(dim=(2,3))

          # Calculate the angles
          angles = torch.where(
               (nominator != 0) | (denominator != 0),
               (torch.pi + torch.atan2(nominator, denominator)) / 2,
               torch.zeros_like(nominator)
          )

          return angles

     def calculate_angles_avg(self, im, W):
          # Make sure the image is on the GPU and is a float32
          im = im.to('cuda:0').float()

          # Calculate the gradients
          Gx_, Gy_ = torch.gradient(im, axis=(1, 0))

          # reshape tensors to allow vectorized computation
          Gx_ = Gx_.unfold(0, W, W).unfold(1, W, W)
          Gy_ = Gy_.unfold(0, W, W).unfold(1, W, W)

          # Calculate the local nominator and denominator
          nominator = 2 * Gx_ * Gy_
          denominator = Gx_ ** 2 - Gy_ ** 2
          print(nominator)
          # Sum over the local windows
          nominator = nominator.sum(dim=(2,3))
          denominator = denominator.sum(dim=(2,3))

          # Calculate the angles
          angles = torch.where(
               (nominator != 0) | (denominator != 0),
               (torch.pi + torch.atan2(nominator, denominator)) / 2,
               torch.zeros_like(nominator)
          )

          # Determine unstable angles where denominator and nominator are close to zero
          unstable_angles = (denominator.abs() < 30000) & (nominator.abs() < 30000)

          # Create kernel for averaging
          avg_kernel = torch.ones((1, 1, 3, 3)).to(im.device) / 9

          # Average angles using the kernel
          avg_angles = F.conv2d(angles[None, :, :], avg_kernel, padding=1)[0]

          # Replace unstable angles with the average of the neighbors
          stable_angles = torch.where(unstable_angles, avg_angles, angles)

          return stable_angles

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
          return xy_list, result

     def refine_mask(self, color_mask, deep_mask, grid_size):
          # 마스크의 크기를 가져옵니다.
          height, width = color_mask.shape

          # 격자의 크기를 정의합니다.
          grid_height, grid_width = height // grid_size, width // grid_size

          # 결과를 저장할 빈 마스크를 만듭니다.
          refined_mask = np.zeros_like(color_mask)

          # 각 격자를 순회합니다.
          for i in range(grid_size):
               for j in range(grid_size):
                    # 격자의 좌표를 가져옵니다.
                    start_y, end_y = i*grid_height, (i+1)*grid_height
                    start_x, end_x = j*grid_width, (j+1)*grid_width

                    # 두 마스크가 겹치는 부분이 있는지 확인합니다.
                    overlap = np.logical_and(color_mask[start_y:end_y, start_x:end_x],
                                             deep_mask[start_y:end_y, start_x:end_x])

                    # 겹치는 부분이 있다면, 색 추출 마스크를 반환합니다.
                    if np.any(overlap):
                         refined_mask[start_y:end_y, start_x:end_x] = color_mask[start_y:end_y, start_x:end_x]
                    else:
                         refined_mask[start_y:end_y, start_x:end_x] = deep_mask[start_y:end_y, start_x:end_x]

          return refined_mask

     def process_image(self, frame):
          frame = cv2.flip(frame,1)
          #frame = self.equalize_image(frame)
          frame, M_inv = self.align_face(frame)

          #frame = coh.coherence_filter(frame, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=1)

          _, clothes_mask, partial_masks  = self.create_partial_mask(frame)

          frame = self.equalize_histogram_color_masked(frame, partial_masks)

          hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
          lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          color_mask = self.create_color_mask(lab, avg_color, std_color)

          #clothes_mask = self.apply_dilate(clothes_mask, 31)


          #mask_partial_, contour_image = self.correct_mask_with_contours(partial_masks, color_mask)
          hair_mask = color_mask - clothes_mask

          #color_mask = self.apply_closing(color_mask, 5)
          hair_mask, _ = self.separate_hair_clothes(hair_mask)

          #hair_mask = self.create_hair_mask(partial_masks, color_mask)
          # 각 마스크를 다른 색으로 표시
          # debug = np.zeros((*hair_mask.shape, 3), dtype=np.uint8)
          # debug[hair_mask == 255] = [255, 0, 0]  # Red for hair
          # debug[clothes_mask == 255] = [0, 255, 0]  # Green for clothes

          color_map, orientation_map = self.seg_hair(frame, hair_mask)
          if M_inv.any():
               color_map = self.restore_image(color_map, M_inv)
               orientation_map = self.restore_image(orientation_map, M_inv)
               frame = self.restore_image(frame, M_inv)

          # path, path_map = self.find_hair_strands(frame, orientation_map, (256,128), 10, step_size=10)
          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)

          angle_map = self.calculate_angles(torch.Tensor(gray_map), 15)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, 15)

          #labels = self.segment_hairs(gray_map, direction_threshold=10)
          #labels_rgb = skimage.color.label2rgb(labels)  # convert labels to an RGB image
          #color_map = coh.coherence_filter(color_map, sigma=3, str_sigma=15, blend=0.7, iter_n=3, gray_on=1)
          combined = np.hstack((frame, angle_visual, color_map, cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR),  cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)))
          return combined

     def process_image_gabor(self, frame):
          frame = cv2.flip(frame,1)
          #frame = self.equalize_image(frame)
          frame, M_inv = self.align_face(frame)

          #frame = coh.coherence_filter(frame, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=1)

          _, clothes_mask, partial_masks  = self.create_partial_mask(frame)

          frame = self.equalize_histogram_color_masked(frame, partial_masks)

          # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
          # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          # avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          # color_mask = self.create_color_mask(lab, avg_color, std_color)

          #clothes_mask = self.apply_dilate(clothes_mask, 31)


          #mask_partial_, contour_image = self.correct_mask_with_contours(partial_masks, color_mask)
          # hair_mask = color_mask - clothes_mask

          #color_mask = self.apply_closing(color_mask, 5)
          # hair_mask, _ = self.separate_hair_clothes(hair_mask)

          #hair_mask = self.create_hair_mask(partial_masks, color_mask)
          # 각 마스크를 다른 색으로 표시
          # debug = np.zeros((*hair_mask.shape, 3), dtype=np.uint8)
          # debug[hair_mask == 255] = [255, 0, 0]  # Red for hair
          # debug[clothes_mask == 255] = [0, 255, 0]  # Green for clothes
          hair_mask = partial_masks
          color_map, orientation_map = self.seg_hair(frame, hair_mask)
          if M_inv.any():
               color_map = self.restore_image(color_map, M_inv)
               orientation_map = self.restore_image(orientation_map, M_inv)
               frame = self.restore_image(frame, M_inv)

          # path, path_map = self.find_hair_strands(frame, orientation_map, (256,128), 10, step_size=10)
          # gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)

          # angle_map = self.calculate_angles(torch.Tensor(gray_map), 15)
          # xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, 15)

          #labels = self.segment_hairs(gray_map, direction_threshold=10)
          #labels_rgb = skimage.color.label2rgb(labels)  # convert labels to an RGB image
          #color_map = coh.coherence_filter(color_map, sigma=3, str_sigma=15, blend=0.7, iter_n=3, gray_on=1)
          combined = np.hstack((frame,color_map, cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR)))
          return combined


     def process_image_st(self, frame):
          frame = cv2.flip(frame,1)
          frame, M_inv = self.align_face(frame)
          frame = self.equalize_image(frame)


          _, clothes_mask, partial_masks  = self.create_partial_mask(frame)

          # frame = self.equalize_histogram_color_masked(frame, partial_masks)

          # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
          lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          color_mask = self.create_color_mask(lab, avg_color, std_color)

          #clothes_mask = self.apply_dilate(clothes_mask, 31)


          #mask_partial_, contour_image = self.correct_mask_with_contours(partial_masks, color_mask)
          #hair_mask = color_mask - clothes_mask

          #color_mask = self.apply_closing(color_mask, 5)
          #hair_mask, _ = self.separate_hair_clothes(hair_mask)

          hair_mask = self.create_hair_mask(partial_masks, color_mask)
          # 각 마스크를 다른 색으로 표시
          # debug = np.zeros((*hair_mask.shape, 3), dtype=np.uint8)
          # debug[hair_mask == 255] = [255, 0, 0]  # Red for hair
          # debug[clothes_mask == 255] = [0, 255, 0]  # Green for clothes
          #hair_mask = partial_masks
          color_map, orientation_map = self.seg_hair(frame, hair_mask)
          # if M_inv.any():
          #     color_map = self.restore_image(color_map, M_inv)
          #     orientation_map = self.restore_image(orientation_map, M_inv)
          #     frame = self.restore_image(frame, M_inv)
          #     hair_mask = self.restore_image(hair_mask, M_inv)

          # path, path_map = self.find_hair_strands(frame, orientation_map, (256,128), 10, step_size=10)
          #gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)

          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
          ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
          cof_map = coh.coherence_filter(otsu_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)

          W = 15

          angle_map = self.calculate_angles(torch.Tensor(cof_map), W)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          debug_map = cv2.addWeighted(frame,0.5, angle_visual,0.5, 2.2)

          _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          angle_color= img_as_ubyte(angle_color / np.max(angle_color))
          angle_color_map = cv2.applyColorMap(angle_color, cv2.COLORMAP_JET)

          # angle_map_w1w2 = self.calculate_angles_W1W2(torch.Tensor(gray_map))
          # _, angle_visual_w1w2= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map_w1w2, W)


          # angle_map_weight = self.angle_calculator.calculate_angles(torch.Tensor(gray_map), W)
          # _, angle_visual_weight= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map_weight, W)


          # angle_map_avg = self.calculate_angles_avg(torch.Tensor(gray_map), W)
          # _, angle_visual_avg= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map_avg, W)


          # _, angle_color_avg= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map_avg, W)
          # angle_color_avg = img_as_ubyte(angle_color_avg / np.max(angle_color_avg))
          # angle_color_avg_map = cv2.applyColorMap(angle_color_avg, cv2.COLORMAP_JET)

          # angle_de_map = self.calculate_angles_denoise(torch.Tensor(gray_map), W)
          # _, angle_visual_denoise = self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_de_map, W)

          #labels = self.segment_hairs(gray_map, direction_threshold=10)
          #labels_rgb = skimage.color.label2rgb(labels)  # convert labels to an RGB image
          #color_map = coh.coherence_filter(color_map, sigma=3, str_sigma=15, blend=0.7, iter_n=3, gray_on=1)
          combined = np.hstack((frame, debug_map, angle_visual, angle_color_map, color_map, cv2.cvtColor(cof_map, cv2.COLOR_GRAY2BGR)))
          return combined

     def process_image_color(self, frame):
          frame = cv2.flip(frame,1)
          frame, frame_mp, M_inv = self.align_face(frame)
          frame = self.equalize_image(frame)

          _, clothes_mask, partial_masks  = self.create_partial_mask(frame)

          lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          avg_color = np.asarray(avg_color)
          std_color = np.asarray(std_color)
          a = 0
          if len(self.avg_color) == 0:
               self.avg_color = avg_color
               self.std_color = std_color
          else:
               self.avg_color = a*self.avg_color + (1-a)*avg_color
               self.std_color = a*self.std_color + (1-a)*std_color

          color_mask = self.create_color_mask(lab, self.avg_color, self.std_color)

          # if len(self.before_hair_mask) == 0:

          # else:
          #      refine_mask = self.refine_mask(color_mask, self.before_hair_mask, 8)
          refine_mask = self.refine_mask(color_mask, partial_masks, 8)
          #hair_mask = self.create_hair_mask(partial_masks, color_mask)
          hair_mask = refine_mask

          self.before_hair_mask = hair_mask

          hair_mask = self.apply_dilate(hair_mask, 5)
          hair_mask = self.apply_erode(hair_mask, 5)
          hair_mask = self.apply_erode(hair_mask, 5)
          hair_mask = self.apply_dilate(hair_mask, 5)

          color_map, orientation_map = self.seg_hair(frame, hair_mask)

          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
          ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

          cof_map = coh.coherence_filter(otsu_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)
          #cof_map = otsu_map
          W = 15

          angle_map = self.calculate_angles(torch.Tensor(cof_map), W)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          debug_map = cv2.addWeighted(frame,0.5, angle_visual,0.5, 2.2)

          _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          angle_color= img_as_ubyte(angle_color / np.max(angle_color))
          angle_color_map = cv2.applyColorMap(angle_color, cv2.COLORMAP_JET)
          combined = np.hstack((frame, frame_mp, debug_map, angle_visual, angle_color_map, color_map, cv2.cvtColor(cof_map, cv2.COLOR_GRAY2BGR)))
          return combined

     def process_image_ros(self, frame):
          # frame = cv2.flip(frame,1)
          #frame, M_inv = self.align_face(frame)
          frame = self.equalize_image(frame)

          _, clothes_mask, partial_masks  = self.create_partial_mask(frame)
          
          lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          avg_color = np.asarray(avg_color)
          std_color = np.asarray(std_color)
          a = 0
          if len(self.avg_color) == 0:
               self.avg_color = avg_color
               self.std_color = std_color
          else:
               self.avg_color = a*self.avg_color + (1-a)*avg_color
               self.std_color = a*self.std_color + (1-a)*std_color

          color_mask = self.create_color_mask(lab, self.avg_color, self.std_color)
          refine_mask = self.refine_mask(color_mask, partial_masks, 8)

          #hair_mask = self.create_hair_mask(partial_masks, color_mask)
          #hair_mask = np.full_like(frame[:,:,0], 255)
          hair_mask = refine_mask

          erode_mask = self.apply_erode(hair_mask, 3)
          hair_mask = self.apply_dilate(erode_mask, 3)


          color_map, orientation_map = self.seg_hair(frame, hair_mask)

          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
          ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

          cof_map = coh.coherence_filter(otsu_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)

          W = 15

          angle_map = self.calculate_angles(torch.Tensor(cof_map), W)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          debug_map = cv2.addWeighted(frame,0.5, angle_visual,0.5, 2.2)

          return debug_map, xy_list

     def process_image_mp(self, frame):
          frame = cv2.flip(frame,1)
          # frame, frame_mp, M_inv = self.align_face(frame)
          frame = self.equalize_image(frame)

          partial_masks  = self.mp_hair_mask(frame)

          if partial_masks is None:
               return frame


          # hair_mask = self.apply_dilate(hair_mask, 5)
          # hair_mask = self.apply_erode(hair_mask, 5)
          # hair_mask = self.apply_erode(hair_mask, 5)
          # hair_mask = self.apply_dilate(hair_mask, 5)

          hair_mask = partial_masks
          color_map, orientation_map = self.seg_hair(frame, hair_mask)

          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
          ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

          cof_map = coh.coherence_filter(otsu_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)
          #cof_map = otsu_map
          W = 15

          angle_map = self.calculate_angles(torch.Tensor(cof_map), W)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          debug_map = cv2.addWeighted(frame,0.5, angle_visual,0.5, 2.2)

          _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          if np.max(angle_color) != 0:
               angle_color= img_as_ubyte(angle_color / np.max(angle_color))
          else:
               angle_color= img_as_ubyte(angle_color)
          angle_color_map = cv2.applyColorMap(angle_color, cv2.COLORMAP_JET)
          combined = np.hstack((frame, debug_map, angle_visual, angle_color_map, color_map, cv2.cvtColor(cof_map, cv2.COLOR_GRAY2BGR)))
          return combined

     def process_image_mp_color(self, frame):
          frame = cv2.flip(frame,1)
          # frame, frame_mp, M_inv = self.align_face(frame)
          frame = self.equalize_image(frame)

          partial_masks  = self.mp_hair_mask(frame)

          if partial_masks is None:
               return frame

          lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

          avg_color, std_color = self.extract_hair_color(lab, partial_masks)
          avg_color = np.array(avg_color)
          std_color = np.array(std_color)
          a = 0
          if len(self.avg_color) == 0:
               self.avg_color = avg_color
               self.std_color = std_color
          else:
               self.avg_color = a*self.avg_color + (1-a)*avg_color
               self.std_color = a*self.std_color + (1-a)*std_color

          color_mask = self.create_color_mask(lab, self.avg_color, self.std_color)

          # if len(self.before_hair_mask) == 0:
          #refine_mask = self.create_hair_mask(partial_masks, color_mask)

          # else:
          #      refine_mask = self.refine_mask(color_mask, self.before_hair_mask, 8)
          refine_mask = self.refine_mask(color_mask, partial_masks, 8)
          #hair_mask = self.create_hair_mask(partial_masks, color_mask)
          hair_mask = refine_mask

          self.before_hair_mask = hair_mask

          # hair_mask = self.apply_dilate(hair_mask, 5)
          # hair_mask = self.apply_erode(hair_mask, 5)
          # hair_mask = self.apply_erode(hair_mask, 5)
          # hair_mask = self.apply_dilate(hair_mask, 5)

          color_map, orientation_map = self.seg_hair(frame, hair_mask)

          gray_map=cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
          ret, otsu_map = cv2.threshold(gray_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

          cof_map = coh.coherence_filter(otsu_map, sigma=3, str_sigma=15, blend=0.5, iter_n=3, gray_on=0)
          #cof_map = otsu_map
          W = 15

          angle_map = self.calculate_angles(torch.Tensor(cof_map), W)
          xy_list, angle_visual= self.visualize_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          debug_map = cv2.addWeighted(frame,0.5, angle_visual,0.5, 2.2)

          _, angle_color= self.visualize_color_angles(gray_map, torch.Tensor(hair_mask), angle_map, W)
          if np.max(angle_color) != 0:
               angle_color= img_as_ubyte(angle_color / np.max(angle_color))
          else:
               angle_color= img_as_ubyte(angle_color)
          angle_color_map = cv2.applyColorMap(angle_color, cv2.COLORMAP_JET)
          combined = np.hstack((frame, debug_map, angle_visual, angle_color_map, color_map, cv2.cvtColor(cof_map, cv2.COLOR_GRAY2BGR), cv2.cvtColor(partial_masks, cv2.COLOR_GRAY2BGR)))
          return combined
