import cv2
import numpy as np

def compute_normal_map(depth_image, ksize=3):
    rows, cols = depth_image.shape

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=ksize)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal_ = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal_ + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    return normal_bgr, normal_
