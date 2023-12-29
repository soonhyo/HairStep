import torch
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def coherence_filter(img, sigma=3, str_sigma=15, blend=0.9, iter_n=4, gray_on=1):
    h, w = img.shape[:2]

    img = torch.from_numpy(img).to(device).float()
    img.requires_grad = False

    for i in range(iter_n):
        gray = img
        if gray_on == 1:
            gray = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2GRAY)
            gray = torch.from_numpy(gray).to(device)

        # Compute eigenvalues and eigenvectors
        eigen = cv2.cornerEigenValsAndVecs(gray.cpu().numpy(), str_sigma, 3)
        eigen = torch.from_numpy(eigen).to(device)
        eigen = eigen.view(h, w, 3, 2)

        x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1]

        # Compute the directional derivatives
        gxx = cv2.Sobel(gray.cpu().numpy(), cv2.CV_32F, 2, 0, ksize=sigma)
        gxx = torch.from_numpy(gxx).to(device)
        gxy = cv2.Sobel(gray.cpu().numpy(), cv2.CV_32F, 1, 1, ksize=sigma)
        gxy = torch.from_numpy(gxy).to(device)
        gyy = cv2.Sobel(gray.cpu().numpy(), cv2.CV_32F, 0, 2, ksize=sigma)
        gyy = torch.from_numpy(gyy).to(device)

        gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
        m = gvv < 0

        # Perform erosion and dilation
        ero = cv2.erode(img.cpu().numpy(), None)
        ero = torch.from_numpy(ero).to(device)
        dil = cv2.dilate(img.cpu().numpy(), None)
        dil = torch.from_numpy(dil).to(device)

        img1 = ero
        img1[m] = dil[m]
        img = img * (1.0 - blend) + img1 * blend
        img = img.to(torch.uint8)

    return img.cpu().numpy()
