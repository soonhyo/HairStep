import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.hourglass import Model

import matplotlib.pyplot as plt
import cv2

def img2depth(opt, rgb_img, mask):
    model = Model().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.checkpoint_img2depth))

    model.eval()

    rgb_img = rgb_img[:,:, 0:3] / 255.
    mask = (mask/255.>0.5)
    # rgb_img = rgb_img*mask
    rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()

    depth_pred = model(rgb_img)
    depth_pred = depth_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    depth_pred_masked = depth_pred[:, :, 0] * mask - (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred)))

    max_val = np.nanmax(depth_pred_masked)
    min_val = np.nanmin(depth_pred_masked + 2 * (1 - mask) * (np.abs(np.nanmax(depth_pred_masked)) + np.abs(np.nanmin(depth_pred))))
    depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val)*mask
    depth_pred_norm = np.clip(depth_pred_norm, 0., 1.)

    # masked_img = depth_pred * mask + (1 - mask) * ((depth_pred * mask - (1 - mask) * 100000).max()) 
    # set the value of un-mask to the min-val in mask
    # norm_masked_depth = masked_img / (np.nanmax(masked_img) - np.nanmin(masked_img))  # norm

    depth_color_map = cv2.applyColorMap(np.uint8(depth_pred_norm*255), cv2.COLORMAP_JET)

    # np.save(os.path.join(output_depth_path, item[:-3]+'npy'), depth_pred_norm)

    # depth2vis(mask, depth_pred_norm, os.path.join(output_depth_vis_path, item))
    print(depth_color_map.shape)
    return depth_color_map

if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2depth(opt)
