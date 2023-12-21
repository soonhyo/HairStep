import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.UNet import Model

def img2strand(opt, rgb_img, mask):
    print("convert image to strand map")

    model = Model().cuda()
    model.load_state_dict(torch.load(opt.checkpoint_img2strand))

    model.eval()

    rgb_img = rgb_img[:,:, 0:3] / 255.

    mask = (mask/255.>0.5)[:,:,None]

    rgb_img = rgb_img*mask

    rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()

    strand_pred = model(rgb_img)
    strand_pred = strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    strand_pred = strand_pred * 2 - 1
    # strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)  # 512 * 512 *60

    strand_pred = np.concatenate([mask, strand_pred * mask], axis=-1)
    # print("strand", strand_pred.shape)
    # print("strand_concate", strand_pred.shape)
    # print("mask", mask*255)
    g = strand_pred[:,:,1]
    b = strand_pred[:,:,2]
    # g = strand_pred[:,:,1] *2 -1
    # b = strand_pred[:,:,2] *2 -1

    print("g:", g)
    print("b:", b)
    theta = np.arctan2(b, g)

    print("angle:", theta*180/np.pi)

    return (strand_pred*255).astype(np.uint8)

if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2strand(opt)
