import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from UNet import Model

def img2strand(checkpoint_img2strand, rgb_img, mask):
    # print("convert image to strand map")

    model = Model().cuda()
    model.load_state_dict(torch.load(checkpoint_img2strand))

    model.eval()

    rgb_img = rgb_img[:,:, 0:3] / 255.

    mask = (mask/255.>0.5)[:,:,None]

    rgb_img = rgb_img*mask

    rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()
    strand_pred = model(rgb_img)
    strand_pred = strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    strand_pred = strand_pred * 2 - 1

    g = strand_pred[:,:,0]
    b = -strand_pred[:,:,1]
    theta = np.arctan2(g, b) # + np.pi/2

    # strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)  # 512 * 512 *60
    strand_pred = np.concatenate([mask, strand_pred * mask], axis=-1)


    # print("angle:", theta*180/np.pi)
    # print("max angle:", np.max(theta*180/np.pi))
    # print("min angle:", np.min(theta*180/np.pi))

    angle_map = theta

    return (strand_pred*255).astype(np.uint8), angle_map

def compute_3d_orientation_map(normal_map, orientation_map_2d, hair_mask):
    """
    Compute the 3D orientation map.

    Parameters:
    normal_map (numpy.ndarray): The normal map (3 channels, float32)
    orientation_map_2d (numpy.ndarray): The 2D orientation map (single channel, float32)

    Returns:
    numpy.ndarray: The 3D orientation map (3 channels, float32)
    """
    # height, width = orientation_map_2d.shape
    # orientation_map_3d = np.zeros((height, width, 3), dtype=np.float32)

    angles = torch.from_numpy(orientation_map_2d).to("cuda").float()
    normal_map = torch.from_numpy(normal_map).to("cuda").float()
    vec_2d_x = torch.cos(angles)
    vec_2d_y = torch.sin(angles)
    zeros = torch.zeros_like(vec_2d_x)
    vec_2d = torch.stack((vec_2d_x, vec_2d_y, zeros), dim=-1)

    # Calculate perpendicular vectors
    perp_vec = torch.cross(normal_map, vec_2d)
    perp_vec /= torch.norm(perp_vec, dim=-1, keepdim=True)

    # Rotate the 2D vectors to 3D space
    vec_3d = torch.cross(normal_map, perp_vec)
    vec_3d = vec_3d * torch.from_numpy(hair_mask[:,:,np.newaxis]).to("cuda")/255

    # for y in range(height):
    #     for x in range(width):
    #         if hair_mask[int(y), int(x)] > 0:
    #             # Extract the normal vector at this pixel
    #             normal = normal_map[y, x]

    #             # Calculate the angle in radians
    #             angle = orientation_map_2d[y, x]

    #             # Generate a 2D vector in the plane
    #             vec_2d = np.array([np.cos(angle), np.sin(angle), 0])

    #             # Find a vector that is perpendicular to the normal
    #             perp_vec = np.cross(normal, vec_2d)
    #             perp_vec /= np.linalg.norm(perp_vec)

    #             # Rotate the 2D vector into 3D space
    #             vec_3d = np.cross(normal, perp_vec)

    #             # Store the result
    #             orientation_map_3d[y, x] = vec_3d

    # return orientation_map_3d
    return vec_3d

def visualize_orientation_map(orientation_map_3d):
    """
    Visualize the 3D orientation map by mapping vectors to RGB colors.

    Parameters:
    orientation_map_3d (numpy.ndarray): The 3D orientation map (3 channels, float32)

    Returns:
    numpy.ndarray: The visualized orientation map (3 channels, uint8)
    """
    # Map from [-1, 1] to [0, 255]
    vis_map = ((orientation_map_3d + 1) / 2 * 255).astype(np.uint8)
    return vis_map

if __name__ == "__main__":
    img2strand()
