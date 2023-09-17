import os
import torch
import os
import json
import imageio
import numpy as np
import mmcv
import cv2
import time


def load_data(data_folder, case_name):
    """TODO: load check point if exists"""
    try:
        checkpoint = torch.load(os.path.join(data_folder, "data.pt"))
        rays_o_all = checkpoint['rays_o']
        rays_d_all = checkpoint['rays_d']
        viewdirs_all = checkpoint['viewdirs']
        rgb_all = checkpoint['rgb_all']
        if 'ray_mask_all' in checkpoint:
            ray_mask_all = checkpoint['ray_mask_all']
        else:
            ray_mask_all = torch.ones([rgb_all.shape[0], rgb_all.shape[2]])

    except:
        with open("data" + case_name + "/all_data.json") as f:
            data_info = json.load(f)
        

    return rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all


def static_train():
    """this method includes two static train steps
    :the first one trains a static sense using sdf-nerf functions (--now using neus) without the object
    the second one trains a static original mesh without sense, similarly using sdf-nerf functions (-- now using neus)"""


if __name__ == '__main__':
    # TODO: raed parser
    print('config_parser -- reading the setting of \"THE WORLD\"')
    load_data("data/example", "example")
    # read config in neus, referring from exp_train.py

    # TODO:train static
    print('background SDF and moving object')

    # TODO: extract moving object's mesh
    # wirte a .obj file
    # give the name of the .obj file
    print('extract moving object\'s mesh')

    # TODO:initial velocity optimization
    # methods? veloicty of each finite element
    # read .obj file according to the name
    print('initial velocity optimization')

    # TODO:train dynamic
    print('physical parameters of moving object')
