
import os
import time
import json
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from tqdm import trange
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.rigid_body import rigid_body_simulator
from models.common import *
from argparse import ArgumentParser
from exp_runner import Runner

def load_data(camera_params_path, images_path, masks_path, frames_count): # assmue load from a json file
    print("---------------------Loading image data-------------------------------------")
    with open(camera_params_path, "r") as json_file:
        camera_params_list = json.load(json_file)   
    images, masks, cameras_K, cameras_M = [], [], [], []  # cameras_M should be c2w mat
    for i in range(0, frames_count):
        picture_name = str(i) + ".png"
        image_I_path = images_path + "/transform" + f"{i:04}" + picture_name 
        image = cv.imread(image_I_path)
        images.append(image) + ".png"
        mask_I_path = masks_path + "/transform" + f"{i:04}_mask" + picture_name 
        mask = cv.imread(mask_I_path)
        masks.append(mask)
        cameras_name = str(i)
        camera_K = camera_params_list[cameras_name + "_K"]
        cameras_K.append(camera_K)
        camera_M = camera_params_list[cameras_name + "_M"]
        cameras_M.append(camera_M)
    print("---------------------Loading image data finished------------------------------")

    return images, masks, cameras_K, cameras_M  # returns numpy arrays

def generate_rays_at(transform_matrix, intrinsic_mat, W, H, resolution_level):  # transform mat should be c2w mat
    transform_matrix = torch.from_numpy(transform_matrix)
    transform_matrix.cuda()   # add to cuda
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    intrinsic_mat_inv.from_numpy(transform_matrix)
    intrinsic_mat_inv.cuda()

    l = resolution_level
    tx = torch.linspace(0, W - 1, W // l)
    ty = torch.linspace(0, H - 1, H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    p = torch.matmul(intrinsic_mat_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3, start from transform
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

def generate_all_rays(imgs, masks, cameras_K, cameras_c2w):
    # this function generate rays from given img and camera_K & c2w, also returns rays_gt as reference
    # assume input raw images are 255-uint, this function transformed to 1.0-up float0
    # stack the result into [frames_count, W*H, 3] format, assume all frames has the same resolution
    
    shape = imgs[0].shape()
    W, H = shape[1], shape[0]
    frames_count = len(imgs)
    rays_o_all, rays_v_all, rays_gt_all, rays_mask_all = [], [], [], []
    for i in range(0, frames_count):
        rays_gt, rays_mask = imgs[i], masks[i] ## check if is  H, W, 3

        rays_gt = rays_gt / 256.0
        rays_gt = rays_gt.reshape(-1, 3)
        rays_mask = rays_mask / 255.0 
        rays_mask = np.where(rays_mask > 0, 1, 0)
        rays_mask = rays_mask.reshape(-1, 3)
        rays_o, rays_v = generate_rays_at(cameras_c2w[i], cameras_K[i], W, H, resolution_level=1) ## check if is  H, W, 3
        rays_o = rays_o.reshape(-1, 3)
        rays_v = rays_v.reshape(-1, 3)
        rays_o_all.append(rays_o)
        rays_v_all.append(rays_v)
        rays_gt_all.append(rays_gt)
        rays_mask_all.append(rays_mask)


    # returns rays_o_all, rays_v_all, rays_gt_all, rays_mask_all formulate by frames
    return rays_o_all, rays_v_all, rays_gt_all, rays_mask_all

class GenshinStart(torch.nn.Module):
    def __init__(self, setting_json_path):
        super(GenshinStart, self).__init__()
        self.flag = 0
        self.device = 'cuda:0'
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        static_mesh = motion_data["static_mesh_path"]
        option = {'frames': 2,
                  'ke': 0.1,
                  'mu': 0.8,
                  'transform': [0.0, 0.0, 0.9985088109970093, 0.0, 0.0, 0.0],
                  'linear_damping': 0.999,
                  'angular_damping': 0.998}
        self.physical_simulator = rigid_body_simulator(static_mesh, option)
        self.max_frames = 1
        self.translation, self.quaternion = [], []
        self.static_object_conf_path =    motion_data["neus_object_conf_path"]
        self.static_object_name =     motion_data['neus_static_object_name']
        self.static_object_continue =     motion_data['neus_static_object_continue']

        self.static_background_conf_path = motion_data["neus_background_conf_path"]       
        self.static_background_name = motion_data['neus_static_background_name']
        self.static_background_continue = motion_data['neus_static_background_continue']
        # in this step, use 'train' mode as default
        self.runner_object = \
            Runner.get_runner(self.static_object_conf_path, self.static_object_name, self.static_object_continue) 
        # self.runner_background = \
        #     Runner.get_runner(self.static_background_conf_path, self.static_background_name, self.static_background_continue)
        
        with torch.no_grad():
            self.init_mu = torch.zeros([1], requires_grad=True, device=self.device)
            self.init_ke = torch.zeros([1], requires_grad=True, device=self.device)
            self.init_translation = torch.zeros([3], requires_grad=True, device=self.device)
            self.init_quaternion = torch.zeros([4], requires_grad=True, device=self.device)
            self.init_v = torch.zeros([3], requires_grad=True, device=self.device)
            self.init_omega = torch.zeros([3], requires_grad=True, device=self.device)

        # TODO: need to be completed， should be torch tensor here
        self.batch_size = motion_data["batch_size"]
        self.frame_counts = motion_data["frame_counts"]
        self.images_path = motion_data["images_path"]
        self.masks_path = motion_data["masks_path"]
        self.camera_setting_path = motion_data["cameras_setting_path"]
        
        images, masks, cameras_K, cameras_M = load_data(self.images_path, self.masks_path, self.camera_setting_path, self.frame_counts)
        rays_o_all, rays_v_all, rays_gt_all, rays_mask_all = generate_all_rays(images, masks, cameras_K, cameras_M)
        import pdb
        pdb.set_trace()

        self.rays_o_all = torch.from_numpy(rays_o_all).to(self.device)
        self.rays_v_all = torch.from_numpy(rays_v_all).to(self.device)
        self.rays_gt_all = torch.from_numpy(rays_gt_all).to(self.device)
        self.rays_mask_all = torch.from_numpy(rays_mask_all).to(self.device)

    
    def forward(self, max_f:int):       
        pbar = trange(max_f) 
        pbar.set_description('\033[5;41mForward\033[0m')
        global_loss = 0
        self.physical_simulator.clear()
        self.physical_simulator.clear_gradients()
        for i in pbar:                      
            translation, quaternion = self.physical_simulator.forward(i)
            self.translation.append(translation)
            self.quaternion.append(quaternion)
            global_loss = torch.tensor(torch.nan, device=self.device)
            rays_o, rays_d, rays_gt, rays_mask = self.rays_o_all[i], self.rays_v_all[i], self.rays_gt_all[i], self.rays_mask_all[i]

            rays_o, rays_d, rays_gt = rays_o[rays_mask], rays_d[rays_mask], rays_gt[rays_mask]
            rays_o, rays_d, rays_gt = rays_o.split(self.batch_size), rays_d.split(self.batch_size), rays_gt.split(self.batch_size)
            for rays_o_batch, rays_d_batch, rays_gt_batch in zip(rays_o, rays_d, rays_gt):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
                # this render out contains grad & img loss, find out its reaction with phy simualtion
                render_out = self.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch, near=near, far=far,
                                                        T=translation, R=quaternion
                                                        , cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                        background_rgb=background_rgb)
                color_fine = render_out["color_fine"]
                color_error = (color_fine - rays_gt_batch)
                mask_sum = self.batch_size
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),
                                            reduction='sum') / mask_sum  # normalize
                global_loss += color_fine_loss.clone().detach()
                color_fine_loss.backward()  # img_loss for refine R & T
                torch.cuda.synchronize()
                del render_out
            pbar.set_description(f"[Forward] loss: {global_loss.item()}")
    
        return global_loss

    def backward(self, max_f:np.int32):
        pbar = trange(max_f)
        pbar.set_description('\033[5;30m[Backward]\033[0m')
        for i in pbar:
            f = max_f - 1 - i
            with torch.no_grad():
                translation_grad = self.translation[f].grad
                quaternion_grad = self.quaternion[f].grad
            if f > 0:
                self.physical_simulator.set_motion_grad(f, translation_grad, quaternion_grad)
                self.physical_simulator.backward(f)
            else:
                v_grad, omega_grad, ke_grad, mu_grad, translation_grad, quaternion_grad = \
                    self.physical_simulator.backwards(f)
                self.init_v.backward(retain_graph=True, gradient=v_grad)
                self.init_omega.backward(retain_graph=True, gradient=omega_grad)
                self.init_ke.backward(retain_graph=True, gradient=ke_grad)
                self.init_mu.backward(retain_graph=True, gradient=mu_grad)
                self.init_translation.backward(retain_graph=True, gradient=translation_grad)
                self.init_quaternion.backward(retain_graph=True, gradient=quaternion_grad)

def get_optimizer(mode, genshinStart):
    
    optimizer = None
    if mode == "train_static":
        optimizer = torch.optim.Adam(
            [
                {"params": getattr(), 'lr': 1e-1}
            ]
        )
    elif mode == "train_velocity":
                optimizer = torch.optim.Adam(
            [
                {"params": getattr(), 'lr': 1e-1}
            ]
        )
    elif mode == "train_dynamic":
        optimizer = torch.optim.Adam(
            [
                {"params": getattr(genshinStart,'init_translation'), 'lr': 1e-1},
                {'params': getattr(genshinStart,'init_quaternion'), 'lr':1e-1}
            ]
            ,
            amsgrad=False
        )

    return optimizer

def train_static(self):
    static = 0 
    # train static object -> export as a obj mesh

    # train static background -> export as a obj mesh

    # also need to train R0 & T0?
    
def train_velocity(self):
    velocity = 1

def train_dynamic(cfg, genshinStart, optimizer, device):
    def train_forward(optimizer):
        optimizer.zero_grad()
        loss = torch.tensor(np.nan, device=device)
        while loss.isnan():
            loss = genshinStart.forward(cfg['max_frame'])

        return loss
    
    optimizer = get_optimizer('train_dynamic',genshinStart)
    for i in range(cfg['iter']):
        loss = train_forward(optimizer=optimizer)
        genshinStart.backward(cfg['max_frame'])
        optimizer.step()


if __name__ == '__main__':
    print_blink('Genshin Nerf, start!!!')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')


    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./dynamic_test/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    genshinStart = GenshinStart(args.conf)
    if args.mode == "train":
        train_static()
        train_velocity()
        train_dynamic()
    else:
        train_dynamic()

    
# python genshin_start.py --mode debug --conf ./dynamic_test/genshin_start.json --case bird --is_continue 
