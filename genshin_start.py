import json
import logging
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import trange
from models.common import *
from argparse import ArgumentParser
from exp_runner import Runner
import time
import math
import trimesh
from pathlib import Path
import os

def load_cameras_and_images(images_path, masks_path, camera_params_path, frames_count, with_fixed_camera=False,
                            camera_params_list=None, pic_mode="png"):  # assmue load from a json file
    print("---------------------Loading image data-------------------------------------")
    global_K, global_M = None, None
    if with_fixed_camera and camera_params_list is not None:
        # in this case, we assume all frames share with the same K & M
        global_K = camera_params_list['K']
        global_M = camera_params_list['M']
    else:   # not pre-defined list
        with open(camera_params_path, "r") as json_file:
            camera_params_list = json.load(json_file)

    images, masks, cameras_K, cameras_M = [], [], [], []  # cameras_M should be c2w mat
    for i in range(1, frames_count + 1):
        picture_name = f"{i:03}"
        image_I_path = images_path + "/" + picture_name + "." + pic_mode
        image = cv.imread(image_I_path)
        images.append(np.array(image))
        mask_I_path = masks_path + "/" + picture_name + "." + pic_mode
        mask = cv.imread(mask_I_path)
        masks.append(np.array(mask))
        if with_fixed_camera:
            cameras_K.append(np.array(global_K))
            cameras_M.append(np.array(global_M))
        else:
            cameras_name = str(i)
            camera_K = camera_params_list[cameras_name + "_K"]
            cameras_K.append(np.array(camera_K))
            camera_M = camera_params_list[cameras_name + "_M"]
            cameras_M.append(np.array(camera_M))
    print("---------------------Load image data finished-------------------------------")
    return images, masks, cameras_K, cameras_M  # returns numpy arrays

def generate_rays_with_K_and_M(transform_matrix, intrinsic_mat, W, H, resolution_level=1):  # transform mat should be c2w mat, and numpy as input
    transform_matrix = torch.from_numpy(transform_matrix.astype(np.float32)).to('cuda')  # add to cuda
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    intrinsic_mat_inv = torch.from_numpy(intrinsic_mat_inv.astype(np.float32)).to('cuda')
    tx = torch.linspace(0, W - 1, W // resolution_level)
    ty = torch.linspace(0, H - 1, H // resolution_level)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    p = torch.matmul(intrinsic_mat_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3, start from transform
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

def generate_all_rays(imgs, masks, cameras_K, cameras_c2w, W_all, H_all):
    # this function generate rays from given img and camera_K & c2w, also returns rays_gt as reference
    # assume input raw images are 255-uint, this function transformed to 1.0-up float0
    # stack the result into [frames_count, W*H, 3] format, assume all frames has the same resolution with W, H
    frames_count = len(imgs)
    rays_o_all, rays_v_all, rays_gt_all, rays_mask_all = [], [], [], []
    for i in range(0, frames_count):
        rays_gt, rays_mask = imgs[i], masks[i]  ## check if is  H, W, 3
        rays_gt = rays_gt / 256.0
        rays_gt = rays_gt.reshape(-1, 3)
        rays_gt = torch.from_numpy(rays_gt.astype(np.float32)).to("cuda")
        rays_mask = rays_mask / 255.0
        rays_mask = np.where(rays_mask > 0, 1, 0).reshape(-1, 3)
        rays_mask = torch.from_numpy(rays_mask.astype(np.bool_)).to("cuda")
        rays_o, rays_v = generate_rays_with_K_and_M(cameras_c2w[i], cameras_K[i], W_all, H_all)  ## check if is  H, W, 3
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
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        self.device = motion_data["device"]
        option = {'mesh':motion_data["static_mesh_path"],
                  'frames': motion_data["frame_counts"],
                  'frame_dt': motion_data["frame_dt"],
                  'delta_frame': motion_data['delta_frame'],
                  'substep':motion_data["substep"],
                  'kn': 0.6,
                  'mu': 0.2,
                  'translation': motion_data['T0'],
                  'quaterion': motion_data['R0'],
                  'linear_damping': 0.999,
                  'angular_damping': 0.998}

        self.physical_init(options=option)
        if 'planar_contact' in motion_data:
            self.add_planar_contact(slope_degree=motion_data['planar_contact'][0], init_height=motion_data['planar_contact'][1])
        self.static_object_conf_path = motion_data["neus_object_conf_path"]
        self.static_object_name = motion_data['neus_static_object_name']
        self.static_object_continue = motion_data['neus_static_object_continue']

        self.static_background_conf_path = motion_data["neus_background_conf_path"]
        self.static_background_name = motion_data['neus_static_background_name']
        self.static_background_continue = motion_data['neus_static_background_continue']
        self.runner_object = \
            Runner.get_runner(self.static_object_conf_path, self.static_object_name, self.static_object_continue)
        self.runner_background = \
            Runner.get_runner(self.static_background_conf_path, self.static_background_name, self.static_background_continue)
        self.batch_size = motion_data["batch_size"]
        self.frame_counts = motion_data["frame_counts"]
        self.images_path = motion_data["images_path"]
        self.masks_path = motion_data["masks_path"]
        self.camera_setting_path = None
        self.with_fixed_camera = motion_data["with_fixed_camera"]
        camera_params_list = None
        if self.with_fixed_camera:
            camera_params_list = motion_data['fixed_camera_setting']
        else:  # need to specify the camera path of the motion
            self.camera_setting_path = motion_data["cameras_setting_path"]
        images, masks, cameras_K, cameras_M = (
            load_cameras_and_images(self.images_path, self.masks_path, self.camera_setting_path, self.frame_counts
                                    , with_fixed_camera=self.with_fixed_camera, camera_params_list=camera_params_list))
        self.cameras_K, self.cameras_M = cameras_K, cameras_M
        self.W, self.H = images[0].shape[1], images[0].shape[0]
        with torch.no_grad():
            self.rays_o_all, self.rays_v_all, self.rays_gt_all, self.rays_mask_all = generate_all_rays(images, masks,
             cameras_K, cameras_M,self.W, self.H)
    
    def physical_init(self, options):
        self.substep = options['substep']
        self.frames = options['frames']
        self.dt = 1.0 / 60.0 / 10
        self.mesh = trimesh.load_mesh(str(Path(options['mesh'])))
        print('mass_center:{}'.format(self.mesh.center_mass))
        # convert vertices to numpy array
        vertices = np.array(self.mesh.vertices) - self.mesh.center_mass
        self.raw_translation = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True) # this para is raw as it is just detected by a qr code, not precise
        self.raw_quaternion = torch.tensor([1, 0, 0, 0], dtype=torch.float32, requires_grad=True) # 
        self.translation = []
        self.quaternion = []
        self.v = []
        self.omega = []
        # torch tensors
        self.mass_center = torch.tensor(self.mesh.center_mass, dtype=torch.float32)
        self.x = torch.tensor(vertices, dtype=torch.float32, requires_grad=True)
        for i in range(self.frames * self.substep):
            self.translation.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.quaternion.append(torch.zeros(4, dtype=torch.float32, requires_grad=True))
            self.v.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.omega.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
        self.kn = torch.nn.Parameter(torch.tensor([options['kn']], requires_grad=True))
        self.mu = torch.nn.Parameter(torch.tensor([options['mu']], requires_grad=True))
        self.linear_damping = torch.nn.Parameter(torch.tensor([options['linear_damping']]))
        self.angular_damping = torch.nn.Parameter(torch.tensor([options['angular_damping']]))
        self.init_v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        self.mass = torch.tensor([0.0], dtype=torch.float32)
        self.inv_mass = torch.tensor([0.0], dtype=torch.float32)
        self.target = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)
        with torch.no_grad():
            self.mass = 0
            self.inertia_referance = torch.zeros(3, 3, dtype=torch.float32)
            mass = 1.0
            for i in range(self.mesh.vertices.shape[0]):
                self.mass += mass
                r = self.x[i] - self.mass_center
                # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
                # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
                I_i = mass * (r.dot(r) * torch.eye(3) - torch.outer(r, r))
                self.inertia_referance += I_i
            self.inv_mass = 1.0 / self.mass
        self.set_init_translation(options['translation'])
        # self.set_init_quaternion_from_euler(options['transform'][3:6])
        self.set_init_quaternion(options['quaterion'])

    def add_planar_contact(self, slope_degree, init_height):
        self.c = np.cos(np.deg2rad(slope_degree))
        self.s = np.sin(np.deg2rad(slope_degree))
        self.init_height = init_height

    def set_init_translation(self, init_translation):
        with torch.no_grad():
            self.translation[0] = torch.tensor(init_translation, dtype=torch.float32)
    
    def set_init_quaternion_from_euler(self, init_euler_angle):
        with torch.no_grad():
            self.quaternion[0] = torch.tensor(self.from_euler(init_euler_angle), dtype=torch.float32)
    
    def set_init_quaternion(self, init_quaternion):
        with torch.no_grad():
            self.quaternion[0] = torch.tensor(init_quaternion, dtype=torch.float32)

    def set_init_v(self):
        with torch.no_grad():
            self.v[0] = self.init_v

    def write_out_paras(self, file_path):
        out_dict = {}
        out_kn = self.kn.detach().clone().cpu().numpy().tolist()
        out_mu = self.mu.detach().clone().cpu().numpy().tolist()
        out_r, out_t = {}, {}
        for i in range(self.frames * self.substep):
            if i % self.substep == 0:
                out_t[i // self.substep] = self.translation[i].detach().clone().cpu().numpy().tolist()
                out_r[i // self.substep] = self.quaternion[i].detach().clone().cpu().numpy().tolist()
        out_dict['out_kn'] = out_kn
        out_dict['out_mu'] = out_mu
        out_dict['out_r'] = out_r
        out_dict['out_t'] = out_t
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, "w") as f:
            json.dump(out_dict, f, indent=4)
        return

    # the euler angle is in degree, we first conver it to radian
    def from_euler(self, euler_angle):
        # convert euler angle to quaternion
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        phi = math.radians(euler_angle[0] / 2)
        theta = math.radians(euler_angle[1] / 2)
        psi = math.radians(euler_angle[2] / 2)
        w = math.cos(phi) * math.cos(theta) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
        x = math.sin(phi) * math.cos(theta) * math.cos(psi) - math.cos(phi) * math.sin(theta) * math.sin(psi)
        y = math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.cos(theta) * math.sin(psi)
        z = math.cos(phi) * math.cos(theta) * math.sin(psi) - math.sin(phi) * math.sin(theta) * math.cos(psi)
        return [w, x, y, z]

    def quat_mul(self, a, b):
        return torch.tensor([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                      a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])
    
    def quat_mul_scalar(self, a, b):
        return torch.tensor([a[0] * b, a[1] * b, a[2] * b, a[3] * b])
    
    def quat_add(self, a, b):
        return torch.tensor([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
    
    def quat_subtraction(self, a, b):
        return torch.tensor([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
    
    def quat_normal(self, a)->torch.int32:
        return torch.tensor([a[0] / torch.norm(a), a[1] / torch.norm(a), a[2] / torch.norm(a), a[3] / torch.norm(a)])
    
    def quat_conjugate(self, a):
        return torch.tensor([a[0], -a[1], -a[2], -a[3]])

    def quat_rotate_vector(self, q, v):
        return self.quat_mul(self.quat_mul(q, torch.tensor([0, v[0], v[1], v[2]])), self.quat_conjugate(q))[1:]
    
    def quat_to_matrix(self, q):
        q = q / torch.norm(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.tensor([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
    
    def quat_inverse(self, q):
        return self.quat_conjugate(q) / torch.norm(q)
    
    def GetCrossMatrix(self, a):
        return torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    
    def slope_collision(self, f:torch.int32):
        contact_normal = torch.tensor([-self.s,  self.c, 0.0 ], dtype=torch.float32)
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] +  torch.matmul(self.x, mat_R.t()) + self.mass_center[None] + torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
        vi = self.v[f] + torch.cross(self.omega[f].unsqueeze(0),  torch.matmul(self.x, mat_R.t()), dim=1)
        d = torch.einsum('bi,i->b', xi, contact_normal)
        rel_v = torch.einsum('bi,i->b', vi, contact_normal)
        contact_condition = (d < (-self.c * self.init_height)) & (rel_v < 0.0)
        sum_position = torch.zeros(3, dtype=torch.float32)
        # caculate the how many points are in contact with the plane
        num_collision = torch.sum(contact_condition.int())
        v_out = torch.zeros(3, dtype=torch.float32)
        omega_out = torch.zeros(3, dtype=torch.float32)
        if num_collision > 0:
            contact_mask = contact_condition.float()[:, None]  # add a new axis to broadcast
            # calculate the sum of the contact points
            sum_position = torch.sum(self.x * contact_mask, dim=0)
            # calculate the average of the contact points
            collision_ri = sum_position / num_collision
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = self.v[f] + self.omega[f].cross(collision_Ri)
            v_i_n = vi.dot(contact_normal) * contact_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.kn) * (torch.norm(v_i_n) / torch.norm(v_i_t)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha  * v_i_t
            # print('f: {}, alpha:{}, vt_new:{}, vt:{}, item: vi_t_normal: {}, vi_n_normal: {}'.format(f, alpha, vt_new, v_i_t, torch.norm(v_i_t), torch.norm(v_i_n)))
            vi_new = vn_new + vt_new
            I = mat_R @ self.inertia_referance @ mat_R.t()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0],\
                        [0.0, self.inv_mass, 0.0],\
                        [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ I.inverse() @ collision_Rri_mat
            J = k.inverse() @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + I.inverse() @ collision_Rri_mat @ J
        return v_out, omega_out

    def sdf_collision(self, f:torch.int32):
        # # collision detect
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] +  torch.matmul(self.x, mat_R.t()) + self.mass_center
        sdf_value, sdf_grad = self.query_background_sdf(xi)
        sdf_value = sdf_value.reshape(-1)
        vi = self.v[f] + torch.cross(self.omega[f].unsqueeze(0),  torch.matmul(self.x, mat_R.t()), dim=1)
        rel_v = torch.einsum('bi, bi->b', vi, sdf_grad)
        contact_condition = (sdf_value < 0.0) & (rel_v < 0.0)
        v_out = torch.zeros(3, dtype=torch.float32)
        omega_out = torch.zeros(3, dtype=torch.float32)
        # caculate the how many points are in contact with the plane
        num_collision = torch.sum(contact_condition.int())
        # collision response
        # impluse method
        if num_collision > 0:
            contact_mask = contact_condition.float()[:, None]  # add a new axis to broadcast
            # calculate the sum of the contact points
            sum_position = torch.sum(self.x * contact_mask, dim=0)           
            # calculate the average of the contact points
            collision_ri = sum_position / num_collision
            # calculate the collision point
            collision_x = self.translation[f] + mat_R @ collision_ri + self.mass_center
            value, collision_normal = self.query_background_sdf(collision_x.reshape(1,3))
            # print('collision_ri:{}'.format(collision_ri))
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = self.v[f] + self.omega[f].cross(collision_Ri)

            v_i_n = vi.dot(collision_normal) * collision_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.kn) * (torch.norm(v_i_n) / (torch.norm(v_i_t) + 1e-6)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha * v_i_t
            vi_new = vn_new + vt_new
            inertial_inv = torch.inverse(mat_R @ self.inertia_referance @ mat_R.t())
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ inertial_inv @ collision_Rri_mat
            J = torch.inverse(k) @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + inertial_inv @ collision_Rri_mat @ J
        return v_out, omega_out

    def physical_forward(self, f:torch.int32):
        # advect
        v_out = (self.v[f] + torch.tensor([0.0, 0.0, -9.8]) * self.dt) * self.linear_damping
        omega_out = self.omega[f] * self.angular_damping
        v_out_, omega_out_ = self.sdf_collision(f=f)
        # v_out_, omega_out_ = self.slope_collision(f=f)
        v_out = v_out + v_out_
        omega_out = omega_out + omega_out_
        # J = F · Δt = m · Δv,  F = m · Δv / Δt = J / Δt
        # torque = r × F = r × (J / Δt) = (r × J) / Δt
        # Δω = I^(-1) · torque · Δt = I^(-1) · (r × J) / Δt · Δt = I^(-1) · (r × J)
        # update state
        wt = omega_out * self.dt * 0.5
        dq = self.quat_mul(torch.tensor([0.0, wt[0], wt[1], wt[2]], dtype=torch.float32), self.quaternion[f])
        self.translation[f + 1] = self.translation[f] + self.dt * v_out
        self.omega[f + 1] = omega_out
        self.v[f + 1] = v_out
        quat_new = self.quaternion[f] + dq
        self.quaternion[f + 1] = quat_new / torch.norm(quat_new)

    def get_transform_matrix(self, translation, quaternion):
        w, x, y, z = quaternion
        transform_matrix = torch.tensor([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w), translation[0]],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w), translation[1]],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2), translation[2]],
            [0, 0, 0, 1.0]
        ], device=self.device, requires_grad=True, dtype=torch.float32)
        transform_matrix_inv = torch.inverse(transform_matrix)  # make an inverse
        transform_matrix_inv.requires_grad_(True)
        return transform_matrix, transform_matrix_inv
        
    def query_background_sdf(self, pts: torch.Tensor):
        # return None, None
        sdf = self.runner_background.sdf_network.sdf(pts).contiguous()
        sdf_grad = self.runner_background.sdf_network.gradient(pts).squeeze().contiguous()
        return sdf, sdf_grad

    def forward(self, max_f: int, vis_folder=None):
        pbar = trange(1, max_f)
        pbar.set_description('\033[5;41mForward\033[0m')
        global_loss = 0
        print('optimizer init v = ', self.init_v)
        for i in pbar:
            print_blink(f'frame id : {i}')
            orgin_mat_c2w = torch.from_numpy(self.cameras_M[i].astype(np.float32)).to(self.device)
            # orgin_mat_K_inv = torch.from_numpy(np.linalg.inv(self.cameras_K[i].astype(np.float32))).to(self.device)
            for f in range(self.substep * (i - 1), self.substep * i):
                self.physical_forward(f)
            rays_gt, rays_mask, rays_o, rays_d = self.rays_gt_all[i], self.rays_mask_all[i], self.rays_o_all[i], \
            self.rays_v_all[i]
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[
                rays_mask].reshape(-1, 3)  # reshape is used for after mask, it become [len*3]
            rays_sum = len(rays_o)
            debug_rgb = []
            for rays_o_batch, rays_d_batch, rays_gt_batch in zip(rays_o.split(self.batch_size),
                                                                 rays_d.split(self.batch_size),
                                                                 rays_gt.split(self.batch_size)):
                near, far = self.runner_object.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = None
                # this render out contains grad & img loss, find out its reaction with phy simualtion
                render_out = self.runner_object.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                        near=near, far=far,
                                                                        R=self.quaternion[f + 1], T=self.translation[f + 1],
                                                                        camera_c2w=orgin_mat_c2w,
                                                                        cos_anneal_ratio=self.runner_object.get_cos_anneal_ratio(),
                                                                        background_rgb=background_rgb)               
                color_fine = render_out["color_fine"]
                color_error = (color_fine - rays_gt_batch)
                debug_rgb.append(color_fine.clone().detach().cpu().numpy())
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),
                                            reduction='sum') / rays_sum / max_f  # normalize
                global_loss += color_fine_loss.clone().detach()
                color_fine_loss.backward(retain_graph=True)  # img_loss for refine R & T
                torch.cuda.synchronize()
                del render_out
            ### img_debug should has same shape as rays_gt
            debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8)
            W, H, cnt = self.W, self.H, 0
            rays_mask = (rays_mask.detach().cpu().numpy()).reshape(H, W, 3)
            debug_img = np.zeros_like(rays_mask).astype(np.float32)
            for index in range(0, H):
                for j in range(0, W):
                    if rays_mask[index][j][0]:
                        debug_img[index][j][0] = debug_rgb[cnt][0]
                        debug_img[index][j][1] = debug_rgb[cnt][1]
                        debug_img[index][j][2] = debug_rgb[cnt][2]
                        cnt = cnt + 1
            print_blink("saving debug image at " + str(i) + " index")
            if vis_folder !=None:
                cv.imwrite((vis_folder / (str(i) + ".png")).as_posix(), debug_img)
            pbar.set_description(f"[Forward] loss: {global_loss.item()}")
            # self.export_mesh(f + 1)
        return global_loss
    
    def export_mesh(self, f:torch.int32):
        with torch.no_grad():
            mat_R = self.quat_to_matrix(self.quaternion[f])
            xi = self.translation[f] +  torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
            faces = self.mesh.faces
            mesh = trimesh.Trimesh(vertices=xi.clone().detach().cpu().numpy(), faces=faces)
            mesh.export(str(Path('mesh_result') / '{}.obj'.format(f // self.substep)))

    def backward(self, max_f: np.int32):
        pbar = trange(1, max_f)
        pbar.set_description('\033[5;30m[Backward]\033[0m')
        for i in pbar:
            f = max_f - i - 1
            with torch.no_grad():
                translation_grad = self.translation[f].grad
                quaternion_grad = self.quaternion[f].grad
            if f > 0:
                self.physical_simulator.set_motion_grad(f, translation_grad, quaternion_grad)
                self.physical_simulator.backward(f, lambda x: self.query_background_sdf(x))
            else:
                v_grad, omega_grad, ke_grad, mu_grad, translation_grad, quaternion_grad = \
                    self.physical_simulator.backward(f, lambda x: self.query_background_sdf(x))
                print_ok('init_v grad = ', v_grad)
                self.init_v.backward(retain_graph=True, gradient=v_grad)
                self.init_omega.backward(retain_graph=True, gradient=omega_grad)
                self.init_ke.backward(retain_graph=True, gradient=ke_grad)
                self.init_mu.backward(retain_graph=True, gradient=mu_grad)
                self.init_translation.backward(retain_graph=True, gradient=translation_grad)
                self.init_quaternion.backward(retain_graph=True, gradient=quaternion_grad)
        return
    
    def refine_RT(self, image_id = 0, vis_folder=None, iter_id=-1, write_out_result=True):
        # this function is used to refine RT to fit the initial dynamic scene (as frame 0)
        global_loss = 0
        orgin_mat_c2w = torch.from_numpy(self.cameras_M[image_id].astype(np.float32)).to(self.device)
        rays_gt, rays_mask, rays_o, rays_d = self.rays_gt_all[image_id], self.rays_mask_all[image_id], self.rays_o_all[image_id], self.rays_v_all[image_id]
        rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[
            rays_mask].reshape(-1, 3)  # reshape is used for after mask, it become [len*3]
        rays_sum = len(rays_o)
        debug_rgb = []
        for rays_o_batch, rays_d_batch, rays_gt_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size), rays_gt.split(self.batch_size)):
            near, far = self.runner_object.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = None
            render_out = self.runner_object.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                    near=near, far=far, R=self.raw_quaternion, T=self.raw_translation,
                                                                    camera_c2w=orgin_mat_c2w,
                                                                    cos_anneal_ratio=self.runner_object.get_cos_anneal_ratio(),
                                                                    background_rgb=background_rgb)    
            color_fine = render_out["color_fine"]
            color_error = (color_fine - rays_gt_batch)
            debug_rgb.append(color_fine.clone().detach().cpu().numpy())
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),reduction='sum') / rays_sum  # normalize
            global_loss += color_fine_loss.clone().detach()
            color_fine_loss.backward(retain_graph=True)  # img_loss for refine R & T
            torch.cuda.synchronize()
            del render_out
        ### img_debug should has same shape as rays_gt
        debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8)
        W, H, cnt = self.W, self.H, 0
        rays_mask = (rays_mask.detach().cpu().numpy()).reshape(H, W, 3)
        debug_img = np.zeros_like(rays_mask).astype(np.float32)
        for index in range(0, H):
            for j in range(0, W):
                if rays_mask[index][j][0]:
                    debug_img[index][j][0] = debug_rgb[cnt][0]
                    debug_img[index][j][1] = debug_rgb[cnt][1]
                    debug_img[index][j][2] = debug_rgb[cnt][2] 
                    cnt = cnt + 1
        if vis_folder !=None and write_out_result:
            print_blink("saving debug image at " + str(iter_id) + "th validation, with image inedex " + str(image_id))
            cv.imwrite((vis_folder / (str(iter_id) + "_" + str(image_id) + ".png")).as_posix(), debug_img)
        R_ek_loss = torch.abs(torch.norm(self.raw_quaternion) - 1) 
        R_ek_loss.backward(retain_graph=True)
        global_loss = R_ek_loss + global_loss
        return global_loss

    def render_depth_core(self, rays_o, rays_d, translation, quaternion, original_camera_c2w
                          , query_background_flag=0, zero_sdf_thereshold=1e-3, inf_depth_thereshold=2.0, near = 1e-2, far=2.0):
        # this function is written for checking two nerf networks' depth, use sdf to calculate
        # returns [len(rays_o), 1](float)) as a render result, query_background_flag = 0 means background, 1 means object, 
        # when queried sdf < zero_sdf_thereshold means reach the surface, sdf > inf_depth_thereshold means reach the inf place, too far
        ### maybe failed because the sdf field is periodic
        # calc the real rays_o & rays_d for R + T as to object
        def query_sdf_points(pts, flag): # pts should be an [len, 3] shape torch tensor
            sdfs = None
            if flag == 0 : # use object
                sdfs = self.runner_object.sdf_network.sdf(pts).contiguous()
            elif flag == 1:# use background
                sdfs = self.runner_background.sdf_network.sdf(pts).contiguous()
            return sdfs
        mat_R = self.quat_to_matrix(quaternion)
        tmp_transform_matrix = torch.zeros((4, 4), dtype=torch.float32)
        tmp_transform_matrix[:3, :3] = mat_R
        tmp_transform_matrix[:3, 3] = translation
        tmp_transform_matrix[3, 3] = 1.0
        transform_matrix_inv = torch.inverse(tmp_transform_matrix)  # make an inverse
        camera_pos = torch.matmul(transform_matrix_inv, original_camera_c2w) # equivalent camera pose
        rays_d = torch.matmul(transform_matrix_inv[None, :3, :3], rays_d[:, :, None]).squeeze()  # W, H, 3
        rays_o = camera_pos[None, :3, 3].expand(rays_d.shape)  # block size, 3
        rays_o = rays_o.clone()
        depths, sdfs = torch.zeros(len(rays_o), dtype=torch.float32), torch.ones(len(rays_o), dtype=torch.float32) # block size
        rays_mask, zero_mask, inf_mask =  torch.ones((len(rays_o)), dtype=torch.bool), \
            torch.ones((len(rays_o)), dtype=torch.bool), torch.ones((len(rays_o)), dtype=torch.bool)
        while torch.sum(rays_mask) > 0:
            pts, dirs = rays_o[rays_mask], rays_d[rays_mask]
            tmp_sdfs = query_sdf_points(pts, flag=query_background_flag).squeeze()
            pts = pts + dirs * (tmp_sdfs.repeat(3, 1).T)
            rays_o[rays_mask] = pts # update current_rays
            depths[rays_mask] = depths[rays_mask] + tmp_sdfs
            sdfs[rays_mask] = tmp_sdfs
            zero_mask, inf_mask = sdfs < zero_sdf_thereshold, sdfs > inf_depth_thereshold
            rays_mask = zero_mask + inf_mask
            rays_mask = ~rays_mask
        depths = (depths.clip(near, far)) / far
        # this is a calculation using progressive photon mapping to calculate depth for each single ray, upper is its refine, use batch calculation
        # depths = (1 / depths - 1 / near) / (1 / far - 1 / near)
        # for depth_index in range(0, len(rays_o)):
        #     ray_o, ray_d, tmp_sdf, acc_distance = rays_o[depth_index], rays_d[depth_index], 1, 0 # use 1 meter as default start
        #     while tmp_sdf > zero_sdf_thereshold:
        #         tmp_sdf, _ = query_sdf_single_point(ray_o, query_background_flag)
        #         ray_o = ray_o + tmp_sdf * ray_d # move the sample point
        #         acc_distance = acc_distance + tmp_sdf # add the acc_dis
        #         if tmp_sdf > inf_depth_thereshold:
        #             break
        #     acc_distance = max(near, min(far, acc_distance))
        #     acc_distance = (1 / acc_distance - 1 / near) / (1 / far - 1 / near) # turn into depth
            # depths[depth_index] = acc_distance
        return depths

    def render_with_depth(self, translation, quaternion, image_index=0, resolution_level=1):
        def feasible(key):
            return (key in render_out) and (render_out[key] is not None)
        # returns final render result
        orgin_mat_c2w = torch.from_numpy(self.cameras_M[image_index].astype(np.float32)).to(self.device)
        if resolution_level > 1:
            camera_K = self.cameras_K[image_index].astype(np.float32)
            camera_M = self.cameras_M[image_index].astype(np.float32)
            rays_o, rays_d = generate_rays_with_K_and_M(transform_matrix=camera_M, intrinsic_mat=camera_K, W=self.W, H=self.H, resolution_level=resolution_level)
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        else:
            rays_o, rays_d = self.rays_o_all[image_index],  self.rays_v_all[image_index] # full resolution
        out_rgb_fine, backgorund_depth_fine_all, object_depth_fine_all = [], [], [] # final result
        rays_count, total_rays = 0, len(rays_o)
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            near, far = self.runner_object.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = None
            object_color_fine, object_depth_fine, backgorund_color_fine, background_depth_fine = [], [], [], []
            # this render out contains grad & img loss, find out its reaction with phy simualtion
            render_out = self.runner_object.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                    near=near, far=far,
                                                                    T=translation, R=quaternion,
                                                                    camera_c2w=orgin_mat_c2w,
                                                                    cos_anneal_ratio=self.runner_object.get_cos_anneal_ratio(),
                                                                    background_rgb=background_rgb)
            if feasible('color_fine'):
                object_color_fine = (render_out['color_fine'].detach().cpu().numpy())
            object_depth_fine = self.render_depth_core(rays_o=rays_o_batch, rays_d=rays_d_batch, translation=translation, quaternion=quaternion, 
                                                       original_camera_c2w=orgin_mat_c2w, query_background_flag=0).detach().cpu().numpy()
            object_depth_fine_all.append(np.expand_dims(object_depth_fine, axis = -1).repeat(3, 1))
            # for background, the sdf calc does not need to considerate RT
            R, T = torch.tensor([1, 0, 0, 0], dtype=torch.float32), torch.tensor([0, 0, 0], dtype=torch.float32)
            render_out = self.runner_background.renderer.render(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                    near=near, far=far,
                                                                    cos_anneal_ratio=self.runner_background.get_cos_anneal_ratio(),
                                                                    background_rgb=background_rgb)
            if feasible('color_fine'):
                backgorund_color_fine = (render_out['color_fine'].detach().cpu().numpy())
            background_depth_fine = self.render_depth_core(rays_o=rays_o_batch, rays_d=rays_d_batch, translation=T, quaternion=R, 
                                                       original_camera_c2w=orgin_mat_c2w, query_background_flag=1).detach().cpu().numpy()
            # compare depth, use small depth pirior
            backgorund_depth_fine_all.append(np.expand_dims(background_depth_fine, axis = -1).repeat(3, 1))
            out_object_mask = np.where(object_depth_fine < background_depth_fine, 1, 0).astype(np.bool_)
            out_rgb_fine_block = backgorund_color_fine
            out_rgb_fine_block[out_object_mask] = object_color_fine[out_object_mask]
            out_rgb_fine.append(out_rgb_fine_block)
            rays_count = rays_count + self.batch_size
            print("process: ", rays_count, " /", total_rays)
        object_depth_fine_all = (np.concatenate(object_depth_fine_all, axis=0).reshape([int(self.H / resolution_level), int(self.W / resolution_level), 3]) * 255).clip(0, 255).astype(np.uint8)    
        backgorund_depth_fine_all = (np.concatenate(backgorund_depth_fine_all, axis=0).reshape([int(self.H / resolution_level), int(self.W / resolution_level), 3]) * 255).clip(0, 255).astype(np.uint8)    
        out_rgb_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([int(self.H / resolution_level), int(self.W / resolution_level), 3]) * 255).clip(0, 255).astype(np.uint8)    
        return out_rgb_fine, object_depth_fine_all, backgorund_depth_fine_all
 
    def render_with_mask(self, translation, quaternion, image_index=0, black_color_thereshold=1e-2):
        # black_color_thereshold is used to confirm whether the color is black as the background
        def feasible(key):
            return (key in render_out) and (render_out[key] is not None)
        # returns final render result
        orgin_mat_c2w = torch.from_numpy(self.cameras_M[image_index].astype(np.float32)).to(self.device)
        rays_o, rays_d = self.rays_o_all[image_index],  self.rays_v_all[image_index]
        rays_mask = torch.ones_like(rays_o, dtype=torch.bool)
        rays_o, rays_d = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3)   # reshape is used for after mask, it become [len*3]
        out_rgb_fine = [] # final result
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            near, far = self.runner_object.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = None
            object_color_fine, object_depth_fine, backgorund_color_fine, background_depth_fine = [], [], [], []
            # this render out contains grad & img loss, find out its reaction with phy simualtion
            render_out = self.runner_object.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                    near=near, far=far,
                                                                    T=translation, R=quaternion,
                                                                    camera_c2w=orgin_mat_c2w,
                                                                    cos_anneal_ratio=self.runner_object.get_cos_anneal_ratio(),
                                                                    background_rgb=background_rgb)
            if feasible('color_fine'):
                object_color_fine = (render_out['color_fine'].detach().cpu().numpy())
            if feasible('depth_map'):
                object_depth_fine = (render_out['depth_map'].detach().cpu().numpy())
            render_out = self.runner_background.renderer.render(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                    near=near, far=far,
                                                                    cos_anneal_ratio=self.runner_background.get_cos_anneal_ratio(),
                                                                    background_rgb=background_rgb)
            if feasible('color_fine'):
                backgorund_color_fine = (render_out['color_fine'].detach().cpu().numpy())
            if feasible('depth_map'):
                background_depth_fine = (render_out['depth_map'].detach().cpu().numpy())
            # compare depth, use small depth pirior
            out_object_mask = np.where(object_color_fine < black_color_thereshold, 0, 1).astype(np.bool_)
            out_rgb_fine_block = np.where(out_object_mask, object_color_fine, backgorund_color_fine)
            object_depth_fine = np.where(out_object_mask, object_depth_fine, 1) # make the black background as the farest
            out_rgb_fine.append(out_rgb_fine_block)
 
        out_rgb_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([self.H, self.W, 3]) * 256).clip(0, 255)    
        return out_rgb_fine
    
    # TODO: Finish new loss calculation setting
    
def get_optimizer(mode, genshinStart):
    optimizer = None
    if mode == "train_static":
        optimizer = torch.optim.Adam(
            [
                {"params": getattr(), 'lr': 1e-1}
            ]
        )
    elif mode == "train_velocity":
        optimizer = torch.optim.LBFGS(
            [
                {"params": getattr(genshinStart, 'init_v'), 'lr': 1e-1}
            ]
        )
    elif mode == "train_dynamic":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(genshinStart, 'mu'), 'lr': 1e-2},
                {'params': getattr(genshinStart, 'kn'), 'lr': 1e-2},
                {"params": getattr(genshinStart, 'init_v'), 'lr': 1e-2}
            ],
            amsgrad=False
        )
    elif mode == "refine_rt":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(genshinStart, 'raw_translation'), 'lr': 1e-2},
                {'params': getattr(genshinStart, 'raw_quaternion'), 'lr': 1e-2},
            ],
            amsgrad=False
        )
    return optimizer

def train_dynamic(max_f, iters, genshinStart):
    def train_forward(optimizer, vis_folder= None):
        optimizer.zero_grad()
        if vis_folder  != None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        loss = torch.tensor(np.nan)
        while loss.isnan():
            loss = genshinStart.forward(max_f, vis_folder)
        return loss

    optimizer = get_optimizer('train_dynamic', genshinStart)
    for i in range(iters):
        genshinStart.set_init_v()
        loss = train_forward(optimizer=optimizer, vis_folder=Path('train_dynamic') / ('iter_' + str(i)))
        if loss.norm() < 1e-6:
            break
        optimizer.step()
        out_json_path = "./debug/train_dynamic/out_jsons/" + str(i) + ".json"
        genshinStart.write_out_paras(out_json_path)
        print('mu: {}, kn: {}'.format(genshinStart.mu, genshinStart.kn))
        
def refine_RT(genshinStart, iters=100, init_R=None, init_T=None, require_init=False, image_id=0):
    def refine_rt_forward(optimizer, vis_folder= None, iter_id=-1):
        optimizer.zero_grad()
        if vis_folder != None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        loss = genshinStart.refine_RT(vis_folder=vis_folder, iter_id=iter_id, image_id=image_id)

        return loss    
    if require_init: # not set init rt for the first loop, need init
            if init_R is None or init_T is None:
                init_R, init_T = [], []  # should be len(4) and len(3) array 
    optimizer = get_optimizer('refine_rt', genshinStart=genshinStart)
    genshinStart.raw_translation, genshinStart.raw_quaternion = init_T, init_R

    optimizer = get_optimizer('refine_rt', genshinStart)
    for i in range(iters):
        genshinStart.set_init_v()
        loss = refine_rt_forward(optimizer=optimizer, vis_folder=Path('refine_rt'), iter_id=i)
        if loss.norm() < 1e-3:
            break
        optimizer.step()
        out_json_path = "./debug/refine_rt_single/out_jsons/" + str(i) + ".json"
        genshinStart.write_out_paras(out_json_path)
        print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(genshinStart.raw_translation, genshinStart.raw_quaternion, loss.norm()))
    return

    """this function is used to call refine_RT function for calculating a set of RT in dynamic video sequence
    this function requires the lenth of the input image sequence to finish RT calculation"""
def refine_RT_seqnuece(genshinStart, sequence_length, iters=1, init_R=None, init_T=None, write_out_folder=None):
    def refine_rt_forward(optimizer, image_id=0, vis_folder= None, iter_id=-1):
        optimizer.zero_grad()
        if vis_folder != None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        loss = genshinStart.refine_RT(image_id=image_id, vis_folder=vis_folder, iter_id=iter_id, write_out_result=True)
        return loss   
    refined_RT_in_sequence = {} # store result json
    for image_id in range(0, sequence_length):
        # refine image_id th image
        optimizer = get_optimizer('refine_rt', genshinStart=genshinStart)
        if image_id == 0:
            genshinStart.raw_translation, genshinStart.raw_quaternion = init_T, init_R # only the first frame needs to init
        optimizer = get_optimizer('refine_rt', genshinStart)
        for i in range(iters):
            genshinStart.set_init_v()
            if write_out_folder is not None:
                loss = refine_rt_forward(optimizer=optimizer, image_id=image_id, vis_folder= write_out_folder / str(image_id), iter_id=i)
            else:
                loss = refine_rt_forward(optimizer=optimizer, image_id=image_id, iter_id=i)      
            if loss.norm() < 1e-2:
                break
            optimizer.step()
            # normalize init R
            print("calculated norm: " + str(loss.norm))
            print('raw_translation: {}, raw_quaternion: {}'.format(genshinStart.raw_translation, genshinStart.raw_quaternion))
        # refine finished, store RT result
        tmp_name = str(image_id) + "_"
        refined_RT_in_sequence[tmp_name + "R"] = genshinStart.raw_quaternion.detach().cpu().numpy().tolist()
        refined_RT_in_sequence[tmp_name + "T"] = genshinStart.raw_translation.detach().cpu().numpy().tolist()
        print("new RT after refine " + str(refined_RT_in_sequence[tmp_name + "R"]) + " " + str(refined_RT_in_sequence[tmp_name + "T"]))
    if write_out_folder is not None: # write out result
        result_json_path= str(write_out_folder) + str("/out.json")
        with open(result_json_path, "w") as f:
            json.dump(refined_RT_in_sequence, f, indent=4)       
    return

def render_with_depth(genshinStart, translation, quaternion, write_out_path=None, image_index=0, resolution_level=1): # need to changed into a full sequence render
    render_out_rgb, bg_depth, obj_depth = genshinStart.render_with_depth(translation=translation, quaternion=quaternion, image_index=image_index, resolution_level=resolution_level)
    if write_out_path is not None:
        print_blink("saving result image at " + write_out_path)
        cv.imwrite(write_out_path, render_out_rgb)
        cv.imwrite(write_out_path[0:-4] + "_bg_depth.png", bg_depth)
        cv.imwrite(write_out_path[0:-4] + "_obj_depth.png", obj_depth)
    return 

def render_full_sequence(genshinStart, rt_json_path, write_out_dir, image_count=1): # need to changed into a full sequence render
    if not Path(write_out_dir).exists():
        print("making new dir " + write_out_dir)
        os.makedirs(Path(write_out_dir)) # make dir
    rt_params_list = None
    with open(rt_json_path, "r") as json_file:
        rt_params_list = json.load(json_file)
    for index in range(0, image_count): # render rgb for sequence by mask, need to be replaced into depth-based render in the future
        # json loads as 0_R, 0_T and so on
        translation, quaternion = rt_params_list[str(index) + "_T"], rt_params_list[str(index) + "_R"]
        translation = torch.tensor(translation, dtype=torch.float32)
        quaternion = torch.tensor(quaternion, dtype=torch.float32)
        render_out_rgb, _, _ = genshinStart.render_with_depth(translation=translation, quaternion=quaternion, image_index=index)
        write_out_path = write_out_dir + "/" + str(index) + ".png"
        print_blink("saving result image at " + write_out_path)
        cv.imwrite(write_out_path, render_out_rgb)
    return 

if __name__ == '__main__':
    print_blink('Genshin Nerf, start!!!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/json/base.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image_id', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    genshinStart = GenshinStart(args.conf)
    optimizer = get_optimizer('train_dynamic', genshinStart=genshinStart)
    if args.mode == "train":
        train_dynamic()
    elif args.mode == "refine_rt":
        init_R, init_T = torch.tensor([0.8908351063728333, -0.36010658740997314, -0.10959087312221527, 0.254412978887558], dtype=torch.float32, requires_grad=True), torch.tensor([0.1520, -0.1390,  0.3170], dtype=torch.float32, requires_grad=True) # use 0 as default
        refine_RT(genshinStart=genshinStart, init_R=init_R, init_T=init_T, image_id=args.image_id, iters=100)
    elif args.mode == "refine_rt_sequence":
        init_R, init_T = torch.tensor([0.8908351063728333, -0.36010658740997314, -0.10959087312221527, 0.254412978887558], dtype=torch.float32, requires_grad=True), torch.tensor([0.1520, -0.1390,  0.3170], dtype=torch.float32, requires_grad=True)
        refine_RT_seqnuece(genshinStart=genshinStart, init_R=init_R, init_T=init_T, sequence_length = 21, write_out_folder=Path("debug", "refine_rt_sequence"), iters=50)
    elif args.mode == 'render_with_depth':
        init_R, init_T = torch.tensor([0.8908351063728333, -0.36010658740997314, -0.10959087312221527, 0.254412978887558], dtype=torch.float32, requires_grad=True), torch.tensor([0.15002988278865814, -0.1365453451871872, 0.3106136620044708], dtype=torch.float32, requires_grad=True)
        write_out_path = Path("debug", "render_with_depth")
        if not write_out_path.exists():
            os.makedirs(write_out_path)
        write_out_path = str(write_out_path) + "/0.png"
        render_with_depth(genshinStart=genshinStart, image_index=0, translation=init_T, quaternion=init_R, write_out_path=write_out_path, resolution_level=1)
    elif args.mode == 'render_result_full':
        rt_json_path = Path("debug", "out2.json")
        write_out_dir = Path("debug", "render_result_full_sequence_for_train_dynamic")
        render_full_sequence(genshinStart=genshinStart, rt_json_path=str(rt_json_path), write_out_dir=str(write_out_dir), image_count=21)
    else:
        train_dynamic(genshinStart.frame_counts, iters=1000, genshinStart=genshinStart)
""" 
D:\gitwork\genshinnerf> python genshin_start_copy.py --mode debug --conf ./dynamic_test/genshin_start.json --case bird
python genshin_start.py --mode debug --conf ./dynamic_test/genshin_start.json --case bird
python genshin_start.py --mode debug --conf ./confs/json/furina.json
python genshin_start.py --mode refine_rt --conf ./confs/json/nahida.json --gpu 1
python genshin_start.py --mode debug --conf ./confs/json/nahida.json --gpu 1
python genshin_start.py --mode refine_rt_sequence --conf ./confs/json/nahida.json
python genshin_start.py --mode render_with_depth --conf ./confs/json/nahida.json --gpu 0
python genshin_start.py --mode render_result_full --conf ./confs/json/nahida.json --gpu 3
"""
