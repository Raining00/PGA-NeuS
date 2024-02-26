import os
import time
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
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import json


def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[93m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            if self.dataset.focus_rays_in_mask:
                rays_o, rays_d, true_rgb, mask = self.dataset.select_random_rays_in_masks(
                    image_perm[self.iter_step % len(image_perm)], self.batch_size)
            else:
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
                rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            # center = torch.Tensor([0.05, -0.1, 0]).cuda()
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.zeros([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def train_dynamic(self):
        # TODO use render_dynamic to pass img_loss
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'dynamic_logs'))
        return

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        depth_map = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('depth_map'):
                depth_map.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out
        # depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        # import pdb; pdb.set_trace()
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        if len(depth_map) > 0:
            depth_map = (np.concatenate(depth_map, axis=0).reshape([H, W, -1]) * 256).clip(0, 255).astype(np.uint8)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(depth_map) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_depth.png'.format(self.iter_step, i, idx)),
                           depth_map[..., i])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def render_novel_image_at(self, camera_pose, resolution_level, intrinsic_inv=None):
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(camera_pose, resolution_level=resolution_level,intrinsic_inv=intrinsic_inv)
        
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        # import pdb; pdb.set_trace()
        
        out_rgb_fine = []
        normal_fine = []
        n_samples = self.renderer.n_samples + self.renderer.n_importance
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            # normal_fine.append((render_out['gradients'] * render_out['weights'][:, :n_samples, None])).detach().cpu().numpy()
            del render_out
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine, normal_fine

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)), encoding='ascii')
        print("save at " + os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,'{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),fourcc, 30, (w, h))
        for image in images:
            writer.write(image)
        writer.release()

    def save_render_pic_at(self, setting_json_path):
        img = self.render_novel_image_at(camera_pose, 2)
        set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
        file_name_with_extension = os.path.basename(setting_json_path)
        case_name, file_extension = os.path.splitext(file_name_with_extension)
        render_path = set_dir + "/" + case_name + ".png"
        print("Saving render img at " + render_path)
        cv.imwrite(render_path, img)

    def render_motion(self, setting_json_path):
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        if motion_data["frames"] is None:
            print_error("must provide a sequence of motion information")
            exit()
        frames = motion_data["frames"]
        print_info(f"{frames} frames will be rendered.")
        motion_transforms = motion_data["results"]
        original_mat = motion_data["1_1_M"]
        if original_mat == None:
            print_error("static camera information must be provided")
        for i in tqdm(range(1)):
            motion_transform = motion_transforms[i]
            assert i == motion_transform["frame_id"], "invalid frame sequence"
            t, q = motion_transform['translation'], motion_transform['rotation'],
            q = [0.9515, 0.1449, 0.2685, 0.0381]
            t = [0000, 0.0000, 0.8659]
            w, x, y, z = q
            rotate_mat = np.array([
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
            ])
            transform_matrix = np.zeros((4, 4))
            transform_matrix[0:3, 0:3] = rotate_mat
            transform_matrix[0:3, 3] = t
            transform_matrix[3, 3] = 1.0
            inverse_matrix = np.linalg.inv(transform_matrix)
            camera_pose = np.array(original_mat)

            img = self.render_novel_image_at(camera_pose, 2)
            # img loss
            set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
            file_name_with_extension = os.path.basename(setting_json_path)
            case_name, file_extension = os.path.splitext(file_name_with_extension)
            render_path = f"{set_dir}/test_render_motion{i:04d}.png"
            print("Saving render img at " + render_path)
            cv.imwrite(render_path, img)
            print_info(f"finish rendering frame:{i}")

        print_ok(f"{frames} images has been rendered!")

    def render_novel_image_with_RTKM(self, post_fix=1):
        q, t = [1, 0, 0, 0], [0, 0, 0] # this is a default setting
        q = [0.46320000290870667, -0.04439999908208847, 0.03220000118017197, -0.8848999738693237]
        t =  [-0.09430000185966492, -0.022700000554323196, 0.1590999960899353]
        w, x, y, z = q
        rotate_mat = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, 0:3] = rotate_mat
        transform_matrix[0:3, 3] = t
        transform_matrix[3, 3] = 1.0
        inverse_matrix = np.linalg.inv(transform_matrix)
        original_mat = np.array(
 [[ 0.94636756, -0.1777333,   0.2698134,   -0.14891088],
  [-0.31938437, -0.6407732,   0.69814277,  -1.0693798 ],
  [ 0.04880596, -0.7468739,  -0.6631722,    0.8686853 ],
  [ 0.,          0.,          0.,          1.        ]]
       )
        
        intrinsic_mat = np.array(
 [[ 4.35143164e+03, -3.63514664e-05,  1.19279785e+03,  0.00000000e+00],
  [ 0.00000000e+00,  4.40391943e+03,  5.34325195e+02,  0.00000000e+00],
  [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        )
        intrinsic_inv = torch.from_numpy(np.linalg.inv(intrinsic_mat).astype(np.float32)).cuda()
        camera_pose = np.array(original_mat)
        transform_matrix = inverse_matrix @ camera_pose
        # import pdb; pdb.set_trace()
        self.dataset.W = 2336
        self.dataset.H = 1080
        # transform_matrix =transform_matrix.astype(np.float32).cuda()
        img, normal = self.render_novel_image_at(transform_matrix, resolution_level=6, intrinsic_inv=intrinsic_inv)
        # img loss
        # set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
        # file_name_with_extension = os.path.basename(setting_json_path)
        # case_name, file_extension = os.path.splitext(file_name_with_extension)
        render_path = os.path.join(self.base_exp_dir, "test_" + str(post_fix) + ".png")
        print("Saving render img at " + render_path)
        cv.imwrite(render_path, img)

    def get_runner(neus_conf_path, case_name, is_continue):
        return Runner(neus_conf_path, mode="train", case=case_name, is_continue=is_continue)


if __name__ == '__main__':
    print('Genshin Nerf, start!!!')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(args.gpu)

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--post_fix', type=int, default=0)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=128, threshold=args.mcube_threshold)
    elif args.mode == 'render_at':
        runner.save_render_pic_at(args.render_at_pose_path)
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'render_motion':
        runner.render_motion(args.render_at_pose_path)
    elif args.mode == 'train_dynamic':
        runner.train_dynamic_single_frame(args.render_at_pose_path)
    elif args.mode == 'render_rtkm':
        runner.render_novel_image_with_RTKM(post_fix=args.post_fix)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

"""
conda activate neus
cd D:/gitwork/NeuS
python exp_runner.py --mode train_dynamic --conf ./confs/wmask.conf --case bird --is_continue --render_at_pose_path D:/gitwork/genshinnerf/dynamic_test/train_dynamic_setting.json
python exp_runner.py --mode validate_mesh --conf ./confs/wmask_blender_bunny.conf --case bunny2 --is_continue
python exp_runner.py --mode render_rtkm --conf ./confs/wmask_blender_bunny.conf --case bunny_original --is_continue
python exp_runner.py --mode validate_image --conf ./confs/thin_structure_white_bkgd.conf --case soap2_merge --is_continue --gpu 5
python exp_runner.py --mode validate_image --conf ./confs/thin_structure.conf --case scene1 --is_continue --gpu 4
python exp_runner.py --mode render_rtkm --conf ./confs/thin_structure_white_bkgd.conf --case soap2_merge --is_continue --gpu 5
python exp_runner.py --mode train --conf ./confs/wmask_blender_bunny.conf --case bunny2
python exp_runner.py --mode render_rtkm --conf ./confs/thin_structure_white_bkgd.conf --case tree
"""
