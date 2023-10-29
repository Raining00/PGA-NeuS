import sys
sys.path.append('../')

import os
from pathlib import Path
import numpy as np
from PIL import Image
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from argparse import ArgumentParser
from glob import glob

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.hex_mesh import hex2obj, hex2obj_with_textures
from py_diff_pd.common.tet_mesh import tet2obj, tet2obj_with_textures
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.tet_mesh import generate_tet_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d,TetMesh3d

import multiprocessing as mp
from multiprocessing import Pool

from tqdm import trange

def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"The path '{path}' has been created.")
    else:
        print(f"The path '{path}' already exists.")

def generate_mesh(args):
    arg = args[0]
    i = args[1]
    bin_file_name = args[2]
    output_path = args[3]
    if arg.input_mesh_type == 'tet':
        mesh = TetMesh3d()
        mesh.Initialize(str(bin_file_name))
        mesh.Scale(arg.mesh_scale)
        output_mesh_name = output_path / '{:04d}.obj'.format(i)
        tet2obj(mesh, obj_file_name=output_mesh_name)
    if arg.input_mesh_type == 'hex':
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        mesh.Scale(arg.mesh_scale)
        output_mesh_name = output_path / '{:04d}.obj'.format(i)
        hex2obj(mesh, obj_file_name=output_mesh_name)

# def mesh_generate(args, verbose=False):
#     assert args.input_file_path, 'Provide path to bin file'
#     assert args.output_mesh_path, 'Provide path of output meshes'
#     assert args.input_mesh_type, 'Provide input mesh type(tet or hex)'

#     bin_file_list = sorted(glob(os.path.join(args.input_file_path, '*.bin')))
#     check_and_create_path(args.output_mesh_path)
#     output_path = Path(args.output_mesh_path)
#     print_info('totle files:', bin_file_list.__len__())
#     print_info('mesh_scale', args.mesh_scale)
#     pbar = trange(bin_file_list.__len__(), desc='mesh generate')
#     for i in pbar:
#         bin_file_name = bin_file_list[i]
#         # use mp to accelerate
#         generate_mesh(args, i, bin_file_name, output_path)
    
#     print_info('mesh generate done!')

def mesh_generate(args, verbose=False):
    assert args.input_file_path, 'Provide path to bin file'
    assert args.output_mesh_path, 'Provide path of output meshes'
    assert args.input_mesh_type, 'Provide input mesh type(tet or hex)'

    bin_file_list = sorted(glob(os.path.join(args.input_file_path, '*.bin')))
    check_and_create_path(args.output_mesh_path)
    output_path = Path(args.output_mesh_path)
    print_info('totle files:', len(bin_file_list))
    print_info('mesh_scale', args.mesh_scale)

    # 使用 Pool 对象并行处理。
    with Pool(processes=mp.cpu_count()) as pool:
        tasks = [(args, i, bin_file_name, output_path) for i, bin_file_name in enumerate(bin_file_list)]
        pbar = trange(len(bin_file_list), desc='mesh generate')
        for _ in pool.imap_unordered(generate_mesh, tasks):
            pbar.update()
    
    print_info('mesh generate done!')

if __name__ == '__main__':
    verbose = True
    parser = ArgumentParser()
    parser.add_argument('--input_file_path', type=str, default=None)
    parser.add_argument('--output_mesh_path',type=str, default=None)
    parser.add_argument('--input_mesh_type', type=str, default='tet', help='tet, hex')
    parser.add_argument('--mesh_scale', type=float, default=1.0)
    args = parser.parse_args()

    mesh_generate(args, verbose)


# python bin_to_mesh.py --input_file_path nerf_sdf_3d/groundtruth --output_mesh_path /mnt/p/pd_eigen_mesh
# python bin_to_mesh.py --input_file_path nerf_sdf_3d/groundtruth/ --output_mesh_path nerf_sdf_3d/mesh