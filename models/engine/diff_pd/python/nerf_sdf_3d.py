import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.nerf_sdf_env_3d import NeRFEnv3d
from tqdm import trange

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('nerf_sdf_3d')
    refinement = 8
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    env = NeRFEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
          'poissons_ratio': poissons_ratio,
          'refinement': refinement,
          'mesh_type':'tet',
          'obj_file_name':  'mc.ply'})
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 16
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('pd_eigen', 'newton_cholesky')
    opts = (pd_opt, newton_opt)

    dt = 1e-3
    frame_num = 30
    substep = 10

    # Initial state.
    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    z_offset = 0.2
    q0[2::3] += z_offset
    v0 = np.zeros(dofs)
    init_com_v = np.array([0,0,0.0],dtype=np.float64)
    init_v = (v0.reshape((-1, 3)) + init_com_v).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num * substep)]
    f0 = [np.zeros(dofs) for _ in range(frame_num * substep)]
    
    env.initialize(options={'frame_num':frame_num, 'substep':substep}, q0=q0, v0=v0, f_ext=f0, act=a0)
    # Generate groundtruth motion.
    # env.simulate(dt, frame_num, methods[0], opts[0], q0, init_v, a0, f0, require_grad=False, vis_folder='groundtruth', render_frame_skip=10)
    pbar = trange(frame_num + 1, desc='groundtruth simulation')
    for i in pbar:
        if i ==0:
            continue
        info = env.forward(dt=dt, method=methods[0], opt=opts[0], vis_folder='groundtruth', frame_num=i)
        # if isinstance(info, str):
        #     print_info(info)
        # elif isinstance(info, dict):
        #     print_info(info['forward_time'])
    print_ok('Done with groundtruth simulation.')

    pbar = trange(frame_num + 1, desc='backward simulation')
    for i in pbar:
        f = frame_num - i
        grad, info = env.backward(dt=dt, frame_num=f, method=methods[0], backward_opt=opts[0], sim_info={'grad_v':np.ones(dofs), 'grad_q':np.ones(dofs)})
    print_ok('Done with backward simulation.')