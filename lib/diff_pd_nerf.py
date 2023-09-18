import sys

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from physical_engine.diff_pd.python.lib.common import ndarray, create_folder
from physical_engine.diff_pd.python.lib.common import print_info, print_ok, print_error
from physical_engine.diff_pd.python.lib.grad_check import check_gradients
from mesh2fem_env import NeRFEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    folder = Path('diff_pd_nerf')
    env = NeRFEnv3d(seed, folder, {
        'state_force_parameters': [0, -9.81, 0, 1e5, 0.025, 1e4],
        'slope_degree': 20,
        'initial_height': 1.0 })
    deformable = env.deformable()

    # optimization parameters
    thread_ct = 16
    pd_opt = { 'max_pd_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = {'pd_eigen', }
    opts = (pd_opt, )

    dt = 1e-3
    frame_num = 14

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]