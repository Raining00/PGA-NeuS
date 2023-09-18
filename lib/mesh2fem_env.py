import time
from pathlib import Path

import numpy as np
import os

from physical_engine.diff_pd.python.py_diff_pd.env.env_base import EnvBase
from physical_engine.diff_pd.python.py_diff_pd.common.common import create_folder, ndarray, print_info
from physical_engine.diff_pd.python.py_diff_pd.common.tet_mesh import tetrahedralize, read_tetgen_file, generate_tet_mesh, tet2obj
from physical_engine.diff_pd.python.py_diff_pd.common.tri_mesh import generate_tri_mesh
from physical_engine.diff_pd.python.py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from physical_engine.diff_pd.python.py_diff_pd.common.project_path import root_path
from physical_engine.diff_pd.python.py_diff_pd.common.display import export_gif
from physical_engine.diff_pd.python.py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from physical_engine.diff_pd.python.py_diff_pd.common.renderer import PbrtRenderer
from physical_engine.diff_pd.python.py_diff_pd.common.project_path import root_path, print_error

class NeRFEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = 4e7
        poissons_ratio = 0.45
        state_force_parameters = options['state_force_parameters']
        center = ndarray(options['center'])
        start_degree = float(options['start_degree'])
        end_degree = float(options['end_degree'])
        start_rad = np.deg2rad(start_degree)
        end_rad = np.deg2rad(end_degree)
        radius = float(options['radius'])
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        
        # Mesh parameters.
        obj_file_name = ''
        tmp_bin_file_name = '.tmp.bin'
        if 'obj_name' in options:
            obj_file_name = Path(root_path) / 'asset' / 'mesh' / options['obj_name']
        else:
            print_error('obj_name not found in options')
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False, options={ 'minratio': 1.1 })
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # External force.
        g = state_force_parameters[:3]
        deformable.AddStateForce('gravity', g)

        # Contact.
        kn, kf, mu = state_force_parameters[3:]

        # TODO: SDF Penalty-based contact force


        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Initial conditions.
        dofs = deformable.dofs()
        q0 = ndarray(mesh.py_vertices()).reshape((-1, 3))
        q0 = q0.ravel()
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__spp = int(options['spp']) if 'spp' in options else 4

    
    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total