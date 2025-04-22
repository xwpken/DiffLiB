"""Benchmark problem with the ‘Marquis2019’ parameter set from PyBaMM"""

import numpy as onp
import jax.numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family":'Helvetica',
    "font.size": 25,
    "lines.linewidth": 3
})

from cmsl.difflib.para import ParameterSets
from cmsl.difflib.mesh import generate_mesh
from cmsl.difflib.models import rectangle_cube
from cmsl.difflib.wrap import fast_wrapper

# Mesh
mesh_infos = {'z_an':100., 'n_an':20,
              'z_se':25.,  'n_se':20,
              'z_ca':100., 'n_ca':20,
              'y':10.,     'n_y':2,
              'r':1.,      'n_r':10}
mesh = generate_mesh(mesh_infos)

# forward prediction wrapper
jax_wrapper = fast_wrapper(ParameterSets, mesh, rectangle_cube)

if __name__ == "__main__":
    
    # Current loadings
    t_eval = (0,3600)
    dt = 5.
    c_rate = 1.
    
    # DiffLiB
    fwd_pred, paramsets = jax_wrapper(t_eval, dt, c_rate=1.)
    theta = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                      paramsets.ks[0], paramsets.ks[1],
                      paramsets.tp,
                      paramsets.ds_an, paramsets.ds_ca,
                      paramsets.cs0_an, paramsets.cs0_ca])
    sol_bs_jax, sol_macro_time, sol_micro_time, time_cost = fwd_pred(theta)

    # PyBaMM
    sol_bs_pbm = onp.load('./input/pybamm_data.npy')
    
    # Postprocessing
    plt.figure(figsize=(10, 8))
    x = dt * np.arange(1, len(sol_bs_jax))
    plt.plot(x, sol_bs_jax[1:], color='b', label='DiffLiB')
    plt.plot(x, sol_bs_pbm[1:], color='r', ls='--', label='PyBaMM')
    plt.legend()
    plt.xticks([0,1200,2400,3600])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Terminal voltage (V)')
    plt.savefig('./output/voltage.jpg', dpi=300, format="jpg",bbox_inches='tight')