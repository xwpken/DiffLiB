"""Benchmark problem with the ‘Marquis2019’ parameter set from PyBaMM"""

import os
import pickle

import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family":'serif',
    "font.size": 25,
    "lines.linewidth": 2.5
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
    dt = 5.
    c_rates = [0.2, 0.5, 1.0, 1.5, 2.]
    ts_max = [18475, 7330, 3620, 2385, 1770]
    
    # DiffLiB
    dfb_voltages = []
    for c_rate, t_max in zip(c_rates, ts_max):
        t_eval = (0,t_max)
        fwd_pred, paramsets = jax_wrapper(t_eval, dt, c_rate=c_rate)
        theta = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                          paramsets.ks[0], paramsets.ks[1],
                          paramsets.tp,
                          paramsets.cs0_an, paramsets.cs0_ca])
        sol_bs, sol_macro_time, sol_micro_time, time_cost = fwd_pred(theta)
        dfb_voltages.append(onp.array(sol_bs))
            
    # PyBaMM
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'input', 'pybamm_data.pkl')
    with open(input_path, 'rb') as f:
        pbm_voltages = pickle.load(f)
    
    # Postprocessing
    output_dir = os.path.join(script_dir, 'output') 
    count = 0
    plt.figure(figsize=(10, 8))
    for c_rate, dfb_data, pbm_data in zip(c_rates, dfb_voltages, pbm_voltages):
        x = dt * onp.arange(1, len(pbm_data))/int(3600/c_rate) * 24
        plt.plot(x, dfb_data[1:], label=fr'${c_rate}\,\rm C$')
        plt.scatter(x, pbm_data[1:], s=60, color=f'C{count}', facecolor='None', label=None)
        count += 1
    plt.xlim([-1, 25])
    plt.xticks([0,6,12,18,24])
    plt.ylim([3.1, 3.9])
    plt.yticks([3.1,3.3,3.5,3.7,3.9])
    plt.xlabel(r'Discharge capacity $(\rm Ah/m^2)$')
    plt.ylabel(r'Terminal voltage $(\rm V)$')
    plt.legend(frameon=False, loc='lower left')
    output_path = os.path.join(output_dir, 'voltage.png')
    plt.savefig(output_path, dpi=300, format="png",bbox_inches='tight')
    