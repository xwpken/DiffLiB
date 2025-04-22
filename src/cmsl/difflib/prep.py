# Copyright (C) 2025 DiffLiB authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
"""Preparation."""

import jax.numpy as np

def assign_init_sol(params, mesh_macro, problem_micro, x_vars):
    
    # Get the initial solutions with the matched dimension for JAX-FEM
    
    # For p, c, s, j
    ndof_macro = 4 
    nnode_macro = mesh_macro.num_nodes
    
    ndof_micro = 1 
    nnode_micro = problem_micro.num_nodes
    
    nodes_anode = mesh_macro.nodes_anode
    nodes_cathode = mesh_macro.nodes_cathode
    
    dofs_p_an, dofs_c_an, dofs_s_an = mesh_macro.dofs_an[0:-1]
    dofs_p_ca, dofs_c_ca, dofs_s_ca = mesh_macro.dofs_ca[0:-1]
    dofs_p_se, dofs_c_se = mesh_macro.dofs_se
    
    dofs_p = np.concatenate([dofs_p_an, dofs_p_ca, dofs_p_se])
    dofs_c = np.concatenate([dofs_c_an, dofs_c_ca, dofs_c_se])
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        dofs_s_acc, dofs_s_ccc = mesh_macro.dofs_cc
        dofs_s_a = np.concatenate([dofs_s_an, dofs_s_acc])
        dofs_s_c = np.concatenate([dofs_s_ca, dofs_s_ccc])
    else:
        dofs_s_a = dofs_s_an
        dofs_s_c = dofs_s_ca
    
    sol_init_macro = np.zeros((ndof_macro*nnode_macro)) 
    
    # The separator has no pore wall flux j, however, it is still considered here.
    sol_init_micro = np.zeros((nnode_macro, nnode_micro*ndof_micro)) 
    
    # macro solution
    cs0_an, cs0_ca = x_vars[-2:]
    
    phi0_an = params.calcUoc_neg(cs0_an/params.cs_max[0])
    phi0_ca = params.calcUoc_pos(cs0_ca/params.cs_max[1])
    
    sol_init_macro = sol_init_macro.at[dofs_p].set(-1*phi0_an)
    sol_init_macro = sol_init_macro.at[dofs_c].set(params.cl0/params.cl_ref)
    sol_init_macro = sol_init_macro.at[dofs_s_a].set(0.)
    sol_init_macro = sol_init_macro.at[dofs_s_c].set(phi0_ca - phi0_an)
    
    # micro solution
    sol_init_micro = sol_init_micro.at[nodes_anode,:].set(cs0_an)
    sol_init_micro = sol_init_micro.at[nodes_cathode,:].set(cs0_ca)
    
    sol_init_macro = np.array(sol_init_macro)
    sol_init_micro = np.array(sol_init_micro)
        
    return sol_init_macro, sol_init_micro    

