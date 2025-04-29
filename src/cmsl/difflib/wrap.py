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
"""Differentiable problem wrapper."""

import time

import jax
import jax.numpy as np

from jax_fem import logger
from jax_fem.solver import ad_wrapper

from .micro import pre_micro_problem


def fast_wrapper(ParameterSets, mesh, model,
                 solver_options=None, additional_info={}):
    
    '''
    Return the function 'jax_wrapper' which gets time and c-rate params, and returns 
    
    the forward prediction functon 'fwd_pred_seq'
    
    '''
    
    mesh_macro, mesh_micro = mesh
    
    obj_dof = mesh_macro.dofs[2][mesh_macro.terminal]
    
    if solver_options is not None:
        fwd_options, inv_options = solver_options
    else:
        fwd_options = {'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'lu'}, 'tol':1e-8, 'rel_tol':1e-8}
        inv_options = {'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'lu'}, 'tol':1e-8, 'rel_tol':1e-8}
    
    # initial solution & guess
    if 'assign_init_sol' in additional_info:
        assign_init_sol = additional_info['assign_init_sol']
    else:
        from .prep import assign_init_sol
    
    # fwd problem wrapper
    def jax_wrapper(t_eval, dt, c_rate, use_diff=False, gauss_jax=True):
        
        steps = int((t_eval[1]-t_eval[0])/dt+1)
        
        # parameters
        paramsets = ParameterSets(dt, c_rate)

        # problem definitions
        problem_micro = pre_micro_problem(mesh_micro, paramsets, dt,
                                          use_diff, gauss_jax)
        
        problem_macro = model(paramsets, mesh_macro, problem_micro)
        
        fwd_pred = ad_wrapper(problem_macro, 
                              solver_options=fwd_options, 
                              adjoint_solver_options=inv_options)
        
        # Some auxiliary variables and functions
        solver_csI, solver_csII = problem_micro.solver_csI, problem_micro.solver_csII
        bound_right = problem_micro.bound_right
        
        # nodes_tag, _, _ = mesh_macro.nodes_vars
        # nodes_lag = problem_macro.get_lag_coeffs(nodes_tag)
        
        _, nodes_micro, nodes_micro_tag = mesh_macro.nodes_vars
        nodes_lag = problem_macro.get_lag_coeffs(nodes_micro_tag)
        assemble_micro = jax.jit(lambda full, part: full.at[nodes_micro,:].set(part))
        
        _, dofs_c, _, dofs_j = mesh_macro.dofs
        
        def fwd_pred_seq(x_vars):
            
            x_vars = np.insert(x_vars, 6, np.array([paramsets.ds_an, paramsets.ds_ca]))
            
            sol_macro_old, sol_micro_old = assign_init_sol(paramsets, mesh_macro, problem_micro, x_vars)
            
            t0 = time.time() 
        
            sol_macro_time = [sol_macro_old]
            sol_micro_time = [sol_micro_old]
            
            sol_bs_jax = [sol_macro_old[obj_dof]]
            
            for itime in range(1, steps):
                
                logger.info(f"Step {itime} in {steps-1}")
                
                macro_t0 = time.time()
                
                try:
                    
                    sol_c_old = sol_macro_old[dofs_c] # c_crt - c_old = 0
                    
                    # sol_csI = sol_micro_old + solver_csI(sol_micro_old, nodes_lag[0,:], nodes_lag[1,:])
                    
                    sol_csI0 = sol_micro_old[nodes_micro,:] + solver_csI(sol_micro_old[nodes_micro,:], nodes_lag[0,:], nodes_lag[1,:])
                    sol_csI = assemble_micro(sol_micro_old, sol_csI0)
                    
                    fwd_options['initial_guess'] = sol_macro_old
                    
                    sol_list = fwd_pred([x_vars[0:-2], sol_c_old, sol_csI[:,bound_right]])
                    
                    sol_macro_old, _ = jax.flatten_util.ravel_pytree(sol_list)
                    
                    sol_bs_jax.append(sol_macro_old[obj_dof])
                    
                    micro_t0 = time.time()
                    
                    # sol_micro_old = sol_csI + solver_csII(sol_list[-1], nodes_lag[0,:], nodes_lag[1,:])
                    
                    sol_micro_j = sol_list[-1][nodes_micro] * paramsets.j_ref
                    
                    sol_cs = sol_csI0 + solver_csII(sol_micro_j, nodes_lag[0,:], nodes_lag[1,:])
                    
                    sol_micro_old = assemble_micro(sol_micro_old, sol_cs)
                    
                    t_micro = time.time() - micro_t0
                    
                    logger.info(f"Time cost: {time.time()-macro_t0} with {t_micro} [s]")

                    sol_macro_time.append(sol_macro_old)
                    sol_micro_time.append(sol_micro_old)
                    
                except:
                    
                    break

            t1 = time.time()
            
            total_time = t1 - t0
            
            sol_bs_jax = np.array(sol_bs_jax).reshape(-1)
            
            sol_macro_time = np.array(sol_macro_time).T
            sol_micro_time = np.transpose(np.array(sol_micro_time), axes=(1,2,0))
            
            # sol_macro_time = 0.
            # sol_micro_time = 0.
            
            return sol_bs_jax, sol_macro_time, sol_micro_time, total_time
        
        return fwd_pred_seq, paramsets
    
    return jax_wrapper

