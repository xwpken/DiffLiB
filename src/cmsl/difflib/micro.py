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
"""Microscopic problems."""

from dataclasses import dataclass

import functools

import jax
import jax.numpy as np

import basix
import scipy
import numpy as onp

from .core import pre_BV_fns

@dataclass
class P2D_micro:
    '''
    to store micro problem-relateed variables
    '''
    dt:float # time step size


def pre_micro_problem(mesh_micro, params, dt, use_diff=False, gauss_jax=True):
    '''
    This function initializes the mciro problem
    '''
    
    def basis_interval():
        '''
        For 1D element
        '''
        # Lagrange
        element_family = basix.ElementFamily.P
        basix_ele = basix.CellType.interval
        degree = 1
        gauss_order = 2
        
        # Reference domain
        # Following codes mainly come from jax_fem.basis
        quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
        element = basix.create_element(element_family, basix_ele, degree)
        vals_and_grads = element.tabulate(1, quad_points)[:, :, :, :]
        shape_values = vals_and_grads[0, :, :, 0]
        shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
        
        return shape_values, shape_grads_ref, weights, quad_points
    
    def get_shape_grads(problem):
        '''
        Get FEM-related variables in physical domain
        
        Follwing codes are mainly copied from jax_fem.fe.get_shape_grads
        '''
        physical_coos = onp.take(problem.points, problem.cells,
                                 axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   problem.shape_grads_ref[None, :, :, None, :],
                                   axis=2,
                                   keepdims=True)
        jacobian_det = onp.linalg.det(
                            jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (problem.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * problem.quad_weights[None, :]
        
        # (num_cells, num_quads, num_nodes, 1, dim)
        v_grads_JxW = shape_grads_physical[:, :, :, None, :] * JxW[:, :, None, None, None]
        
        return shape_grads_physical, JxW, v_grads_JxW
    
    # variable sets
    problem = P2D_micro(dt)
    
    # Mesh
    problem.points = mesh_micro.points
    problem.cells = mesh_micro.cells
    problem.num_nodes = len(problem.points)
    problem.ndofs = len(problem.points)
    problem.right = onp.max(problem.points) # x = 1.
    problem.bound_right = mesh_micro.bound_right.item()
    
    # 1D FE
    problem.shape_vals, problem.shape_grads_ref, problem.quad_weights, problem.quad_points = basis_interval()
    problem.shape_grads, problem.JxW , problem.v_grads_JxW = get_shape_grads(problem)
    
    if gauss_jax:
        # Notes: the quadrature interval in basix is [0,1]
        # (num_quads, )
        quad_coords_ref = problem.quad_points.reshape(-1)
        cells_coords = onp.take(problem.points, problem.cells)
        quad_coords_physical = ((cells_coords[:,1] - cells_coords[:,0])[:,None]* 
                                quad_coords_ref[None,:] + cells_coords[:,0][:,None])
        # (num_cells, num_quads)
        problem.quad_coords_physical = quad_coords_physical
        
    else:
        # To be consistent with MATLAB. However, this seems wrong!
        quad_coords_ref = onp.array([-1/np.sqrt(3),1/np.sqrt(3)])
        # (num_cells, num_quads) --> (1, num_quads)
        problem.quad_coords_physical = quad_coords_ref[None,:]
        
    # Index
    problem.I = np.repeat(problem.cells,repeats=2,axis=1).reshape(-1)
    problem.J = np.repeat(problem.cells,repeats=2,axis=0).reshape(-1)
    
    micro_fns_list = pre_micro_fns(problem, params, use_diff)
    
    problem.flux_res_fns, problem.solver_csI, problem.solver_csII = micro_fns_list
    
    return problem


def pre_micro_res_fns(problem, mode = 'full'):
    
    # (num_quads, num_nodes)
    shape_vals = problem.shape_vals
    # (num_cells, num_quads, num_nodes, dim)
    shape_grads = problem.shape_grads
    # (num_cells, num_quads)
    JxW = problem.JxW
    # (num_cells, num_quads, num_nodes, 1, dim)
    v_grads_JxW = problem.v_grads_JxW
    # (num_cells, num_quads)
    x = problem.quad_coords_physical
    
    if mode == 'full':
        
        def compute_micro_res(sol_micro, sol_micro_old, ds, cs_dt_coeff):
            
            '''
            Compute the full residual for micro problems
            
            '''
            # (num_cells, num_nodes, vec)
            sol_micro = np.take(sol_micro, problem.cells)[:,:,None]
            # (num_cells, num_nodes, vec)
            sol_micro_old = np.take(sol_micro_old, problem.cells)[:,:,None]
            
            # Residual
            # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
            sol_micro_crt = np.sum(sol_micro[:,None,:,:] * shape_vals[None,:,:,None], axis=2)
            sol_micro_old = np.sum(sol_micro_old[:,None,:,:] * shape_vals[None,:,:,None], axis=2)
            d_cs = sol_micro_crt - sol_micro_old
            
            # (num_cells, num_quads, 1, 1) * (num_cells, num_quads, 1, vec) *
            # (1, num_quads, num_nodes, 1) * (num_cells, num_quads, 1, 1) --> (num_cells,num_nodes,vec)
            res = 4 * np.pi * cs_dt_coeff * np.sum(x[:,:,None,None]**2 * d_cs[:,:,None,:] 
                                                * shape_vals[None,:,:,None] * JxW[:,:,None,None],axis=1)
            # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, vec, dim)
            cs_grads = np.sum(sol_micro[:, None, :, :, None] * shape_grads[:, :, :, None, :],axis=2)
            # (num_cells, num_quads, 1, 1, 1) * (num_cells, num_quads, 1, vec, dim) * (num_cells, num_quads, num_nodes, 1, dim) -> 
            res = res + 4 * np.pi * ds * np.sum(x[:,:,None,None,None]**2 * cs_grads[:,:,None,:,:] * v_grads_JxW, axis=(1,-1))
            
            weak_form = np.zeros((len(problem.points)))
            weak_form = weak_form.at[problem.cells.reshape(-1)].add(res.reshape(-1))
            
            return weak_form
        
    elif mode == 'fast':
        
        def compute_micro_res(sol_micro, ds):
            
            '''
            Compute the non-zero residual for micro problems
            
            '''
            # (num_cells, num_nodes, vec)
            sol_micro = np.take(sol_micro, problem.cells)[:,:,None]

            # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, vec, dim)
            cs_grads = np.sum(sol_micro[:, None, :, :, None] * shape_grads[:, :, :, None, :],axis=2)
            # (num_cells, num_quads, 1, 1, 1) * (num_cells, num_quads, 1, vec, dim) * (num_cells, num_quads, num_nodes, 1, dim) -> 
            res = 4 * np.pi * ds * np.sum(x[:,:,None,None,None]**2 * cs_grads[:,:,None,:,:] * v_grads_JxW, axis=(1,-1))
            
            weak_form = np.zeros((len(problem.points)))
            weak_form = weak_form.at[problem.cells.reshape(-1)].add(res.reshape(-1))
            
            return weak_form
        
    return compute_micro_res


def explicit_jac(problem, cs_dt_coeff, ds):
    '''
    Compute micro jac explicitly
    
    '''

    # (num_quads, num_nodes)
    shape_vals = problem.shape_vals
    # (num_cells, num_quads, num_nodes, dim)
    shape_grads = problem.shape_grads
    # (num_cells, num_quads)
    JxW = problem.JxW
    # (num_cells, num_quads, num_nodes, 1, dim)
    # v_grads_JxW = problem.v_grads_JxW
    
    x = problem.quad_coords_physical
    
    # (num_cells, num_quads, 1, 1) * 
    # (1, num_quads, num_nodes, 1) @ (1, num_quads, 1, num_nodes) *
    # (num_cells, num_quads, 1, 1)
    # +
    # (num_cells, num_quads, 1, 1) * 
    # (num_cells, num_quads, num_nodes, 1) @ (num_cells, num_quads, 1, num_nodes) *
    # (num_cells, num_quads, 1, 1)
    
    V = (cs_dt_coeff * 4 * np.pi * x[:,:,None,None]**2 *
            shape_vals[None,:,:,None] @ shape_vals[None,:,None,:] *
            JxW[:,:,None,None]
            +
            ds * 4 * np.pi * x[:,:,None,None]**2 *
            shape_grads[:,:,:,None,0] @ shape_grads[:,:,None,:,0] * JxW[:,:,None,None]
            ) 
    
    # (num_cells, num_nodes, num_nodes)
    cells_jac = np.sum(V,axis=1)
    
    return cells_jac


def compute_micro_jac(problem, params, inv=False):
    
    '''
    Compute micro jac with initial micro parameters
    
    '''
    
    # anode
    cs_an_dt_coeff = 1 / params.dt * params.l_ref_an**2 / params.ds_an
    ds_an = params.ds_an / params.ds_an
    V_an = explicit_jac(problem, cs_an_dt_coeff, ds_an).reshape(-1)
    
    # cathode
    cs_ca_dt_coeff = 1 / params.dt * params.l_ref_ca**2 / params.ds_ca
    ds_ca = params.ds_ca / params.ds_ca
    V_ca = explicit_jac(problem, cs_ca_dt_coeff, ds_ca).reshape(-1)
    
    I, J = problem.I, problem.J

    A_an = scipy.sparse.csr_matrix((V_an, (I, J)), 
                                   shape=(problem.ndofs, problem.ndofs))
    A_ca = scipy.sparse.csr_matrix((V_ca, (I, J)), 
                                   shape=(problem.ndofs, problem.ndofs))
    # inverse matrix (dense!)
    if inv:
        A_an = np.array(onp.linalg.inv(A_an.toarray()), dtype=np.float64)
        A_ca = np.array(onp.linalg.inv(A_ca.toarray()), dtype=np.float64)
    
    return A_an, A_ca


def pre_micro_fns(problem, params, use_diff):
    
    '''
    The similar micro problem treatment as the MATLAB code, 
    
    also some algorithm from the fast P2D solver (arXiv)
    
    '''
    
    Bulter_Volmer_fns = pre_BV_fns(params)
    
    def value_and_jacfwd(f, x):
        pushfwd = functools.partial(jax.jvp, f, (x, ))
        basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, x.size)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
        return y, jac
    
    I, J = problem.I, problem.J

    if use_diff:
        
        # differentiable but maybe slow when the micro mesh is fine
        
        compute_micro_res = pre_micro_res_fns(problem, mode = 'full')
        
        def solver(sol_micro_old, node_flux, ds, cs_dt_coeff, q_bc, lag_an, lag_ca):
            
            # previous time step solutions as initial solutions
            sol_micro = sol_micro_old
            
            # def newton_update_helper(sol_micro):
            #     res = compute_res_micro(sol_micro, sol_micro_old, ds, cs_dt_coeff)
            #     return res
            
            # res = newton_update_helper(sol_micro)
            
            # res, jac = value_and_jacfwd(newton_update_helper, sol_micro.reshape(-1))
            
            res = compute_micro_res(sol_micro, sol_micro_old, ds, cs_dt_coeff)
            
            jac = np.zeros((problem.ndofs, problem.ndofs))
            jac = jac.at[I,J].add(explicit_jac(cs_dt_coeff, ds).reshape(-1))
            
            # Neumann BCs
            res = res.at[problem.bound_right].add(q_bc * node_flux)
            
            sol_micro_inc = np.linalg.solve(jac, -res.reshape(-1,1))
            
            sol_micro = sol_micro + sol_micro_inc
            
            return sol_micro
    
    else:
        
        # not differentiable for micro parameters but could be faster
        
        compute_micro_res = pre_micro_res_fns(problem, mode = 'fast')
        
        A_inv_an, A_inv_ca = compute_micro_jac(problem, params, inv=True)
        
        bound_right = problem.bound_right
        
        I_gamma = onp.zeros(problem.num_nodes)
        I_gamma[bound_right] = 1.
        
        q_bc_an = 4 * np.pi * params.r_an**2 / params.ds_an / params.l_ref_an
        q_bc_ca = 4 * np.pi * params.r_ca**2 / params.ds_ca / params.l_ref_ca
        
        A_inv_gamma_an = A_inv_an @ I_gamma
        A_inv_gamma_ca = A_inv_ca @ I_gamma
        
        def solver_csI(sol_cs_old, lag_an, lag_ca):
            
            ds = (lag_an * params.ds_an / params.ds_an 
                + lag_ca * params.ds_ca / params.ds_ca)
            
            A_inv = lag_an * A_inv_an + lag_ca * A_inv_ca
            
            res = compute_micro_res(sol_cs_old, ds)
            
            return A_inv @ -res
            
        
        def solver_csII(j, lag_an, lag_ca):
            A_inv_gamma = lag_an * A_inv_gamma_an + lag_ca * A_inv_gamma_ca
            q_bc = lag_an * q_bc_an + lag_ca * q_bc_ca            
            return A_inv_gamma * -1 * q_bc * j
        
        
        def flux_res(node_p, node_c, node_s, node_j, sol_csI, ks, lag_an, lag_ca):
            # css
            css = sol_csI + solver_csII(node_j, lag_an, lag_ca)[bound_right]
            # BV equation
            j = Bulter_Volmer_fns(node_p, node_c, node_s, css, ks, lag_an, lag_ca)
            # residual of j
            res = node_j - j
            return res
            
    return [jax.jit(jax.vmap(flux_res)), jax.jit(jax.vmap(solver_csI)), jax.jit(jax.vmap(solver_csII))]

