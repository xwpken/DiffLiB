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
"""Implementation of the full-order DFN model."""

from dataclasses import dataclass

import jax
import jax.numpy as np

from jax_fem.problem import Problem

# -------------------- Anode + Separator + Cathode --------------------
@dataclass
class P2D(Problem):
    
    def custom_init(self, params, cells_vars, flux_res_fns):
        
        self.fe_p = self.fes[0]
        self.fe_c = self.fes[1]
        self.fe_s = self.fes[2]
        self.fe_j = self.fes[3]
        
        self.num_cell_nodes = self.fe_p.num_nodes
        
        self.params = params
        self.cells_vars = cells_vars
        self.flux_res_fns = flux_res_fns
        
        # electrolyte conductivity
        if hasattr(params,'calcKappa'):
            self.calcKappa = params.calcKappa
        else:
            kappa = np.array([params.kappa])
            self.calcKappa = lambda c_e: kappa
        
        # electrolyte diffusivity
        if hasattr(params,'calcDf'):
            self.calcDf = params.calcDf
        else:
            df = np.array([params.df])
            self.calcDf = lambda ce: df
    
    
    def get_lag_coeffs(self, tag):
        
        # element/node domain (for lag interpolation)
        
        lag_an = 1./2 * (tag-2)*(tag-3)      # anode - 1
        lag_ca =  -1. * (tag-1)*(tag-3)      # cathode - 2
        lag_se = 1./2 * (tag-1)*(tag-2)      # seperator - 3
        
        return np.array([lag_an, lag_ca, lag_se])
        
    
    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, 
                             cell_x_vars, cell_c_quad_old, cell_sol_csI_bound, cell_tag, cell_nodes_tag, cell_nodes_sum):
            
            # ---- Split the input variables ----
            
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            # sol of the cell
            # (num_nodes, vec) -> (4, 1)
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j = cell_sol_list
            
            # shape function gradients of the cell
            # (num_quads, num_nodes, dim) -> (4, 4, 2)
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p, cell_shape_grads_c, cell_shape_grads_s, _ = cell_shape_grads_list
            
            # grad(v)*JxW of the cell
            # (num_quads, num_nodes, 1, dim) -> (4, 4, 1, 2)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p, cell_v_grads_JxW_c, cell_v_grads_JxW_s, _ = cell_v_grads_JxW_list
            
            # JxW of the cell -> (num_quads,) -> (4,)
            cell_JxW_p, cell_JxW_c, cell_JxW_s, _ = cell_JxW[0], cell_JxW[1], cell_JxW[2], cell_JxW[3]
            
            
            # scaling
            cell_sol_c = cell_sol_c * self.params.cl_ref   
            cell_c_quad_old = cell_c_quad_old * self.params.cl_ref
            cell_sol_j = cell_sol_j * self.params.j_ref
            
            
            # design variables
            alpha_an, alpha_ca, alpha_se, ks_an, ks_ca, tp, ds_an, ds_ca = cell_x_vars
            
            # lagrangian coefficients
            lag = self.get_lag_coeffs(cell_tag)
            lag_an, lag_ca, lag_se = lag[0], lag[1], lag[2]
            
            # Process variabls
            alpha = (lag_an * alpha_an +
                     lag_ca * alpha_ca +
                     lag_se * alpha_se)
            
            epsl = (lag_an * self.params.epsl_an + 
                    lag_ca * self.params.epsl_ca + 
                    lag_se * self.params.epsl_se)
            
            epss = (lag_an * self.params.epss_an +
                    lag_ca * self.params.epss_ca)
            
            r = (lag_an * self.params.r_an + 
                 lag_ca * self.params.r_ca + 
                 lag_se * 1.)
            
            sour_c = (self.params.l_ref)**2 / self.params.df_ref * (1 - tp)
            
            sour_s = (lag_an * self.params.sour_s_an + 
                      lag_ca * self.params.sour_s_ca)
            
            svr = 3 * epss / r
            
            ratio_epsl = (epsl)**(alpha)
            ratio_epss = (epss)**(alpha)
            
            # ---- Residual of potential in electrolyte  ----
            
            # R_p = (ka_eff * inner(grad(p), grad(v_p) - kad_eff * inner(grad(ln(c))), grad(v_p) - a*F*j*v_p)*dx
            
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec) -> (num_quads,)
            c = np.sum(cell_sol_c[None,:,:] * self.fe_c.shape_vals[:,:,None], axis=1)[:,0]
            # **** Notes: following params are location dependent (quadrature point location dependent) ****
            # (num_quads,)
            kappa = self.calcKappa(c)
            ka_eff = kappa / self.params.ka_ref * ratio_epsl
            # **** Notes: kad_eff seems not to be consistent with the formulations in the PDF ****
            kad_eff = 2 * ka_eff * self.params.R * self.params.T / self.params.F *(1 - tp)
            
            
            # Handles the term `ka_eff * inner(grad(p), grad(v_p)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            p_grads = cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :]
            p_grads = np.sum(p_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v1 = np.sum(ka_eff[:,None,None,None] * p_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))

            
            # Handles the term `kad_eff * inner(grad(ln(c))), grad(v_p)*dx`
            # **** Notes: grad(ln(c)) can be written as d(ln(c))/d(c) * grad(c) = 1/c * grad(c) ****
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            c_grads = cell_sol_c[None, :, :, None] * cell_shape_grads_c[:, :, None, :]
            c_grads = np.sum(c_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v2 = np.sum(kad_eff[:,None,None,None] * 1./c[:,None,None,None] * c_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))
            
            
            # Handles the term `a*F*j*v_p*dx` (sour_p * a *j * v_p *dx)
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            j = np.sum(cell_sol_j[None,:,:] * self.fe_j.shape_vals[:,:,None], axis=1)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            Rp_v3 = self.params.sour_p * svr * np.sum(j[:,None,:] * self.fe_p.shape_vals[:,:,None] * cell_JxW_p[:,None,None], axis=0)
            
            Rp = Rp_v1 - Rp_v2 - Rp_v3
            
            
            # ---- Residual of diffusion in electrolyte  ----
            
            # R_c = (c_crt - c_old)/dt * v_c *dx + df_eff * inner(grad(c),grad(v_c))*dx - (1-t_e)*a*j*v_c*dx
            
            df_eff = self.calcDf(c) * ratio_epsl / self.params.df_ref
            
            # Handles the term ` (c_crt - c_old)/dt * v_c *dx`
            dc = c - cell_c_quad_old
            # (num_quads, 1, 1) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            Rc_v1 = self.params.c_dt_coeff * epsl * np.sum(dc[:,None,None] * self.fe_c.shape_vals[:,:,None]* cell_JxW_c[:,None,None], axis=0)
            
            
            # Handles the term `df_eff * inner(grad(c),grad(v_c))*dx`
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rc_v2 = np.sum(df_eff[:, None, None, None] * c_grads[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))
            
            
            # Handles the term `(1-t_e)*a*j*v_c*dx`
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            
            Rc_v3 = sour_c * svr * np.sum(j[:,None,:] * self.fe_c.shape_vals[:,:,None] * cell_JxW_c[:,None,None], axis=0)
            
            Rc = Rc_v1 + Rc_v2 - Rc_v3
            
            
            # ---- Residual of potential in electrode  ----
            
            # Only in anode and cathode!
            
            # R_s = (sig_eff * grad(s) * grad(v_s) + a*F*j*v_s)*dx
            
            sigma = (lag_an * self.params.sigma_an / self.params.sigma_ref + 
                     lag_ca * self.params.sigma_ca / self.params.sigma_ref)
            
            sigma_eff = sigma * ratio_epss
            
            # Handles the term `sig_eff * grad(s) * grad(v_s)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            s_grads = cell_sol_s[None, :, :, None] * cell_shape_grads_s[:, :, None, :]
            s_grads = np.sum(s_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rs_v1 = sigma_eff * np.sum(s_grads[:, None, :, :] * cell_v_grads_JxW_s, axis=(0, -1))
            
            # Handles the term `a*F*j*v_s*dx`
            Rs_v2 = sour_s * svr * np.sum(j[:,None,:] * self.fe_s.shape_vals[:,:,None] * cell_JxW_s[:,None,None], axis=0)
            
            Rs = Rs_v1 + Rs_v2
            
            
            # ---- Residual of pore wall flux ----
            
            # cell_sol_list [(num_nodes,vec),...]
            
            cell_node_lag = jax.vmap(self.get_lag_coeffs)(cell_nodes_tag)
            
            cell_node_ks = cell_node_lag[:,0] * ks_an + cell_node_lag[:,1] * ks_ca
            
            Rj = self.flux_res_fns(cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j,
                                   cell_sol_csI_bound,
                                   cell_node_ks, 
                                   cell_node_lag[:,0], cell_node_lag[:,1])
            
            Rj = Rj / cell_nodes_sum
            
            weak_form = [Rp, Rc, Rs, Rj] # [(num_nodes, vec), ...]
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel
    
    
    def get_universal_kernels_surface(self):
        '''
        Neumann boundary conditions for phi_s
        '''
        def current(u):
            return np.array([self.params.I_bc * self.params.I_bc_coeff])
        
        def current_neumann(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_face_quads, dim)
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # face_nanson_scale: (num_vars, num_face_quads)
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_s = cell_sol_list[2]
            face_shape_vals = face_shape_vals[:, -self.fe_s.num_nodes:]
            face_nanson_scale = face_nanson_scale[0]
            
            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol_s[None, :, :] * face_shape_vals[:, :, None], axis=1)
            
            u_physics = jax.vmap(current)(u)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val_s = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)
            
            # only for the electrode potential
            val = [val_s*0.,val_s*0.,val_s,val_s*0.]
            
            return jax.flatten_util.ravel_pytree(val)[0]
        
        return [current_neumann]


    def set_params(self, params):
        """
        Input variables to the kernel
        """
        x_vars, sol_c_old, sol_csI_bound = params
        
        cells_tag, cells_nodes_tag, cells_nodes_sum = self.cells_vars
        
        cells_x_vars = np.repeat(x_vars[None,:],axis=0,repeats=self.num_cells)
        
        # (num_cells, num_quads)
        sol_c_old = self.fe_c.convert_from_dof_to_quad(sol_c_old.reshape((-1,1)))[:,:,0]
        # (num_cells, num_nodes, num_micro_nodes)
        sol_csI_bound = sol_csI_bound[self.cells_list[0]]
        
        self.internal_vars = [cells_x_vars, sol_c_old, sol_csI_bound, cells_tag, cells_nodes_tag, cells_nodes_sum]


        
# -------------------- ACC + Anode + Separator + Cathode + CCC --------------------

class P2DCC(P2D):
    
    def get_lag_coeffs(self, tag):
        
        # element/node domain (for lag interpolation)
        lag_an =  1./(1.*2.*3.*4.)*(tag-2)*(tag-3)*(tag-4)*(tag-5)   # anode - 1
        lag_ca = -1./(1.*1.*2.*3.)*(tag-1)*(tag-3)*(tag-4)*(tag-5)   # cathode - 2
        lag_se =  1./(2.*1.*1.*2.)*(tag-1)*(tag-2)*(tag-4)*(tag-5)   # seperator - 3
        lag_acc = -1./(3.*2.*1.*1.)*(tag-1)*(tag-2)*(tag-3)*(tag-5)   # anode current collector - 4
        lag_ccc =  1./(4.*3.*2.*1.)*(tag-1)*(tag-2)*(tag-3)*(tag-4)   # cathode current collector - 5
        
        return np.array([lag_an, lag_ca, lag_se, lag_acc, lag_ccc])
        
    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, 
                             cell_x_vars, cell_c_quad_old, cell_sol_csI_bound, cell_tag, cell_nodes_tag, cell_nodes_sum):
            
            # ---- Split the input variables ----
            
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            # sol of the cell
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            # (num_nodes, vec) -> (4, 1)
            cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j = cell_sol_list
            
            # shape function gradients of the cell
            # (num_quads, num_nodes, dim) -> (4, 4, 2)
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p, cell_shape_grads_c, cell_shape_grads_s, _ = cell_shape_grads_list
            
            # grad(v)*JxW of the cell
            # (num_quads, num_nodes, 1, dim) -> (4, 4, 1, 2)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p, cell_v_grads_JxW_c, cell_v_grads_JxW_s, _ = cell_v_grads_JxW_list
            
            # JxW of the cell -> (num_quads,) -> (4,)
            cell_JxW_p, cell_JxW_c, cell_JxW_s, _ = cell_JxW[0], cell_JxW[1], cell_JxW[2], cell_JxW[3]
            
            
            # scaling
            cell_sol_c = cell_sol_c * self.params.cl_ref   
            cell_c_quad_old = cell_c_quad_old * self.params.cl_ref
            cell_sol_j = cell_sol_j * self.params.j_ref
            
            
            # design variables
            alpha_an, alpha_ca, alpha_se, ks_an, ks_ca, tp, ds_an, ds_ca = cell_x_vars
            
            # lagrangian coefficients
            lag = self.get_lag_coeffs(cell_tag)
            lag_an, lag_ca, lag_se, lag_acc, lag_ccc = lag[0], lag[1], lag[2], lag[3], lag[4]
            
            # Process variabls
            alpha = (lag_an * alpha_an +
                     lag_ca * alpha_ca +
                     lag_se * alpha_se +
                     lag_acc * 1. +
                     lag_ccc * 1.)
            
            epsl = (lag_an * self.params.epsl_an + 
                    lag_ca * self.params.epsl_ca + 
                    lag_se * self.params.epsl_se)
            
            epss = (lag_an * self.params.epss_an +
                    lag_ca * self.params.epss_ca + 
                    lag_acc * 1. + 
                    lag_ccc * 1.)
            
            r = (lag_an  * self.params.r_an + 
                 lag_ca  * self.params.r_ca + 
                 lag_se  * 1. + 
                 lag_acc * 1. + 
                 lag_ccc * 1.)
            
            sour_p = (lag_an + lag_ca + lag_se) * self.params.sour_p
            
            sour_c = (lag_an + lag_ca + lag_se) * (self.params.l_ref)**2 / self.params.df_ref * (1 - tp)
                     
            sour_s = (lag_an * self.params.sour_s_an + 
                      lag_ca * self.params.sour_s_ca)
            
            svr = 3 * epss / r
            
            ratio_epsl = (epsl)**(alpha)
            ratio_epss = (epss)**(alpha)
            
            # ---- Residual of potential in electrolyte  ----
            
            # R_p = (ka_eff * inner(grad(p), grad(v_p) - kad_eff * inner(grad(ln(c))), grad(v_p) - a*F*j*v_p)*dx
            
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec) -> (num_quads,)
            c = np.sum(cell_sol_c[None,:,:] * self.fe_c.shape_vals[:,:,None], axis=1)[:,0]
            c = (lag_an + lag_ca + lag_se) * c + (lag_acc + lag_ccc) * 1.
            # **** Notes: following params are location dependent (quadrature point location dependent) ****
            # (num_quads,)
            kappa = self.calcKappa(c)
            ka_eff = kappa / self.params.ka_ref * ratio_epsl
            # **** Notes: kad_eff seems not to be consistent with the formulations in the PDF ****
            kad_eff = 2 * ka_eff * self.params.R * self.params.T / self.params.F *(1 - tp)
            
            
            # Handles the term `ka_eff * inner(grad(p), grad(v_p)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            p_grads = cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :]
            p_grads = np.sum(p_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v1 = np.sum(ka_eff[:,None,None,None] * p_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))

            
            # Handles the term `kad_eff * inner(grad(ln(c))), grad(v_p)*dx`
            # **** Notes: grad(ln(c)) can be written as d(ln(c))/d(c) * grad(c) = 1/c * grad(c) ****
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            c_grads = cell_sol_c[None, :, :, None] * cell_shape_grads_c[:, :, None, :]
            c_grads = np.sum(c_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v2 = np.sum(kad_eff[:,None,None,None] * 1./c[:,None,None,None] * c_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))
            
            
            # Handles the term `a*F*j*v_p*dx` (sour_p * a *j * v_p *dx)
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            j = np.sum(cell_sol_j[None,:,:] * self.fe_j.shape_vals[:,:,None], axis=1)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            Rp_v3 = sour_p * svr * np.sum(j[:,None,:] * self.fe_p.shape_vals[:,:,None] * cell_JxW_p[:,None,None], axis=0)
            
            Rp = Rp_v1 - Rp_v2 - Rp_v3
            
            
            # ---- Residual of diffusion in electrolyte  ----
            
            # R_c = (c_crt - c_old)/dt * v_c *dx + df_eff * inner(grad(c),grad(v_c))*dx - (1-t_e)*a*j*v_c*dx
            
            df_eff = self.calcDf(c) * ratio_epsl / self.params.df_ref
            
            # Handles the term ` (c_crt - c_old)/dt * v_c *dx`
            dc = c - cell_c_quad_old
            # (num_quads, 1, 1) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            Rc_v1 = self.params.c_dt_coeff * epsl * np.sum(dc[:,None,None] * self.fe_c.shape_vals[:,:,None]* cell_JxW_c[:,None,None], axis=0)
            
            
            # Handles the term `df_eff * inner(grad(c),grad(v_c))*dx`
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rc_v2 = np.sum(df_eff[:, None, None, None] * c_grads[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))
            
            
            # Handles the term `(1-t_e)*a*j*v_c*dx`
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            Rc_v3 = sour_c * svr * np.sum(j[:,None,:] * self.fe_c.shape_vals[:,:,None] * cell_JxW_c[:,None,None], axis=0)
            
            Rc = Rc_v1 + Rc_v2 - Rc_v3
            
            
            # ---- Residual of potential in electrode  ----
            
            # Only in anode and cathode!
            
            # R_s = (sig_eff * grad(s) * grad(v_s) + a*F*j*v_s)*dx
            
            sigma = (lag_an * self.params.sigma_an / self.params.sigma_ref + 
                     lag_ca * self.params.sigma_ca  / self.params.sigma_ref+
                     lag_acc * self.params.sigma_acc / self.params.sigma_ref+
                     lag_ccc * self.params.sigma_ccc / self.params.sigma_ref)
            
            sigma_eff = sigma * ratio_epss
            
            # Handles the term `sig_eff * grad(s) * grad(v_s)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            s_grads = cell_sol_s[None, :, :, None] * cell_shape_grads_s[:, :, None, :]
            s_grads = np.sum(s_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rs_v1 = sigma_eff * np.sum(s_grads[:, None, :, :] * cell_v_grads_JxW_s, axis=(0, -1))
            
            # Handles the term `a*F*j*v_s*dx`
            Rs_v2 = sour_s * svr * np.sum(j[:,None,:] * self.fe_s.shape_vals[:,:,None] * cell_JxW_s[:,None,None], axis=0)
            
            Rs = Rs_v1 + Rs_v2
            
            
            # ---- Residual of pore wall flux ----
            
            # cell_sol_list [(num_nodes,vec),...]
            
            cell_node_lag = jax.vmap(self.get_lag_coeffs)(cell_nodes_tag)
            
            cell_node_ks = cell_node_lag[:,0] * ks_an + cell_node_lag[:,1] * ks_ca
            
            Rj = self.flux_res_fns(cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j,
                                   cell_sol_csI_bound,
                                   cell_node_ks, 
                                   cell_node_lag[:,0], cell_node_lag[:,1])
            
            Rj = Rj / cell_nodes_sum
            
            weak_form = [Rp, Rc, Rs, Rj] # [(num_nodes, vec), ...]
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel


# -------------------- Bulter-Volmer equations --------------------

def pre_BV_fns(params):
    
    F = params.F      # Faraday's constant (C/mol)
    R = params.R      # Gas constant
    T = params.T      # Absolute temperature
    
    cs_max_an, cs_max_ca = params.cs_max
    
    calcUoc_neg = params.calcUoc_neg
    calcUoc_pos = params.calcUoc_pos
    
    def calcJ0(ce, css, ks, cs_max):
        
        # ks: nominal Reaction rates (A/m^2)*(mol^3/mol)^(1+alpha)
        # cs_max: maximum conc in electrode active material (mol/m^3)
        
        alpha_a = 0.5
        alpha_c = 0.5
        
        # delta_cs = np.where(cs_max - css > 0, cs_max - css, 1.)
        
        delta_cs = cs_max - css
        
        j0 = ks * (delta_cs)**alpha_a * (ce)**alpha_a * (css)**alpha_c
        
        return j0
    
    
    def calcBV(eta):
        
        BV = np.exp(0.5 * F / R / T * eta ) - np.exp(-0.5 * F / R / T * eta )
        
        return BV
    
    
    def Bulter_Volmer_fns(sol_p, sol_c, sol_s, css, 
                          ks, lag_an, lag_ca):
        '''
        Bulter-Volmer equation
        '''
        
        # j0
        cs_max = lag_an * cs_max_an + lag_ca * cs_max_ca
        j0 = calcJ0(sol_c, css, ks, cs_max)
        # Uoc
        sto = lag_an * css/cs_max_an + lag_ca * css/cs_max_ca
        Uoc = lag_an * calcUoc_neg(sto) + lag_ca * calcUoc_pos(sto)
        # eta
        eta = sol_s - sol_p - Uoc
        # BV
        BV = calcBV(eta)
        # j
        sol_j = j0 * BV
        
        return sol_j
    
    return Bulter_Volmer_fns

