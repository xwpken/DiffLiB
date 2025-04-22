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
"""Pre-defined models."""


import jax.numpy as np

from jax_fem.generate_mesh import Mesh

from .core import P2D, P2DCC


def rectangle_cube(params, mesh_macro, problem_micro):
    '''
    Rectangle & Cube LIB models
    
    '''
    
    dim = mesh_macro.dim
    
    if dim == 2:
        ele_type = 'QUAD4'
    else:
        ele_type = 'HEX8'
    
    # mesh for JAX-FEM
    jax_mesh = Mesh(mesh_macro.points, mesh_macro.cells, ele_type=ele_type)
    
    # boundary conditions
    # only for the electrode potential (s) 
    
    x_max = (jax_mesh.points).max(0)[0] 
    x_min = (jax_mesh.points).min(0)[0] 
    
    nodes_se = mesh_macro.nodes_separator

    def zero_dirichlet(point):
        return 0.
    
    def current_pos(point):
        return np.isclose(point[0], x_max, atol=1e-5)
    
    def ground_pos(point):
        return np.isclose(point[0], x_min, atol=1e-5)
    
    def separator(point, ind):
        return np.isin(ind, nodes_se) 
    
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        # with current collectors
        
        problem = P2DCC
        
        nodes_acc = mesh_macro.nodes_acc
        nodes_ccc = mesh_macro.nodes_ccc
        
        def anode_cc(point, ind):
            return np.isin(ind, nodes_acc) 
        
        def cathode_cc(point, ind):
            return np.isin(ind, nodes_ccc) 
        
        diri_bc_info_p = [[anode_cc, cathode_cc], [0]*2, [zero_dirichlet]*2]
        diri_bc_info_c = [[anode_cc, cathode_cc], [0]*2, [zero_dirichlet]*2]
        diri_bc_info_s = [[separator, ground_pos], [0]*2, [zero_dirichlet]*2]
        diri_bc_info_j = [[separator, anode_cc, cathode_cc], [0]*3, [zero_dirichlet]*3]
        
    else:
        # without current collectors
        
        problem = P2D
        
        diri_bc_info_p = None
        diri_bc_info_c = None
        diri_bc_info_s = [[ground_pos, separator], [0]*2, [zero_dirichlet]*2]
        diri_bc_info_j = [[separator], [0], [zero_dirichlet]]
    
    location_fns_s = [current_pos]
    
    cells_vars = mesh_macro.cells_vars

    # macro problem for (p,c,s,j)
    problem = problem([jax_mesh]*4, vec = [1]*4, dim=dim, ele_type = [ele_type]*4, gauss_order=[2]*4,
                      dirichlet_bc_info = [diri_bc_info_p,diri_bc_info_c,diri_bc_info_s,diri_bc_info_j],
                      location_fns = location_fns_s, 
                      additional_info=(params, cells_vars, problem_micro.flux_res_fns))
    
    return problem



def pouch(params, mesh_macro, problem_micro):
    '''
    Pouch-type LIB models
    
    '''
    
    ele_type = 'HEX8'
    
    # mesh for JAX-FEM
    jax_mesh = Mesh(mesh_macro.points, mesh_macro.cells, ele_type=ele_type)
    
    # boundary conditions
    # only for the electrode potential (s) 
    
    y_max = (jax_mesh.points).max(0)[1] 
    y_min = (jax_mesh.points).min(0)[1] 
    
    nodes_se = mesh_macro.nodes_separator
    
    def zero_dirichlet(point):
        return 0.
    
    def current_pos(point):
        return np.isclose(point[1], y_min, atol=1e-5)
    
    def ground_pos(point):
        return np.isclose(point[1], y_max, atol=1e-5)
        
    def separator(point, ind):
        return np.isin(ind, nodes_se) 
    
    # with current collectors
    
    problem = P2DCC
    
    nodes_acc = mesh_macro.nodes_acc
    nodes_ccc = mesh_macro.nodes_ccc
    
    def anode_cc(point, ind):
        return np.isin(ind, nodes_acc) 
    
    def cathode_cc(point, ind):
        return np.isin(ind, nodes_ccc) 
    
    diri_bc_info_p = [[anode_cc, cathode_cc], [0]*2, [zero_dirichlet]*2]
    diri_bc_info_c = [[anode_cc, cathode_cc], [0]*2, [zero_dirichlet]*2]
    diri_bc_info_s = [[separator, ground_pos], [0]*2, [zero_dirichlet]*2]
    diri_bc_info_j = [[separator, anode_cc, cathode_cc], [0]*3, [zero_dirichlet]*3]

    location_fns_s = [current_pos]
    
    cells_vars = mesh_macro.cells_vars
    
    # macro problem for (p,c,s,j)
    problem = problem([jax_mesh]*4, vec = [1]*4, dim=3, ele_type = [ele_type]*4, gauss_order=[2]*4,
                      dirichlet_bc_info = [diri_bc_info_p,diri_bc_info_c,diri_bc_info_s,diri_bc_info_j],
                      location_fns = location_fns_s, 
                      additional_info=(params, cells_vars, problem_micro.flux_res_fns))
    
    return problem
