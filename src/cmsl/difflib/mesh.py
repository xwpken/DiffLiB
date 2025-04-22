# Copyright (C) 2025 DiffLiB authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) an_y later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT An_y WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
"""Mesh."""

from dataclasses import dataclass

import re
import numpy as onp

from jax_fem.generate_mesh import rectangle_mesh


@dataclass
class Mesh:
    
    name:str
    
    def to_dict(self):
        return self.__dict__


def generate_mesh(infos):
    
    if 'file' not in infos.keys():
        mesh_macro = generate_quad_mesh(infos['z_an'], infos['n_an'], 
                                        infos['z_se'], infos['n_se'], 
                                        infos['z_ca'], infos['n_ca'], 
                                        infos['y'], infos['n_y'])
    else:
        mesh_macro = read_mph_mesh(infos)
    
    mesh_micro = generate_interval_mesh(infos['r'],infos['n_r'])
    
    return [mesh_macro, mesh_micro]


def generate_interval_mesh(hr,Nr,name='jax_mesh_micro'):
    
    mesh_micro = Mesh(name) 
    points = onp.linspace(0, hr, Nr+1).reshape(-1,1)
    mesh_micro.points = points
    mesh_micro.cells = onp.hstack((onp.linspace(0,len(points)-2,len(points)-1).reshape(-1,1),
                        onp.linspace(1,len(points)-1,len(points)-1).reshape(-1,1))).astype(onp.int32)
    
    mesh_micro.num_nodes = len(points)
    mesh_micro.num_cells = len(mesh_micro.cells)
    
    nodes_total = onp.linspace(0, len(points)-1, len(points), dtype=onp.int32)
    mesh_micro.bound_right = nodes_total[onp.isclose(points[:,0], hr)]
    
    return mesh_micro


def generate_quad_mesh(z_an,n_an,
                        z_sp,n_sp,
                        z_ca,n_ca,
                        y,n_y):
    
    '''
    
    Anode + Separator + Cathode
    
    Only for structured QUAD4 mesh (macro) and structured interval mesh (micro)
    '''
    
    # =========== macro ===========
    
    anmesh = rectangle_mesh(n_an, n_y, domain_x=z_an, domain_y=y)
    spmesh = rectangle_mesh(n_sp, n_y, domain_x=z_sp, domain_y=y)
    camesh = rectangle_mesh(n_ca, n_y, domain_x=z_ca, domain_y=y)
    aux_mesh = rectangle_mesh(n_an+n_sp+n_ca, n_y, domain_x=z_an+z_sp+z_ca, domain_y=y)
    
    cells = aux_mesh.cells_dict['quad']
    
    pos_an = anmesh.points
    pos_sp = onp.vstack([spmesh.points[:,0]+z_an,spmesh.points[:,1]]).T
    pos_ca = onp.vstack([camesh.points[:,0]+z_an+z_sp,camesh.points[:,1]]).T
    points = onp.vstack((pos_an, pos_sp[n_y+1:,:],pos_ca[n_y+1:,:]))
    
    # points
    nodes_total = onp.linspace(0, len(points)-1, len(points), dtype=onp.int32)
    nind_an = int((n_y+1)*(n_an+1))
    nind_ca = int((n_y+1)*(n_ca+1))
    
    # cells
    cind_an = int(n_y*n_an)
    cind_ca = int(n_y*n_ca)
    
    # assembly
    mesh_macro = Mesh('jax_mesh_macro')
    
    mesh_macro.model = 'P2D'
    
    mesh_macro.cell_type = 'quad'
    
    mesh_macro.dim = 2
    
    mesh_macro.points = points
    mesh_macro.num_nodes = len(points)
    mesh_macro.nodes_anode = nodes_total[:nind_an]
    mesh_macro.nodes_separator = nodes_total[nind_an:-nind_ca]
    mesh_macro.nodes_cathode = nodes_total[-nind_ca:]   
    mesh_macro.nodes_bound_anright = nodes_total[onp.isclose(points[:,0], z_an)]
    mesh_macro.nodes_bound_caleft = nodes_total[onp.isclose(points[:,0], z_an+z_sp)]
    
    mesh_macro.cells = cells
    mesh_macro.cells_anode = cells[:cind_an]
    mesh_macro.cells_separator = cells[cind_an:-cind_ca]
    mesh_macro.cells_cathode = cells[-cind_ca:]
    
    # terminal position
    mesh_macro.terminal = len(mesh_macro.points)-2
    
    # assign dofs and auxiliary variables
    mesh_macro = assign_macro_dofs(mesh_macro)
    mesh_macro = assign_aux_vars(mesh_macro)
    
    return mesh_macro


def quad_mesh_p2dcc(z_acc,n_acc,
                     z_an,n_an,
                     z_sp,n_sp,
                     z_ca,n_ca,
                     z_ccc,n_ccc,
                     y,n_y):
    
    '''
    
    ACC + Anode + Separator + Cathode + CCC
    
    Only for structured QUAD4 mesh (macro) and structured interval mesh (micro)
    '''
    
    # =========== macro ===========
    
    accmesh = rectangle_mesh(n_acc, n_y, domain_x=z_acc, domain_y=y)
    anmesh = rectangle_mesh(n_an, n_y, domain_x=z_an, domain_y=y)
    spmesh = rectangle_mesh(n_sp, n_y, domain_x=z_sp, domain_y=y)
    camesh = rectangle_mesh(n_ca, n_y, domain_x=z_ca, domain_y=y)
    cccmesh = rectangle_mesh(n_ccc, n_y, domain_x=z_ccc, domain_y=y)
    
    aux_mesh = rectangle_mesh(n_acc+n_an+n_sp+n_ca+n_ccc, n_y, 
                              domain_x = z_acc+z_an+z_sp+z_ca+z_ccc, domain_y = y)
    
    cells = aux_mesh.cells_dict['quad']
    
    pos_acc = accmesh.points
    pos_an = onp.vstack([anmesh.points[:,0]+z_acc,anmesh.points[:,1]]).T
    pos_sp = onp.vstack([spmesh.points[:,0]+z_acc+z_an,spmesh.points[:,1]]).T
    pos_ca = onp.vstack([camesh.points[:,0]+z_acc+z_an+z_sp,camesh.points[:,1]]).T
    pos_ccc = onp.vstack([cccmesh.points[:,0]+z_acc+z_an+z_sp+z_ca,cccmesh.points[:,1]]).T
    points = onp.vstack((pos_acc, pos_an[n_y+1:,:], pos_sp[n_y+1:,:],pos_ca[n_y+1:,:],pos_ccc[n_y+1:,:]))
    
    # points
    nodes_total = onp.linspace(0, len(points)-1, len(points), dtype=onp.int32)
    nind_acc = int((n_y+1)*(n_acc+1-1))
    nind_an = int((n_y+1)*(n_an+1)) + nind_acc
    nind_se = int((n_y+1)*(n_sp+1-2)) + nind_an
    nind_ca = int((n_y+1)*(n_ca+1)) + nind_se
    nind_ccc = int((n_y+1)*(n_ccc+1-1)) + nind_ca
    
    # cells
    cind_acc = int(n_y*n_acc)
    cind_an = int(n_y*n_an) + cind_acc
    cind_se = int(n_y*n_sp) + cind_an
    cind_ca = int(n_y*n_ca) + cind_se
    cind_ccc = int(n_y*n_ccc) + cind_ca
    
    # assembly
    ccmesh_macro = Mesh('jax_ccmesh_macro')
    
    ccmesh_macro.model = 'P2DCC'
    
    ccmesh_macro.cell_type = 'quad'
    
    ccmesh_macro.points = points
    ccmesh_macro.num_nodes = len(points)
    
    ccmesh_macro.nodes_acc = nodes_total[:nind_acc]
    ccmesh_macro.nodes_anode = nodes_total[nind_acc:nind_an]
    ccmesh_macro.nodes_separator = nodes_total[nind_an:nind_se]
    ccmesh_macro.nodes_cathode = nodes_total[nind_se:nind_ca]  
    ccmesh_macro.nodes_ccc = nodes_total[nind_ca:nind_ccc]  
    
    
    ccmesh_macro.nodes_bound_anright = nodes_total[onp.isclose(points[:,0], z_acc+z_an)]
    ccmesh_macro.nodes_bound_caleft = nodes_total[onp.isclose(points[:,0], z_acc+z_an+z_sp)]
    
    ccmesh_macro.cells = cells
    ccmesh_macro.cells_acc = cells[:cind_acc]
    ccmesh_macro.cells_anode = cells[cind_acc:cind_an]
    ccmesh_macro.cells_separator = cells[cind_an:cind_se]
    ccmesh_macro.cells_cathode = cells[cind_se:cind_ca]
    ccmesh_macro.cells_ccc = cells[cind_ca:cind_ccc]
    
    # assign dofs and auxiliary variables
    ccmesh_macro = assign_macro_dofs(ccmesh_macro)
    ccmesh_macro = assign_aux_vars(ccmesh_macro)
    
    return ccmesh_macro



def read_mph_mesh(infos):
    
    '''
    read comsol mesh file (.mphtxt)
    
    '''
    
    name = infos['name']
    file_path = infos['file']
    domian_dicts = infos['domains']
    
    def get_comsol_mesh_type(ele_type):
        
        if ele_type == 'quad':
            cell_type = 'quad'
        elif ele_type == 'hex':
            cell_type = 'hexahedron'
        else:
            raise NotImplementedError
            
        return cell_type
    
    
    def reorder_cells(ele_type, cells):
        '''
        COMSOL has different cell vertices from JAX-FEM
        ''' 
        if ele_type == 'hexahedron':
            order = [0, 1, 3, 2, 4, 5, 7, 6]
        elif ele_type == 'quad':
            order = [0, 1, 3, 2]
        else:
            raise NotImplementedError
            
        cells = cells[:,order]
        return cells
    
    # read .mphtxt file from COMSOL
    with open(file_path, "r") as f:  
        data = f.readline()
        while data:
            data = f.readline()
            
            if 'sdim' in data:
                dim = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
        
            if 'number of mesh vertices' in data:
                num_nodes = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                points = onp.zeros((num_nodes,dim))
            
            if 'Mesh vertex coordinates' in data:
                print(f'Reading {num_nodes} element nodes in {dim}D model ...')
                for i in range(num_nodes):
                    points[i,:] = onp.fromstring(f.readline().strip(), sep=' ')
            
            if 'type name' in data:
                ele_type = re.sub(r'\d', '', data.split('#')[0].replace(" ", ""))
                ele_type = get_comsol_mesh_type(ele_type)
            
            if 'number of vertices per element' in data:
                num_cell_nodes = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                
            if 'number of elements' in data:
                num_cells = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                cells = onp.zeros((num_cells,num_cell_nodes))
                cells_tag = onp.zeros((num_cells,1))
            
            if data.strip() == '# Elements' :
                print(f'Reading {num_cells} {ele_type} elements in {dim}D model ...')
                for i in range(num_cells):
                    cells[i,:] = onp.fromstring(f.readline().strip(), sep=' ')
            
            if data.strip() == '# Geometric entity indices' :
                for i in range(num_cells):
                    cells_tag[i,0] = float(f.readline().strip())
    
    cells = reorder_cells(ele_type, cells).astype(onp.int32)
    cells_tag = cells_tag.reshape(-1)
    
    # assembly
    mesh_macro = Mesh('comsol_mesh_macro')
    
    mesh_macro.cell_type = ele_type
    
    mesh_macro.dim = dim
    
    mesh_macro.name = name
    
    mesh_macro.model = 'P2D'
    
    mesh_macro.points = points
    mesh_macro.num_nodes = len(points)
    mesh_macro.cells = cells
    
    # Be careful about the domain ID in COMSOL!
    
    # domain cells
    mesh_macro.cells_anode = cells[cells_tag==domian_dicts['anode'],:]
    mesh_macro.cells_cathode = cells[cells_tag==domian_dicts['cathode'],:]
    mesh_macro.cells_separator = cells[cells_tag==domian_dicts['separator'],:]
    
    def eliminate_ac_nodes(cells_domain):
        nodes_domain = onp.unique(cells_domain)
        nind_domian = onp.logical_or(onp.isin(nodes_domain,mesh_macro.nodes_anode),
                                      onp.isin(nodes_domain,mesh_macro.nodes_cathode))
        return nodes_domain[~nind_domian]
    
    # domain nodes
    mesh_macro.nodes_anode = onp.unique(mesh_macro.cells_anode)
    mesh_macro.nodes_cathode = onp.unique(mesh_macro.cells_cathode)
    mesh_macro.nodes_separator = eliminate_ac_nodes(mesh_macro.cells_separator)
    
    if len(onp.unique(cells_tag)) == 5:
        mesh_macro.cells_acc = cells[cells_tag==domian_dicts['acc'],:]
        mesh_macro.cells_ccc = cells[cells_tag==domian_dicts['ccc'],:]
        mesh_macro.nodes_acc = eliminate_ac_nodes(mesh_macro.cells_acc)
        mesh_macro.nodes_ccc = eliminate_ac_nodes(mesh_macro.cells_ccc)
        
        mesh_macro.model = 'P2DCC'
    
    # assign dofs and auxiliary variables
    mesh_macro = assign_macro_dofs(mesh_macro)
    mesh_macro = assign_aux_vars(mesh_macro)    
    
    return mesh_macro


    domian_dicts = infos['domains']
    
    def get_comsol_mesh_type(ele_type):
        
        if ele_type == 'quad':
            cell_type = 'quad'
        elif ele_type == 'hex':
            cell_type = 'hexahedron'
        else:
            raise NotImplementedError
            
        return cell_type
    
    
    def reorder_cells(ele_type, cells):
        '''
        COMSOL has different cell vertices from JAX-FEM
        ''' 
        if ele_type == 'hexahedron':
            order = [0, 1, 3, 2, 4, 5, 7, 6]
        elif ele_type == 'quad':
            order = [0, 1, 3, 2]
        else:
            raise NotImplementedError
            
        cells = cells[:,order]
        return cells
    
    # read .mphtxt file from COMSOL
    with open(file_path, "r") as f:  
        data = f.readline()
        while data:
            data = f.readline()
            
            if 'sdim' in data:
                dim = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
        
            if 'number of mesh vertices' in data:
                num_nodes = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                points = onp.zeros((num_nodes,dim))
            
            if 'Mesh vertex coordinates' in data:
                print(f'Reading {num_nodes} element nodes in {dim}D model ...')
                for i in range(num_nodes):
                    points[i,:] = onp.fromstring(f.readline().strip(), sep=' ')
            
            if 'type name' in data:
                ele_type = re.sub(r'\d', '', data.split('#')[0].replace(" ", ""))
                ele_type = get_comsol_mesh_type(ele_type)
            
            if 'number of vertices per element' in data:
                num_cell_nodes = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                
            if 'number of elements' in data:
                num_cells = onp.array(re.findall(r"\d+\.?\d*",data),dtype=onp.int32)[0]
                cells = onp.zeros((num_cells,num_cell_nodes))
                cells_tag = onp.zeros((num_cells,1))
            
            if data.strip() == '# Elements' :
                print(f'Reading {num_cells} {ele_type} elements in {dim}D model ...')
                for i in range(num_cells):
                    cells[i,:] = onp.fromstring(f.readline().strip(), sep=' ')
            
            if data.strip() == '# Geometric entity indices' :
                for i in range(num_cells):
                    cells_tag[i,0] = float(f.readline().strip())
    
    cells = reorder_cells(ele_type, cells).astype(onp.int32)
    cells_tag = cells_tag.reshape(-1)
    
    # assembly
    mesh_macro = Mesh('comsol_mesh_macro')
    
    mesh_macro.cell_type = ele_type
    
    mesh_macro.dim = dim
    
    mesh_macro.name = name
    
    mesh_macro.model = 'P2D'
    
    mesh_macro.points = points
    mesh_macro.num_nodes = len(points)
    mesh_macro.cells = cells
    
    # Be careful about the domain ID in COMSOL!
    
    # domain cells
    mesh_macro.cells_anode = cells[cells_tag==domian_dicts['anode'],:]
    mesh_macro.cells_cathode = cells[cells_tag==domian_dicts['cathode'],:]
    mesh_macro.cells_separator = cells[cells_tag==domian_dicts['separator'],:]
    
    def eliminate_ac_nodes(cells_domain):
        nodes_domain = onp.unique(cells_domain)
        nind_domian = onp.logical_or(onp.isin(nodes_domain,mesh_macro.nodes_anode),
                                      onp.isin(nodes_domain,mesh_macro.nodes_cathode))
        return nodes_domain[~nind_domian]
    
    # domain nodes
    mesh_macro.nodes_anode = onp.unique(mesh_macro.cells_anode)
    mesh_macro.nodes_cathode = onp.unique(mesh_macro.cells_cathode)
    mesh_macro.nodes_separator = eliminate_ac_nodes(mesh_macro.cells_separator)
    
    if len(onp.unique(cells_tag)) == 5:
        mesh_macro.cells_acc = cells[cells_tag==domian_dicts['acc'],:]
        mesh_macro.cells_ccc = cells[cells_tag==domian_dicts['ccc'],:]
        mesh_macro.nodes_acc = eliminate_ac_nodes(mesh_macro.cells_acc)
        mesh_macro.nodes_ccc = eliminate_ac_nodes(mesh_macro.cells_ccc)
        
        mesh_macro.model = 'P2DCC'
    
    # assign dofs and auxiliary variables
    mesh_macro = assign_macro_dofs(mesh_macro)
    mesh_macro = assign_aux_vars(mesh_macro)    
    
    return mesh_macro


def assign_macro_dofs(mesh_macro):
    
    '''
    fns to assign dofs and auxiliary variables
    '''
    
    # unpack
    nnode_macro = mesh_macro.num_nodes
    
    nodes_anode = mesh_macro.nodes_anode
    nodes_cathode = mesh_macro.nodes_cathode
    nodes_separator = mesh_macro.nodes_separator
    
    # dofs
    dofs_p = onp.linspace(0, nnode_macro-1, nnode_macro, dtype=onp.int32)
    dofs_p_an = nodes_anode
    dofs_p_ca = nodes_cathode
    dofs_p_se = nodes_separator

    dofs_c = dofs_p + nnode_macro
    dofs_c_an = nnode_macro + nodes_anode
    dofs_c_ca = nnode_macro + nodes_cathode
    dofs_c_se = nnode_macro + nodes_separator

    # dofs s are defined in all domain but only meaningful in electrode and CC
    dofs_s = dofs_c + nnode_macro 
    dofs_s_an = 2*nnode_macro + nodes_anode
    dofs_s_ca = 2*nnode_macro + nodes_cathode

    # dofs j are defined in all domain but only meaningful in electrode
    dofs_j = dofs_s + nnode_macro
    dofs_j_an = 3*nnode_macro + nodes_anode
    dofs_j_ca = 3*nnode_macro + nodes_cathode
    
    # assembly
    mesh_macro.dofs = [dofs_p, dofs_c, dofs_s, dofs_j]
    mesh_macro.dofs_an = [dofs_p_an, dofs_c_an, dofs_s_an, dofs_j_an]
    mesh_macro.dofs_ca = [dofs_p_ca, dofs_c_ca, dofs_s_ca, dofs_j_ca]
    mesh_macro.dofs_se = [dofs_p_se, dofs_c_se]
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        # current collector
        nodes_acc = mesh_macro.nodes_acc
        nodes_ccc = mesh_macro.nodes_ccc
        dofs_s_acc = 2*nnode_macro + nodes_acc
        dofs_s_ccc = 2*nnode_macro + nodes_ccc
        mesh_macro.dofs_cc = [dofs_s_acc, dofs_s_ccc]
    
    return mesh_macro


def assign_aux_vars(mesh_macro):
    
    points = mesh_macro.points
    cells = mesh_macro.cells
    
    num_nodes = mesh_macro.num_nodes
    
    nodes_anode = mesh_macro.nodes_anode
    nodes_separator = mesh_macro.nodes_separator
    nodes_cathode = mesh_macro.nodes_cathode
    
    # nodes tag
    nodes_tag = onp.zeros((len(points)))
    nodes_tag[nodes_anode] = 1.
    nodes_tag[nodes_cathode] = 2.
    nodes_tag[nodes_separator] = 3.
    
    nodes_micro = onp.hstack((nodes_anode.reshape(-1), 
                              nodes_cathode.reshape(-1)))
    nodes_micro_tag = nodes_tag[nodes_micro]
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        nodes_acc = mesh_macro.nodes_acc
        nodes_ccc = mesh_macro.nodes_ccc
        nodes_tag[nodes_acc] = 4.
        nodes_tag[nodes_ccc] = 5.
    
    # cell nodes tag
    cells_nodes_tag = onp.take(nodes_tag, cells)
    
    # cells tag
    cells_tag = cells_nodes_tag.astype(onp.int32)
    cind_se = onp.unique(onp.nonzero(cells_tag==3.)[0])
    cells_tag[cind_se,:] = 3.
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        cind_acc = onp.unique(onp.nonzero(cells_tag==4)[0])
        cind_ccc = onp.unique(onp.nonzero(cells_tag==5)[0])
        cells_tag[cind_acc,:] = 4.
        cells_tag[cind_ccc,:] = 5.
    
    cells_tag = onp.sum(cells_tag,axis=1)/cells.shape[1]
    
    # repeat num of each node
    nodes,nodes_sum = onp.unique(cells,return_counts=True)
    cells_nodes_sum = onp.take(nodes_sum,cells)[:,:,None].astype(onp.float64)
    
    mesh_macro.cells_vars = [cells_tag, cells_nodes_tag, cells_nodes_sum]
    mesh_macro.nodes_vars = [nodes_tag, nodes_micro, nodes_micro_tag]
    
    # position and dof of each solution
    
    def match_dof_point(dofs_list):
        '''
        return the position of the input dof
        '''
        x_list = []
        for dofs in dofs_list:
            xinds = dofs - num_nodes*(onp.floor(dofs/num_nodes)).astype(onp.int32)
            x_list.append(points[xinds,:])
        return x_list
    
    mesh_macro.match_dof_point = match_dof_point
    
    dofs_s_an, dofs_s_ca = mesh_macro.dofs_an[2], mesh_macro.dofs_ca[2]
    x_s_an, x_s_ca = match_dof_point([dofs_s_an, dofs_s_ca])
    ind_sol = {'ind_phis_an':[x_s_an, dofs_s_an], 
               'ind_phis_ca':[x_s_ca, dofs_s_ca]}
    
    if hasattr(mesh_macro, 'nodes_acc') and hasattr(mesh_macro, 'nodes_ccc'):
        
        # elimate p and c from cc domain
        # sol_p means p is meaningful on this node
        dofs_p, dofs_c, _, _ = mesh_macro.dofs
        cc_nodes = onp.vstack((mesh_macro.nodes_acc.reshape(-1,1), 
                               mesh_macro.nodes_ccc.reshape(-1,1)))
        # remove the cc_nodes row
        dofs_sol_p = onp.delete(dofs_p, cc_nodes, axis=0)
        dofs_sol_c = onp.delete(dofs_c, cc_nodes, axis=0)
        
        # cc
        dofs_s_acc, dofs_s_ccc = mesh_macro.dofs_cc
        x_s_acc, x_s_ccc = match_dof_point([dofs_s_acc, dofs_s_ccc])
        ind_sol['ind_phis_acc'] = [x_s_acc, dofs_s_acc]
        ind_sol['ind_phis_ccc'] = [x_s_ccc, dofs_s_ccc]
        
    else:
        dofs_sol_p, dofs_sol_c, _, _ = mesh_macro.dofs
        
    x_sol_p, x_sol_c = match_dof_point([dofs_sol_p, dofs_sol_c])
    ind_sol['ind_phil'] = [x_sol_p, dofs_sol_p]
    ind_sol['ind_cl'] = [x_sol_c, dofs_sol_c]

    mesh_macro.ind_sol = ind_sol
    
    return mesh_macro


