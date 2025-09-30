
import numpy as np
import jax.numpy as jnp
import jax

import poisson_utils.tensor
import poisson_utils.user_funcs
import poisson_utils.geometry
import poisson_utils.rhs


import block_assembl_poisson.solver

v_coeff_func = jax.vmap(poisson_utils.user_funcs.coeff_func, [0, 0, 0])


def assemble_elem_stiffness_matrix(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z,
                          gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    """
    
                | Qx Qy Qz || rx ry rz |     | rx sx tx || Qx |
    Ke =                    | sx sy sz | J W | ry sy ty || Qy |
                            | tx ty tz |     | rz sz tz || Qz |
    """
    J, D = poisson_utils.geometry.elem_geo_factors_with_inverse(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, d_gauss_1d)
    gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z = \
        poisson_utils.geometry.transform_elem_coords(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, gll_to_gauss_1d)
    
    

    mu = v_coeff_func(gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z)

    # factor = jnp.zeros(((p_order+1)**3, 6))
    """
                  0  3  4
      factor      3  1  5
                  4  5  2
    """
    factor_0 = (D[0]*D[0] + D[1]*D[1] + D[2]*D[2] ) * J * gauss_w_3d * mu # d2u/dx^2
    factor_1 = (D[3]*D[3] + D[4]*D[4] + D[5]*D[5] ) * J * gauss_w_3d * mu # d2u/dy^2
    factor_2 = (D[6]*D[6] + D[7]*D[7] + D[8]*D[8] ) * J * gauss_w_3d * mu # d2u/dz^2
    
    factor_3 = (D[0]*D[3] + D[1]*D[4] + D[2]*D[5] ) * J * gauss_w_3d * mu # d2u/dxdy
    factor_4 = (D[0]*D[6] + D[1]*D[7] + D[2]*D[8] ) * J * gauss_w_3d * mu # d2u/dxdz
    factor_5 = (D[3]*D[6] + D[4]*D[7] + D[5]*D[8] ) * J * gauss_w_3d * mu # d2u/dydz

    """
    Computing
        Ke =   r.Qx' * diag(factor(:,0)) * r.Qx ...
             + r.Qy' * diag(factor(:,1)) * r.Qy ...
             + r.Qz' * diag(factor(:,2)) * r.Qz ...
             + r.Qx' * diag(factor(:,3)) * r.Qy ...
             + r.Qy' * diag(factor(:,3)) * r.Qx ...
             + r.Qx' * diag(factor(:,4)) * r.Qz ...
             + r.Qz' * diag(factor(:,4)) * r.Qx ...
             + r.Qz' * diag(factor(:,5)) * r.Qy ...
             + r.Qy' * diag(factor(:,5)) * r.Qz ;
        
    """

    Qx = poisson_utils.tensor.kron_kron(gll_to_gauss_1d, gll_to_gauss_1d, d_gauss_1d)
    Qy = poisson_utils.tensor.kron_kron(gll_to_gauss_1d, d_gauss_1d, gll_to_gauss_1d)
    Qz = poisson_utils.tensor.kron_kron(d_gauss_1d, gll_to_gauss_1d, gll_to_gauss_1d)



    QxT_fac0_Qx = jnp.matmul(Qx.T, (factor_0[:, None]  * Qx))
    QyT_fac1_Qy = jnp.matmul(Qy.T, (factor_1[:, None]  * Qy))
    QzT_fac2_Qz = jnp.matmul(Qz.T, (factor_2[:, None]  * Qz))
    QxT_fac3_Qy = jnp.matmul(Qx.T, (factor_3[:, None]  * Qy))
    QyT_fac4_Qx = jnp.matmul(Qy.T, (factor_3[:, None]  * Qx))
    QxT_fac5_Qz = jnp.matmul(Qx.T, (factor_4[:, None]  * Qz))
    QzT_fac6_Qx = jnp.matmul(Qz.T, (factor_4[:, None]  * Qx))
    QzT_fac7_Qy = jnp.matmul(Qz.T, (factor_5[:, None]  * Qy))
    QyT_fac8_Qz = jnp.matmul(Qy.T, (factor_5[:, None]  * Qz))


    return QxT_fac0_Qx + QyT_fac1_Qy + QzT_fac2_Qz + QxT_fac3_Qy + QyT_fac4_Qx + QxT_fac5_Qz + QzT_fac6_Qx + QzT_fac7_Qy + QyT_fac8_Qz

def assemble_block_matrices(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n,
                            gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    n_elems = e_to_n.shape[0]
    nodes_per_element  = (p_order+1)**3
    e_to_n_linear = e_to_n.flatten()

    node_coords_x_scattered = node_coords_x[e_to_n_linear]
    node_coords_x_scattered = jnp.reshape(node_coords_x_scattered, (n_elems, nodes_per_element))
    node_coords_y_scattered = node_coords_y[e_to_n_linear]
    node_coords_y_scattered = jnp.reshape(node_coords_y_scattered, (n_elems, nodes_per_element))
    node_coords_z_scattered = node_coords_z[e_to_n_linear]
    node_coords_z_scattered = jnp.reshape(node_coords_z_scattered, (n_elems, nodes_per_element))


    v_asm_elem_stiffn_mat = jax.vmap(assemble_elem_stiffness_matrix, in_axes=(None, 0, 0, 0, None, None, None))
    block_matrices = v_asm_elem_stiffn_mat(p_order, node_coords_x_scattered, node_coords_y_scattered, node_coords_z_scattered, 
                                           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    
    return block_matrices.reshape(n_elems, nodes_per_element, nodes_per_element)

def blocked_matvec(p_order, block_matrices, e_to_n, e_to_n_sorted, e_to_n_sort_idx, 
                   vec_u, boundary_indices):
    n_elems = e_to_n.shape[0]
    nodes_per_element  = (p_order+1)**3
    e_to_n_linear = e_to_n.flatten()
    vec_u_scattered = jax.lax.optimization_barrier(vec_u[e_to_n_linear])
    vec_u_scattered = jnp.reshape(vec_u_scattered, (n_elems, nodes_per_element))

    vec_Au_scattered = jax.lax.optimization_barrier(jnp.matvec(block_matrices, vec_u_scattered))  # batched matvec
    vec_Au_scattered = vec_Au_scattered.flatten()

    vec_Au_scattered_sorted = vec_Au_scattered[e_to_n_sort_idx]
    vec_Au_zero = jnp.zeros_like(vec_u)
    vec_Au = vec_Au_zero.at[e_to_n_sorted].add(vec_Au_scattered_sorted, indices_are_sorted=True)

    # np.add.at(vec_Au, e_to_n_linear, vec_Au_scattered)

    # do not change values in boundary conditions
    vec_Au = vec_Au.at[boundary_indices].set(vec_u[boundary_indices], unique_indices=True)

    return vec_Au



def solve_poisson(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, maxiter):

    rhs_vec = jax.lax.optimization_barrier(poisson_utils.rhs.rhs(p_order, 
                                                                 node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                                                                 gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))
    
    block_matrices = jax.lax.optimization_barrier(assemble_block_matrices(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))
    

    
    u_numeric = block_assembl_poisson.solver.conjugate_gradient_unrolled(p_order, block_matrices, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  rhs_vec, 1e-16, maxiter)
    

    return u_numeric