import numpy as np
import jax.numpy as jnp
import jax

import poisson_utils.tensor
import poisson_utils.user_funcs
import poisson_utils.geometry
import poisson_utils.rhs


import mat_free_batched_poisson.solver

import mat_free_poisson.poisson

v_coeff_func = jax.vmap(poisson_utils.user_funcs.coeff_func, [0, 0, 0])





def elem_stiffness_matvec_batched(p_order, gll_node_coords_x_batched, gll_node_coords_y_batched, gll_node_coords_z_batched, 
                                  vec_u_scattered_batched, 
                                  gauss_w_3d, batched_gll_to_gauss_1d, batched_d_gauss_1d,
                                  batched_gll_to_gauss_1d_tr, batched_d_gauss_1d_tr):
    """
    
                | Qx Qy Qz || rx ry rz |     | rx sx tx || Qx |
    Ke =                    | sx sy sz | J W | ry sy ty || Qy |
                            | tx ty tz |     | rz sz tz || Qz |
    """
    J, D = poisson_utils.geometry.elem_geo_factors_with_inverse_batched(p_order, 
                                                                        gll_node_coords_x_batched, gll_node_coords_y_batched, gll_node_coords_z_batched, 
                                                                        batched_d_gauss_1d)
    gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z \
        = poisson_utils.geometry.transform_elem_coords_batched(p_order, 
                                                               gll_node_coords_x_batched, gll_node_coords_y_batched, gll_node_coords_z_batched, 
                                                               batched_gll_to_gauss_1d)
    
    gauss_node_coords_x = gauss_node_coords_x.flatten()
    gauss_node_coords_y = gauss_node_coords_y.flatten()
    gauss_node_coords_z = gauss_node_coords_z.flatten()


    mu = v_coeff_func(gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z)
    mu = mu.reshape(gll_node_coords_x_batched.shape[0], gll_node_coords_x_batched.shape[1])

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
        u_new = Ke*u
    """
    batch = gll_node_coords_x_batched.shape[0]
    Qx_u = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d, 
                                  batched_gll_to_gauss_1d, 
                                  batched_d_gauss_1d, vec_u_scattered_batched, p_order+1)
    Qy_u = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d, 
                                  batched_d_gauss_1d, 
                                  batched_gll_to_gauss_1d, vec_u_scattered_batched, p_order+1)
    Qz_u = poisson_utils.tensor.ABCx_batched_bigABC(batched_d_gauss_1d, 
                                  batched_gll_to_gauss_1d, 
                                  batched_gll_to_gauss_1d, vec_u_scattered_batched, p_order+1)

    fac_mult_0 = factor_0  * Qx_u
    fac_mult_1 = factor_1  * Qy_u
    fac_mult_2 = factor_2  * Qz_u
    fac_mult_3 = factor_3  * Qy_u
    fac_mult_4 = factor_3  * Qx_u
    fac_mult_5 = factor_4  * Qz_u
    fac_mult_6 = factor_4  * Qx_u
    fac_mult_7 = factor_5  * Qy_u
    fac_mult_8 = factor_5  * Qz_u

    QxT_mult0 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, fac_mult_0, p_order+1)

    QyT_mult1 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_1, p_order+1)

    QzT_mult2 = poisson_utils.tensor.ABCx_batched_bigABC(batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_2, p_order+1)

    QxT_mult3 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, fac_mult_3, p_order+1)

    QyT_mult4 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_4, p_order+1)

    QxT_mult5 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, fac_mult_5, p_order+1)

    QzT_mult6 = poisson_utils.tensor.ABCx_batched_bigABC(batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_6, p_order+1)

    QzT_mult7 = poisson_utils.tensor.ABCx_batched_bigABC(batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_7, p_order+1)

    QyT_mult8 = poisson_utils.tensor.ABCx_batched_bigABC(batched_gll_to_gauss_1d_tr, 
                                       batched_d_gauss_1d_tr, 
                                       batched_gll_to_gauss_1d_tr, fac_mult_8, p_order+1)

    return QxT_mult0 + QyT_mult1 + QzT_mult2 + QxT_mult3 + QyT_mult4 + QxT_mult5 + QzT_mult6 + QzT_mult7 + QyT_mult8

def mat_free_matvec(p_order, batch_size, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, vec_u, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d,
           batched_gll_to_gauss_1d, batched_d_gauss_1d,
           batched_gll_to_gauss_1d_tr, batched_d_gauss_1d_tr):
    n_elems = e_to_n.shape[0]
    batch_count = n_elems // batch_size
    remaining_count = n_elems % batch_size

    nodes_per_element  = (p_order+1)**3
    e_to_n_linear = e_to_n.flatten()
    vec_u_scattered = vec_u[e_to_n_linear]
    vec_u_scattered = jnp.reshape(vec_u_scattered, (n_elems, nodes_per_element))
    vec_u_scattered_batched = vec_u_scattered[:batch_count*batch_size]
    vec_u_scattered_batched = jnp.reshape(vec_u_scattered_batched, (batch_count, batch_size, nodes_per_element))
    vec_u_scattered_remaining = vec_u_scattered[batch_count*batch_size:]


    node_coords_x_scattered = node_coords_x[e_to_n_linear]
    node_coords_x_scattered = jnp.reshape(node_coords_x_scattered, (n_elems, nodes_per_element))
    node_coords_x_scattered_batched = node_coords_x_scattered[:batch_count*batch_size]
    node_coords_x_scattered_batched = jnp.reshape(node_coords_x_scattered_batched, (batch_count, batch_size, nodes_per_element))
    node_coords_x_scattered_remaining = node_coords_x_scattered[batch_count*batch_size:]

    node_coords_y_scattered = node_coords_y[e_to_n_linear]
    node_coords_y_scattered = jnp.reshape(node_coords_y_scattered, (n_elems, nodes_per_element))
    node_coords_y_scattered_batched = node_coords_y_scattered[:batch_count*batch_size]
    node_coords_y_scattered_batched = jnp.reshape(node_coords_y_scattered_batched, (batch_count, batch_size, nodes_per_element))
    node_coords_y_scattered_remaining = node_coords_y_scattered[batch_count*batch_size:]

    node_coords_z_scattered = node_coords_z[e_to_n_linear]
    node_coords_z_scattered = jnp.reshape(node_coords_z_scattered, (n_elems, nodes_per_element))
    node_coords_z_scattered_batched = node_coords_z_scattered[:batch_count*batch_size]
    node_coords_z_scattered_batched = jnp.reshape(node_coords_z_scattered_batched, (batch_count, batch_size, nodes_per_element))
    node_coords_z_scattered_remaining = node_coords_z_scattered[batch_count*batch_size:]

    # vmap over batches
    v_elem_stiffness_matvec_batched = jax.vmap(elem_stiffness_matvec_batched, in_axes=(None, 0, 0, 0, 0, None, None, None, None, None))
    vec_Au_scattered_batched = jax.lax.optimization_barrier(v_elem_stiffness_matvec_batched(p_order, node_coords_x_scattered_batched, node_coords_y_scattered_batched, node_coords_z_scattered_batched,
                                                               vec_u_scattered_batched,
                                                               gauss_w_3d,
                                                               batched_gll_to_gauss_1d, batched_d_gauss_1d,
                                                               batched_gll_to_gauss_1d_tr, batched_d_gauss_1d_tr))
    # vmap over remaining
    v_elem_stiffness_matvec = jax.vmap(mat_free_poisson.poisson.elem_stiffness_matvec, in_axes=(None, 0, 0, 0, 0, None, None, None))
    vec_Au_scattered_remaining = jax.lax.optimization_barrier(v_elem_stiffness_matvec(p_order, node_coords_x_scattered_remaining, node_coords_y_scattered_remaining, node_coords_z_scattered_remaining,
                                               vec_u_scattered_remaining,
                                               gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))


    vec_Au_scattered = jnp.concatenate([vec_Au_scattered_batched.ravel(), vec_Au_scattered_remaining.ravel()])
    vec_Au_scattered_sorted = jax.lax.optimization_barrier(vec_Au_scattered[e_to_n_sort_idx])
    vec_Au_zero = jnp.zeros_like(vec_u)
    vec_Au = jax.lax.optimization_barrier(vec_Au_zero.at[e_to_n_sorted].add(vec_Au_scattered_sorted, indices_are_sorted=True))

    # np.add.at(vec_Au, e_to_n_linear, vec_Au_scattered)

    # do not change values in boundary conditions
    vec_Au = jax.lax.optimization_barrier(vec_Au.at[boundary_indices].set(vec_u[boundary_indices], unique_indices=True))

    return vec_Au








def solve_poisson(p_order, batch_size, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, batched_gll_to_gauss_1d, batched_d_gauss_1d,
                  batched_gll_to_gauss_1d_tr, batched_d_gauss_1d_tr, maxiter):

    rhs_vec = jax.lax.optimization_barrier(poisson_utils.rhs.rhs(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
        gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))
    

    
    u_numeric = mat_free_batched_poisson.solver.conjugate_gradient_unrolled(p_order, batch_size, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, batched_gll_to_gauss_1d, batched_d_gauss_1d,
                  batched_gll_to_gauss_1d_tr, batched_d_gauss_1d_tr, rhs_vec, 1e-16, maxiter)
    

    return u_numeric