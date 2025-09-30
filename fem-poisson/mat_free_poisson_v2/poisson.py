import numpy as np
import jax.numpy as jnp
import jax

import poisson_utils.tensor
import poisson_utils.user_funcs
import poisson_utils.geometry
import poisson_utils.rhs


import mat_free_poisson_v2.solver

v_coeff_func = jax.vmap(poisson_utils.user_funcs.coeff_func, [0, 0, 0])





def stiffness_matvec(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, vec_u_scattered, 
                     gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    n = p_order + 1
    """
    
                | Qx Qy Qz || rx ry rz |     | rx sx tx || Qx |
    Ke =                    | sx sy sz | J W | ry sy ty || Qy |
                            | tx ty tz |     | rz sz tz || Qz |
    """
    J, D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8 = \
        poisson_utils.geometry.geo_factors_with_inverse_v2(gll_node_coords_x, 
                                                           gll_node_coords_y, 
                                                           gll_node_coords_z, 
                                                           d_gauss_1d)
    gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z = \
        poisson_utils.geometry.transform_coords_v2(gll_node_coords_x, 
                                                   gll_node_coords_y, 
                                                   gll_node_coords_z,
                                                   gll_to_gauss_1d)
    
    

    mu = v_coeff_func(gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z)

    # factor = jnp.zeros(((p_order+1)**3, 6))
    """
                  0  3  4
      factor      3  1  5
                  4  5  2
    """
    factor_0 = (((D_0*D_0 + D_1*D_1 + D_2*D_2 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dx^2
    factor_1 = (((D_3*D_3 + D_4*D_4 + D_5*D_5 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dy^2
    factor_2 = (((D_6*D_6 + D_7*D_7 + D_8*D_8 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dz^2
    
    factor_3 = (((D_0*D_3 + D_1*D_4 + D_2*D_5 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dxdy
    factor_4 = (((D_0*D_6 + D_1*D_7 + D_2*D_8 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dxdz
    factor_5 = (((D_3*D_6 + D_4*D_7 + D_5*D_8 ) * J * mu).reshape(-1, n**3) * gauss_w_3d).flatten()  # d2u/dydz

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

    Qx_u = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d, 
                                  gll_to_gauss_1d, 
                                  d_gauss_1d, vec_u_scattered)
    Qy_u = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d, 
                                  d_gauss_1d, 
                                  gll_to_gauss_1d, vec_u_scattered)
    Qz_u = poisson_utils.tensor.ABC_xs_v2(d_gauss_1d, 
                                  gll_to_gauss_1d, 
                                  gll_to_gauss_1d, vec_u_scattered)

    fac_mult_0 = factor_0  * Qx_u
    fac_mult_1 = factor_1  * Qy_u
    fac_mult_2 = factor_2  * Qz_u
    fac_mult_3 = factor_3  * Qy_u
    fac_mult_4 = factor_3  * Qx_u
    fac_mult_5 = factor_4  * Qz_u
    fac_mult_6 = factor_4  * Qx_u
    fac_mult_7 = factor_5  * Qy_u
    fac_mult_8 = factor_5  * Qz_u

    QxT_mult0 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, fac_mult_0)

    QyT_mult1 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_1)

    QzT_mult2 = poisson_utils.tensor.ABC_xs_v2(d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_2)

    QxT_mult3 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, fac_mult_3)

    QyT_mult4 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_4)

    QxT_mult5 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, fac_mult_5)

    QzT_mult6 = poisson_utils.tensor.ABC_xs_v2(d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_6)

    QzT_mult7 = poisson_utils.tensor.ABC_xs_v2(d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_7)

    QyT_mult8 = poisson_utils.tensor.ABC_xs_v2(gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_8)

    return QxT_mult0 + QyT_mult1 + QzT_mult2 + QxT_mult3 + QyT_mult4 + QxT_mult5 + QzT_mult6 + QzT_mult7 + QyT_mult8

def mat_free_matvec(p_order, node_coords_x, node_coords_y, node_coords_z, 
                    e_to_n, e_to_n_sorted, e_to_n_sort_idx, vec_u, boundary_indices,
                    gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    n_elems = e_to_n.shape[0]
    nodes_per_element  = (p_order+1)**3
    e_to_n_linear = e_to_n.flatten()
    vec_u_scattered = vec_u[e_to_n_linear]

    node_coords_x_scattered = node_coords_x[e_to_n_linear]
    node_coords_y_scattered = node_coords_y[e_to_n_linear]
    node_coords_z_scattered = node_coords_z[e_to_n_linear]

    vec_Au_scattered = jax.lax.optimization_barrier(stiffness_matvec(p_order,
                                                                     node_coords_x_scattered, node_coords_y_scattered, node_coords_z_scattered,
                                                                     vec_u_scattered,
                                                                     gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))

    vec_Au_scattered_sorted = jax.lax.optimization_barrier(vec_Au_scattered[e_to_n_sort_idx])
    vec_Au_zero = jnp.zeros_like(vec_u)
    vec_Au = jax.lax.optimization_barrier(vec_Au_zero.at[e_to_n_sorted].add(vec_Au_scattered_sorted, indices_are_sorted=True))

    # np.add.at(vec_Au, e_to_n_linear, vec_Au_scattered)

    # do not change values in boundary conditions
    vec_Au = jax.lax.optimization_barrier(vec_Au.at[boundary_indices].set(vec_u[boundary_indices], unique_indices=True))

    return vec_Au








def solve_poisson(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, maxiter):

    rhs_vec = jax.lax.optimization_barrier(poisson_utils.rhs.rhs(p_order, node_coords_x, node_coords_y, node_coords_z, 
                                                                 e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
        gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))
    

    
    u_numeric = mat_free_poisson_v2.solver.conjugate_gradient_unrolled(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, rhs_vec, 1e-16, maxiter)
    

    return u_numeric