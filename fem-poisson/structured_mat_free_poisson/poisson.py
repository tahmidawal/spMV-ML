import numpy as np
import jax.numpy as jnp
import jax

import poisson_utils.tensor
import poisson_utils.rhs


import structured_mat_free_poisson.solver






def elem_stiffness_matvec(p_order, const_J, rx_sy_tz, vec_u_section, 
                          gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    """
    
                | Qx Qy Qz || rx ry rz |     | rx sx tx || Qx |
    Ke =                    | sx sy sz | J W | ry sy ty || Qy |
                            | tx ty tz |     | rz sz tz || Qz |
    """

    # factor = jnp.zeros(((p_order+1)**3, 6))
    """
                  0  3  4
      factor      3  1  5
                  4  5  2
    """
    factor_0 = (rx_sy_tz *rx_sy_tz ) * const_J * gauss_w_3d # d2u/dx^2, d2u/dy^2, d2u/dz^2

    

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

    Qx_u = poisson_utils.tensor.ABCx(gll_to_gauss_1d, 
                                  gll_to_gauss_1d, 
                                  d_gauss_1d, vec_u_section, p_order+1)
    Qy_u = poisson_utils.tensor.ABCx(gll_to_gauss_1d, 
                                  d_gauss_1d, 
                                  gll_to_gauss_1d, vec_u_section, p_order+1)
    Qz_u = poisson_utils.tensor.ABCx(d_gauss_1d, 
                                  gll_to_gauss_1d, 
                                  gll_to_gauss_1d, vec_u_section, p_order+1)

    fac_mult_0 = factor_0  * Qx_u
    fac_mult_1 = factor_0  * Qy_u
    fac_mult_2 = factor_0  * Qz_u


    QxT_mult0 = poisson_utils.tensor.ABCx(gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, fac_mult_0, p_order+1)

    QyT_mult1 = poisson_utils.tensor.ABCx(gll_to_gauss_1d.T, 
                                       d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_1, p_order+1)

    QzT_mult2 = poisson_utils.tensor.ABCx(d_gauss_1d.T, 
                                       gll_to_gauss_1d.T, 
                                       gll_to_gauss_1d.T, fac_mult_2, p_order+1)

    

    return QxT_mult0 + QyT_mult1 + QzT_mult2

def mat_free_matvec(p_order, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, vec_u, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    n_elems = e_to_n.shape[0]
    nodes_per_element  = (p_order+1)**3
    e_to_n_linear = e_to_n.flatten()
    vec_u_scattered = vec_u[e_to_n_linear]
    vec_u_scattered = jnp.reshape(vec_u_scattered, (n_elems, nodes_per_element))


    v_elem_stiffness_matvec = jax.vmap(elem_stiffness_matvec, in_axes=(None, None, None, 0, None, None, None))
    vec_Au_scattered = v_elem_stiffness_matvec(p_order, const_J, rx_sy_tz, 
                                               vec_u_scattered,
                                               gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)

    vec_Au_scattered = vec_Au_scattered.flatten()
    vec_Au_scattered_sorted = vec_Au_scattered[e_to_n_sort_idx]
    vec_Au_zero = jnp.zeros_like(vec_u)
    vec_Au = vec_Au_zero.at[e_to_n_sorted].add(vec_Au_scattered_sorted, indices_are_sorted=True)

    # np.add.at(vec_Au, e_to_n_linear, vec_Au_scattered)

    # do not change values in boundary conditions
    vec_Au = vec_Au.at[boundary_indices].set(vec_u[boundary_indices], unique_indices=True)

    return vec_Au








def solve_poisson(p_order, node_coords, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, maxiter):

    rhs_vec = jax.lax.optimization_barrier(poisson_utils.rhs.rhs(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
        gauss_w_3d, gll_to_gauss_1d, d_gauss_1d))
    

    
    u_numeric = structured_mat_free_poisson.solver.conjugate_gradient_unrolled(p_order, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, rhs_vec, 1e-16, maxiter)
    

    return u_numeric