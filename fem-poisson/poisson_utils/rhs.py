import jax.numpy as jnp
import jax

import poisson_utils.tensor
import poisson_utils.user_funcs
import poisson_utils.geometry

v_source_func = jax.vmap(poisson_utils.user_funcs.source_func, [0, 0, 0])


def elem_rhs(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, 
             gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    
    J = poisson_utils.geometry.elem_geo_factors(p_order, 
                                                gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, 
                                                d_gauss_1d)
    
    gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z = \
        poisson_utils.geometry.transform_elem_coords(p_order, 
                                                     gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, 
                                                     gll_to_gauss_1d)
    source_f_vals = v_source_func(gauss_node_coords_x, gauss_node_coords_y, gauss_node_coords_z)
    Jd = gauss_w_3d * J * source_f_vals

    return poisson_utils.tensor.ABCx(gll_to_gauss_1d.T, 
                                     gll_to_gauss_1d.T, 
                                     gll_to_gauss_1d.T, Jd, p_order+1)

def rhs(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
        gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    n_elems = e_to_n.shape[0]
    n_nodes = node_coords_x.shape[0]
    nodes_per_element  = (p_order+1)**3
    
    e_to_n_linear = e_to_n.flatten()

    node_coords_x_scattered = node_coords_x[e_to_n_linear]
    node_coords_x_scattered = jnp.reshape(node_coords_x_scattered, (n_elems, nodes_per_element))
    node_coords_y_scattered = node_coords_y[e_to_n_linear]
    node_coords_y_scattered = jnp.reshape(node_coords_y_scattered, (n_elems, nodes_per_element))
    node_coords_z_scattered = node_coords_z[e_to_n_linear]
    node_coords_z_scattered = jnp.reshape(node_coords_z_scattered, (n_elems, nodes_per_element))


    v_elem_rhs = jax.vmap(elem_rhs, in_axes=(None, 0, 0, 0, None, None, None))
    rhs_vec_scattered = v_elem_rhs(p_order, 
                                   node_coords_x_scattered, node_coords_y_scattered, node_coords_z_scattered, 
                                   gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)


    rhs_vec_scattered = rhs_vec_scattered.flatten()
    rhs_vec_scattered_sorted = rhs_vec_scattered[e_to_n_sort_idx]

    rhs_vec_zero = jnp.zeros((n_nodes,))
    
    rhs_vec = rhs_vec_zero.at[e_to_n_sorted].add(rhs_vec_scattered_sorted, indices_are_sorted=True)

    # np.add.at(rhs_vec, e_to_n_linear, rhs_vec_scattered)

    v_boundary_func = jax.vmap(poisson_utils.user_funcs.boundary_func, [0, 0, 0])
    bdry_node_coords_x = node_coords_x[boundary_indices]
    bdry_node_coords_y = node_coords_y[boundary_indices]
    bdry_node_coords_z = node_coords_z[boundary_indices]

    rhs_vec = rhs_vec.at[boundary_indices].set(v_boundary_func(bdry_node_coords_x, bdry_node_coords_y, bdry_node_coords_z),
                                               unique_indices=True)
    # rhs_vec[boundary_indices] = 0


    return rhs_vec