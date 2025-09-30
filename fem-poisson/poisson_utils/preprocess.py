import poisson_utils.gmsh_utils
import poisson_utils.geometry
import poisson_utils.refel
import poisson_utils.user_funcs
import poisson_utils.hexmesh
import numpy as np
import jax.numpy as jnp

import jax
import os



BACKEND = 'cpu'  # Changed from 'gpu' to 'cpu' for compatibility

# def lagrange_derivative(l_pts, d_pts):
#     l_n = len(l_pts)
#     d_n = len(d_pts)
#     D = np.zeros((l_n, d_n))

#     for i in range(l_n):
#         for j in range(d_n):
#             L = 1.0
#             mult = 0
#             for k in range(l_n):
#                 if i!=k:
#                     L*= (d_pts[j] - l_pts[k]) / (l_pts[i] - l_pts[k])
#                     mult += 1/ (d_pts[j] - l_pts[k])
#             D[i, j] = L*mult

#     return D.T


def gmsh_coords_to_gll_coords(p_order, 
                              gmsh_coords_x, gmsh_coords_y, gmsh_coords_z , 
                              e_to_n, transformation_1d):
    n_elems = e_to_n.shape[0]
    nodes_per_element  = (p_order+1)**3

    e_to_n_linear = e_to_n.flatten()
    
    gll_coords_x_scattered = gmsh_coords_x[e_to_n_linear]
    gll_coords_x_scattered = jnp.reshape(gll_coords_x_scattered, (n_elems, nodes_per_element))
    gll_coords_y_scattered = gmsh_coords_y[e_to_n_linear]
    gll_coords_y_scattered = jnp.reshape(gll_coords_y_scattered, (n_elems, nodes_per_element))
    gll_coords_z_scattered = gmsh_coords_z[e_to_n_linear]
    gll_coords_z_scattered = jnp.reshape(gll_coords_z_scattered, (n_elems, nodes_per_element))

    v_transform_elem_coords = jax.vmap(poisson_utils.geometry.transform_elem_coords, in_axes=(None, 0, 0, 0, None))

    gll_coords_x_scattered, gll_coords_y_scattered, gll_coords_z_scattered = \
        v_transform_elem_coords(p_order, 
                                gll_coords_x_scattered, gll_coords_y_scattered, gll_coords_z_scattered, 
                                transformation_1d)


    gll_coords_x_scattered = jnp.reshape(gll_coords_x_scattered, (n_elems * nodes_per_element, ))
    gll_coords_y_scattered = jnp.reshape(gll_coords_y_scattered, (n_elems * nodes_per_element, ))
    gll_coords_z_scattered = jnp.reshape(gll_coords_z_scattered, (n_elems * nodes_per_element, ))


    gll_coords_x_zero = jnp.zeros_like(gmsh_coords_x)
    gll_coords_y_zero = jnp.zeros_like(gmsh_coords_y)
    gll_coords_z_zero = jnp.zeros_like(gmsh_coords_z)



    # there are duplicates but duplicates are equivalent
    gll_coords_x = gll_coords_x_zero.at[e_to_n_linear].set(gll_coords_x_scattered)
    gll_coords_y = gll_coords_y_zero.at[e_to_n_linear].set(gll_coords_y_scattered)
    gll_coords_z = gll_coords_z_zero.at[e_to_n_linear].set(gll_coords_z_scattered)


    return gll_coords_x, gll_coords_y, gll_coords_z


def compute_u_exact(gll_coords_x, gll_coords_y, gll_coords_z):
    v_u_sol_func = jax.vmap(poisson_utils.user_funcs.u_sol_func, in_axes=(0, 0, 0))

    u_exact = v_u_sol_func(gll_coords_x, gll_coords_y, gll_coords_z)

    return u_exact




def compile_for_inputs(func, static_argnums, *args):
    return jax.jit(func, static_argnums=static_argnums, backend=BACKEND).trace(*args).lower().compile()
    # return jax.jit(func, static_argnums=static_argnums, backend=BACKEND)


def sort_e_to_n(e_to_n):
    e_to_n_flattened = e_to_n.flatten()
    e_to_n_sort_idx = np.argsort(e_to_n_flattened)
    e_to_n_sorted = e_to_n_flattened[e_to_n_sort_idx]
    return e_to_n_sort_idx, e_to_n_sorted

def preprocess_poisson(p_order, mesh_file_path, surface_tags):
    
    jax.config.update("jax_default_device", jax.devices(BACKEND)[0])
    jax.config.update('jax_compiler_enable_remat_pass', False)

    jax.config.update("jax_compilation_cache_dir", f"{os.getcwd()}/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    jax.config.update('jax_default_matmul_precision', 'float32')



    all_pts, e_to_n, bdry_indices = poisson_utils.gmsh_utils.read_hex_mesh(mesh_file_path, p_order, surface_tags)
    
    all_pts_x = np.copy(all_pts[:, 0])
    all_pts_y = np.copy(all_pts[:, 1])
    all_pts_z = np.copy(all_pts[:, 2])


    # some artificial transformation
    # all_pts[:, 1] = all_pts[:, 1] + 0.08*np.sin(8*all_pts[:, 0])



    ordering_idx = poisson_utils.gmsh_utils.gmsh_to_XYZ_order(p_order)

    e_to_n = e_to_n[:, ordering_idx]
    # all_pts, e_to_n, bdry_indices = poisson_utils.hexmesh.gen_hex_mesh(p_order, -1, 1, -1, 1, -2, 2, 10, 10, 25)

    # we can put sorting inside solve_poisson kernel but then the jax persistent cache fails
    e_to_n_sort_idx, e_to_n_sorted = sort_e_to_n(e_to_n)

    refel = poisson_utils.refel.Refel(p_order)

    gmsh_to_gll_1d = poisson_utils.geometry.transformation_1d(np.linspace(-1, 1, p_order + 1), refel.gll_x_1d)

    gmsh_coords_to_gll_coords_compiled = compile_for_inputs(gmsh_coords_to_gll_coords, (0,), 
                                                            p_order, all_pts_x, all_pts_y, all_pts_z, e_to_n, gmsh_to_gll_1d)


    all_pts_gll_x, all_pts_gll_y, all_pts_gll_z = gmsh_coords_to_gll_coords_compiled(all_pts_x, all_pts_y, all_pts_z, 
                                                                                     e_to_n, gmsh_to_gll_1d)

    all_pts_gll_x.block_until_ready()
    all_pts_gll_y.block_until_ready()
    all_pts_gll_z.block_until_ready()


    compute_u_exact_compiled = compile_for_inputs(compute_u_exact, [], all_pts_gll_x, all_pts_gll_y, all_pts_gll_z)

    u_exact = compute_u_exact_compiled(all_pts_gll_x, all_pts_gll_y, all_pts_gll_z).block_until_ready()

    return all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n, e_to_n_sort_idx, e_to_n_sorted, bdry_indices, refel, u_exact