import numpy as np
import jax.numpy as jnp
import jax
import nvtx
import argparse

import poisson_utils.preprocess
import structured_mat_free_poisson.poisson
import poisson_utils.postprocess
import poisson_utils.geometry

import sys

def const_jacobian(p_order, gll_node_coords, d_gauss_1d):
    J, D = poisson_utils.geometry.elem_geo_factors_with_inverse(p_order, gll_node_coords, d_gauss_1d)
    # print("J, ", J)
    # print("D, ", D)

    return J[0], D[1][0]        # TODO: have to manually check which D is constant


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("p_order", type=int, help="Element order for FEM")
    parser.add_argument("mesh_file_path", type=str, help="Mesh file path")
    parser.add_argument("surface_tags", type=int, nargs='+', help="Physical group tags of boundary surfaces")

    args = parser.parse_args()

    p_order         = args.p_order
    mesh_file_path  = args.mesh_file_path
    surface_tags    = args.surface_tags


    print("p_order:", p_order)
    print("mesh_file_path:", mesh_file_path)
    print("surface_tags:", surface_tags)


    
    all_pts_gll, e_to_n, e_to_n_sort_idx, e_to_n_sorted, bdry_indices, \
        refel, u_exact =  poisson_utils.preprocess.preprocess_poisson(p_order, 
                                                    mesh_file_path, 
                                                    surface_tags)

    u_exact = np.array(u_exact, copy=True)

    n_per_e = (p_order+1)**3

    all_pts_gll = all_pts_gll
    n_pts = all_pts_gll.shape[0]
    n_elems = e_to_n.shape[0]

    # use elem_0 to get const jacobian
    elem_0_node_coords = all_pts_gll[e_to_n[0]]
    # elem_0_node_coords = jnp.reshape(elem_0_node_coords, (1, n_per_e, 3))
    # print(elem_0_node_coords)

    const_J, rx_sy_tz = const_jacobian(p_order, elem_0_node_coords, refel.d_gauss_1d)

    # sys.exit(0)

    matvec_compiled = poisson_utils.preprocess.compile_for_inputs(structured_mat_free_poisson.poisson.mat_free_matvec, (0,), p_order, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, u_exact, bdry_indices,
                                      refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d)

    print(f"order : {p_order}\t est. (structured-mat-free) matvec cost: {matvec_compiled.cost_analysis()['flops']}")
    

    print("compilation started..")

    nvtx.push_range("solver_compiling")
    solve_poisson_compiled = poisson_utils.preprocess.compile_for_inputs(structured_mat_free_poisson.poisson.solve_poisson, (0,), p_order, all_pts_gll, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, bdry_indices,
                                      refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d, 2)
    nvtx.pop_range()
    print("compilation done!", flush=True)

    print("Solver Starting", flush=True)

    nvtx.push_range("solver_running")
    u_numeric = solve_poisson_compiled(all_pts_gll, const_J, rx_sy_tz, e_to_n, e_to_n_sorted, e_to_n_sort_idx, bdry_indices,
                                      refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d, 500).block_until_ready()


    # force copy back to device
    u_numeric = np.array(u_numeric, copy=True)
    nvtx.pop_range()
    
    print("Solver Finished!", flush=True)

    poisson_utils.postprocess.postprocess_poisson(all_pts_gll, e_to_n, p_order, u_exact, u_numeric)

    print("", flush=True)
