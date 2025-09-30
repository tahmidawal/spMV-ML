import numpy as np
import jax.numpy as jnp
import jax
import nvtx
import argparse

import poisson_utils.preprocess
import mat_free_poisson.poisson
import poisson_utils.postprocess




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


    
    all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n, e_to_n_sort_idx, e_to_n_sorted, bdry_indices, \
        refel, u_exact =  poisson_utils.preprocess.preprocess_poisson(p_order, 
                                                    mesh_file_path, 
                                                    surface_tags)

    u_exact = np.array(u_exact, copy=True)

    n_per_e = (p_order+1)**3


    n_pts = all_pts_gll_x.shape[0]
    n_elems = e_to_n.shape[0]

    matvec_compiled = poisson_utils.preprocess.compile_for_inputs(mat_free_poisson.poisson.mat_free_matvec, (0,), 
                                                                  p_order, 
                                                                  all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, 
                                                                  e_to_n, e_to_n_sorted, e_to_n_sort_idx, u_exact, bdry_indices,
                                                                  refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d)

    print(f"order : {p_order}\t est. (mat-free) matvec cost: {matvec_compiled.cost_analysis()['flops']}")
    

    print("compilation started..")

    nvtx.push_range("solver_compiling")
    solve_poisson_compiled = poisson_utils.preprocess.compile_for_inputs(mat_free_poisson.poisson.solve_poisson, (0,), 
                                                                         p_order, 
                                                                         all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, 
                                                                         e_to_n, e_to_n_sorted, e_to_n_sort_idx, bdry_indices,
                                                                         refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d, 2)
    nvtx.pop_range()
    print("compilation done!", flush=True)

    print("Solver Starting", flush=True)

    nvtx.push_range("solver_running")
    u_numeric = solve_poisson_compiled(all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, 
                                       e_to_n, e_to_n_sorted, e_to_n_sort_idx, bdry_indices,
                                      refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d, 500).block_until_ready()


    # force copy back to device
    u_numeric = np.array(u_numeric, copy=True)
    nvtx.pop_range()
    
    print("Solver Finished!", flush=True)

    poisson_utils.postprocess.postprocess_poisson(all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n, p_order, u_exact, u_numeric)

    print("", flush=True)
