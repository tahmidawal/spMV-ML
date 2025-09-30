import jax

import jax.experimental.sparse
import jax.numpy as jnp
from scipy.sparse import csr_matrix
import nvtx

import global_assembl_poisson.solver
import block_assembl_poisson.poisson

import poisson_utils.rhs
import poisson_utils.preprocess


def assemble_global(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n,
                    gauss_w_3d, gll_to_gauss_1d, d_gauss_1d):
    
    nodes_per_element = (p_order+1)**3
    n_pts = node_coords_x.shape[0]


    assemble_block_matrices_compiled = \
        poisson_utils.preprocess.compile_for_inputs(block_assembl_poisson.poisson.assemble_block_matrices, (0,), 
                                                    p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n,
                                      gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    nvtx.push_range("assembling_global")
    block_matrices = assemble_block_matrices_compiled(node_coords_x, node_coords_y, node_coords_z, e_to_n,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    
    # block_matrices = block_assembl_poisson.poisson.assemble_block_matrices(p_order, node_coords, e_to_n,
    #                                   gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    

    sparse_rows = jnp.repeat(e_to_n, nodes_per_element, axis=1).flatten()
    sparse_cols = jnp.repeat(e_to_n, nodes_per_element, axis=0).flatten()
    sparse_mat = csr_matrix((block_matrices.flatten().block_until_ready(), 
                                (sparse_rows.block_until_ready(), sparse_cols.block_until_ready())), 
                            shape=(n_pts, n_pts))
    
    csr_row_ptrs = sparse_mat.indptr
    csr_cols = sparse_mat.indices
    csr_vals = sparse_mat.data


    jax_csr_mat = jax.experimental.sparse.CSR(( csr_vals, 
                                            csr_cols, 
                                            csr_row_ptrs), 
                                          shape=(n_pts, n_pts)).block_until_ready()
    nvtx.pop_range()
    return jax_csr_mat



def sparse_matvec(csr_mat, vec_u, boundary_indices):
    vec_Au = jax.experimental.sparse.csr_matvec(csr_mat, vec_u)

    # do not change values in boundary conditions
    vec_Au = vec_Au.at[boundary_indices].set(vec_u[boundary_indices], unique_indices=True)

    return vec_Au



def solve_poisson(p_order, csr_mat, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, maxiter):

    rhs_vec = poisson_utils.rhs.rhs(p_order, node_coords_x, node_coords_y, node_coords_z, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
        gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    
    

    
    u_numeric = global_assembl_poisson.solver.conjugate_gradient_unrolled(csr_mat, boundary_indices,
                  rhs_vec, 1e-16, maxiter)
    

    return u_numeric