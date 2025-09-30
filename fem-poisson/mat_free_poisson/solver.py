import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, cg
import jax
import jax.numpy as jnp
import mat_free_poisson.poisson

def scipy_gmres(matvec, b, n_dof, guess, tol=1e-8):
    A_linop = LinearOperator((n_dof,n_dof), matvec=matvec, dtype=np.float32)
    cb = lambda x_k: print(np.linalg.norm(b - matvec(x_k), np.inf))
    x, info = gmres(A_linop, b, guess, rtol=tol, callback=cb, callback_type="x")
    print(info)
    return x

def scipy_cg(matvec, b, n_dof, guess, tol=1e-8):
    A_linop = LinearOperator((n_dof,n_dof), matvec=matvec, dtype=np.float32)
    cb = lambda x_k: print(np.linalg.norm(b - matvec(x_k), np.inf))
    x, info = cg(A_linop, b, guess, rtol=tol, callback=cb)
    print(info)
    return x

# def conjugate_gradient(matvec_func, b, x0=None, tol=1e-8, maxiter=None):
#     """
#     Solve A x = b using matrix-free Conjugate Gradient.

#     Parameters:
#         matvec_func : function
#             A function that computes A @ x for any x.
#         b : ndarray
#             Right-hand side vector.
#         x0 : ndarray or None
#             Initial guess (if None, uses zero vector).
#         tol : float
#             Convergence tolerance on residual norm.
#         maxiter : int or None
#             Maximum number of iterations (if None, defaults to len(b)).

#     Returns:
#         x : ndarray
#             Approximate solution.
#     """
#     n = b.shape[0]
#     if x0 is None:
#         x = np.zeros_like(b)
#     else:
#         x = x0.copy()

#     r = b - matvec_func(x)
#     p = r.copy()
#     rs_old = np.dot(r, r)

#     if maxiter is None:
#         maxiter = n

#     for i in range(maxiter):
#         Ap = matvec_func(p)
#         alpha = rs_old / np.dot(p, Ap)
#         x += alpha * p
#         r -= alpha * Ap
#         rs_new = np.dot(r, r)
#         print("iter:", i+1,"\tres:", np.sqrt(rs_new))

#         if np.sqrt(rs_new) < tol:
#             print(f'Converged in {i+1} iterations.')
#             break

#         beta = rs_new / rs_old
#         p = r + beta * p
#         rs_old = rs_new

#     return x



# def conjugate_gradient(matvec_func, b, tol, maxiter):
#     """
#     Matrix-free Conjugate Gradient in pure JAX (compilable).
#     """
#     # n = b.shape[0]
#     x0 = jnp.zeros_like(b)

#     def cond_fun(state):
#         i, rs_new, *_ = state
#         return jnp.logical_and(i < maxiter, jnp.sqrt(rs_new) >= tol)

#     def body_fun(state):
#         i, rs_old, x, r, p = state

#         Ap = matvec_func(p)
#         alpha = rs_old / jnp.dot(p, Ap)
#         x_new = x + alpha * p
#         r_new = r - alpha * Ap
#         rs_new = jnp.dot(r_new, r_new)
#         beta = rs_new / rs_old
#         p_new = r_new + beta * p
#         return (i+1, rs_new, x_new, r_new, p_new)

#     # Initial residual
#     r0 = b - matvec_func(x0)
#     p0 = r0
#     rs0 = jnp.dot(r0, r0)

#     init_state = (0, rs0, x0, r0, p0)
#     final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
#     _, _, x_final, _, _ = final_state
#     return x_final



def conjugate_gradient(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, b, tol, maxiter):
    """
    Matrix-free Conjugate Gradient in pure JAX (compilable).
    """
    # n = b.shape[0]
    x0 = jnp.zeros_like(b)

    def cond_fun(state):
        i, rs_new, *_ = state
        return jnp.logical_and(i < maxiter, jnp.sqrt(rs_new) >= tol)

    def body_fun(state):
        i, rs_old, x, r, p = state

        Ap = mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, p, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
        alpha = rs_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return (i+1, rs_new, x_new, r_new, p_new)

    # Initial residual
    r0 = b - mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, x0, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    p0 = r0
    rs0 = jnp.dot(r0, r0)

    init_state = (0, rs0, x0, r0, p0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, x_final, _, _ = final_state
    return x_final


def conjugate_gradient_fixed(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, b, tol, maxiter):
    """
    Matrix-free Conjugate Gradient in pure JAX (compilable).
    """
    # n = b.shape[0]
    x0 = jnp.zeros_like(b)

    def cond_fun(state):
        i, rs_new, *_ = state
        return jnp.logical_and(i < maxiter, jnp.sqrt(rs_new) >= tol)

    def body_fun(state):
        i, rs_old, x, r, p = state

        Ap = mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, p, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
        alpha = rs_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return (i+1, rs_new, x_new, r_new, p_new)

    # Initial residual
    r0 = b - mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords, e_to_n, e_to_n_sorted, e_to_n_sort_idx, x0, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    p0 = r0
    rs0 = jnp.dot(r0, r0)

    init_state = (0, rs0, x0, r0, p0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, x_final, _, _ = final_state
    return x_final



def conjugate_gradient_unrolled(p_order, node_coords_x, node_coords_y, node_coords_z, 
                                e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                                gauss_w_3d, gll_to_gauss_1d, d_gauss_1d, b, tol, maxiter):

    x0 = jnp.zeros_like(b)

    # Initial residual
    r0 = b - mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords_x, node_coords_y, node_coords_z, 
                                                      e_to_n, e_to_n_sorted, e_to_n_sort_idx, x0, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
    p0 = r0
    rs0 = jnp.dot(r0, r0)

    def single_step(i, rs_old, x, r, p):
        Ap = mat_free_poisson.poisson.mat_free_matvec(p_order, node_coords_x, node_coords_y, node_coords_z, 
                                                      e_to_n, e_to_n_sorted, e_to_n_sort_idx, p, boundary_indices,
           gauss_w_3d, gll_to_gauss_1d, d_gauss_1d)
        alpha = rs_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return i+1, rs_new, x_new, r_new, p_new

    def body_fun(state):
        i, rs_old, x, r, p = state

        # unroll
        i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)
        # i, rs_old, x, r, p = single_step(i, rs_old, x, r, p)

        return (i, rs_old, x, r, p)

    def cond_fun(state):
        i, rs_new, *_ = state
        return jnp.logical_and(i < maxiter, jnp.sqrt(rs_new) >= tol)

    init_state = (0, rs0, x0, r0, p0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, x_final, _, _ = final_state
    return x_final