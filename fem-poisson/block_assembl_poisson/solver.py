
import numpy as np
import jax
import jax.numpy as jnp
import block_assembl_poisson.poisson


def conjugate_gradient_unrolled(p_order, block_matrices, e_to_n, e_to_n_sorted, e_to_n_sort_idx, boundary_indices,
                  b, tol, maxiter):

    x0 = jnp.zeros_like(b)

    # Initial residual
    r0 = b - block_assembl_poisson.poisson.blocked_matvec(p_order, block_matrices, e_to_n, e_to_n_sorted, e_to_n_sort_idx, x0, boundary_indices)
    p0 = r0
    rs0 = jnp.dot(r0, r0)

    def single_step(i, rs_old, x, r, p):
        Ap = block_assembl_poisson.poisson.blocked_matvec(p_order, block_matrices, e_to_n, e_to_n_sorted, e_to_n_sort_idx, p, boundary_indices)
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