import jax.numpy as jnp
import numpy as np

import poisson_utils.tensor

def transformation_1d(from_x, to_x):
    """
    Create 1D transformation matrix from from_x nodes to to_x nodes
    using Lagrange interpolation
    """
    n_from = len(from_x)
    n_to = len(to_x)
    T = np.zeros((n_to, n_from))

    for i in range(n_to):
        for j in range(n_from):
            # Evaluate jth Lagrange polynomial at ith n_to point
            L_val = 1.0
            for k in range(n_from):
                if k != j:
                    L_val *= (to_x[i] - from_x[k]) / (from_x[j] - from_x[k])
            T[i, j] = L_val
    return T


def transform_elem_coords(p_order, 
                          coords_x, coords_y, coords_z, 
                          transformation_1d):
    output_coords_x = poisson_utils.tensor.ABCx(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_x, 
                                             p_order+1 )
    
    output_coords_y = poisson_utils.tensor.ABCx(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_y, 
                                             p_order+1 )
    output_coords_z = poisson_utils.tensor.ABCx(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_z, 
                                             p_order+1 )
    
    return output_coords_x, output_coords_y, output_coords_z


def transform_elem_coords_batched(p_order, coords_x_batched, coords_y_batched, coords_z_batched, 
                                  transformation_1d_batched):
    batch = coords_x_batched.shape[0]
    output_coords_x = poisson_utils.tensor.ABCx_batched_bigABC(transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             coords_x_batched, 
                                             p_order+1)
    
    output_coords_y = poisson_utils.tensor.ABCx_batched_bigABC(transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             coords_y_batched, 
                                             p_order+1)
    output_coords_z = poisson_utils.tensor.ABCx_batched_bigABC(transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             transformation_1d_batched, 
                                             coords_z_batched, 
                                             p_order+1)
    
    return output_coords_x, output_coords_y, output_coords_z


def elem_geo_factors(p_order, 
                     gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, 
                     d_gauss_1d):
    """
    returns |J| for gauss-legendre quadrature points
    """

    xr, xs, xt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_x, p_order+1)
    yr, ys, yt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_y, p_order+1)
    zr, zs, zt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_z, p_order+1)

    J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
    return J


def elem_geo_factors_with_inverse(p_order, gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, d_gauss_1d):
    """
    returns |J| and inverse J for gauss-legendre quadrature points
    """

    xr, xs, xt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_x, p_order+1)
    yr, ys, yt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_y, p_order+1)
    zr, zs, zt = poisson_utils.tensor.grad3(d_gauss_1d, gll_node_coords_z, p_order+1)

    J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
    # D = jnp.zeros((9, (p_order+1)**3))

    """
    | rx ry rz |
    | sx sy sz |
    | tx ty tz |

    D[0] = rx   D[1] = ry   D[2] = rz
    D[3] = sx   D[4] = sy   D[5] = sz
    D[6] = tx   D[7] = ty   D[8] = tz

    """


    D_0 =  (ys*zt - zs*yt)/J
    D_1 = -(xs*zt - zs*xt)/J
    D_2 =  (xs*yt - ys*xt)/J
    
    D_3 = -(yr*zt - zr*yt)/J
    D_4 =  (xr*zt - zr*xt)/J
    D_5 = -(xr*yt - yr*xt)/J
    
    D_6 =  (yr*zs - zr*ys)/J
    D_7 = -(xr*zs - zr*xs)/J
    D_8 =  (xr*ys - yr*xs)/J

    return J, jnp.vstack([D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8])


def elem_geo_factors_with_inverse_batched(p_order, 
                                          gll_node_coords_x_batched, gll_node_coords_y_batched, gll_node_coords_z_batched,
                                          batched_d_gauss_1d):
    """
    returns |J| and inverse J for gauss-legendre quadrature points
    """
    batch = gll_node_coords_x_batched.shape[0]

    xr, xs, xt = poisson_utils.tensor.grad3_batched_bigA(batched_d_gauss_1d, gll_node_coords_x_batched, p_order+1)
    yr, ys, yt = poisson_utils.tensor.grad3_batched_bigA(batched_d_gauss_1d, gll_node_coords_y_batched, p_order+1)
    zr, zs, zt = poisson_utils.tensor.grad3_batched_bigA(batched_d_gauss_1d, gll_node_coords_z_batched, p_order+1)

    J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
    # D = jnp.zeros((9, (p_order+1)**3))

    """
    | rx ry rz |
    | sx sy sz |
    | tx ty tz |

    D[0] = rx   D[1] = ry   D[2] = rz
    D[3] = sx   D[4] = sy   D[5] = sz
    D[6] = tx   D[7] = ty   D[8] = tz

    """


    D_0 =  (ys*zt - zs*yt)/J
    D_1 = -(xs*zt - zs*xt)/J
    D_2 =  (xs*yt - ys*xt)/J
    
    D_3 = -(yr*zt - zr*yt)/J
    D_4 =  (xr*zt - zr*xt)/J
    D_5 = -(xr*yt - yr*xt)/J
    
    D_6 =  (yr*zs - zr*ys)/J
    D_7 = -(xr*zs - zr*xs)/J
    D_8 =  (xr*ys - yr*xs)/J

    return J, jnp.stack([D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8], axis=0)




# ================== v2 functions =========

def transform_coords_v2(coords_x, coords_y, coords_z, 
                        transformation_1d):
    output_coords_x = poisson_utils.tensor.ABC_xs_v2(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_x)
    
    output_coords_y = poisson_utils.tensor.ABC_xs_v2(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_y)
    output_coords_z = poisson_utils.tensor.ABC_xs_v2(transformation_1d, 
                                             transformation_1d, 
                                             transformation_1d, 
                                             coords_z)
    
    return output_coords_x, output_coords_y, output_coords_z

def geo_factors_with_inverse_v2(gll_node_coords_x, gll_node_coords_y, gll_node_coords_z, d_gauss_1d):
    """
    returns |J| and inverse J for gauss-legendre quadrature points
    """

    xr, xs, xt = poisson_utils.tensor.grad3_v2(d_gauss_1d, gll_node_coords_x)
    yr, ys, yt = poisson_utils.tensor.grad3_v2(d_gauss_1d, gll_node_coords_y)
    zr, zs, zt = poisson_utils.tensor.grad3_v2(d_gauss_1d, gll_node_coords_z)

    J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
    # D = jnp.zeros((9, (p_order+1)**3))

    """
    | rx ry rz |
    | sx sy sz |
    | tx ty tz |

    D[0] = rx   D[1] = ry   D[2] = rz
    D[3] = sx   D[4] = sy   D[5] = sz
    D[6] = tx   D[7] = ty   D[8] = tz

    """


    D_0 =  (ys*zt - zs*yt)/J
    D_1 = -(xs*zt - zs*xt)/J
    D_2 =  (xs*yt - ys*xt)/J
    
    D_3 = -(yr*zt - zr*yt)/J
    D_4 =  (xr*zt - zr*xt)/J
    D_5 = -(xr*yt - yr*xt)/J
    
    D_6 =  (yr*zs - zr*ys)/J
    D_7 = -(xr*zs - zr*xs)/J
    D_8 =  (xr*ys - yr*xs)/J

    return J, D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8