import numpy as np
import jax.numpy as jnp
import poisson_utils.preprocess
import nvtx
"""
interpolation points in 3D
corner_pts should be gmsh linear node ordering
Hexahedron:            

       v
3----------2           
|\     ^   |\          
| \    |   | \         
|  \   |   |  \        
|   7------+---6       
|   |  +-- |-- | -> u  
0---+---\--1   |       
 \  |    \  \  |       
  \ |     \  \ |       
   \|      w  \|       
    4----------5       
"""
def element_nodes(corner_pts, ref_intr_pts_1d, N):
    assert ref_intr_pts_1d.shape[0] == (N+1)
    

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(ref_intr_pts_1d, ref_intr_pts_1d, ref_intr_pts_1d, indexing='ij')
    
    # Flatten and stack - transpose to get correct order (x fastest varying)
    ref_intr_pts_3d = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])
    
    ref_corner_pts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ])
    
    interpolated_points = np.zeros(((N+1)**3, 3))
    
    # Extract reference coordinates
    xi, eta, zeta = ref_intr_pts_3d[:, 0], ref_intr_pts_3d[:, 1], ref_intr_pts_3d[:, 2]
    
    # Trilinear shape functions for hexahedron
    # N1 = (1-xi)(1-eta)(1-zeta)/8    for node (-1,-1,-1)
    # N2 = (1+xi)(1-eta)(1-zeta)/8    for node (1,-1,-1)
    # N3 = (1+xi)(1+eta)(1-zeta)/8    for node (1,1,-1)
    # N4 = (1-xi)(1+eta)(1-zeta)/8    for node (-1,1,-1)
    # N5 = (1-xi)(1-eta)(1+zeta)/8    for node (-1,-1,1)
    # N6 = (1+xi)(1-eta)(1+zeta)/8    for node (1,-1,1)
    # N7 = (1+xi)(1+eta)(1+zeta)/8    for node (1,1,1)
    # N8 = (1-xi)(1+eta)(1+zeta)/8    for node (-1,1,1)
    
    N = np.zeros(((N+1)**3, 8))
    N[:, 0] = (1 - xi) * (1 - eta) * (1 - zeta) / 8  # N1
    N[:, 1] = (1 + xi) * (1 - eta) * (1 - zeta) / 8  # N2
    N[:, 2] = (1 + xi) * (1 + eta) * (1 - zeta) / 8  # N3
    N[:, 3] = (1 - xi) * (1 + eta) * (1 - zeta) / 8  # N4
    N[:, 4] = (1 - xi) * (1 - eta) * (1 + zeta) / 8  # N5
    N[:, 5] = (1 + xi) * (1 - eta) * (1 + zeta) / 8  # N6
    N[:, 6] = (1 + xi) * (1 + eta) * (1 + zeta) / 8  # N7
    N[:, 7] = (1 - xi) * (1 + eta) * (1 + zeta) / 8  # N8
    
    # Interpolate coordinates
    for i in range(3):  # x, y, z coordinates
        interpolated_points[:, i] = np.sum(N * corner_pts[:, i], axis=1)
    
    return interpolated_points





def gen_hex_mesh_core(p_order, x_min, x_max, y_min, y_max, z_min, z_max,
                 x_elem_n, y_elem_n, z_elem_n):
    total_elem_n = x_elem_n * y_elem_n * z_elem_n
    x_node_n = (p_order + 1) * x_elem_n - (x_elem_n - 1)
    y_node_n = (p_order + 1) * y_elem_n - (y_elem_n - 1)
    z_node_n = (p_order + 1) * z_elem_n - (z_elem_n - 1)

    # === Generate Node Coordinates ===
    x_ = jnp.linspace(x_min, x_max, x_node_n)
    y_ = jnp.linspace(y_min, y_max, y_node_n)
    z_ = jnp.linspace(z_min, z_max, z_node_n)
    X, Y, Z = jnp.meshgrid(x_, y_, z_, indexing='ij')
        # # Add waviness along the depth (Z) dimension
    X += 0.2*jnp.sin(5*jnp.pi*(Z - z_min)/(z_max - z_min))

    node_coords = jnp.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])

    # === Build Element to Node Connectivity ===
    q = jnp.arange(p_order + 1)
    dx, dy, dz = jnp.meshgrid(q, q, q, indexing='ij')
    dx = dx.ravel('F')
    dy = dy.ravel('F')
    dz = dz.ravel('F')

    local_offsets = dx + dy * x_node_n + dz * x_node_n * y_node_n

    ei = jnp.arange(total_elem_n)
    ex = ei % x_elem_n
    ey = (ei // x_elem_n) % y_elem_n
    ez = ei // (x_elem_n * y_elem_n)

    base = ex * p_order + ey * p_order * x_node_n + ez * p_order * x_node_n * y_node_n
    e_to_n = base[:, None] + local_offsets[None, :]

    # === Surface Node Tags ===
    # Bottom and Top XY planes
    xy1 = jnp.arange(x_node_n * y_node_n)
    xy2 = xy1 + x_node_n * y_node_n * (z_node_n - 1)

    # XZ planes (y = 0 and y = max)
    x_range = jnp.arange(x_node_n)
    z_range = jnp.arange(z_node_n)
    xz1 = (z_range[:, None] * (x_node_n * y_node_n) + x_range).ravel()
    xz2 = xz1 + x_node_n * (y_node_n - 1)

    # YZ planes (x = 0 and x = max)
    yz1 = (z_range[:, None] * (x_node_n * y_node_n) + jnp.arange(0, x_node_n * y_node_n, x_node_n)).ravel()
    yz2 = yz1 + (x_node_n - 1)

    all_surf = jnp.concatenate([xy1, xy2, xz1, xz2, yz1, yz2])
    # surface_unique_count = 2*(x_node_n*y_node_n + x_node_n*z_node_n + y_node_n*z_node_n) - 2*8 - 4*(x_node_n + y_node_n + z_node_n - 6)
    # surface_node_tags = jnp.sort(jnp.unique(all_surf, surface_unique_count))
    # surface_node_tags = np.unique(all_surf)


    return node_coords, e_to_n, all_surf

def gen_hex_mesh(p_order, x_min, x_max, y_min, y_max, z_min, z_max,
                 x_elem_n, y_elem_n, z_elem_n):
    gen_hex_mesh_core_compiled = poisson_utils.preprocess.compile_for_inputs(gen_hex_mesh_core, 
                                                                             (0,1,2,3,4,5,6,7,8,9),
                                                                             p_order, x_min, x_max, y_min, y_max, z_min, z_max,
                                                                             x_elem_n, y_elem_n, z_elem_n)
    nvtx.push_range("meshing")
    node_coords, e_to_n, surface_node_tags_duplicated = gen_hex_mesh_core_compiled()
    surface_node_tags = np.sort(np.unique(surface_node_tags_duplicated.block_until_ready()))
    nvtx.pop_range()
    return node_coords, e_to_n, surface_node_tags

if __name__ == "__main__":
    gen_hex_mesh(3, 0, 1, 0, 1, 0, 2, 2, 3, 3)