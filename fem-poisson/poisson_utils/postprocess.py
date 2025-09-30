import numpy as np

import poisson_utils.vtk_utils

def postprocess_poisson(all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, 
                        e_to_n, p_order, u_exact, u_numeric):
    print("L2 relative error = ", 100*np.linalg.norm((u_exact - u_numeric))/ np.linalg.norm(u_exact), "%")

    print("VTK export starting", flush=True)
    poisson_utils.vtk_utils.export_cells_to_vtk(all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, 
                                                e_to_n, p_order,
                                                np.column_stack([u_exact, u_numeric]),
                                                ["u_exact", "u_numeric"],
                                                "sample.vtk")

    print("VTK export finished!", flush=True)