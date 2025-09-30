import numpy as np
import vtk
import vtk.util.numpy_support as vnp

# def export_points_to_vtk(points, filename):
#     """
#     Export a set of 3D points to a VTK file with scalar values equal to point indices.
    
#     Parameters:
#     -----------
#     points : array-like
#         Array of shape (n, 3) containing (x, y, z) coordinates
#     filename : str
#         Output filename (should end with .vtk)
#     """
#     points = np.array(points)
#     n_points = len(points)
    
#     # Create VTK points object
#     vtk_points = vtk.vtkPoints()
#     for point in points:
#         vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
#     # Create polydata object
#     polydata = vtk.vtkPolyData()
#     polydata.SetPoints(vtk_points)
    
#     # Create vertices (so points are visible)
#     vertices = vtk.vtkCellArray()
#     for i in range(n_points):
#         vertices.InsertNextCell(1, [i])
#     polydata.SetVerts(vertices)
    
#     # Add scalar data (index values)
#     scalars = vtk.vtkIntArray()
#     scalars.SetName("index")
#     scalars.SetNumberOfComponents(1)
#     scalars.SetNumberOfTuples(n_points)
    
#     for i in range(n_points):
#         scalars.SetValue(i, i)
    
#     polydata.GetPointData().SetScalars(scalars)
    
#     # Write to file
#     writer = vtk.vtkPolyDataWriter()
#     writer.SetFileName(filename)
#     writer.SetInputData(polydata)
#     writer.Write()

def export_points_to_vtk_modern(points, filename):
    """
    Alternative version using vtkPolyData with modern VTK patterns.
    Includes both legacy (.vtk) and XML (.vtp) format support.
    """
    points = np.array(points)
    n_points = len(points)
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(points))
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Create vertices
    vertices = vtk.vtkCellArray()
    for i in range(n_points):
        vertices.InsertNextCell(1, [i])
    polydata.SetVerts(vertices)
    
    # Add index scalars
    index_array = vnp.numpy_to_vtk(np.arange(n_points))
    index_array.SetName("index")
    polydata.GetPointData().SetScalars(index_array)
    

    # # Choose writer based on file extension
    # if filename.endswith('.vtp'):
    #     writer = vtk.vtkXMLPolyDataWriter()
    # else:
    #     writer = vtk.vtkPolyDataWriter()

    writer = vtk.vtkPolyDataWriter()
    
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def export_cells_to_vtk(node_coords_x, node_coords_y, node_coords_z, 
                        e_to_n, p_order,
                        node_scalars, scalar_names,
                        output_filename):
    """

    Args
    ----------
        node_coords_x   : np.ndarray of shape (node_count, )
            node x coordinates
        node_coords_y   : np.ndarray of shape (node_count, )
            node y coordinates
        node_coords_z   : np.ndarray of shape (node_count, )
            node z coordinates
        e_to_n          : np.ndarray of shape (element_count, (p_order+1)^3)
            element to node mapping
        node_scalars    : np.ndarray of shape (node_count, s_count)
            scalars on each node
        scalar_names    : np.ndarray of shape (s_count, )
        output_filename : str
    """
    print("starting vtk cells export")
    vtk_points = vtk.vtkPoints()

    node_coords_xyz = np.column_stack((node_coords_x, node_coords_y, node_coords_z))

    vtk_points.SetData(vnp.numpy_to_vtk(node_coords_xyz))

    # Create a VTK unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)




    # Add scalar fields to point data
    vtk_point_data = grid.GetPointData()

    for i in range(node_scalars.shape[1]):
        data_vtk = vnp.numpy_to_vtk(node_scalars[:, i])
        data_vtk.SetName(scalar_names[i])
        vtk_point_data.AddArray(data_vtk)
    e_n = e_to_n.shape[0]
    points_per_cell = (p_order+1)**3
    points_per_face = (p_order+1)**2
    points_per_edge = (p_order+1)

    for e_i in range(e_n):
        hex_cell = vtk.vtkLagrangeHexahedron()
        hex_cell.GetPointIds().SetNumberOfIds(points_per_cell)
        for n_i in range(points_per_cell):
            i = n_i % points_per_edge
            j = (n_i // points_per_edge) % points_per_edge
            k = (n_i // points_per_face) % points_per_edge
            local_node_id = vtk.vtkLagrangeHexahedron.PointIndexFromIJK(
                        i, j, k, [p_order, p_order, p_order, points_per_cell])
            global_node_id = e_to_n[e_i, n_i]
            hex_cell.GetPointIds().SetId(local_node_id, global_node_id)
        grid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(f"{output_filename}")
    writer.SetInputData(grid)
    writer.Write()
    print(f"Mesh exported to {output_filename}")
    return