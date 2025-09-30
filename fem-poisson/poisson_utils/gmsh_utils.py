import gmsh
import numpy as np




order_to_elem_type = {
    1: 5,
    2: 12,
    3: 92,
    4: 93,
    5: 94,
    6: 95,
    7: 96,
    8: 97,
    9: 98
}

def read_hex_mesh(mesh_file, order, boundary_surface_physical_tags):
    """
    reads a 3D hex mesh, converts to given polynomial order
    """
    
    gmsh.initialize()
    

    gmsh.open(mesh_file)

    gmsh.model.mesh.setOrder(order)
    
    gmsh.model.geo.synchronize()


    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()


    # Create mapping from gmsh node tags to array indices
    node_tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
    
    # Reshape coordinates (gmsh returns flattened array)
    node_coords = node_coords.reshape(-1, 3)
    
    
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(3)
    

    hex_node_tags = None
    
    for i, elem_type in enumerate(element_types):
        if elem_type == order_to_elem_type[order]:  # 8-node hexahedron
            hex_node_tags = element_node_tags[i]
            
            # Reshape to get element-to-node connectivity
            hex_node_tags = hex_node_tags.reshape(-1, (order+1)**3)
            
            break
    
    if hex_node_tags is None:
        raise ValueError("No hexahedral element nodes found in the mesh file")
    

    # extract node tags of boundary surface

    surface_node_tags = set()

    for b_t in boundary_surface_physical_tags:
        dim = 2
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, b_t)

        for entity_tag in entity_tags:
            
            _, _, elem_node_tags = gmsh.model.mesh.getElements(dim, entity_tag)
            
            for node_list in elem_node_tags:
                surface_node_tags.update(node_list) 

    # _, _, elem_node_tags = gmsh.model.mesh.getElements(2, 0)
    
    # for node_list in elem_node_tags:
    #     surface_node_tags.update(node_list) 

    surface_node_indices = []

    for s_n in surface_node_tags:
        surface_node_indices.append(node_tag_to_index[s_n])
    
    surface_node_indices = np.sort(np.array(surface_node_indices))


    gmsh.finalize()
    print(f"Mesh statistics:")
    print(f"  Nodes: {len(node_tags)}")
    print(f"  Hexahedral elements: {hex_node_tags.shape[0]}")
    print(f"  Node coordinate range:")
    print(f"    X: [{node_coords[:, 0].min():.6f}, {node_coords[:, 0].max():.6f}]")
    print(f"    Y: [{node_coords[:, 1].min():.6f}, {node_coords[:, 1].max():.6f}]")
    print(f"    Z: [{node_coords[:, 2].min():.6f}, {node_coords[:, 2].max():.6f}]")
    


    
    # Extract connectivity (columns 1 through 8)
    # Note: gmsh uses 1-based indexing, convert to 0-based if needed
    connectivity = hex_node_tags.astype(int)
    
    
    # Convert connectivity from gmsh node tags to array indices
    connectivity_indexed = np.zeros_like(connectivity)
    for i in range(connectivity.shape[0]):
        for j in range((order+1)**3):
            connectivity_indexed[i, j] = node_tag_to_index[connectivity[i, j]]
    
    return node_coords, connectivity_indexed, surface_node_indices

        

def gmsh_to_XYZ_order(order):
    """
    For a 3D hexahedron of given order (with (order+1)^3 nodes)
    returns a mapping from gmsh node ordering to XYZ ordering where X varies 
    fastest, then Y, then Z
    """


    gmsh_to_xyz = [
        None,
        [0, 1, 3, 2, 4, 5, 7, 6],
        [0, 8, 1, 9, 20, 11, 3, 13, 2, 10, 21, 12, 22, 26, 23, 15, 24, 14, 4, 16, 5, 17, 25, 18, 7, 19, 6],
        [0, 8, 9, 1, 10, 32, 35, 14, 11, 33, 34, 15, 3, 19, 18, 2, 12, 36, 37, 16, 40, 56, 57, 44, 43, 59, 58, 45, 22, 49, 48, 20, 13, 39, 38, 17, 41, 60, 61, 47, 42, 63, 62, 46, 23, 50, 51, 21, 4, 24, 25, 5, 26, 52, 53, 28, 27, 55, 54, 29, 7, 31, 30, 6],
        [0, 8, 9, 10, 1, 11, 44, 51, 47, 17, 12, 48, 52, 50, 18, 13, 45, 49, 46, 19, 3, 25, 24, 23, 2, 14, 53, 57, 54, 20, 62, 98, 106, 99, 71, 69, 107, 118, 109, 75, 65, 101, 111, 100, 72, 29, 81, 84, 80, 26, 15, 60, 61, 58, 21, 66, 108, 119, 110, 78, 70, 120, 124, 121, 79, 68, 113, 122, 112, 76, 30, 85, 88, 87, 27, 16, 56, 59, 55, 22, 63, 102, 114, 103, 74, 67, 115, 123, 116, 77, 64, 105, 117, 104, 73, 31, 82, 86, 83, 28, 4, 32, 33, 34, 5, 35, 89, 93, 90, 38, 36, 96, 97, 94, 39, 37, 92, 95, 91, 40, 7, 43, 42, 41, 6],
        [0, 8, 9, 10, 11, 1, 12, 56, 67, 66, 59, 20, 13, 60, 68, 71, 65, 21, 14, 61, 69, 70, 64, 22, 15, 57, 62, 63, 58, 23, 3, 31, 30, 29, 28, 2, 16, 72, 76, 77, 73, 24, 88, 152, 160, 161, 153, 104, 99, 162, 184, 187, 166, 108, 98, 163, 185, 186, 167, 109, 91, 155, 171, 170, 154, 105, 36, 121, 125, 124, 120, 32, 17, 83, 84, 85, 78, 25, 92, 164, 188, 189, 168, 115, 100, 192, 208, 209, 196, 116, 103, 195, 211, 210, 197, 117, 97, 174, 201, 200, 172, 110, 37, 126, 133, 132, 131, 33, 18, 82, 87, 86, 79, 26, 93, 165, 191, 190, 169, 114, 101, 193, 212, 213, 199, 119, 102, 194, 215, 214, 198, 118, 96, 175, 202, 203, 173, 111, 38, 127, 134, 135, 130, 34, 19, 75, 81, 80, 74, 27, 89, 156, 176, 177, 157, 107, 94, 178, 204, 205, 180, 113, 95, 179, 207, 206, 181, 112, 90, 159, 183, 182, 158, 106, 39, 122, 128, 129, 123, 35, 4, 40, 41, 42, 43, 5, 44, 136, 140, 141, 137, 48, 45, 147, 148, 149, 142, 49, 46, 146, 151, 150, 143, 50, 47, 139, 145, 144, 138, 51, 7, 55, 54, 53, 52, 6],
        [0, 8, 9, 10, 11, 12, 1, 13, 68, 83, 82, 81, 71, 23, 14, 72, 84, 91, 87, 80, 24, 15, 73, 88, 92, 90, 79, 25, 16, 74, 85, 89, 86, 78, 26, 17, 69, 75, 76, 77, 70, 27, 3, 37, 36, 35, 34, 33, 2, 18, 93, 97, 98, 99, 94, 28, 118, 218, 226, 227, 228, 219, 143, 133, 229, 262, 269, 265, 235, 147, 132, 230, 266, 270, 268, 236, 148, 131, 231, 263, 267, 264, 237, 149, 121, 221, 243, 242, 241, 220, 144, 43, 169, 174, 173, 172, 168, 38, 19, 108, 109, 113, 110, 100, 29, 122, 232, 271, 275, 272, 238, 158, 134, 280, 316, 324, 317, 289, 159, 141, 287, 325, 336, 327, 293, 163, 137, 283, 319, 329, 318, 290, 160, 130, 247, 299, 302, 298, 244, 150, 44, 175, 185, 188, 184, 183, 39, 20, 107, 116, 117, 114, 101, 30, 123, 233, 278, 279, 276, 239, 157, 138, 284, 326, 337, 328, 296, 166, 142, 288, 338, 342, 339, 297, 167, 140, 286, 331, 340, 330, 294, 164, 129, 248, 303, 306, 305, 245, 151, 45, 176, 189, 192, 191, 182, 40, 21, 106, 112, 115, 111, 102, 31, 124, 234, 274, 277, 273, 240, 156, 135, 281, 320, 332, 321, 292, 162, 139, 285, 333, 341, 334, 295, 165, 136, 282, 323, 335, 322, 291, 161, 128, 249, 300, 304, 301, 246, 152, 46, 177, 186, 190, 187, 181, 41, 22, 96, 105, 104, 103, 95, 32, 119, 222, 250, 251, 252, 223, 146, 125, 253, 307, 311, 308, 256, 155, 126, 254, 314, 315, 312, 257, 154, 127, 255, 310, 313, 309, 258, 153, 120, 225, 261, 260, 259, 224, 145, 47, 170, 178, 179, 180, 171, 42, 4, 48, 49, 50, 51, 52, 5, 53, 193, 197, 198, 199, 194, 58, 54, 208, 209, 213, 210, 200, 59, 55, 207, 216, 217, 214, 201, 60, 56, 206, 212, 215, 211, 202, 61, 57, 196, 205, 204, 203, 195, 62, 7, 67, 66, 65, 64, 63, 6],
        [0, 8, 9, 10, 11, 12, 13, 1, 14, 80, 99, 98, 97, 96, 83, 26, 15, 84, 100, 111, 110, 103, 95, 27, 16, 85, 104, 112, 115, 109, 94, 28, 17, 86, 105, 113, 114, 108, 93, 29, 18, 87, 101, 106, 107, 102, 92, 30, 19, 81, 88, 89, 90, 91, 82, 31, 3, 43, 42, 41, 40, 39, 38, 2, 20, 116, 120, 121, 122, 123, 117, 32, 152, 296, 304, 305, 306, 307, 297, 188, 171, 308, 352, 363, 362, 355, 316, 192, 170, 309, 356, 364, 367, 361, 317, 193, 169, 310, 357, 365, 366, 360, 318, 194, 168, 311, 353, 358, 359, 354, 319, 195, 155, 299, 327, 326, 325, 324, 298, 189, 50, 225, 231, 230, 229, 228, 224, 44, 21, 135, 136, 140, 141, 137, 124, 33, 156, 312, 368, 372, 373, 369, 320, 207, 172, 384, 448, 456, 457, 449, 400, 208, 183, 395, 458, 480, 483, 462, 404, 212, 182, 394, 459, 481, 482, 463, 405, 213, 175, 387, 451, 467, 466, 450, 401, 209, 167, 332, 417, 421, 420, 416, 328, 196, 51, 232, 245, 249, 248, 244, 243, 45, 22, 134, 147, 148, 149, 142, 125, 34, 157, 313, 379, 380, 381, 374, 321, 206, 176, 388, 460, 484, 485, 464, 411, 219, 184, 396, 488, 504, 505, 492, 412, 220, 187, 399, 491, 507, 506, 493, 413, 221, 181, 393, 470, 497, 496, 468, 406, 214, 166, 333, 422, 429, 428, 427, 329, 197, 52, 233, 250, 257, 256, 255, 242, 46, 23, 133, 146, 151, 150, 143, 126, 35, 158, 314, 378, 383, 382, 375, 322, 205, 177, 389, 461, 487, 486, 465, 410, 218, 185, 397, 489, 508, 509, 495, 415, 223, 186, 398, 490, 511, 510, 494, 414, 222, 180, 392, 471, 498, 499, 469, 407, 215, 165, 334, 423, 430, 431, 426, 330, 198, 53, 234, 251, 258, 259, 254, 241, 47, 24, 132, 139, 145, 144, 138, 127, 36, 159, 315, 371, 377, 376, 370, 323, 204, 173, 385, 452, 472, 473, 453, 403, 211, 178, 390, 474, 500, 501, 476, 409, 217, 179, 391, 475, 503, 502, 477, 408, 216, 174, 386, 455, 479, 478, 454, 402, 210, 164, 335, 418, 424, 425, 419, 331, 199, 54, 235, 246, 252, 253, 247, 240, 48, 25, 119, 131, 130, 129, 128, 118, 37, 153, 300, 336, 337, 338, 339, 301, 191, 160, 340, 432, 436, 437, 433, 344, 203, 161, 341, 443, 444, 445, 438, 345, 202, 162, 342, 442, 447, 446, 439, 346, 201, 163, 343, 435, 441, 440, 434, 347, 200, 154, 303, 351, 350, 349, 348, 302, 190, 55, 226, 236, 237, 238, 239, 227, 49, 4, 56, 57, 58, 59, 60, 61, 5, 62, 260, 264, 265, 266, 267, 261, 68, 63, 279, 280, 284, 285, 281, 268, 69, 64, 278, 291, 292, 293, 286, 269, 70, 65, 277, 290, 295, 294, 287, 270, 71, 66, 276, 283, 289, 288, 282, 271, 72, 67, 263, 275, 274, 273, 272, 262, 73, 7, 79, 78, 77, 76, 75, 74, 6],
        [0, 8, 9, 10, 11, 12, 13, 14, 1, 15, 92, 115, 114, 113, 112, 111, 95, 29, 16, 96, 116, 131, 130, 129, 119, 110, 30, 17, 97, 120, 132, 139, 135, 128, 109, 31, 18, 98, 121, 136, 140, 138, 127, 108, 32, 19, 99, 122, 133, 137, 134, 126, 107, 33, 20, 100, 117, 123, 124, 125, 118, 106, 34, 21, 93, 101, 102, 103, 104, 105, 94, 35, 3, 49, 48, 47, 46, 45, 44, 43, 2, 22, 141, 145, 146, 147, 148, 149, 142, 36, 190, 386, 394, 395, 396, 397, 398, 387, 239, 213, 399, 454, 469, 468, 467, 457, 409, 243, 212, 400, 458, 470, 477, 473, 466, 410, 244, 211, 401, 459, 474, 478, 476, 465, 411, 245, 210, 402, 460, 471, 475, 472, 464, 412, 246, 209, 403, 455, 461, 462, 463, 456, 413, 247, 193, 389, 423, 422, 421, 420, 419, 388, 240, 57, 289, 296, 295, 294, 293, 292, 288, 50, 23, 164, 165, 169, 170, 171, 166, 150, 37, 194, 404, 479, 483, 484, 485, 480, 414, 262, 214, 504, 604, 612, 613, 614, 605, 529, 263, 229, 519, 615, 648, 655, 651, 621, 533, 267, 228, 518, 616, 652, 656, 654, 622, 534, 268, 227, 517, 617, 649, 653, 650, 623, 535, 269, 217, 507, 607, 629, 628, 627, 606, 530, 264, 208, 429, 555, 560, 559, 558, 554, 424, 248, 58, 297, 313, 318, 317, 316, 312, 311, 51, 24, 163, 180, 181, 185, 182, 172, 151, 38, 195, 405, 494, 495, 499, 496, 486, 415, 261, 218, 508, 618, 657, 661, 658, 624, 544, 278, 230, 520, 666, 702, 710, 703, 675, 545, 279, 237, 527, 673, 711, 722, 713, 679, 549, 283, 233, 523, 669, 705, 715, 704, 676, 546, 280, 226, 516, 633, 685, 688, 684, 630, 536, 270, 207, 430, 561, 571, 574, 570, 569, 425, 249, 59, 298, 319, 329, 332, 328, 327, 310, 52, 25, 162, 179, 188, 189, 186, 173, 152, 39, 196, 406, 493, 502, 503, 500, 487, 416, 260, 219, 509, 619, 664, 665, 662, 625, 543, 277, 234, 524, 670, 712, 723, 714, 682, 552, 286, 238, 528, 674, 724, 728, 725, 683, 553, 287, 236, 526, 672, 717, 726, 716, 680, 550, 284, 225, 515, 634, 689, 692, 691, 631, 537, 271, 206, 431, 562, 575, 578, 577, 568, 426, 250, 60, 299, 320, 333, 336, 335, 326, 309, 53, 26, 161, 178, 184, 187, 183, 174, 153, 40, 197, 407, 492, 498, 501, 497, 488, 417, 259, 220, 510, 620, 660, 663, 659, 626, 542, 276, 231, 521, 667, 706, 718, 707, 678, 548, 282, 235, 525, 671, 719, 727, 720, 681, 551, 285, 232, 522, 668, 709, 721, 708, 677, 547, 281, 224, 514, 635, 686, 690, 687, 632, 538, 272, 205, 432, 563, 572, 576, 573, 567, 427, 251, 61, 300, 321, 330, 334, 331, 325, 308, 54, 27, 160, 168, 177, 176, 175, 167, 154, 41, 198, 408, 482, 491, 490, 489, 481, 418, 258, 215, 505, 608, 636, 637, 638, 609, 532, 266, 221, 511, 639, 693, 697, 694, 642, 541, 275, 222, 512, 640, 700, 701, 698, 643, 540, 274, 223, 513, 641, 696, 699, 695, 644, 539, 273, 216, 506, 611, 647, 646, 645, 610, 531, 265, 204, 433, 556, 564, 565, 566, 557, 428, 252, 62, 301, 314, 322, 323, 324, 315, 307, 55, 28, 144, 159, 158, 157, 156, 155, 143, 42, 191, 390, 434, 435, 436, 437, 438, 391, 242, 199, 439, 579, 583, 584, 585, 580, 444, 257, 200, 440, 594, 595, 599, 596, 586, 445, 256, 201, 441, 593, 602, 603, 600, 587, 446, 255, 202, 442, 592, 598, 601, 597, 588, 447, 254, 203, 443, 582, 591, 590, 589, 581, 448, 253, 192, 393, 453, 452, 451, 450, 449, 392, 241, 63, 290, 302, 303, 304, 305, 306, 291, 56, 4, 64, 65, 66, 67, 68, 69, 70, 5, 71, 337, 341, 342, 343, 344, 345, 338, 78, 72, 360, 361, 365, 366, 367, 362, 346, 79, 73, 359, 376, 377, 381, 378, 368, 347, 80, 74, 358, 375, 384, 385, 382, 369, 348, 81, 75, 357, 374, 380, 383, 379, 370, 349, 82, 76, 356, 364, 373, 372, 371, 363, 350, 83, 77, 340, 355, 354, 353, 352, 351, 339, 84, 7, 91, 90, 89, 88, 87, 86, 85, 6],
        [0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 16, 104, 131, 130, 129, 128, 127, 126, 107, 32, 17, 108, 132, 151, 150, 149, 148, 135, 125, 33, 18, 109, 136, 152, 163, 162, 155, 147, 124, 34, 19, 110, 137, 156, 164, 167, 161, 146, 123, 35, 20, 111, 138, 157, 165, 166, 160, 145, 122, 36, 21, 112, 139, 153, 158, 159, 154, 144, 121, 37, 22, 113, 133, 140, 141, 142, 143, 134, 120, 38, 23, 105, 114, 115, 116, 117, 118, 119, 106, 39, 3, 55, 54, 53, 52, 51, 50, 49, 48, 2, 24, 168, 172, 173, 174, 175, 176, 177, 169, 40, 232, 488, 496, 497, 498, 499, 500, 501, 489, 296, 259, 502, 568, 587, 586, 585, 584, 571, 514, 300, 258, 503, 572, 588, 599, 598, 591, 583, 515, 301, 257, 504, 573, 592, 600, 603, 597, 582, 516, 302, 256, 505, 574, 593, 601, 602, 596, 581, 517, 303, 255, 506, 575, 589, 594, 595, 590, 580, 518, 304, 254, 507, 569, 576, 577, 578, 579, 570, 519, 305, 235, 491, 531, 530, 529, 528, 527, 526, 490, 297, 64, 361, 369, 368, 367, 366, 365, 364, 360, 56, 25, 195, 196, 200, 201, 202, 203, 197, 178, 41, 236, 508, 604, 608, 609, 610, 611, 605, 520, 323, 260, 640, 784, 792, 793, 794, 795, 785, 676, 324, 279, 659, 796, 840, 851, 850, 843, 804, 680, 328, 278, 658, 797, 844, 852, 855, 849, 805, 681, 329, 277, 657, 798, 845, 853, 854, 848, 806, 682, 330, 276, 656, 799, 841, 846, 847, 842, 807, 683, 331, 263, 643, 787, 815, 814, 813, 812, 786, 677, 325, 253, 538, 713, 719, 718, 717, 716, 712, 532, 306, 65, 370, 389, 395, 394, 393, 392, 388, 387, 57, 26, 194, 215, 216, 220, 221, 217, 204, 179, 42, 237, 509, 623, 624, 628, 629, 625, 612, 521, 322, 264, 644, 800, 856, 860, 861, 857, 808, 695, 343, 280, 660, 872, 936, 944, 945, 937, 888, 696, 344, 291, 671, 883, 946, 968, 971, 950, 892, 700, 348, 290, 670, 882, 947, 969, 970, 951, 893, 701, 349, 283, 663, 875, 939, 955, 954, 938, 889, 697, 345, 275, 655, 820, 905, 909, 908, 904, 816, 684, 332, 252, 539, 720, 733, 737, 736, 732, 731, 533, 307, 66, 371, 396, 409, 413, 412, 408, 407, 386, 58, 27, 193, 214, 227, 228, 229, 222, 205, 180, 43, 238, 510, 622, 635, 636, 637, 630, 613, 522, 321, 265, 645, 801, 867, 868, 869, 862, 809, 694, 342, 284, 664, 876, 948, 972, 973, 952, 899, 707, 355, 292, 672, 884, 976, 992, 993, 980, 900, 708, 356, 295, 675, 887, 979, 995, 994, 981, 901, 709, 357, 289, 669, 881, 958, 985, 984, 956, 894, 702, 350, 274, 654, 821, 910, 917, 916, 915, 817, 685, 333, 251, 540, 721, 738, 745, 744, 743, 730, 534, 308, 67, 372, 397, 414, 421, 420, 419, 406, 385, 59, 28, 192, 213, 226, 231, 230, 223, 206, 181, 44, 239, 511, 621, 634, 639, 638, 631, 614, 523, 320, 266, 646, 802, 866, 871, 870, 863, 810, 693, 341, 285, 665, 877, 949, 975, 974, 953, 898, 706, 354, 293, 673, 885, 977, 996, 997, 983, 903, 711, 359, 294, 674, 886, 978, 999, 998, 982, 902, 710, 358, 288, 668, 880, 959, 986, 987, 957, 895, 703, 351, 273, 653, 822, 911, 918, 919, 914, 818, 686, 334, 250, 541, 722, 739, 746, 747, 742, 729, 535, 309, 68, 373, 398, 415, 422, 423, 418, 405, 384, 60, 29, 191, 212, 219, 225, 224, 218, 207, 182, 45, 240, 512, 620, 627, 633, 632, 626, 615, 524, 319, 267, 647, 803, 859, 865, 864, 858, 811, 692, 340, 281, 661, 873, 940, 960, 961, 941, 891, 699, 347, 286, 666, 878, 962, 988, 989, 964, 897, 705, 353, 287, 667, 879, 963, 991, 990, 965, 896, 704, 352, 282, 662, 874, 943, 967, 966, 942, 890, 698, 346, 272, 652, 823, 906, 912, 913, 907, 819, 687, 335, 249, 542, 723, 734, 740, 741, 735, 728, 536, 310, 69, 374, 399, 410, 416, 417, 411, 404, 383, 61, 30, 190, 199, 211, 210, 209, 208, 198, 183, 46, 241, 513, 607, 619, 618, 617, 616, 606, 525, 318, 261, 641, 788, 824, 825, 826, 827, 789, 679, 327, 268, 648, 828, 920, 924, 925, 921, 832, 691, 339, 269, 649, 829, 931, 932, 933, 926, 833, 690, 338, 270, 650, 830, 930, 935, 934, 927, 834, 689, 337, 271, 651, 831, 923, 929, 928, 922, 835, 688, 336, 262, 642, 791, 839, 838, 837, 836, 790, 678, 326, 248, 543, 714, 724, 725, 726, 727, 715, 537, 311, 70, 375, 390, 400, 401, 402, 403, 391, 382, 62, 31, 171, 189, 188, 187, 186, 185, 184, 170, 47, 233, 492, 544, 545, 546, 547, 548, 549, 493, 299, 242, 550, 748, 752, 753, 754, 755, 749, 556, 317, 243, 551, 767, 768, 772, 773, 769, 756, 557, 316, 244, 552, 766, 779, 780, 781, 774, 757, 558, 315, 245, 553, 765, 778, 783, 782, 775, 758, 559, 314, 246, 554, 764, 771, 777, 776, 770, 759, 560, 313, 247, 555, 751, 763, 762, 761, 760, 750, 561, 312, 234, 495, 567, 566, 565, 564, 563, 562, 494, 298, 71, 362, 376, 377, 378, 379, 380, 381, 363, 63, 4, 72, 73, 74, 75, 76, 77, 78, 79, 5, 80, 424, 428, 429, 430, 431, 432, 433, 425, 88, 81, 451, 452, 456, 457, 458, 459, 453, 434, 89, 82, 450, 471, 472, 476, 477, 473, 460, 435, 90, 83, 449, 470, 483, 484, 485, 478, 461, 436, 91, 84, 448, 469, 482, 487, 486, 479, 462, 437, 92, 85, 447, 468, 475, 481, 480, 474, 463, 438, 93, 86, 446, 455, 467, 466, 465, 464, 454, 439, 94, 87, 427, 445, 444, 443, 442, 441, 440, 426, 95, 7, 103, 102, 101, 100, 99, 98, 97, 96, 6]
    ]

    return np.array(gmsh_to_xyz[order])

    ### 
    ### 
    ### 

    """
    ============
    Above pre-computed arrays were formed by the following procedure

    Step-1
    Create a regular cube centered at zero.
    Corner points should be (+-order, +-order, +-order).
    This "forces" gmsh higher-prder (equidistanced) nodes to be in integer 
    coordinates

    Step-2
    Mesh it with given polynomial order

    Step-3
    Extract nodes and node coordinates. These nodes are equidistanced and are in
    gmsh node ordering. 
    https://gmsh.info/doc/texinfo/gmsh.html#High_002dorder-elements

    Step-4
    Sort the nodes by following steps
        1. Sort by X coord
        2. Stable sort by Y coord
        3. Stable sort by Z coord
    After this nodes are sorted such that X varies fastest, then Y, then Z

    Step-5
    Return the mapping from gmsh ordering to new ordering
    
    ============
    """

    # first we create one reference hexahedron where nodes coordinates are 
    # (almost) integers


    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    
    # Create a new model
    gmsh.model.add("single_hex")
    
    h = order 
    
    # Bottom face vertices (z = -h)
    p1 = gmsh.model.geo.addPoint(-h, -h, -h)
    p2 = gmsh.model.geo.addPoint( h, -h, -h)
    p3 = gmsh.model.geo.addPoint( h,  h, -h)
    p4 = gmsh.model.geo.addPoint(-h,  h, -h)
    
    # Top face vertices (z = h)
    p5 = gmsh.model.geo.addPoint(-h, -h,  h)
    p6 = gmsh.model.geo.addPoint( h, -h,  h)
    p7 = gmsh.model.geo.addPoint( h,  h,  h)
    p8 = gmsh.model.geo.addPoint(-h,  h,  h)
    
    # Create edges
    # Bottom face edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    # Top face edges
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    
    # Vertical edges
    l9  = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)
    
    # Create curve loops for faces
    # Bottom face
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    # Top face
    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    # Side faces
    cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])   # front
    cl4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])  # right
    cl5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])  # back
    cl6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])   # left
    
    # Create surfaces
    s1 = gmsh.model.geo.addPlaneSurface([cl1])  # bottom
    s2 = gmsh.model.geo.addPlaneSurface([cl2])  # top
    s3 = gmsh.model.geo.addPlaneSurface([cl3])  # front
    s4 = gmsh.model.geo.addPlaneSurface([cl4])  # right
    s5 = gmsh.model.geo.addPlaneSurface([cl5])  # back
    s6 = gmsh.model.geo.addPlaneSurface([cl6])  # left
    
    # Create surface loop and volume
    sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    vol = gmsh.model.geo.addVolume([sl])
    
    # Synchronize the geometry
    gmsh.model.geo.synchronize()
    
    # Set mesh algorithm to generate structured mesh (hexahedral)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 6)  # Frontal for 3D
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine into quads/hexes
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)  # Recombine 3D elements
    
    # Set element order
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    
    # Set mesh size to ensure only one element
    # We want exactly one hexahedron, so set a large characteristic length
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * 2)
    
    # Alternative approach: use transfinite meshing for structured grid
    # Set number of divisions on each edge to 1
    for line_tag in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag, 2)  # 2 nodes = 1 element
    
    # Set surfaces as transfinite
    for surf_tag in [s1, s2, s3, s4, s5, s6]:
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_tag)
        gmsh.model.geo.mesh.setRecombine(2, surf_tag)  # Recombine into quads
    
    # Set volume as transfinite
    gmsh.model.geo.mesh.setTransfiniteVolume(vol)
    
    # Synchronize after transfinite settings
    gmsh.model.geo.synchronize()
    
    # Generate mesh
    gmsh.model.mesh.generate(3)

    gmsh.model.geo.synchronize()

    output_file = "reference-hex.msh"

    # Write mesh file
    gmsh.write(output_file)
    
    # Print mesh information
    print(f"Mesh generated with element order: {order}")
    print(f"Output file: {output_file}")
    
    # Get mesh statistics
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    
    print(f"Number of nodes: {len(node_tags)}")
    for i, elem_type in enumerate(element_types):
        print(f"Element type {elem_type}: {len(element_tags[i])} elements")
    


    ## now read the nodes in gmsh order and create a mapping by sorting

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    # Create mapping from gmsh node tags to array indices
    node_tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
    
    # Reshape coordinates (gmsh returns flattened array)
    node_coords = node_coords.reshape(-1, 3)

    # make coordinates really integers so that sorting will be exact
    node_coords = np.rint(node_coords)
    
    
    # Get hexahedral elements 
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()

  

    nodes_gmsh_order = None
    for i, elem_type in enumerate(element_types):
        if elem_type == order_to_elem_type[order]: 
            nodes_gmsh_order = element_node_tags[i]
            
            # Reshape to get element-to-node connectivity
            # nodes_gmsh_order = nodes_gmsh_order.reshape(-1, (order+1)**3)
            break
    
    if nodes_gmsh_order is None:
        raise ValueError("No hexahedral element nodes found in the mesh file")
    gmsh.finalize()
    print(f"Mesh statistics:")
    print(f"  Nodes: {len(node_tags)}")
    print(f"  Node coordinate range:")
    print(f"    X: [{node_coords[:, 0].min():.6f}, {node_coords[:, 0].max():.6f}]")
    print(f"    Y: [{node_coords[:, 1].min():.6f}, {node_coords[:, 1].max():.6f}]")
    print(f"    Z: [{node_coords[:, 2].min():.6f}, {node_coords[:, 2].max():.6f}]")
    
    assert len(nodes_gmsh_order) == (order+1)**3

    node_coords_gmsh_order = []
    for node in nodes_gmsh_order:
        node_coords_gmsh_order.append(node_coords[node_tag_to_index[node]])

    node_coords_gmsh_order = np.array(node_coords_gmsh_order)
    idx = np.arange((order+1)**3)   # to obtain the mapping

    nodes_n_coords_gmsh_order = np.column_stack((idx, nodes_gmsh_order, node_coords_gmsh_order))

    # sort by x coord, then y, then z

    nodes_sorted = nodes_n_coords_gmsh_order[nodes_n_coords_gmsh_order[:,2].argsort(stable=True)]
    nodes_sorted = nodes_sorted[nodes_sorted[:,3].argsort(stable=True)]
    nodes_sorted = nodes_sorted[nodes_sorted[:,4].argsort(stable=True)]



    # for i in range((order+1)**2):
    #     print(nodes_sorted[i*(order+1): (i+1)*(order+1), 1])
        # print(nodes_sorted[i*(order+1): (i+1)*(order+1), 2:5])


    return nodes_sorted[:, 0].astype(int)

        


    