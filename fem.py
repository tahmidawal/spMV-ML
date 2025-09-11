import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import matplotlib.tri as mtri
import os

class FEMPoissonSolver:
    def __init__(self, nx=40, ny=40, mesh_type='unstructured'):
        """
        Initialize FEM solver for generalized Poisson equation
        -∇·(θ∇u) = f on domain [0,1]×[0,1]
        with u = 0 on boundary (Dirichlet BC)
        
        Parameters:
        - nx, ny: mesh resolution parameters
        - mesh_type: 'structured', 'unstructured', or 'adaptive'
        """
        self.nx = nx
        self.ny = ny
        self.mesh_type = mesh_type
        self.setup_mesh(mesh_type)
        
    def setup_mesh(self, mesh_type='unstructured'):
        """Create triangular mesh using various strategies"""
        if mesh_type == 'structured':
            self._create_structured_mesh()
        elif mesh_type == 'unstructured':
            self._create_unstructured_mesh()
        elif mesh_type == 'adaptive':
            self._create_adaptive_mesh()
        else:
            raise ValueError("mesh_type must be 'structured', 'unstructured', or 'adaptive'")
        
        self.n_nodes = len(self.points)
        self.n_elements = len(self.triangles)
        
        # Identify boundary nodes
        self.identify_boundary_nodes()
    
    def _create_structured_mesh(self):
        """Create regular grid mesh (original method)"""
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        self.tri = Delaunay(points)
        self.points = self.tri.points
        self.triangles = self.tri.simplices
    
    def _create_unstructured_mesh(self):
        """Create truly unstructured mesh with random interior points"""
        np.random.seed(42)  # For reproducibility
        
        # Start with boundary points to ensure proper boundary
        boundary_points = []
        
        # Bottom boundary
        x_bottom = np.linspace(0, 1, max(10, self.nx//4))
        boundary_points.extend([(x, 0.0) for x in x_bottom])
        
        # Right boundary  
        y_right = np.linspace(0, 1, max(10, self.ny//4))
        boundary_points.extend([(1.0, y) for y in y_right[1:]])
        
        # Top boundary
        x_top = np.linspace(1, 0, max(10, self.nx//4))
        boundary_points.extend([(x, 1.0) for x in x_top[1:]])
        
        # Left boundary
        y_left = np.linspace(1, 0, max(10, self.ny//4))
        boundary_points.extend([(0.0, y) for y in y_left[1:-1]])
        
        # Add random interior points with clustering
        n_interior = max(100, (self.nx * self.ny) // 2)
        interior_points = []
        
        # Add some clustered regions for complexity
        cluster_centers = [(0.3, 0.3), (0.7, 0.7), (0.2, 0.8), (0.8, 0.2)]
        points_per_cluster = n_interior // 6
        
        for cx, cy in cluster_centers:
            for _ in range(points_per_cluster):
                # Gaussian cluster around center
                x = np.random.normal(cx, 0.1)
                y = np.random.normal(cy, 0.1)
                # Keep within domain
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                interior_points.append((x, y))
        
        # Add completely random points
        remaining = n_interior - len(interior_points)
        for _ in range(remaining):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            interior_points.append((x, y))
        
        # Combine all points
        all_points = boundary_points + interior_points
        points = np.array(all_points)
        
        # Create Delaunay triangulation
        self.tri = Delaunay(points)
        self.points = self.tri.points
        self.triangles = self.tri.simplices
    
    def _create_adaptive_mesh(self):
        """Create adaptively refined mesh with local refinement"""
        np.random.seed(43)
        
        # Start with coarse structured mesh
        nx_coarse, ny_coarse = max(8, self.nx//5), max(8, self.ny//5)
        x = np.linspace(0, 1, nx_coarse)
        y = np.linspace(0, 1, ny_coarse)
        X, Y = np.meshgrid(x, y)
        points = [(X[i,j], Y[i,j]) for i in range(nx_coarse) for j in range(ny_coarse)]
        
        # Define regions for refinement (based on features)
        refinement_regions = [
            {'center': (0.25, 0.25), 'radius': 0.2, 'density': 3},
            {'center': (0.75, 0.75), 'radius': 0.15, 'density': 4},
            {'center': (0.5, 0.8), 'radius': 0.1, 'density': 5},
            {'center': (0.2, 0.7), 'radius': 0.12, 'density': 3},
        ]
        
        # Add refined points in specified regions
        for region in refinement_regions:
            cx, cy = region['center']
            radius = region['radius']
            density = region['density']
            
            # Add points in this region
            n_points = density * 20
            for _ in range(n_points):
                # Random point in circle
                r = np.random.uniform(0, radius)
                theta = np.random.uniform(0, 2*np.pi)
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                
                # Keep within domain
                if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98:
                    points.append((x, y))
        
        # Add some random points for irregularity
        for _ in range(50):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append((x, y))
        
        points = np.array(points)
        self.tri = Delaunay(points)
        self.points = self.tri.points
        self.triangles = self.tri.simplices
        
    def identify_boundary_nodes(self):
        """Identify nodes on the boundary"""
        eps = 1e-10
        self.boundary_nodes = []
        for i, point in enumerate(self.points):
            if (abs(point[0]) < eps or abs(point[0] - 1) < eps or
                abs(point[1]) < eps or abs(point[1] - 1) < eps):
                self.boundary_nodes.append(i)
        self.boundary_nodes = np.array(self.boundary_nodes)
        
        # Interior nodes
        all_nodes = set(range(self.n_nodes))
        boundary_set = set(self.boundary_nodes)
        self.interior_nodes = np.array(list(all_nodes - boundary_set))
        
    def element_stiffness_matrix(self, vertices, theta_elem):
        """
        Compute element stiffness matrix for a triangle with variable coefficient θ
        For equation: -∇·(θ∇u) = f
        """
        # Extract coordinates
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        
        # Area of triangle
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        # Gradients of basis functions
        b = np.array([y2-y3, y3-y1, y1-y2]) / (2*area)
        c = np.array([x3-x2, x1-x3, x2-x1]) / (2*area)
        
        # Element stiffness matrix with theta
        # Using average theta over element (1-point quadrature)
        theta_avg = np.mean(theta_elem)
        
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = theta_avg * area * (b[i]*b[j] + c[i]*c[j])
                
        return K_e
    
    def element_load_vector(self, vertices, f_values):
        """
        Compute element load vector for a triangle
        Using 1-point quadrature at centroid
        """
        # Area of triangle
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        # Average of nodal values (1-point quadrature)
        f_avg = np.mean(f_values)
        
        # Element load vector (equal distribution to nodes)
        f_e = f_avg * area * np.ones(3) / 3.0
        
        return f_e
    
    def assemble_system(self, f_field, theta_field):
        """Assemble global stiffness matrix and load vector with variable θ"""
        # Initialize sparse matrix and load vector
        self.K = lil_matrix((self.n_nodes, self.n_nodes))
        self.f = np.zeros(self.n_nodes)
        
        # Loop over elements
        for elem_idx, triangle in enumerate(self.triangles):
            # Get vertices
            vertices = self.points[triangle]
            
            # Get theta values at element nodes
            theta_elem = theta_field[triangle]
            
            # Compute element matrices
            K_e = self.element_stiffness_matrix(vertices, theta_elem)
            
            # Get f values at nodes
            f_values = f_field[triangle]
            f_e = self.element_load_vector(vertices, f_values)
            
            # Assemble into global system
            for i in range(3):
                for j in range(3):
                    self.K[triangle[i], triangle[j]] += K_e[i, j]
                self.f[triangle[i]] += f_e[i]
        
        # Convert to CSR format for efficient solving
        self.K = csr_matrix(self.K)
        
    def apply_boundary_conditions(self):
        """Apply Dirichlet boundary conditions (u = 0 on boundary)"""
        # Create reduced system (only interior nodes)
        n_int = len(self.interior_nodes)
        self.K_reduced = lil_matrix((n_int, n_int))
        self.f_reduced = np.zeros(n_int)
        
        # Map from global to reduced indices
        global_to_reduced = {}
        for i, node in enumerate(self.interior_nodes):
            global_to_reduced[node] = i
        
        # Extract submatrix and subvector
        for i, node_i in enumerate(self.interior_nodes):
            self.f_reduced[i] = self.f[node_i]
            for j, node_j in enumerate(self.interior_nodes):
                self.K_reduced[i, j] = self.K[node_i, node_j]
        
        self.K_reduced = csr_matrix(self.K_reduced)
        
    def solve(self):
        """Solve the linear system"""
        # Solve reduced system
        u_reduced = spsolve(self.K_reduced, self.f_reduced)
        
        # Reconstruct full solution
        self.u = np.zeros(self.n_nodes)
        self.u[self.interior_nodes] = u_reduced
        # Boundary nodes remain zero (Dirichlet BC)
        
        return self.u
    
    def create_sinusoidal_field(self, k1=None, k2=None, seed=None, normalize=True):
        """
        Create sinusoidal source function: f(x,y) = sin(2π*k1*x) * sin(2π*k2*y)
        where k1, k2 ∈ {1,2,3,4,5,6,7,8,9}
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Define frequency choices
        frequency_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # Select random frequencies if not provided
        if k1 is None:
            k1 = np.random.choice(frequency_choices)
        if k2 is None:
            k2 = np.random.choice(frequency_choices)
        
        # Store the frequencies for reference
        self.k1, self.k2 = k1, k2
        
        # Compute sinusoidal field at mesh nodes
        field = np.zeros(self.n_nodes)
        for i, point in enumerate(self.points):
            x, y = point
            field[i] = np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)
        
        # Normalize to [-1, 1] if requested
        if normalize:
            field = self.normalize_field(field)
        
        return field
    
    def normalize_field(self, field, target_range=(-1, 1)):
        """
        Normalize field to target range (default [-1, 1])
        """
        if np.max(np.abs(field)) > 0:
            # First normalize to [-1, 1]
            field_normalized = field / np.max(np.abs(field))
            
            # Then scale to target range if different
            if target_range != (-1, 1):
                a, b = target_range
                field_normalized = a + (field_normalized + 1) * (b - a) / 2
            
            return field_normalized
        else:
            return field
    
    def create_theta_field(self, min_val=0.5, max_val=2.0, n_regions=3, seed=None, normalize=True):
        """
        Create spatially varying theta field
        Represents material properties or diffusion coefficient
        """
        if seed is not None:
            np.random.seed(seed + 100)  # Different seed from f field
        
        # Initialize with base value
        theta = np.ones(self.n_nodes) * (min_val + max_val) / 2
        
        # Add smooth variations
        for _ in range(n_regions):
            # Random center and radius
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            radius = np.random.uniform(0.1, 0.3)
            value = np.random.uniform(min_val, max_val)
            
            # Apply smooth transition
            for i, point in enumerate(self.points):
                x, y = point
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius:
                    # Smooth transition using cosine
                    weight = 0.5 * (1 + np.cos(np.pi * dist / radius))
                    theta[i] = theta[i] * (1 - weight) + value * weight
        
        # Add some small-scale variation
        noise = np.random.randn(self.n_nodes) * 0.1
        theta += noise
        
        # Ensure bounds
        theta = np.clip(theta, min_val, max_val)
        
        # Normalize to [-1, 1] if requested
        if normalize:
            theta = self.normalize_field(theta)
        
        return theta

def visualize_results(solver, f_field, theta_field, u_solution):
    """Create comprehensive visualization including theta field"""
    fig = plt.figure(figsize=(20, 16))
    
    # Create triangulation for plotting
    triang = mtri.Triangulation(solver.points[:, 0], solver.points[:, 1], solver.triangles)
    
    # 1. Theta field (coefficient)
    ax1 = plt.subplot(3, 3, 1)
    contour_theta = ax1.tricontourf(triang, theta_field, levels=20, cmap='plasma')
    plt.colorbar(contour_theta, ax=ax1)
    ax1.set_title('θ(x,y) Field\n(Normalized [-1,1])', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    
    # 2. Input field (2D)
    ax2 = plt.subplot(3, 3, 2)
    contour_f = ax2.tricontourf(triang, f_field, levels=20, cmap='RdBu_r')
    plt.colorbar(contour_f, ax=ax2)
    ax2.set_title(f'Input Field f(x,y)\nsin(2π·{solver.k1}·x) × sin(2π·{solver.k2}·y) [Normalized]', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    
    # 3. Combined visualization (θ and f)
    ax3 = plt.subplot(3, 3, 3)
    # Show f as contour lines over theta as filled contours
    contour_theta2 = ax3.tricontourf(triang, theta_field, levels=15, cmap='YlOrRd', alpha=0.6)
    contour_f2 = ax3.tricontour(triang, f_field, levels=10, colors='blue', linewidths=1)
    ax3.clabel(contour_f2, inline=True, fontsize=8)
    plt.colorbar(contour_theta2, ax=ax3, label='θ')
    ax3.set_title('Combined: θ (filled) and f (contours)', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    
    # 4. 1D Input vector
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(solver.f, 'b-', linewidth=0.5)
    ax4.set_title('1D Input Vector (RHS)\n(Normalized [-1,1])', fontsize=12)
    ax4.set_xlabel('Node index')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    
    # 5. Sparse matrix structure
    ax5 = plt.subplot(3, 3, 5)
    ax5.spy(solver.K, markersize=0.5)
    ax5.set_title(f'Sparse Stiffness Matrix with θ\n({solver.K.shape[0]}×{solver.K.shape[1]}, {solver.K.nnz} non-zeros)', fontsize=12)
    ax5.set_xlabel('Column index')
    ax5.set_ylabel('Row index')
    
    # 6. Output vector (1D)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(u_solution, 'r-', linewidth=0.5)
    ax6.set_title('1D Output Vector\n(Normalized [-1,1])', fontsize=12)
    ax6.set_xlabel('Node index')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3)
    
    # 7. Output field (2D)
    ax7 = plt.subplot(3, 3, 7)
    contour_u = ax7.tricontourf(triang, u_solution, levels=20, cmap='viridis')
    plt.colorbar(contour_u, ax=ax7)
    ax7.set_title('Output Field u(x,y)\n(Normalized [-1,1])', fontsize=12)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_aspect('equal')
    
    # 8. Cross-section comparison
    ax8 = plt.subplot(3, 3, 8)
    # Extract values along y=0.5 line
    mid_nodes = []
    mid_x = []
    for i, point in enumerate(solver.points):
        if abs(point[1] - 0.5) < 0.05:  # Near y=0.5
            mid_nodes.append(i)
            mid_x.append(point[0])
    
    mid_nodes = np.array(mid_nodes)
    mid_x = np.array(mid_x)
    sort_idx = np.argsort(mid_x)
    
    ax8.plot(mid_x[sort_idx], theta_field[mid_nodes[sort_idx]], 'g-', label='θ(x, 0.5)', alpha=0.7)
    ax8.plot(mid_x[sort_idx], f_field[mid_nodes[sort_idx]], 'b-', label=f'f(x, 0.5) [k₁={solver.k1}, k₂={solver.k2}]', alpha=0.7)
    ax8.plot(mid_x[sort_idx], u_solution[mid_nodes[sort_idx]], 'r-', label='u(x, 0.5)', alpha=0.7)
    ax8.set_title('Cross-sections at y=0.5', fontsize=12)
    ax8.set_xlabel('x')
    ax8.set_ylabel('Value')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Sparse matrix values histogram
    ax9 = plt.subplot(3, 3, 9)
    # Get non-zero values from sparse matrix
    sparse_values = solver.K.data
    # Remove very small values (numerical zeros)
    sparse_values = sparse_values[np.abs(sparse_values) > 1e-12]
    
    ax9.hist(sparse_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax9.set_title(f'Sparse Matrix Values Distribution\n({len(sparse_values)} non-zero entries)', fontsize=12)
    ax9.set_xlabel('Matrix Entry Value')
    ax9.set_ylabel('Frequency')
    ax9.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(sparse_values):.3e}\nStd: {np.std(sparse_values):.3e}\nMin: {np.min(sparse_values):.3e}\nMax: {np.max(sparse_values):.3e}'
    ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    plt.suptitle(f'Enhanced FEM Solver: -∇·(θ∇u) = f with Variable Coefficient θ\n({solver.mesh_type.capitalize()} Mesh)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'fem_{solver.mesh_type}_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   - Plot saved as: {filename}")
    plt.close()  # Close the figure to free memory


def generate_fem_sinusoid_dataset(num_samples=300,
                                  k_values=(1, 2, 3, 4, 5),
                                  train_ratio=0.8,
                                  nx=35,
                                  ny=35,
                                  mesh_type='unstructured',
                                  theta_min=0.5,
                                  theta_max=3.0,
                                  theta_regions=4,
                                  seed=123,
                                  out_dir='ml_data',
                                  dataset_name=None):
    """
    Generate a dataset of (RHS, solution) pairs using FEM on a fixed mesh and theta.

    - Inputs are assembled RHS vectors f (global, including boundary entries).
    - Outputs are full solution vectors u with Dirichlet boundary nodes equal to zero.
    - RHS functions are sin(2π k1 x) sin(2π k2 y) with k1,k2 sampled from k_values.
    - Theta field is strictly positive (no normalization) for ellipticity.

    Returns (X_train, Y_train, X_val, Y_val, meta, solver)
    and also saves an NPZ file under out_dir.
    """
    rng = np.random.default_rng(seed)

    # Initialize solver and mesh
    solver = FEMPoissonSolver(nx=nx, ny=ny, mesh_type=mesh_type)

    # Create strictly positive theta (no normalization)
    theta_field = solver.create_theta_field(min_val=theta_min,
                                            max_val=theta_max,
                                            n_regions=theta_regions,
                                            seed=seed,
                                            normalize=False)

    n_nodes = solver.n_nodes
    X = np.zeros((num_samples, n_nodes), dtype=np.float64)
    Y = np.zeros((num_samples, n_nodes), dtype=np.float64)
    ks = np.zeros((num_samples, 2), dtype=np.int32)

    # Sample frequencies
    for i in range(num_samples):
        k1 = int(rng.choice(k_values))
        k2 = int(rng.choice(k_values))
        ks[i] = (k1, k2)

        # Create RHS field (no normalization)
        f_field = solver.create_sinusoidal_field(k1=k1, k2=k2, seed=None, normalize=False)

        # Assemble and solve (theta fixed for all samples)
        solver.assemble_system(f_field, theta_field)
        solver.apply_boundary_conditions()
        u = solver.solve()

        X[i] = solver.f.copy()
        Y[i] = u.copy()

    # Split into train/val
    n_train = int(train_ratio * num_samples)
    indices = rng.permutation(num_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, Y_train, ks_train = X[train_idx], Y[train_idx], ks[train_idx]
    X_val, Y_val, ks_val = X[val_idx], Y[val_idx], ks[val_idx]

    # Metadata
    meta = {
        'points': solver.points,
        'triangles': solver.triangles,
        'boundary_nodes': solver.boundary_nodes,
        'interior_nodes': solver.interior_nodes,
        'theta_field': theta_field,
        'k_values': np.array(k_values, dtype=np.int32),
        'ks_train': ks_train,
        'ks_val': ks_val,
        'mesh_type': solver.mesh_type,
        'nx': nx,
        'ny': ny,
    }

    # Save dataset
    os.makedirs(out_dir, exist_ok=True)
    if dataset_name is None:
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = f'fem_sinusoid_k1k2_{min(k_values)}to{max(k_values)}_{num_samples}s_{ts}.npz'
    save_path = os.path.join(out_dir, dataset_name)
    np.savez(
        save_path,
        X_train=X_train.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        Y_val=Y_val.astype(np.float32),
        ks_train=ks_train,
        ks_val=ks_val,
        points=meta['points'].astype(np.float32),
        triangles=meta['triangles'].astype(np.int32),
        boundary_nodes=meta['boundary_nodes'].astype(np.int32),
        interior_nodes=meta['interior_nodes'].astype(np.int32),
        theta_field=meta['theta_field'].astype(np.float32),
        nx=nx,
        ny=ny,
        mesh_type=solver.mesh_type
    )
    print(f"Dataset saved to: {save_path}")

    return X_train, Y_train, X_val, Y_val, meta, solver


def generate_fem_forward_dataset(num_samples=300,
                                 k_values=(1, 2, 3, 4, 5),
                                 train_ratio=0.8,
                                 nx=35,
                                 ny=35,
                                 mesh_type='unstructured',
                                 theta_min=0.5,
                                 theta_max=3.0,
                                 theta_regions=4,
                                 seed=123,
                                 out_dir='ml_data',
                                 dataset_name=None):
    """
    Generate dataset for learning FORWARD sparse matrix multiplication: u = K @ f
    Instead of solving K @ u = f for u, we compute K @ f directly.
    
    This is a simpler problem than the inverse - we just need to learn SpMV!
    
    Parameters:
    - num_samples: Number of samples to generate
    - k_values: Frequency choices for sinusoidal functions
    - train_ratio: Fraction of data for training
    - nx, ny: Mesh resolution
    - mesh_type: Type of mesh ('structured', 'unstructured', 'adaptive')
    - theta_min, theta_max: Range for theta field (material properties)
    - theta_regions: Number of smooth regions in theta field
    - seed: Random seed
    - out_dir: Output directory
    - dataset_name: Name for saved dataset
    
    Returns (X_train, Y_train, X_val, Y_val, meta, solver)
    where X is input vector f, Y is output K @ f
    """
    rng = np.random.default_rng(seed)
    
    # Initialize solver and mesh
    solver = FEMPoissonSolver(nx=nx, ny=ny, mesh_type=mesh_type)
    print(f"Created {solver.mesh_type} mesh: {solver.n_nodes} nodes, {solver.n_elements} elements")
    
    # Create theta field (fixed for all samples) - NO normalization to keep ellipticity
    theta_field = solver.create_theta_field(min_val=theta_min,
                                           max_val=theta_max,
                                           n_regions=theta_regions,
                                           seed=seed,
                                           normalize=False)
    
    print(f"Theta field range: [{theta_field.min():.3f}, {theta_field.max():.3f}] (elliptic, no normalization)")
    
    # Build the sparse matrix K once (since θ is fixed for all samples)
    # Use dummy f just to assemble K - the f values don't affect K matrix
    dummy_f = np.zeros(solver.n_nodes)
    solver.assemble_system(dummy_f, theta_field)
    K = solver.K  # This is our sparse stiffness matrix
    
    print(f"Assembled sparse matrix K: {K.shape}, {K.nnz} non-zeros")
    print(f"Sparsity: {100*(1-K.nnz/(K.shape[0]*K.shape[1])):.2f}%")
    
    n_nodes = solver.n_nodes
    X = np.zeros((num_samples, n_nodes), dtype=np.float64)
    Y = np.zeros((num_samples, n_nodes), dtype=np.float64)
    ks = np.zeros((num_samples, 2), dtype=np.int32)
    
    print(f"Generating {num_samples} samples via forward SpMV...")
    
    # Generate samples
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
            
        k1 = int(rng.choice(k_values))
        k2 = int(rng.choice(k_values))
        ks[i] = (k1, k2)
        
        # Create input vector f (sinusoidal field evaluated at nodes)
        # Normalize input to [-1, 1] as before
        f_field = solver.create_sinusoidal_field(k1=k1, k2=k2, seed=None, normalize=True)
        
        # Compute output: u = K @ f (forward sparse matrix-vector multiplication)
        u_raw = K @ f_field  # Raw SpMV result
        
        # Apply frequency-aware normalization to output
        # Based on theoretical relationship: |K@f| ∝ (k1² + k2²)
        frequency_factor = k1*k1 + k2*k2
        reference_frequency = 1*1 + 1*1  # Reference: k1=1, k2=1 case
        
        # Normalize output to have similar range as reference frequency
        u_normalized = u_raw * (reference_frequency / frequency_factor)
        
        X[i] = f_field.copy()
        Y[i] = u_normalized.copy()
    
    print(f"Forward SpMV dataset generation complete!")
    
    # Split into train/val
    n_train = int(train_ratio * num_samples)
    indices = rng.permutation(num_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, Y_train, ks_train = X[train_idx], Y[train_idx], ks[train_idx]
    X_val, Y_val, ks_val = X[val_idx], Y[val_idx], ks[val_idx]
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Metadata
    meta = {
        'points': solver.points,
        'triangles': solver.triangles,
        'boundary_nodes': solver.boundary_nodes,
        'interior_nodes': solver.interior_nodes,
        'theta_field': theta_field,
        'k_values': np.array(k_values, dtype=np.int32),
        'ks_train': ks_train,
        'ks_val': ks_val,
        'mesh_type': solver.mesh_type,
        'nx': nx,
        'ny': ny,
        'K_sparse_matrix': K,  # Store the sparse matrix for verification
        'problem_type': 'forward_spmv_freq_normalized',  # Identifier for forward problem with freq normalization
        'normalization_type': 'frequency_aware',
        'reference_frequency_factor': reference_frequency,  # k1=1, k2=1 → factor=2
        'normalization_formula': 'output = (K @ f) * (reference_freq / (k1² + k2²))'
    }
    
    # Save dataset
    os.makedirs(out_dir, exist_ok=True)
    if dataset_name is None:
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = f'fem_forward_spmv_freq_norm_k1k2_{min(k_values)}to{max(k_values)}_{num_samples}s_{ts}.npz'
    
    save_path = os.path.join(out_dir, dataset_name)
    
    # Also save the sparse matrix separately for easy loading
    from scipy.sparse import save_npz
    sparse_matrix_path = os.path.join(out_dir, f'sparse_matrix_K_{ts}.npz')
    save_npz(sparse_matrix_path, K)
    
    np.savez(
        save_path,
        X_train=X_train.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        Y_val=Y_val.astype(np.float32),
        ks_train=ks_train,
        ks_val=ks_val,
        points=meta['points'].astype(np.float32),
        triangles=meta['triangles'].astype(np.int32),
        boundary_nodes=meta['boundary_nodes'].astype(np.int32),
        interior_nodes=meta['interior_nodes'].astype(np.int32),
        theta_field=meta['theta_field'].astype(np.float32),
        nx=nx,
        ny=ny,
        mesh_type=solver.mesh_type,
        problem_type='forward_spmv_freq_normalized',
        normalization_type='frequency_aware',
        reference_frequency_factor=reference_frequency
    )
    
    print(f"Forward SpMV dataset saved to: {save_path}")
    print(f"Sparse matrix K saved to: {sparse_matrix_path}")
    
    # Print some statistics
    print("\nDataset Statistics (with frequency-aware normalization):")
    print(f"  Input (f) range: [{X.min():.3e}, {X.max():.3e}]")
    print(f"  Output (normalized K@f) range: [{Y.min():.3e}, {Y.max():.3e}]")
    print(f"  Input std: {X.std():.3e}")
    print(f"  Output std: {Y.std():.3e}")
    print(f"  Frequency-aware normalization applied: output ∝ (k1² + k2²)⁻¹")
    print(f"  Reference frequency: k1=1, k2=1 (factor=2)")
    print(f"  All outputs normalized to have similar magnitude as k1=1, k2=1 case")

    return X_train, Y_train, X_val, Y_val, meta, solver


def plot_fem_dataset_samples(solver,
                             X_split,
                             Y_split,
                             ks_split,
                             split_name='train',
                             num_samples=5,
                             save_path=None):
    """
    Plot a few samples (RHS and solution) on the unstructured mesh using triangulation.
    """
    assert X_split.shape[0] >= num_samples and Y_split.shape[0] >= num_samples, "Not enough samples to plot"

    # Create triangulation for plotting
    triang = mtri.Triangulation(solver.points[:, 0], solver.points[:, 1], solver.triangles)

    # Choose indices to visualize
    idx = np.linspace(0, X_split.shape[0] - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2.4 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])
    for row, i in enumerate(idx):
        f_vec = X_split[i]
        u_vec = Y_split[i]
        k1, k2 = ks_split[i]

        ax_f = axes[row, 0]
        cf = ax_f.tricontourf(triang, f_vec, levels=20, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax_f, fraction=0.046, pad=0.04)
        ax_f.set_title(f'{split_name} f (k1={k1}, k2={k2})')
        ax_f.set_aspect('equal')

        ax_u = axes[row, 1]
        cu = ax_u.tricontourf(triang, u_vec, levels=20, cmap='viridis')
        plt.colorbar(cu, ax=ax_u, fraction=0.046, pad=0.04)
        ax_u.set_title(f'{split_name} u')
        ax_u.set_aspect('equal')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Samples plot saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        print("Warning: No save_path provided for dataset samples plot")


def plot_forward_spmv_verification(solver, X_split, Y_split, ks_split, K_matrix, 
                                   split_name='train', num_samples=5, save_path=None, 
                                   frequency_normalized=False):
    """
    Enhanced verification of forward SpMV: compare Y (stored K@f) with actual K@X computation.
    Includes both 1D vector plots and 2D field visualizations.
    """
    assert X_split.shape[0] >= num_samples and Y_split.shape[0] >= num_samples, "Not enough samples to plot"
    
    # Create triangulation for 2D plotting
    triang = mtri.Triangulation(solver.points[:, 0], solver.points[:, 1], solver.triangles)
    
    # Choose indices to visualize
    idx = np.linspace(0, X_split.shape[0] - 1, num_samples, dtype=int)
    
    # Create comprehensive plot: 1D verification + 2D visualization
    fig = plt.figure(figsize=(20, 5 * num_samples))
    
    max_error = 0.0
    
    for row, i in enumerate(idx):
        f_vec = X_split[i]  # Input vector
        stored_output = Y_split[i]  # Stored K @ f
        k1, k2 = ks_split[i]
        
        # Recompute K @ f 
        computed_output_raw = K_matrix @ f_vec  # Raw SpMV result
        
        # Apply frequency-aware normalization if the dataset uses it
        if frequency_normalized:
            frequency_factor = k1*k1 + k2*k2
            reference_frequency = 1*1 + 1*1  # k1=1, k2=1 reference
            computed_output = computed_output_raw * (reference_frequency / frequency_factor)
        else:
            computed_output = computed_output_raw
        
        # Compute error
        error = np.abs(stored_output - computed_output)
        max_error = max(max_error, np.max(error))
        rmse = np.sqrt(np.mean(error**2))
        
        # Create subplot grid for this sample
        base_idx = row * 5 + 1
        
        # Plot 1: Input vector f (1D)
        ax1 = plt.subplot(num_samples, 5, base_idx)
        ax1.plot(f_vec, 'b-', linewidth=0.8, alpha=0.7)
        boundary_mask = np.zeros(len(f_vec), dtype=bool)
        boundary_mask[solver.boundary_nodes] = True
        ax1.scatter(np.where(boundary_mask)[0], f_vec[boundary_mask], 
                   c='red', s=15, alpha=0.8, label=f'Boundary (zeros)')
        ax1.set_title(f'Input f (k1={k1}, k2={k2})')
        ax1.set_xlabel('Node index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: Input field f (2D)
        ax2 = plt.subplot(num_samples, 5, base_idx + 1)
        cf = ax2.tricontourf(triang, f_vec, levels=20, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f'Input f (2D)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # Plot 3: Stored vs Computed K@f (1D)
        ax3 = plt.subplot(num_samples, 5, base_idx + 2)
        ax3.plot(stored_output, 'r-', linewidth=0.8, alpha=0.8, label='Stored K@f')
        ax3.plot(computed_output, 'g--', linewidth=0.8, alpha=0.8, label='Computed K@f')
        ax3.set_title(f'Output Comparison\n(RMSE={rmse:.2e})')
        ax3.set_xlabel('Node index')
        ax3.set_ylabel('Value')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Output field K@f (2D) - using stored values
        ax4 = plt.subplot(num_samples, 5, base_idx + 3)
        cu = ax4.tricontourf(triang, stored_output, levels=20, cmap='plasma')
        plt.colorbar(cu, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_title(f'Output K@f (2D)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_aspect('equal')
        
        # Plot 5: Error visualization (both 1D and 2D)
        ax5 = plt.subplot(num_samples, 5, base_idx + 4)
        
        # 1D error plot (main)
        line_error = ax5.semilogy(error + 1e-16, 'k-', linewidth=0.8, alpha=0.8, 
                                 label=f'Max: {np.max(error):.2e}')
        ax5.set_xlabel('Node index')
        ax5.set_ylabel('|Stored - Computed|')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # Add small inset for 2D error visualization
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax5_inset = inset_axes(ax5, width="40%", height="40%", loc='upper right')
        error_2d = ax5_inset.tricontourf(triang, error, levels=15, cmap='Reds')
        ax5_inset.set_xticks([])
        ax5_inset.set_yticks([])
        ax5_inset.set_title('2D Error', fontsize=8)
        
        ax5.set_title(f'Error Analysis')
    
    plt.suptitle(f'Enhanced Forward SpMV Verification - {split_name.title()} Set\n'
                 f'Max error across all samples: {max_error:.2e}\n'
                 f'Columns: Input f (1D), Input f (2D), Output Comparison (1D), Output K@f (2D), Error Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Enhanced forward SpMV verification plot saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        print("Warning: No save_path provided for forward SpMV verification plot")
    
    return max_error


def plot_forward_spmv_samples_2d(solver, X_split, Y_split, ks_split, 
                                 split_name='train', num_samples=3, save_path=None, K_matrix=None):
    """
    Plot comprehensive 2D visualization of forward SpMV samples: mesh, input f, output K@f, and sparse matrix K.
    """
    assert X_split.shape[0] >= num_samples and Y_split.shape[0] >= num_samples, "Not enough samples to plot"
    
    # Create triangulation for plotting
    triang = mtri.Triangulation(solver.points[:, 0], solver.points[:, 1], solver.triangles)
    
    # Choose indices to visualize
    idx = np.linspace(0, X_split.shape[0] - 1, num_samples, dtype=int)
    
    # Determine number of columns based on whether we have K_matrix
    n_cols = 4 if K_matrix is not None else 3
    fig_width = 6 * n_cols
    
    # Create comprehensive plot: mesh + data + sparse matrix (if available)
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(fig_width, 4.5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
        
    for row, i in enumerate(idx):
        f_vec = X_split[i]  # Input: sinusoidal field
        output_vec = Y_split[i]  # Output: K @ f
        k1, k2 = ks_split[i]
        
        # Plot 1: Mesh structure with boundary/interior nodes highlighted
        ax_mesh = axes[row, 0]
        ax_mesh.triplot(triang, 'k-', linewidth=0.3, alpha=0.4)
        
        # Highlight boundary and interior nodes
        boundary_points = solver.points[solver.boundary_nodes]
        interior_points = solver.points[solver.interior_nodes]
        
        ax_mesh.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                       c='red', s=15, alpha=0.8, label=f'Boundary ({len(solver.boundary_nodes)})')
        ax_mesh.scatter(interior_points[:, 0], interior_points[:, 1], 
                       c='blue', s=8, alpha=0.6, label=f'Interior ({len(solver.interior_nodes)})')
        
        ax_mesh.set_title(f'{split_name} Sample {i}: Mesh Structure\n'
                         f'{solver.mesh_type.title()} ({solver.n_nodes} nodes, {solver.n_elements} elements)')
        ax_mesh.set_xlabel('x')
        ax_mesh.set_ylabel('y')
        ax_mesh.set_aspect('equal')
        ax_mesh.legend(fontsize=8)
        ax_mesh.grid(True, alpha=0.2)
        
        # Plot 2: Input field f(x,y) with enhanced visualization
        ax_f = axes[row, 1]
        cf = ax_f.tricontourf(triang, f_vec, levels=25, cmap='RdBu_r', extend='both')
        
        # Add contour lines for better visualization
        contour_lines = ax_f.tricontour(triang, f_vec, levels=10, colors='black', 
                                       linewidths=0.5, alpha=0.6)
        ax_f.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Highlight zero values at boundaries
        zero_mask = np.abs(f_vec) < 1e-10
        if np.any(zero_mask):
            zero_points = solver.points[zero_mask]
            ax_f.scatter(zero_points[:, 0], zero_points[:, 1], 
                        c='white', s=20, edgecolors='black', linewidths=0.5, 
                        marker='s', alpha=0.8, label=f'Zeros ({np.sum(zero_mask)})')
            ax_f.legend(fontsize=8, loc='upper right')
        
        cbar_f = plt.colorbar(cf, ax=ax_f, fraction=0.046, pad=0.04)
        cbar_f.set_label('f value', fontsize=10)
        
        ax_f.set_title(f'{split_name} Input: f = sin(2π·{k1}·x)sin(2π·{k2}·y)\n'
                      f'Range: [{f_vec.min():.3f}, {f_vec.max():.3f}]')
        ax_f.set_xlabel('x')
        ax_f.set_ylabel('y')
        ax_f.set_aspect('equal')
        
        # Plot 3: Output field K @ f with enhanced visualization
        ax_u = axes[row, 2]
        cu = ax_u.tricontourf(triang, output_vec, levels=25, cmap='plasma', extend='both')
        
        # Add contour lines
        contour_lines_u = ax_u.tricontour(triang, output_vec, levels=10, colors='white', 
                                         linewidths=0.5, alpha=0.7)
        ax_u.clabel(contour_lines_u, inline=True, fontsize=8, fmt='%.2f')
        
        cbar_u = plt.colorbar(cu, ax=ax_u, fraction=0.046, pad=0.04)
        cbar_u.set_label('K@f value', fontsize=10)
        
        ax_u.set_title(f'{split_name} Output: u = K @ f (SpMV Result)\n'
                      f'Range: [{output_vec.min():.3f}, {output_vec.max():.3f}]')
        ax_u.set_xlabel('x')
        ax_u.set_ylabel('y')
        ax_u.set_aspect('equal')
        
        # Plot 4: Sparse Matrix K (if available)
        if K_matrix is not None and n_cols == 4:
            ax_K = axes[row, 3]
            
            # Show sparse matrix structure
            ax_K.spy(K_matrix, markersize=0.5, alpha=0.8, color='darkblue')
            ax_K.set_title(f'Sparse Matrix K Structure\n'
                          f'{K_matrix.shape[0]}×{K_matrix.shape[1]}, {K_matrix.nnz:,} non-zeros\n'
                          f'Sparsity: {100*(1-K_matrix.nnz/(K_matrix.shape[0]*K_matrix.shape[1])):.2f}%')
            ax_K.set_xlabel('Column index')
            ax_K.set_ylabel('Row index')
            
            # Add matrix statistics as text
            if row == 0:  # Only add stats to first row to avoid repetition
                stats_text = (f'Matrix Stats:\n'
                             f'Size: {K_matrix.shape[0]}×{K_matrix.shape[1]}\n'
                             f'Non-zeros: {K_matrix.nnz:,}\n'
                             f'Density: {100*K_matrix.nnz/(K_matrix.shape[0]*K_matrix.shape[1]):.3f}%\n'
                             f'Max entry: {K_matrix.data.max():.2e}\n'
                             f'Min entry: {K_matrix.data.min():.2e}')
                ax_K.text(1.02, 0.5, stats_text, transform=ax_K.transAxes, 
                         verticalalignment='center', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    columns_desc = "Columns: Mesh, Input f, Output K@f" + (", Sparse Matrix K" if K_matrix is not None else "")
    plt.suptitle(f'Forward SpMV Dataset - {split_name.title()} Samples\n'
                 f'Learning Task: Neural Network(f) ≈ K @ f\n'
                 f'{columns_desc}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Leave space for suptitle
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Enhanced forward SpMV 2D samples plot saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        print("Warning: No save_path provided for forward SpMV 2D samples plot")


def plot_sparse_matrix_analysis(K_matrix, solver, save_path=None):
    """
    Create detailed analysis of the sparse matrix K structure and properties.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Sparse matrix spy plot
    ax1 = axes[0, 0]
    ax1.spy(K_matrix, markersize=0.3, alpha=0.8, color='darkblue')
    ax1.set_title(f'Sparse Matrix K Structure\n{K_matrix.shape[0]}×{K_matrix.shape[1]}, {K_matrix.nnz:,} non-zeros')
    ax1.set_xlabel('Column index')
    ax1.set_ylabel('Row index')
    
    # Plot 2: Non-zero values histogram
    ax2 = axes[0, 1]
    nonzero_vals = K_matrix.data[np.abs(K_matrix.data) > 1e-12]
    ax2.hist(nonzero_vals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_title(f'Matrix Entry Values Distribution\n({len(nonzero_vals):,} entries)')
    ax2.set_xlabel('Matrix Entry Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Min: {nonzero_vals.min():.3e}\nMax: {nonzero_vals.max():.3e}\nMean: {nonzero_vals.mean():.3e}\nStd: {nonzero_vals.std():.3e}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    # Plot 3: Row-wise non-zero count
    ax3 = axes[0, 2]
    row_nnz = np.array([K_matrix.getrow(i).nnz for i in range(K_matrix.shape[0])])
    ax3.plot(row_nnz, 'b-', alpha=0.7, linewidth=0.8)
    ax3.set_title(f'Non-zeros per Row\nAvg: {row_nnz.mean():.1f}, Std: {row_nnz.std():.2f}')
    ax3.set_xlabel('Row index')
    ax3.set_ylabel('Number of non-zeros')
    ax3.grid(True, alpha=0.3)
    
    # Highlight boundary nodes
    if hasattr(solver, 'boundary_nodes'):
        boundary_nnz = row_nnz[solver.boundary_nodes]
        ax3.scatter(solver.boundary_nodes, boundary_nnz, c='red', s=15, alpha=0.8, label='Boundary')
        ax3.legend()
    
    # Plot 4: Matrix bandwidth visualization
    ax4 = axes[1, 0]
    # Compute bandwidth for each row
    bandwidths = []
    for i in range(K_matrix.shape[0]):
        row = K_matrix.getrow(i)
        if row.nnz > 1:
            cols = row.indices
            bandwidth = max(cols) - min(cols)
            bandwidths.append(bandwidth)
        else:
            bandwidths.append(0)
    
    ax4.plot(bandwidths, 'g-', alpha=0.7, linewidth=0.8)
    ax4.set_title(f'Matrix Bandwidth per Row\nMax: {max(bandwidths)}, Avg: {np.mean(bandwidths):.1f}')
    ax4.set_xlabel('Row index')
    ax4.set_ylabel('Bandwidth')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Mesh connectivity visualization
    ax5 = axes[1, 1]
    if hasattr(solver, 'points') and hasattr(solver, 'triangles'):
        triang = mtri.Triangulation(solver.points[:, 0], solver.points[:, 1], solver.triangles)
        ax5.triplot(triang, 'k-', linewidth=0.3, alpha=0.5)
        ax5.scatter(solver.points[:, 0], solver.points[:, 1], c=row_nnz, 
                   s=20, cmap='viridis', alpha=0.8)
        cbar = plt.colorbar(ax5.collections[-1], ax=ax5, fraction=0.046, pad=0.04)
        cbar.set_label('Non-zeros per node', fontsize=10)
        ax5.set_title('Mesh with Matrix Connectivity')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_aspect('equal')
    
    # Plot 6: Sparsity pattern statistics
    ax6 = axes[1, 2]
    # Compute some sparsity metrics
    density = K_matrix.nnz / (K_matrix.shape[0] * K_matrix.shape[1])
    sparsity = 1 - density
    
    # Create a summary plot
    metrics = ['Density (%)', 'Sparsity (%)', 'Avg NNZ/Row', 'Max Bandwidth', 'Condition Est.']
    values = [density*100, sparsity*100, row_nnz.mean(), max(bandwidths), 0]  # Condition number placeholder
    
    # Try to estimate condition number for small matrices
    if K_matrix.shape[0] < 1000:
        try:
            from scipy.sparse.linalg import norm
            K_norm = norm(K_matrix)
            values[4] = K_norm
        except:
            values[4] = np.nan
    
    bars = ax6.bar(range(4), values[:4], color=['blue', 'red', 'green', 'orange'])
    ax6.set_xticks(range(4))
    ax6.set_xticklabels(metrics[:4], rotation=45, ha='right')
    ax6.set_title('Matrix Characteristics')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, values[:4])):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Sparse Matrix K Analysis\n'
                 f'FEM Stiffness Matrix ({solver.mesh_type.title()} Mesh, {solver.n_nodes} nodes, {solver.n_elements} elements)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Sparse matrix analysis plot saved to: {save_path}")
        plt.close()
    else:
        print("Warning: No save_path provided for sparse matrix analysis plot")


# Main execution
def main():
    print("=" * 70)
    print("ENHANCED FEM SOLVER FOR GENERALIZED POISSON EQUATION")
    print("Solving: -∇·(θ∇u) = f with variable coefficient θ")
    print("=" * 70)
    
    # Create solver with unstructured mesh
    print("\n1. Initializing FEM solver with unstructured mesh...")
    solver = FEMPoissonSolver(nx=35, ny=35, mesh_type='unstructured')
    print(f"   - Mesh type: {solver.mesh_type}")
    print(f"   - Mesh: {solver.n_nodes} nodes, {solver.n_elements} elements")
    print(f"   - Boundary nodes: {len(solver.boundary_nodes)}")
    print(f"   - Interior nodes: {len(solver.interior_nodes)}")
    
    # Create theta field (variable coefficient)
    print("\n2. Creating normalized theta field (variable coefficient)...")
    theta_field = solver.create_theta_field(min_val=0.5, max_val=3.0, n_regions=4, seed=42, normalize=True)
    print(f"   - Theta range: [{theta_field.min():.3f}, {theta_field.max():.3f}]")
    print(f"   - Theta mean: {theta_field.mean():.3f}")
    print(f"   - Normalized to [-1, 1]")
    
    # Create input field
    print("\n3. Creating normalized sinusoidal source function...")
    f_field = solver.create_sinusoidal_field(seed=43, normalize=True)
    print(f"   - Sinusoidal function: sin(2π·{solver.k1}·x) × sin(2π·{solver.k2}·y)")
    print(f"   - Frequencies: k1={solver.k1}, k2={solver.k2}")
    print(f"   - Field range: [{f_field.min():.3f}, {f_field.max():.3f}]")
    print(f"   - Normalized to [-1, 1]")
    
    # Assemble system
    print("\n4. Assembling FEM system with variable θ...")
    solver.assemble_system(f_field, theta_field)
    print(f"   - Stiffness matrix: {solver.K.shape} with {solver.K.nnz} non-zeros")
    print(f"   - Sparsity: {100*(1 - solver.K.nnz/(solver.K.shape[0]*solver.K.shape[1])):.2f}%")
    
    # Analyze matrix complexity
    from scipy.sparse import triu
    K_upper = triu(solver.K, k=1)  # Upper triangular part
    max_bandwidth = 0
    avg_row_nnz = solver.K.nnz / solver.K.shape[0]
    
    # Compute bandwidth and irregularity metrics
    row_nnz_var = 0
    for i in range(solver.K.shape[0]):
        row_nnz = solver.K.getrow(i).nnz
        row_nnz_var += (row_nnz - avg_row_nnz) ** 2
        if row_nnz > 0:
            row_data = solver.K.getrow(i)
            col_indices = row_data.indices
            if len(col_indices) > 1:
                bandwidth = max(col_indices) - min(col_indices)
                max_bandwidth = max(max_bandwidth, bandwidth)
    
    row_nnz_var = np.sqrt(row_nnz_var / solver.K.shape[0])
    print(f"   - Max bandwidth: {max_bandwidth}")
    print(f"   - Avg non-zeros per row: {avg_row_nnz:.1f}")
    print(f"   - Row nnz std deviation: {row_nnz_var:.2f} (irregularity measure)")
    
    # Apply boundary conditions
    print("\n5. Applying boundary conditions...")
    solver.apply_boundary_conditions()
    print(f"   - Reduced system size: {solver.K_reduced.shape}")
    
    # Normalize RHS vector
    print("\n5.5. Normalizing RHS vector...")
    f_rhs_raw = solver.f.copy()
    solver.f = solver.normalize_field(solver.f)
    solver.f_reduced = solver.normalize_field(solver.f_reduced)
    print(f"   - Raw RHS range: [{f_rhs_raw.min():.3f}, {f_rhs_raw.max():.3f}]")
    print(f"   - Normalized RHS range: [{solver.f.min():.3f}, {solver.f.max():.3f}]")
    print(f"   - RHS normalized to [-1, 1]")
    
    # Solve
    print("\n6. Solving linear system...")
    u_solution_raw = solver.solve()
    print(f"   - Raw solution range: [{u_solution_raw.min():.3f}, {u_solution_raw.max():.3f}]")
    
    # Normalize solution
    u_solution = solver.normalize_field(u_solution_raw)
    print(f"   - Normalized solution range: [{u_solution.min():.3f}, {u_solution.max():.3f}]")
    print(f"   - Solution normalized to [-1, 1]")
    
    # Verify solution (residual)
    residual = solver.K @ u_solution - solver.f
    residual[solver.boundary_nodes] = 0  # Exclude boundary
    print(f"   - Residual norm: {np.linalg.norm(residual):.2e}")
    
    # Energy norm
    energy = 0.5 * u_solution.T @ (solver.K @ u_solution) - u_solution.T @ solver.f
    print(f"   - Energy functional: {energy:.6f}")
    
    # Visualize
    print("\n7. Creating comprehensive visualizations...")
    visualize_results(solver, f_field, theta_field, u_solution)
    
    print("\n" + "=" * 70)
    print("SOLVER COMPLETE")
    print("=" * 70)
    
    # Additional analysis
    print("\nADDITIONAL ANALYSIS:")
    print("-" * 30)
    
    # Analysis of normalized fields
    theta_mean = theta_field.mean()
    f_max = np.max(np.abs(f_field))
    u_max = np.max(np.abs(u_solution))
    rhs_max = np.max(np.abs(solver.f))
    print(f"Average θ: {theta_mean:.3f} (normalized)")
    print(f"Max |f|: {f_max:.3f} (normalized)")
    print(f"Max |RHS|: {rhs_max:.3f} (normalized)")
    print(f"Max |u|: {u_max:.3f} (normalized)")
    print(f"All fields normalized to [-1, 1] for consistency")
    
    # Condition number estimate (for small systems)
    if solver.n_nodes < 1000:
        from scipy.sparse.linalg import norm
        K_norm = norm(solver.K_reduced)
        print(f"Matrix norm: {K_norm:.2e}")

if __name__ == "__main__":
    main()