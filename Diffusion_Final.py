import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline

class DiffusionSolver2D:
    def __init__(self, lx, ly, nx, ny):
        """
        Initialize the 2D diffusion solver
        
        Parameters:
        lx, ly: domain dimensions
        nx, ny: number of nodes in each dimension
        """
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.hx = lx / (nx - 1)  # Grid spacing in x
        self.hy = ly / (ny - 1)  # Grid spacing in y
        
    def assemble_matrix(self, j_left, j_right, rho_upper, rho_lower):
        """
        Assemble the coefficient matrix and RHS vector with boundary conditions
        """
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        D = 1.0  # Diffusion coefficient (normalized)
        
        # Coefficients for the central difference scheme
        cx = D / (self.hx ** 2)
        cy = D / (self.hy ** 2)
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                
                # Upper boundary
                if j == 0:
                    A[idx, idx] = 1
                    b[idx] = rho_upper
                # Lower boundary
                elif j == self.ny - 1:
                    A[idx, idx] = 1
                    b[idx] = rho_lower
                # Left boundary
                elif i == 0:
                    A[idx, idx] = -3/(2*self.hx)
                    A[idx, idx+1] = 2/self.hx
                    A[idx, idx+2] = -1/(2*self.hx)
                    b[idx] = -j_left/D
                # Right boundary
                elif i == self.nx - 1:
                    A[idx, idx] = 3/(2*self.hx)
                    A[idx, idx-1] = -2/self.hx
                    A[idx, idx-2] = 1/(2*self.hx)
                    b[idx] = j_right/D
                # Interior nodes
                else:
                    A[idx, idx] = -2 * (cx + cy)
                    A[idx, idx-1] = cx  # Left neighbor
                    A[idx, idx+1] = cx  # Right neighbor
                    A[idx, idx-self.nx] = cy  # Upper neighbor
                    A[idx, idx+self.nx] = cy  # Lower neighbor
        
        return A.tocsc(), b
    
    def solve(self, j_left, j_right, rho_upper, rho_lower):
        """
        Solve the diffusion equation with given boundary conditions
        """
        A, b = self.assemble_matrix(j_left, j_right, rho_upper, rho_lower)
        rho = spsolve(A, b)
        return rho.reshape((self.ny, self.nx))

    def plot_solution(self, rho, title=""):
        """
        Plot the solution as a 2D heatmap
        """
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y)
        
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, rho, shading='auto', cmap='viridis')
        plt.colorbar(label='Concentration (mol/m³)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title)
        plt.axis('equal')
        plt.show()

# Separate plot functions for each test case
def plot_test1():
    solver = DiffusionSolver2D(lx=15, ly=15, nx=15, ny=15)
    rho = solver.solve(j_left=0, j_right=0, rho_upper=100, rho_lower=100)
    solver.plot_solution(rho, "Test 1: Constant concentration")

def plot_test2():
    solver = DiffusionSolver2D(lx=15, ly=15, nx=15, ny=15)
    rho = solver.solve(j_left=0, j_right=0, rho_upper=100, rho_lower=50)
    solver.plot_solution(rho, "Test 2: Linear concentration gradient")

def plot_test3():
    solver = DiffusionSolver2D(lx=15, ly=15, nx=15, ny=15)
    test_cases = [
        (200, 100),
        (50, 30) , (100, 40) , (10, 50)
    ]
    for rho_up, rho_low in test_cases:
        rho = solver.solve(j_left=0, j_right=0, rho_upper=rho_up, rho_lower=rho_low)
        solver.plot_solution(rho, f"Test 3: ρ_upper={rho_up}, ρ_lower={rho_low}")

def plot_test4():
    lx = ly = 15
    grid_sizes = [5, 40 ]
    errors = []
    
    def analytical_solution(x, y, rho_upper, rho_lower, ly):
        return rho_lower + (rho_upper - rho_lower) * (ly - y) / ly

    for n in grid_sizes:
        solver = DiffusionSolver2D(lx, ly, nx=n, ny=n)
        rho = solver.solve(j_left=0, j_right=0, rho_upper=100, rho_lower=50)

        # Compare with analytical solution
        y = np.linspace(0, ly, n)
        x = np.linspace(0, lx, n)
        X, Y = np.meshgrid(x, y)
        rho_analytical = analytical_solution(X, Y, 100, 50, ly)
        error = np.max(np.abs(rho - rho_analytical))
        errors.append(error)

        solver.plot_solution(rho, f"Solution for grid size {n}x{n}")


'''
def plot_convergence_study():
    """Study convergence using a high-resolution reference solution with complex boundary conditions"""
    # Domain parameters
    lx = ly = 1.0
    
    # Create reference solution with 300x300 grid
    ref_nx = ref_ny = 300
    ref_solver = DiffusionSolver2D(lx, ly, ref_nx, ref_ny)
    
    # Use boundary conditions that create a more complex 2D pattern
    reference = ref_solver.solve(
        j_left=2.0,      # Strong inflow from left
        j_right=-1.0,    # Moderate outflow from right
        rho_upper=150,   # Higher concentration at top
        rho_lower=50     # Lower concentration at bottom
    )
    
    # Grid sizes reducing by factor of ~2
    grid_sizes = [5,10,20,30]
    errors = []
    
    # Reference grid points
    x_ref = np.linspace(0, lx, ref_nx)
    y_ref = np.linspace(0, ly, ref_ny)
    
    for n in grid_sizes:
        # Solve with current grid size
        solver = DiffusionSolver2D(lx, ly, n, n)
        solution = solver.solve(
            j_left=2.0,
            j_right=-1.0,
            rho_upper=150,
            rho_lower=50
        )
        
        # Create grids for current solution
        x = np.linspace(0, lx, n)
        y = np.linspace(0, ly, n)
        
        # Interpolate solution to match reference grid points
        interpolator = RectBivariateSpline(y, x, solution)
        interpolated = interpolator(y_ref, x_ref)
        
        # Calculate error
        error = np.max(np.abs(interpolated - reference))
        errors.append(error)



    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.plot(grid_sizes, errors, 'o-')
    plt.xlabel('Grid size')
    plt.ylabel('Maximum error')
    plt.grid(True)
    plt.show()

'''

def plot_convergence_study():

    # Domain parameters
    lx = ly = 1.0  

    # Create a high-resolution reference solution
    ref_nx = ref_ny = 300  
    ref_solver = DiffusionSolver2D(lx, ly, ref_nx, ref_ny)

    # Apply a flux-based boundary condition
    reference = ref_solver.solve(
        j_left=2.0,      # Inflow on the left
        j_right=-1.0,    # Outflow on the right
        rho_upper=150,   # Upper boundary fixed concentration
        rho_lower=50     # Lower boundary fixed concentration
    )

    # Compute the total integrated concentration for the reference solution
    dx_ref = lx / (ref_nx - 1)
    dy_ref = ly / (ref_ny - 1)
    integral_reference = np.sum(reference) * dx_ref * dy_ref  # Numerical integration

    # Grid sizes to test
    grid_sizes = [5, 10, 20, 30]
    errors = []

    for n in grid_sizes:
        # Solve for the current grid size
        solver = DiffusionSolver2D(lx, ly, n, n)
        solution = solver.solve(
            j_left=2.0,
            j_right=-1.0,
            rho_upper=150,
            rho_lower=50
        )

        # Compute total concentration integral
        dx = lx / (n - 1)
        dy = ly / (n - 1)
        integral_solution = np.sum(solution) * dx * dy

        # Compute relative error
        error = abs(integral_solution - integral_reference) / integral_reference
        errors.append(error)

    # Plot the convergence trend
    plt.figure(figsize=(8, 6))
    plt.plot(grid_sizes, errors, 'o-', label="Relative error in total concentration")
    plt.xlabel("Grid size")
    plt.ylabel("Relative Error")
    plt.yscale("log") 
    plt.grid(True)
    plt.legend()
    plt.title("Convergence of Integrated Concentration with Grid Refinement")
    plt.show()





def plot_flux_study():
    """
    Study the flux distribution with equal left/right fluxes and equal upper/lower concentrations
    """
    # Domain parameters
    lx = ly = 1.0
    nx = ny = 50  # Use 50x50 grid for clear visualization
    
    solver = DiffusionSolver2D(lx, ly, nx, ny)
    
    # Set equal boundary conditions
    rho_upper = 120.0  # Upper concentration
    rho_lower = 120.0  # Lower concentration (equal to upper)
    j_left = 1.0       # Left flux
    j_right = 1.0      # Right flux (equal to left)
    
    # Solve the system
    solution = solver.solve(
        j_left=j_left,
        j_right=j_right,
        rho_upper=rho_upper,
        rho_lower=rho_lower
    )
    
    # Create the plot
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(10, 8))
    
    # Plot using pcolormesh for better visualization
    plt.pcolormesh(X, Y, solution, cmap='viridis', shading='auto')
    plt.colorbar(label='Concentration (mol/m³)')
    
    # Add grid lines to match the reference image
    plt.grid(True, color='black', alpha=0.2)
    
    # Add labels
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Flux Distribution (j_left = j_right, ρ_upper = ρ_lower)')
    
    plt.show()
    
    return solution






def plot_flux_study_extended():
    """
    Study the flux distribution with varied flux values and matching in- and out-flux.
    """
    # Domain parameters
    lx = ly = 1.0
    nx = ny = 50  # Use 50x50 grid for clear visualization
    
    solver = DiffusionSolver2D(lx, ly, nx, ny)
    
    # Define test cases with varied flux values
    test_cases = [
        (1.0, -1.0, 120.0, 80.0),  # Positive inflow, negative outflow
        (-1.0, 1.0, 150.0, 50.0),  # Negative inflow, positive outflow
        (0.5, -0.5, 100.0, 100.0),  # Symmetrical fluxes, constant concentrations
        (2.0, -2.0, 200.0, 50.0)   # Higher inflow/outflow values
    ]
    
    for i, (j_left, j_right, rho_upper, rho_lower) in enumerate(test_cases):
        # Solve the system for each test case
        solution = solver.solve(
            j_left=j_left,
            j_right=j_right,
            rho_upper=rho_upper,
            rho_lower=rho_lower
        )
        
        # Create the plot
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y)
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, solution, cmap='viridis', shading='auto')
        plt.colorbar(label='Concentration (mol/m³)')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title(f'Flux Study {i+1}: j_left={j_left}, j_right={j_right}, '
                  f'ρ_upper={rho_upper}, ρ_lower={rho_lower}')
        plt.axis('equal')
        plt.grid(True, color='black', alpha=0.2)
        plt.show()








if __name__ == "__main__":
    plot_test1()
    plot_test2()
    plot_test3()
    plot_test4()

    # Run convergence study
    plot_convergence_study()
    
    # Run flux study
    plot_flux_study()

    # Run extended flux study
    plot_flux_study_extended()


















