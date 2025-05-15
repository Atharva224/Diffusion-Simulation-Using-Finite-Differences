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
    
Keeping the original work safe. Email atharvasinnarkar@gmail.com for the full code file and mention the proper usecase.
















