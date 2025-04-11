# 2D Diffusion Simulation using Finite Differences 🌊🧮

Simulation of stationary diffusion in a 2D domain using the Finite Difference Method (FDM). This project solves anisotropic and isotropic diffusion equations under Dirichlet and Neumann boundary conditions. Implemented in Python using sparse matrix solvers and visualized with Matplotlib.

---

## 📑 Contents

- `Diffusion_Final.py` – Python implementation for solving and visualizing 2D diffusion
- `diffusion_FD_report.pdf` – Report with theoretical derivations, matrix formulation, and results
- `Practical 1 - Diffusion and FD.pdf` – Course sheet with mathematical background and tasks

---

## 🧠 Key Features

- ✅ Handles both Dirichlet and Neumann boundary conditions
- ✅ Grid discretization with customizable size
- ✅ Supports varying flux and concentration boundary setups
- ✅ Visualizes concentration heatmaps and convergence
- ✅ Validates numerical results against analytical solutions

---

## 💡 Test Cases

- **Test 1**: Constant boundary concentrations → Flat profile
- **Test 2**: Linear gradient between top and bottom
- **Test 3**: Variable ρ_upper and ρ_lower pairs
- **Test 4**: Numerical vs analytical comparison and convergence plot
- **Flux Studies**: Equal and varied flux in/out with symmetric boundary values

---

## 🚀 How to Run

1. Install required packages:
```bash
pip install numpy matplotlib scipy
