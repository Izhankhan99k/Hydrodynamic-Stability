
import numpy as np
import dedalus.public as de
import matplotlib.pyplot as plt
import logging

# Set up logging to print Dedalus output cleanly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. Physical Parameters (High Inertia / Advection Dominated Regime)
# -------------------------------------------------------------------------
Ra = 1e4        # Rayleigh number (Buoyancy)
Pe = 5000.0     # Péclet number (High advection)
Pr = 1.0        # Prandtl number (Low viscosity -> Re = Pe/Pr = 5000)
S = 2.0         # Stefan number (Coupled phase-change)
Lambda = 0.5    # Solid/Liquid thickness ratio
k = 1.0         # Streamwise wavenumber (m=0 implies 2D)

# -------------------------------------------------------------------------
# 2. Domain & Bases Setup
# -------------------------------------------------------------------------
# Increased size to 128 to capture the continuous wave branches properly
zcoord = de.Coordinate('z')
dist = de.Distributor(zcoord, dtype=np.complex128)
zb = de.ChebyshevT(zcoord, size=128, bounds=(0, 1))
z = dist.local_grid(zb)

# -------------------------------------------------------------------------
# 3. Fields & First-Order Variables
# -------------------------------------------------------------------------
# Base State Velocity (Couette-like linear profile)
U = dist.Field(name='U', bases=zb)
U['g'] = 1 - z

# State vector fields (First-order reduction)
w = dist.Field(name='w', bases=zb)
wz = dist.Field(name='wz', bases=zb)
wzz = dist.Field(name='wzz', bases=zb)
wzzz = dist.Field(name='wzzz', bases=zb)

Tl = dist.Field(name='Tl', bases=zb)
Tlz = dist.Field(name='Tlz', bases=zb)

Ts = dist.Field(name='Ts', bases=zb)
Tsz = dist.Field(name='Tsz', bases=zb)

# Perturbation interface height (Scalar eigenvalue variable)
h = dist.Field(name='h')

# Growth rate (Eigenvalue)
sigma = dist.Field(name='sigma')

# Tau fields for tau-method boundary enforcement
tau_w1 = dist.Field(name='tau_w1')
tau_w2 = dist.Field(name='tau_w2')
tau_w3 = dist.Field(name='tau_w3')
tau_w4 = dist.Field(name='tau_w4')
tau_Tl1 = dist.Field(name='tau_Tl1')
tau_Tl2 = dist.Field(name='tau_Tl2')
tau_Ts1 = dist.Field(name='tau_Ts1')
tau_Ts2 = dist.Field(name='tau_Ts2')

# -------------------------------------------------------------------------
# 4. Operators
# -------------------------------------------------------------------------
D = lambda A: de.Differentiate(A, zcoord)

# Proper basis lifting for Dedalus v3 boundary enforcement
lift_basis = zb.derivative_basis(1)
lift = lambda A: de.Lift(A, lift_basis, -1)

# -------------------------------------------------------------------------
# 5. Generalized Eigenvalue Problem (EVP) Formulation
# -------------------------------------------------------------------------
variables = [w, wz, wzz, wzzz, Tl, Tlz, Ts, Tsz, h,
             tau_w1, tau_w2, tau_w3, tau_w4, 
             tau_Tl1, tau_Tl2, tau_Ts1, tau_Ts2]

problem = de.EVP(variables, eigenvalue=sigma)

# Auxiliary First-Order Equations
problem.add_equation((D(w) - wz + lift(tau_w1), 0))
problem.add_equation((D(wz) - wzz + lift(tau_w2), 0))
problem.add_equation((D(wzz) - wzzz + lift(tau_w3), 0))
problem.add_equation((D(Tl) - Tlz + lift(tau_Tl1), 0))
problem.add_equation((D(Ts) - Tsz + lift(tau_Ts1), 0))

# Orr-Sommerfeld Equation (Liquid)
eq_os = (sigma*(wzz - k**2*w) 
         - Pr*(D(wzzz) - 2*k**2*wzz + k**4*w) 
         + Pe*1j*k*U*(wzz - k**2*w) 
         + (Ra*Pr/Pe)*k**2*Tl 
         + lift(tau_w4))
problem.add_equation((eq_os, 0))

# Heat Equation (Liquid)
eq_Tl = (sigma*Tl 
         - (D(Tlz) - k**2*Tl) 
         + Pe*1j*k*U*Tl 
         - Pe*w 
         + lift(tau_Tl2))
problem.add_equation((eq_Tl, 0))

# Heat Equation (Solid - Mapped to z in [0, 1])
eq_Ts = (sigma*Ts 
         - (1/Lambda**2)*D(Tsz) 
         + k**2*Ts 
         + lift(tau_Ts2))
problem.add_equation((eq_Ts, 0))

# -------------------------------------------------------------------------
# 6. Boundary & Interface Conditions
# -------------------------------------------------------------------------
# At z = 0 (Liquid bottom wall & mapped Solid top wall)
problem.add_equation((w(z=0), 0))
problem.add_equation((wz(z=0), 0))
problem.add_equation((Tl(z=0), 0))
problem.add_equation((Ts(z=0), 0))

# At z = 1 (Liquid-Solid Phase Boundary Interface)
problem.add_equation((w(z=1), 0))
problem.add_equation((wz(z=1), 0))
problem.add_equation((Tl(z=1) - h, 0))
problem.add_equation((Ts(z=1) - h, 0))

# Time-Dependent Stefan Condition (closes the system for h)
eq_stefan = sigma*h + 1/(Lambda*S) * ( (1/Lambda)*Tsz(z=1) + Tlz(z=1) )
problem.add_equation((eq_stefan, 0))

# -------------------------------------------------------------------------
# 7. Solve Dense Matrix & Filter Eigenvalues
# -------------------------------------------------------------------------
logger.info("Building solver and calculating dense eigenvalues...")
solver = problem.build_solver()

# Calculate the eigenvalues (this populates solver.eigenvalues)
solver.solve_dense(solver.subproblems[0])
evals = solver.eigenvalues

# Filter out numerical infinities and NaNs
valid_evals = evals[np.isfinite(evals)]

# Sort by largest real part (most unstable / least stable modes first)
sorted_evals = valid_evals[np.argsort(valid_evals.real)[::-1]]
leading_sigma = sorted_evals[0]
logger.info(f"Leading Eigenvalue (σ): Re = {leading_sigma.real:.6f}, Im = {leading_sigma.imag:.6f}")

# -------------------------------------------------------------------------
# 8. Plot the Eigenvalue Spectrum
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.scatter(sorted_evals.real, sorted_evals.imag, marker='o', alpha=0.7, s=20)

# Broad viewing window to capture the high-Pe wave branches
plt.xlim(-150, 10)  
plt.ylim(-5500, 500) 

plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Re(σ) - Growth Rate")
plt.ylabel("Im(σ) - Wave Frequency")
plt.title("Filtered Eigenvalue Spectrum (High Inertia Regime)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 9. Extract & Plot the Leading Eigenfunctions
# -------------------------------------------------------------------------
# Find the exact index of our leading mode in the original, unfiltered solver array
leading_idx = np.argmin(np.abs(evals - leading_sigma))
solver.set_state(leading_idx, solver.subproblems[0])

# Extract the 1D spatial grid and the spatial mode arrays
z_grid = z.evaluate()['g'][0] 
w_mode = w['g']
Tl_mode = Tl['g']
Ts_mode = Ts['g']

# Create the eigenfunction plot
plt.figure(figsize=(12, 5))

# Subplot 1: Velocity perturbation (Liquid only)
plt.subplot(1, 2, 1)
plt.plot(z_grid, w_mode.real, label='Re($\hat{w}$)', color='blue', linewidth=2)
plt.axvline(1, color='black', linestyle=':', label='Phase Boundary (z=1)')
plt.title(f"Vertical Velocity Perturbation\nMode: $\sigma$ = {leading_sigma.real:.2f} + {leading_sigma.imag:.2f}i")
plt.xlabel("Mapped Domain (z)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Subplot 2: Temperature perturbations (Liquid and Solid)
plt.subplot(1, 2, 2)
plt.plot(z_grid, Tl_mode.real, label='Re($\hat{\theta}_l$)', color='red', linewidth=2)
plt.plot(z_grid, Ts_mode.real, label='Re($\hat{\theta}_s$)', color='orange', linewidth=2)
plt.axvline(1, color='black', linestyle=':')
plt.title("Temperature Perturbations")
plt.xlabel("Mapped Domain (z)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print the interface perturbation amplitude (the scalar variable 'h')
logger.info(f"Interface perturbation amplitude (h): {h['c'][0].real:.6f} + {h['c'][0].imag:.6f}i")