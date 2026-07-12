import numpy as np
import dedalus.public as d3
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Fixed parameters
a = 0.0
G = 0.0
Bo = 1000.0

# Resolution grid
N_k = 60
N_Re = 60
k_space = np.logspace(-2, 2, N_k)
Re_space = np.logspace(-1, 5, N_Re)

def solve_evp(N_res, Re_val, k_val):
    """Solve the EVP for a single (Re, k) pair at given resolution."""
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N_res, bounds=(-1, 0))
    z = dist.local_grid(basis)
    
    # Base flow
    U = dist.Field(name='U', bases=basis)
    U['g'] = a * z**2 + (a + 1) * z + 1
    Uz = dist.Field(name='Uz', bases=basis)
    Uz['g'] = 2 * a * z + (a + 1)
    Uzz = dist.Field(name='Uzz', bases=basis)
    Uzz['g'] = 2 * a * np.ones_like(z)
    
    # Fields
    phi = dist.Field(name='phi', bases=basis)
    phiz = dist.Field(name='phiz', bases=basis)
    Lphi = dist.Field(name='Lphi', bases=basis)
    Lphiz = dist.Field(name='Lphiz', bases=basis)
    eta = dist.Field(name='eta')
    c = dist.Field(name='c')
    
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    tau3 = dist.Field(name='tau3')
    tau4 = dist.Field(name='tau4')
    
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    
    ns = dict(
        Re=Re_val, k=k_val, a=a, G=G, Bo=Bo,
        phi=phi, phiz=phiz, Lphi=Lphi, Lphiz=Lphiz, eta=eta, c=c,
        tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4,
        dz=dz, lift=lift, U=U, Uz=Uz, Uzz=Uzz
    )
    
    problem = d3.EVP([phi, phiz, Lphi, Lphiz, eta, tau1, tau2, tau3, tau4],
                     eigenvalue=c, namespace=ns)
    problem.add_equation("phiz - dz(phi) + lift(tau1) = 0")
    problem.add_equation("Lphi - dz(phiz) + (k**2)*phi + lift(tau2) = 0")
    problem.add_equation("Lphiz - dz(Lphi) + lift(tau3) = 0")
    problem.add_equation("1j*k*Re*(U*Lphi - Uzz*phi) - (dz(Lphiz) - (k**2)*Lphi) - 1j*k*Re*c*Lphi + lift(tau4) = 0")
    
    problem.add_equation("phi(z=-1) = 0")
    problem.add_equation("phiz(z=-1) = 0")
    problem.add_equation("-eta*U(z=0) - phi(z=0) + c*eta = 0")
    problem.add_equation("eta*Uzz(z=0) + Lphi(z=0) + 2*(k**2)*phi(z=0) = 0")
    problem.add_equation("Lphiz(z=0) - 2*(k**2)*phiz(z=0) + 1j*k*Re*(Uz(z=0)*phi(z=0) - U(z=0)*phiz(z=0)) - 1j*k*Re*G*eta*(1 + (k**2)/Bo) + 1j*k*Re*c*phiz(z=0) = 0")
    
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

def convergence_filter(ev_lo, ev_hi, tol=0.01):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    phys_mask = np.isfinite(ev_lo)
    if len(good) == 0:
        return ev_lo[phys_mask]
    converged = np.array(good)
    return ev_lo[converged[phys_mask[converged]]]

def compute_point(args):
    """Worker function for a single (i, j) grid point."""
    i, j, Re_val, k_val = args
    # Use two resolutions
    ev1 = solve_evp(180, Re_val, k_val)
    ev2 = solve_evp(200, Re_val, k_val)
    evals = convergence_filter(ev1, ev2, tol=0.01)
    if len(evals) > 0:
        max_ci = np.max(evals.imag)
    else:
        max_ci = -999.0
    return i, j, max_ci

def main():
    # Prepare all parameter pairs with indices
    tasks = []
    for i, Re_val in enumerate(Re_space):
        for j, k_val in enumerate(k_space):
            tasks.append((i, j, Re_val, k_val))
    
    total = len(tasks)
    print(f"Computing {total} points using multiprocessing...")
    
    # Use number of processes = CPU cores (leave one free)
    n_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"Using {n_workers} workers.")
    
    # Initialize result array
    max_ci_grid = np.zeros((N_Re, N_k))
    
    # Submit tasks and collect results
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(compute_point, task): task for task in tasks}
        
        # Process completed futures as they arrive (for progress)
        completed = 0
        for future in as_completed(futures):
            i, j, val = future.result()
            max_ci_grid[i, j] = val
            completed += 1
            if completed % 300 == 0 or completed == total:
                print(f"Progress: {completed}/{total} points done.")
    
    # Save data
    np.savez("a0g0.npz",
             k_grid=k_space,          # Note: we store 1D arrays, not meshgrid
             Re_grid=Re_space,
             max_ci_grid=max_ci_grid,
             a=a, G=G, Bo=Bo)
    print("Data saved successfully to 'a0g0.npz'.")

if __name__ == "__main__":
    main()