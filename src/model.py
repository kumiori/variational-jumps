import numpy as np
from scipy.optimize import root_scalar


class PeelingModel:
    def __init__(self, params):
        self.params = params
        self.G0 = params["G0"]
        self.G1 = params["G1"]
        self.L0 = params["L0"]
        self.L1 = params["L1"]
        self.N = params["N"]
        self.P = self.L0 + self.L1
        self.G_c = self._compile_toughness_function()

    def _compile_toughness_function(self):
        def G_c(x):
            mod_x = np.mod(x, self.P)
            return np.where(mod_x < self.L0, self.G0, self.G1)

        return np.vectorize(G_c)

    def get_right_Gc(self, l, tol=1e-3):
        mod_l = np.mod(l + tol, self.P)
        return self.G0 if mod_l < self.L0 else self.G1

    def energy(
        self,
        l,
        t,
    ):
        P = self.P
        L0 = self.L0
        L1 = self.L1
        G0 = self.G0
        G1 = self.G1

        k = np.floor(l / P)
        rem = l - k * P
        rem_G0 = np.minimum(rem, L0)
        rem_G1 = np.maximum(0, rem - L0)
        toughness_integral = k * (G0 * L0 + G1 * L1) + G0 * rem_G0 + G1 * rem_G1
        return -self.N * t**2 / (2 * l) + toughness_integral

    def dE(self, l, t, tol=1e-3):
        return self.N * t**2 / (2 * l**2) + self.G_c(l)

    def d2E(self, l, t, tol=1e-4):
        P = self.P

        smooth_curvature = -self.N * t**2 / (l**3)
        singular_curvature = 0.0
        k = np.floor(l / P)
        rem = l - k * P
        G0, G1 = self.G0, self.G1

        if np.isclose(rem, self.L0, atol=tol):
            # print("singular curvature +")
            singular_curvature += G1 - G0
        elif np.isclose(rem, self.L0 + self.L1, atol=tol):
            # print("singular curvature -")
            singular_curvature += G0 - G1
        else:
            singular_curvature = 0.0

        return smooth_curvature + singular_curvature

    def integrate_fast_flow(self, l0, t, s_max=5.0, dt=0.01):
        l_vals = [l0]
        s_vals = [0.0]
        diss = 0.0
        drives = [-self.dE(l0, t)]
        l = l0
        s = 0.0

        while s < s_max:
            # drive = fast_flow_rhs(l, t)
            drive = max(0.0, -self.dE(l, t))
            if drive <= 1e-6:
                break  # stability reached

            dl = drive * dt
            l += dl
            s += dt
            diss += drive**2 * dt

            l_vals.append(l)
            s_vals.append(s)
            drives.append(drive)

        return np.array(s_vals), np.array(l_vals), diss, drives

    # Define a function that solves the energy conservation equation
    def solve_energy_conservation_jump(
        self, L, T, L_max=5.0, dx=1e-3, tol=1e-3, num_subintervals=5
    ):
        """
        Solves the equation:
            -N T^2 / (2*L) = -N T^2 / (2*L_+) + ∫_{L}^{L_+} G_c(x) dx
        for L_+ > L given an instability at (L, T).

        Parameters:
            L : float       -- current (unstable) crack length
            T : float       -- fixed loading time
            L_max : float   -- upper bound for crack search
            dx : float      -- step size for numerical integration
            tol : float     -- tolerance for root solver

        Returns:
            L_plus : float  -- solution of the energy conservation equation
        """
        G_c_func = self.G_c
        # LHS is fixed
        N_val = self.N
        lhs_val = -N_val * T**2 / (2 * L)

        # RHS = -N T^2 / 2* L_+ + ∫ G_c from L to L_+
        def rhs(L_plus):
            xs = np.arange(L, L_plus, dx)
            gc_vals = G_c_func(xs)
            integral = np.sum(gc_vals) * dx
            return -N_val * T**2 / (2 * L_plus) + integral

        # Residual
        def residual(L_plus):
            return rhs(L_plus) - lhs_val

        # Divide into subintervals
        brackets = np.linspace(L + tol, L_max, num_subintervals + 1)

        for i in range(num_subintervals):
            a, b = brackets[i], brackets[i + 1]
            try:
                fa, fb = residual(a), residual(b)
                # print(f"Bracket ({a:.4f}, {b:.4f}) with fa={fa:.4f}, fb={fb:.4f}")
                if fa * fb < 0:
                    # print(f"Trying subinterval ({a:.4f}, {b:.4f})")
                    sol = root_scalar(
                        residual, bracket=(a, b), method="brentq", xtol=tol
                    )
                    if sol.converged:
                        return sol.root
            except Exception as e:
                print(f"  Error in subinterval ({a:.4f}, {b:.4f}): {e}")
                continue

    def run_variational_flow(
        self,
        t_grid,
        l_0,
    ):
        get_right_Gc = self.get_right_Gc
        energy = self.energy
        dE = self.dE
        d2E = self.d2E
        N_val = self.N
        jump = self.integrate_fast_flow

        l_path = np.zeros_like(t_grid)
        dissipation_path = np.zeros_like(t_grid)
        drive_path = np.zeros_like(t_grid)
        curvature_path = np.zeros_like(t_grid)

        l_current = l_0
        dissipation_current = 0.0
        delta = 1e-3

        for i, t in enumerate(t_grid):
            if i == 0:
                l_path[i] = l_0
                continue

            _local_Gc = get_right_Gc(l_path[i - 1], tol=delta)
            slope = np.sqrt(-N_val / (2 * _local_Gc))
            l_guess = slope * t
            l_current = max(l_guess, l_path[i - 1])
            l_path[i] = l_current

            drive = dE(l_current, t, tol=delta)
            curvature = d2E(l_current, t, tol=1e-2)

            drive_path[i] = drive
            curvature_path[i] = curvature

            if drive < -1e-10:
                _s, _l, diss, drives = jump(l_path[i], t, s_max=5.0, dt=0.01)
                dissipation_current += diss
                l_path[i] = _l[-1]

            dissipation_path[i] = dissipation_current

        E_total = [energy(l, t) for l, t in zip(l_path, t_grid)]

        return {
            "l_path": l_path,
            "t_grid": t_grid,
            "E_path": E_total,
            "dissipation_path": dissipation_path,
            "drive_path": drive_path,
            "curvature_path": curvature_path,
        }

    def run_energy_conserving(
        self,
        t_grid,
        l_0,
    ):
        jump = self.solve_energy_conservation_jump
        get_right_Gc = self.get_right_Gc
        energy = self.energy
        dE = self.dE
        d2E = self.d2E
        N_val = self.N

        l_path = np.zeros_like(t_grid)
        drive_path = np.zeros_like(t_grid)
        curvature_path = np.zeros_like(t_grid)
        delta = 1e-3

        for i, t in enumerate(t_grid):
            if i == 0:
                l_path[i] = l_0
                continue

            _local_Gc = get_right_Gc(l_path[i - 1], tol=delta)
            slope = np.sqrt(-N_val / (2 * _local_Gc))
            l_guess = slope * t
            l_current = max(l_guess, l_path[i - 1])
            l_path[i] = l_current

            drive = dE(l_current, t, tol=delta)
            curvature = d2E(l_current, t, tol=1e-2)

            drive_path[i] = drive
            curvature_path[i] = curvature

            if drive < -1e-10:
                L_val = l_current
                T_val = t
                L_plus = jump(
                    L=L_val,
                    T=T_val,
                    L_max=L_val + 3,
                    tol=1e-2,
                    num_subintervals=5,
                )
                l_path[i] = L_plus

        E_total = [energy(l, t) for l, t in zip(l_path, t_grid)]

        return {
            "l_path": l_path,
            "t_grid": t_grid,
            "E_path": E_total,
            "dissipation_path": np.zeros_like(t_grid),  # Optional: fill if needed
            "drive_path": drive_path,
            "curvature_path": curvature_path,
        }

    def run_global_minimizer(
        self,
        t_grid,
        l_0,
        l_max=5.0,
        dl=1e-3,
        delta=1e-2,
        **kwargs,
    ):
        energy = self.energy
        dE = self.dE
        d2E = self.d2E
        # get_right_Gc = self.get_right_Gc,

        l_path = np.full_like(t_grid, np.nan)
        l_prev = l_0
        drive_path = np.zeros_like(t_grid)
        curvature_path = np.zeros_like(t_grid)

        for i, t in enumerate(t_grid):
            if i == 0:
                l_path[i] = l_0
                continue

            # Candidate search range
            l_horiz = l_prev + 5 if l_prev + 5 < l_max else l_max
            # l_horiz = l_max
            l_candidates = np.arange(l_prev, l_horiz, dl)
            energies = [energy(l, t) for l in l_candidates]
            l_current = l_candidates[np.argmin(energies)]

            l_path[i] = l_current
            l_prev = l_current

            drive = dE(l_current, t, tol=delta)
            curvature = d2E(l_current, t, tol=1e-2)

            drive_path[i] = drive
            curvature_path[i] = curvature

        E_total = [energy(l, t) for l, t in zip(l_path, t_grid)]

        return {
            "l_path": l_path,
            "t_grid": t_grid,
            "E_path": E_total,
            "dissipation_path": np.zeros_like(t_grid),  # Optional: fill if needed
            "drive_path": drive_path,
            "curvature_path": curvature_path,
        }
