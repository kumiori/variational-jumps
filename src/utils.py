import sympy as sp
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import root_scalar


x = sp.symbols("x")


def set_model_params(params):
    global G0_val, G1_val, L0_val, L1_val, N_val, _G_c, P

    G0_val = params["G0"]
    G1_val = params["G1"]
    L0_val = params["L0"]
    L1_val = params["L1"]
    N_val = params["N"]

    # Define piecewise toughness function
    P = L0_val + L1_val

    def G_c(x):
        mod_x = np.mod(x, P)
        return np.where(mod_x < L0_val, G0_val, G1_val)

    _G_c = np.vectorize(G_c)

    P = L0_val + L1_val


params = {"G0": 1, "G1": 2, "L0": 0.3, "L1": 0.7, "N": -1, "t": 1.0}

set_model_params(params)

G_c_expr = sp.Piecewise(
    (G0_val, (x % P) < L0_val),
    (G1_val, True),  # G1 applies for the rest of the period
)
_G_c = sp.lambdify(x, G_c_expr, modules="numpy")


# Map toughness to grayscale color (lighter = weaker, darker = tougher)
def toughness_to_gray(G_val, G_min, G_max, alpha=0.3):
    shade = 1.0 - (G_val - G_min) / (G_max - G_min)
    return (shade, shade, shade, alpha)


# Plot background toughness stripes
def add_toughness_background(
    ax, x_max, G0, G1, L0, L1, alpha=0.3, orientation="vertical"
):
    P = L0 + L1
    G_min, G_max = min(G0, G1), max(G0, G1)
    i = 0
    while i * P < x_max:
        start0 = i * P
        end0 = start0 + L0
        end1 = start0 + P

        if orientation == "vertical":
            # Patch for G0 (weaker)
            ylim = ax.get_ylim()[1]
            ax.add_patch(
                Rectangle(
                    (start0, 0),
                    L0,
                    ylim,  # height = 1.0 normalized (scale will stretch it)
                    facecolor=toughness_to_gray(G0, G_min, G_max, alpha),
                    edgecolor="none",
                )
            )

            # Patch for G1 (tougher)
            ax.add_patch(
                Rectangle(
                    (end0, 0),
                    L1,
                    ylim,
                    facecolor=toughness_to_gray(G1, G_min, G_max, alpha),
                    edgecolor="none",
                )
            )
        else:
            xlim = ax.get_xlim()[1]
            # Patch for G0 (weaker)
            ax.add_patch(
                Rectangle(
                    (0, start0),
                    xlim,
                    L0,  # width = 1.0 normalized (scale will stretch it)
                    facecolor=toughness_to_gray(G0, G_min, G_max, alpha),
                    edgecolor="none",
                )
            )

            # Patch for G1 (tougher)
            ax.add_patch(
                Rectangle(
                    (0, end0),
                    xlim,
                    L1,
                    facecolor=toughness_to_gray(G1, G_min, G_max, alpha),
                    edgecolor="none",
                )
            )
        i += 1


t_val = 0


def get_right_Gc(l, tol=1e-3):
    mod_l = np.mod(l + tol, P)
    return G0_val if mod_l < L0_val else G1_val


# Define pointwise energy, gradient, and curvature (first and second derivatives)
def energy_pointwise(l, t=t_val):
    k = np.floor(l / P)
    rem = l - k * P
    rem_G0 = np.minimum(rem, L0_val)
    rem_G1 = np.maximum(0, rem - L0_val)
    toughness_integral = (
        k * (G0_val * L0_val + G1_val * L1_val) + G0_val * rem_G0 + G1_val * rem_G1
    )
    return -N_val * t**2 / (2 * l) + toughness_integral


def dE_pointwise(l, t=t_val, tol=1e-3):
    mod_l = np.mod(l + tol, P)
    G = G0_val if mod_l < L0_val else G1_val
    # print(f"mod_l: {mod_l}, G: {G}, {mod_l+tol} < {L0_val}")
    return N_val * t**2 / (2 * l**2) + G


def d2E_pointwise(l, t=t_val, tol=1e-4):
    smooth_curvature = -N_val * t**2 / (l**3)
    singular_curvature = 0.0
    k = np.floor(l / P)
    rem = l - k * P

    if np.isclose(rem, L0_val, atol=tol):
        # print("singular curvature +")
        singular_curvature += G1_val - G0_val
    elif np.isclose(rem, L0_val + L1_val, atol=tol):
        # print("singular curvature -")
        singular_curvature += G0_val - G1_val
    else:
        singular_curvature = 0.0

    return smooth_curvature + singular_curvature


def l_vs_t(l_path, t_grid, t):
    # return interpolated value of l_path at time t
    if t < t_grid[0] or t > t_grid[-1]:
        raise ValueError("t is out of bounds of the time grid")
    return np.interp(t, t_grid, l_path)


def run_evo_with_params(
    params,
    evo,
    kwargs={},
):
    assert params["L0"] + params["L1"] > 0, "L0 and L1 must be positive"
    assert params["L0"] + params["L1"] == 1, "P:=L0 + L1 must be equal to 1"
    set_model_params(params)

    # print("Running evolution with parameters")
    t_grid = np.linspace(params["T_min"], params["T_max"], params["timesteps"])
    l_0 = params["l_0"]

    data = evo(
        t_grid=t_grid,
        l_0=params["l_0"],
        get_right_Gc=get_right_Gc,
        energy_pointwise=energy_pointwise,
        dE_pointwise=dE_pointwise,
        d2E_pointwise=d2E_pointwise,
        **kwargs,
        # jump=integrate_fast_flow,
    )
    data["params"] = params
    return data


def compute_metrics(data, label=None, tol=1e-3):
    l_path = data["l_path"]
    t_grid = data["t_grid"]
    diss = data.get("dissipation_path", np.zeros_like(t_grid))

    # Compute jump sizes
    dl = np.diff(l_path)
    jump_sizes = dl[dl > tol]
    return {
        "label": label,
        "final_length": l_path[-1],
        "max_jump": np.max(jump_sizes) if len(jump_sizes) else 0.0,
        "n_jumps": len(jump_sizes),
        "jump_sizes": jump_sizes,
        "total_dissipation": diss[-1],
        "avg_jump": np.mean(jump_sizes) if len(jump_sizes) else 0.0,
        "var_jump": np.var(jump_sizes) if len(jump_sizes) else 0.0,
    }


def compute_energy(l_array, t_val=t_val):
    k = np.floor(l_array / P)
    rem = l_array - k * P
    rem_G0 = np.minimum(rem, L0_val)
    rem_G1 = np.maximum(0, rem - L0_val)
    toughness_integral = (
        k * (G0_val * L0_val + G1_val * L1_val) + G0_val * rem_G0 + G1_val * rem_G1
    )
    energy = -N_val * t_val**2 / (2 * l_array) + toughness_integral
    return energy


def compute_surface_energy(l_array):
    k = np.floor(l_array / P)
    rem = l_array - k * P
    rem_G0 = np.minimum(rem, L0_val)
    rem_G1 = np.maximum(0, rem - L0_val)
    surface_energy = (
        k * (G0_val * L0_val + G1_val * L1_val) + G0_val * rem_G0 + G1_val * rem_G1
    )
    return surface_energy


def compute_energy_derivative(l_array, side="right", t_val=t_val):
    dE_dl = -N_val * t_val**2 / (2 * l_array**2) + compute_surface_energy_derivative(
        l_array, side
    )

    return dE_dl


def compute_surface_energy_derivative(l_array, side="right"):
    k = np.floor(l_array / P)
    rem = np.mod(l_array, P)
    if side == "left":
        dE_dl = np.where(rem <= L0_val, G0_val, G1_val)
    elif side == "right":
        dE_dl = np.where(rem < L0_val, G0_val, G1_val)
    else:
        raise ValueError("Side must be 'left' or 'right'")

    return dE_dl


def fast_flow_rhs(l, t):
    f = N_val * t**2 / (2 * l**2) - _G_c(l)
    return max(0.0, f)


def integrate_fast_flow(l0, t, s_max=5.0, dt=0.01):
    l_vals = [l0]
    s_vals = [0.0]
    diss = 0.0
    drives = [-dE_pointwise(l0, t)]
    l = l0
    s = 0.0

    while s < s_max:
        # drive = fast_flow_rhs(l, t)
        drive = max(0.0, -dE_pointwise(l, t))
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
    L, T, G_c_func=_G_c, L_max=5.0, dx=1e-3, tol=1e-3, num_subintervals=5
):
    """
    Solves the equation:
        -N T^2 / (2*L) = -N T^2 / (2*L_+) + ∫_{L}^{L_+} G_c(x) dx
    for L_+ > L given an instability at (L, T).

    Parameters:
        L : float       -- current (unstable) crack length
        T : float       -- fixed loading time
        G_c_func : func -- toughness function G_c(x)
        L_max : float   -- upper bound for crack search
        dx : float      -- step size for numerical integration
        tol : float     -- tolerance for root solver

    Returns:
        L_plus : float  -- solution of the energy conservation equation
    """
    # LHS is fixed
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

    # Try solving
    # try:
    #     print(f"Finding root for L_plus in ({L + tol}, {L_max})")
    #     sol = root_scalar(residual, bracket=(L + tol, L_max), method="brentq", xtol=tol)
    #     if sol.converged:
    #         return sol.root
    #     else:
    #         return None
    # except Exception as e:
    #     print(f"Error in root finding: {e}")
    #     return None

    # Divide into subintervals
    brackets = np.linspace(L + tol, L_max, num_subintervals + 1)

    for i in range(num_subintervals):
        a, b = brackets[i], brackets[i + 1]
        try:
            fa, fb = residual(a), residual(b)
            # print(f"Bracket ({a:.4f}, {b:.4f}) with fa={fa:.4f}, fb={fb:.4f}")
            if fa * fb < 0:
                # print(f"Trying subinterval ({a:.4f}, {b:.4f})")
                sol = root_scalar(residual, bracket=(a, b), method="brentq", xtol=tol)
                if sol.converged:
                    return sol.root
        except Exception as e:
            print(f"  Error in subinterval ({a:.4f}, {b:.4f}): {e}")
            continue


def run_variational_flow(
    t_grid,
    l_0,
    get_right_Gc=get_right_Gc,
    energy_pointwise=energy_pointwise,
    dE_pointwise=dE_pointwise,
    d2E_pointwise=d2E_pointwise,
    jump=integrate_fast_flow,
):
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

        drive = dE_pointwise(l_current, t, tol=delta)
        curvature = d2E_pointwise(l_current, t, tol=1e-2)

        drive_path[i] = drive
        curvature_path[i] = curvature

        if drive < -1e-10:
            _s, _l, diss, drives = jump(l_path[i], t, s_max=5.0, dt=0.01)
            dissipation_current += diss
            l_path[i] = _l[-1]

        dissipation_path[i] = dissipation_current

    E_total = [energy_pointwise(l, t) for l, t in zip(l_path, t_grid)]

    return {
        "l_path": l_path,
        "t_grid": t_grid,
        "E_path": E_total,
        "dissipation_path": dissipation_path,
        "drive_path": drive_path,
        "curvature_path": curvature_path,
    }


def run_energy_conserving(
    t_grid,
    l_0,
    get_right_Gc=get_right_Gc,
    energy_pointwise=energy_pointwise,
    dE_pointwise=dE_pointwise,
    d2E_pointwise=d2E_pointwise,
    jump=solve_energy_conservation_jump,
):
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

        drive = dE_pointwise(l_current, t, tol=delta)
        curvature = d2E_pointwise(l_current, t, tol=1e-2)

        drive_path[i] = drive
        curvature_path[i] = curvature

        if drive < -1e-10:
            L_val = l_current
            T_val = t
            L_plus = solve_energy_conservation_jump(
                L=L_val,
                T=T_val,
                G_c_func=_G_c,
                L_max=L_val + 3,
                tol=1e-2,
                num_subintervals=5,
            )
            l_path[i] = L_plus

    E_total = [energy_pointwise(l, t) for l, t in zip(l_path, t_grid)]

    return {
        "l_path": l_path,
        "t_grid": t_grid,
        "E_path": E_total,
        "dissipation_path": np.zeros_like(t_grid),  # Optional: fill if needed
        "drive_path": drive_path,
        "curvature_path": curvature_path,
    }


def run_global_minimizer(
    t_grid,
    l_0,
    energy_pointwise,
    get_right_Gc=get_right_Gc,
    l_max=5.0,
    dl=1e-3,
    delta=1e-2,
    **kwargs,
):
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
        energies = [energy_pointwise(l, t) for l in l_candidates]
        l_current = l_candidates[np.argmin(energies)]

        l_path[i] = l_current
        l_prev = l_current

        drive = dE_pointwise(l_current, t, tol=delta)
        curvature = d2E_pointwise(l_current, t, tol=1e-2)

        drive_path[i] = drive
        curvature_path[i] = curvature

    E_total = [energy_pointwise(l, t) for l, t in zip(l_path, t_grid)]

    return {
        "l_path": l_path,
        "t_grid": t_grid,
        "E_path": E_total,
        "dissipation_path": np.zeros_like(t_grid),  # Optional: fill if needed
        "drive_path": drive_path,
        "curvature_path": curvature_path,
    }
