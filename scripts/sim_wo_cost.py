"""
Monte Carlo Simulation of Discrete-Time Hedging without Transaction Costs
"""

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dex_sim.constants import FIGURE_PATH


BINS = 60

@dataclass(frozen=True)
class Params:
    """
    Parameters for the simulation.
    """
    # GBM drift, volatility, and growth rate
    mu: float = 0.01
    sigma: float = 0.05
    r: float = 0.0005
    gamma: float = 0.003


    # Initital asset price and liquidity state
    S0: float = 100.0
    L0: float = 10.0

    # Simulation controls
    T: float = 30.0
    dt: float = 1.0 / 24.0
    n_paths: int = 10000
    seed: int = 42

def simulate_paths(p: Params) -> dict[str, np.ndarray]:
    """
    Simulate many paths and return terminal PnLs:
    
    Arguments:
        p: Parameters for the simulation.

    Returns:
        A dictionary containing:
            'realized_pnl': (Pi_T - Pi_0) for each path using exact bookkeeping.
            'theory_pnl': sum of theoretical Delta Pi approximation along each path.
            'mae_step': mean absolute step error |Delta Pi_real - Delta Pi_theory| along each path.
    """

    rng = np.random.default_rng(p.seed)

    n_steps = int(round(p.T / p.dt))
    dt = p.dt

    realized_pnl = np.empty(p.n_paths, dtype=float)
    theory_pnl = np.empty(p.n_paths, dtype=float)
    mae_step = np.empty(p.n_paths, dtype=float)

    for i in range(p.n_paths):
        # Initial asset price and liquidity state
        S = p.S0
        L = p.L0

        # GBM drift, volatility, and growth rate
        mu = p.mu
        sigma = p.sigma
        r = p.r
        gamma = p.gamma

        # Initial portfolio value and Delta given by continuous-time hedging without transaction costs
        V = 2.0 * L * math.sqrt(S)
        Delta = L / math.sqrt(S)

        # Self-Financing Setup
        B = 100.0
        H = Delta * S + B
        Pi = V - H
        Pi0 = Pi

        theory_sum = 0.0
        abs_err_sum = 0.0

        for _ in range(n_steps):
            # White noise for this step
            Z = rng.standard_normal()

            # Update asset price S using GBM dynamics
            dS = mu * dt * S + sigma * math.sqrt(dt) * Z * S
            S_next = S + dS

            # Liquidity growth
            L_next = L * math.exp(gamma * dt)
            # Update LP value at t+dt
            V_next = 2.0 * L_next * math.sqrt(S_next)

            # Compute new Delta at t+dt (rebalance at t+dt)
            Delta_next = L_next / math.sqrt(S_next)

            # Self-financing cash update (no transaction costs)
            B_next = (1.0 + r * dt) * B - (Delta_next - Delta) * S_next

            # Hedge value and hedged position
            H_next = Delta_next * S_next + B_next
            Pi_next = V_next - H_next

            # Realized increment
            dPi_real = Pi_next - Pi

            # Theoretical increment approximation
            dPi_th = (
                r * Pi * dt
                + V * (gamma - 0.5 * r) * dt
                - (sigma ** 2 / 8.0) * V * (Z ** 2) * dt
            )

            theory_sum += dPi_th
            abs_err_sum += abs(dPi_real - dPi_th)

            # Roll state
            S, L = S_next, L_next
            V, Delta = V_next, Delta_next
            B, H, Pi = B_next, H_next, Pi_next

        realized_pnl[i] = Pi - Pi0
        theory_pnl[i] = theory_sum
        mae_step[i] = abs_err_sum 

    return {
        'realized_pnl': realized_pnl,
        'theory_pnl': theory_pnl,
        'mae_step': mae_step,
    }

if __name__ == "__main__":
    p = Params()
    results = simulate_paths(p)

    fig, ax = plt.subplots(figsize=(6, 5))

    SIM_COLOR = "#6BAED6"
    THEORY_COLOR = "#D62728"

    ax.hist(
        results["realized_pnl"],
        bins=BINS,
        color=SIM_COLOR,
        alpha=0.75,
        edgecolor=None,
        linewidth=0,
        label="Simulation"
    )

    ax.hist(
        results["theory_pnl"],
        bins=BINS,
        histtype="bar",
        facecolor="none",
        edgecolor=THEORY_COLOR,
        linewidth=1.2,
        label="Theory"
    )

    ax.set_xlabel(r"Hedged Position PnL")
    ax.grid(alpha=0.4)
    ax.legend(frameon=False, loc="upper left")

    # Inset scatter
    RIGHT_X = 0.62
    WIDTH   = 0.35

    # Scatter inset (top)
    axins = inset_axes(
        ax,
        width="100%", height="80%",
        bbox_to_anchor=(RIGHT_X, 0.58, WIDTH, 0.40), 
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    axins.scatter(
        results["theory_pnl"],
        results["realized_pnl"],
        s=5, alpha=0.35, color="#333333"
    )

    mn = min(results["theory_pnl"].min(), results["realized_pnl"].min())
    mx = max(results["theory_pnl"].max(), results["realized_pnl"].max())
    axins.plot([mn, mx], [mn, mx], color=THEORY_COLOR, linewidth=1)

    axins.set_xlabel("Theory", fontsize=9)
    axins.set_ylabel("Simulation", fontsize=9)
    axins.tick_params(labelsize=8)
    axins.grid(alpha=0.3)

    # MAE inset (bottom)
    axins_mae = inset_axes(
        ax,
        width="100%", height="80%",
        bbox_to_anchor=(RIGHT_X, 0.32, WIDTH, 0.22),  # directly under scatter
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    error = results["theory_pnl"] - results["realized_pnl"]

    axins_mae.hist(error, bins=30, color="#888888", alpha=0.7)
    mae = error.mean()
    axins_mae.axvline(mae, color=THEORY_COLOR, linestyle="--", linewidth=1)

    axins_mae.set_xlabel("Theory - Simulation", fontsize=9)
    axins_mae.tick_params(labelsize=8)
    axins_mae.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "sim_wo_cost.pdf", dpi=300)
    plt.show()


