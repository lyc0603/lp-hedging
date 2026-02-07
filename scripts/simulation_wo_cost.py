"""
Monte Carlo Simulation of Discrete-Time Hedging without Transaction Costs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Params:
    """
    Parameters for the simulation.
    """
    # GBM drift and volatility, and risk-free rate
    mu: float = 0.01         
    sigma: float = 0.05       
    r: float = 0.0005         

    # AMM fee accrual
    gamma: float = 0.003

    # Simulation controls
    S0: float = 1.0
    L0: float = 1000.0
    T: float = 30.0           
    dt: float = 1.0 / 24.0  
    n_paths: int = 5000
    seed: int = 42


def simulate_paths(p: Params) -> Dict[str, np.ndarray]:
    """
    Simulate many paths and return terminal PnLs:
      realized_pnl[i]    = (Π_T - Π_0) for path i using exact bookkeeping
      theory_pnl[i]      = sum of theoretical ΔΠ approximation along path i
      mae_step[i]        = mean abs step error |ΔΠ_real - ΔΠ_theory| along path i
    """
    rng = np.random.default_rng(p.seed)

    n_steps = int(round(p.T / p.dt))
    dt = p.dt

    realized_pnl = np.empty(p.n_paths, dtype=float)
    theory_pnl = np.empty(p.n_paths, dtype=float)
    mae_step = np.empty(p.n_paths, dtype=float)

    for j in range(p.n_paths):
        # Asset price and liquidity state
        S = p.S0
        L = p.L0

        V = 2.0 * L * math.sqrt(S)
        Delta = L / math.sqrt(S)

        # choose an initial self-financing setup:
        # simplest: start with hedge portfolio H0 = Δ0 S0 (i.e., B0 = 0)
        B = 10.0
        H = Delta * S + B
        Pi = V - H
        Pi0 = Pi

        theory_sum = 0.0
        abs_err_sum = 0.0

        for _ in range(n_steps):
            # draw Z for this step
            Z = rng.standard_normal()

            # price step (discrete GBM)
            dS = p.mu * S * dt + p.sigma * S * math.sqrt(dt) * Z
            S_next = max(S + dS, 1e-12)  # keep positive (rare with small dt)

            # liquidity growth
            L_next = L * math.exp(p.gamma * dt)

            # update LP value at t+dt
            V_next = 2.0 * L_next * math.sqrt(S_next)

            # compute new delta at t+dt (rebalance at t+dt)
            Delta_next = L_next / math.sqrt(S_next)

            # self-financing cash update (no transaction costs)
            B_next = (1.0 + p.r * dt) * B - (Delta_next - Delta) * S_next

            # hedge value and hedged position
            H_next = Delta_next * S_next + B_next
            Pi_next = V_next - H_next

            # realized increment
            dPi_real = Pi_next - Pi

            # theoretical increment approximation
            dPi_th = (
                p.r * Pi * dt
                + V * (p.gamma - 0.5 * p.r) * dt
                - (p.sigma ** 2 / 8.0) * V * (Z ** 2) * dt
            )

            theory_sum += dPi_th
            abs_err_sum += abs(dPi_real - dPi_th)

            # roll state
            S, L = S_next, L_next
            V, Delta = V_next, Delta_next
            B, H, Pi = B_next, H_next, Pi_next

        realized_pnl[j] = Pi - Pi0
        theory_pnl[j] = theory_sum
        mae_step[j] = abs_err_sum / n_steps

    return {"realized_pnl": realized_pnl, "theory_pnl": theory_pnl, "mae_step": mae_step}


def summarize(x: np.ndarray) -> Tuple[float, float]:
    """Summarize an array with mean and sample variance."""
    return float(np.mean(x)), float(np.var(x, ddof=1))


if __name__ == "__main__":
    p = Params(
        mu=0.00,
        sigma=0.05,
        r=0.0005,
        gamma=0.003,
        S0=1.0,
        L0=1000.0,
        T=30.0,
        dt=1.0 / 24.0, 
        n_paths=3000,
        seed=123,
    )

    out = simulate_paths(p)

    m_real, v_real = summarize(out["realized_pnl"])
    m_th, v_th = summarize(out["theory_pnl"])
    m_mae, v_mae = summarize(out["mae_step"])

    print("=== Discrete-Time Hedging (No Transaction Costs) ===")
    print(f"Paths: {p.n_paths:,} | Steps: {int(round(p.T/p.dt)):,} | dt={p.dt:g} | T={p.T:g}")
    print(f"Params: mu={p.mu}, sigma={p.sigma}, r={p.r}, gamma={p.gamma}, S0={p.S0}, L0={p.L0}")
    print()
    print("Terminal hedged PnL (Π_T - Π_0):")
    print(f"  Realized bookkeeping:   mean={m_real:.6f}, var={v_real:.6f}")
    print(f"  Theoretical approx sum: mean={m_th:.6f}, var={v_th:.6f}")
    print()
    print("Per-step approximation error |ΔΠ_real - ΔΠ_theory|:")
    print(f"  mean(MAE)={m_mae:.6e}, var(MAE)={v_mae:.6e}")

    # Histogram comparison
    plt.figure()
    plt.hist(out["realized_pnl"], bins=60, alpha=0.6, label="Realized (bookkeeping)")
    plt.hist(out["theory_pnl"], bins=60, alpha=0.6, label="Theory approx (sum of ΔΠ)")
    plt.xlabel(r"$\Pi_T - \Pi_0$")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Distribution of terminal hedged PnL")

    # Scatter: realized vs theory
    plt.figure()
    plt.scatter(out["theory_pnl"], out["realized_pnl"], s=6, alpha=0.4)
    mn = min(out["theory_pnl"].min(), out["realized_pnl"].min())
    mx = max(out["theory_pnl"].max(), out["realized_pnl"].max())
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("Theory approx PnL")
    plt.ylabel("Realized PnL")
    plt.title("Realized vs Theoretical Approximation")

    plt.show()

