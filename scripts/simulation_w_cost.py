"""
Monte Carlo Simulation of Discrete-Time Hedging with Transaction Costs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Params:
    """Parameters for the simulation."""
    # Market
    mu: float = 0.01
    sigma: float = 0.05
    r: float = 0.0005

    # LP fee accrual (liquidity growth)
    gamma: float = 0.003

    # Proportional transaction cost on risky-asset rebalancing
    k: float = 0.001

    # Simulation controls
    S0: float = 1.0
    L0: float = 1000.0
    T: float = 30.0
    dt: float = 1.0 / 24.0
    n_paths: int = 5000
    seed: int = 42


def simulate_with_costs(p: Params, *, include_fee_term: bool = False) -> Dict[str, np.ndarray]:
    """
    Returns:
      realized_pnl[i] = simulated Π_T - Π_0 with transaction costs
      theory_pnl[i]   = theoretical approximation sum of ΔΠ with transaction costs (Prop 5.1),
                        optionally plus fee term if include_fee_term=True
      tc_paid[i]      = total transaction costs paid along the path (in currency units)
      mae_step[i]     = mean abs step error |ΔΠ_real - ΔΠ_theory| along the path
    """
    rng = np.random.default_rng(p.seed)
    n_steps = int(round(p.T / p.dt))
    dt = p.dt

    realized_pnl = np.empty(p.n_paths, dtype=float)
    theory_pnl = np.empty(p.n_paths, dtype=float)
    tc_paid = np.empty(p.n_paths, dtype=float)
    mae_step = np.empty(p.n_paths, dtype=float)

    for j in range(p.n_paths):
        # --- init state
        S = p.S0
        L = p.L0

        V = 2.0 * L * math.sqrt(S)
        Delta = L / math.sqrt(S)

        # Any initial cash is allowed; it cancels out in Π_T - Π_0 if you measure increments consistently.
        B = 0.0

        H = Delta * S + B
        Pi = V - H
        Pi0 = Pi

        theory_sum = 0.0
        abs_err_sum = 0.0
        tc_sum = 0.0

        for _ in range(n_steps):
            Z = rng.standard_normal()

            # --- price step (Euler discrete GBM)
            dS = p.mu * S * dt + p.sigma * S * math.sqrt(dt) * Z
            S_next = max(S + dS, 1e-12)

            # --- liquidity growth
            L_next = L * math.exp(p.gamma * dt)

            # --- LP value and delta at t+dt
            V_next = 2.0 * L_next * math.sqrt(S_next)
            Delta_next = L_next / math.sqrt(S_next)

            # --- transaction cost paid at this rebalance (in currency units)
            dDelta = Delta_next - Delta
            tc_step = p.k * abs(dDelta) * S_next
            tc_sum += tc_step

            # --- cash update with proportional transaction costs
            # B_{t+dt} = (1 + r dt) B_t - (Δ_{t+dt}-Δ_t) S_{t+dt} - k|Δ_{t+dt}-Δ_t| S_{t+dt}
            B_next = (1.0 + p.r * dt) * B - dDelta * S_next - tc_step

            # --- hedge value and hedged position
            H_next = Delta_next * S_next + B_next
            Pi_next = V_next - H_next

            # --- realized increment
            dPi_real = Pi_next - Pi

            # --- theoretical increment (Prop 5.1)
            dPi_th = (
                p.r * Pi * dt
                +  V * (p.gamma - 0.5 * p.r) * dt
                - (p.sigma ** 2 / 8.0) * V * (Z ** 2) * dt
                - (p.k * p.sigma / 4.0) * V * abs(Z) * math.sqrt(dt)
            )

            # Optional: include fee-drift term (from the no-cost expansion)
            if include_fee_term:
                dPi_th += V * (p.gamma - 0.5 * p.r) * dt

            theory_sum += dPi_th
            abs_err_sum += abs(dPi_real - dPi_th)

            # roll
            S, L = S_next, L_next
            V, Delta = V_next, Delta_next
            B, H, Pi = B_next, H_next, Pi_next

        realized_pnl[j] = Pi - Pi0
        theory_pnl[j] = theory_sum
        tc_paid[j] = tc_sum
        mae_step[j] = abs_err_sum / n_steps

    return {
        "realized_pnl": realized_pnl,
        "theory_pnl": theory_pnl,
        "tc_paid": tc_paid,
        "mae_step": mae_step,
    }


def summarize(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.var(x, ddof=1))


if __name__ == "__main__":
    p = Params(
        mu=0.00,
        sigma=0.05,
        r=0.0005,
        gamma=0.003,
        k=0.001,
        S0=1.0,
        L0=1000.0,
        T=30.0,
        dt=1.0 / 24.0,   # 1 hour
        n_paths=3000,
        seed=123,
    )

    include_fee_term = False  # set True if you want to add +V(γ - r/2)dt into the theoretical increment

    out = simulate_with_costs(p, include_fee_term=include_fee_term)

    m_real, v_real = summarize(out["realized_pnl"])
    m_th, v_th = summarize(out["theory_pnl"])
    m_tc, v_tc = summarize(out["tc_paid"])
    m_mae, v_mae = summarize(out["mae_step"])

    print("=== Discrete-Time Hedging WITH Transaction Costs ===")
    print(f"Paths: {p.n_paths:,} | Steps: {int(round(p.T/p.dt)):,} | dt={p.dt:g} | T={p.T:g}")
    print(f"Params: mu={p.mu}, sigma={p.sigma}, r={p.r}, gamma={p.gamma}, k={p.k}, S0={p.S0}, L0={p.L0}")
    print(f"Theory includes fee term (+V(γ-r/2)dt): {include_fee_term}")
    print()
    print("Terminal hedged PnL (Π_T - Π_0):")
    print(f"  Simulated (with TC):     mean={m_real:.6f}, var={v_real:.6f}")
    print(f"  Theoretical approx:      mean={m_th:.6f}, var={v_th:.6f}")
    print()
    print("Transaction costs paid (sum k|ΔΔ|S):")
    print(f"  mean={m_tc:.6f}, var={v_tc:.6f}")
    print()
    print("Per-step approximation error |ΔΠ_real - ΔΠ_theory|:")
    print(f"  mean(MAE)={m_mae:.6e}, var(MAE)={v_mae:.6e}")

    # Histogram comparison (PnL only, with costs)
    plt.figure()
    plt.hist(out["realized_pnl"], bins=60, alpha=0.6, label="Simulated PnL (with TC)")
    plt.hist(out["theory_pnl"], bins=60, alpha=0.6, label="Theory approx (with TC)")
    plt.xlabel(r"$\Pi_T - \Pi_0$")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Terminal Hedged PnL WITH Transaction Costs")

    # Scatter: realized vs theory (with costs)
    plt.figure()
    plt.scatter(out["theory_pnl"], out["realized_pnl"], s=6, alpha=0.4)
    mn = min(out["theory_pnl"].min(), out["realized_pnl"].min())
    mx = max(out["theory_pnl"].max(), out["realized_pnl"].max())
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("Theory approx PnL (with TC)")
    plt.ylabel("Simulated PnL (with TC)")
    plt.title("Simulated vs Theoretical (WITH Transaction Costs)")

    # Optional: transaction cost distribution
    plt.figure()
    plt.hist(out["tc_paid"], bins=60, alpha=0.7)
    plt.xlabel("Total transaction costs paid")
    plt.ylabel("Count")
    plt.title("Distribution of Total Transaction Costs")

    plt.show()
