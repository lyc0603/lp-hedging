"""Simulation for the dynamics of  ."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Theorem 2 (equations 35–36) ----------
def exp_pnl(T, N, V0, r, sigma, k):
    """
    E[Π_T - Π_0] ≈ V0 * ((r/2) - σ^2/8) * T  -  (k σ V0 / 4) * sqrt(2/π) * T / sqrt(δt)
                 with δt = T/N  =>  T / sqrt(δt) = sqrt(T) * sqrt(N)
    """
    term1 = V0 * ((r / 2.0) - (sigma**2) / 8.0) * T
    term2 = (k * sigma * V0 / 4.0) * np.sqrt(2.0 / np.pi) * np.sqrt(T) * np.sqrt(N)
    return term1 - term2


def var_pnl(T, N, V0, sigma, k):
    """
    Var[Π_T - Π_0] ≈ (σ^4 V0^2 / 32) * T δt  +  (k^2 σ^2 V0^2 / 16) * (1 - 2/π) * T
                   = (σ^4 V0^2 / 32) * T^2 / N  +  const * T
    """
    dt = T / N
    term_disc = (sigma**4 * V0**2 / 32.0) * T * dt  # goes to 0 as 1/N
    term_cost = (
        (k**2 * sigma**2 * V0**2 / 16.0) * (1.0 - 2.0 / np.pi) * T
    )  # lower bound
    return term_disc + term_cost, term_disc, term_cost


# ---------- Sensitivity runner ----------
def run_sensitivity_N(
    N_list=(5, 10, 20, 40, 80, 160, 320, 640),
    T=10.0,
    V0=1000.0,
    r=0.05,
    sigma=0.8,
    k=0.003,
):
    rows = []
    for N in N_list:
        dt = T / N
        E = exp_pnl(T, N, V0, r, sigma, k)
        V, V_disc, V_cost = var_pnl(T, N, V0, sigma, k)
        rows.append(
            dict(
                N=N,
                dt=dt,
                sqrtN=np.sqrt(N),
                E_PnL=E,
                Var=V,
                Var_disc_term=V_disc,
                Var_cost_floor=V_cost,
            )
        )
    return pd.DataFrame(rows)


def main():
    # --- Parameters you can tweak ---
    T = 10.0
    V0 = 1000.0
    r = 0.05
    sigma = 0.8
    k = 0.003
    N_list = (5, 10, 20, 40, 80, 160, 320, 640, 1280)

    df = run_sensitivity_N(N_list, T=T, V0=V0, r=r, sigma=sigma, k=k)
    print(df.to_string(index=False))

    # Plots (matplotlib only; simple defaults)
    # 1) Expectation vs N (should decrease roughly ∝ -sqrt(N))
    plt.figure(figsize=(7, 5))
    plt.plot(df["N"], df["E_PnL"], marker="o")
    plt.xlabel("N (number of rebalancing steps)")
    plt.ylabel("E[Π_T - Π_0]")
    plt.title("Expectation vs N (δt = T/N)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional diagnostic: Expectation vs sqrt(N) to reveal linear scaling
    plt.figure(figsize=(7, 5))
    plt.plot(df["sqrtN"], df["E_PnL"], marker="o")
    plt.xlabel("sqrt(N)")
    plt.ylabel("E[Π_T - Π_0]")
    plt.title("Expectation vs sqrt(N) (Theorem 2 scaling)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Variance vs N (should approach a positive floor as N ↑)
    plt.figure(figsize=(7, 5))
    plt.plot(df["N"], df["Var"], marker="o", label="Total variance")
    plt.plot(
        df["N"], df["Var_disc_term"], marker="x", label="Discretization term (∝ 1/N)"
    )
    plt.plot(
        df["N"], df["Var_cost_floor"], marker="s", label="Cost floor (constant in N)"
    )
    plt.xlabel("N (number of rebalancing steps)")
    plt.ylabel("Var[Π_T - Π_0]")
    plt.title("Variance vs N (δt = T/N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
