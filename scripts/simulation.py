"""Dynamics of the hedged position under discrete rebalancing and transaction costs."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dex_sim.constants import FIGURE_PATH


FONT_SIZE = 14
COLORS = ["gold", "darkorange", "red"]
T = 30 * 24  # total time in hours


def exp(
    V0: float = 2000.0,
    r: float = 0.0001,
    sigma: float = 0.05,
    T: float = 30 * 24,
    k: float = 0.003,
    dt: float = 1.0,
) -> float:
    """Expected E[Π_T - Π_0] under discrete rebalancing and transaction costs."""

    return V0 * ((r / 2.0) - (sigma**2) / 8.0) * T - (k * sigma * V0 / 4.0) * np.sqrt(
        2.0 / np.pi
    ) * (T / np.sqrt(dt))


def var(
    V0: float = 2000.0,
    sigma: float = 0.05,
    T: float = 30 * 24,
    k: float = 0.003,
    dt: float = 1.0,
):
    """Variance Var[Π_T - Π_0] under discrete rebalancing and transaction costs."""

    return (sigma**4 * V0**2 / 32.0) * T * dt + (
        k**2 * sigma**2 * V0**2 / 16.0
    ) * (1.0 - 2.0 / np.pi) * T, (k**2 * sigma**2 * V0**2 / 16.0) * (
        1.0 - 2.0 / np.pi
    ) * T


# Plot sensitivity of E[Π_T - Π_0] to k and sigma
plt.figure(figsize=(3.5, 3))
ks = [0.001, 0.003, 0.005]

for k, color in zip(ks, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, k=k),
                "var": var(dt=dt, k=k)[0],
                "floor": var(dt=dt, k=k)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["exp"],
        label=k,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{k}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.ylabel(
    r"$\mathbf{\mathbb{E}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "exp_k.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# sigma sensitivity
plt.figure(figsize=(3.5, 3))
sigmas = [0.03, 0.05, 0.08]

for sigma, color in zip(sigmas, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, sigma=sigma),
                "var": var(dt=dt, sigma=sigma)[0],
                "floor": var(dt=dt, sigma=sigma)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["exp"],
        label=sigma,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{\sigma}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.ylabel(
    r"$\mathbf{\mathbb{E}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "exp_sigma.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# r sensitivity
plt.figure(figsize=(3.5, 3))
rs = [0.0001, 0.0005, 0.001]

for r, color in zip(rs, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, r=r),
                "var": var(dt=dt)[0],
                "floor": var(dt=dt)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["exp"],
        label=r,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{r}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.ylabel(
    r"$\mathbf{\mathbb{E}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "exp_r.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Plot sensitivity of Var[Π_T - Π_0] to N
plt.figure(figsize=(3.5, 3))
ks = [0.05, 0.025, 0.001]

for k, color in zip(ks, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, k=k),
                "var": var(dt=dt, k=k)[0],
                "floor": var(dt=dt, k=k)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["var"],
        label=k,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )
    plt.plot(
        df.index,
        df["floor"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{k}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
    x=0.3,
)
plt.ylabel(
    r"$\mathbf{\text{Var}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "var_k.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# V_0 sensitivity
plt.figure(figsize=(3.5, 3))
V0s = [2000.0, 1000.0, 500.0]

for V0, color in zip(V0s, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, V0=V0),
                "var": var(dt=dt, V0=V0)[0],
                "floor": var(dt=dt, V0=V0)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["var"],
        label=V0,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )
    plt.plot(
        df.index,
        df["floor"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{V}_{\boldsymbol{0}}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
    x=0.3,
)
plt.ylabel(
    r"$\mathbf{\text{Var}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "var_V0.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Sigma sensitivity
plt.figure(figsize=(3.5, 3))
sigmas = [0.03, 0.05, 0.08]

for sigma, color in zip(sigmas, COLORS):
    rows = []
    for N in np.linspace(1, 30 * 24, 100)[1:]:
        dt = T / N
        rows.append(
            {
                "N": N,
                "dt": dt,
                "exp": exp(dt=dt, sigma=sigma),
                "var": var(dt=dt, sigma=sigma)[0],
                "floor": var(dt=dt, sigma=sigma)[1],
            }
        )
    df = pd.DataFrame(rows).set_index("N")
    plt.plot(
        df.index,
        df["var"],
        label=sigma,
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
        color=color,
    )
    plt.plot(
        df.index,
        df["floor"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        color=color,
    )

plt.legend(
    title=r"$\boldsymbol{\sigma}$",
    frameon=False,
    loc="center right",
    fontsize=FONT_SIZE,
    prop={"weight": "bold"},
)
plt.xlabel(
    r"Number of Time Intervals, $\boldsymbol{N = \frac{T}{\delta t}}$",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
    x=0.3,
)
plt.ylabel(
    r"$\mathbf{\text{Var}[\Pi_{t + 1} - \Pi_{t}]}$, USD",
    fontsize=FONT_SIZE - 3,
    fontweight="bold",
)
plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "var_sigma.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
