"""Script to run a simulation."""

import pandas as pd
import matplotlib.pyplot as plt

from dex_sim.cpmm import PoolV2
from dex_sim.agents import NoiseTraderV2, StaticLPV2, HedgedLPV2, ArbitragerV2
from dex_sim.price import PriceSimulator
from dex_sim.tracker import PoolV2Tracker, LPV2AgentTracker
from dex_sim.constants import FIGURE_PATH

SEED = 6
FONT_SIZE = 14

T = 30 * 24  # total time in hours
dt_list = [24, 1.0, 1.0 / 60]
min_dt = min(dt_list)
sim = PriceSimulator(initial_price=50, mu=0.0, sigma=0.05, T=T, seed=SEED)
cex_prices = sim.simulate(dt=min_dt)


def simulate(fee: float, k: float, dt: float) -> None:
    """Run the simulation and plot results."""
    # cex_prices = paths[dt]  # identical Brownian path across all dt
    steps = len(cex_prices) - 1

    print(f"Running simulation with fee={fee} and k={k} and dt={dt}")
    # Initialize the pool and agents
    pool = PoolV2(fee=fee, token0="ETH", token1="USDC")
    print(f"Initial empty pool: {pool}")

    unhedged_liquidity_provider = StaticLPV2(
        name="Unhedged LP",
        amount0=1,
        amount1=1000,
    )
    print(f"Initial unhedged liquidity provider: {unhedged_liquidity_provider}")

    hedged_liquidity_provider = HedgedLPV2(
        name="Hedged LP", amount0=1, amount1=1000, k=k, r=0.0
    )
    print(f"Initial hedged liquidity provider: {hedged_liquidity_provider}")

    arbitrager = ArbitragerV2(name="Arbitrager", tolerance=0.001)
    print(f"Initial arbitrager: {arbitrager}")

    noise_trader = NoiseTraderV2(
        name="Noise Trader",
        poisson_lambda=10,
        gaussian_mean=0,
        gaussian_std=0.1,
        seed=SEED,
    )
    print(f"Initial noise trader: {noise_trader}")

    # Simulate Arbitrager and Liquidity Provider Actions
    pool_tracker = PoolV2Tracker(pool)
    lp1_tracker = LPV2AgentTracker(unhedged_liquidity_provider)
    lp2_tracker = LPV2AgentTracker(hedged_liquidity_provider)

    BnH = {
        "timestep": [],
        "position": [],
    }
    m = int(round(dt / min_dt))

    # for timestep, external_price in enumerate(cex_prices):
    for timestep in range(steps + 1):
        pct_step = timestep / steps * T
        external_price = cex_prices[timestep]
        unhedged_liquidity_provider.act(pool, external_price)
        # Hedged LP action at coarser dt
        if timestep % m == 0:
            hedged_liquidity_provider.act(pool, dt=dt, external_price=external_price)
        noise_trader.act(pool, dt=min_dt)
        arbitrager.act(pool, external_price)
        BnH["timestep"].append(pct_step)
        BnH["position"].append(1 * external_price + 1000)
        # trackers
        pool_tracker.update(pool, pct_step)
        lp1_tracker.update(unhedged_liquidity_provider, pool, pct_step)
        lp2_tracker.update(hedged_liquidity_provider, pool, pct_step)

    BnH = pd.DataFrame(BnH).set_index("timestep")["position"]
    plt.figure(figsize=(3.5, 3))
    plt.plot(
        lp2_tracker.to_dataframe()[0].set_index("timestep")[["Pi"]],
        label="Hedged LP",
        color="blue",
        linewidth=1.5,
        linestyle="-",
        alpha=0.8,
    )
    plt.plot(
        lp1_tracker.to_dataframe()[0].set_index("timestep")[["V"]],
        label="Unhedged LP",
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
    )
    plt.plot(
        BnH,
        label="Buy and Hold",
        color="orange",
        linewidth=1.5,
        linestyle=":",
        alpha=0.8,
    )
    plt.legend(
        frameon=False,
        bbox_to_anchor=(1.05, 1.05),
        loc="upper right",
        fontsize=FONT_SIZE,
        prop={"weight": "bold"},
    )
    plt.xlabel("Hour, $\mathbf{t}$", fontsize=FONT_SIZE - 3, fontweight="bold")
    plt.ylabel("Position (USD)", fontsize=FONT_SIZE - 3, fontweight="bold")
    plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
    plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"simulation_fee_{fee:.3f}_k_{k:.3f}_dt_{dt:.3f}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


for fee, k in [
    (0, 0),
    (0.003, 0),
    (0.003, 0.001),
]:
    for dt in dt_list:
        simulate(fee=fee, k=k, dt=dt)
