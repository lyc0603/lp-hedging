"""Script to run a simulation."""

import pandas as pd
import matplotlib.pyplot as plt

from dex_sim.cpmm import PoolV2
from dex_sim.agents import NoiseTraderV2, StaticLPV2, HedgedLPV2, ArbitragerV2
from dex_sim.price import PriceSimulator
from dex_sim.tracker import PoolV2Tracker, LPV2AgentTracker
from dex_sim.constants import FIGURE_PATH

# SEED = 42
SEED = 16
FONT_SIZE = 14

for fee in [0.0, 0.003]:
    for k in [0.0, 0.002]:
        print(f"Running simulation with fee={fee} and k={k}")
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
            name="Hedged LP", amount0=1, amount1=1000, k=k, r=0.0001
        )
        print(f"Initial hedged liquidity provider: {hedged_liquidity_provider}")

        arbitrager = ArbitragerV2(name="Arbitrager", tolerance=0.0001)
        print(f"Initial arbitrager: {arbitrager}")

        noise_trader = NoiseTraderV2(
            name="Noise Trader",
            poisson_lambda=1,
            gaussian_mean=0,
            gaussian_std=0.5,
            seed=SEED,
        )
        print(f"Initial noise trader: {noise_trader}")

        # Initialize the price simulator
        simulator = PriceSimulator(
            initial_price=1000, mu=0, sigma=0.05, dt=1, seed=SEED
        )  # 1% hourly volatility ~ 100% annual volatility
        print(f"Initialize price simulator: {simulator}")

        # Simulate price changes over a month (24 hours * 30 days)
        cex_prices = simulator.simulate(steps=1000)
        print(
            "Simulated price changes over time."
            f"First price = {cex_prices[0]}, Last price = {cex_prices[-1]}"
        )

        # Simulate Arbitrager and Liquidity Provider Actions
        pool_tracker = PoolV2Tracker(pool)
        lp1_tracker = LPV2AgentTracker(unhedged_liquidity_provider)
        lp2_tracker = LPV2AgentTracker(hedged_liquidity_provider)

        BnH = []

        for timestep, external_price in enumerate(cex_prices):
            unhedged_liquidity_provider.act(pool, external_price)
            hedged_liquidity_provider.act(pool, dt=1, external_price=external_price)
            noise_trader.act(pool)
            arbitrager.act(pool, external_price)
            BnH.append(1 * external_price + 1000)
            # trackers
            pool_tracker.update(pool, timestep)
            lp1_tracker.update(unhedged_liquidity_provider, pool, timestep)
            lp2_tracker.update(hedged_liquidity_provider, pool, timestep)

        plt.figure(figsize=(3.5, 3))
        plt.plot(
            pd.DataFrame(hedged_liquidity_provider.history)[["Pi"]],
            label="Hedged LP",
            color="black",
            linewidth=1.5,
        )
        plt.plot(
            pd.DataFrame(unhedged_liquidity_provider.history)[["V"]],
            label="Unhedged LP",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        plt.plot(BnH, label="Buy and Hold", color="gray", linewidth=1.5)
        plt.legend(
            frameon=False,
            bbox_to_anchor=(1.05, 1.05),
            loc="upper right",
            fontsize=FONT_SIZE,
            prop={"weight": "bold"},
        )
        plt.xlabel(
            "Step, $\mathbf{\delta t}$", fontsize=FONT_SIZE - 3, fontweight="bold"
        )
        plt.ylabel("Position (USD)", fontsize=FONT_SIZE - 3, fontweight="bold")
        plt.xticks(fontsize=FONT_SIZE - 4, fontweight="bold")
        plt.yticks(fontsize=FONT_SIZE - 4, fontweight="bold")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            FIGURE_PATH / f"simulation_fee_{fee:.3f}_k_{k:.3f}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
