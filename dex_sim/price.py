"""
GBM (Geometric Brownian motion) price simulator module.
"""

import numpy as np
import matplotlib.pyplot as plt


class PriceSimulator:
    """Simulate price movements using Geometric Brownian Motion (GBM)."""

    def __init__(
        self, initial_price: float, mu: float, sigma: float, dt: float, seed: int = None
    ):
        """Initialize the price simulator."""
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, steps: int) -> np.ndarray:
        """Simulate price movements over a specified number of steps."""
        prices = np.zeros(steps + 1)
        price = self.initial_price
        prices[0] = price
        drift = (self.mu * -0.5 * self.sigma**2) * self.dt
        volatility = self.sigma * np.sqrt(self.dt)
        for i in range(steps):
            shock = np.random.normal(loc=drift, scale=volatility)
            price *= np.exp(shock)
            prices[i + 1] = price

        return prices


if __name__ == "__main__":
    simulator = PriceSimulator(
        initial_price=100, mu=0.05, sigma=0.2, dt=1 / 252, seed=42
    )
    prices = simulator.simulate(steps=252)

    plt.plot(prices)
    plt.title("Simulated Price Path using GBM")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.show()
