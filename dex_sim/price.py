"""
GBM (Geometric Brownian Motion) price simulator module.
- Correct drift: (mu - 0.5*sigma^2)*dt
- Uses a private RNG (Generator) for reproducibility
- Provides 'simulate_coupled' to compare different dt fairly by sharing one Brownian path
"""

import numpy as np


class PriceSimulator:
    """Simulate price movements using Geometric Brownian Motion (GBM)."""

    def __init__(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        T: float,
        seed: int,
    ):
        """
        Parameters
        ----------
        initial_price : float
            S_0
        mu : float
            Drift (annualized if dt is in years)
        sigma : float
            Volatility (annualized if dt is in years)
        dt : float
            Time step for this simulator instance
        seed : int | None
            Seed for reproducibility (uses numpy Generator, not global RNG)
        """
        self.initial_price = float(initial_price)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.T = float(T)
        self.rng = np.random.default_rng(seed)

    def simulate(self, dt: float) -> np.ndarray:
        """Simulate price movements over a specified number of steps."""
        steps = int(self.T / dt)
        prices = np.zeros(steps + 1)
        price = self.initial_price
        prices[0] = price
        drift = (self.mu * -0.5 * self.sigma**2) * dt
        volatility = self.sigma * np.sqrt(dt)
        for i in range(steps):
            z = self.rng.standard_normal()
            shock = drift + z * volatility
            price *= np.exp(shock)
            prices[i + 1] = price

        return prices

    # def simulate_coupled(
    #     self,
    #     dt_list: List[float],
    #     ensure_last: bool = True,
    # ) -> Dict[float, np.ndarray]:
    #     """
    #     Generate *coupled* price paths for multiple dt on the SAME Brownian path.

    #     Strategy:
    #     - Simulate once on the finest grid (min_dt).
    #     - Downsample the price path to each coarser dt.
    #     - This keeps the underlying Brownian motion identical across dt,
    #       isolating the effect of hedge frequency.

    #     Returns
    #     -------
    #     dict: {dt: np.ndarray of prices of length steps_dt+1}
    #     """
    #     dt_list = sorted(dt_list)
    #     min_dt = min(dt_list)
    #     # make T exactly divisible by min_dt (tolerate tiny rounding)
    #     steps_min = int(round(self.T / min_dt))

    #     S_base = self.simulate(steps=steps_min, dt=min_dt)

    #     out: Dict[float, np.ndarray] = {}
    #     for dt in dt_list:
    #         m = int(round(dt / min_dt))
    #         assert m >= 1, "Each dt must be an integer multiple of the finest min_dt."
    #         path = S_base[::m]

    #         # optionally ensure the very last time T_eff is included
    #         if ensure_last and path[-1] != S_base[-1]:
    #             path = np.append(path, S_base[-1])

    #         out[dt] = path

    #     return out


if __name__ == "__main__":
    sim = PriceSimulator(initial_price=100, mu=0.05, sigma=0.2, T=1, seed=42)
    prices = sim.simulate(dt=1 / 252)
    import matplotlib.pyplot as plt

    plt.plot(prices)
    plt.title("Simulated GBM Price Path")
    plt.xlabel("Steps (Daily)")
    plt.ylabel("Price")
    plt.show()
