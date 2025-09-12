"""Agents Module"""

import logging
import numpy as np
from dex_sim.cpmm import PoolV2

logger = logging.getLogger(__name__)


def lp_share_fraction(pool: PoolV2, lp_liquidity: float) -> float:
    """Calculate the fraction of the pool owned by the LP."""
    if pool.total_liquidity <= 0:
        return 0.0
    return lp_liquidity / pool.total_liquidity


def lp_mark_to_market_value(pool: PoolV2, s: float, S: float) -> float:
    """Calculate the mark-to-market value of the LP's position."""

    return s * (pool.x * S + pool.y)


class Agent:
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name: str = name

    def act(self, pool: PoolV2) -> None:
        """Perform actions in the pool."""
        pass


class NoiseTraderV2(Agent):
    """
    Noise trader that generates trades based on a Poisson process and a Gaussian distribution.
    """

    def __init__(
        self,
        name: str,
        poisson_lambda: float,
        gaussian_mean: float,
        gaussian_std: float,
        seed: int = None,
    ):
        super().__init__(name)
        self.rng = np.random.default_rng(seed)
        self.poission_lambda = poisson_lambda
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std

    def act(self, pool: PoolV2, dt: float = None) -> None:
        """Perform actions in the pool."""
        n_traders = self.rng.poisson(self.poission_lambda * dt)
        logger.debug("%s generating %d trades", self.name, n_traders)
        for _ in range(n_traders):
            amt = self.rng.normal(
                self.gaussian_mean * dt, self.gaussian_std * np.sqrt(dt)
            )
            logger.debug("%s executing trade with amount: %f", self.name, amt)
            # Execute the trade in the pool
            if amt > 0:
                pool.swap(amount_in=amt, token0_for_token1=True)
            else:
                pool.swap(amount_in=-amt * pool.price, token0_for_token1=False)


class StaticLPV2(Agent):
    """
    Static liquidity provider that adds liquidity to the pool and earns fees.
    """

    def __init__(self, name: str, amount0: float, amount1: float) -> None:
        super().__init__(name)
        self.amount0 = amount0
        self.amount1 = amount1
        self.lp_position = None
        self.lp_liquidity = 0.0
        self.V = 0.0

        # tracking
        self.history = []

    def act(self, pool: PoolV2, external_price: float) -> None:
        """Perform actions in the pool."""
        if self.lp_position is None:
            logger.debug("%s adding liquidity", self.name)
            pos, used0, used1, minted = pool.add_liquidity(
                self.amount0, self.amount1, is_tracked=True
            )
            self.lp_position = pos
            self.lp_liquidity = minted
            self.amount0 -= used0
            self.amount1 -= used1
            logger.debug(
                "%s added liquidity: used0=%.6f, used1=%.6f, minted=%.6f",
                self.name,
                used0,
                used1,
                minted,
            )
        s = lp_share_fraction(pool, self.lp_liquidity)
        self.V = lp_mark_to_market_value(pool, s, external_price)
        self.history.append(
            {
                "amount0": self.amount0,
                "amount1": self.amount1,
                "lp_liquidity": self.lp_liquidity,
                "V": self.V,
            }
        )

    def close(self, pool: PoolV2, fraction: float = 1.0) -> tuple[float, float]:
        """Remove liquidity from the pool."""
        if self.lp_position is None:
            return 0.0, 0.0
        out0, out1 = pool.remove_liquidity(self.lp_position.id, fraction=fraction)
        self.amount0 += out0
        self.amount1 += out1
        self.lp_liquidity *= 1 - fraction
        if self.lp_liquidity <= 0:
            self.lp_position = None
        logger.debug(
            "%s removed liquidity: out0=%.6f, out1=%.6f",
            self.name,
            out0,
            out1,
        )
        return out0, out1


class HedgedLPV2(Agent):
    """
    Hedged static liquidity provider
    """

    def __init__(
        self,
        name: str,
        amount0: float,
        amount1: float,
        k: float = 0.0,
        r: float = 0.0,
    ) -> None:
        super().__init__(name)

        # Wallet cash balances
        self.amount0: float = amount0
        self.amount1: float = amount1
        self.cash_B: float = 0.0
        self.Pi: float = 0.0  # total hedged wealth

        # LP
        self.lp_position = None
        self.lp_liquidity: float = 0.0

        # Hedging portfolio
        self.delta = 0.0
        self.last_S = None

        # friction parameters
        self.k = k  # proportional transaction cost
        self.r = r  # risk free rate

        # tracking
        self.history = []

    def _ensure_lp(self, pool: PoolV2) -> None:
        """Ensure the agent has an active LP position."""
        if self.lp_position is not None:
            return
        pos, used0, used1, minted = pool.add_liquidity(
            self.amount0, self.amount1, is_tracked=True
        )
        self.lp_position = pos
        self.lp_liquidity = minted
        self.amount0 -= used0
        self.amount1 -= used1
        logger.debug(
            "%s added liquidity: used0=%.6f, used1=%.6f, minted=%.6f",
            self.name,
            used0,
            used1,
            minted,
        )

    def act(self, pool: PoolV2, dt: float, external_price: float | None = None) -> None:
        """Perform actions in the pool."""
        self._ensure_lp(pool)

        S = float(external_price) if external_price is not None else float(pool.price)
        if S <= 0:
            return

        # Cash carry before re-hedge
        if self.cash_B != 0.0 and dt > 0.0 and self.r != 0.0:
            # self.cash_B += self.r * self.cash_B * dt
            self.cash_B *= np.exp(self.r * dt)

        # LP value and target delta
        s = lp_share_fraction(pool, self.lp_liquidity)
        V = lp_mark_to_market_value(pool, s, S)
        # target = replicating delta (>=0), but we hold the short
        delta_rep = V / (2.0 * S)
        delta_star = -delta_rep  # we are short Δ

        dDelta = delta_star - self.delta

        # trade: if dDelta<0 we buy back (use cash); if dDelta>0 we sell more (receive cash)
        self.cash_B += -dDelta * S  # note the plus here (opposite sign to your code)

        tc = abs(dDelta) * S * self.k if self.k > 0.0 else 0.0
        if tc:
            self.cash_B -= tc

        self.delta = delta_star

        # value of the hedge (short underlying + cash)
        H = self.delta * S + self.cash_B  # self.delta is negative
        # total hedged wealth = LP + hedge
        wealth = V + H
        self.Pi = wealth  # report Π as total hedged wealth

        self.history.append(
            {
                "S": S,
                "V": V,
                "delta": self.delta,
                "B": self.cash_B,
                "H": H,
                "Pi": self.Pi,
                "tc": tc,
                "s": s,
                "wealth": V + H,
            }
        )

        # Prepare for next step
        self.last_S = S

        logger.debug(
            "%s hedge step: S=%s, V=%s, Δ*=%s, dΔ=%s, TC=%s, B=%s, H=%s, Π=%s",
            self.name,
            S,
            V,
            delta_star,
            dDelta,
            tc,
            self.cash_B,
            H,
            self.Pi,
        )

    def close(self, pool: PoolV2, fraction: float = 1.0) -> tuple[float, float]:
        """Remove liquidity from the pool and close hedging."""
        if self.lp_position is None:
            return 0.0, 0.0
        out0, out1 = pool.remove_liquidity(self.lp_position.id, fraction=fraction)
        self.amount0 += out0
        self.amount1 += out1
        self.lp_liquidity *= 1 - fraction
        if self.lp_liquidity <= 0:
            self.lp_position = None
        logger.debug(
            "%s removed liquidity: out0=%.6f, out1=%.6f",
            self.name,
            out0,
            out1,
        )
        return out0, out1


class ArbitragerV2(Agent):
    """
    Arbitrager that exploits price differences between the pool and an external price.
    """

    def __init__(self, name: str, tolerance: float):
        super().__init__(name)
        self.tolerance = tolerance

    def act(self, pool: PoolV2, external_price: float) -> None:
        """Perform actions in the pool."""
        S = pool.price
        if external_price <= 0:
            return

        if abs(S - external_price) / external_price < self.tolerance:
            return

        amt_in = self.compute_amount_to_trade(
            pool,
            external_price,
            token0_for_token1=(S > external_price),
        )
        if amt_in <= 0.0:
            return
        pool.swap(amt_in, token0_for_token1=(S > external_price))

    @staticmethod
    def compute_amount_to_trade(
        pool: PoolV2, target_price: float, token0_for_token1: bool
    ) -> float:
        """Compute the amount to trade to move the pool price to the target price."""
        S = pool.price
        f = getattr(pool, "swap_fee", 0.0)
        one_minus_f = 1.0 - f

        if token0_for_token1:
            if target_price >= S:
                return 0.0

            x = pool.x
            dx_after_fee = x * (np.sqrt(S / target_price) - 1.0)
            return dx_after_fee / max(one_minus_f, 1e-18)
        else:
            if target_price <= S:
                return 0.0

            y = pool.y
            dy_after_fee = y * (np.sqrt(target_price / S) - 1.0)
            return dy_after_fee / max(one_minus_f, 1e-18)
