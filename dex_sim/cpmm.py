"""Constant Product Market Maker (CPMM) implementation."""

import math
import logging

logger = logging.getLogger(__name__)


class PositionV2:
    """A class to represent a liquidity provider's position in a CPMM pool."""

    _id_counter = 0

    def __init__(self, liquidity: float, is_tracked: bool = False) -> None:
        self.id = PositionV2._id_counter
        PositionV2._id_counter += 1
        self.liquidity = liquidity
        self.is_tracked = is_tracked

    def __repr__(self) -> str:
        return (
            f"PositionV2(id={self.id}, "
            f"liquidity={self.liquidity:.6f}, "
            f"is_tracked={self.is_tracked})"
        )


class PoolV2:
    """A class to represent a Constant Product Market Maker (CPMM) pool."""

    amount_precision = 6

    def __init__(
        self,
        fee: float,
        token0: str = "token0",
        token1: str = "token1",
        withdrawal_fee: float = 0.0,
    ) -> None:
        self.x = 0.0
        self.y = 0.0
        self.fee = fee
        self.withdrawal_fee = withdrawal_fee
        self.token0 = token0
        self.token1 = token1

        # LP token liquidity
        self.total_liquidity = math.sqrt(self.x * self.y)

        # State: positions
        self.positions = []

        # Accounting
        self.total_fees_token0 = 0.0
        self.total_fees_token1 = 0.0
        self.total_withdrawal_fees_token0 = 0.0
        self.total_withdrawal_fees_token1 = 0.0
        self.total_swaps_token0_in = 0.0
        self.total_swaps_token1_in = 0.0
        self.total_swaps_token0_out = 0.0
        self.total_swaps_token1_out = 0.0

        logger.debug(self.__repr__())

    def __repr__(self) -> str:
        return (
            f"PoolV2(x={self.x:.{self.amount_precision}f}, "
            f"y={self.y:.{self.amount_precision}f}, "
            f"price={self.price:.{self.amount_precision}f}, "
            f"fee={self.fee:.4f}, "
            f"token0='{self.token0}', "
            f"token1='{self.token1}', "
            f"total_liquidity={self.total_liquidity:.{self.amount_precision}f})"
        )

    @property
    def price(self) -> float:
        """Get the current price of token0 in terms of token1."""
        return self.y / self.x if self.x > 0 else float("NaN")

    def k(self) -> float:
        """Get the current invariant k = x * y."""
        return self.x * self.y

    def add_liquidity(
        self, amount0: float, amount1: float, is_tracked: bool = False
    ) -> tuple[PositionV2, float, float, float]:
        """Add liquidity to the pool and return a PositionV2 object."""
        if amount0 <= 0 or amount1 <= 0:
            raise ValueError("Amounts must be positive.")

        logger.debug(
            f"Adding liquidity: amount0=%s.{self.amount_precision}f, "
            f"amount1=%s.{self.amount_precision}f",
            amount0,
            amount1,
        )

        # Minting formula
        if self.total_liquidity == 0:
            liquidity_minted = math.sqrt(amount0 * amount1)
            if liquidity_minted <= 0:
                raise ValueError("Liquidity minted must be positive.")
            required0 = amount0
            required1 = amount1
        else:
            liquidity_from_0 = (amount0 * self.total_liquidity) / self.x
            liquidity_from_1 = (amount1 * self.total_liquidity) / self.y
            liquidity_minted = min(liquidity_from_0, liquidity_from_1)

            if liquidity_minted <= 0:
                raise ValueError("Liquidity minted must be positive.")

            # Adjust actual amounts consumed based on liquidity minted
            required0 = (liquidity_minted * self.x) / self.total_liquidity
            required1 = (liquidity_minted * self.y) / self.total_liquidity

        self.x += required0
        self.y += required1
        self.total_liquidity += liquidity_minted

        pos = PositionV2(liquidity=liquidity_minted, is_tracked=is_tracked)
        self.positions.append(pos)

        logger.debug(
            (
                "Minted LP = %s.{self.amount_precision}f, "
                "required0 = %s.{self.amount_precision}f, "
                "required1 = %s.{self.amount_precision}f, "
                "New state: %s"
            ),
            liquidity_minted,
            required0,
            required1,
            self,
        )

        return pos, required0, required1, liquidity_minted

    def remove_liquidity(
        self, position_id: int, liquidity_to_burn: float = None, fraction: float = None
    ) -> tuple[float, float]:
        """Remove liquidity from the pool and return amounts of token0 and token1 withdrawn."""
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos is None:
            raise ValueError(f"PositionV2 with id = {position_id} not found.")

        if liquidity_to_burn is None and fraction is None:
            raise ValueError("Either liquidity_to_burn or fraction must be provided.")

        if fraction is not None:
            if not 0 < fraction <= 1:
                raise ValueError("Fraction must be in (0, 1].")
            liquidity_to_burn = pos.liquidity * fraction

        if liquidity_to_burn <= 0 or liquidity_to_burn > pos.liquidity:
            raise ValueError("Invalid liquidity to burn.")

        logger.debug(
            "Removing liquidity: position_id=%d, liquidity_to_burn=%s.{self.amount_precision}f",
            position_id,
            liquidity_to_burn,
        )

        # Pro-rata amounts
        share = liquidity_to_burn / self.total_liquidity
        amount0 = share * self.x
        amount1 = share * self.y

        # Apply withdrawal fee
        fee0 = amount0 * self.withdrawal_fee
        fee1 = amount1 * self.withdrawal_fee

        amount0_after = amount0 - fee0
        amount1_after = amount1 - fee1

        # Update pool and accounting
        self.x -= amount0
        self.y -= amount1
        self.total_liquidity -= liquidity_to_burn

        pos.liquidity -= liquidity_to_burn
        if pos.liquidity == 0:
            self.positions.remove(pos)

        self.total_withdrawal_fees_token0 += fee0
        self.total_withdrawal_fees_token1 += fee1

        logger.debug(
            (
                "Burned LP = %s.{self.amount_precision}f, "
                "amount0 = %s.{self.amount_precision}f, "
                "amount1 = %s.{self.amount_precision}f, "
                "fee0 = %s.{self.amount_precision}f, "
                "fee1 = %s.{self.amount_precision}f, "
                "New state: %s"
            ),
            liquidity_to_burn,
            amount0_after,
            amount1_after,
            fee0,
            fee1,
            self,
        )

    def swap(self, amount_in: float, token0_for_token1: bool = True) -> float:
        """Swap tokens in the pool and return the amount of output tokens received."""

        if amount_in <= 0:
            raise ValueError("Amount in must be positive.")

        logger.debug(
            "Swapping: amount_in=%s.{self.amount_precision}f, token0_for_token1=%s",
            amount_in,
            token0_for_token1,
        )

        if token0_for_token1:
            reserve_in, reserve_out = self.x, self.y
        else:
            reserve_in, reserve_out = self.y, self.x

        amount_in_after_fee = amount_in * (1.0 - self.fee)
        # Track the fee component separately
        fee_amount = amount_in - amount_in_after_fee
        if token0_for_token1:
            self.total_fees_token0 += fee_amount
        else:
            self.total_fees_token1 += fee_amount

        amount_out = (amount_in_after_fee * reserve_out) / (
            reserve_in + amount_in_after_fee
        )
        if amount_out <= 0 or amount_out >= reserve_out:
            raise ValueError(
                "Invalid swap resulting in non-positive or excessive output."
            )

        # Update reserves
        if token0_for_token1:
            self.x += amount_in_after_fee + fee_amount
            self.y -= amount_out
            self.total_swaps_token0_in += amount_in
            self.total_swaps_token1_out += amount_out
        else:
            self.y += amount_in_after_fee + fee_amount
            self.x -= amount_out
            self.total_swaps_token1_in += amount_in
            self.total_swaps_token0_out += amount_out

        logger.debug(
            ("amount_out = %s.{self.amount_precision}f, " "New state: %s"),
            amount_out,
            self,
        )
        return amount_out
