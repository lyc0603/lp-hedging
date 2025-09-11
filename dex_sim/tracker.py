"""Trackers for CPMM simulation."""

import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd

from dex_sim.cpmm import PoolV2, PositionV2

logger = logging.getLogger(__name__)


def _lp_share_fraction(pool: PoolV2, lp_liquidity: float) -> float:
    """Calculate the fraction of the pool owned by the LP."""
    tl = getattr(pool, "total_liquidity", 0.0)
    if tl <= 0:
        return 0.0
    return lp_liquidity / tl


class PositionV2Tracker:
    """Tracks a single PositionV2 over time in a PoolV2."""

    def __init__(self, position: PositionV2):
        if position is None:
            raise ValueError("Position cannot be None.")
        if not position.is_tracked:
            raise ValueError("Position must be tracked to be able to track it.")
        self.position_id = position.id
        self.data = []

    def update_data(self, pool: PoolV2, timestep: int):
        """Update the tracked data for the position at the given timestep."""

        pos = next((p for p in pool.positions if p.id == self.position_id), None)
        if pos is None:
            raise ValueError(
                f"PositionV2 with id = {self.position_id} not found in pool."
            )

        s = _lp_share_fraction(pool, pos.liquidity)
        amount_token0 = s * pool.x
        amount_token1 = s * pool.y
        vaslue_token1 = s * (pool.x * pool.price + pool.y)

        tracked_values = {
            "timestep": timestep,
            "id": pos.id,
            "liquidity": pos.liquidity,
            "share": s,
            "amount_token0": amount_token0,
            "amount_token1": amount_token1,
            "value_token1": vaslue_token1,
        }
        logger.debug("Tracked CPMM position values: %s", tracked_values)
        self.data.append(tracked_values)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the tracked data to a pandas DataFrame."""
        return pd.DataFrame(self.data)


class MultiPositionV2Tracker:
    """Tracks multiple PositionV2 instances over time in a PoolV2."""

    def __init__(self):
        self.position_ids = set()
        self.position_trackers: Dict[int, PositionV2Tracker] = {}

    def add_position(self, position: PositionV2):
        """Add a PositionV2 to be tracked."""
        if position.id in self.position_ids:
            logger.warning("Position %d is already tracked.", position.id)
            return
        if not getattr(position, "is_tracked", False):
            raise ValueError("Position must be tracked to be able to track it.")
        self.position_ids.add(position.id)
        self.position_trackers[position.id] = PositionV2Tracker(position)

    def add_positions(self, positions: List[PositionV2]):
        """Add multiple PositionV2 instances to be tracked."""
        for position in positions:
            self.add_position(position)

    def update_data(self, pool: PoolV2, timestep: int):
        """Update the tracked data for all positions at the given timestep."""
        for _id, tracker in self.position_trackers.items():
            tracker.update_data(pool, timestep)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the tracked data for all positions to a single pandas DataFrame."""
        dfs = [tracker.to_dataframe() for tracker in self.position_trackers.values()]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


class PoolV2Tracker:
    """Tracks the state of a PoolV2 over time."""

    def __init__(self, pool: PoolV2):
        self.pool_id = id(pool)
        self.data = []

    def update(self, pool: PoolV2, timestep: int):
        """Update the tracked data for the pool at the given timestep."""
        if id(pool) != self.pool_id:
            raise ValueError("Tracking a different pool instance is not supported.")

        tracked_values = {
            "timestep": timestep,
            "price": pool.price,
            "reserve0_x": pool.x,
            "reserve1_y": pool.y,
            "total_liquidity": getattr(pool, "total_liquidity", 0.0),
            "total_fees_token0": getattr(pool, "total_fees_token0", 0.0),
            "total_fees_token1": getattr(pool, "total_fees_token1", 0.0),
            "total_swap_token0_in": getattr(pool, "total_swap_token0_in", 0.0),
            "total_swap_token0_out": getattr(pool, "total_swap_token0_out", 0.0),
            "total_swap_token1_in": getattr(pool, "total_swap_token1_in", 0.0),
            "total_swap_token1_out": getattr(pool, "total_swap_token1_out", 0.0),
        }

        logger.debug("Tracked CPMM pool values: %s", tracked_values)
        self.data.append(tracked_values)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the tracked data to a pandas DataFrame."""
        return pd.DataFrame(self.data)


class LPV2AgentTracker:
    """Tracks a liquidity provider agent over time in a PoolV2."""

    def __init__(self, agent):
        self.agent_id = id(agent)
        self.positions_tracker = MultiPositionV2Tracker()
        self.data = []

    def _get_agent_position(self, agent) -> Optional[PositionV2]:
        """Get the current position of the agent."""
        pos = getattr(agent, "lp_position", None)
        if pos is None:
            pos = getattr(agent, "position", None)

        return pos

    def update(self, agent, pool: PoolV2, timestep: int):
        """Update the tracked data for the agent at the given timestep."""
        if id(agent) != self.agent_id:
            raise ValueError("Tracking a different agent instance is not supported.")

        pos = self._get_agent_position(agent)
        lp_liquidity = getattr(agent, "lp_liquidity", getattr(pos, "liquidity", 0.0))
        lp_position_id = getattr(pos, "id", None)

        tracked_values = {
            "timestep": timestep,
            "lp_position_id": lp_position_id,
            "lp_liquidity": lp_liquidity,
            "V": getattr(agent, "V", None),
            # Optional fields for hedged LPs:
            "Pi": getattr(agent, "Pi", None),
            "Delta": getattr(agent, "delta", None),
            "B": getattr(agent, "cash_B", None),
        }
        logger.debug("Tracked CPMM agent values: %s", tracked_values)
        self.data.append(tracked_values)

        # Track the LPShare itself (if any and tracked)
        if pos is not None and getattr(pos, "is_tracked", False):
            if pos.id not in self.positions_tracker.position_ids:
                self.positions_tracker.add_position(pos)
            self.positions_tracker.update_data(pool, timestep)

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert the tracked data to pandas DataFrames."""
        agent_df = pd.DataFrame(self.data)
        position_df = self.positions_tracker.to_dataframe()
        return agent_df, position_df
