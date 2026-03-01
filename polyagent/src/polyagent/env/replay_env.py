from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from polyagent.config import EnvConfig
from polyagent.env.fees import compute_fee
from polyagent.env.fills import simulate_fill


@dataclass(slots=True)
class PositionState:
    inventory: float = 0.0
    cash: float = 0.0


class ReplayEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, market_df: pd.DataFrame, cfg: EnvConfig) -> None:
        super().__init__()
        self.df = market_df.reset_index(drop=True)
        self.cfg = cfg
        self.max_steps = min(cfg.episode_length, len(self.df) - 2)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )

        self._idx = 0
        self._step_count = 0
        self._position = PositionState()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        self._idx = 0
        self._step_count = 0
        self._position = PositionState()
        obs = self._build_obs(self._idx)
        return obs, self._info(0.0, 0.0, 0.0, 0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        row = self.df.iloc[self._idx]
        next_row = self.df.iloc[self._idx + 1]
        requested_action = action

        if self._position.inventory >= self.cfg.max_inventory and action in (1, 2):
            action = 0
        elif self._position.inventory <= -self.cfg.max_inventory and action in (3, 4):
            action = 0

        if action == 5:
            self._position = PositionState()
            fill_qty = 0.0
            fill_notional = 0.0
            fee = 0.0
        else:
            fill = simulate_fill(
                action=action,
                order_size=self.cfg.order_size,
                best_bid=float(row.best_bid),
                best_ask=float(row.best_ask),
                next_trade_price=float(next_row.trade_price),
                next_trade_side=int(next_row.trade_side),
                tick_size=self.cfg.tick_size,
            )
            fill_qty = fill.filled_qty
            fill_notional = fill.fill_price * fill_qty
            fee_bps = self.cfg.maker_fee_bps if fill.is_maker else self.cfg.taker_fee_bps
            fee = compute_fee(fill_notional, fee_bps)
            self._position.inventory += fill_qty
            self._position.cash -= fill_notional
            self._position.cash -= fee

        mid_now = 0.5 * (float(row.best_bid) + float(row.best_ask))
        mid_next = 0.5 * (float(next_row.best_bid) + float(next_row.best_ask))
        mtm_pnl = self._position.inventory * (mid_next - mid_now)
        inventory_penalty = self.cfg.inventory_penalty * abs(self._position.inventory)
        impact_penalty = self.cfg.impact_penalty * abs(fill_qty)
        reward = mtm_pnl - inventory_penalty - impact_penalty - fee

        self._idx += 1
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = False
        obs = self._build_obs(self._idx)
        info = self._info(mtm_pnl, inventory_penalty, impact_penalty, fee)
        info["requested_action"] = requested_action
        info["executed_action"] = action
        return obs, float(reward), terminated, truncated, info

    def _build_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        mid = 0.5 * (float(row.best_bid) + float(row.best_ask))
        spread = float(row.best_ask) - float(row.best_bid)
        flow = float(row.trade_side) * float(row.trade_size)
        t_norm = idx / max(len(self.df) - 1, 1)
        remain = max(self.max_steps - self._step_count, 0) / max(self.max_steps, 1)

        features = np.array(
            [
                float(row.best_bid),
                float(row.best_ask),
                mid,
                spread,
                float(row.bid_size_1),
                float(row.ask_size_1),
                float(row.bid_size_2),
                float(row.ask_size_2),
                float(row.bid_size_3),
                float(row.ask_size_3),
                float(row.trade_price),
                float(row.trade_size),
                float(row.trade_side),
                flow,
                self._position.inventory,
                self._position.cash,
                t_norm,
                remain,
                0.0,  # optional news embedding placeholder dim 1
                0.0,  # optional news embedding placeholder dim 2
            ],
            dtype=np.float32,
        )
        return features

    def _info(
        self,
        mtm_pnl: float,
        inventory_penalty: float,
        impact_penalty: float,
        fee: float,
    ) -> dict[str, Any]:
        return {
            "inventory": self._position.inventory,
            "cash": self._position.cash,
            "mtm_pnl": mtm_pnl,
            "inventory_penalty": inventory_penalty,
            "impact_penalty": impact_penalty,
            "fee": fee,
        }
