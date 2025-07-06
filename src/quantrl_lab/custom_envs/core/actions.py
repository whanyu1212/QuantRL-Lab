from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    Hold = 0
    Buy = 1
    Sell = 2
    LimitBuy = 3
    LimitSell = 4
    StopLoss = 5
    TakeProfit = 6
