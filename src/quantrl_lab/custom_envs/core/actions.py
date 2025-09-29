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


class HedgingActions(IntEnum):

    Hold = 0  # Maintain the current position (either cash or an existing hedge).
    Liquidate = 1  # Close any open hedged position and return to cash.
    BuyA_SellB = 2  # Enter a long spread position.
    SellA_BuyB = 3  # Enter a short spread position.
