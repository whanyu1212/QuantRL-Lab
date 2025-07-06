from __future__ import annotations


class Portfolio:
    """A minimal, generic asset-agnostic portfolio class that can be
    used in various trading environments."""

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance

        # --- Core State ---
        self.balance: float = 0.0
        self.units_held: int = 0  # Generic term for any asset (shares, contracts, coins etc.)

        # Initialize the state
        self.reset()

    def reset(self) -> None:
        """Resets the portfolio's core state to its initial values."""
        self.balance = self.initial_balance
        self.units_held = 0

    def get_value(self, current_price: float) -> float:
        """Calculates the total value of the portfolio at a given
        price."""
        return self.balance + (self.units_held * current_price)
