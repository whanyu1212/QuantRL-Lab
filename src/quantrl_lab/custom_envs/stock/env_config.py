from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """
    Configuration for the StockTradingEnv.

    Pydantic model for data validation and type hinting.
    """

    # ellipsis (...) indicate required fields for Pydantic models
    initial_balance: float = Field(..., gt=0, description="The initial cash balance for the agent.")
    window_size: int = Field(
        ...,
        gt=0,
        description="The number of past time steps to include in the observation.",
    )
    price_column_index: int = Field(
        ...,
        ge=0,
        description="The column index in the data array that contains the price.",
    )
    transaction_cost_pct: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="Transaction cost as a percentage of the trade value.",
    )
    slippage: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="Slippage as a percentage for market orders.",
    )
    order_expiration_steps: int = Field(
        default=5,
        gt=0,
        description="Number of steps after which a limit order expires.",
    )

    class Config:
        from_attributes = True
