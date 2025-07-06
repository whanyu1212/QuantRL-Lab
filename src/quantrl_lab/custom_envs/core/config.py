from pydantic import BaseModel, Field


class CoreEnvConfig(BaseModel):
    """Core environment configuration."""

    # Required fields for Pydantic models are indicated by ellipsis (...)
    # gt (greater than) and ge (greater than or equal to) are used for
    # numeric fields to enforce constraints
    # lt (less than) and le (less than or equal to) can also be used for constraints

    initial_balance: float = Field(..., gt=0, description="The initial cash balance for the agent.")
    window_size: int = Field(..., gt=0, description="The size of the observation window.")
    price_column_index: int = Field(..., ge=0, description="The column index for the price data.")
    transaction_cost_pct: float = Field(default=0.0, ge=0, lt=1, description="The percentage fee for each transaction.")
    slippage: float = Field(default=0.0, ge=0, lt=1, description="The slippage percentage for market orders.")
    order_expiration_steps: int = Field(
        default=5, gt=0, description="The number of steps before a pending order expires."
    )

    class Config:
        from_attributes = True
