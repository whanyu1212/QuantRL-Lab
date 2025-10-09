from typing import List, Optional

from pydantic import BaseModel, Field


class HedgeRecommendation(BaseModel):
    """Individual hedge recommendation model."""

    symbol: str = Field(..., description="Hedge instrument ticker symbol")
    name: str = Field(..., description="Company or fund name")
    hedge_type: str = Field(
        ...,
        description=(
            "Type of hedge: negative_correlation, inverse_sector, " "volatility_hedge, commodity_hedge, currency_hedge"
        ),
    )
    correlation: Optional[str] = Field(None, description="Historical correlation coefficient if available")
    rationale: str = Field(..., description="Why this is an effective hedge")
    hedge_ratio: str = Field(..., description="Suggested position sizing relative to target")
    effectiveness_conditions: List[str] = Field(..., description="Market conditions when this hedge works best")
    limitations: List[str] = Field(..., description="When this hedge may fail")
    liquidity: str = Field(..., description="Execution liquidity: high, medium, low")


class HedgeScreeningResult(BaseModel):
    """Complete hedge screening result model."""

    target_stock: str = Field(..., description="Stock being hedged")
    hedge_criteria: str = Field(..., description="Criteria used for screening")
    hedge_recommendations: List[HedgeRecommendation] = Field(..., description="List of hedge recommendations")
    overall_strategy: str = Field(..., description="Summary of the hedging approach")
    disclaimer: str = Field(..., description="Risk management disclaimer")
