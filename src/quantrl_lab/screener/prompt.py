"""
Prompt builders for LLM hedge screening.

This module centralizes the system and user prompts used by the
LLMStockScreener. Keeping prompts here makes them easier to edit and
reuse without touching the screener logic or response schemas.
"""

from __future__ import annotations

from typing import Tuple


def build_hedge_screening_prompts(target_stock: str, hedge_criteria: str = "") -> Tuple[str, str]:
    """
    Build prompts for the search-enabled hedge screening flow.

    Returns a tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a professional risk management analyst with access to real-time market data.

Use the search tool to find current market information, correlations, and hedge effectiveness data.

When recommending hedge instruments:
1. Search for current correlation data between target and potential hedges
2. Look up recent performance of hedge instruments
3. Check current market conditions affecting hedge effectiveness
4. Verify liquidity and trading volumes
5. Consider sector rotation opportunities (cyclical vs defensive)
6. Look for volatility hedges (VIX products, low-beta stocks)
7. Check inverse/short ETFs in the same sector

Always provide:
1. Stock symbols and names of recommended hedges
2. Type of hedge relationship (negative correlation, inverse sector, etc.)
3. Current correlation data when available
4. Specific hedge ratios or position sizing suggestions
5. Risks and limitations of each hedge
6. Market conditions where the hedge works best/worst

Be specific about hedge effectiveness and provide actionable recommendations with current market context."""

    criteria_text = f" with additional criteria: {hedge_criteria}" if hedge_criteria else ""
    user_prompt = f"""Find effective hedge instruments for {target_stock}{criteria_text}.

Please search for:
1. Current correlation data for {target_stock} with potential hedge instruments
2. Recent market performance of defensive/inverse instruments
3. Current volatility and market conditions

Then provide specific hedge recommendations with current market context."""

    return system_prompt, user_prompt


def build_structured_hedge_screening_prompts(target_stock: str, hedge_criteria: str = "") -> Tuple[str, str]:
    """
    Build prompts for the structured JSON hedge screening flow.

    Returns a tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a professional risk management analyst.

For hedge_type field, use one of these values: negative_correlation, inverse_sector,
volatility_hedge, commodity_hedge, currency_hedge
For liquidity field, use one of: high, medium, low

When recommending hedge instruments, consider:
1. Historical correlation data between target and potential hedges
2. Recent performance patterns of hedge instruments
3. Current market conditions affecting hedge effectiveness
4. Liquidity and trading volumes
5. Sector rotation opportunities (cyclical vs defensive)
6. Volatility hedges (VIX products, low-beta stocks)
7. Inverse/short ETFs in the same sector

Provide hedge recommendations based on your training data and market knowledge."""

    criteria_text = f" with additional criteria: {hedge_criteria}" if hedge_criteria else ""
    user_prompt = f"""Find hedge instruments for {target_stock}{criteria_text}.

Analyze and provide hedge recommendations considering:
1. Historical correlation patterns for {target_stock}
2. Performance characteristics of potential hedge instruments
3. Current market environment and conditions
4. Liquidity and execution considerations

Provide structured hedge recommendations."""

    return system_prompt, user_prompt
