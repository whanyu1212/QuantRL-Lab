import json
import logging
from typing import List, Optional

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class LLMStockScreener:
    """Simple, robust LLM-based stock screener using LiteLLM."""

    def __init__(self, model_name: str, validate_on_init: bool = True):
        """
        Initialize the screener.

        Args:
            model_name: LiteLLM model name (e.g., "openai/gpt-4", "gemini/gemini-pro")
            validate_on_init: Whether to test the connection on initialization
        """
        self.model_name = model_name

        if validate_on_init:
            self._quick_test()

    def _quick_test(self) -> None:
        """
        Quick test to ensure the model works.

        Uses LiteLLM's built-in error handling for API key validation.
        """
        try:
            response = completion(model=self.model_name, messages=[{"role": "user", "content": "Hi"}], max_tokens=5)
            if not response.choices:
                raise RuntimeError(f"Model {self.model_name} returned empty response")
            logger.info(f"✓ Model {self.model_name} is ready")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {self.model_name}. " f"Check your API keys and model name. Error: {str(e)}"
            ) from e

    def run(self, query: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Run a query against the LLM.

        Args:
            query: The user question or prompt to send
            system_prompt: Optional system instruction to guide the model's behavior
            **kwargs: Additional parameters for the completion call (including tools, tool_choice, etc.)

        Returns:
            str: The LLM's response
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Build messages array with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        try:
            response = completion(model=self.model_name, messages=messages, **kwargs)

            # Handle tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                # If tools were called, return a structured response
                tool_calls = response.choices[0].message.tool_calls
                tool_info = []
                for tool_call in tool_calls:
                    tool_info.append(f"Tool: {tool_call.function.name}, Args: {tool_call.function.arguments}")

                content = response.choices[0].message.content or ""
                tool_summary = "\n".join(tool_info)
                return f"{content}\n\nTool calls made:\n{tool_summary}".strip()

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty response from model")

            return content.strip()

        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}") from e

    def screen_hedge_stocks(self, target_stock: str, hedge_criteria: str = "", **kwargs) -> str:
        """
        Screen for stocks that can be used to hedge against a target
        stock with real-time search.

        Args:
            target_stock: The stock symbol or description to hedge against (e.g., "TSLA", "Tesla")
            hedge_criteria: Additional criteria for the hedge (e.g., "low correlation", "inverse ETFs",
                "puts available")
            **kwargs: Additional parameters for the completion call

        Returns:
            str: Hedge stock recommendations with real-time data
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
        # Add tools to kwargs if not already present
        if 'tools' not in kwargs:
            kwargs['tools'] = [{"googleSearch": {}}]
        if 'tool_choice' not in kwargs:
            kwargs['tool_choice'] = 'auto'

        return self.run(user_prompt, system_prompt=system_prompt, **kwargs)

    def screen_hedge_stocks_structured(
        self, target_stock: str, hedge_criteria: str = "", **kwargs
    ) -> HedgeScreeningResult:
        """
        Screen for hedge stocks with structured Pydantic output and
        real-time search.

        Args:
            target_stock: The stock to hedge against
            hedge_criteria: Additional hedge criteria
            **kwargs: Additional parameters for the completion call

        Returns:
            HedgeScreeningResult: Structured hedge recommendations with real-time data
        """
        # Define the response schema
        response_schema = {
            "type": "object",
            "properties": {
                "target_stock": {"type": "string", "description": "Stock being hedged"},
                "hedge_criteria": {"type": "string", "description": "Criteria used for screening"},
                "hedge_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Hedge instrument ticker symbol"},
                            "name": {"type": "string", "description": "Company or fund name"},
                            "hedge_type": {
                                "type": "string",
                                "enum": [
                                    "negative_correlation",
                                    "inverse_sector",
                                    "volatility_hedge",
                                    "commodity_hedge",
                                    "currency_hedge",
                                ],
                                "description": "Type of hedge",
                            },
                            "correlation": {
                                "type": "string",
                                "description": "Historical correlation coefficient if available",
                            },
                            "rationale": {"type": "string", "description": "Why this is an effective hedge"},
                            "hedge_ratio": {
                                "type": "string",
                                "description": "Suggested position sizing relative to target",
                            },
                            "effectiveness_conditions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Market conditions when this hedge works best",
                            },
                            "limitations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "When this hedge may fail",
                            },
                            "liquidity": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Execution liquidity",
                            },
                        },
                        "required": [
                            "symbol",
                            "name",
                            "hedge_type",
                            "rationale",
                            "hedge_ratio",
                            "effectiveness_conditions",
                            "limitations",
                            "liquidity",
                        ],
                    },
                },
                "overall_strategy": {"type": "string", "description": "Summary of the hedging approach"},
                "disclaimer": {"type": "string", "description": "Risk management disclaimer"},
            },
            "required": ["target_stock", "hedge_criteria", "hedge_recommendations", "overall_strategy", "disclaimer"],
        }

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

        # Use enhanced JSON mode with schema validation
        kwargs['response_format'] = {
            "type": "json_object",
            "response_schema": response_schema,
            "enforce_validation": True,
        }

        # Remove tools if present since they conflict with structured output
        if 'tools' in kwargs:
            del kwargs['tools']
        if 'tool_choice' in kwargs:
            del kwargs['tool_choice']

        response = self.run(user_prompt, system_prompt=system_prompt, **kwargs)

        # Response should be JSON string when using json_object mode
        if isinstance(response, str) and response.strip():
            try:
                response_data = json.loads(response)
                return HedgeScreeningResult(**response_data)
            except (json.JSONDecodeError, Exception) as e:
                raise RuntimeError(f"Failed to parse JSON response: {e}. Response: {response[:500]}...") from e

        # Check if response is already a dict (unlikely with json_object mode)
        if isinstance(response, dict):
            return HedgeScreeningResult(**response)

        # If we get here, something went wrong
        raise RuntimeError(f"Unexpected response format. Type: {type(response)}, Content: {response[:200]}...")


if __name__ == "__main__":
    # Example usage
    load_dotenv()  # Load environment variables if needed
    screener = LLMStockScreener(model_name="gemini/gemini-2.5-flash")

    # print("=== Hedge Screening (with real-time search but no structured output) ===")
    # try:
    #     hedge_result = screener.screen_hedge_stocks("MU", "protect against tech sector downturn")
    #     print(hedge_result)
    # except Exception as e:
    #     print(f"Search-enabled hedge screening failed: {e}")

    print("\n=== Structured Hedge Screening ===")
    try:
        structured_hedge = screener.screen_hedge_stocks_structured("MU", "protect against tech sector downturn")
        print(f"Target stock: {structured_hedge.target_stock}")
        print(f"Strategy: {structured_hedge.overall_strategy}")
        print(f"Number of hedge recommendations: {len(structured_hedge.hedge_recommendations)}")

        print(structured_hedge)
        if structured_hedge.hedge_recommendations:
            first_hedge = structured_hedge.hedge_recommendations[0]
            print(f"Top hedge symbol: {first_hedge.symbol}")
            print(f"Top hedge name: {first_hedge.name}")
            print(f"Correlation: {first_hedge.correlation}")
            print(f"Hedge ratio: {first_hedge.hedge_ratio}")
            print(f"Effectiveness conditions: {', '.join(first_hedge.effectiveness_conditions)}")
            print(f"Limitations: {', '.join(first_hedge.limitations)}")
            print(f"Liquidity: {first_hedge.liquidity}")
            print(f"Hedge type: {first_hedge.hedge_type}")
            print(f"Rationale: {first_hedge.rationale}")
    except Exception as e:
        print(f"Structured hedge screening failed: {e}")

    # Example of disabling search tools if needed
    # print("\n=== Hedge Screening without search tools ===")
    # try:
    #     no_search_result = screener.screen_hedge_stocks(
    #         "MU",
    #         "protect against tech sector downturn",
    #         tools=[]  # Disable tools
    #     )
    #     print(no_search_result)
    # except Exception as e:
    #     print(f"Non-search hedge screening failed: {e}")
