import json
from typing import Optional

from dotenv import load_dotenv
from litellm import completion
from loguru import logger

from quantrl_lab.screener.data_models import HedgeScreeningResult
from quantrl_lab.screener.prompt import (
    build_hedge_screening_prompts,
    build_structured_hedge_screening_prompts,
)
from quantrl_lab.screener.response_schemas import HEDGE_SCREENING_RESPONSE_SCHEMA


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
            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
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
        system_prompt, user_prompt = build_hedge_screening_prompts(target_stock, hedge_criteria)
        # Add tools to kwargs if not already present
        if "tools" not in kwargs:
            kwargs["tools"] = [{"googleSearch": {}}]
        if "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "auto"

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
        system_prompt, user_prompt = build_structured_hedge_screening_prompts(target_stock, hedge_criteria)

        # Use enhanced JSON mode with schema validation
        kwargs["response_format"] = {
            "type": "json_object",
            "response_schema": HEDGE_SCREENING_RESPONSE_SCHEMA,
            "enforce_validation": True,
        }

        # Remove tools if present since they conflict with structured output
        if "tools" in kwargs:
            del kwargs["tools"]
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]

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
    load_dotenv()
    screener = LLMStockScreener(model_name="gemini/gemini-2.5-pro")

    # logger.info("=== Hedge Screening (with real-time search but no structured output) ===")
    # try:
    #     hedge_result = screener.screen_hedge_stocks("MU", "protect against tech sector downturn")
    #     logger.info(hedge_result)
    # except Exception as e:
    #     logger.error(f"Search-enabled hedge screening failed: {e}")

    logger.info("\n=== Structured Hedge Screening ===")
    try:
        structured_hedge = screener.screen_hedge_stocks_structured("MU", "protect against tech sector downturn")
        logger.info(f"Target stock: {structured_hedge.target_stock}")
        logger.info(f"Strategy: {structured_hedge.overall_strategy}")
        logger.info(f"Number of hedge recommendations: {len(structured_hedge.hedge_recommendations)}")

        logger.info(structured_hedge)
        if structured_hedge.hedge_recommendations:
            first_hedge = structured_hedge.hedge_recommendations[0]
            logger.info(f"Top hedge symbol: {first_hedge.symbol}")
            logger.info(f"Top hedge name: {first_hedge.name}")
            logger.info(f"Correlation: {first_hedge.correlation}")
            logger.info(f"Hedge ratio: {first_hedge.hedge_ratio}")
            logger.info(f"Effectiveness conditions: {', '.join(first_hedge.effectiveness_conditions)}")
            logger.info(f"Limitations: {', '.join(first_hedge.limitations)}")
            logger.info(f"Liquidity: {first_hedge.liquidity}")
            logger.info(f"Hedge type: {first_hedge.hedge_type}")
            logger.info(f"Rationale: {first_hedge.rationale}")
    except Exception as e:
        logger.error(f"Structured hedge screening failed: {e}")

    # Example of disabling search tools if needed
    # logger.info("\n=== Hedge Screening without search tools ===")
    # try:
    #     no_search_result = screener.screen_hedge_stocks(
    #         "MU",
    #         "protect against tech sector downturn",
    #         tools=[]  # Disable tools
    #     )
    #     logger.info(no_search_result)
    # except Exception as e:
    #     logger.error(f"Non-search hedge screening failed: {e}")
