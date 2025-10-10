HEDGE_SCREENING_RESPONSE_SCHEMA = {
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
