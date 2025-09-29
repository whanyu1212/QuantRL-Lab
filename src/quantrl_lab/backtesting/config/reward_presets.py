from itertools import combinations

from quantrl_lab.backtesting.config.reward_config import RewardCombination, RewardStrategyConfig
from quantrl_lab.custom_envs.stock.strategies.rewards import (
    HoldPenalty,
    InvalidActionPenalty,
    PortfolioValueChangeReward,
    PositionSizingRiskReward,
    TrendFollowingReward,
)

reward_configs = [
    RewardStrategyConfig(name="portfolio_reward", strategy=PortfolioValueChangeReward),
    RewardStrategyConfig(name="invalid_penalty", strategy=InvalidActionPenalty, params={"penalty": -1.0}),
    RewardStrategyConfig(name="trend_reward", strategy=TrendFollowingReward),
    RewardStrategyConfig(name="hold_penalty", strategy=HoldPenalty, params={"penalty": -0.5}),
    RewardStrategyConfig(name="position_sizing_reward", strategy=PositionSizingRiskReward),
]

# Define preset reward combinations
reward_presets = {
    "balanced": RewardCombination(
        name="balanced",
        strategies=reward_configs,
        weights=[0.4, 0.2, 0.2, 0.1, 0.1],
    ),
    "conservative": RewardCombination(
        name="conservative",
        strategies=reward_configs,
        weights=[0.5, 1.0, 0.1, 0.2, 0.2],
    ),
    "aggressive": RewardCombination(
        name="aggressive",
        strategies=reward_configs,
        weights=[0.5, 0.1, 0.3, 0.05, 0.05],
    ),
    "risk_managed": RewardCombination(
        name="risk_managed",
        strategies=reward_configs,
        weights=[0.3, 0.5, 0.1, 0.05, 0.3],
    ),
}


if __name__ == "__main__":
    from pprint import pprint

    # Generate combinations of reward strategies
    for i in range(1, 5):
        for combo in combinations(reward_configs, i):
            combo_name = "_".join(c.name for c in combo)
            reward_presets[combo_name] = RewardCombination(
                name=combo_name,
                strategies=list(combo),
                weights=[1.0 / len(combo)] * len(combo),  # Equal weights as a starting point
            )
            print(f"Added preset: {combo_name}")
            pprint(reward_presets[combo_name])
            print("-" * 40)
