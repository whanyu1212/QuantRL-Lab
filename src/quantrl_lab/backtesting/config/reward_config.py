from typing import Any, Dict, List, Type

from pydantic import BaseModel

from quantrl_lab.custom_envs.stock.strategies.rewards import WeightedCompositeReward
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import BaseRewardStrategy


class RewardStrategyConfig(BaseModel):
    name: str
    strategy: Type[BaseRewardStrategy]
    params: Dict[str, Any] = {}

    def create_instance(self) -> BaseRewardStrategy:
        return self.strategy(**self.params)


class RewardCombination(BaseModel):
    name: str
    strategies: List[RewardStrategyConfig]
    weights: List[float]


def create_reward_strategy_from_combination(combination: 'RewardCombination') -> WeightedCompositeReward:
    """
    Creates a WeightedCompositeReward instance from a RewardCombination.

    Args:
        combination (RewardCombination): The reward combination configuration.

    Returns:
        WeightedCompositeReward: The created weighted composite reward instance.
    """
    strategy_instances = [config.create_instance() for config in combination.strategies]
    return WeightedCompositeReward(strategies=strategy_instances, weights=combination.weights)
