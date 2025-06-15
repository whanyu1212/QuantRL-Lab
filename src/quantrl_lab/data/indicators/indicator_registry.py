from typing import Callable, Dict, List

import pandas as pd


class IndicatorRegistry:

    # Mapping of indicator names to the respective functions to be applied
    _indicators: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str = None) -> Callable:
        """
        Register an indicator function.

        Args:
            name (str, optional): Defaults to None.

        Returns:
            Callable: Decorator function
        """

        def decorator(func: Callable):
            # register the function with either
            # the provided name or the function name itself
            indicator_name = name or func.__name__
            cls._indicators[indicator_name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get the indicator function by name.

        Args:
            name (str): Name of the indicator

        Raises:
            KeyError: If the name is not found in the registry

        Returns:
            Callable: Indicator function
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not registered")
        return cls._indicators[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered indicators.

        Returns:
            List[str]: a list of indicator names
        """
        return list(cls._indicators.keys())

    @classmethod
    def apply(cls, name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the indicator function to the dataframe.

        Args:
            name (str): Name of the indicator
            df (pd.DataFrame): input dataframe
            **kwargs: Additional keyword arguments to
            be passed to the indicator function

        Returns:
            pd.DataFrame: DataFrame with the indicator addedAdd commentMore actions
        """
        indicator_func = cls.get(name)
        return indicator_func(df, **kwargs)
