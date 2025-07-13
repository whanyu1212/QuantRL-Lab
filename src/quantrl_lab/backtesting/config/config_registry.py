from typing import Type, Union

from stable_baselines3 import A2C, PPO, SAC

from .algorithm_config import A2CConfig, PPOConfig, SACConfig


class AlgorithmConfigRegistry:
    """Registry for algorithm configurations with predefined presets."""

    @staticmethod
    def get_default_config(algo_class: Type) -> Union[PPOConfig, A2CConfig, SACConfig]:
        """
        Get default configuration for algorithm.

        Args:
            algo_class (Type): The algorithm class for which to get the default config.

        Raises:
            ValueError: If the algorithm class is not supported.

        Returns:
            Union[PPOConfig, A2CConfig, SACConfig]: The default configuration for the algorithm.
        """
        if algo_class == PPO:
            return PPOConfig()
        elif algo_class == A2C:
            return A2CConfig()
        elif algo_class == SAC:
            return SACConfig()
        else:
            raise ValueError(f"Unsupported algorithm class: {algo_class.__name__}")

    @staticmethod
    def get_explorative_config(algo_class: Type) -> Union[PPOConfig, A2CConfig, SACConfig]:
        """
        Get explorative configuration for algorithm.

        Args:
            algo_class (Type): The algorithm class for which to get the explorative config.

        Raises:
            ValueError: If the algorithm class is not supported.

        Returns:
            Union[PPOConfig, A2CConfig, SACConfig]: The explorative configuration for the algorithm.
        """
        if algo_class == PPO:
            return PPOConfig(
                ent_coef=0.05,  # Higher entropy coefficient for more exploration
                learning_rate=0.001,  # Higher learning rate for faster adaptation
                clip_range=0.3,  # Wider clip range for more aggressive updates
                n_steps=1024,  # Shorter rollouts for more frequent updates
                batch_size=32,  # Smaller batch size for more variance
                n_epochs=5,  # Fewer epochs to maintain exploration
                gae_lambda=0.9,  # Lower GAE lambda for shorter-term rewards
            )
        elif algo_class == A2C:
            return A2CConfig(
                ent_coef=0.1,  # Much higher entropy for exploration
                learning_rate=0.002,  # Higher learning rate
                n_steps=3,  # Very short rollouts for frequent updates
                vf_coef=0.1,  # Lower value function coefficient
                max_grad_norm=1.0,  # Higher gradient norm for more aggressive updates
            )
        elif algo_class == SAC:
            return SACConfig(
                learning_rate=0.002,  # Higher learning rate
                learning_starts=50,  # Start learning earlier
                batch_size=128,  # Smaller batch for more variance
                tau=0.01,  # Faster target network updates
                train_freq=1,  # Train every step
                gradient_steps=2,  # More gradient steps per update
                ent_coef="auto",  # Auto entropy tuning for exploration
                target_entropy="auto",  # Auto target entropy
            )
        else:
            raise ValueError(f"Unsupported algorithm class: {algo_class.__name__}")

    @staticmethod
    def get_conservative_config(algo_class: Type) -> Union[PPOConfig, A2CConfig, SACConfig]:
        """
        Get conservative configuration for algorithm (lower learning
        rates, etc.).

        Args:
            algo_class (Type): The algorithm class for which to get the conservative config.

        Raises:
            ValueError: If the algorithm class is not supported.

        Returns:
            Union[PPOConfig, A2CConfig, SACConfig]: The conservative configuration for the algorithm.
        """

        if algo_class == PPO:
            return PPOConfig(ent_coef=0.0, learning_rate=0.0001, clip_range=0.1)
        elif algo_class == A2C:
            return A2CConfig(ent_coef=0.001, learning_rate=0.0003)
        elif algo_class == SAC:
            return SACConfig(learning_rate=0.0001)
        else:
            raise ValueError(f"Unsupported algorithm class: {algo_class.__name__}")

    @staticmethod
    def get_preset_config(algo_class: Type, preset: str) -> Union[PPOConfig, A2CConfig, SACConfig]:
        """
        Get configuration for algorithm based on preset name.

        Args:
            algo_class (Type): The algorithm class for which to get the config.
            preset (str): The preset name ("default", "explorative", "conservative").

        Raises:
            ValueError: If the algorithm class or preset is not supported.

        Returns:
            Union[PPOConfig, A2CConfig, SACConfig]: The configuration for the algorithm and preset.
        """
        if preset == "default":
            return AlgorithmConfigRegistry.get_default_config(algo_class)
        elif preset == "explorative":
            return AlgorithmConfigRegistry.get_explorative_config(algo_class)
        elif preset == "conservative":
            return AlgorithmConfigRegistry.get_conservative_config(algo_class)
        else:
            raise ValueError(f"Unsupported preset: {preset}. Available presets: default, explorative, conservative")
