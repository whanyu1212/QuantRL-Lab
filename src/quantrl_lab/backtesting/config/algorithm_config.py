from dataclasses import asdict, dataclass


@dataclass
class PPOConfig:
    """Default configuration for PPO algorithm."""

    ent_coef: float = 0.0
    learning_rate: float = 0.0003
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Remark: you can use string-based type annotation
    # for dynamic typing in Python 3.8+
    def update(self, **kwargs) -> 'PPOConfig':
        """
        Update configuration with new parameters.

        Raises:
            ValueError: If an unknown parameter is provided.

        Returns:
            PPOConfig: The updated configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def copy(self) -> 'PPOConfig':
        """
        Create a copy of this configuration.

        Returns:
            PPOConfig: A copy of the current configuration.
        """
        return PPOConfig(**asdict(self))


@dataclass
class A2CConfig:
    """Default configuration for A2C algorithm."""

    ent_coef: float = 0.01
    learning_rate: float = 0.0007
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    use_rms_prop: bool = True

    def update(self, **kwargs) -> 'A2CConfig':
        """
        _summary_

        Raises:
            ValueError: If an unknown parameter is provided.

        Returns:
            A2CConfig: The updated configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def copy(self) -> 'A2CConfig':
        """
        Create a copy of this configuration.

        Returns:
            A2CConfig: A copy of the current configuration.
        """
        return A2CConfig(**asdict(self))


@dataclass
class SACConfig:
    """Default configuration for SAC algorithm."""

    learning_rate: float = 0.0003
    buffer_size: int = 1000000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = 'auto'
    target_update_interval: int = 1
    target_entropy: str = 'auto'

    def update(self, **kwargs) -> 'SACConfig':
        """
        Update configuration with new parameters.

        Raises:
            ValueError: If an unknown parameter is provided.

        Returns:
            SACConfig: The updated configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def copy(self) -> 'SACConfig':
        """
        Create a copy of this configuration.

        Returns:
            SACConfig: A copy of the current configuration.
        """
        return SACConfig(**asdict(self))
