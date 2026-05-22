# Data sources
# Configuration
from .config import DataConfig, config
from .crypto import fetch_crypto_data

# Exceptions
from .exceptions import (
    APIConnectionError,
    AuthenticationError,
    DataNotAvailableError,
    DataSourceError,
    DataValidationError,
    InvalidParametersError,
    RateLimitError,
)

# Indicators
from .indicators.registry import IndicatorMetadata, IndicatorRegistry

# Partitioning
from .partitioning import DataSplitter, DateRangeSplitter, RatioSplitter

# Processing
from .processing import DataProcessor

# Feature Generators
from .processing.features.base import FeatureGenerator
from .processing.features.sentiment import SentimentFeatureGenerator
from .processing.features.technical import TechnicalFeatureGenerator
from .processing.pipeline import DataPipeline

# Sentiment Providers
from .processing.sentiment import (
    HuggingFaceConfig,
    HuggingFaceProvider,
    SentimentConfig,
    SentimentProvider,
)
from .processing.steps import (
    ColumnCleanupStep,
    NumericConversionStep,
    ProcessingStep,
    SentimentEnrichmentStep,
    TechnicalIndicatorStep,
)
from .source_registry import DataSourceRegistry
from .sources import (
    AlpacaDataLoader,
    AlphaVantageDataLoader,
    FMPDataSource,
    YFinanceDataLoader,
)

__all__ = [
    # Sources
    "AlpacaDataLoader",
    "YFinanceDataLoader",
    "AlphaVantageDataLoader",
    "FMPDataSource",
    # Processing
    "DataProcessor",
    "DataPipeline",
    "ProcessingStep",
    "TechnicalIndicatorStep",
    "SentimentEnrichmentStep",
    "ColumnCleanupStep",
    "NumericConversionStep",
    "DataSourceRegistry",
    "IndicatorRegistry",
    "IndicatorMetadata",
    # Partitioning
    "DataSplitter",
    "RatioSplitter",
    "DateRangeSplitter",
    # Sentiment
    "SentimentProvider",
    "SentimentConfig",
    "HuggingFaceConfig",
    "HuggingFaceProvider",
    "SentimentFeatureGenerator",
    # Feature Generators
    "FeatureGenerator",
    "TechnicalFeatureGenerator",
    # Configuration
    "DataConfig",
    "config",
    # Exceptions
    "DataSourceError",
    "DataNotAvailableError",
    "APIConnectionError",
    "InvalidParametersError",
    "DataValidationError",
    "RateLimitError",
    "AuthenticationError",
]
