"""
Data processing modules for feature engineering and transformation.

This package contains processors and mappings for transforming raw
market data.
"""

from .metadata import ProcessingMetadata
from .pipeline import DataPipeline
from .pipeline_config import (
    CleanupConfig,
    CrossSectionalConfig,
    ProcessingPipelineConfig,
    SplitConfig,
)
from .processor import DataProcessor

__all__ = [
    "CleanupConfig",
    "CrossSectionalConfig",
    "DataPipeline",
    "DataProcessor",
    "ProcessingMetadata",
    "ProcessingPipelineConfig",
    "SplitConfig",
]
