# QuantRL-Lab
A Python testbed for Reinforcement Learning in finance, designed to enable researchers and developers to experiment with and evaluate RL algorithms in financial contexts. The project emphasizes modularity and configurability, allowing users to tailor the environment, data sources, and algorithmic settings to their specific needs

The repository demonstrates a complete workflow:

1. Load market data from multiple sources
2. Process data with technical indicators
3. Configure trading environments with different parameters
4. Train RL algorithms with custom or preset configurations
5. Evaluate performance across multiple metrics
6. Generate comprehensive comparison reports

---

### Why Configurability Matters
QuantRL-Lab is built with configurability at its core, ensuring that:
- **Flexibility**: Users can easily adapt the testbed to different financial instruments, data sources, and RL algorithms.
- **Reproducibility**: Configurable settings make it straightforward to replicate experiments and share results.
- **Scalability**: Modular design allows for seamless integration of new features, such as custom environments, policies, or data pipelines.
- **Efficiency**: By enabling fine-grained control over configurations, users can optimize computational resources and focus on specific aspects of their research.

Whether you're exploring single-stock trading strategies or multi-agent portfolio optimization, QuantRL-Lab provides the tools and framework to accelerate your research and development.


---
### Setup Guide

1. Clone the Repository
```bash
git clone https://github.com/whanyu1212/QuantRL-Lab.git
```

2. Install Poetry for dependency management
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Sync dependencies (It also installs the current project in dev mode)
```bash
poetry install
```

4. Activate virtual environment (Note that the `shell` command is deprecated in the latest poetry version)
```bash
poetry env activate
# a venv path will be printed in the terminal, just copy and run it
# e.g.,
source /home/codespace/.cache/pypoetry/virtualenvs/multi-agent-quant-cj6_z41n-py3.12/bin/activate
```

5. Install jupyter kernel
```bash
# You can change the name and display name according to your preference
python -m ipykernel install --user --name multi-agent-quant --display-name "Multi Agent Quant"
```

6. Set up environment variables
```bash
# Copy the example environment file
cp .env.example .env

# Open .env file and replace the placeholder values with your actual credentials
# You can use any text editor, here using VS Code
code .env
```

Make sure to replace all placeholder values in the `.env` file with your actual API keys and credentials. Never commit the `.env` file to version control.

<br>

7. Set up pre-commit hooks
```bash
# Install pre-commit
poetry add pre-commit

# Install the git hooks
pre-commit install

# Optional: run pre-commit on all files
pre-commit run --all-files
```

The pre-commit hooks will check for:
- Code formatting (black)
- Import sorting (isort)
- Code linting (flake8)
- Docstring formatting (docformatter)
- Basic file checks (trailing whitespace, YAML validation, etc.)

To skip pre-commit hooks temporarily:
```bash
git commit -m "your message" --no-verify
```

For more details, please refer to `.pre-commit-config.yaml` file.

## Architecture Overview

QuantRL-Lab is built around a modular architecture with two primary layers that enable flexible experimentation with reinforcement learning in financial markets.

### 🗄️ Data Layer

The data layer provides a unified, extensible interface for market data acquisition, processing, and technical analysis. It's designed to abstract away the complexities of different data providers while maintaining consistency across all data sources.

#### Core Components

**1. Data Source Interface & Registry**
- **Abstract Base Classes**: All data sources implement the `DataSource` interface with standardized methods (`connect()`, `disconnect()`, `get_historical_ohlcv_data()`, etc.)
- **Protocol-Based Capabilities**: Mixins for different data types:
  - `HistoricalDataCapable`: OHLCV historical data
  - `NewsDataCapable`: News and sentiment data
  - `LiveDataCapable`: Real-time market data
  - `StreamingCapable`: Live data streaming
- **Centralized Registry**: `DataSourceRegistry` manages multiple data sources with configurable primary/secondary sources

**2. Data Loaders (Implemented)**
- ✅ **Alpaca Markets**: Full OHLCV + news data integration
- ✅ **Yahoo Finance**: Historical data with multi-symbol support
- 🔄 **Alpha Vantage**: Partial implementation (standardization in progress)
- 📋 **OANDA**: Placeholder for forex data

**3. Technical Indicators System**
- **Registry Pattern**: `IndicatorRegistry` enables plugin-style indicator registration
- **Implemented Indicators** (8 total):
  - SMA, EMA (Moving Averages)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - ATR (Average True Range)
  - BB (Bollinger Bands)
  - STOCH (Stochastic Oscillator)
  - OBV (On-Balance Volume)
- **Flexible Configuration**: Support for multiple parameter sets (e.g., `{"SMA": {"window": [10, 20, 50]}}`)

**4. Data Processing Pipeline**
- **Unified DataProcessor**: Central class for data transformation and enrichment
- **News Sentiment Analysis**: Integration with transformer models for sentiment scoring
- **Data Validation**: Column type checking, missing data handling
- **Flexible Processing**: Configurable pipelines with method chaining

#### What's Working ✅
- End-to-end data pipeline from raw market data to processed features
- Multiple data source integration with unified interface
- 8 technical indicators with extensible registry system
- News sentiment analysis with transformer models
- Flexible parameter configuration for indicators
- Data validation and type conversion utilities

#### Roadmap 🔄
- **Data Source Expansion**:
  - Complete Alpha Vantage integration
  - Add OANDA forex data support
  - Implement fundamental data sources (earnings, financials)
- **Technical Indicators**:
  - Add 10+ more indicators (Ichimoku, Williams %R, CCI, etc.)
  - Volume-based indicators
  - Custom indicator framework
- **Alternative Data**:
  - Economic calendar events
  - Sector performance data

### 🧪 Experiment Layer

*[Placeholder - This section will detail the reinforcement learning experiment framework, including environment management, algorithm configuration, training pipelines, and evaluation systems.]*

---

### Literature Review


### TODO
