<!-- omit in toc -->
# QuantRL-Lab

[![PyPI version](https://badge.fury.io/py/quantrl-lab.svg)](https://badge.fury.io/py/quantrl-lab)
[![Python](https://img.shields.io/pypi/pyversions/quantrl-lab.svg)](https://pypi.org/project/quantrl-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python testbed for Reinforcement Learning in finance, designed to enable researchers and developers to experiment with and evaluate RL algorithms in financial contexts. The project emphasizes modularity and configurability, allowing users to tailor the environment, data sources, and algorithmic settings to their specific needs

## Installation

```bash
pip install quantrl-lab
```

## Table of Contents
- [Installation](#installation)
- [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [System Diagrams](#system-diagrams)
  - [Example usage:](#example-usage)
  - [Roadmap 🔄](#roadmap-)
  - [Development Setup](#development-setup)
    - [For Contributors and Developers](#for-contributors-and-developers)
- [Contributing](#contributing)
  - [How to Contribute](#how-to-contribute)
  - [Code of Conduct](#code-of-conduct)
- [Contributors](#contributors)
  - [Main Contributors](#main-contributors)
  - [How to Become a Contributor](#how-to-become-a-contributor)
  - [Literature Review](#literature-review)

### Motivation

**Addressing the Monolithic Environment Problem**

Most existing RL frameworks for finance suffer from tightly coupled, monolithic designs where action spaces, observation spaces, and reward functions are hardcoded into the environment initialization. This creates several critical limitations:

- **Limited Experimentation**: Users cannot easily test different reward formulations or action spaces without doing a lot of rewriting of the environments
- **Poor Scalability**: Adding new asset classes, trading strategies, or market conditions requires significant code restructuring
- **Reduced Reproducibility**: Inconsistent interfaces across different environment configurations make fair comparisons difficult
- **Development Overhead**: Simple modifications like testing different reward functions or adding new observation features require extensive refactoring


<u>The framework tries to demonstrate the following workflow:</u>
1. **Flexible Data Acquisition**: Aggregate market data from multiple heterogeneous sources with unified interfaces
2. **Feature Engineering**: Systematic selection and analysis of technical indicators (based on vectorized backtesting) for optimal signal generation
3. **Data Processing**: Enrich datasets with technical indicators and sentiment analysis from news sources
4. **Environment Configuration**: Define trading environments with customizable parameters (portfolio allocation, transaction costs, slippage, observation windows)
5. **Algorithm Training & Tuning**: Execute RL algorithm training with preset or configurable hyperparameters
6. **Performance Evaluation**: Assess model performance and action distribution
7. **Comparative Analysis**: Generate detailed performance reports

---

### System Diagrams

<details>
<summary><b>📋 Workflow Diagram</b> - End-to-end process from data acquisition to evaluation</summary>

```mermaid
flowchart TB
    A[Fetch Historical Data] --> B[Configure Pipeline]

    B --> C[Compute Indicators: RSI, MACD, etc.]

    C --> D[Instantiate Environment with Strategies]

    D --> E[Action Strategy]
    D --> F[Observation Strategy]
    D --> G[Reward Strategy]

    E --> H
    F --> H
    G --> H

    H[Train RL Agent: PPO/SAC/A2C] --> I[Evaluate vs Benchmarks]

    I --> J[Analyze Results]

    J --> K{Iterate?}

    K -->|Yes| B
    K -->|No| L[End]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#fce4ec
    style I fill:#e0f2f1
    style J fill:#f1f8e9
    style K fill:#fce4ec
    style L fill:#c8e6c9
```

</details>

<details>
<summary><b>🏗️ High-Level Architecture</b> - Layered system design with data, environment, and experiment layers</summary>

```mermaid
graph TB
    subgraph DL["📊 Data Layer"]
        DS[Data Sources<br/>Alpaca, Alpha Vantage<br/>Yahoo Finance, Polygon]
        UI[Unified Interface<br/>DataFetcher]
        PP[Processing Pipeline<br/>Technical Indicators<br/>Feature Engineering]
        DS --> UI
        UI --> PP
    end

    subgraph EL["🏪 Environment Layer"]
        TE[Trading Environment<br/>Gymnasium-based]

        subgraph PS["Pluggable Strategies"]
            AS[Action Strategy<br/>Market/Limit/Stop Orders<br/>Position Sizing]
            OS[Observation Strategy<br/>Portfolio State<br/>Market Conditions<br/>Risk Metrics]
            RS[Reward Strategy<br/>Conservative/Explorative<br/>Custom Composite]
        end

        AS -.-> TE
        OS -.-> TE
        RS -.-> TE
    end

    subgraph XL["🤖 Experiment Layer"]
        RL[RL Agents<br/>PPO, SAC, A2C<br/>Stable-Baselines3]
        HPT[Hyperparameter Tuning<br/>Optuna]
        EVAL[Evaluation & Analysis<br/>Backtesting<br/>Performance Metrics<br/>Benchmarking]

        RL --> HPT
        RL --> EVAL
    end

    subgraph UL["🛠️ Utilities"]
        FS[Feature Selection<br/>Indicator Optimization]
        VIS[Visualization<br/>Results Analysis]
        LOG[Logging & Monitoring]
    end

    PP ==>|Processed Data| TE
    TE ==>|State/Reward| RL
    RL ==>|Actions| TE

    FS -.->|Optimal Features| PP
    EVAL -.->|Insights| VIS
    RL -.->|Metrics| LOG

    style DL fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style EL fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style XL fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style UL fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style PS fill:#f1f8e9,stroke:#33691e,stroke-width:1px,stroke-dasharray: 5 5

    classDef dataNode fill:#bbdefb,stroke:#1976d2
    classDef envNode fill:#c8e6c9,stroke:#388e3c
    classDef expNode fill:#f8bbd0,stroke:#c2185b
    classDef utilNode fill:#ffe0b2,stroke:#f57c00

    class DS,UI,PP dataNode
    class TE,AS,OS,RS envNode
    class RL,HPT,EVAL expNode
    class FS,VIS,LOG utilNode
```

</details>

<details>
<summary><b>🔌 Strategy Pattern Implementation</b> - How pluggable strategies interact with the trading environment</summary>

```mermaid
graph TB
    subgraph Client["👤 Client Code"]
        CONFIG[Environment Configuration]
    end

    subgraph Core["🎯 Core Trading Environment"]
        ENV[TradingEnv<br/>Gymnasium Interface]

        subgraph State["Internal State"]
            PORTFOLIO[Portfolio Manager<br/>Balance, Holdings, Positions]
            MARKET[Market Data<br/>Price History, Indicators]
        end
    end

    subgraph Strategies["🔌 Pluggable Strategy Components"]
        direction TB

        AS[ActionStrategy<br/>━━━━━━━━━━━━<br/>process_action]
        OS[ObservationStrategy<br/>━━━━━━━━━━━━<br/>get_observation]
        RS[RewardStrategy<br/>━━━━━━━━━━━━<br/>calculate_reward]

        subgraph ASImpl["Action Implementations"]
            AS1[DiscreteActionStrategy<br/>Buy/Hold/Sell]
            AS2[ContinuousActionStrategy<br/>Position Sizing]
            AS3[MultiOrderStrategy<br/>Market/Limit/Stop]
        end

        subgraph OSImpl["Observation Implementations"]
            OS1[SimpleObservation<br/>Price + Balance]
            OS2[RichObservation<br/>Portfolio + Risk Metrics]
            OS3[CustomObservation<br/>User-defined Features]
        end

        subgraph RSImpl["Reward Implementations"]
            RS1[ConservativeReward<br/>Sharpe-based]
            RS2[ExplorativeReward<br/>Return-based]
            RS3[CompositeReward<br/>Multi-objective]
        end
    end

    subgraph Agent["🤖 RL Agent"]
        ALGO[Algorithm<br/>PPO/SAC/A2C]
    end

    CONFIG -->|1. Inject Strategies| ENV
    CONFIG -.->|Configure| AS
    CONFIG -.->|Configure| OS
    CONFIG -.->|Configure| RS

    AS1 -.->|implements| AS
    AS2 -.->|implements| AS
    AS3 -.->|implements| AS

    OS1 -.->|implements| OS
    OS2 -.->|implements| OS
    OS3 -.->|implements| OS

    RS1 -.->|implements| RS
    RS2 -.->|implements| RS
    RS3 -.->|implements| RS

    ALGO -->|2. action| ENV
    ENV -->|3. delegates to| AS
    AS -->|4. validated action| PORTFOLIO

    PORTFOLIO -.->|5. state change| ENV
    ENV -->|6. delegates to| OS
    OS -->|7. reads| PORTFOLIO
    OS -->|8. reads| MARKET
    OS -->|9. observation| ENV

    ENV -->|10. delegates to| RS
    RS -->|11. reads| PORTFOLIO
    RS -->|12. reward| ENV

    ENV -->|13. obs, reward, done, info| ALGO

    style ENV fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    style AS fill:#2196f3,stroke:#1565c0,stroke-width:2px,color:#fff
    style OS fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    style RS fill:#9c27b0,stroke:#6a1b9a,stroke-width:2px,color:#fff

    style PORTFOLIO fill:#c8e6c9,stroke:#388e3c
    style MARKET fill:#c8e6c9,stroke:#388e3c
    style CONFIG fill:#e1f5fe,stroke:#01579b
    style ALGO fill:#fce4ec,stroke:#c2185b

    style ASImpl fill:#bbdefb,stroke:#1976d2,stroke-dasharray: 5 5
    style OSImpl fill:#ffe0b2,stroke:#f57c00,stroke-dasharray: 5 5
    style RSImpl fill:#e1bee7,stroke:#7b1fa2,stroke-dasharray: 5 5

    classDef strategyInterface fill:#1976d2,stroke:#0d47a1,color:#fff
    classDef implementation fill:#fff,stroke:#666,stroke-dasharray: 3 3

    class AS,OS,RS strategyInterface
    class AS1,AS2,AS3,OS1,OS2,OS3,RS1,RS2,RS3 implementation
```

</details>

<details>
<summary><b>📊 Data Flow</b> - How data moves from sources through processing to the RL agent</summary>

```mermaid
flowchart TB
    subgraph Sources["📡 Raw Data Sources"]
        ALP[Alpaca API<br/>Stocks & Crypto]
        AV[Alpha Vantage<br/>Market Data]
        YF[Yahoo Finance<br/>Historical Prices]
        POL[Polygon.io<br/>Real-time Data]
    end

    subgraph Layer1["🔄 Data Acquisition Layer"]
        UI[Unified DataFetcher Interface<br/>━━━━━━━━━━━━━━━━━━━<br/>fetch_data<br/>validate_schema<br/>normalize_format]

        ALP --> UI
        AV --> UI
        YF --> UI
        POL --> UI

        CACHE[(Data Cache<br/>Historical Storage)]
        UI <--> CACHE
    end

    subgraph Layer2["⚙️ Processing Pipeline"]
        direction TB

        RAW[Raw OHLCV Data<br/>Open, High, Low<br/>Close, Volume]

        FS{Feature Selection<br/>Module?}

        PROC[Technical Indicator<br/>Computation Engine]

        subgraph Indicators["Technical Indicators"]
            direction LR
            IND1[Trend<br/>SMA, EMA<br/>MACD]
            IND2[Momentum<br/>RSI, Stochastic<br/>CCI]
            IND3[Volatility<br/>Bollinger Bands<br/>ATR]
            IND4[Volume<br/>OBV, MFI<br/>VWAP]
        end

        UI -->|normalized data| RAW
        RAW --> FS
        FS -->|optimal features| PROC
        FS -.->|skip| PROC

        PROC --> IND1
        PROC --> IND2
        PROC --> IND3
        PROC --> IND4
    end

    subgraph EnvData["🎯 Environment Data Structures"]
        direction TB

        MARKET[Market State<br/>━━━━━━━━━━<br/>Price History<br/>Technical Indicators<br/>Window Buffer]

        PORT[Portfolio State<br/>━━━━━━━━━━<br/>Cash Balance<br/>Holdings<br/>Positions<br/>Transaction History]

        IND1 --> MARKET
        IND2 --> MARKET
        IND3 --> MARKET
        IND4 --> MARKET
    end

    subgraph Env["🏪 Trading Environment"]
        direction TB

        OBS[Observation Strategy<br/>━━━━━━━━━━━━━━<br/>Reads: Market + Portfolio<br/>Returns: State Vector]

        STEP[Environment Step<br/>━━━━━━━━━━━━<br/>1. Process Action<br/>2. Update Portfolio<br/>3. Get Observation<br/>4. Calculate Reward]

        MARKET --> OBS
        PORT --> OBS

        OBS --> STEP
    end

    subgraph Agent["🤖 RL Agent"]
        direction TB

        POLICY[Policy Network<br/>━━━━━━━━━━<br/>Neural Network<br/>PPO/SAC/A2C]

        BUFFER[Experience Buffer<br/>━━━━━━━━━━━━<br/>Transitions<br/>s, a, r, s', done]

        LEARN[Learning Algorithm<br/>━━━━━━━━━━━━<br/>Gradient Updates<br/>Policy Optimization]
    end

    subgraph Output["📊 Analysis & Evaluation"]
        direction TB

        METRICS[Performance Metrics<br/>━━━━━━━━━━━━━<br/>Sharpe Ratio<br/>Total Return<br/>Max Drawdown<br/>Win Rate]

        VIZ[Visualization<br/>━━━━━━━━<br/>Equity Curve<br/>Action Distribution<br/>Portfolio Allocation]

        BENCH[Benchmark Comparison<br/>━━━━━━━━━━━━━━<br/>Buy & Hold<br/>Equal Weight<br/>Market Index]
    end

    STEP -->|observation| POLICY
    POLICY -->|action| STEP

    STEP -->|transition| BUFFER
    BUFFER --> LEARN
    LEARN -.->|updated weights| POLICY

    STEP -->|episode data| METRICS
    METRICS --> VIZ
    METRICS --> BENCH

    PORT -.->|state history| METRICS

    style Sources fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Layer1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Layer2 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Indicators fill:#ffe0b2,stroke:#f57c00,stroke-dasharray: 5 5
    style EnvData fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style Env fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Agent fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style Output fill:#f1f8e9,stroke:#558b2f,stroke-width:2px

    classDef dataSource fill:#90caf9,stroke:#1565c0
    classDef processor fill:#ffcc80,stroke:#e65100
    classDef storage fill:#ce93d8,stroke:#6a1b9a
    classDef component fill:#80deea,stroke:#01579b
    classDef agent fill:#f48fb1,stroke:#c2185b
    classDef output fill:#aed581,stroke:#558b2f

    class ALP,AV,YF,POL dataSource
    class UI,PROC,FS processor
    class CACHE,MARKET,PORT,BUFFER storage
    class OBS,STEP,POLICY,LEARN component
    class METRICS,VIZ,BENCH output
```

</details>

<details>
<summary><b>🧩 Pre-built Components Overview</b> - Out-of-the-box strategies and configurations</summary>

```mermaid
graph TB
    subgraph AS["🎯 Action Strategies"]
        AS1[StandardMarketActionStrategy<br/>━━━━━━━━━━━━━━━━━<br/>• 3D Continuous Action Space<br/>• Market Orders<br/>• Limit Orders<br/>• Stop-Loss & Take-Profit<br/>• Dynamic Position Sizing]

        AS2[Custom Action Strategies<br/>━━━━━━━━━━━━━━━━━<br/>Extend BaseActionStrategy<br/>to create your own]
    end

    subgraph OS["👁️ Observation Strategies"]
        OS1[PortfolioWithTrendObservation<br/>━━━━━━━━━━━━━━━━━━━━━━<br/>9-Feature Observation Space]

        subgraph OSF["Feature Breakdown"]
            F1[1. Balance Ratio<br/>Cash / Initial Balance]
            F2[2. Position Size Ratio<br/>Holdings Value / Portfolio]
            F3[3. Unrealized P/L %<br/>Position Gain/Loss]
            F4[4. Risk/Reward Ratio<br/>Potential Loss vs Gain]
            F5[5-6. Stop/Target Distance<br/>Price Distance to Limits]
            F6[7. Trend Strength<br/>Market Direction Signal]
            F7[8. Volatility<br/>Price Fluctuation Metric]
            F8[9. High/Low Context<br/>Recent Price Range]
        end

        OS1 --> OSF

        OS2[Custom Observation<br/>━━━━━━━━━━━━━<br/>Extend BaseObservationStrategy]
    end

    subgraph RS["🎁 Reward Strategies"]
        direction TB

        RS0[Individual Reward Components<br/>━━━━━━━━━━━━━━━━━━━━]

        RS1[1. PortfolioValueChangeReward<br/>Returns-based reward]
        RS2[2. InvalidActionPenalty<br/>Penalizes illegal actions]
        RS3[3. TrendFollowingReward<br/>Rewards trend alignment]
        RS4[4. HoldPenalty<br/>Discourages inaction]
        RS5[5. PositionSizingRiskReward<br/>Optimal position management]
        RS6[6. CashFlowRiskManagement<br/>Cash utilization optimization]
        RS7[7. ExcessiveCashUsagePenalty<br/>Prevents over-leveraging]

        RS0 --> RS1
        RS0 --> RS2
        RS0 --> RS3
        RS0 --> RS4
        RS0 --> RS5
        RS0 --> RS6
        RS0 --> RS7

        COMP[WeightedCompositeReward<br/>━━━━━━━━━━━━━━━━━<br/>Combines multiple rewards<br/>with custom weights]

        RS1 -.-> COMP
        RS2 -.-> COMP
        RS3 -.-> COMP
        RS4 -.-> COMP
        RS5 -.-> COMP
        RS6 -.-> COMP
        RS7 -.-> COMP

        subgraph PRESETS["Preset Combinations"]
            P1[Conservative<br/>High penalty weighting]
            P2[Balanced<br/>Equal distribution]
            P3[Aggressive<br/>High trend following]
            P4[Risk Managed<br/>Focus on risk metrics]
        end

        COMP --> PRESETS
    end

    style AS fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style OS fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style RS fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style OSF fill:#ffe0b2,stroke:#f57c00,stroke-dasharray: 3 3
    style PRESETS fill:#e1bee7,stroke:#7b1fa2,stroke-dasharray: 3 3

    style AS1 fill:#bbdefb,stroke:#1976d2
    style OS1 fill:#ffcc80,stroke:#e65100
    style COMP fill:#ce93d8,stroke:#6a1b9a

    style AS2 fill:#fff,stroke:#666,stroke-dasharray: 5 5
    style OS2 fill:#fff,stroke:#666,stroke-dasharray: 5 5

    classDef feature fill:#fff9c4,stroke:#f57f17
    classDef preset fill:#f8bbd0,stroke:#c2185b

    class F1,F2,F3,F4,F5,F6,F7,F8 feature
    class P1,P2,P3,P4 preset
```

</details>

<details>
<summary><b>🔧 Extensibility & Customization</b> - How to extend the framework with custom strategies</summary>

```mermaid
classDiagram
    class BaseActionStrategy {
        <<abstract>>
        +define_action_space()*
        +process_action(action)*
        +get_action_space_info()*
    }

    class BaseObservationStrategy {
        <<abstract>>
        +define_observation_space()*
        +build_observation()*
    }

    class BaseRewardStrategy {
        <<abstract>>
        +calculate_reward()*
        +reset()
    }

    class TradingEnv {
        +action_strategy
        +observation_strategy
        +reward_strategy
        +step()
        +reset()
    }

    %% Pre-built Implementations
    class StandardMarketActionStrategy {
        +3D continuous action space
        +Market/Limit/Stop orders
        +Position sizing
    }

    class PortfolioWithTrendObservation {
        +9-feature observation
        +Trend & volatility metrics
        +Risk/reward calculations
    }

    class WeightedCompositeReward {
        +Combines multiple rewards
        +Configurable weights
    }

    %% Custom User Implementations
    class CustomActionStrategy {
        +Your custom logic
        +define_action_space()
        +process_action()
    }

    class CustomObservationStrategy {
        +Your custom features
        +define_observation_space()
        +build_observation()
    }

    class CustomRewardStrategy {
        +Your custom objectives
        +calculate_reward()
    }

    %% Inheritance relationships
    BaseActionStrategy <|-- StandardMarketActionStrategy : implements
    BaseActionStrategy <|-- CustomActionStrategy : extend

    BaseObservationStrategy <|-- PortfolioWithTrendObservation : implements
    BaseObservationStrategy <|-- CustomObservationStrategy : extend

    BaseRewardStrategy <|-- WeightedCompositeReward : implements
    BaseRewardStrategy <|-- CustomRewardStrategy : extend

    %% Composition relationships
    TradingEnv o-- BaseActionStrategy : uses
    TradingEnv o-- BaseObservationStrategy : uses
    TradingEnv o-- BaseRewardStrategy : uses

    %% Notes
    note for TradingEnv "Strategies are injected\nvia dependency injection.\n\nMix and match:\n• Pre-built components\n• Custom implementations\n• Hybrid approaches"

    note for CustomActionStrategy "Extend base classes\nto create your own:\n\n1. Inherit from base\n2. Implement abstract methods\n3. Add custom logic\n4. Inject into environment"

    %% Styling
    style BaseActionStrategy fill:#2196f3,stroke:#1565c0,color:#fff
    style BaseObservationStrategy fill:#ff9800,stroke:#e65100,color:#fff
    style BaseRewardStrategy fill:#9c27b0,stroke:#6a1b9a,color:#fff

    style StandardMarketActionStrategy fill:#bbdefb,stroke:#1976d2
    style PortfolioWithTrendObservation fill:#ffe0b2,stroke:#f57c00
    style WeightedCompositeReward fill:#e1bee7,stroke:#7b1fa2

    style CustomActionStrategy fill:#c8e6c9,stroke:#388e3c
    style CustomObservationStrategy fill:#c8e6c9,stroke:#388e3c
    style CustomRewardStrategy fill:#c8e6c9,stroke:#388e3c

    style TradingEnv fill:#4caf50,stroke:#2e7d32,color:#fff
```

</details>

<details>
<summary><b>🔍 Protocol Pattern in Action</b> - Structural typing for flexible, decoupled design</summary>

```mermaid
graph TB
    subgraph Concept["💡 Protocol Concept"]
        direction TB
        PROTO_DEF["Protocol defines interface<br/>━━━━━━━━━━━━━━━━<br/>• No inheritance required<br/>• Duck typing with type checking<br/>• Structural subtyping"]

        PROTO_ADV["Advantages<br/>━━━━━━━━━━<br/>✓ Loose coupling<br/>✓ Multiple capabilities<br/>✓ Runtime checkable<br/>✓ No diamond problem"]
    end

    subgraph DataProtocols["📡 Data Source Protocols"]
        direction TB

        BASE[DataSource<br/>━━━━━━━━━━<br/>Abstract Base Class<br/>Common interface]

        P1[HistoricalDataCapable<br/>━━━━━━━━━━━━━━━━<br/>get_historical_ohlcv_data]
        P2[LiveDataCapable<br/>━━━━━━━━━━━━━━<br/>get_latest_quote<br/>get_latest_trade]
        P3[NewsDataCapable<br/>━━━━━━━━━━━━━<br/>get_news_data]
        P4[StreamingCapable<br/>━━━━━━━━━━━━━<br/>subscribe, start/stop_streaming]
        P5[ConnectionManaged<br/>━━━━━━━━━━━━━━<br/>connect, disconnect, is_connected]
        P6[FundamentalDataCapable<br/>━━━━━━━━━━━━━━━━━<br/>get_fundamental_data]
        P7[MacroDataCapable<br/>━━━━━━━━━━━━━<br/>get_macro_data]
    end

    subgraph EnvProtocol["🏪 Environment Protocol"]
        direction TB

        EP[TradingEnvProtocol<br/>━━━━━━━━━━━━━━━<br/>Defines required attributes<br/>& methods for trading envs]

        EP_ATTRS["Required Attributes:<br/>• data: np.ndarray<br/>• current_step: int<br/>• window_size: int<br/>• action_space<br/>• observation_space"]

        EP_METHODS["Required Methods:<br/>• step<br/>• reset<br/>• render<br/>• close"]

        EP --> EP_ATTRS
        EP --> EP_METHODS
    end

    subgraph Implementations["🔧 Concrete Implementations"]
        direction TB

        YF[YfinanceDataloader<br/>━━━━━━━━━━━━━━<br/>Inherits: DataSource<br/>Implements: HistoricalDataCapable<br/>           FundamentalDataCapable]

        ALP[AlpacaDataLoader<br/>━━━━━━━━━━━━━<br/>Inherits: DataSource<br/>Implements: HistoricalDataCapable<br/>           LiveDataCapable<br/>           StreamingCapable<br/>           NewsDataCapable<br/>           ConnectionManaged]

        TE[TradingEnv<br/>━━━━━━━━━<br/>Implements: TradingEnvProtocol<br/>Has all required attributes<br/>& methods]
    end

    subgraph Usage["💼 Usage Pattern"]
        direction TB

        CHECK["Runtime Check<br/>━━━━━━━━━━━━<br/>isinstance(obj, Protocol)<br/>Checks structural compatibility"]

        FEATURE["Feature Detection<br/>━━━━━━━━━━━━━━<br/>if isinstance(source, LiveDataCapable):<br/>    # Use live data features<br/>else:<br/>    # Fall back to historical"]

        COMPOSE["Compose Capabilities<br/>━━━━━━━━━━━━━━━━<br/>Class can implement<br/>multiple protocols<br/>to gain multiple capabilities"]
    end

    subgraph Benefits["✨ Benefits in QuantRL-Lab"]
        direction TB

        B1["Flexibility<br/>━━━━━━━━━<br/>Data sources can<br/>implement any combination<br/>of capabilities"]

        B2["Type Safety<br/>━━━━━━━━━<br/>Static type checkers<br/>validate protocol<br/>compliance"]

        B3["Decoupling<br/>━━━━━━━━━<br/>Code depends on<br/>protocols, not<br/>concrete classes"]

        B4["Discoverability<br/>━━━━━━━━━━━<br/>supported_features()<br/>checks which protocols<br/>are implemented"]
    end

    PROTO_DEF -.-> P1
    PROTO_DEF -.-> P2
    PROTO_DEF -.-> EP

    BASE --> YF
    BASE --> ALP

    P1 -.->|structural typing| YF
    P6 -.->|structural typing| YF

    P1 -.->|structural typing| ALP
    P2 -.->|structural typing| ALP
    P3 -.->|structural typing| ALP
    P4 -.->|structural typing| ALP
    P5 -.->|structural typing| ALP

    EP -.->|structural typing| TE

    YF --> CHECK
    ALP --> CHECK
    CHECK --> FEATURE
    FEATURE --> COMPOSE

    COMPOSE --> B1
    COMPOSE --> B2
    COMPOSE --> B3
    COMPOSE --> B4

    style PROTO_DEF fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style PROTO_ADV fill:#b3e5fc,stroke:#0277bd

    style BASE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style P1 fill:#ffe0b2,stroke:#f57c00
    style P2 fill:#ffe0b2,stroke:#f57c00
    style P3 fill:#ffe0b2,stroke:#f57c00
    style P4 fill:#ffe0b2,stroke:#f57c00
    style P5 fill:#ffe0b2,stroke:#f57c00
    style P6 fill:#ffe0b2,stroke:#f57c00
    style P7 fill:#ffe0b2,stroke:#f57c00

    style EP fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style EP_ATTRS fill:#e1bee7,stroke:#7b1fa2
    style EP_METHODS fill:#e1bee7,stroke:#7b1fa2

    style YF fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style ALP fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style TE fill:#c8e6c9,stroke:#388e3c,stroke-width:2px

    style CHECK fill:#fff9c4,stroke:#f57f17
    style FEATURE fill:#fff9c4,stroke:#f57f17
    style COMPOSE fill:#fff9c4,stroke:#f57f17

    style B1 fill:#b2dfdb,stroke:#00695c
    style B2 fill:#b2dfdb,stroke:#00695c
    style B3 fill:#b2dfdb,stroke:#00695c
    style B4 fill:#b2dfdb,stroke:#00695c

    classDef protocol fill:#ffccbc,stroke:#d84315
    class P1,P2,P3,P4,P5,P6,P7,EP protocol
```

**How Protocols Work in QuantRL-Lab:**

1. **Protocol Definition**: Instead of forcing inheritance, protocols define what methods/attributes a class must have
2. **Structural Typing**: A class automatically satisfies a protocol if it has the required methods/attributes
3. **Multiple Capabilities**: Data sources can implement multiple protocols (e.g., Alpaca implements 5 protocols)
4. **Runtime Checking**: Use `isinstance(obj, Protocol)` to check if an object supports certain capabilities
5. **Feature Discovery**: The `supported_features` property checks which protocols are implemented
6. **Type Safety**: Static type checkers (mypy, pyright) validate protocol compliance at development time

**Example:**
```python
# Any class with these methods satisfies HistoricalDataCapable
class CustomDataSource:
    def get_historical_ohlcv_data(self, symbols, start, end, timeframe):
        # Implementation
        pass

# No inheritance needed! This works:
if isinstance(custom_source, HistoricalDataCapable):
    data = custom_source.get_historical_ohlcv_data(...)
```

</details>

<details>
<summary><b>📋 Registry Pattern for Technical Indicators</b> - Centralized, extensible indicator management</summary>

```mermaid
graph TB
    subgraph Pattern["🎯 Registry Pattern Concept"]
        direction TB

        REG_IDEA["Centralized Registration<br/>━━━━━━━━━━━━━━━━━<br/>• Single source of truth<br/>• Dynamic discovery<br/>• Decoupled architecture<br/>• Plugin-like extensibility"]

        REG_FLOW["Registration Flow<br/>━━━━━━━━━━━━━━<br/>1. Decorator marks function<br/>2. Function added to registry<br/>3. Available for lookup<br/>4. Can be applied dynamically"]
    end

    subgraph Registry["📚 IndicatorRegistry Class"]
        direction TB

        REG_STORE["_indicators: Dict[str, Callable]<br/>━━━━━━━━━━━━━━━━━━━━━━<br/>Central storage for all indicators"]

        REG_METHODS["Core Methods<br/>━━━━━━━━━━━"]

        M1["@register(name)<br/>Decorator to add indicators"]
        M2["get(name)<br/>Retrieve indicator function"]
        M3["list_all()<br/>Show all registered indicators"]
        M4["apply(name, df, **kwargs)<br/>Execute indicator on data"]

        REG_METHODS --> M1
        REG_METHODS --> M2
        REG_METHODS --> M3
        REG_METHODS --> M4
    end

    subgraph Indicators["📊 Registered Technical Indicators"]
        direction TB

        I1["@register('SMA')<br/>Simple Moving Average"]
        I2["@register('EMA')<br/>Exponential Moving Average"]
        I3["@register('RSI')<br/>Relative Strength Index"]
        I4["@register('MACD')<br/>Moving Average Convergence"]
        I5["@register('ATR')<br/>Average True Range"]
        I6["@register('BB')<br/>Bollinger Bands"]
        I7["@register('STOCH')<br/>Stochastic Oscillator"]
        I8["@register('OBV')<br/>On-Balance Volume"]
        I9["... more indicators<br/>Easily extensible"]
    end

    subgraph Usage["💼 Usage in DataProcessor"]
        direction TB

        DP[DataProcessor<br/>━━━━━━━━━━━<br/>Processes OHLCV data<br/>with technical indicators]

        STEP1["1. Check Available<br/>━━━━━━━━━━━━━━<br/>IndicatorRegistry.list_all()<br/>Returns: ['SMA', 'EMA', 'RSI', ...]"]

        STEP2["2. Apply Indicator<br/>━━━━━━━━━━━━━━<br/>IndicatorRegistry.apply('RSI', df, window=14)<br/>Returns: DataFrame with RSI_14 column"]

        STEP3["3. Batch Processing<br/>━━━━━━━━━━━━━━━<br/>Loop through multiple indicators<br/>Apply with different parameters"]

        DP --> STEP1
        STEP1 --> STEP2
        STEP2 --> STEP3
    end

    subgraph Extension["🔧 Adding Custom Indicators"]
        direction TB

        CUSTOM["How to Extend<br/>━━━━━━━━━━━━"]

        CODE1["1. Define Function<br/>━━━━━━━━━━━━━━<br/>def my_indicator(df, **params):<br/>    # Your logic<br/>    return df"]

        CODE2["2. Register with Decorator<br/>━━━━━━━━━━━━━━━━━━<br/>@IndicatorRegistry.register('MyIndicator')<br/>def my_indicator(df, **params):"]

        CODE3["3. Use Immediately<br/>━━━━━━━━━━━━━━━<br/>IndicatorRegistry.apply('MyIndicator', df)<br/>No modification to core code!"]

        CUSTOM --> CODE1
        CODE1 --> CODE2
        CODE2 --> CODE3
    end

    subgraph Benefits["✨ Benefits"]
        direction TB

        B1["No Hard-Coding<br/>━━━━━━━━━━━━<br/>Indicators are<br/>discovered dynamically"]

        B2["Easy Testing<br/>━━━━━━━━━━<br/>Swap indicators<br/>without code changes"]

        B3["Plugin Architecture<br/>━━━━━━━━━━━━━━<br/>Add new indicators<br/>without modifying registry"]

        B4["Consistent Interface<br/>━━━━━━━━━━━━━━━<br/>All indicators follow<br/>same pattern"]

        B5["Feature Selection<br/>━━━━━━━━━━━━━<br/>Programmatically test<br/>different combinations"]
    end

    REG_IDEA --> REG_STORE
    REG_STORE --> REG_METHODS

    M1 -.->|registers| I1
    M1 -.->|registers| I2
    M1 -.->|registers| I3
    M1 -.->|registers| I4
    M1 -.->|registers| I5
    M1 -.->|registers| I6
    M1 -.->|registers| I7
    M1 -.->|registers| I8
    M1 -.->|registers| I9

    I1 --> REG_STORE
    I2 --> REG_STORE
    I3 --> REG_STORE
    I4 --> REG_STORE
    I5 --> REG_STORE
    I6 --> REG_STORE
    I7 --> REG_STORE
    I8 --> REG_STORE
    I9 --> REG_STORE

    REG_METHODS --> STEP1
    REG_STORE --> STEP2

    CODE2 -.->|extends| M1
    CODE3 --> STEP2

    STEP3 --> B1
    STEP3 --> B2
    STEP3 --> B3
    STEP3 --> B4
    STEP3 --> B5

    style REG_IDEA fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style REG_FLOW fill:#b3e5fc,stroke:#0277bd

    style REG_STORE fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style REG_METHODS fill:#ffe0b2,stroke:#f57c00,stroke-width:2px

    style M1 fill:#ffccbc,stroke:#d84315
    style M2 fill:#ffccbc,stroke:#d84315
    style M3 fill:#ffccbc,stroke:#d84315
    style M4 fill:#ffccbc,stroke:#d84315

    style I1 fill:#c8e6c9,stroke:#388e3c
    style I2 fill:#c8e6c9,stroke:#388e3c
    style I3 fill:#c8e6c9,stroke:#388e3c
    style I4 fill:#c8e6c9,stroke:#388e3c
    style I5 fill:#c8e6c9,stroke:#388e3c
    style I6 fill:#c8e6c9,stroke:#388e3c
    style I7 fill:#c8e6c9,stroke:#388e3c
    style I8 fill:#c8e6c9,stroke:#388e3c
    style I9 fill:#c8e6c9,stroke:#388e3c,stroke-dasharray: 5 5

    style DP fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style STEP1 fill:#e1bee7,stroke:#7b1fa2
    style STEP2 fill:#e1bee7,stroke:#7b1fa2
    style STEP3 fill:#e1bee7,stroke:#7b1fa2

    style CUSTOM fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style CODE1 fill:#fff59d,stroke:#f9a825
    style CODE2 fill:#fff59d,stroke:#f9a825
    style CODE3 fill:#fff59d,stroke:#f9a825

    style B1 fill:#b2dfdb,stroke:#00695c
    style B2 fill:#b2dfdb,stroke:#00695c
    style B3 fill:#b2dfdb,stroke:#00695c
    style B4 fill:#b2dfdb,stroke:#00695c
    style B5 fill:#b2dfdb,stroke:#00695c
```

**How the Registry Pattern Works:**

1. **Registration Phase**: Indicators are decorated with `@IndicatorRegistry.register(name)` which adds them to the central registry dictionary

2. **Discovery Phase**: Use `list_all()` to see all available indicators without hardcoding names

3. **Application Phase**: Call `apply(name, df, **kwargs)` to execute any registered indicator dynamically

4. **Extension Phase**: Add new indicators by simply decorating functions - no need to modify the registry class

**Example Usage:**
```python
from quantrl_lab.data.indicators import IndicatorRegistry

# See what's available
print(IndicatorRegistry.list_all())
# Output: ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BB', 'STOCH', 'OBV']

# Apply indicator dynamically
df_with_rsi = IndicatorRegistry.apply('RSI', df, window=14)
df_with_sma = IndicatorRegistry.apply('SMA', df, window=20, column='Close')

# Add your own indicator
@IndicatorRegistry.register('CustomIndicator')
def custom_indicator(df, param1, param2):
    # Your calculation
    return df

# Use it immediately
df = IndicatorRegistry.apply('CustomIndicator', df, param1=10, param2=20)
```

**Key Advantage**: The DataProcessor can loop through indicators programmatically, making it trivial to test hundreds of indicator combinations for feature selection without code changes!

</details>

<details>
<summary><b>⚡ Reward Strategy Pattern</b> - How reward strategies decouple from environment instantiation</summary>

```mermaid
graph LR
    subgraph Creation["1️⃣ Create Reward Strategy"]
        direction TB

        BASE["BaseRewardStrategy<br/>━━━━━━━━━━━━━━<br/>Abstract Interface"]

        R1["PortfolioValueChangeReward<br/>calculate_reward(env)"]
        R2["TrendFollowingReward<br/>calculate_reward(env)"]
        R3["InvalidActionPenalty<br/>calculate_reward(env)"]

        COMPOSITE["WeightedCompositeReward<br/>━━━━━━━━━━━━━━━━━━<br/>Combines multiple rewards"]

        BASE -.-> R1
        BASE -.-> R2
        BASE -.-> R3
        BASE -.-> COMPOSITE

        R1 -.-> COMPOSITE
        R2 -.-> COMPOSITE
        R3 -.-> COMPOSITE
    end

    subgraph Injection["2️⃣ Inject into Environment"]
        direction TB

        STRAT["reward_strategy = <br/>WeightedCompositeReward([<br/>  PortfolioValueChangeReward(),<br/>  TrendFollowingReward()<br/>], weights=[0.7, 0.3])"]

        ENV["env = SingleStockTradingEnv(<br/>  data=df,<br/>  config=config,<br/>  reward_strategy=reward_strategy<br/>)"]

        STORE["Environment stores reference:<br/>self.reward_strategy = reward_strategy"]

        STRAT --> ENV
        ENV --> STORE
    end

    subgraph Usage["3️⃣ Environment Delegates at Runtime"]
        direction TB

        STEP["env.step(action)"]

        EXECUTE["Inside step():<br/>━━━━━━━━━━━━<br/>1. Process action<br/>2. Update portfolio<br/>3. Call: reward = <br/>   self.reward_strategy<br/>     .calculate_reward(self)<br/>4. Return observation, reward"]

        CALC["Reward Strategy Executes:<br/>━━━━━━━━━━━━━━━━━━<br/>• Accesses env.portfolio<br/>• Reads current state<br/>• Computes reward<br/>• Returns scalar value"]

        STEP --> EXECUTE
        EXECUTE --> CALC
    end

    subgraph Benefits["✨ Benefits"]
        direction TB

        B1["🔄 Easy Swapping<br/>Change reward logic<br/>without touching env"]

        B2["🧪 A/B Testing<br/>Test multiple reward<br/>formulations"]

        B3["🎯 Single Responsibility<br/>Reward logic isolated<br/>in strategy class"]
    end

    COMPOSITE --> STRAT
    STORE --> STEP
    CALC --> B1
    CALC --> B2
    CALC --> B3

    style Creation fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style BASE fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style R1 fill:#ce93d8,stroke:#8e24aa
    style R2 fill:#ce93d8,stroke:#8e24aa
    style R3 fill:#ce93d8,stroke:#8e24aa
    style COMPOSITE fill:#ba68c8,stroke:#7b1fa2,stroke-width:2px

    style Injection fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style STRAT fill:#81d4fa,stroke:#0288d1
    style ENV fill:#4fc3f7,stroke:#039be5,stroke-width:2px
    style STORE fill:#29b6f6,stroke:#0277bd

    style Usage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style STEP fill:#a5d6a7,stroke:#43a047
    style EXECUTE fill:#81c784,stroke:#388e3c
    style CALC fill:#66bb6a,stroke:#2e7d32,stroke-width:2px

    style Benefits fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style B1 fill:#ffe0b2,stroke:#f57c00
    style B2 fill:#ffe0b2,stroke:#f57c00
    style B3 fill:#ffe0b2,stroke:#f57c00
```

**How It Works:**

```python
# 1️⃣ Create reward strategy (outside environment)
reward_strategy = WeightedCompositeReward([
    PortfolioValueChangeReward(),
    TrendFollowingReward(),
], weights=[0.7, 0.3])

# 2️⃣ Inject into environment
env = SingleStockTradingEnv(
    data=df,
    config=config,
    reward_strategy=reward_strategy  # ← Injected here!
)

# 3️⃣ Environment delegates during step
obs, reward, done, truncated, info = env.step(action)
# Inside step():
#   reward = self.reward_strategy.calculate_reward(self)
```

**Key Insight**: The environment doesn't know *how* rewards are calculated. It just calls `calculate_reward()` and the strategy does the rest. Want different rewards? Just inject a different strategy!

</details>

---


### Example usage:

```python
# Easily swappable strategies for experimentation
# For in depth example, please refer to the backtesting_example.ipynb

sample_env_config = BacktestRunner.create_env_config_factory(
    train_data=train_data_df,
    test_data=test_data_df,
    action_strategy=action_strategy,
    reward_strategy=reward_strategies["conservative"],
    observation_strategy=observation_strategy,
    initial_balance=100000.0,
    transaction_cost_pct=0.001,
    window_size=20
)

runner = BacktestRunner(verbose=1)

# Single experiment
results = runner.run_single_experiment(
    SAC,          # Algorithm to use
    sample_env_config,
    # config=custom_sac_config,  # an optional input arg
    total_timesteps=50000,  # Total timesteps for training
    num_eval_episodes=3
)

BacktestRunner.inspect_single_experiment(results)

# More combinations
presets = ["default", "explorative", "conservative"]

algorithms = [PPO, A2C, SAC]

comprehensive_results = runner.run_comprehensive_backtest(
    algorithms=algorithms,
    env_configs=env_configs,
    presets=presets,
    # custom_configs=custom_configs,  # either use presets or customize config by yourself
    total_timesteps=50000,
    n_envs=4,
    num_eval_episodes=3
)
```

For more detailed use cases, please refer to the notebooks:
- Feature and window size selection: [`notebooks/feature_selection.ipynb`](notebooks/feature_selection.ipynb)
- Data processing example: [`notebooks/data_processing.ipynb`](notebooks/data_processing.ipynb)
- Backtesting: [`notebooks/backtesting_example.ipynb`](notebooks/backtesting_example.ipynb)
- Hyperparameter tuning for stablebaseline algo: [`notebooks/hyperparameter_tuning.ipynb`](notebooks/hyperparameter_tuning.ipynb)
- LLM hedge pair screener (for upcoming multi stock env): [`notebooks/llm_hedge_screener.ipynb`](notebooks/llm_hedge_screener.ipynb)



---

### Roadmap 🔄
- **Data Source Expansion**:
  - Complete Integration for more (free) data sources
  - Add Cryto data support
  - Add OANDA forex data support
- **Technical Indicators**:
  - Add more indicators (Ichimoku, Williams %R, CCI, etc.)
- **Trading Environments**:
  - (In-progress) Multi-stock trading environment with hedging pair capabilities
- **Alternative Data for consideration in observable space**:
  - Fundamental data (earnings, balance sheets, income statements, cash flow)
  - Macroeconomic indicators (GDP, inflation, unemployment, interest rates)
  - Economic calendar events
  - Sector performance data

---

### Development Setup

> **Note:** This section is for developers who want to contribute to QuantRL-Lab or run it from source. If you just want to use the library, simply install it with `pip install quantrl-lab`.

#### For Contributors and Developers

1. Clone the Repository
```bash
git clone https://github.com/whanyu1212/QuantRL-Lab.git
cd QuantRL-Lab
```

2. Install Poetry for dependency management
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies (installs the project in development mode)
```bash
poetry install
```

4. Activate virtual environment
```bash
# The shell command is deprecated, use this instead:
poetry env info  # This shows the venv path
# Then activate it manually, e.g.:
source /path/to/virtualenv/bin/activate  # macOS/Linux
```

5. (Optional) Install jupyter kernel for notebook examples
```bash
python -m ipykernel install --user --name quantrl-lab --display-name "QuantRL-Lab"
```

6. Set up environment variables (for data sources like Alpaca, Alpha Vantage, etc.)
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Never commit the .env file to version control
```

7. Set up pre-commit hooks (for code quality)
```bash
# Install the git hooks
pre-commit install

# (Optional) Run on all files to test
pre-commit run --all-files
```

The pre-commit hooks will automatically check:
- Code formatting (black)
- Import sorting (isort)
- Code linting (flake8)
- Docstring formatting (docformatter)
- File checks (trailing whitespace, YAML validation, etc.)

To skip pre-commit hooks temporarily:
```bash
git commit -m "your message" --no-verify
```

---

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or suggesting ideas, your help is appreciated.

### How to Contribute

1. **Fork the repository** and create a branch for your feature or fix
2. **Make your changes** following our coding standards
3. **Write tests** for new functionality
4. **Submit a pull request** with a clear description

Please read our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on:
- Setting up your development environment
- Coding standards and style guidelines
- Testing requirements
- Pull request process

### Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build something great together!

---

## Contributors

This project exists thanks to all the people who contribute.

### Main Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/whanyu1212">
        <img src="https://github.com/whanyu1212.png" width="100px;" alt="whanyu1212"/>
        <br />
        <sub><b>whanyu1212</b></sub>
      </a>
      <br />
      <sub>Creator & Maintainer</sub>
    </td>
  </tr>
</table>

### How to Become a Contributor

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) to get started.

---

### Literature Review
