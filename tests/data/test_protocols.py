"""
Comprehensive protocol conformance tests for data sources.

Tests verify that data sources correctly implement the protocol
interfaces they claim to support, and don't claim protocols they don't
implement.
"""

from unittest.mock import patch

from quantrl_lab.data.interface import (
    AnalystDataCapable,
    CompanyProfileCapable,
    ConnectionManaged,
    FundamentalDataCapable,
    HistoricalDataCapable,
    LiveDataCapable,
    MacroDataCapable,
    NewsDataCapable,
    SectorDataCapable,
    StreamingCapable,
)
from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader
from quantrl_lab.data.sources.alpha_vantage_loader import AlphaVantageDataLoader
from quantrl_lab.data.sources.fmp_loader import FMPDataSource
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader


class TestYfinanceProtocols:
    """Protocol conformance tests for YFinanceDataLoader."""

    def test_implements_historical_data_protocol(self):
        """YFinanceDataLoader should implement HistoricalDataCapable."""
        loader = YFinanceDataLoader()
        assert isinstance(loader, HistoricalDataCapable)
        assert hasattr(loader, "get_historical_ohlcv_data")

    def test_implements_fundamental_data_protocol(self):
        """YFinanceDataLoader should implement
        FundamentalDataCapable."""
        loader = YFinanceDataLoader()
        assert isinstance(loader, FundamentalDataCapable)
        assert hasattr(loader, "get_fundamental_data")

    def test_does_not_implement_live_data_protocol(self):
        """YFinanceDataLoader should NOT implement LiveDataCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, LiveDataCapable)

    def test_does_not_implement_news_protocol(self):
        """YFinanceDataLoader should NOT implement NewsDataCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, NewsDataCapable)

    def test_does_not_implement_streaming_protocol(self):
        """YFinanceDataLoader should NOT implement StreamingCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, StreamingCapable)

    def test_does_not_implement_macro_data_protocol(self):
        """YFinanceDataLoader should NOT implement MacroDataCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, MacroDataCapable)

    def test_does_not_implement_analyst_data_protocol(self):
        """YFinanceDataLoader should NOT implement
        AnalystDataCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, AnalystDataCapable)

    def test_does_not_implement_sector_data_protocol(self):
        """YFinanceDataLoader should NOT implement SectorDataCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, SectorDataCapable)

    def test_does_not_implement_company_profile_protocol(self):
        """YFinanceDataLoader should NOT implement
        CompanyProfileCapable."""
        loader = YFinanceDataLoader()
        assert not isinstance(loader, CompanyProfileCapable)

    def test_supported_features_accuracy(self):
        """supported_features property should reflect actual
        capabilities."""
        loader = YFinanceDataLoader()
        features = loader.supported_features

        assert "historical_bars" in features
        assert "fundamental_data" in features
        assert "connection_managed" not in features
        assert "instrument_discovery" not in features
        assert "live_data" not in features
        assert "news" not in features
        assert "streaming" not in features
        assert "macro_data" not in features
        assert "analyst_data" not in features
        assert "sector_data" not in features
        assert "company_profile" not in features


class TestAlpacaProtocols:
    """Protocol conformance tests for AlpacaDataLoader."""

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_historical_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should implement HistoricalDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert isinstance(loader, HistoricalDataCapable)
        assert hasattr(loader, "get_historical_ohlcv_data")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_live_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should implement LiveDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert isinstance(loader, LiveDataCapable)
        assert hasattr(loader, "get_latest_quote")
        assert hasattr(loader, "get_latest_trade")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_news_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should implement NewsDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert isinstance(loader, NewsDataCapable)
        assert hasattr(loader, "get_news_data")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_streaming_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should implement StreamingCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert isinstance(loader, StreamingCapable)
        assert hasattr(loader, "subscribe_to_updates")
        assert hasattr(loader, "start_streaming")
        assert hasattr(loader, "stop_streaming")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_connection_managed_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should implement ConnectionManaged."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert isinstance(loader, ConnectionManaged)
        assert hasattr(loader, "connect")
        assert hasattr(loader, "disconnect")
        assert hasattr(loader, "is_connected")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_does_not_implement_fundamental_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should NOT implement
        FundamentalDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert not isinstance(loader, FundamentalDataCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_does_not_implement_macro_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should NOT implement MacroDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert not isinstance(loader, MacroDataCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_does_not_implement_analyst_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should NOT implement AnalystDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert not isinstance(loader, AnalystDataCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_does_not_implement_sector_data_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should NOT implement SectorDataCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert not isinstance(loader, SectorDataCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_does_not_implement_company_profile_protocol(self, mock_stream, mock_client):
        """AlpacaDataLoader should NOT implement
        CompanyProfileCapable."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        assert not isinstance(loader, CompanyProfileCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_supported_features_accuracy(self, mock_stream, mock_client):
        """supported_features property should reflect actual
        capabilities."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        features = loader.supported_features

        assert "historical_bars" in features
        assert "live_data" in features
        assert "news" in features
        assert "streaming" in features
        assert "connection_managed" in features
        assert "fundamental_data" not in features
        assert "macro_data" not in features
        assert "analyst_data" not in features
        assert "sector_data" not in features
        assert "company_profile" not in features


class TestAlphaVantageProtocols:
    """Protocol conformance tests for AlphaVantageDataLoader."""

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_implements_historical_data_protocol(self):
        """AlphaVantageDataLoader should implement
        HistoricalDataCapable."""
        loader = AlphaVantageDataLoader()
        assert isinstance(loader, HistoricalDataCapable)
        assert hasattr(loader, "get_historical_ohlcv_data")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_implements_fundamental_data_protocol(self):
        """AlphaVantageDataLoader should implement
        FundamentalDataCapable."""
        loader = AlphaVantageDataLoader()
        assert isinstance(loader, FundamentalDataCapable)
        assert hasattr(loader, "get_fundamental_data")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_implements_macro_data_protocol(self):
        """AlphaVantageDataLoader should implement MacroDataCapable."""
        loader = AlphaVantageDataLoader()
        assert isinstance(loader, MacroDataCapable)
        assert hasattr(loader, "get_macro_data")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_implements_news_protocol(self):
        """AlphaVantageDataLoader should implement NewsDataCapable."""
        loader = AlphaVantageDataLoader()
        assert isinstance(loader, NewsDataCapable)
        assert hasattr(loader, "get_news_data")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_does_not_implement_live_data_protocol(self):
        """AlphaVantageDataLoader should NOT implement
        LiveDataCapable."""
        loader = AlphaVantageDataLoader()
        assert not isinstance(loader, LiveDataCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_does_not_implement_streaming_protocol(self):
        """AlphaVantageDataLoader should NOT implement
        StreamingCapable."""
        loader = AlphaVantageDataLoader()
        assert not isinstance(loader, StreamingCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_does_not_implement_analyst_data_protocol(self):
        """AlphaVantageDataLoader should NOT implement
        AnalystDataCapable."""
        loader = AlphaVantageDataLoader()
        assert not isinstance(loader, AnalystDataCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_does_not_implement_sector_data_protocol(self):
        """AlphaVantageDataLoader should NOT implement
        SectorDataCapable."""
        loader = AlphaVantageDataLoader()
        assert not isinstance(loader, SectorDataCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_does_not_implement_company_profile_protocol(self):
        """AlphaVantageDataLoader should NOT implement
        CompanyProfileCapable."""
        loader = AlphaVantageDataLoader()
        assert not isinstance(loader, CompanyProfileCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_supported_features_accuracy(self):
        """supported_features property should reflect actual
        capabilities."""
        loader = AlphaVantageDataLoader()
        features = loader.supported_features

        assert "historical_bars" in features
        assert "fundamental_data" in features
        assert "macro_data" in features
        assert "news" in features
        assert "connection_managed" not in features
        assert "instrument_discovery" not in features
        assert "live_data" not in features
        assert "streaming" not in features
        assert "analyst_data" not in features
        assert "sector_data" not in features
        assert "company_profile" not in features


class TestFMPProtocols:
    """Protocol conformance tests for FMPDataSource."""

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_implements_historical_data_protocol(self):
        """FMPDataSource should implement HistoricalDataCapable."""
        loader = FMPDataSource()
        assert isinstance(loader, HistoricalDataCapable)
        assert hasattr(loader, "get_historical_ohlcv_data")

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_implements_analyst_data_protocol(self):
        """FMPDataSource should implement AnalystDataCapable."""
        loader = FMPDataSource()
        assert isinstance(loader, AnalystDataCapable)
        assert hasattr(loader, "get_historical_grades")
        assert hasattr(loader, "get_historical_rating")

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_implements_sector_data_protocol(self):
        """FMPDataSource should implement SectorDataCapable."""
        loader = FMPDataSource()
        assert isinstance(loader, SectorDataCapable)
        assert hasattr(loader, "get_historical_sector_performance")
        assert hasattr(loader, "get_historical_industry_performance")

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_implements_company_profile_protocol(self):
        """FMPDataSource should implement CompanyProfileCapable."""
        loader = FMPDataSource()
        assert isinstance(loader, CompanyProfileCapable)
        assert hasattr(loader, "get_company_profile")

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_does_not_implement_live_data_protocol(self):
        """FMPDataSource should NOT implement LiveDataCapable."""
        loader = FMPDataSource()
        assert not isinstance(loader, LiveDataCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_does_not_implement_news_protocol(self):
        """FMPDataSource should NOT implement NewsDataCapable."""
        loader = FMPDataSource()
        assert not isinstance(loader, NewsDataCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_does_not_implement_streaming_protocol(self):
        """FMPDataSource should NOT implement StreamingCapable."""
        loader = FMPDataSource()
        assert not isinstance(loader, StreamingCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_does_not_implement_fundamental_data_protocol(self):
        """FMPDataSource should NOT implement FundamentalDataCapable."""
        loader = FMPDataSource()
        assert not isinstance(loader, FundamentalDataCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_does_not_implement_macro_data_protocol(self):
        """FMPDataSource should NOT implement MacroDataCapable."""
        loader = FMPDataSource()
        assert not isinstance(loader, MacroDataCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_supported_features_accuracy(self):
        """supported_features property should reflect actual
        capabilities."""
        loader = FMPDataSource()
        features = loader.supported_features

        assert "historical_bars" in features
        assert "analyst_data" in features
        assert "sector_data" in features
        assert "company_profile" in features
        assert "connection_managed" not in features
        assert "instrument_discovery" not in features
        assert "live_data" not in features
        assert "news" not in features
        assert "streaming" not in features
        assert "fundamental_data" not in features
        assert "macro_data" not in features


class TestProtocolMethodSignatures:
    """Test that protocol methods have correct signatures."""

    def test_historical_data_protocol_signature(self):
        """HistoricalDataCapable methods should have expected
        signatures."""
        loader = YFinanceDataLoader()

        # Check method exists and is callable
        assert callable(getattr(loader, "get_historical_ohlcv_data", None))

        # Check signature accepts expected parameters
        import inspect

        sig = inspect.signature(loader.get_historical_ohlcv_data)
        params = list(sig.parameters.keys())

        assert "symbols" in params
        assert "start" in params
        # end and timeframe may have defaults, so just check they're recognized
        assert "end" in params or "end" in str(sig)
        assert "timeframe" in params or "timeframe" in str(sig)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_fundamental_data_protocol_signature(self):
        """FundamentalDataCapable methods should have expected
        signatures."""
        loader = AlphaVantageDataLoader()

        assert callable(getattr(loader, "get_fundamental_data", None))

        import inspect

        sig = inspect.signature(loader.get_fundamental_data)
        params = list(sig.parameters.keys())

        # AlphaVantage uses 'symbol' (singular), not 'symbols'
        assert "symbol" in params
        assert "metrics" in params

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_macro_data_protocol_signature(self):
        """MacroDataCapable methods should have expected signatures."""
        loader = AlphaVantageDataLoader()

        assert callable(getattr(loader, "get_macro_data", None))

        import inspect

        sig = inspect.signature(loader.get_macro_data)
        params = list(sig.parameters.keys())

        assert "indicators" in params
        assert "start" in params
        assert "end" in params

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_analyst_data_protocol_signature(self):
        """AnalystDataCapable methods should have expected
        signatures."""
        loader = FMPDataSource()

        # Check get_historical_grades
        assert callable(getattr(loader, "get_historical_grades", None))
        import inspect

        sig = inspect.signature(loader.get_historical_grades)
        params = list(sig.parameters.keys())
        assert "symbol" in params

        # Check get_historical_rating
        assert callable(getattr(loader, "get_historical_rating", None))
        sig = inspect.signature(loader.get_historical_rating)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "limit" in params

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_sector_data_protocol_signature(self):
        """SectorDataCapable methods should have expected signatures."""
        loader = FMPDataSource()

        # Check get_historical_sector_performance
        assert callable(getattr(loader, "get_historical_sector_performance", None))
        import inspect

        sig = inspect.signature(loader.get_historical_sector_performance)
        params = list(sig.parameters.keys())
        assert "sector" in params

        # Check get_historical_industry_performance
        assert callable(getattr(loader, "get_historical_industry_performance", None))
        sig = inspect.signature(loader.get_historical_industry_performance)
        params = list(sig.parameters.keys())
        assert "industry" in params

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_company_profile_protocol_signature(self):
        """CompanyProfileCapable methods should have expected
        signatures."""
        loader = FMPDataSource()

        # Check get_company_profile
        assert callable(getattr(loader, "get_company_profile", None))
        import inspect

        sig = inspect.signature(loader.get_company_profile)
        params = list(sig.parameters.keys())
        assert "symbol" in params

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_live_data_protocol_signature(self, mock_stream, mock_client):
        """LiveDataCapable methods should have expected signatures."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()

        # Check get_latest_quote
        assert callable(getattr(loader, "get_latest_quote", None))
        import inspect

        sig = inspect.signature(loader.get_latest_quote)
        params = list(sig.parameters.keys())
        # Alpaca uses 'symbol' (singular), not 'symbols'
        assert "symbol" in params

        # Check get_latest_trade
        assert callable(getattr(loader, "get_latest_trade", None))
        sig = inspect.signature(loader.get_latest_trade)
        params = list(sig.parameters.keys())
        # Alpaca uses 'symbol' (singular), not 'symbols'
        assert "symbol" in params


class TestSupportsFeaturesMethod:
    """Test the supports_feature() helper method."""

    def test_yfinance_supports_feature(self):
        """Test YFinanceDataLoader supports_feature method."""
        loader = YFinanceDataLoader()

        assert loader.supports_feature("historical_bars") is True
        assert loader.supports_feature("fundamental_data") is True
        assert loader.supports_feature("live_data") is False
        assert loader.supports_feature("news") is False
        assert loader.supports_feature("streaming") is False

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_alpaca_supports_feature(self, mock_stream, mock_client):
        """Test AlpacaDataLoader supports_feature method."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()

        assert loader.supports_feature("historical_bars") is True
        assert loader.supports_feature("live_data") is True
        assert loader.supports_feature("news") is True
        assert loader.supports_feature("streaming") is True
        assert loader.supports_feature("fundamental_data") is False
        assert loader.supports_feature("macro_data") is False

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_alpha_vantage_supports_feature(self):
        """Test AlphaVantageDataLoader supports_feature method."""
        loader = AlphaVantageDataLoader()

        assert loader.supports_feature("historical_bars") is True
        assert loader.supports_feature("fundamental_data") is True
        assert loader.supports_feature("macro_data") is True
        assert loader.supports_feature("news") is True
        assert loader.supports_feature("live_data") is False
        assert loader.supports_feature("streaming") is False

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_fmp_supports_feature(self):
        """Test FMPDataSource supports_feature method."""
        loader = FMPDataSource()

        assert loader.supports_feature("historical_bars") is True
        assert loader.supports_feature("analyst_data") is True
        assert loader.supports_feature("sector_data") is True
        assert loader.supports_feature("company_profile") is True
        assert loader.supports_feature("live_data") is False
        assert loader.supports_feature("news") is False
        assert loader.supports_feature("fundamental_data") is False
