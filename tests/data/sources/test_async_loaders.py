"""
Tests for async methods added to data loaders.

Covers:
- YFinanceDataLoader.async_fetch_ohlcv
- FMPDataSource.async_fetch_ohlcv
- FMPDataSource.async_fetch_ratings
- FMPDataSource.async_fetch_company_profile
- FMPDataSource.async_fetch_sector_perf
- FMPDataSource.async_fetch_industry_perf
- AlpacaDataLoader.async_fetch_news
"""

import re
from unittest.mock import patch

import aiohttp
import pandas as pd
import pytest
from aioresponses import aioresponses

from quantrl_lab.data.exceptions import APIConnectionError

# ── YFinanceDataLoader ────────────────────────────────────────────────────────


class TestYFinanceAsyncFetchOhlcv:
    """Tests for YFinanceDataLoader.async_fetch_ohlcv."""

    @pytest.fixture
    def loader(self):
        from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader

        return YFinanceDataLoader()

    @pytest.fixture
    def mock_ohlcv_df(self):
        return pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=3),
                "Open": [150.0, 151.0, 152.0],
                "High": [153.0, 154.0, 155.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [151.0, 152.0, 153.0],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
                "Symbol": ["AAPL", "AAPL", "AAPL"],
            }
        )

    @pytest.mark.asyncio
    async def test_returns_symbol_and_dataframe(self, loader, mock_ohlcv_df):
        with patch.object(loader, "_fetch_single_symbol", return_value=mock_ohlcv_df):
            sym, df = await loader.async_fetch_ohlcv("AAPL", start="2023-01-01", end="2023-01-31")

        assert sym == "AAPL"
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.asyncio
    async def test_returns_empty_df_on_exception(self, loader):
        with patch.object(loader, "_fetch_single_symbol", side_effect=Exception("network error")):
            sym, df = await loader.async_fetch_ohlcv("AAPL", start="2023-01-01", end="2023-01-31")

        assert sym == "AAPL"
        assert df.empty

    @pytest.mark.asyncio
    async def test_multiple_symbols_concurrently(self, loader, mock_ohlcv_df):
        import asyncio

        def make_df(sym):
            df = mock_ohlcv_df.copy()
            df["Symbol"] = sym
            return df

        with patch.object(loader, "_fetch_single_symbol", side_effect=lambda sym, *a, **kw: make_df(sym)):
            results = await asyncio.gather(
                loader.async_fetch_ohlcv("AAPL", start="2023-01-01", end="2023-01-31"),
                loader.async_fetch_ohlcv("MSFT", start="2023-01-01", end="2023-01-31"),
                loader.async_fetch_ohlcv("GOOG", start="2023-01-01", end="2023-01-31"),
            )

        symbols = [sym for sym, _ in results]
        assert set(symbols) == {"AAPL", "MSFT", "GOOG"}
        assert all(not df.empty for _, df in results)

    @pytest.mark.asyncio
    async def test_uses_asyncio_to_thread(self, loader, mock_ohlcv_df):
        """Verify the call is dispatched via asyncio.to_thread (non-
        blocking)."""
        with patch("asyncio.to_thread", return_value=mock_ohlcv_df) as mock_to_thread:
            sym, df = await loader.async_fetch_ohlcv("AAPL", start="2023-01-01")

        mock_to_thread.assert_called_once()
        assert sym == "AAPL"


# ── FMPDataSource ─────────────────────────────────────────────────────────────


class TestFMPAsyncMethods:
    """Tests for FMPDataSource async methods."""

    @pytest.fixture
    def fmp(self):
        with patch.dict("os.environ", {"FMP_API_KEY": "test_key"}):
            from quantrl_lab.data.sources.fmp_loader import FMPDataSource

            return FMPDataSource()

    @pytest.fixture
    def base_url(self):
        return "https://financialmodelingprep.com/stable"

    @pytest.mark.asyncio
    async def test_async_fetch_ohlcv_returns_dataframe(self, fmp, base_url):
        payload = [
            {"date": "2023-01-03", "open": 130.0, "high": 133.0, "low": 129.0, "close": 132.0, "volume": 5_000_000},
            {"date": "2023-01-04", "open": 132.0, "high": 135.0, "low": 131.0, "close": 134.0, "volume": 4_800_000},
        ]
        with aioresponses() as m:
            m.get(re.compile(r".*/historical-price-eod/full.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sym, df = await fmp.async_fetch_ohlcv(session, "AAPL", start="2023-01-01", end="2023-01-31")
                assert sym == "AAPL"
                assert isinstance(df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_async_fetch_ohlcv_raises_on_api_failure(self, fmp, base_url):
        with aioresponses() as m:
            for _ in range(4):
                m.get(re.compile(r".*/historical-price-eod/full.*"), status=500)
            async with aiohttp.ClientSession() as session:
                with pytest.raises(APIConnectionError):
                    await fmp.async_fetch_ohlcv(session, "AAPL", start="2023-01-01", end="2023-01-31")

    @pytest.mark.asyncio
    async def test_async_fetch_ratings_returns_dataframe(self, fmp, base_url):
        payload = [
            {"date": "2023-01-03", "symbol": "AAPL", "ratingScore": 4, "ratingRecommendation": "Buy"},
            {"date": "2023-01-10", "symbol": "AAPL", "ratingScore": 3, "ratingRecommendation": "Hold"},
        ]
        with aioresponses() as m:
            m.get(re.compile(r".*/ratings-historical.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sym, df = await fmp.async_fetch_ratings(session, "AAPL", limit=500)
                assert sym == "AAPL"
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 2

    @pytest.mark.asyncio
    async def test_async_fetch_ratings_returns_empty_on_empty_response(self, fmp, base_url):
        with aioresponses() as m:
            m.get(re.compile(r".*/ratings-historical.*"), payload=[])
            async with aiohttp.ClientSession() as session:
                sym, df = await fmp.async_fetch_ratings(session, "AAPL")
                assert sym == "AAPL"
                assert df.empty

    @pytest.mark.asyncio
    async def test_async_fetch_company_profile_returns_dataframe(self, fmp, base_url):
        payload = [
            {"symbol": "AAPL", "companyName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"}
        ]
        with aioresponses() as m:
            m.get(re.compile(r".*/profile.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sym, df = await fmp.async_fetch_company_profile(session, "AAPL")
                assert sym == "AAPL"
                assert not df.empty
                assert df.iloc[0]["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_async_fetch_company_profile_returns_empty_on_failure(self, fmp, base_url):
        with aioresponses() as m:
            m.get(re.compile(r".*/profile.*"), payload=[])
            async with aiohttp.ClientSession() as session:
                sym, df = await fmp.async_fetch_company_profile(session, "AAPL")
                assert sym == "AAPL"
                assert df.empty

    @pytest.mark.asyncio
    async def test_async_fetch_sector_perf_returns_dataframe(self, fmp, base_url):
        payload = [
            {"date": "2023-01-03", "sector": "Technology", "changesPercentage": "0.5"},
            {"date": "2023-01-04", "sector": "Technology", "changesPercentage": "1.2"},
        ]
        with aioresponses() as m:
            m.get(re.compile(r".*/historical-sector-performance.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sector, df = await fmp.async_fetch_sector_perf(session, "Technology", "2023-01-01", "2023-01-31")
                assert sector == "Technology"
                assert len(df) == 2

    @pytest.mark.asyncio
    async def test_async_fetch_industry_perf_returns_dataframe(self, fmp, base_url):
        payload = [
            {"date": "2023-01-03", "industry": "Consumer Electronics", "changesPercentage": "0.3"},
        ]
        with aioresponses() as m:
            m.get(re.compile(r".*/historical-industry-performance.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                industry, df = await fmp.async_fetch_industry_perf(
                    session, "Consumer Electronics", "2023-01-01", "2023-01-31"
                )
                assert industry == "Consumer Electronics"
                assert len(df) == 1

    @pytest.mark.asyncio
    async def test_multiple_fmp_calls_concurrently(self, fmp, base_url):
        """Test that multiple FMP async calls can be gathered
        concurrently."""
        import asyncio

        ratings_payload = [{"date": "2023-01-03", "ratingScore": 4}]
        profile_payload = [{"symbol": "AAPL", "sector": "Technology"}]

        with aioresponses() as m:
            m.get(re.compile(r".*/ratings-historical.*"), payload=ratings_payload)
            m.get(re.compile(r".*/profile.*"), payload=profile_payload)
            async with aiohttp.ClientSession() as session:
                (sym_r, df_r), (sym_p, df_p) = await asyncio.gather(
                    fmp.async_fetch_ratings(session, "AAPL"),
                    fmp.async_fetch_company_profile(session, "AAPL"),
                )
                assert sym_r == "AAPL"
                assert sym_p == "AAPL"
                assert not df_r.empty
                assert not df_p.empty


# ── AlpacaDataLoader ──────────────────────────────────────────────────────────


class TestAlpacaAsyncFetchNews:
    """Tests for AlpacaDataLoader.async_fetch_news."""

    @pytest.fixture
    def alpaca(self):
        with patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            with patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient"):
                with patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream"):
                    from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader

                    return AlpacaDataLoader()

    @pytest.fixture
    def news_url(self):
        from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader

        return AlpacaDataLoader.NEWS_API_BASE_URL

    @pytest.mark.asyncio
    async def test_returns_symbol_and_dataframe(self, alpaca, news_url):
        payload = {
            "news": [
                {"id": 1, "headline": "Apple hits record high", "symbols": ["AAPL"]},
                {"id": 2, "headline": "Apple reports earnings", "symbols": ["AAPL"]},
            ],
            "next_page_token": None,
        }
        with aioresponses() as m:
            m.get(re.compile(r".*alpaca\.markets.*news.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sym, df = await alpaca.async_fetch_news(session, "AAPL", start="2023-01-01", end="2023-01-31")
                assert sym == "AAPL"
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_df_on_no_news(self, alpaca, news_url):
        payload = {"news": [], "next_page_token": None}
        with aioresponses() as m:
            m.get(re.compile(r".*alpaca\.markets.*news.*"), payload=payload)
            async with aiohttp.ClientSession() as session:
                sym, df = await alpaca.async_fetch_news(session, "AAPL", start="2023-01-01", end="2023-01-31")
                assert sym == "AAPL"
                assert df.empty

    @pytest.mark.asyncio
    async def test_returns_empty_df_on_http_error(self, alpaca, news_url):
        with aioresponses() as m:
            m.get(re.compile(r".*alpaca\.markets.*news.*"), status=403)
            async with aiohttp.ClientSession() as session:
                sym, df = await alpaca.async_fetch_news(session, "AAPL", start="2023-01-01", end="2023-01-31")
                assert sym == "AAPL"
                assert df.empty

    @pytest.mark.asyncio
    async def test_paginates_through_multiple_pages(self, alpaca, news_url):
        page1 = {
            "news": [{"id": 1, "headline": "Story 1"}],
            "next_page_token": "token_page2",
        }
        page2 = {
            "news": [{"id": 2, "headline": "Story 2"}],
            "next_page_token": None,
        }
        with aioresponses() as m:
            m.get(re.compile(r".*alpaca\.markets.*news.*"), payload=page1)
            m.get(re.compile(r".*alpaca\.markets.*news.*"), payload=page2)
            async with aiohttp.ClientSession() as session:
                sym, df = await alpaca.async_fetch_news(session, "AAPL", start="2023-01-01", end="2023-01-31")
                assert sym == "AAPL"
                assert len(df) == 2
                assert set(df["id"].tolist()) == {1, 2}

    @pytest.mark.asyncio
    async def test_multiple_symbols_fetched_concurrently(self, alpaca, news_url):
        import asyncio

        def news_payload(sym):
            return {"news": [{"id": 1, "headline": f"{sym} news"}], "next_page_token": None}

        with aioresponses() as m:
            for sym in ["AAPL", "MSFT", "GOOG"]:
                m.get(re.compile(r".*alpaca\.markets.*news.*"), payload=news_payload(sym))
            async with aiohttp.ClientSession() as session:
                results = await asyncio.gather(
                    alpaca.async_fetch_news(session, "AAPL", start="2023-01-01", end="2023-01-31"),
                    alpaca.async_fetch_news(session, "MSFT", start="2023-01-01", end="2023-01-31"),
                    alpaca.async_fetch_news(session, "GOOG", start="2023-01-01", end="2023-01-31"),
                )
                assert len(results) == 3
                assert all(not df.empty for _, df in results)
