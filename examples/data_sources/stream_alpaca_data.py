"""
Example: Streaming real-time data from Alpaca.

This script demonstrates how to:
1. Connect to Alpaca's WebSocket stream
2. Subscribe to trade updates
3. Handle the continuous data stream

Requirements:
- Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file
- Install requirements (including 'alpaca-py')
"""

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

from quantrl_lab.data.sources import AlpacaDataLoader

# Load environment variables
load_dotenv()


async def main():
    # Configure logger to show DEBUG messages (where the data is logged by default handlers)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("Initializing Alpaca Data Loader...")
    loader = AlpacaDataLoader()

    if not loader.is_connected():
        logger.error("Alpaca credentials not found. Check your .env file.")
        return

    # Symbol to subscribe to
    # Note: For free tier, IEX feed (stocks) or Crypto feed is available.
    # 'AAPL' might require a paid subscription for real-time SIP data,
    # but usually works with IEX (delayed or partial) on free tier.
    # NOTE: The current AlpacaDataLoader is configured for Stock Data only.
    # Crypto pairs (like BTC/USD) require CryptoDataStream and will fail with 400.
    symbol = "SPY"

    logger.info(f"Subscribing to trades for {symbol}...")

    # Subscribe to trades
    # The default handler in AlpacaDataLoader logs received trades at DEBUG level
    await loader.subscribe_to_updates(symbol=symbol, data_type="trades")

    # Example: Subscribe to quotes (if needed)
    # await loader.subscribe_to_updates(symbol=symbol, data_type="quotes")

    logger.info("Starting stream. Press Ctrl+C to stop.")

    try:
        # This will run forever until interrupted
        await loader.start_streaming()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await loader.stop_streaming()
        logger.success("Stream closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
