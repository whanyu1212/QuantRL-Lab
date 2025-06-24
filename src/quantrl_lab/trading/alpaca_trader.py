import os
from typing import List, Union

import alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    AssetStatus,
    OrderClass,
    OrderSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class AlpacaTradingClient:
    """
    Alpaca Trading Client interface to interact with Alpaca Trading API.

    We will be using the python sdk primarily and not the REST API
    directly.
    """

    def __init__(self, api_key: str = None, secret_key: str = None, paper_trading: bool = True):
        self.api_key = api_key if api_key else os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key if secret_key else os.getenv("ALPACA_SECRET_KEY")
        self.paper_trading = paper_trading

    def connect_client(self):
        """Initialize the Alpaca Trading Client with the provided API
        Key and Secret Key."""
        try:
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_trading)
            logger.info("Successfully connected to Alpaca Trading API")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca Trading API: {e}")

    def get_account_details(self) -> alpaca.trading.models.TradeAccount:
        account = self.trading_client.get_account()
        logger.info(f"Account details have been retrieved:" f"{account}")
        return account

    # streams for a long time, not very useful, keeping here for completeness
    def get_all_assets(
        self,
        asset_class: AssetClass = AssetClass.US_EQUITY,
        asset_status: AssetStatus = AssetStatus.ACTIVE,
    ) -> List[alpaca.trading.models.Asset]:
        """
        Get all assets that are currently available for trading on
        Alpaca.

        Args:
            asset_class (AssetClass, optional): Defaults to AssetClass.US_EQUITY.
            asset_status (AssetStatus, optional): Defaults to AssetStatus.ACTIVE.

        Returns:
            List[alpaca.trading.models.Asset]: List of assets that match the search criteria.
        """
        search_params = GetAssetsRequest(asset_class=asset_class, status=asset_status)
        assets = self.trading_client.get_all_assets(search_params)
        return assets

    def get_account_assets_by_symbol(self, symbol: str) -> alpaca.trading.models.Asset:
        """
        Get asset details for a specific symbol.

        Args:
            symbol (str): Symbol of the asset to get details for.

        Returns:
            alpaca.trading.models.Asset: Asset details for the provided symbol.
        """
        asset = self.trading_client.get_asset(symbol)
        logger.info(f"Asset details for {symbol} have been retrieved:" f"{asset}")
        return asset

    # oto and oco orders are not supported here due to complexity
    # and the fact that they are not used very often
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Union[int, float] = None,
        notional: float = None,
        order_type: str = "market",  # market, limit, stop, stop_limit, trailing_stop, bracket
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: float = None,
        stop_price: float = None,
        trail_percent: float = None,
        take_profit_price: float = None,
        stop_loss_price: float = None,
    ) -> alpaca.trading.models.Order:
        """
        Create an order with the specified parameters, automatically
        selecting the appropriate order type.

        Args:
            symbol (str): Symbol of the asset to place the order for.
            side (OrderSide): Buy or sell order.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            order_type (str, optional): Type of order to create. Defaults to "market".
                Options: market, limit, stop, stop_limit, trailing_stop, bracket
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.
            limit_price (float, optional): Required for limit and stop_limit orders.
            stop_price (float, optional): Required for stop and stop_limit orders.
            trail_percent (float, optional): Required for trailing_stop orders.
            take_profit_price (float, optional): Required for bracket orders.
            stop_loss_price (float, optional): Required for bracket orders.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
            Additional parameters are required based on order_type:
            - limit: limit_price
            - stop: stop_price
            - stop_limit: stop_price and limit_price
            - trailing_stop: trail_percent
            - bracket: take_profit_price and stop_loss_price
        """
        # Validate qty or notional is provided, but not both
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Normalize order_type to lowercase for case-insensitive comparison
        order_type = order_type.lower()

        # Market Order
        if order_type == "market":
            return self._create_market_order(
                symbol=symbol,
                qty=qty,
                notional=notional,
                side=side,
                time_in_force=time_in_force,
            )

        # Limit Order
        elif order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")

            return self._create_limit_order(
                symbol=symbol,
                limit_price=limit_price,
                qty=qty,
                notional=notional,
                side=side,
                time_in_force=time_in_force,
            )

        # Stop Order
        elif order_type == "stop":
            if stop_price is None:
                raise ValueError("stop_price is required for stop orders")

            return self._create_stop_order(
                symbol=symbol,
                stop_price=stop_price,
                qty=qty,
                notional=notional,
                side=side,
                time_in_force=time_in_force,
            )

        # Stop Limit Order
        elif order_type == "stop_limit":
            if stop_price is None or limit_price is None:
                raise ValueError("Both stop_price and limit_price are required for stop_limit orders")

            return self._create_stop_limit_order(
                symbol=symbol,
                stop_price=stop_price,
                limit_price=limit_price,
                qty=qty,
                notional=notional,
                side=side,
                time_in_force=time_in_force,
            )

        # Trailing Stop Order
        elif order_type == "trailing_stop":
            if trail_percent is None:
                raise ValueError("trail_percent is required for trailing_stop orders")

            return self._create_trailing_stop_order(
                symbol=symbol,
                trail_percent=trail_percent,
                qty=qty,
                notional=notional,
                side=side,
                time_in_force=time_in_force,
            )

        # Bracket Order
        elif order_type == "bracket":
            if take_profit_price is None or stop_loss_price is None:
                raise ValueError("Both take_profit_price and stop_loss_price are required for bracket orders")

            return self._create_bracket_order(
                symbol=symbol,
                qty=qty,
                notional=notional,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                side=side,
                time_in_force=time_in_force,
            )

        else:
            raise ValueError(
                f"Unsupported order_type: {order_type}. Supported types are: "
                "market, limit, stop, stop_limit, trailing_stop, bracket"
            )

    def _create_market_order(
        self,
        symbol: str,
        qty: Union[int, float] = None,
        notional: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a market order using either quantity of shares or dollar
        amount (notional).

        Args:
            symbol (str): Symbol of the asset to place the order for.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Create market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
        )

        market_order = self.trading_client.submit_order(market_order_data)

        if qty:
            logger.info(f"Market Order of {qty} shares for {symbol} has been placed.")
        else:
            logger.info(f"Market Order of ${notional} for {symbol} has been placed.")

        return market_order

    def _create_limit_order(
        self,
        symbol: str,
        limit_price: float,
        qty: Union[int, float] = None,
        notional: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a limit order using either quantity of shares or dollar
        amount (notional).

        Args:
            symbol (str): Symbol of the asset to place the order for.
            limit_price (float): Price at which the order should be executed.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Create limit order request
        limit_order_data = LimitOrderRequest(
            symbol=symbol,
            limit_price=limit_price,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
        )

        limit_order = self.trading_client.submit_order(limit_order_data)

        if qty:
            logger.info(f"Limit Order of {qty} shares for {symbol} at ${limit_price} has been placed.")
        else:
            logger.info(f"Limit Order of ${notional} for {symbol} at ${limit_price} has been placed.")

        return limit_order

    # Stop order is a market order that is
    # only executed when the price of the asset reaches a certain level.
    def _create_stop_order(
        self,
        symbol: str,
        stop_price: float,
        qty: Union[int, float] = None,
        notional: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a stop order using either quantity of shares or dollar
        amount (notional).

        Args:
            symbol (str): Symbol of the asset to place the order for.
            stop_price (float): Price at which the stop order becomes active.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Create stop order request
        stop_order_data = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
            stop_price=stop_price,
        )

        stop_order = self.trading_client.submit_order(stop_order_data)

        if qty:
            logger.info(f"Stop Order of {qty} shares for {symbol} at ${stop_price} has been placed.")
        else:
            logger.info(f"Stop Order of ${notional} for {symbol} at ${stop_price} has been placed.")

        return stop_order

    def _create_stop_limit_order(
        self,
        symbol: str,
        stop_price: float,
        limit_price: float,
        qty: Union[int, float] = None,
        notional: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a stop limit order using either quantity of shares or
        dollar amount (notional).

        Args:
            symbol (str): Symbol of the asset to place the order for.
            stop_price (float): Price at which the stop order becomes active.
            limit_price (float): Maximum/minimum price to buy/sell at once triggered.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Create stop limit order request
        stop_limit_order_data = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
            stop_price=stop_price,
            limit_price=limit_price,
        )

        stop_limit_order = self.trading_client.submit_order(stop_limit_order_data)

        if qty:
            logger.info(
                f"Stop Limit Order of {qty} shares for {symbol} with stop price ${stop_price} "
                f"and limit price ${limit_price} has been placed."
            )
        else:
            logger.info(
                f"Stop Limit Order of ${notional} for {symbol} with stop price ${stop_price} "
                f"and limit price ${limit_price} has been placed."
            )

        return stop_limit_order

    def _create_trailing_stop_order(
        self,
        symbol: str,
        trail_percent: float = 0.2,
        qty: Union[int, float] = None,
        notional: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a trailing stop order using either quantity of shares or
        dollar amount (notional).

        Args:
            symbol (str): Symbol of the asset to place the order for.
            trail_percent (float, optional): Percentage distance for the trailing stop. Defaults to 0.2.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        # Create trailing stop order request
        trailing_stop_order_data = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
            trail_percent=trail_percent,
        )

        trailing_stop_order = self.trading_client.submit_order(trailing_stop_order_data)

        if qty:
            logger.info(
                f"Trailing Stop Order of {qty} shares for {symbol} with {trail_percent}% trail has been placed."
            )
        else:
            logger.info(f"Trailing Stop Order of ${notional} for {symbol} with {trail_percent}% trail has been placed.")

        return trailing_stop_order

    def _create_bracket_order(
        self,
        symbol: str,
        qty: Union[int, float] = None,
        notional: float = None,
        take_profit_price: float = None,
        stop_loss_price: float = None,
        side: OrderSide = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> alpaca.trading.models.Order:
        """
        Create a bracket order with take profit and stop loss using
        either quantity or notional value.

        A bracket order is a market order with take profit and stop loss orders attached.

        Args:
            symbol (str): Symbol of the asset to place the order for.
            qty (Union[int, float], optional): Quantity of shares to buy/sell.
            notional (float, optional): Dollar amount to buy/sell.
            take_profit_price (float): Limit price for the take profit order.
            stop_loss_price (float): Stop price for the stop loss order.
            side (OrderSide): Buy or sell order.
            time_in_force (TimeInForce, optional): How long the order should be active. Defaults to TimeInForce.DAY.

        Returns:
            alpaca.trading.models.Order: Details of the submitted order

        Note:
            You must provide either qty OR notional, not both.
            Both take_profit_price and stop_loss_price must be provided.
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("You must provide either qty OR notional, not both or neither.")

        if take_profit_price is None or stop_loss_price is None:
            raise ValueError("Both take_profit_price and stop_loss_price must be provided.")

        # Validate stop loss and take profit prices based on side
        if side == OrderSide.BUY:
            if take_profit_price <= stop_loss_price:
                raise ValueError("For BUY orders, take_profit_price must be higher than stop_loss_price.")
        elif side == OrderSide.SELL:
            if take_profit_price >= stop_loss_price:
                raise ValueError("For SELL orders, take_profit_price must be lower than stop_loss_price.")

        # Create bracket order request
        bracket_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            time_in_force=time_in_force,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )

        bracket_order = self.trading_client.submit_order(bracket_order_data)

        if qty:
            logger.info(
                f"Bracket Order of {qty} shares for {symbol} with take profit at ${take_profit_price} "
                f"and stop loss at ${stop_loss_price} has been placed."
            )
        else:
            logger.info(
                f"Bracket Order of ${notional} for {symbol} with take profit at ${take_profit_price} "
                f"and stop loss at ${stop_loss_price} has been placed."
            )

        return bracket_order

    def get_all_orders(self, symbol: Union[str, List[str]]) -> List[alpaca.trading.models.Order]:
        """
        Get all orders (regardless of status) for the specified
        symbol(s).

        Args:
            symbol (Union[str, List[str]]): Symbol(s) of the asset(s) to get orders for.

        Returns:
            List[alpaca.trading.models.Order]: List of orders that match the search criteria.
        """

        if isinstance(symbol, str):
            symbol = [symbol]

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=symbol)
        orders = self.trading_client.get_orders(req)
        return orders

    def get_all_open_orders(self, symbol: Union[str, List[str]]) -> List[alpaca.trading.models.Order]:
        """
        Get all open orders for the specified symbol(s).

        Args:
            symbol (Union[str, List[str]]): Symbol(s) of the asset(s) to get open orders for.

        Returns:
            List[alpaca.trading.models.Order]: List of open orders that match the search criteria.
        """
        if isinstance(symbol, str):
            symbol = [symbol]
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        open_orders = self.trading_client.get_orders(req)
        return open_orders

    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the account."""
        self.trading_client.cancel_orders()

    def get_all_positions(self) -> List[alpaca.trading.models.Position]:
        """
        Get all open positions for the account.

        Returns:
            List[alpaca.trading.models.Position]: List of open positions.
        """
        positions = self.trading_client.get_all_positions()
        return positions

    def get_open_position_by_symbol(self, symbol: str) -> alpaca.trading.models.Position:
        """
        Get open position for a specific symbol.

        Args:
            symbol (str): Symbol of the asset to get position for.

        Returns:
            alpaca.trading.models.Position: Open position for the provided symbol.
        """
        position = self.trading_client.get_open_position(symbol_or_asset_id=symbol)
        return position

    def close_position(self, symbol: str, qty: Union[int, float]) -> None:
        """
        Close the position for the specified symbol by selling a
        specific quantity of shares.

        Args:
            symbol (str): Symbol of the asset to close the position for.
            qty (Union[int, float]): Quantity of shares to sell.
        """
        self.trading_client.close_position(
            symbol_or_asset_id=symbol,
            close_options=ClosePositionRequest(
                qty=qty,
            ),
        )

    def close_all_positions(self) -> None:
        """Close all open positions for the account."""
        self.trading_client.close_all_positions()


if __name__ == "__main__":
    alpaca_client = AlpacaTradingClient()
    alpaca_client.connect_client()
    account_details = alpaca_client.get_account_details()
    account_details = alpaca_client.get_account_assets_by_symbol("AAPL")
    market_order = alpaca_client._create_market_order("AAPL", 1, OrderSide.BUY)
