"""
datamodel.py — minimal replica of IMC Prosperity's datamodel.
Keeps the same class names and signatures so the real Trader code
runs unchanged inside the backtester.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


class OrderDepth:
    """
    Holds the current limit-order book for one product.
    buy_orders  : {price: volume}   (positive volumes, bids)
    sell_orders : {price: volume}   (negative volumes, asks — matches IMC convention)
    """
    def __init__(self):
        self.buy_orders:  Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


@dataclass
class Order:
    symbol:   str
    price:    int
    quantity: int   # positive = buy, negative = sell


@dataclass
class Trade:
    symbol:   str
    price:    float
    quantity: int
    buyer:    str = ""
    seller:   str = ""
    timestamp: int = 0


class TradingState:
    def __init__(
        self,
        timestamp:    int,
        listings:     dict,
        order_depths: Dict[str, OrderDepth],
        own_trades:   Dict[str, List[Trade]],
        market_trades:Dict[str, List[Trade]],
        position:     Dict[str, int],
        observations: dict,
        traderData:   str = "",
    ):
        self.timestamp     = timestamp
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations
        self.traderData    = traderData
