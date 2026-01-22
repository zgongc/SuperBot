"""
components/exchanges
Exchange API implementations

A single entry point for all exchange APIs.
"""

from __future__ import annotations

from components.exchanges.base_api import BaseExchangeAPI
from components.exchanges.binance_api import BinanceAPI

# TODO: CCXT wrapper will be added for other exchanges.
# from components.exchanges.ccxt_wrapper import CCXTWrapper
# from components.exchanges.bybit_api import BybitAPI
# from components.exchanges.okx_api import OkxAPI
# from components.exchanges.kucoin_api import KucoinAPI
# from components.exchanges.gateio_api import GateioAPI

__all__ = [
    # Base class
    'BaseExchangeAPI',

    # Exchange implementations
    'BinanceAPI',
    # 'BybitAPI',      # TODO: Implement with CCXT wrapper
    # 'OkxAPI',        # TODO: With CCXT wrapper
    # 'KucoinAPI',     # TODO: Implement with CCXT wrapper
    # 'GateioAPI',     # TODO: Use CCXT wrapper
]
