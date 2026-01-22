#!/usr/bin/env python3
"""
components/exchanges/bybit_api.py
SuperBot - Bybit Exchange API
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

Bybit Exchange API implementation

Features:
- Uses the CCXT library
- Supports testnet/production environments
- Supports spot and futures trading

Usage:
    from components.exchanges import BybitAPI
    from core.config_engine import ConfigEngine

    config = ConfigEngine().get_config('connectors')['bybit']
    bybit = BybitAPI(config=config)

    ticker = await bybit.get_ticker('BTC/USDT')

Dependencies:
    - ccxt>=4.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.exchanges.ccxt_wrapper import CCXTWrapper


# ============================================================================
# BYBIT API
# ============================================================================

class BybitAPI(CCXTWrapper):
    """
    Bybit Exchange API

    CCXT library kullanarak Bybit API'yi wrap eder
    Config source: config/connectors.yaml -> bybit section
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bybit API.

        Args:
            config: bybit section from the config file
        """
        super().__init__(exchange_name='bybit', config=config)

        # Bybit-specific configurations
        self.features = config.get('features', {})
        self.spot_enabled = self.features.get('spot_trading', True)
        self.futures_enabled = self.features.get('futures_trading', True)

    # Bybit-specific methods buraya eklenebilir
    # For now, the methods inherited from CCXTWrapper are sufficient.


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BybitAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 BybitAPI Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  Bybit API test:")

        # Test config
        test_config = {
            "enabled": True,
            "testnet": True,
            "endpoints": {
                "testnet": {
                    "api_key": "test_key",
                    "secret_key": "test_secret"
                }
            },
            "features": {
                "spot_trading": True,
                "futures_trading": True
            },
            "timeout": {
                "read": 30
            }
        }

        try:
            # Create BybitAPI
            bybit = BybitAPI(config=test_config)
            print(f"   ✅ BybitAPI created")
            print(f"   - Exchange: {bybit.exchange_name}")
            print(f"   - Testnet: {bybit.testnet}")
            print(f"   - Spot enabled: {bybit.spot_enabled}")
            print(f"   - Futures enabled: {bybit.futures_enabled}")
            print(f"   - Health: {bybit.health_check()}")

            # Stats
            stats = bybit.get_stats()
            print(f"   - Stats: {stats}")

            # Close
            await bybit.close()

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
