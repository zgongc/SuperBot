#!/usr/bin/env python3
"""
components/exchanges/kucoin_api.py
SuperBot - KuCoin Exchange API
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

KuCoin Exchange API implementation

Features:
- Uses the CCXT library
- Supports testnet/production environments
- Spot trading

Usage:
    from components.exchanges import KucoinAPI
    from core.config_engine import ConfigEngine

    config = ConfigEngine().get_config('connectors')['kucoin']
    kucoin = KucoinAPI(config=config)

    ticker = await kucoin.get_ticker('BTC/USDT')

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
# KUCOIN API
# ============================================================================

class KucoinAPI(CCXTWrapper):
    """
    KuCoin Exchange API

    CCXT library kullanarak KuCoin API'yi wrap eder
    Config source: config/connectors.yaml -> kucoin section
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KuCoin API.

        Args:
            config: Kucoin section from the configuration file.
        """
        super().__init__(exchange_name='kucoin', config=config)

    # KuCoin-specific methods buraya eklenebilir
    # For now, the methods inherited from CCXTWrapper are sufficient.


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'KucoinAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 KucoinAPI Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  KuCoin API test:")

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
            "timeout": {
                "read": 30
            }
        }

        try:
            # Create KucoinAPI
            kucoin = KucoinAPI(config=test_config)
            print(f"   ✅ KucoinAPI created")
            print(f"   - Exchange: {kucoin.exchange_name}")
            print(f"   - Testnet: {kucoin.testnet}")
            print(f"   - Health: {kucoin.health_check()}")

            # Stats
            stats = kucoin.get_stats()
            print(f"   - Stats: {stats}")

            # Close
            await kucoin.close()

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
