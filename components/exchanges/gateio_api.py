#!/usr/bin/env python3
"""
components/exchanges/gateio_api.py
SuperBot - Gate.io Exchange API
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

Gate.io Exchange API implementation

Features:
- Uses the CCXT library
- Supports testnet/production environments
- Spot trading

Usage:
    from components.exchanges import GateioAPI
    from core.config_engine import ConfigEngine

    config = ConfigEngine().get_config('connectors')['gateio']
    gateio = GateioAPI(config=config)

    ticker = await gateio.get_ticker('BTC/USDT')

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
# GATEIO API
# ============================================================================

class GateioAPI(CCXTWrapper):
    """
    Gate.io Exchange API

    CCXT library kullanarak Gate.io API'yi wrap eder
    Config source: config/connectors.yaml -> gateio section
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gate.io API.

        Args:
            config: Gate.io section from the configuration file.
        """
        super().__init__(exchange_name='gateio', config=config)

    # Gate.io-specific methods buraya eklenebilir
    # For now, the methods inherited from CCXTWrapper are sufficient.


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'GateioAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 GateioAPI Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  Gate.io API test:")

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
            # Create GateioAPI
            gateio = GateioAPI(config=test_config)
            print(f"   ✅ GateioAPI created")
            print(f"   - Exchange: {gateio.exchange_name}")
            print(f"   - Testnet: {gateio.testnet}")
            print(f"   - Health: {gateio.health_check()}")

            # Stats
            stats = gateio.get_stats()
            print(f"   - Stats: {stats}")

            # Close
            await gateio.close()

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
