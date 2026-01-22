#!/usr/bin/env python3
"""
components/exchanges/okx_api.py
SuperBot - OKX Exchange API
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

OKX Exchange API implementation

Features:
- Uses the CCXT library
- Supports testnet/production environments
- Supports passphrase (required for OKX)
- Supports spot and futures trading

Usage:
    from components.exchanges import OkxAPI
    from core.config_engine import ConfigEngine

    config = ConfigEngine().get_config('connectors')['okx']
    okx = OkxAPI(config=config)

    ticker = await okx.get_ticker('BTC/USDT')

Dependencies:
    - ccxt>=4.0.0

Not:
    OKX requires passphrase in addition to api_key and secret_key
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
# OKX API
# ============================================================================

class OkxAPI(CCXTWrapper):
    """
    OKX Exchange API

    CCXT library kullanarak OKX API'yi wrap eder
    Config source: config/connectors.yaml -> okx section

    NOTE: In addition to OKX, api_key and secret_key, a passphrase is required.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OKX API.

        Args:
            config: okx section from the configuration file
                    (must include: endpoints.testnet.passphrase)
        """
        super().__init__(exchange_name='okx', config=config)

        # OKX-specific configurations
        self.features = config.get('features', {})
        self.spot_enabled = self.features.get('spot_trading', True)
        self.futures_enabled = self.features.get('futures_trading', True)

    # OKX-specific methods buraya eklenebilir
    # For now, the methods inherited from CCXTWrapper are sufficient.


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'OkxAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 OkxAPI Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  OKX API test:")

        # Test config (with passphrase)
        test_config = {
            "enabled": True,
            "testnet": True,
            "endpoints": {
                "testnet": {
                    "api_key": "test_key",
                    "secret_key": "test_secret",
                    "passphrase": "test_passphrase"  # Required for OKX
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
            # Create OkxAPI
            okx = OkxAPI(config=test_config)
            print(f"   ✅ OkxAPI created")
            print(f"   - Exchange: {okx.exchange_name}")
            print(f"   - Testnet: {okx.testnet}")
            print(f"   - Spot enabled: {okx.spot_enabled}")
            print(f"   - Futures enabled: {okx.futures_enabled}")
            print(f"   - Health: {okx.health_check()}")

            # Stats
            stats = okx.get_stats()
            print(f"   - Stats: {stats}")

            # Close
            await okx.close()

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
