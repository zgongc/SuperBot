"""Account business logic"""
import json
from pathlib import Path
from .base_service import BaseService

class AccountService(BaseService):
    """Account management service"""

    def __init__(self, data_manager, exchange_client, logger):
        super().__init__(data_manager, logger)
        self.exchange_client = exchange_client

    async def get_account_balance(self, account_type='SPOT'):
        """Get account balance from database"""
        account_info = await self.data_manager.get_account_info(
            account_type=account_type
        )
        return account_info

    async def sync_account_balance(self, account_type='SPOT'):
        """
        Sync account balance from exchange

        Flow:
        1. Fetch from exchange
        2. Save to JSON (backup)
        3. Save to database
        """
        # Step 1: Get account balance from exchange
        try:
            account_info = await self.exchange_client.get_balance()
        except Exception as balance_error:
            # Check if API Secret not configured
            if 'API Secret required' in str(balance_error) or 'API-secret' in str(balance_error):
                is_testnet = self.exchange_client.testnet if hasattr(self.exchange_client, 'testnet') else False
                endpoint = 'testnet' if is_testnet else 'production'
                env_vars = 'BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET' if is_testnet else 'BINANCE_API_KEY and BINANCE_API_SECRET'
                raise Exception(f'API keys not configured. Please add {env_vars} to config/.env and verify config/connectors.yaml endpoints.{endpoint}.api_key is set.')
            else:
                raise

        # Step 2: Save to JSON (same pattern as symbols)
        json_dir = Path('data/json')
        json_dir.mkdir(parents=True, exist_ok=True)

        json_filename = f'account_{account_type.lower()}.json'
        json_path = json_dir / json_filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(account_info, f, indent=2, ensure_ascii=False)

        self.logger.info(f'Saved {account_type} account info to {json_path}')

        # Step 3: Save to database
        success = await self.data_manager.save_account_info(
            account_type=account_type,
            account_data=account_info
        )

        return success
