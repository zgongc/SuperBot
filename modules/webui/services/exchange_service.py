"""
modules/webui/services/exchange_service.py
SuperBot - Exchange Account Service Layer
Author: SuperBot Team
Date: 2025-10-30
Versiyon: 1.0.0

Exchange account business logic and operations.
"""

from typing import Dict, List, Any, Optional
from components.datamanager import DataManager
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class ExchangeService:
    """Exchange account service"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    # ============================================
    # Exchange Account CRUD
    # ============================================

    async def get_all_exchange_accounts(self) -> List[Dict[str, Any]]:
        """Get all exchange accounts (without sensitive data)"""
        try:
            accounts = await self.data_manager.get_all_exchange_accounts()

            # Add portfolio count for each account
            for account in accounts:
                portfolios = await self.data_manager.get_all_portfolios()
                account['portfolio_count'] = len([
                    p for p in portfolios
                    if p.get('exchange_account_id') == account['id']
                ])

            return accounts
        except Exception as e:
            logger.error(f"❌ Get all exchange accounts error: {e}")
            return []

    async def get_exchange_account_by_id(self, account_id: int) -> Optional[Dict[str, Any]]:
        """Get exchange account by ID (includes API credentials)"""
        try:
            account = await self.data_manager.get_exchange_account_by_id(account_id)

            if account:
                # Add portfolio count
                portfolios = await self.data_manager.get_all_portfolios()
                account['portfolio_count'] = len([
                    p for p in portfolios
                    if p.get('exchange_account_id') == account_id
                ])

            return account
        except Exception as e:
            logger.error(f"❌ Get exchange account by ID error: {e}")
            return None

    async def create_exchange_account(
        self,
        name: str,
        exchange: str,
        environment: str = 'production',
        account_type: str = 'spot',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> Optional[int]:
        """Create new exchange account"""
        try:
            # TODO: Encrypt API credentials before saving
            # Will be implemented with security module

            account_id = await self.data_manager.create_exchange_account(
                name=name,
                exchange=exchange,
                environment=environment,
                account_type=account_type,
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                settings=settings,
                notes=notes
            )

            if account_id:
                logger.info(f"✅ Exchange account created: {name} ({exchange})")

            return account_id
        except Exception as e:
            logger.error(f"❌ Create exchange account error: {e}")
            return None

    async def update_exchange_account(
        self,
        account_id: int,
        **kwargs
    ) -> bool:
        """Update exchange account"""
        try:
            # TODO: Encrypt API credentials if they are being updated

            success = await self.data_manager.update_exchange_account(
                account_id,
                **kwargs
            )

            if success:
                logger.info(f"✅ Exchange account updated: {account_id}")

            return success
        except Exception as e:
            logger.error(f"❌ Update exchange account error: {e}")
            return False

    async def delete_exchange_account(self, account_id: int) -> Dict[str, Any]:
        """Delete exchange account and its linked portfolios (cascade delete)"""
        try:
            # Find all portfolios linked to this exchange
            portfolios = await self.data_manager.get_all_portfolios()
            linked_portfolios = [
                p for p in portfolios
                if p.get('exchange_account_id') == account_id
            ]

            # Delete linked portfolios first (cascade delete)
            for portfolio in linked_portfolios:
                portfolio_id = portfolio.get('id')
                logger.info(f"🗑️ Deleting linked portfolio: {portfolio.get('name')} (ID: {portfolio_id})")
                await self.data_manager.delete_portfolio(portfolio_id)

            # Now delete the exchange account
            success = await self.data_manager.delete_exchange_account(account_id)

            if success:
                logger.info(f"✅ Exchange account deleted: {account_id} (with {len(linked_portfolios)} portfolio(s))")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Delete failed'}

        except Exception as e:
            logger.error(f"❌ Delete exchange account error: {e}")
            return {'success': False, 'error': str(e)}

    # ============================================
    # Exchange Connection & Testing
    # ============================================

    async def test_connection(self, account_id: int) -> Dict[str, Any]:
        """Test exchange connection"""
        try:
            account = await self.data_manager.get_exchange_account_by_id(account_id)
            if not account:
                return {'success': False, 'error': 'Account not found'}

            # TODO: Implement actual exchange connection test
            # This will be implemented in Phase 7.5 with exchange clients
            # For now, just validate credentials exist

            if not account.get('api_key') or not account.get('api_secret'):
                return {
                    'success': False,
                    'error': 'API credentials not configured'
                }

            # Simulate connection test (Phase 7.5 will implement real test)
            result = await self.data_manager.test_exchange_connection(account_id)

            return result
        except Exception as e:
            logger.error(f"❌ Test connection error: {e}")
            return {'success': False, 'error': str(e)}

    async def get_exchange_balance(self, account_id: int) -> Dict[str, Any]:
        """Get exchange account balance (Phase 7.5)"""
        try:
            account = await self.data_manager.get_exchange_account_by_id(account_id)
            if not account:
                return {'success': False, 'error': 'Account not found'}

            # TODO: Implement exchange API client integration
            # This will fetch real balance from exchange

            return {
                'success': False,
                'error': 'Exchange integration not yet implemented (Phase 7.5)'
            }
        except Exception as e:
            logger.error(f"❌ Get exchange balance error: {e}")
            return {'success': False, 'error': str(e)}

    # ============================================
    # Import/Sync Operations (Phase 7.5)
    # ============================================

    async def sync_positions(self, account_id: int, portfolio_id: int) -> Dict[str, Any]:
        """Sync positions from exchange to portfolio (Phase 7.5)"""
        try:
            # TODO: Implement exchange position syncing
            # 1. Fetch positions from exchange
            # 2. Compare with DB positions
            # 3. Calculate differences
            # 4. Return import preview

            return {
                'success': False,
                'error': 'Exchange integration not yet implemented (Phase 7.5)'
            }
        except Exception as e:
            logger.error(f"❌ Sync positions error: {e}")
            return {'success': False, 'error': str(e)}

    # ============================================
    # Statistics & Analytics
    # ============================================

    async def get_exchange_accounts_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all exchange accounts"""
        try:
            accounts = await self.data_manager.get_all_exchange_accounts()
            portfolios = await self.data_manager.get_all_portfolios()

            # Count by exchange
            exchange_counts = {}
            for account in accounts:
                exchange = account['exchange']
                exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1

            # Count by environment
            env_counts = {
                'production': len([a for a in accounts if a['environment'] == 'production']),
                'testnet': len([a for a in accounts if a['environment'] == 'testnet'])
            }

            # Connection status
            status_counts = {
                'connected': len([a for a in accounts if a['connection_status'] == 'connected']),
                'disconnected': len([a for a in accounts if a['connection_status'] == 'disconnected']),
                'error': len([a for a in accounts if a['connection_status'] == 'error'])
            }

            return {
                'total_accounts': len(accounts),
                'active_accounts': len([a for a in accounts if a['is_active']]),
                'total_portfolios': len(portfolios),
                'exchange_counts': exchange_counts,
                'environment_counts': env_counts,
                'connection_status': status_counts
            }
        except Exception as e:
            logger.error(f"❌ Get exchange accounts summary error: {e}")
            return {
                'total_accounts': 0,
                'active_accounts': 0,
                'total_portfolios': 0,
                'exchange_counts': {},
                'environment_counts': {},
                'connection_status': {}
            }


# Service instance is created and stored in Flask app context (app.exchange_service)
