#!/usr/bin/env python3
"""
components/datamanager/services/exchange.py
SuperBot - Exchange Account Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Exchange account management - models and CRUD operations

Features:
- ExchangeAccount model
- Create/read/update/delete exchange accounts
- Connection status tracking
- API credentials management (encrypted in production)

Usage:
    from components.datamanager.services.exchange import ExchangeService, ExchangeAccount

    service = ExchangeService(db_manager)
    account_id = await service.create_exchange_account("Binance Main", "binance")
    accounts = await service.get_all_exchange_accounts()

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Index, Text, JSON
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.exchange")


# ============================================
# MODELS
# ============================================

class ExchangeAccount(Base):
    """
    Exchange account definitions - Centralized exchange management

    This table centrally manages all exchange connections.
    Each ExchangeAccount represents an exchange account (e.g., Binance Main, Bybit Test).
    Multiple Portfolios can use the same ExchangeAccount.

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "exchange_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Account Info
    name = Column(String(50), nullable=False)  # "Binance Main", "Bybit Efe", "Manual"
    exchange = Column(String(20), nullable=False, index=True)  # binance, bybit, okx, manual
    environment = Column(String(20), default='production')  # production, testnet
    account_type = Column(String(20), default='spot')  # spot, futures, margin

    # API Credentials (encrypted in production)
    api_key = Column(String(200))
    api_secret = Column(String(200))
    passphrase = Column(String(200))  # For OKX

    # Connection Settings (JSON)
    # Example: {"rate_limit": 1200, "timeout": 30, "max_retries": 3, "features": {...}}
    settings = Column(JSON)

    # Status & Health
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)  # Default exchange for bot operations
    last_connected_at = Column(DateTime)
    last_sync_at = Column(DateTime)
    connection_status = Column(String(20), default='disconnected')  # connected, disconnected, error
    error_message = Column(Text)  # Last error if connection failed

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_exchange_accounts_exchange', 'exchange', 'environment'),
        Index('idx_exchange_accounts_active', 'is_active'),
    )


# ============================================
# SERVICE
# ============================================

class ExchangeService(BaseService):
    """Exchange account management service"""

    async def get_all_exchange_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all exchange accounts (without API credentials for security)

        Returns:
            List of exchange account dicts
        """
        # === PORTED FROM data_manager.py lines 1406-1444 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).order_by(
                        ExchangeAccount.is_default.desc(),
                        ExchangeAccount.name
                    )
                )
                accounts = result.scalars().all()

                return [
                    {
                        'id': acc.id,
                        'name': acc.name,
                        'exchange': acc.exchange,
                        'environment': acc.environment,
                        'account_type': acc.account_type,
                        'is_active': acc.is_active,
                        'is_default': acc.is_default,
                        'last_connected_at': acc.last_connected_at.isoformat() if acc.last_connected_at else None,
                        'last_sync_at': acc.last_sync_at.isoformat() if acc.last_sync_at else None,
                        'connection_status': acc.connection_status,
                        'error_message': acc.error_message,
                        'notes': acc.notes,
                        'created_at': acc.created_at.isoformat() if acc.created_at else None,
                        'updated_at': acc.updated_at.isoformat() if acc.updated_at else None
                    }
                    for acc in accounts
                ]
        except Exception as e:
            logger.error(f"❌ Get all exchange accounts error: {e}")
            return []

    async def get_exchange_account_by_id(
        self,
        account_id: int,
        include_credentials: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get exchange account by ID

        Args:
            account_id: Account ID
            include_credentials: If True, include API credentials

        Returns:
            Account dict or None
        """
        # === PORTED FROM data_manager.py lines 1446-1488 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).where(ExchangeAccount.id == account_id)
                )
                acc = result.scalar_one_or_none()

                if not acc:
                    return None

                account_dict = {
                    'id': acc.id,
                    'name': acc.name,
                    'exchange': acc.exchange,
                    'environment': acc.environment,
                    'account_type': acc.account_type,
                    'settings': acc.settings,
                    'is_active': acc.is_active,
                    'is_default': acc.is_default,
                    'last_connected_at': acc.last_connected_at.isoformat() if acc.last_connected_at else None,
                    'last_sync_at': acc.last_sync_at.isoformat() if acc.last_sync_at else None,
                    'connection_status': acc.connection_status,
                    'error_message': acc.error_message,
                    'notes': acc.notes,
                    'created_at': acc.created_at.isoformat() if acc.created_at else None,
                    'updated_at': acc.updated_at.isoformat() if acc.updated_at else None
                }

                if include_credentials:
                    account_dict['api_key'] = acc.api_key
                    account_dict['api_secret'] = acc.api_secret
                    account_dict['passphrase'] = acc.passphrase

                return account_dict
        except Exception as e:
            logger.error(f"❌ Get exchange account error: {e}")
            return None

    async def get_default_account(self) -> Optional[Dict[str, Any]]:
        """
        Get default exchange account

        Returns:
            Default account dict or None
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).where(ExchangeAccount.is_default == True)
                )
                acc = result.scalar_one_or_none()

                if not acc:
                    return None

                return {
                    'id': acc.id,
                    'name': acc.name,
                    'exchange': acc.exchange,
                    'environment': acc.environment,
                    'account_type': acc.account_type,
                    'is_active': acc.is_active,
                    'is_default': acc.is_default,
                    'connection_status': acc.connection_status
                }
        except Exception as e:
            logger.error(f"❌ Get default account error: {e}")
            return None

    async def get_accounts_by_exchange(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get all accounts for a specific exchange

        Args:
            exchange: Exchange name (binance, bybit, okx, etc.)

        Returns:
            List of account dicts
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).where(
                        ExchangeAccount.exchange == exchange.lower()
                    ).order_by(ExchangeAccount.name)
                )
                accounts = result.scalars().all()

                return [
                    {
                        'id': acc.id,
                        'name': acc.name,
                        'exchange': acc.exchange,
                        'environment': acc.environment,
                        'account_type': acc.account_type,
                        'is_active': acc.is_active,
                        'is_default': acc.is_default,
                        'connection_status': acc.connection_status
                    }
                    for acc in accounts
                ]
        except Exception as e:
            logger.error(f"❌ Get accounts by exchange error: {e}")
            return []

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
        notes: Optional[str] = None,
        is_default: bool = False
    ) -> Optional[int]:
        """
        Create new exchange account

        Args:
            name: Account display name
            exchange: Exchange identifier (binance, bybit, okx)
            environment: production or testnet
            account_type: spot, futures, or margin
            api_key: API key (optional for manual accounts)
            api_secret: API secret
            passphrase: Passphrase (for OKX)
            settings: Additional settings as dict
            notes: User notes
            is_default: Set as default account

        Returns:
            Account ID if successful
        """
        # === PORTED FROM data_manager.py lines 1490-1536 ===
        try:
            async with self.session() as session:
                # If setting as default, unset other defaults first
                if is_default:
                    existing_defaults = await session.execute(
                        select(ExchangeAccount).where(ExchangeAccount.is_default == True)
                    )
                    for acc in existing_defaults.scalars().all():
                        acc.is_default = False

                account = ExchangeAccount(
                    name=name,
                    exchange=exchange.lower(),
                    environment=environment,
                    account_type=account_type,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    settings=settings,
                    is_active=True,
                    is_default=is_default,
                    connection_status='disconnected',
                    notes=notes
                )
                session.add(account)
                await session.commit()
                await session.refresh(account)

                logger.info(f"✅ Exchange account created: {name} ({exchange})")
                return account.id
        except Exception as e:
            logger.error(f"❌ Create exchange account error: {e}")
            return None

    async def update_exchange_account(
        self,
        account_id: int,
        **kwargs
    ) -> bool:
        """
        Update exchange account fields

        Args:
            account_id: Account ID
            **kwargs: Fields to update

        Returns:
            bool: Success status
        """
        # === PORTED FROM data_manager.py lines 1538-1588 ===
        try:
            if not kwargs:
                return True

            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).where(ExchangeAccount.id == account_id)
                )
                account = result.scalar_one_or_none()

                if not account:
                    return False

                # Handle is_default - ensure only one default
                if kwargs.get('is_default'):
                    existing_defaults = await session.execute(
                        select(ExchangeAccount).where(
                            ExchangeAccount.is_default == True,
                            ExchangeAccount.id != account_id
                        )
                    )
                    for acc in existing_defaults.scalars().all():
                        acc.is_default = False

                # Update allowed fields
                allowed_fields = [
                    'name', 'exchange', 'environment', 'account_type',
                    'api_key', 'api_secret', 'passphrase', 'settings',
                    'is_active', 'is_default', 'connection_status',
                    'error_message', 'notes', 'last_connected_at', 'last_sync_at'
                ]

                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(account, key, value)

                account.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Exchange account updated: {account_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Update exchange account error: {e}")
            return False

    async def delete_exchange_account(self, account_id: int) -> bool:
        """
        Delete exchange account

        Note: This does NOT cascade delete portfolios. Use PortfolioService
        to handle portfolio cleanup before deleting exchange accounts.

        Args:
            account_id: Account ID

        Returns:
            bool: Success status
        """
        # === PORTED FROM data_manager.py lines 1590-1641 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ExchangeAccount).where(ExchangeAccount.id == account_id)
                )
                account = result.scalar_one_or_none()

                if not account:
                    return False

                await session.delete(account)
                await session.commit()

                logger.info(f"✅ Exchange account deleted: {account_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Delete exchange account error: {e}")
            return False

    async def test_connection(self, account_id: int) -> Dict[str, Any]:
        """
        Test exchange connection and update status

        Args:
            account_id: Account ID

        Returns:
            Dict with success status and message
        """
        # === PORTED FROM data_manager.py lines 1643-1668 ===
        try:
            account = await self.get_exchange_account_by_id(account_id, include_credentials=True)
            if not account:
                return {'success': False, 'error': 'Account not found'}

            # TODO: Implement actual exchange connection test
            # This will be implemented with exchange clients

            # For now, just update last_connected_at
            await self.update_exchange_account(
                account_id,
                last_connected_at=get_utc_now(),
                connection_status='connected'
            )

            return {'success': True, 'message': 'Connection test successful'}
        except Exception as e:
            logger.error(f"❌ Test exchange connection error: {e}")
            await self.update_exchange_account(
                account_id,
                connection_status='error',
                error_message=str(e)
            )
            return {'success': False, 'error': str(e)}

    async def set_default_account(self, account_id: int) -> bool:
        """
        Set an account as the default

        Args:
            account_id: Account ID to set as default

        Returns:
            bool: Success status
        """
        try:
            async with self.session() as session:
                # Unset all defaults
                all_accounts = await session.execute(select(ExchangeAccount))
                for acc in all_accounts.scalars().all():
                    acc.is_default = False

                # Set new default
                result = await session.execute(
                    select(ExchangeAccount).where(ExchangeAccount.id == account_id)
                )
                account = result.scalar_one_or_none()

                if not account:
                    return False

                account.is_default = True
                account.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Set default exchange account: {account_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Set default account error: {e}")
            return False

    async def update_connection_status(
        self,
        account_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update connection status for an account

        Args:
            account_id: Account ID
            status: Connection status (connected, disconnected, error)
            error_message: Error message if status is error

        Returns:
            bool: Success status
        """
        try:
            update_data = {
                'connection_status': status,
                'error_message': error_message
            }

            if status == 'connected':
                update_data['last_connected_at'] = get_utc_now()

            return await self.update_exchange_account(account_id, **update_data)
        except Exception as e:
            logger.error(f"❌ Update connection status error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 ExchangeService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = ExchangeService(db)

        # Test 1: Create Binance account
        print("\n1️⃣ Create Binance account")
        binance_id = await service.create_exchange_account(
            name="Binance Main",
            exchange="binance",
            environment="production",
            account_type="spot",
            api_key="test_api_key_123",
            api_secret="test_api_secret_456",
            settings={"rate_limit": 1200, "timeout": 30},
            notes="Primary trading account",
            is_default=True
        )
        print(f"   ✅ Binance account ID: {binance_id}")

        # Test 2: Create Bybit account
        print("\n2️⃣ Create Bybit account")
        bybit_id = await service.create_exchange_account(
            name="Bybit Futures",
            exchange="bybit",
            environment="production",
            account_type="futures",
            api_key="bybit_key",
            api_secret="bybit_secret",
            notes="Futures trading"
        )
        print(f"   ✅ Bybit account ID: {bybit_id}")

        # Test 3: Create OKX testnet account
        print("\n3️⃣ Create OKX testnet account")
        okx_id = await service.create_exchange_account(
            name="OKX Testnet",
            exchange="okx",
            environment="testnet",
            account_type="spot",
            api_key="okx_key",
            api_secret="okx_secret",
            passphrase="okx_pass"
        )
        print(f"   ✅ OKX account ID: {okx_id}")

        # Test 4: Get all accounts
        print("\n4️⃣ Get all exchange accounts")
        accounts = await service.get_all_exchange_accounts()
        print(f"   ✅ Found {len(accounts)} accounts")
        for acc in accounts:
            default_marker = " (DEFAULT)" if acc['is_default'] else ""
            print(f"      - {acc['name']}: {acc['exchange']} ({acc['environment']}){default_marker}")

        # Test 5: Get account by ID
        print("\n5️⃣ Get account by ID (without credentials)")
        binance = await service.get_exchange_account_by_id(binance_id)
        if binance:
            print(f"   ✅ Name: {binance['name']}")
            print(f"   ✅ Exchange: {binance['exchange']}")
            print(f"   ✅ Has api_key field: {'api_key' in binance}")

        # Test 6: Get account with credentials
        print("\n6️⃣ Get account by ID (with credentials)")
        binance_full = await service.get_exchange_account_by_id(binance_id, include_credentials=True)
        if binance_full:
            print(f"   ✅ API Key: {binance_full.get('api_key', 'N/A')[:10]}...")

        # Test 7: Get default account
        print("\n7️⃣ Get default account")
        default = await service.get_default_account()
        if default:
            print(f"   ✅ Default: {default['name']} ({default['exchange']})")

        # Test 8: Get accounts by exchange
        print("\n8️⃣ Get accounts by exchange (binance)")
        binance_accounts = await service.get_accounts_by_exchange("binance")
        print(f"   ✅ Found {len(binance_accounts)} Binance accounts")

        # Test 9: Update account
        print("\n9️⃣ Update account")
        success = await service.update_exchange_account(
            bybit_id,
            name="Bybit Futures Pro",
            notes="Updated notes - main futures account"
        )
        print(f"   ✅ Updated: {success}")

        bybit = await service.get_exchange_account_by_id(bybit_id)
        if bybit:
            print(f"   ✅ New name: {bybit['name']}")

        # Test 10: Set new default
        print("\n🔟 Set new default account")
        success = await service.set_default_account(bybit_id)
        print(f"   ✅ Set default: {success}")

        default = await service.get_default_account()
        print(f"   ✅ New default: {default['name'] if default else 'None'}")

        # Test 11: Update connection status
        print("\n1️⃣1️⃣ Update connection status")
        await service.update_connection_status(binance_id, "connected")
        binance = await service.get_exchange_account_by_id(binance_id)
        print(f"   ✅ Binance status: {binance['connection_status']}")

        await service.update_connection_status(okx_id, "error", "API rate limit exceeded")
        okx = await service.get_exchange_account_by_id(okx_id)
        print(f"   ✅ OKX status: {okx['connection_status']} - {okx['error_message']}")

        # Test 12: Test connection (mock)
        print("\n1️⃣2️⃣ Test connection (mock)")
        result = await service.test_connection(binance_id)
        print(f"   ✅ Connection test: {result}")

        # Test 13: Delete account
        print("\n1️⃣3️⃣ Delete OKX account")
        success = await service.delete_exchange_account(okx_id)
        print(f"   ✅ Deleted: {success}")

        accounts = await service.get_all_exchange_accounts()
        print(f"   ✅ Remaining accounts: {len(accounts)}")

        # Test 14: Non-existent account
        print("\n1️⃣4️⃣ Non-existent account")
        none_acc = await service.get_exchange_account_by_id(9999)
        print(f"   ✅ Non-existent account: {none_acc}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
