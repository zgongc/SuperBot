#!/usr/bin/env python3
"""
components/datamanager/services/account.py
SuperBot - Account Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Account information management - models and CRUD operations

Features:
- AccountInfo model
- Save/retrieve exchange account snapshots
- Balance tracking and history

Usage:
    from components.datamanager.services.account import AccountService, AccountInfo

    service = AccountService(db_manager)
    await service.save_account_info("SPOT", account_data)
    info = await service.get_account_info("SPOT")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, BigInteger, Boolean, DateTime, Index, JSON, update
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.account")


# ============================================
# MODELS
# ============================================

class AccountInfo(Base):
    """
    Exchange account information snapshot

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "account_info"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Account details
    account_type = Column(String(20), nullable=False)  # SPOT or FUTURES
    user_id = Column(String(50), default="default", index=True)

    # Permissions
    can_trade = Column(Boolean, default=False)
    can_withdraw = Column(Boolean, default=False)
    can_deposit = Column(Boolean, default=False)

    # Balances (JSON array of assets)
    balances = Column(JSON)  # [{"asset": "BTC", "free": 0.5, "locked": 0.1, "total": 0.6}]

    # Summary
    total_assets = Column(Integer, default=0)
    total_btc = Column(Float, default=0.0)
    total_usdt = Column(Float, default=0.0)

    # Timing
    synced_at = Column(DateTime, default=datetime.utcnow, index=True)
    exchange_update_time = Column(BigInteger)  # Exchange timestamp

    # Indexes
    __table_args__ = (
        Index('idx_user_account_type', 'user_id', 'account_type'),
        Index('idx_synced_at', 'synced_at'),
    )


# ============================================
# SERVICE
# ============================================

class AccountService(BaseService):
    """Account information management service"""

    async def save_account_info(
        self,
        account_type: str,
        account_data: Dict[str, Any],
        user_id: str = "default"
    ) -> bool:
        """
        Save account info snapshot to database

        Args:
            account_type: 'SPOT' or 'FUTURES'
            account_data: Account info from exchange (from get_balance())
            user_id: User identifier

        Returns:
            bool: Success status
        """
        # === PORTED FROM data_manager.py lines 1674-1775 ===
        try:
            # Parse balances
            balances = []
            total_btc = 0.0
            total_usdt = 0.0

            if 'balances' in account_data:
                for balance in account_data['balances']:
                    free = float(balance.get('free', 0))
                    locked = float(balance.get('locked', 0))
                    total = free + locked

                    # Only save assets with balance > 0
                    if total > 0.0001:
                        balances.append({
                            'asset': balance['asset'],
                            'free': free,
                            'locked': locked,
                            'total': total
                        })

                        # Track totals
                        if balance['asset'] == 'BTC':
                            total_btc = total
                        elif balance['asset'] == 'USDT':
                            total_usdt = total

            # Sort by total value
            balances.sort(key=lambda x: x['total'], reverse=True)

            async with self.session() as session:
                stmt = select(AccountInfo).where(
                    AccountInfo.user_id == user_id,
                    AccountInfo.account_type == account_type
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing record
                    update_stmt = (
                        update(AccountInfo)
                        .where(
                            AccountInfo.user_id == user_id,
                            AccountInfo.account_type == account_type
                        )
                        .values(
                            balances=balances,
                            can_trade=account_data.get('canTrade', False),
                            can_withdraw=account_data.get('canWithdraw', False),
                            can_deposit=account_data.get('canDeposit', False),
                            total_assets=len(balances),
                            total_btc=total_btc,
                            total_usdt=total_usdt,
                            synced_at=get_utc_now(),
                            exchange_update_time=account_data.get('updateTime', 0)
                        )
                    )
                    await session.execute(update_stmt)
                else:
                    # Insert new record
                    account_info = AccountInfo(
                        account_type=account_type,
                        user_id=user_id,
                        balances=balances,
                        can_trade=account_data.get('canTrade', False),
                        can_withdraw=account_data.get('canWithdraw', False),
                        can_deposit=account_data.get('canDeposit', False),
                        total_assets=len(balances),
                        total_btc=total_btc,
                        total_usdt=total_usdt,
                        exchange_update_time=account_data.get('updateTime', 0)
                    )
                    session.add(account_info)

                await session.commit()
                logger.info(f"✅ Account info saved: {account_type} ({len(balances)} assets)")
                return True

        except Exception as e:
            logger.error(f"❌ Save account info error: {e}")
            return False

    async def get_account_info(
        self,
        account_type: str = "SPOT",
        user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest account info from database

        Args:
            account_type: 'SPOT' or 'FUTURES'
            user_id: User identifier

        Returns:
            Dict with account info or None
        """
        # === PORTED FROM data_manager.py lines 1777-1829 ===
        try:
            async with self.session() as session:
                stmt = (
                    select(AccountInfo)
                    .where(
                        AccountInfo.user_id == user_id,
                        AccountInfo.account_type == account_type
                    )
                    .order_by(AccountInfo.synced_at.desc())
                    .limit(1)
                )

                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if not account:
                    return None

                return {
                    'id': account.id,
                    'account_type': account.account_type,
                    'user_id': account.user_id,
                    'balances': account.balances or [],
                    'can_trade': account.can_trade,
                    'can_withdraw': account.can_withdraw,
                    'can_deposit': account.can_deposit,
                    'summary': {
                        'total_assets': account.total_assets,
                        'total_btc': account.total_btc,
                        'total_usdt': account.total_usdt
                    },
                    'synced_at': account.synced_at.isoformat() if account.synced_at else None,
                    'update_time': account.exchange_update_time
                }

        except Exception as e:
            logger.error(f"❌ Get account info error: {e}")
            return None

    async def get_all_accounts(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Get all account types for a user

        Args:
            user_id: User identifier

        Returns:
            List of account info dicts
        """
        try:
            async with self.session() as session:
                stmt = (
                    select(AccountInfo)
                    .where(AccountInfo.user_id == user_id)
                    .order_by(AccountInfo.account_type)
                )
                result = await session.execute(stmt)
                accounts = result.scalars().all()

                return [
                    {
                        'id': acc.id,
                        'account_type': acc.account_type,
                        'user_id': acc.user_id,
                        'balances': acc.balances or [],
                        'can_trade': acc.can_trade,
                        'can_withdraw': acc.can_withdraw,
                        'can_deposit': acc.can_deposit,
                        'summary': {
                            'total_assets': acc.total_assets,
                            'total_btc': acc.total_btc,
                            'total_usdt': acc.total_usdt
                        },
                        'synced_at': acc.synced_at.isoformat() if acc.synced_at else None,
                        'update_time': acc.exchange_update_time
                    }
                    for acc in accounts
                ]
        except Exception as e:
            logger.error(f"❌ Get all accounts error: {e}")
            return []

    async def get_account_balance(
        self,
        asset: str,
        account_type: str = "SPOT",
        user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific asset balance from account

        Args:
            asset: Asset symbol (BTC, ETH, USDT, etc.)
            account_type: 'SPOT' or 'FUTURES'
            user_id: User identifier

        Returns:
            Balance dict or None
        """
        try:
            account = await self.get_account_info(account_type, user_id)
            if not account:
                return None

            for balance in account.get('balances', []):
                if balance['asset'] == asset:
                    return balance

            return None
        except Exception as e:
            logger.error(f"❌ Get account balance error: {e}")
            return None

    async def delete_account_info(
        self,
        account_type: str,
        user_id: str = "default"
    ) -> bool:
        """
        Delete account info record

        Args:
            account_type: 'SPOT' or 'FUTURES'
            user_id: User identifier

        Returns:
            bool: Success status
        """
        try:
            async with self.session() as session:
                stmt = select(AccountInfo).where(
                    AccountInfo.user_id == user_id,
                    AccountInfo.account_type == account_type
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account:
                    await session.delete(account)
                    await session.commit()
                    logger.info(f"✅ Deleted account info: {account_type} for {user_id}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete account info error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 AccountService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = AccountService(db)

        # Test 1: Save SPOT account info
        print("\n1️⃣ Save SPOT account info")
        spot_data = {
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 1705900000000,
            "balances": [
                {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                {"asset": "ETH", "free": "10.0", "locked": "2.0"},
                {"asset": "USDT", "free": "50000.0", "locked": "10000.0"},
                {"asset": "BNB", "free": "100.0", "locked": "0"},
                {"asset": "DOGE", "free": "0.00001", "locked": "0"},  # Should be filtered
            ]
        }
        success = await service.save_account_info("SPOT", spot_data, "user1")
        print(f"   ✅ SPOT saved: {success}")

        # Test 2: Save FUTURES account info
        print("\n2️⃣ Save FUTURES account info")
        futures_data = {
            "canTrade": True,
            "canWithdraw": False,
            "canDeposit": True,
            "updateTime": 1705900001000,
            "balances": [
                {"asset": "USDT", "free": "25000.0", "locked": "5000.0"},
                {"asset": "BNB", "free": "50.0", "locked": "10.0"},
            ]
        }
        success = await service.save_account_info("FUTURES", futures_data, "user1")
        print(f"   ✅ FUTURES saved: {success}")

        # Test 3: Get SPOT account info
        print("\n3️⃣ Get SPOT account info")
        spot = await service.get_account_info("SPOT", "user1")
        if spot:
            print(f"   ✅ Account type: {spot['account_type']}")
            print(f"   ✅ Can trade: {spot['can_trade']}")
            print(f"   ✅ Total assets: {spot['summary']['total_assets']}")
            print(f"   ✅ Total BTC: {spot['summary']['total_btc']}")
            print(f"   ✅ Total USDT: {spot['summary']['total_usdt']}")
            print(f"   ✅ Balances count: {len(spot['balances'])}")

        # Test 4: Get all accounts for user
        print("\n4️⃣ Get all accounts for user1")
        accounts = await service.get_all_accounts("user1")
        print(f"   ✅ Found {len(accounts)} accounts")
        for acc in accounts:
            print(f"      - {acc['account_type']}: {acc['summary']['total_assets']} assets")

        # Test 5: Get specific asset balance
        print("\n5️⃣ Get specific asset balance (BTC)")
        btc_balance = await service.get_account_balance("BTC", "SPOT", "user1")
        if btc_balance:
            print(f"   ✅ BTC balance: free={btc_balance['free']}, locked={btc_balance['locked']}, total={btc_balance['total']}")

        # Test 6: Update existing account (upsert)
        print("\n6️⃣ Update SPOT account (upsert)")
        updated_spot_data = {
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 1705900002000,
            "balances": [
                {"asset": "BTC", "free": "2.0", "locked": "0.5"},
                {"asset": "ETH", "free": "15.0", "locked": "3.0"},
                {"asset": "USDT", "free": "75000.0", "locked": "15000.0"},
            ]
        }
        success = await service.save_account_info("SPOT", updated_spot_data, "user1")
        print(f"   ✅ SPOT updated: {success}")

        spot = await service.get_account_info("SPOT", "user1")
        if spot:
            print(f"   ✅ New Total USDT: {spot['summary']['total_usdt']}")

        # Test 7: Get non-existent asset
        print("\n7️⃣ Get non-existent asset (XRP)")
        xrp_balance = await service.get_account_balance("XRP", "SPOT", "user1")
        print(f"   ✅ XRP balance: {xrp_balance}")

        # Test 8: Multiple users
        print("\n8️⃣ Multiple users test")
        user2_data = {
            "canTrade": True,
            "canWithdraw": False,
            "canDeposit": True,
            "updateTime": 1705900003000,
            "balances": [
                {"asset": "USDT", "free": "1000.0", "locked": "0"},
            ]
        }
        await service.save_account_info("SPOT", user2_data, "user2")
        user2_spot = await service.get_account_info("SPOT", "user2")
        print(f"   ✅ User2 SPOT USDT: {user2_spot['summary']['total_usdt'] if user2_spot else 'N/A'}")

        # Test 9: Delete account info
        print("\n9️⃣ Delete FUTURES account")
        success = await service.delete_account_info("FUTURES", "user1")
        print(f"   ✅ Deleted: {success}")

        accounts = await service.get_all_accounts("user1")
        print(f"   ✅ Remaining accounts: {len(accounts)}")

        # Test 10: Non-existent user
        print("\n🔟 Non-existent user")
        no_account = await service.get_account_info("SPOT", "nonexistent")
        print(f"   ✅ Non-existent account: {no_account}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
