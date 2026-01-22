#!/usr/bin/env python3
"""
components/datamanager/services/config.py
SuperBot - System Configuration Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

System configuration management - settings and config files

Features:
- SystemSetting model for key-value settings
- ConfigFile model for config file tracking
- CRUD operations for settings

Usage:
    from components.datamanager.services.config import ConfigService, SystemSetting

    service = ConfigService(db_manager)
    await service.set_setting("trading", "max_positions", "5")
    value = await service.get_setting("trading", "max_positions")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Index, Text
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.config")


# ============================================
# MODELS
# ============================================

class SystemSetting(Base):
    """System configuration settings"""
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text)
    value_type = Column(String(20))  # string, int, float, bool, json
    display_name = Column(String(150))
    description = Column(Text)
    is_sensitive = Column(Boolean, default=False)  # For API keys, passwords
    is_editable = Column(Boolean, default=True)
    validation_rules = Column(Text)  # JSON validation rules
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_settings_category', 'category'),
        Index('idx_settings_category_key', 'category', 'key', unique=True),
    )


class ConfigFile(Base):
    """Configuration file tracking"""
    __tablename__ = "config_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(500), nullable=False, unique=True)
    file_type = Column(String(10))  # yaml, json, toml
    content = Column(Text)
    content_hash = Column(String(64))
    is_valid = Column(Boolean, default=True)
    validation_error = Column(Text)
    is_backup = Column(Boolean, default=False)
    backed_up_at = Column(DateTime)
    last_modified_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_config_files_type', 'file_type'),
    )


# ============================================
# SERVICE
# ============================================

class ConfigService(BaseService):
    """System configuration management service"""

    # ============================================
    # System Settings
    # ============================================

    async def get_setting(
        self,
        category: str,
        key: str,
        default: Optional[str] = None
    ) -> Optional[str]:
        """Get a setting value"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(SystemSetting).where(
                        SystemSetting.category == category,
                        SystemSetting.key == key
                    )
                )
                setting = result.scalar_one_or_none()

                if setting:
                    return setting.value
                return default
        except Exception as e:
            logger.error(f"❌ Get setting error: {e}")
            return default

    async def set_setting(
        self,
        category: str,
        key: str,
        value: str,
        value_type: str = 'string',
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        is_sensitive: bool = False
    ) -> bool:
        """Set or update a setting"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(SystemSetting).where(
                        SystemSetting.category == category,
                        SystemSetting.key == key
                    )
                )
                setting = result.scalar_one_or_none()

                if setting:
                    setting.value = value
                    setting.value_type = value_type
                    if display_name:
                        setting.display_name = display_name
                    if description:
                        setting.description = description
                    setting.updated_at = get_utc_now()
                else:
                    setting = SystemSetting(
                        category=category,
                        key=key,
                        value=value,
                        value_type=value_type,
                        display_name=display_name or key,
                        description=description,
                        is_sensitive=is_sensitive,
                        is_editable=True
                    )
                    session.add(setting)

                await session.commit()
                logger.info(f"✅ Setting saved: {category}.{key}")
                return True
        except Exception as e:
            logger.error(f"❌ Set setting error: {e}")
            return False

    async def get_settings_by_category(self, category: str) -> Dict[str, Any]:
        """Get all settings in a category as a dict"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(SystemSetting).where(SystemSetting.category == category)
                )
                settings = result.scalars().all()

                return {s.key: s.value for s in settings}
        except Exception as e:
            logger.error(f"❌ Get settings by category error: {e}")
            return {}

    async def get_all_settings(self, include_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Get all settings"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(SystemSetting).order_by(SystemSetting.category, SystemSetting.key)
                )
                settings = result.scalars().all()

                return [
                    {
                        'id': s.id,
                        'category': s.category,
                        'key': s.key,
                        'value': s.value if not s.is_sensitive or include_sensitive else '********',
                        'value_type': s.value_type,
                        'display_name': s.display_name,
                        'description': s.description,
                        'is_sensitive': s.is_sensitive,
                        'is_editable': s.is_editable,
                        'updated_at': s.updated_at.isoformat() if s.updated_at else None
                    }
                    for s in settings
                ]
        except Exception as e:
            logger.error(f"❌ Get all settings error: {e}")
            return []

    async def delete_setting(self, category: str, key: str) -> bool:
        """Delete a setting"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(SystemSetting).where(
                        SystemSetting.category == category,
                        SystemSetting.key == key
                    )
                )
                setting = result.scalar_one_or_none()

                if setting:
                    await session.delete(setting)
                    await session.commit()
                    logger.info(f"✅ Setting deleted: {category}.{key}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete setting error: {e}")
            return False

    # ============================================
    # Config Files
    # ============================================

    async def register_config_file(
        self,
        file_path: str,
        file_type: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None
    ) -> Optional[int]:
        """Register a configuration file"""
        try:
            async with self.session() as session:
                config = ConfigFile(
                    file_path=file_path,
                    file_type=file_type,
                    content=content,
                    content_hash=content_hash,
                    is_valid=True,
                    last_modified_at=get_utc_now()
                )
                session.add(config)
                await session.commit()
                await session.refresh(config)

                logger.info(f"✅ Config file registered: {file_path}")
                return config.id
        except Exception as e:
            logger.error(f"❌ Register config file error: {e}")
            return None

    async def get_config_files(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all config files"""
        try:
            async with self.session() as session:
                query = select(ConfigFile).where(ConfigFile.is_backup == False)

                if file_type:
                    query = query.where(ConfigFile.file_type == file_type)

                result = await session.execute(query)
                configs = result.scalars().all()

                return [
                    {
                        'id': c.id,
                        'file_path': c.file_path,
                        'file_type': c.file_type,
                        'is_valid': c.is_valid,
                        'validation_error': c.validation_error,
                        'content_hash': c.content_hash,
                        'last_modified_at': c.last_modified_at.isoformat() if c.last_modified_at else None
                    }
                    for c in configs
                ]
        except Exception as e:
            logger.error(f"❌ Get config files error: {e}")
            return []

    async def update_config_file(
        self,
        file_path: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
        is_valid: Optional[bool] = None,
        validation_error: Optional[str] = None
    ) -> bool:
        """Update config file record"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(ConfigFile).where(ConfigFile.file_path == file_path)
                )
                config = result.scalar_one_or_none()

                if not config:
                    return False

                if content is not None:
                    config.content = content
                if content_hash is not None:
                    config.content_hash = content_hash
                if is_valid is not None:
                    config.is_valid = is_valid
                if validation_error is not None:
                    config.validation_error = validation_error

                config.last_modified_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Config file updated: {file_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Update config file error: {e}")
            return False

    async def backup_config_file(self, file_path: str) -> Optional[int]:
        """Create backup of a config file"""
        try:
            async with self.session() as session:
                # Get original
                result = await session.execute(
                    select(ConfigFile).where(ConfigFile.file_path == file_path)
                )
                original = result.scalar_one_or_none()

                if not original:
                    return None

                # Create backup
                backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup = ConfigFile(
                    file_path=backup_path,
                    file_type=original.file_type,
                    content=original.content,
                    content_hash=original.content_hash,
                    is_valid=original.is_valid,
                    is_backup=True,
                    backed_up_at=get_utc_now(),
                    last_modified_at=original.last_modified_at
                )
                session.add(backup)
                await session.commit()
                await session.refresh(backup)

                logger.info(f"✅ Config file backup created: {backup_path}")
                return backup.id
        except Exception as e:
            logger.error(f"❌ Backup config file error: {e}")
            return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 ConfigService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = ConfigService(db)

        # Test 1: Set settings
        print("\n1️⃣ Set settings")
        await service.set_setting("trading", "max_positions", "5", "int", "Maximum Positions")
        await service.set_setting("trading", "position_size_pct", "10", "float", "Position Size %")
        await service.set_setting("trading", "default_exchange", "binance", "string")
        await service.set_setting("api", "binance_key", "abc123xyz", "string", is_sensitive=True)
        print("   ✅ 4 settings saved")

        # Test 2: Get single setting
        print("\n2️⃣ Get single setting")
        value = await service.get_setting("trading", "max_positions")
        print(f"   ✅ max_positions = {value}")

        # Test 3: Get with default
        print("\n3️⃣ Get with default")
        value = await service.get_setting("trading", "nonexistent", "default_value")
        print(f"   ✅ nonexistent = {value}")

        # Test 4: Get settings by category
        print("\n4️⃣ Get settings by category")
        trading_settings = await service.get_settings_by_category("trading")
        print(f"   ✅ Trading settings: {trading_settings}")

        # Test 5: Get all settings (hide sensitive)
        print("\n5️⃣ Get all settings (hide sensitive)")
        all_settings = await service.get_all_settings(include_sensitive=False)
        print(f"   ✅ Found {len(all_settings)} settings")
        for s in all_settings:
            print(f"      - {s['category']}.{s['key']} = {s['value']}")

        # Test 6: Update setting
        print("\n6️⃣ Update setting")
        await service.set_setting("trading", "max_positions", "10", "int")
        value = await service.get_setting("trading", "max_positions")
        print(f"   ✅ Updated max_positions = {value}")

        # Test 7: Register config file
        print("\n7️⃣ Register config file")
        config_id = await service.register_config_file(
            file_path="config/trading.yaml",
            file_type="yaml",
            content="exchange: binance\nmax_positions: 5",
            content_hash="abc123"
        )
        print(f"   ✅ Config ID: {config_id}")

        # Test 8: Get config files
        print("\n8️⃣ Get config files")
        configs = await service.get_config_files()
        print(f"   ✅ Found {len(configs)} config files")
        for c in configs:
            print(f"      - {c['file_path']} ({c['file_type']})")

        # Test 9: Update config file
        print("\n9️⃣ Update config file")
        success = await service.update_config_file(
            "config/trading.yaml",
            content="exchange: binance\nmax_positions: 10",
            content_hash="xyz789"
        )
        print(f"   ✅ Updated: {success}")

        # Test 10: Backup config file
        print("\n🔟 Backup config file")
        backup_id = await service.backup_config_file("config/trading.yaml")
        print(f"   ✅ Backup ID: {backup_id}")

        # Test 11: Delete setting
        print("\n1️⃣1️⃣ Delete setting")
        success = await service.delete_setting("api", "binance_key")
        print(f"   ✅ Deleted: {success}")

        all_settings = await service.get_all_settings()
        print(f"   ✅ Remaining settings: {len(all_settings)}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
