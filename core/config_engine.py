#!/usr/bin/env python3
"""
core/config_engine.py

SuperBot - Config Management System
Author: SuperBot Team
Date: 2025-10-16
Version: 1.0.0

Features:
- Multi-YAML support (multiple config files)
- Environment variable substitution (${REDIS_HOST})
- Hot reload support (FileWatcher integration)
- Schema validation (Pydantic - optional)
- Nested key access (dot notation: cache.backend)
- Thread-safe config access
- Config versioning & rollback
- Callback system (notify on config change)
- Config merging (base + environment + override)

Usage:
    from core.config_engine import ConfigEngine

    # Initialize
    config = ConfigEngine(base_path="config/")

    # Load all configs
    config.load_all([
        "main.yaml",
        "infrastructure.yaml",
        "connectors.yaml"
    ])

    # Nested key access
    backend = config.get("cache.backend", default="memory")

    # Environment variable override
    redis_host = config.get("redis.host")  # ${REDIS_HOST} → 100.98.224.83

    # Callback on config change
    config.on_change("cache.backend", lambda old, new: print(f"{old} → {new}"))

    # Hot reload
    config.reload()

    # Versioning
    config.save_snapshot("v1.0")
    config.rollback("v1.0")

Dependencies:
    - pyyaml
    - python-dotenv
    - pydantic (optional - for validation)
"""

import os
import yaml
import threading
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.logger_engine import LoggerEngine

# For environment variables
from dotenv import load_dotenv

# For schema validation
try:
    from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Fallback
    ValidationError = Exception
    ConfigDict = None
    field_validator = None

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class ConfigSnapshot:
    """Config snapshot (for versioning)."""
    version: str
    timestamp: datetime
    config_data: Dict[str, Any]


if PYDANTIC_AVAILABLE:
    class ConfigSchema(BaseModel):
        """
        Config schema base class.

        Usage:
            class MyConfigSchema(ConfigSchema):
                api_key: str = Field(..., min_length=10)
                timeout: int = Field(30, ge=1, le=300)
                retry_count: int = Field(3, ge=0, le=10)
        """
        model_config = ConfigDict(
            extra="allow",  # Allow extra fields
            validate_assignment=True  # Validate on assignment
        )


    class RiskManagementSchema(ConfigSchema):
        """Risk management config schema - EXAMPLE"""
        max_position_size: float = Field(..., gt=0, le=100, description="Maximum position size (%)")
        max_risk_per_trade: float = Field(..., gt=0, le=10, description="Max risk per trade (%)")
        max_portfolio_risk: float = Field(..., gt=0, le=50, description="Portfolio max risk (%)")

        @field_validator('max_position_size')
        @classmethod
        def validate_position_size(cls, v):
            if v > 20:
                raise ValueError("Position size cannot exceed 20% (safety)")
            return v
else:
    # Dummy classes if Pydantic not available
    class ConfigSchema:
        pass

    class RiskManagementSchema:
        pass


class ConfigEngine:
    """
    Config management system.

    Features:
    - Multi-YAML config loading
    - Environment variable substitution
    - Hot reload
    - Thread-safe access
    - Versioning & rollback
    - Change callbacks
    """

    def __init__(self, base_path: str = "config/", env_file: str = ".env"):
        """
        Initialize ConfigEngine.

        Args:
            base_path: Directory containing config files
            env_file: .env file name
        """
        self.base_path = Path(base_path)
        self.env_file = self.base_path / env_file
        
        # Config data (merged)
        self._config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        
        # Loaded files tracking
        self._loaded_files: List[str] = []
        self._file_timestamps: Dict[str, float] = {}
        
        # Versioning
        self._snapshots: Dict[str, ConfigSnapshot] = {}
        
        # Callbacks: key -> [callbacks]
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Load .env file
        if self.env_file.exists():
            load_dotenv(self.env_file)
        else:
            logger.warning(f".env file not found: {self.env_file}")

    def load(self, filename: str) -> bool:
        """
        Load a single config file.

        Args:
            filename: Config file name (e.g., main.yaml)

        Returns:
            bool: True if successful
        """
        file_path = self.base_path / filename

        if not file_path.exists():
            logger.error(f"Config file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            # Environment variable substitution
            data = self._substitute_env_vars(data)
            
            with self._config_lock:
                # Merge into main config
                self._merge_config(data)
                
                # Track loaded file
                if filename not in self._loaded_files:
                    self._loaded_files.append(filename)
                
                # Track timestamp
                self._file_timestamps[filename] = file_path.stat().st_mtime

            return True

        except Exception as e:
            logger.error(f"Config loading error {filename}: {e}")
            return False

    def load_all(self, filenames: List[str]) -> bool:
        """
        Load multiple config files.

        Args:
            filenames: List of config file names

        Returns:
            bool: True if all successful
        """
        logger.info(f"✅ Loading {len(filenames)} config files...")

        success = True
        for filename in filenames:
            if not self.load(filename):
                success = False

        if success:
            logger.info(f"✅ All configs loaded ({len(filenames)} files)")
        else:
            logger.warning("⚠️  Some configs failed to load")

        return success

    def reload(self, filename: Optional[str] = None) -> bool:
        """
        Reload config.

        Args:
            filename: Specific file (None for all)

        Returns:
            bool: True if successful
        """
        if filename:
            logger.info(f"✅ Reloading config: {filename}")

            # Save old values (for callback)
            old_config = copy.deepcopy(self._config)
            
            # Reload
            success = self.load(filename)
            
            if success:
                # Trigger callbacks
                self._trigger_change_callbacks(old_config, self._config)
            
            return success
        else:
            logger.info("✅ Reloading all configs...")

            old_config = copy.deepcopy(self._config)

            # Clear and reload all
            with self._config_lock:
                self._config = {}

            success = self.load_all(self._loaded_files.copy())

            if success:
                self._trigger_change_callbacks(old_config, self._config)

            return success

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value (nested key support).

        Args:
            key: Config key (dot notation: "cache.backend")
            default: Default value

        Returns:
            Config value or default

        Example:
            backend = config.get("cache.backend", default="memory")
            max_risk = config.get("trading.risk.max_per_trade", default=2.0)
        """
        with self._config_lock:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value

    def set(self, key: str, value: Any) -> bool:
        """
        Set config value (runtime override).

        Args:
            key: Config key (dot notation)
            value: New value

        Returns:
            bool: True if successful
        """
        try:
            with self._config_lock:
                old_value = self.get(key)

                # Set value
                keys = key.split('.')
                config = self._config

                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]

                config[keys[-1]] = value

                # Trigger callbacks
                if key in self._callbacks:
                    for callback in self._callbacks[key]:
                        try:
                            callback(old_value, value)
                        except Exception as e:
                            logger.error(f"Callback error {key}: {e}")

                logger.debug(f"✅ Config updated: {key} = {value}")
                return True

        except Exception as e:
            logger.error(f"Config set error {key}: {e}")
            return False

    def on_change(self, key: str, callback: Callable[[Any, Any], None]):
        """
        Add callback on config change.

        Args:
            key: Config key to watch
            callback: Callback function (old_value, new_value)

        Example:
            config.on_change("cache.backend",
                lambda old, new: print(f"Cache: {old} → {new}"))
        """
        self._callbacks[key].append(callback)
        logger.debug(f"Callback added: {key}")

    def save_snapshot(self, version: str) -> bool:
        """
        Save current config as snapshot.

        Args:
            version: Snapshot version (e.g., "v1.0", "before-update")

        Returns:
            bool: True if successful
        """
        try:
            with self._config_lock:
                snapshot = ConfigSnapshot(
                    version=version,
                    timestamp=datetime.now(),
                    config_data=copy.deepcopy(self._config)
                )
                
                self._snapshots[version] = snapshot

            logger.info(f"✅ Config snapshot saved: {version}")
            return True

        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return False

    def rollback(self, version: str) -> bool:
        """
        Rollback to a specific snapshot.

        Args:
            version: Snapshot version

        Returns:
            bool: True if successful
        """
        if version not in self._snapshots:
            logger.error(f"Snapshot not found: {version}")
            return False

        try:
            with self._config_lock:
                old_config = copy.deepcopy(self._config)
                snapshot = self._snapshots[version]
                self._config = copy.deepcopy(snapshot.config_data)

                # Trigger callbacks
                self._trigger_change_callbacks(old_config, self._config)

            logger.info(f"✅ Config rollback completed: {version}")
            return True

        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False

    def has_changed(self, filename: str) -> bool:
        """
        Check if config file has changed.

        Args:
            filename: Config file name

        Returns:
            bool: True if changed
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            return False
        
        current_mtime = file_path.stat().st_mtime
        last_mtime = self._file_timestamps.get(filename, 0)
        
        return current_mtime > last_mtime

    def get_all(self) -> Dict[str, Any]:
        """
        Return all config.

        Returns:
            Dict: Config data (copy)
        """
        with self._config_lock:
            return copy.deepcopy(self._config)

    def get_loaded_files(self) -> List[str]:
        """Return loaded config files."""
        return self._loaded_files.copy()

    def get_snapshots(self) -> List[str]:
        """Return available snapshots."""
        return list(self._snapshots.keys())

    def _merge_config(self, new_data: Dict[str, Any]):
        """Merge config (deep merge)."""
        self._config = self._deep_merge(self._config, new_data)

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dicts.

        Args:
            base: Base dict
            update: Update dict

        Returns:
            Dict: Merged dict
        """
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """
        Environment variable substitution.

        ${REDIS_HOST} → os.getenv("REDIS_HOST")

        Args:
            data: Config data (dict, list, str)

        Returns:
            Substituted data
        """
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Find ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'

            def replacer(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))

            return re.sub(pattern, replacer, data)
        else:
            return data

    def _trigger_change_callbacks(self, old_config: Dict, new_config: Dict):
        """Trigger callbacks on config changes."""
        # Check each registered key
        for key in self._callbacks.keys():
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)

            if old_value != new_value:
                for callback in self._callbacks[key]:
                    try:
                        callback(old_value, new_value)
                    except Exception as e:
                        logger.error(f"Callback error {key}: {e}")

    def _get_nested_value(self, data: Dict, key: str) -> Any:
        """Get value from nested key."""
        keys = key.split('.')
        value = data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None

        return value

    def validate(
        self,
        schema: type[BaseModel],
        config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate config against schema.

        Args:
            schema: Pydantic BaseModel schema
            config_path: Config path to validate (None for root)

        Returns:
            dict: Validated config

        Raises:
            ValidationError: If validation fails
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic not installed, validation skipped")
            return {}

        try:
            # Get config
            if config_path:
                config_data = self.get(config_path, {})
            else:
                config_data = self.config

            # Validate
            logger.debug(f"🔍 Config validation starting: {schema.__name__}")
            validated = schema(**config_data)

            logger.info(f"✅ Config validation successful: {schema.__name__}")
            return validated.model_dump()

        except ValidationError as e:
            logger.error(f"❌ Config validation error: {e}")
            # Log errors in detail
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                msg = error['msg']
                logger.error(f"   • {field}: {msg}")
            raise

    def register_schema(
        self,
        config_path: str,
        schema: type[BaseModel],
        auto_validate: bool = True
    ) -> None:
        """
        Register schema for config path.

        Args:
            config_path: Config path (e.g., "risk_management")
            schema: Pydantic schema
            auto_validate: Auto validate on config change
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic not installed, schema registration skipped")
            return

        if not hasattr(self, '_schemas'):
            self._schemas = {}

        self._schemas[config_path] = schema
        logger.info(f"📋 Schema registered: {config_path} -> {schema.__name__}")

        # Validate now
        if auto_validate:
            try:
                self.validate(schema, config_path)
            except ValidationError:
                logger.warning(f"⚠️  Schema validation failed: {config_path}")

    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all registered schemas.

        Returns:
            dict: {config_path: success}
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic not installed, validation skipped")
            return {}

        if not hasattr(self, '_schemas'):
            logger.warning("⚠️  No registered schemas")
            return {}

        results = {}
        for config_path, schema in self._schemas.items():
            try:
                self.validate(schema, config_path)
                results[config_path] = True
            except ValidationError:
                results[config_path] = False

        success_count = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(f"📊 Schema validation: {success_count}/{total} successful")

        return results


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ConfigEngine Test")
    print("=" * 60)

    # Create test config directory
    test_config_dir = Path("config_test")
    test_config_dir.mkdir(exist_ok=True)
    
    # main.yaml
    main_yaml = test_config_dir / "main.yaml"
    main_yaml.write_text("""
system:
  name: SuperBot
  version: 1.0.0

logging:
  level: INFO
  log_dir: data/logs
""")
    
    # infrastructure.yaml
    infra_yaml = test_config_dir / "infrastructure.yaml"
    infra_yaml.write_text("""
cache:
  backend: memory
  ttl: 5

redis:
  host: ${REDIS_HOST}
  port: ${REDIS_PORT}
""")
    
    # .env
    env_file = test_config_dir / ".env"
    env_file.write_text("REDIS_HOST=100.98.224.83\n")
    
    print("\n1️⃣  Creating ConfigEngine...")
    config = ConfigEngine(base_path="config_test/")

    print("\n2️⃣  Loading configs...")
    config.load_all(["main.yaml", "infrastructure.yaml"])

    print("\n3️⃣  Config read tests:")
    print(f"   system.name: {config.get('system.name')}")
    print(f"   cache.backend: {config.get('cache.backend')}")
    print(f"   redis.host: {config.get('redis.host')}")  # ${REDIS_HOST} → 100.98.224.83
    print(f"   nonexistent (default): {config.get('nonexistent', default='DEFAULT')}")

    print("\n4️⃣  Config change callback test:")
    def on_backend_change(old, new):
        print(f"   🔔 Cache backend changed: {old} → {new}")

    config.on_change("cache.backend", on_backend_change)
    config.set("cache.backend", "redis")

    print("\n5️⃣  Snapshot test:")
    config.save_snapshot("v1.0")
    config.set("cache.ttl", 10)
    print(f"   cache.ttl (changed): {config.get('cache.ttl')}")

    config.rollback("v1.0")
    print(f"   cache.ttl (rollback): {config.get('cache.ttl')}")

    print("\n6️⃣  Loaded files:")
    for f in config.get_loaded_files():
        print(f"   - {f}")

    # Schema validation test
    print("\n7️⃣  Schema validation test:")
    if PYDANTIC_AVAILABLE:
        from pydantic import BaseModel, Field

        class TestSchema(BaseModel):
            name: str = Field(..., min_length=3)
            age: int = Field(..., ge=18, le=100)

        # Valid config
        valid_config = {"name": "John", "age": 25}
        config.config = valid_config

        try:
            result = config.validate(TestSchema)
            print(f"   ✅ Valid config: {result}")
        except:
            print("   ❌ Validation failed")

        # Invalid config
        invalid_config = {"name": "Jo", "age": 15}  # name too short, age too low
        config.config = invalid_config

        try:
            result = config.validate(TestSchema)
            print("   ❌ Invalid config passed (ERROR!)")
        except ValidationError as e:
            print(f"   ✅ Invalid config caught: {len(e.errors())} errors")
    else:
        print("   ⚠️  Pydantic not installed, test skipped")

    # Cleanup
    import shutil
    shutil.rmtree(test_config_dir)

    print("\n✅ Test completed!")
    print("=" * 60)


# ============================================================================
# SINGLETON & HELPER FUNCTIONS
# ============================================================================


_config_engine_instance: Optional[ConfigEngine] = None
_config_lock = threading.Lock()


def get_config_engine() -> ConfigEngine:
    """
    Return ConfigEngine singleton instance.

    Returns:
        ConfigEngine: Singleton instance
    """
    global _config_engine_instance
    if _config_engine_instance is None:
        with _config_lock:
            if _config_engine_instance is None:
                _config_engine_instance = ConfigEngine(base_path="config/")
                # Load all config files
                _config_engine_instance.load_all([
                    "main.yaml",
                    "infrastructure.yaml",
                    "connectors.yaml",
                    "daemon.yaml",
                    "trading.yaml"
                ])
    return _config_engine_instance


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Return config value (backward compatibility).

    Args:
        key: Config key (dot notation)
        default: Default value

    Returns:
        Any: Config value or ConfigEngine instance (if key=None)
    """
    engine = get_config_engine()
    if key is None:
        return engine
    return engine.get(key, default)