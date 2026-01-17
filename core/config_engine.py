#!/usr/bin/env python3
"""
core/config_engine.py
SuperBot - Config Yönetim Sistemi
Yazar: SuperBot Team
Tarih: 2025-10-16
Versiyon: 1.0.0

Özellikler:
- Multi-YAML support (birden fazla config dosyası)
- Environment variable substitution (${REDIS_HOST})
- Hot reload support (FileWatcher entegrasyonu)
- Schema validation (Pydantic - opsiyonel)
- Nested key access (dot notation: cache.backend)
- Thread-safe config access
- Config versioning & rollback
- Callback system (config değişince notify)
- Config merging (base + environment + override)

Kullanım:
    from core.config_engine import ConfigEngine
    
    # Initialize
    config = ConfigEngine(base_path="config/")
    
    # Tüm config'leri yükle
    config.load_all([
        "main.yaml",
        "infrastructure.yaml",
        "connectors.yaml"
    ])
    
    # Nested key access
    backend = config.get("cache.backend", default="memory")
    
    # Environment variable override
    redis_host = config.get("redis.host")  # ${REDIS_HOST} → 100.98.224.83
    
    # Config değişikliğinde callback
    config.on_change("cache.backend", lambda old, new: print(f"{old} → {new}"))
    
    # Hot reload
    config.reload()
    
    # Versioning
    config.save_snapshot("v1.0")
    config.rollback("v1.0")

Bağımlılıklar:
    - pyyaml
    - python-dotenv
    - pydantic (opsiyonel - validation için)
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

# Environment variables için
from dotenv import load_dotenv

# Schema validation için
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
    """Config snapshot (versioning için)"""
    version: str
    timestamp: datetime
    config_data: Dict[str, Any]


if PYDANTIC_AVAILABLE:
    class ConfigSchema(BaseModel):
        """
        Config şema base class

        Kullanım:
            class MyConfigSchema(ConfigSchema):
                api_key: str = Field(..., min_length=10)
                timeout: int = Field(30, ge=1, le=300)
                retry_count: int = Field(3, ge=0, le=10)
        """
        model_config = ConfigDict(
            extra="allow",  # Extra field'lara izin ver
            validate_assignment=True  # Assignment'ta validate et
        )


    class RiskManagementSchema(ConfigSchema):
        """Risk management config şeması - ÖRNEK"""
        max_position_size: float = Field(..., gt=0, le=100, description="Maksimum pozisyon büyüklüğü (%)")
        max_risk_per_trade: float = Field(..., gt=0, le=10, description="Trade başına max risk (%)")
        max_portfolio_risk: float = Field(..., gt=0, le=50, description="Portföy max riski (%)")

        @field_validator('max_position_size')
        @classmethod
        def validate_position_size(cls, v):
            if v > 20:
                raise ValueError("Pozisyon büyüklüğü %20'den fazla olamaz (güvenlik)")
            return v
else:
    # Pydantic yoksa dummy class'lar
    class ConfigSchema:
        pass

    class RiskManagementSchema:
        pass


class ConfigEngine:
    """
    Config yönetim sistemi
    
    Özellikler:
    - Multi-YAML config loading
    - Environment variable substitution
    - Hot reload
    - Thread-safe access
    - Versioning & rollback
    - Change callbacks
    """
    
    def __init__(self, base_path: str = "config/", env_file: str = ".env"):
        """
        ConfigEngine'i başlat
        
        Args:
            base_path: Config dosyalarının bulunduğu klasör
            env_file: .env dosyası adı
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
        
        # .env dosyasını yükle
        if self.env_file.exists():
            load_dotenv(self.env_file)
        else:
            logger.warning(f".env dosyası bulunamadı: {self.env_file}")
    
    def load(self, filename: str) -> bool:
        """
        Tek bir config dosyasını yükle
        
        Args:
            filename: Config dosya adı (örn: main.yaml)
            
        Returns:
            bool: Başarılı ise True
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            logger.error(f"Config dosyası bulunamadı: {file_path}")
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
            logger.error(f"Config yükleme hatası {filename}: {e}")
            return False
    
    def load_all(self, filenames: List[str]) -> bool:
        """
        Birden fazla config dosyasını yükle
        
        Args:
            filenames: Config dosya adları listesi
            
        Returns:
            bool: Tümü başarılı ise True
        """
        logger.info(f"✅ Toplam {len(filenames)} config dosyası yükleniyor...")
        
        success = True
        for filename in filenames:
            if not self.load(filename):
                success = False
        
        if success:
            logger.info(f"✅ Tüm config'ler yüklendi ({len(filenames)} dosya)")
        else:
            logger.warning("⚠️  Bazı config'ler yüklenemedi")
        
        return success
    
    def reload(self, filename: Optional[str] = None) -> bool:
        """
        Config'i yeniden yükle
        
        Args:
            filename: Belirli bir dosya (None ise tümü)
            
        Returns:
            bool: Başarılı ise True
        """
        if filename:
            logger.info(f"✅ Config yeniden yükleniyor: {filename}")
            
            # Old value'ları kaydet (callback için)
            old_config = copy.deepcopy(self._config)
            
            # Reload
            success = self.load(filename)
            
            if success:
                # Trigger callbacks
                self._trigger_change_callbacks(old_config, self._config)
            
            return success
        else:
            logger.info("✅ Tüm config'ler yeniden yükleniyor...")
            
            old_config = copy.deepcopy(self._config)
            
            # Clear ve reload all
            with self._config_lock:
                self._config = {}
            
            success = self.load_all(self._loaded_files.copy())
            
            if success:
                self._trigger_change_callbacks(old_config, self._config)
            
            return success
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Config değerini al (nested key support)
        
        Args:
            key: Config key (dot notation: "cache.backend")
            default: Default değer
            
        Returns:
            Config değeri veya default
            
        Örnek:
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
        Config değerini set et (runtime override)
        
        Args:
            key: Config key (dot notation)
            value: Yeni değer
            
        Returns:
            bool: Başarılı ise True
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
                            logger.error(f"Callback hatası {key}: {e}")
                
                logger.debug(f"✅ Config güncellendi: {key} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Config set hatası {key}: {e}")
            return False
    
    def on_change(self, key: str, callback: Callable[[Any, Any], None]):
        """
        Config değişikliğinde callback ekle
        
        Args:
            key: İzlenecek config key
            callback: Callback fonksiyon (old_value, new_value)
            
        Örnek:
            config.on_change("cache.backend", 
                lambda old, new: print(f"Cache: {old} → {new}"))
        """
        self._callbacks[key].append(callback)
        logger.debug(f"Callback eklendi: {key}")
    
    def save_snapshot(self, version: str) -> bool:
        """
        Mevcut config'i snapshot olarak kaydet
        
        Args:
            version: Snapshot version (örn: "v1.0", "before-update")
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            with self._config_lock:
                snapshot = ConfigSnapshot(
                    version=version,
                    timestamp=datetime.now(),
                    config_data=copy.deepcopy(self._config)
                )
                
                self._snapshots[version] = snapshot
            
            logger.info(f"✅ Config snapshot kaydedildi: {version}")
            return True
            
        except Exception as e:
            logger.error(f"Snapshot kaydetme hatası: {e}")
            return False
    
    def rollback(self, version: str) -> bool:
        """
        Belirli bir snapshot'a geri dön
        
        Args:
            version: Snapshot version
            
        Returns:
            bool: Başarılı ise True
        """
        if version not in self._snapshots:
            logger.error(f"Snapshot bulunamadı: {version}")
            return False
        
        try:
            with self._config_lock:
                old_config = copy.deepcopy(self._config)
                snapshot = self._snapshots[version]
                self._config = copy.deepcopy(snapshot.config_data)
                
                # Trigger callbacks
                self._trigger_change_callbacks(old_config, self._config)
            
            logger.info(f"✅ Config rollback yapıldı: {version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback hatası: {e}")
            return False
    
    def has_changed(self, filename: str) -> bool:
        """
        Config dosyası değişti mi kontrol et
        
        Args:
            filename: Config dosya adı
            
        Returns:
            bool: Değiştiyse True
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            return False
        
        current_mtime = file_path.stat().st_mtime
        last_mtime = self._file_timestamps.get(filename, 0)
        
        return current_mtime > last_mtime
    
    def get_all(self) -> Dict[str, Any]:
        """
        Tüm config'i döndür
        
        Returns:
            Dict: Config data (copy)
        """
        with self._config_lock:
            return copy.deepcopy(self._config)
    
    def get_loaded_files(self) -> List[str]:
        """Yüklü config dosyalarını döndür"""
        return self._loaded_files.copy()
    
    def get_snapshots(self) -> List[str]:
        """Mevcut snapshot'ları döndür"""
        return list(self._snapshots.keys())
    
    def _merge_config(self, new_data: Dict[str, Any]):
        """Config'i merge et (deep merge)"""
        self._config = self._deep_merge(self._config, new_data)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        İki dict'i deep merge et
        
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
        Environment variable substitution
        
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
            # ${VAR_NAME} pattern'ini bul
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(pattern, replacer, data)
        else:
            return data
    
    def _trigger_change_callbacks(self, old_config: Dict, new_config: Dict):
        """Config değişikliklerinde callback'leri tetikle"""
        # Her registered key için kontrol et
        for key in self._callbacks.keys():
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)
            
            if old_value != new_value:
                for callback in self._callbacks[key]:
                    try:
                        callback(old_value, new_value)
                    except Exception as e:
                        logger.error(f"Callback hatası {key}: {e}")
    
    def _get_nested_value(self, data: Dict, key: str) -> Any:
        """Nested key'den value al"""
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
        Config'i şemaya göre validate et

        Args:
            schema: Pydantic BaseModel şeması
            config_path: Validate edilecek config path (None ise root)

        Returns:
            dict: Validate edilmiş config

        Raises:
            ValidationError: Validation başarısız olursa
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic yüklü değil, validation atlandı")
            return {}

        try:
            # Config'i al
            if config_path:
                config_data = self.get(config_path, {})
            else:
                config_data = self.config

            # Validate et
            logger.debug(f"🔍 Config validation başlıyor: {schema.__name__}")
            validated = schema(**config_data)

            logger.info(f"✅ Config validation başarılı: {schema.__name__}")
            return validated.model_dump()

        except ValidationError as e:
            logger.error(f"❌ Config validation hatası: {e}")
            # Hataları detaylı logla
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
        Config path için şema kaydet

        Args:
            config_path: Config path (örn: "risk_management")
            schema: Pydantic şeması
            auto_validate: Config değişince otomatik validate et
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic yüklü değil, schema kaydı atlandı")
            return

        if not hasattr(self, '_schemas'):
            self._schemas = {}

        self._schemas[config_path] = schema
        logger.info(f"📋 Şema kaydedildi: {config_path} -> {schema.__name__}")

        # Şimdi validate et
        if auto_validate:
            try:
                self.validate(schema, config_path)
            except ValidationError:
                logger.warning(f"⚠️  Şema validation başarısız: {config_path}")

    def validate_all(self) -> Dict[str, bool]:
        """
        Kayıtlı tüm şemaları validate et

        Returns:
            dict: {config_path: success}
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️  Pydantic yüklü değil, validation atlandı")
            return {}

        if not hasattr(self, '_schemas'):
            logger.warning("⚠️  Kayıtlı şema yok")
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
        logger.info(f"📊 Schema validation: {success_count}/{total} başarılı")

        return results


# Test kodu
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ConfigEngine Test")
    print("=" * 60)
    
    # Test config dosyası oluştur
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
    
    print("\n1️⃣  ConfigEngine oluşturuluyor...")
    config = ConfigEngine(base_path="config_test/")
    
    print("\n2️⃣  Config'ler yükleniyor...")
    config.load_all(["main.yaml", "infrastructure.yaml"])
    
    print("\n3️⃣  Config okuma testleri:")
    print(f"   system.name: {config.get('system.name')}")
    print(f"   cache.backend: {config.get('cache.backend')}")
    print(f"   redis.host: {config.get('redis.host')}")  # ${REDIS_HOST} → 100.98.224.83
    print(f"   nonexistent (default): {config.get('nonexistent', default='DEFAULT')}")
    
    print("\n4️⃣  Config değişikliği callback testi:")
    def on_backend_change(old, new):
        print(f"   🔔 Cache backend değişti: {old} → {new}")
    
    config.on_change("cache.backend", on_backend_change)
    config.set("cache.backend", "redis")
    
    print("\n5️⃣  Snapshot testi:")
    config.save_snapshot("v1.0")
    config.set("cache.ttl", 10)
    print(f"   cache.ttl (değiştirildi): {config.get('cache.ttl')}")
    
    config.rollback("v1.0")
    print(f"   cache.ttl (rollback): {config.get('cache.ttl')}")
    
    print("\n6️⃣  Yüklü dosyalar:")
    for f in config.get_loaded_files():
        print(f"   - {f}")

    # Schema validation testi
    print("\n7️⃣  Schema validation testi:")
    if PYDANTIC_AVAILABLE:
        from pydantic import BaseModel, Field

        class TestSchema(BaseModel):
            name: str = Field(..., min_length=3)
            age: int = Field(..., ge=18, le=100)

        # Geçerli config
        valid_config = {"name": "John", "age": 25}
        config.config = valid_config

        try:
            result = config.validate(TestSchema)
            print(f"   ✅ Geçerli config: {result}")
        except:
            print("   ❌ Validation başarısız")

        # Geçersiz config
        invalid_config = {"name": "Jo", "age": 15}  # name çok kısa, age çok küçük
        config.config = invalid_config

        try:
            result = config.validate(TestSchema)
            print("   ❌ Geçersiz config geçti (HATA!)")
        except ValidationError as e:
            print(f"   ✅ Geçersiz config yakalandı: {len(e.errors())} hata")
    else:
        print("   ⚠️  Pydantic yüklü değil, test atlandı")

    # Cleanup
    import shutil
    shutil.rmtree(test_config_dir)

    print("\n✅ Test tamamlandı!")
    print("=" * 60)


# ============================================================================
# SINGLETON & HELPER FUNCTIONS
# ============================================================================


_config_engine_instance: Optional[ConfigEngine] = None
_config_lock = threading.Lock()


def get_config_engine() -> ConfigEngine:
    """
    ConfigEngine singleton instance'ını döndür.
    
    Returns:
        ConfigEngine: Singleton instance
    """
    global _config_engine_instance
    if _config_engine_instance is None:
        with _config_lock:
            if _config_engine_instance is None:
                _config_engine_instance = ConfigEngine(base_path="config/")
                # Tüm config dosyalarını yükle
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
    Config value döndür (backward compatibility).
    
    Args:
        key: Config key (dot notation)
        default: Default value
        
    Returns:
        Any: Config value veya ConfigEngine instance (key=None ise)
    """
    engine = get_config_engine()
    if key is None:
        return engine
    return engine.get(key, default)