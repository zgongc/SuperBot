#!/usr/bin/env python3
"""
core/logger_engine.py
SuperBot - Merkezi Loglama Sistemi
Yazar: SuperBot Team
Tarih: 2025-10-16
Versiyon: 1.0.0

Özellikler:
- Hibrit format (Console: renkli/okunabilir, File: JSON)
- Custom log levels (SIGNAL, TRADE, POSITION, PNL)
- Thread-safe logging
- Performance monitoring (timing decorator)
- Correlation ID tracking (trade takibi)
- Async/sync support
- Log rotation (günlük + 50MB)
- Config entegrasyonu
- EventBus entegrasyonu (critical log'lar event olur)

Kullanım:
    from core.logger_engine import LoggerEngine, log_execution_time
    
    # Logger al
    logger = LoggerEngine().get_logger("MyModule")
    
    # Normal log
    logger.info("Bot başlatıldı")
    logger.warning("Yüksek volatilite tespit edildi")
    
    # Custom level log
    logger.signal("BUY sinyali oluşturuldu", symbol="BTCUSDT", rsi=28.5)
    logger.trade("Emir gerçekleştirildi", symbol="ETHUSDT", side="SELL")
    logger.position("Pozisyon açıldı", symbol="BNBUSDT", side="LONG")
    logger.pnl("Kâr realize edildi", symbol="BTCUSDT", pnl=125.50)
    
    # Correlation ID ile trade tracking
    with logger_engine.correlation_context() as trade_id:
        logger.info("Trade başlatıldı")
        logger.trade("Emir gönderildi")
    
    # Performance monitoring
    @log_execution_time("MyModule")
    async def slow_function():
        await asyncio.sleep(1)

Bağımlılıklar:
    - rich (opsiyonel - renkli console için)
"""

import logging
import sys
import json
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Rich kütüphanesi (opsiyonel)
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich kütüphanesi bulunamadı. Kurulum: pip install rich")


# Custom log level'lar
VERBOSE_LEVEL = 15   # Verbose logs (DEBUG < VERBOSE < INFO)
SIGNAL_LEVEL = 25    # Trading sinyalleri (INFO ile WARNING arası)
TRADE_LEVEL = 26     # Trade execution
POSITION_LEVEL = 27  # Position değişiklikleri
PNL_LEVEL = 28       # P&L değişiklikleri

# Level'ları logging modülüne kaydet
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")
logging.addLevelName(SIGNAL_LEVEL, "SIGNAL")
logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(POSITION_LEVEL, "POSITION")
logging.addLevelName(PNL_LEVEL, "PNL")


class JSONFormatter(logging.Formatter):
    """
    JSON formatında log formatter
    
    Dosya log'ları için kullanılır - machine-readable format
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Log record'unu JSON formatına çevir"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Correlation ID varsa ekle
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Extra data varsa ekle (trading data için)
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        
        # Exception varsa ekle
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class CorrelationFilter(logging.Filter):
    """
    Correlation ID filter
    
    Thread-local storage kullanarak her thread'e unique ID atar
    Trade tracking için kullanılır
    """
    
    _thread_local = threading.local()
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Mevcut thread için correlation ID set et"""
        cls._thread_local.correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Mevcut thread'in correlation ID'sini al"""
        return getattr(cls._thread_local, 'correlation_id', None)
    
    @classmethod
    def clear_correlation_id(cls):
        """Mevcut thread'in correlation ID'sini temizle"""
        if hasattr(cls._thread_local, 'correlation_id'):
            delattr(cls._thread_local, 'correlation_id')
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Her log record'una correlation ID ekle"""
        correlation_id = self.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class EventBusFilter(logging.Filter):
    """
    EventBus entegrasyonu için filter
    
    CRITICAL ve ERROR log'ları EventBus'a publish eder
    """
    
    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Critical/Error log'ları event olarak yayınla"""
        if self.event_bus and record.levelno >= logging.ERROR:
            try:
                self.event_bus.publish(
                    topic="system.error" if record.levelno == logging.ERROR else "system.critical",
                    data={
                        "level": record.levelname,
                        "module": record.name,
                        "message": record.getMessage(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            except Exception as e:
                # EventBus hatası log sistemini patlatmamalı
                print(f"⚠️  EventBus publish hatası: {e}")
        
        return True


class CustomLogger(logging.Logger):
    """
    Custom log level metodları ile genişletilmiş logger

    Trading'e özel log level'lar ekler:
    - verbose(): Sadece verbose modda görünen detay log'lar
    - signal(): Trading sinyalleri
    - trade(): Trade execution
    - position(): Position değişiklikleri
    - pnl(): P&L değişiklikleri
    """

    def verbose(self, message: str, **kwargs):
        """
        Verbose log (sadece --verbose modda görünür)

        Level 15 (DEBUG=10 < VERBOSE=15 < INFO=20)
        Normal modda log level INFO olduğundan görünmez.
        --verbose modda log level VERBOSE olur ve görünür.
        """
        if self.isEnabledFor(VERBOSE_LEVEL):
            if kwargs:
                self._log(VERBOSE_LEVEL, message, (), extra={'extra_data': kwargs})
            else:
                self._log(VERBOSE_LEVEL, message, ())

    def signal(self, message: str, **kwargs):
        """Trading signal logu"""
        if self.isEnabledFor(SIGNAL_LEVEL):
            self._log(SIGNAL_LEVEL, message, (), extra={'extra_data': kwargs})

    def trade(self, message: str, **kwargs):
        """Trade execution logu"""
        if self.isEnabledFor(TRADE_LEVEL):
            self._log(TRADE_LEVEL, message, (), extra={'extra_data': kwargs})

    def position(self, message: str, **kwargs):
        """Position değişikliği logu"""
        if self.isEnabledFor(POSITION_LEVEL):
            self._log(POSITION_LEVEL, message, (), extra={'extra_data': kwargs})

    def pnl(self, message: str, **kwargs):
        """P&L değişikliği logu"""
        if self.isEnabledFor(PNL_LEVEL):
            self._log(PNL_LEVEL, message, (), extra={'extra_data': kwargs})


# Custom logger class'ını logging modülüne kaydet
logging.setLoggerClass(CustomLogger)


class LoggerEngine:
    """
    Merkezi loglama sistemi
    
    Özellikler:
    - Singleton pattern (tüm uygulama aynı instance'ı kullanır)
    - Thread-safe
    - Async-compatible
    - Config-driven
    - EventBus entegrasyonu
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, event_bus=None):
        """
        LoggerEngine'i başlat
        
        Args:
            config: Config dict (main.yaml logging section)
            event_bus: EventBus instance (opsiyonel)
        """
        if self._initialized:
            return
        
        # Config
        self.config = config or {}
        self.event_bus = event_bus
        
        # Log klasörü
        log_dir = self.config.get('log_dir', 'data/logs')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log level
        log_level = self.config.get('level', 'INFO').upper()
        self.log_level = getattr(logging, log_level, logging.INFO)
        
        # Console setup (Rich)
        if RICH_AVAILABLE:
            # Force UTF-8 encoding for Windows emoji support
            import sys
            import io
            if sys.platform == 'win32':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

            self.console = Console(
                force_terminal=True,
                force_interactive=False,
                theme=Theme({
                    "logging.level.signal": "bold cyan",
                    "logging.level.trade": "bold green",
                    "logging.level.position": "bold yellow",
                    "logging.level.pnl": "bold magenta"
                })
            )
        else:
            self.console = None
        
        # Root logger setup
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)

        # Filters
        self.correlation_filter = CorrelationFilter()
        self.eventbus_filter = EventBusFilter(event_bus)

        # Handlers setup
        self._setup_handlers()

        # ANSI color kontrolü - Flask/Werkzeug logger'larını sustur
        console_config = self.config.get('console', {})
        if not console_config.get('ansi_color', True):
            self._disable_werkzeug_ansi_colors()

        self._initialized = True
    
    def _setup_handlers(self):
        """Console ve file handlers'ları ayarla"""

        # Console handler (Rich veya Standard)
        if RICH_AVAILABLE:
            console_handler = RichHandler(
                console=self.console,
                rich_tracebacks=True,
                show_time=False,
                show_level=True,
                show_path=False,
                markup=True
            )
        else:
            # Fallback: Standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)

        console_handler.setLevel(self.log_level)
        console_handler.addFilter(self.correlation_filter)
        console_handler.addFilter(self.eventbus_filter)
        self.root_logger.addHandler(console_handler)

        # File handlers (JSON format) - SADECE config'de aktifse
        if self.config.get('file', False):
            rotation_config = self.config.get('rotation', {})
            max_bytes = rotation_config.get('max_bytes', 52428800)  # 50MB
            backup_count = rotation_config.get('backup_count', 5)

            # 1. Main log - Tüm log'lar
            self._add_file_handler("main.log", logging.DEBUG, max_bytes, backup_count)

            # 2. Trading log - Sadece trading log'ları
            self._add_file_handler(
                "trading.log", SIGNAL_LEVEL, max_bytes, backup_count,
                filter_levels=[SIGNAL_LEVEL, TRADE_LEVEL, POSITION_LEVEL, PNL_LEVEL]
            )

            # 3. Error log - Sadece error ve critical
            self._add_file_handler("errors.log", logging.ERROR, max_bytes, backup_count)

            # 4. Performance log - Sadece performance log'ları
            self._add_file_handler(
                "performance.log", logging.DEBUG, max_bytes, backup_count,
                filter_name="performance"
            )
    
    def _add_file_handler(
        self, filename: str, level: int, max_bytes: int, backup_count: int,
        filter_levels: Optional[list] = None, filter_name: Optional[str] = None
    ):
        """File handler ekle"""
        file_path = self.log_dir / filename
        
        handler = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        handler.addFilter(self.correlation_filter)
        handler.addFilter(self.eventbus_filter)
        
        if filter_levels:
            handler.addFilter(lambda record: record.levelno in filter_levels)
        
        if filter_name:
            handler.addFilter(lambda record: filter_name.lower() in record.name.lower())
        
        self.root_logger.addHandler(handler)
    
    def get_logger(self, name: str) -> CustomLogger:
        """Modül-specific logger döndür"""
        return logging.getLogger(name)
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Correlation ID context manager"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        CorrelationFilter.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            CorrelationFilter.clear_correlation_id()
    
    def _disable_werkzeug_ansi_colors(self):
        """Flask/Werkzeug ANSI color kodlarını devre dışı bırak"""
        import os

        # ANSI color'ları kapat
        os.environ['TERM'] = 'dumb'
        os.environ['NO_COLOR'] = '1'

        # Werkzeug logger'ını sustur
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.WARNING)

        # Flask'ın kendi logger'ı
        flask_logger = logging.getLogger('flask')
        flask_logger.setLevel(logging.WARNING)

        logger = self.get_logger("LoggerEngine")
        logger.debug("🔇 Flask/Werkzeug ANSI color kodları devre dışı bırakıldı")

    def set_log_level(self, level: str):
        """Global log level'ı değiştir"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "VERBOSE": VERBOSE_LEVEL,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        log_level = level_map.get(level.upper(), logging.INFO)
        self.root_logger.setLevel(log_level)

        # Console handler'ların level'ını da güncelle
        for handler in self.root_logger.handlers:
            if isinstance(handler, (logging.StreamHandler, RichHandler if RICH_AVAILABLE else type(None))):
                handler.setLevel(log_level)

        logger = self.get_logger("LoggerEngine")
        logger.info(f"Log level değiştirildi: {level}")

    def set_verbose_mode(self, enabled: bool = True):
        """
        Verbose mode'u aç/kapat

        Args:
            enabled: True ise VERBOSE level, False ise INFO level
        """
        if enabled:
            self.set_log_level("VERBOSE")
        else:
            self.set_log_level("INFO")


def log_execution_time(logger_name: str = "performance"):
    """Function execution time'ı logla"""
    def decorator(func: Callable):
        logger = logging.getLogger(logger_name)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"⏱️  {func.__name__} tamamlandı",
                    extra={'extra_data': {
                        'execution_time': f"{execution_time:.3f}s",
                        'function': func.__name__
                    }}
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"❌ {func.__name__} başarısız",
                    extra={'extra_data': {
                        'execution_time': f"{execution_time:.3f}s",
                        'function': func.__name__,
                        'error': str(e)
                    }}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"⏱️  {func.__name__} tamamlandı",
                    extra={'extra_data': {'execution_time': f"{execution_time:.3f}s"}})
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {func.__name__} başarısız",
                    extra={'extra_data': {'execution_time': f"{execution_time:.3f}s", 'error': str(e)}})
                raise
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


if __name__ == "__main__":
    import asyncio
    
    print("=" * 60)
    print("🧪 LoggerEngine Test")
    print("=" * 60)
    
    logger_engine = LoggerEngine()
    logger = logger_engine.get_logger("TestModule")
    
    # Normal log'lar
    logger.debug("🔍 Debug mesajı")
    logger.info("✅ Bot başlatıldı")
    logger.warning("⚠️  Yüksek volatilite")
    logger.error("❌ API hatası")
    
    # Trading log'lar
    logger.signal("📊 BUY sinyali", symbol="BTCUSDT", rsi=28.5)
    logger.trade("💰 Emir gerçekleşti", symbol="ETHUSDT")
    logger.position("📈 Pozisyon açıldı", symbol="BNBUSDT")
    logger.pnl("💵 Kâr realize edildi", pnl=125.50)
    
    # Correlation ID
    with logger_engine.correlation_context() as trade_id:
        logger.info(f"Trade ID: {trade_id}")
    
    @log_execution_time("TestModule")
    def test_func():
        time.sleep(0.1)
    
    test_func()
    
    print("\n✅ Test tamamlandı!")
    print("=" * 60)


# ============================================================================
# SINGLETON & HELPER FUNCTIONS
# ============================================================================


_logger_engine_instance: Optional[LoggerEngine] = None
_logger_lock = threading.Lock()


def get_logger_engine() -> LoggerEngine:
    """
    LoggerEngine singleton instance'ını döndür.
    
    Returns:
        LoggerEngine: Singleton instance
    """
    global _logger_engine_instance
    if _logger_engine_instance is None:
        with _logger_lock:
            if _logger_engine_instance is None:
                _logger_engine_instance = LoggerEngine()
    return _logger_engine_instance


def get_logger(module_name: str) -> logging.Logger:
    """
    Logger instance döndür (backward compatibility).

    Args:
        module_name: Module adı

    Returns:
        logging.Logger: Logger instance
    """
    engine = get_logger_engine()
    return engine.get_logger(module_name)


def set_verbose_mode(enabled: bool = True):
    """
    Global verbose mode'u aç/kapat

    Args:
        enabled: True ise verbose log'lar görünür

    Kullanım:
        from core.logger_engine import set_verbose_mode
        set_verbose_mode(True)  # --verbose flag ile
    """
    engine = get_logger_engine()
    engine.set_verbose_mode(enabled)