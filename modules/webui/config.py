"""WebUI Configuration Loader"""
from pathlib import Path
import sys
import logging as stdlib_logging

# Add SuperBot root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config_engine import ConfigEngine
from core.logger_engine import LoggerEngine

class WebUIConfig:
    """WebUI configuration manager"""

    def __init__(self):
        self.config_engine = ConfigEngine()
        self.config_engine.load_all([
            "main.yaml",
            "infrastructure.yaml",
            "connectors.yaml",
            "analysis.yaml"
        ])

        # Setup logger config
        self.logger_config = self._setup_logger_config()
        self.logger_engine = LoggerEngine(config=self.logger_config)
        self.logger = self.logger_engine.get_logger("WebUI")

        # Silence noisy loggers
        self._silence_noisy_loggers()

    def _setup_logger_config(self):
        """Configure logger for WebUI"""
        logger_config = self.config_engine.get("logging", {})

        # Disable file logging for WebUI - prevents log rotation permission errors
        # Multiple processes writing to same log file causes PermissionError on Windows
        if 'file' not in logger_config:
            logger_config['file'] = {}
        logger_config['file']['enabled'] = False

        # Disable Rich colored console for WebUI to prevent conflicts with Flask/Werkzeug
        if 'console' not in logger_config:
            logger_config['console'] = {}
        logger_config['console']['colored'] = False
        logger_config['console']['enabled'] = True
        logger_config['console']['ansi_color'] = False  # Disable Flask/Werkzeug ANSI codes

        # Set log level to INFO for WebUI (no DEBUG spam)
        logger_config['level'] = 'INFO'

        return logger_config

    def _silence_noisy_loggers(self):
        """Silence noisy third-party loggers"""
        # Disable SQLAlchemy echo (DEBUG SQL logs)
        stdlib_logging.getLogger('sqlalchemy.engine').setLevel(stdlib_logging.WARNING)
        stdlib_logging.getLogger('sqlalchemy.pool').setLevel(stdlib_logging.WARNING)

        # Disable aiosqlite DEBUG logs
        stdlib_logging.getLogger('aiosqlite').setLevel(stdlib_logging.WARNING)

    def get_db_config(self):
        """Get database configuration"""
        return self.config_engine.get("infrastructure.database", {})

    def get_connector_config(self):
        """Get connector configuration"""
        return self.config_engine.get("binance", {})

    def get(self, key, default=None):
        """Get configuration value by key"""
        return self.config_engine.get(key, default)
