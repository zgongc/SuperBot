"""
SuperBot WebUI - Flask Application
===============================================================================

Modern, modular architecture with clean separation of concerns.

Architecture:
    - Config Layer: Configuration and logger setup
    - Service Layer: Business logic (symbols, favorites, analysis, etc.)
    - API Layer: RESTful endpoints organized by domain
    - View Layer: Page rendering routes
    - Helper Layer: Utilities (async, validation, responses)

Author: SuperBot Team
Date: 2025-10-29
Version: 3.0.0 (Refactored)
"""
from flask import Flask, session, render_template
from pathlib import Path

# Import configuration
from .config import WebUIConfig
from .extensions import init_extensions

# Import helpers
from .helpers.async_helper import get_event_loop

# Import blueprints
from .views import create_views_blueprint, get_strategy_blueprint
from .api import create_api_blueprint, get_strategy_api_blueprint

def create_app():
    """Application factory"""

    # Create Flask app
    app = Flask(__name__)
    app.secret_key = 'superbot-secret-key-change-in-production'  # TODO: Move to config
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    # Load configuration
    config = WebUIConfig()
    app.config['WEBUI_CONFIG'] = config

    # Configure Werkzeug logging based on config
    flask_config = config.get('logging.flask', {})
    _configure_werkzeug_logging(
        log_output=flask_config.get('log_output', True),
        level=flask_config.get('level', 'INFO')
    )

    config.logger.info("WebUI Flask App initializing...")

    # Initialize extensions
    init_extensions(app)

    # Initialize SuperBot components
    init_components(app, config)

    # Register blueprints
    app.register_blueprint(create_views_blueprint())
    app.register_blueprint(create_api_blueprint())

    # Register strategy blueprints
    app.register_blueprint(get_strategy_blueprint())
    app.register_blueprint(get_strategy_api_blueprint())

    # Theme context processor
    @app.context_processor
    def inject_theme():
        return {'theme': session.get('theme', 'dark')}

    # Legacy API endpoints (for backwards compatibility)
    register_legacy_endpoints(app)

    # Error handlers
    register_error_handlers(app)

    config.logger.info("WebUI initialization complete")

    return app

def init_components(app, config):
    """Initialize DataManager and other components"""
    from components.datamanager.manager import DataManager
    from components.managers.symbols_manager import SymbolsManager
    from components.exchanges.binance_api import BinanceAPI
    from components.strategies.strategy_manager import StrategyManager

    # Import services
    from .services.symbols_service import SymbolsService
    from .services.favorites_service import FavoritesService
    from .services.analysis_service import AnalysisService
    from .services.portfolio_service import PortfolioService
    from .services.account_service import AccountService
    from .services.stats_service import StatsService
    from .services.categories_service import CategoriesService
    from .services.exchange_service import ExchangeService
    from .services.strategy_ui_service import StrategyUIService
    from .services.replay_service import ReplayService
    from .services.smc_service import SMCService
    from .services.download_service import DownloadService
    from .services.monitoring_service import MonitoringService

    config.logger.info("Initializing components...")

    # Get event loop
    event_loop = get_event_loop()

    # Initialize DataManager
    db_config = config.get_db_config()
    config.logger.info(f"Initializing DataManager with config: {db_config}")
    data_manager = DataManager(config=db_config)

    # Start DataManager
    config.logger.info("Starting DataManager...")
    event_loop.run_until_complete(data_manager.start())
    config.logger.info("✅ DataManager started")

    # Initialize Exchange Client
    config.logger.info("Initializing Binance API...")
    exchange_config = config.get('connectors.binance', {})
    exchange_client = BinanceAPI(config=exchange_config)
    config.logger.info("Binance API initialized")

    # Initialize SymbolsManager
    config.logger.info("Creating SymbolsManager instance")
    symbols_manager = SymbolsManager(
        config=config.config_engine,
        logger=config.logger,
        cache_manager=None,
        exchange_client=exchange_client,
        data_manager=data_manager
    )

    # Initialize StrategyManager
    config.logger.info("Creating StrategyManager instance")
    strategy_manager = StrategyManager(
        indicator_manager=None,  # WebUI doesn't need indicator manager
        position_manager=None,   # WebUI doesn't need position manager
        logger=config.logger
    )
    config.logger.info("StrategyManager initialized")

    # Create services
    config.logger.info("Creating service instances...")
    symbols_service = SymbolsService(data_manager, symbols_manager, config.logger)
    favorites_service = FavoritesService(data_manager, config.logger)
    analysis_service = AnalysisService(data_manager, config.logger)
    portfolio_service = PortfolioService(data_manager, config.logger)
    account_service = AccountService(data_manager, exchange_client, config.logger)
    stats_service = StatsService(data_manager, symbols_manager, config.logger)
    categories_service = CategoriesService(data_manager, config.logger)
    exchange_service = ExchangeService(data_manager)
    strategy_ui_service = StrategyUIService(data_manager, strategy_manager, config.logger)

    # Initialize ParquetsEngine for ReplayService and SMCService
    from components.managers.parquets_engine import ParquetsEngine
    parquets_engine = ParquetsEngine(
        config_engine=config.config_engine,
        logger_engine=config.logger_engine
    )
    replay_service = ReplayService(data_manager, config.logger, parquets_engine, strategy_manager)
    smc_service = SMCService(parquets_engine=parquets_engine, logger=config.logger)
    download_service = DownloadService(logger=config.logger)
    monitoring_service = MonitoringService(history_size=60)

    # Store in app context
    app.data_manager = data_manager
    app.symbols_manager = symbols_manager
    app.strategy_manager = strategy_manager
    app.exchange_client = exchange_client
    app.symbols_service = symbols_service
    app.favorites_service = favorites_service
    app.analysis_service = analysis_service
    app.portfolio_service = portfolio_service
    app.exchange_service = exchange_service
    app.strategy_ui_service = strategy_ui_service

    # Also store in config for easier access in API endpoints
    app.config['data_manager'] = data_manager
    app.config['symbols_manager'] = symbols_manager
    app.config['strategy_manager'] = strategy_manager
    app.config['STRATEGY_UI_SERVICE'] = strategy_ui_service
    app.account_service = account_service
    app.stats_service = stats_service
    app.categories_service = categories_service
    app.replay_service = replay_service
    app.smc_service = smc_service
    app.parquets_engine = parquets_engine
    app.download_service = download_service
    app.monitoring_service = monitoring_service

    # Start background event loop for async operations
    from .helpers.async_helper import start_background_loop
    start_background_loop()

    config.logger.info("All components initialized successfully")

def register_legacy_endpoints(app):
    """Register legacy API endpoints for backwards compatibility"""
    from flask import jsonify

    @app.route('/api/scan-status')
    def api_scan_status():
        """Get current scan status (for HTMX polling)"""
        # TODO: Get real status from scanner
        return jsonify({
            'status': 'idle',  # idle / running / completed
            'last_scan': '2025-10-27 21:30:00',
            'opportunities_count': 5,
            'high_score_count': 2
        })

    @app.route('/api/opportunities')
    def api_opportunities():
        """Get opportunities list (for HTMX)"""
        # TODO: Get real opportunities from database
        mock_data = [
            {
                'id': 1,
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'score': 87.5,
                'patterns': 2,
                'timestamp': '2025-10-27 21:30:00'
            },
            {
                'id': 2,
                'symbol': 'ETHUSDT',
                'timeframe': '4h',
                'score': 75.2,
                'patterns': 1,
                'timestamp': '2025-10-27 21:25:00'
            }
        ]
        return jsonify(mock_data)

def _configure_werkzeug_logging(log_output: bool = True, level: str = 'INFO'):
    """Configure Werkzeug logging based on config settings

    Args:
        log_output: If True, show Werkzeug logs; if False, hide completely
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging
    import sys

    # Get werkzeug logger
    werkzeug_logger = logging.getLogger('werkzeug')

    # Disable propagation to prevent duplicate messages
    werkzeug_logger.propagate = False

    # Remove all existing handlers
    for handler in werkzeug_logger.handlers[:]:
        werkzeug_logger.removeHandler(handler)

    # Check if user wants to see Werkzeug logs
    if not log_output:
        # Completely disable Werkzeug logs
        werkzeug_logger.disabled = True
        return

    # Custom formatter to match LoggerEngine format
    class WerkzeugFormatter(logging.Formatter):
        def format(self, record):
            # Format: "FLASK    message"
            message = record.getMessage()
            # Pad to match LoggerEngine's "INFO     " (8 chars + space)
            return f"FLASK    {message}"

    # Show logs with custom format
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(WerkzeugFormatter())
    werkzeug_logger.addHandler(handler)

    # Set log level from config
    log_level = getattr(logging, level.upper(), logging.INFO)
    werkzeug_logger.setLevel(log_level)

def register_error_handlers(app):
    """Register error handlers"""

    @app.errorhandler(404)
    def not_found(error):
        return render_template('error.html',
                             error_code=404,
                             error_message='Page not found'), 404

    @app.errorhandler(500)
    def server_error(error):
        return render_template('error.html',
                             error_code=500,
                             error_message='Internal server error'), 500

# ═══════════════════════════════════════════════════════════════════════════════
# CLI Commands (for development)
# ═══════════════════════════════════════════════════════════════════════════════

# Create app instance at module level (for Flask CLI)
app = create_app()

@app.cli.command()
def init():
    """Initialize WebUI (create necessary directories)"""
    logger = app.config['WEBUI_CONFIG'].logger
    logger.info("Initializing WebUI...")

    # Create directories
    dirs = [
        'static/css',
        'static/js',
        'static/img',
        'templates'
    ]

    for dir_path in dirs:
        path = Path(__file__).parent / dir_path
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {dir_path}")

    logger.info("WebUI initialized!")

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ASCII-safe banner for Windows console
    banner = """
    ======================================================
    SuperBot WebUI v3.0 (Refactored)
    ======================================================

    Dashboard: http://localhost:5000

    Navigation:
      - Dashboard  : /
      - Trade      : /trade
      - Backtest   : /backtest
      - Analiz     : /analiz
      - Ayarlar    : /ayarlar
      - Favorites  : /favorites
      - Portfolio  : /portfolio

    Architecture:
      - Modular design with blueprints
      - Service layer for business logic
      - Clean separation of concerns

    ======================================================
    """
    #print(banner)

    import os

    # venv ve site-packages klasörlerini reloader'dan hariç tut
    extra_dirs = []
    exclude_patterns = ['venv', 'site-packages', '__pycache__', '.git', 'node_modules']

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True,
        reloader_type='stat',  # watchdog yerine stat kullan (daha stabil)
        exclude_patterns=exclude_patterns
    )
