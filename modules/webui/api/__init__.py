"""API Blueprint initialization"""
from flask import Blueprint

def create_api_blueprint():
    """Create and configure API blueprint"""
    api = Blueprint('api', __name__, url_prefix='/api')

    # Import and register sub-routes
    from . import symbols, favorites, analysis, portfolio, account, stats, theme, categories, alerts, notifications, settings, exchanges, replay, smc_analysis, data_download

    symbols.register_routes(api)
    favorites.register_routes(api)
    analysis.register_routes(api)
    portfolio.register_routes(api)
    account.register_routes(api)
    stats.register_routes(api)
    theme.register_routes(api)
    categories.register_routes(api)
    alerts.register_routes(api)
    notifications.register_routes(api)
    settings.register_routes(api)
    replay.register_routes(api)
    smc_analysis.register_routes(api)
    data_download.register_routes(api)

    # Register exchanges blueprint
    api.register_blueprint(exchanges.bp)

    return api

def get_strategy_api_blueprint():
    """Get strategies API blueprint"""
    from . import strategies
    return strategies.bp
