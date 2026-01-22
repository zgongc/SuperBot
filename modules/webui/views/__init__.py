"""Views blueprint initialization"""
from flask import Blueprint

def create_views_blueprint():
    """Create and configure views blueprint"""
    views = Blueprint('views', __name__)

    # Import route handlers
    from . import main, trading, settings, favorites, portfolio, replay, smc

    # Register routes
    main.register_routes(views)
    trading.register_routes(views)
    settings.register_routes(views)
    favorites.register_routes(views)
    portfolio.register_routes(views)
    replay.register_routes(views)
    smc.register_routes(views)

    return views

def get_strategy_blueprint():
    """Get strategies blueprint"""
    from . import strategies
    return strategies.bp
