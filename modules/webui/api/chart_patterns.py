"""
Chart Pattern Analysis API endpoints

Geometric pattern detection:
- Double Top/Bottom
- Head & Shoulders
- Triangles (Ascending, Descending, Symmetric)
- Wedges, Channels
"""

from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response
from ..services.chart_pattern_service import load_analysis_config


def get_chart_pattern_service():
    """Get chart pattern service from app context"""
    from flask import current_app
    if not hasattr(current_app, 'chart_pattern_service'):
        from ..services.chart_pattern_service import ChartPatternService
        current_app.chart_pattern_service = ChartPatternService(
            parquets_engine=getattr(current_app, 'parquets_engine', None),
            logger=current_app.logger
        )
    return current_app.chart_pattern_service


def register_routes(bp):
    """Register chart pattern routes"""

    @bp.route('/chart-patterns/config', methods=['GET'])
    def get_chart_patterns_config():
        """
        GET /api/chart-patterns/config - Get chart pattern config

        Returns:
            {
                "enabled": true,
                "show": true,
                "patterns": {...},
                "price_tolerance_pct": 0.5,
                ...
            }
        """
        try:
            service = get_chart_pattern_service()
            result = run_async(service.get_config())
            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/chart-patterns/info', methods=['GET'])
    def get_chart_patterns_info():
        """
        GET /api/chart-patterns/info - Get pattern definitions

        Returns:
            {
                "patterns": {...},
                "description": {...}
            }
        """
        try:
            service = get_chart_pattern_service()
            result = run_async(service.get_pattern_info())
            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/chart-patterns/analyze', methods=['POST'])
    def analyze_chart_patterns():
        """
        POST /api/chart-patterns/analyze - Analyze chart patterns

        Body:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            limit: 500 (default)
            start_date: 2025-01-01 (optional)
            end_date: 2025-01-31 (optional)

        Returns:
            {
                "symbol": str,
                "timeframe": str,
                "bars": int,
                "patterns": [...],
                "swings": [...],
                "candles": [...],
                "annotations": {...},
                "stats": {...}
            }
        """
        try:
            data = request.get_json() or {}

            symbol = data.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = data.get('timeframe', '5m')
            limit = data.get('limit', 500)
            start_date = data.get('start_date')
            end_date = data.get('end_date')

            service = get_chart_pattern_service()
            result = run_async(service.analyze(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                start_date=start_date,
                end_date=end_date
            ))

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/chart-patterns/reload', methods=['POST'])
    def reload_chart_patterns_config():
        """
        POST /api/chart-patterns/reload - Reload config

        Reloads the chart pattern configuration from analysis.yaml
        """
        try:
            service = get_chart_pattern_service()
            service.reload_config()
            return success_response({'message': 'Config reloaded'})

        except Exception as e:
            return error_response(str(e), 500)
