"""
Candlestick Pattern Analysis API endpoints

Pattern detection:
- Doji, Hammer, Engulfing, etc.
- Single and multi-candle patterns
"""

from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response


def get_patterns_service():
    """Get patterns analysis service from app context"""
    from flask import current_app
    if not hasattr(current_app, 'patterns_service'):
        from ..services.patterns_service import PatternsService
        current_app.patterns_service = PatternsService(
            parquets_engine=getattr(current_app, 'parquets_engine', None),
            logger=current_app.logger
        )
    return current_app.patterns_service


def register_routes(bp):
    """Register pattern analysis routes"""

    @bp.route('/patterns/analyze', methods=['POST'])
    def analyze_patterns():
        """
        POST /api/patterns/analyze - Pattern analysis

        Body:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            limit: 500 (default)
            start_date: 2025-01-01 (optional)
            end_date: 2025-01-31 (optional)

        Returns:
            {
                "summary": {...},
                "patterns": [...],
                "candles": [...],
                "annotations": {...}
            }
        """
        try:
            data = request.get_json() or {}

            symbol = data.get('symbol')
            if not symbol:
                return error_response('symbol is required', 400)

            timeframe = data.get('timeframe', '5m')
            limit = data.get('limit', 500)
            start_date = data.get('start_date')
            end_date = data.get('end_date')

            service = get_patterns_service()
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

    @bp.route('/patterns/info', methods=['GET'])
    def get_pattern_info():
        """
        GET /api/patterns/info - Get pattern definitions

        Returns:
            {
                "hammer": {"name": "Hammer", "type": "bullish", ...},
                ...
            }
        """
        try:
            service = get_patterns_service()
            result = run_async(service.get_pattern_info())
            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)
