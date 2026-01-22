"""Analysis API endpoints"""
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response

def get_analysis_service():
    """Get analysis service from app context"""
    from flask import current_app
    return current_app.analysis_service

def register_routes(bp):
    """Register analysis routes"""

    @bp.route('/analysis/queue', methods=['POST'])
    def queue_analysis():
        """POST /api/analysis/queue - Queue symbol for analysis"""
        try:
            data = request.get_json()
            symbol_ids = data.get('symbol_ids', [])
            symbols = data.get('symbols', [])
            timeframe = data.get('timeframe', '1h')
            strategy_name = data.get('strategy_name')

            service = get_analysis_service()
            result = run_async(service.queue_analysis(
                symbol_ids, symbols, timeframe, strategy_name
            ))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/analysis/queue', methods=['GET'])
    def get_analysis_queue():
        """GET /api/analysis/queue - Get analysis queue"""
        try:
            status = request.args.get('status')
            limit = request.args.get('limit', 100, type=int)

            service = get_analysis_service()
            result = run_async(service.get_analysis_queue(status, limit))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/analysis/results', methods=['GET'])
    def get_analysis_results():
        """GET /api/analysis/results - Get analysis results"""
        try:
            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe')
            signal = request.args.get('signal')
            min_confidence = request.args.get('min_confidence', type=float)
            limit = request.args.get('limit', 100, type=int)

            service = get_analysis_service()
            result = run_async(service.get_analysis_results(
                symbol, timeframe, signal, min_confidence, limit
            ))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)
