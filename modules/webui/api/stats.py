"""Stats API endpoints"""
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response

def get_stats_service():
    """Get stats service from app context"""
    from flask import current_app
    return current_app.stats_service

def register_routes(bp):
    """Register stats routes"""

    @bp.route('/stats/overview')
    def get_stats_overview():
        """GET /api/stats/overview - Get dashboard overview statistics"""
        try:
            service = get_stats_service()
            result = run_async(service.get_overview_stats())

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/stats')
    def get_stats():
        """GET /api/stats - Basic stats (legacy endpoint)"""
        # TODO: Get real stats
        from flask import jsonify
        return jsonify({
            'status': 'success',
            'data': {
                'total_opportunities': 5,
                'high_score_count': 2,
                'medium_score_count': 3,
                'avg_score': 78.3,
                'last_scan': '2025-10-27 21:30:00'
            }
        })
