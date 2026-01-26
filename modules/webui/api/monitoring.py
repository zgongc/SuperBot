"""Monitoring API endpoints"""
from flask import jsonify
from ..helpers.response_helper import success_response, error_response


def get_monitoring_service():
    """Get monitoring service from app context"""
    from flask import current_app
    return current_app.monitoring_service


def register_routes(bp):
    """Register monitoring routes"""

    @bp.route('/monitoring/resources')
    def get_resources():
        """GET /api/monitoring/resources - Get current resource usage"""
        try:
            service = get_monitoring_service()
            result = service.get_current()
            return jsonify(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/monitoring/history')
    def get_history():
        """GET /api/monitoring/history - Get resource history for charts"""
        from flask import request
        try:
            service = get_monitoring_service()
            limit = request.args.get('limit', 60, type=int)
            result = service.get_history(limit)
            return jsonify(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/monitoring/process')
    def get_process():
        """GET /api/monitoring/process - Get current process info"""
        try:
            service = get_monitoring_service()
            result = service.get_process_info()
            return jsonify(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/monitoring/system')
    def get_system():
        """GET /api/monitoring/system - Get system information"""
        try:
            service = get_monitoring_service()
            result = service.get_system_info()
            return jsonify(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/monitoring/alerts/reset', methods=['POST'])
    def reset_alerts():
        """POST /api/monitoring/alerts/reset - Reset alert counters"""
        try:
            service = get_monitoring_service()
            service.reset_alerts()
            return success_response({'message': 'Alerts reset'})
        except Exception as e:
            return error_response(str(e), 500)
