"""
Notifications API - Handle WebUI toast notifications
"""

from flask import Blueprint, jsonify
from ..helpers.response_helper import success_response, error_response
from components.alerts.notification_dispatcher import NotificationDispatcher


bp = Blueprint('notifications_api', __name__, url_prefix='/notifications')


@bp.route('/pending', methods=['GET'])
def get_pending_notifications():
    """
    Get pending WebUI notifications

    Returns all pending notifications and clears the queue
    """
    try:
        # Get all pending notifications
        notifications = NotificationDispatcher.pending_webui_notifications.copy()

        # Clear the queue (they've been delivered)
        NotificationDispatcher.pending_webui_notifications.clear()

        return success_response({
            'notifications': notifications,
            'count': len(notifications)
        })

    except Exception as e:
        return error_response(f'Failed to get notifications: {str(e)}')


@bp.route('/history', methods=['GET'])
def get_notification_history():
    """
    Get notification history from database

    Query params:
    - limit: max notifications to return (default 20)
    - offset: pagination offset (default 0)
    """
    try:
        from flask import request, current_app
        from ..helpers.async_helper import run_async
        import json

        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))

        # Get data manager from Flask app
        data_manager = current_app.config.get('data_manager')
        if not data_manager:
            return error_response('Data manager not available')

        # Query notification history
        query = """
            SELECT
                an.id,
                an.alert_id,
                an.analysis_result_id,
                an.trigger_type,
                an.trigger_details,
                an.triggered_at,
                an.webui_sent,
                an.webui_sent_at,
                an.is_read,
                an.read_at,
                aa.name as alert_name,
                aa.description as alert_description,
                es.symbol
            FROM alert_notifications an
            LEFT JOIN analysis_alerts aa ON an.alert_id = aa.id
            LEFT JOIN exchange_symbols es ON aa.symbol_id = es.id
            WHERE an.webui_sent = 1
            ORDER BY an.triggered_at DESC
            LIMIT ? OFFSET ?
        """

        notifications = run_async(data_manager.fetch_all(query, (limit, offset)))

        # Format for frontend
        formatted = []
        for n in notifications:
            # Safely parse trigger_details
            try:
                trigger_details = json.loads(n['trigger_details']) if n['trigger_details'] else {}
            except (json.JSONDecodeError, TypeError):
                trigger_details = {}

            # Parse trigger_details for title/message if alert info not available
            title = f"Alert: {n['alert_name']}" if n.get('alert_name') else (n['trigger_details'] or 'Notification')
            message = n.get('alert_description') or f"{n['trigger_type']} on {n.get('symbol') or 'category'}"

            formatted.append({
                'id': n['id'],
                'notification_id': n['id'],
                'title': title,
                'message': message,
                'type': 'alert',
                'alert_type': n['trigger_type'],
                'trigger_type': n['trigger_type'],
                'symbol': n.get('symbol'),
                'analysis_result_id': n['analysis_result_id'],
                'timestamp': n['triggered_at'],
                'created_at': n['triggered_at'],
                'is_read': bool(n['is_read']),
                'read_at': n.get('read_at'),
                'trigger_details': trigger_details
            })

        return success_response({
            'notifications': formatted,
            'count': len(formatted)
        })

    except Exception as e:
        return error_response(f'Failed to get history: {str(e)}')


@bp.route('/<int:notification_id>/read', methods=['POST'])
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    try:
        from flask import current_app
        from ..helpers.async_helper import run_async

        data_manager = current_app.config.get('data_manager')
        if not data_manager:
            return error_response('Data manager not available')

        success = run_async(data_manager.mark_notification_read(notification_id))

        if success:
            return success_response({'message': 'Notification marked as read'})
        else:
            return error_response('Failed to mark as read')

    except Exception as e:
        return error_response(f'Failed to mark as read: {str(e)}')


@bp.route('/mark-all-read', methods=['POST'])
def mark_all_read():
    """Mark all notifications as read"""
    try:
        from flask import current_app
        from ..helpers.async_helper import run_async

        data_manager = current_app.config.get('data_manager')
        if not data_manager:
            return error_response('Data manager not available')

        success = run_async(data_manager.mark_all_notifications_read())

        if success:
            return success_response({'message': 'All notifications marked as read'})
        else:
            return error_response('Failed to mark all as read')

    except Exception as e:
        return error_response(f'Failed to mark all as read: {str(e)}')


@bp.route('/<int:notification_id>', methods=['DELETE'])
def delete_notification(notification_id):
    """Delete a notification"""
    try:
        from flask import current_app
        from ..helpers.async_helper import run_async

        data_manager = current_app.config.get('data_manager')
        if not data_manager:
            return error_response('Data manager not available')

        success = run_async(data_manager.delete_notification(notification_id))

        if success:
            return success_response({'message': 'Notification deleted'})
        else:
            return error_response('Failed to delete notification')

    except Exception as e:
        return error_response(f'Failed to delete notification: {str(e)}')


@bp.route('/unread-count', methods=['GET'])
def get_unread_count():
    """Get unread notification count"""
    try:
        from flask import current_app
        from ..helpers.async_helper import run_async

        data_manager = current_app.config.get('data_manager')
        if not data_manager:
            return error_response('Data manager not available')

        count = run_async(data_manager.get_unread_notification_count())

        return success_response({'count': count})

    except Exception as e:
        return error_response(f'Failed to get unread count: {str(e)}')


def register_routes(api_blueprint):
    """Register notification routes with the main API blueprint"""
    api_blueprint.register_blueprint(bp)
