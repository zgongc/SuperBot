"""Alerts API endpoints"""

import json
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response


def get_data_manager():
    """Get data_manager from app context"""
    from flask import current_app
    return current_app.data_manager


def register_routes(bp):
    """Register alert routes"""

    @bp.route('/alerts', methods=['GET'])
    def list_alerts():
        """GET /api/alerts - List all alerts with optional filters"""
        try:
            # Get query parameters
            scope_type = request.args.get('scope_type')  # 'symbol' or 'category'
            alert_type = request.args.get('alert_type')
            is_active = request.args.get('is_active')
            symbol_id = request.args.get('symbol_id', type=int)
            category_id = request.args.get('category_id', type=int)

            # Build query
            query = "SELECT * FROM analysis_alerts WHERE 1=1"
            params = []

            if scope_type:
                query += " AND scope_type = ?"
                params.append(scope_type)

            if alert_type:
                query += " AND alert_type = ?"
                params.append(alert_type)

            if is_active is not None:
                query += " AND is_active = ?"
                params.append(1 if is_active == 'true' else 0)

            if symbol_id:
                query += " AND symbol_id = ?"
                params.append(symbol_id)

            if category_id:
                query += " AND category_id = ?"
                params.append(category_id)

            query += " ORDER BY created_at DESC"

            dm = get_data_manager()
            alerts = run_async(dm.fetch_all(query, tuple(params)))

            # Enrich with symbol/category names
            for alert in alerts:
                if alert['symbol_id']:
                    symbol = run_async(dm.fetch_one(
                        "SELECT symbol FROM exchange_symbols WHERE id = ?",
                        (alert['symbol_id'],)
                    ))
                    alert['symbol_name'] = symbol['symbol'] if symbol else None

                if alert['category_id']:
                    category = run_async(dm.fetch_one(
                        "SELECT name FROM categories WHERE id = ?",
                        (alert['category_id'],)
                    ))
                    alert['category_name'] = category['name'] if category else None

            return success_response({
                'alerts': alerts,
                'total': len(alerts)
            })

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/alerts', methods=['POST'])
    def create_alert():
        """POST /api/alerts - Create a new alert"""
        try:
            data = request.get_json()

            # Validate required fields
            required = ['name', 'scope_type', 'alert_type', 'conditions']
            for field in required:
                if field not in data:
                    return error_response(f"Missing required field: {field}", 400)

            # Validate scope
            scope_type = data['scope_type']
            if scope_type not in ['symbol', 'category']:
                return error_response("scope_type must be 'symbol' or 'category'", 400)

            if scope_type == 'symbol' and 'symbol_id' not in data:
                return error_response("symbol_id required for symbol scope", 400)

            if scope_type == 'category' and 'category_id' not in data:
                return error_response("category_id required for category scope", 400)

            # Validate alert_type
            valid_types = ['pattern', 'signal', 'score', 'trend_change', 'category_aggregate']
            if data['alert_type'] not in valid_types:
                return error_response(f"alert_type must be one of: {', '.join(valid_types)}", 400)

            # Validate conditions (must be dict/JSON)
            conditions = data['conditions']
            if isinstance(conditions, str):
                try:
                    json.loads(conditions)
                except:
                    return error_response("conditions must be valid JSON", 400)
            elif not isinstance(conditions, dict):
                return error_response("conditions must be a dict or JSON string", 400)

            # Convert conditions to JSON string
            conditions_json = json.dumps(conditions) if isinstance(conditions, dict) else conditions

            # Insert alert
            query = """
                INSERT INTO analysis_alerts (
                    name, description, scope_type, symbol_id, category_id,
                    alert_type, conditions,
                    notify_webui, notify_telegram, notify_email,
                    is_active, cooldown_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            dm = get_data_manager()
            alert_id = run_async(dm.execute(
                query,
                (
                    data['name'],
                    data.get('description'),
                    scope_type,
                    data.get('symbol_id'),
                    data.get('category_id'),
                    data['alert_type'],
                    conditions_json,
                    data.get('notify_webui', True),
                    data.get('notify_telegram', False),
                    data.get('notify_email', False),
                    data.get('is_active', True),
                    data.get('cooldown_minutes', 60)
                )
            ))

            return success_response({
                'alert_id': alert_id,
                'message': 'Alert created successfully'
            })

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/alerts/<int:alert_id>', methods=['GET'])
    def get_alert(alert_id):
        """GET /api/alerts/:id - Get a single alert"""
        try:
            dm = get_data_manager()
            alert = run_async(dm.fetch_one(
                "SELECT * FROM analysis_alerts WHERE id = ?",
                (alert_id,)
            ))

            if not alert:
                return error_response("Alert not found", 404)

            # Enrich with symbol/category names
            if alert['symbol_id']:
                symbol = run_async(dm.fetch_one(
                    "SELECT symbol FROM exchange_symbols WHERE id = ?",
                    (alert['symbol_id'],)
                ))
                alert['symbol_name'] = symbol['symbol'] if symbol else None

            if alert['category_id']:
                category = run_async(dm.fetch_one(
                    "SELECT name FROM categories WHERE id = ?",
                    (alert['category_id'],)
                ))
                alert['category_name'] = category['name'] if category else None

            return success_response(alert)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/alerts/<int:alert_id>', methods=['PUT'])
    def update_alert(alert_id):
        """PUT /api/alerts/:id - Update an alert"""
        try:
            data = request.get_json()

            # Check if alert exists
            dm = get_data_manager()
            existing = run_async(dm.fetch_one(
                "SELECT id FROM analysis_alerts WHERE id = ?",
                (alert_id,)
            ))

            if not existing:
                return error_response("Alert not found", 404)

            # Build update query dynamically
            allowed_fields = [
                'name', 'description', 'conditions', 'is_active',
                'notify_webui', 'notify_telegram', 'notify_email', 'cooldown_minutes'
            ]

            updates = []
            params = []

            for field in allowed_fields:
                if field in data:
                    value = data[field]

                    # Special handling for conditions
                    if field == 'conditions':
                        if isinstance(value, dict):
                            value = json.dumps(value)

                    updates.append(f"{field} = ?")
                    params.append(value)

            if not updates:
                return error_response("No fields to update", 400)

            # Add updated_at
            updates.append("updated_at = ?")
            params.append(run_async(dm.execute("SELECT datetime('now')")))

            # Add alert_id
            params.append(alert_id)

            query = f"UPDATE analysis_alerts SET {', '.join(updates)} WHERE id = ?"
            run_async(dm.execute(query, tuple(params)))

            return success_response({
                'alert_id': alert_id,
                'message': 'Alert updated successfully'
            })

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/alerts/<int:alert_id>', methods=['DELETE'])
    def delete_alert(alert_id):
        """DELETE /api/alerts/:id - Delete an alert"""
        try:
            dm = get_data_manager()

            # Check if exists
            existing = run_async(dm.fetch_one(
                "SELECT id FROM analysis_alerts WHERE id = ?",
                (alert_id,)
            ))

            if not existing:
                return error_response("Alert not found", 404)

            # Delete (CASCADE will handle notifications)
            run_async(dm.execute(
                "DELETE FROM analysis_alerts WHERE id = ?",
                (alert_id,)
            ))

            return success_response({
                'message': 'Alert deleted successfully'
            })

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/alerts/<int:alert_id>/notifications', methods=['GET'])
    def get_alert_notifications(alert_id):
        """GET /api/alerts/:id/notifications - Get notification history for an alert"""
        try:
            limit = request.args.get('limit', 10, type=int)
            offset = request.args.get('offset', 0, type=int)

            dm = get_data_manager()

            # Get notifications
            query = """
                SELECT * FROM alert_notifications
                WHERE alert_id = ?
                ORDER BY triggered_at DESC
                LIMIT ? OFFSET ?
            """

            notifications = run_async(dm.fetch_all(query, (alert_id, limit, offset)))

            # Get total count
            count_query = "SELECT COUNT(*) as total FROM alert_notifications WHERE alert_id = ?"
            total_result = run_async(dm.fetch_one(count_query, (alert_id,)))
            total = total_result['total'] if total_result else 0

            return success_response({
                'notifications': notifications,
                'total': total,
                'limit': limit,
                'offset': offset
            })

        except Exception as e:
            return error_response(str(e), 500)
