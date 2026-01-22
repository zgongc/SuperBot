"""Favorites API endpoints"""
from flask import request
from datetime import datetime
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response
from ..helpers.validation import validate_required_fields

def get_favorites_service():
    """Get favorites service from app context"""
    from flask import current_app
    return current_app.favorites_service

def register_routes(bp):
    """Register favorites routes"""

    @bp.route('/favorites', methods=['GET'])
    def get_favorites():
        """GET /api/favorites - Get user favorites"""
        try:
            sort_by = request.args.get('sort_by', 'priority')
            tags_str = request.args.get('tags', '')
            tags = tags_str.split(',') if tags_str else None

            service = get_favorites_service()
            result = run_async(service.get_favorites(sort_by, tags))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/favorites/add', methods=['POST'])
    def add_favorite():
        """POST /api/favorites/add - Add symbol to favorites"""
        try:
            data = request.get_json()

            # Validate required fields
            is_valid, error_msg = validate_required_fields(data, ['symbol', 'base_asset'])
            if not is_valid:
                return error_response(error_msg, 400)

            service = get_favorites_service()
            favorite_id = run_async(service.add_favorite(
                symbol=data['symbol'],
                base_asset=data['base_asset'],
                quote_asset=data.get('quote_asset', 'USDT'),
                market_type=data.get('market_type', 'SPOT'),
                tags=data.get('tags', []),
                priority=data.get('priority', 5),
                notes=data.get('notes', ''),
                color=data.get('color')
            ))

            if favorite_id:
                return success_response({
                    'favorite_id': favorite_id,
                    'symbol': data['symbol'],
                    'added_at': datetime.utcnow().isoformat()
                })
            else:
                return error_response('Failed to add favorite', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/favorites/bulk-add', methods=['POST'])
    def bulk_add_favorites():
        """POST /api/favorites/bulk-add - Add multiple favorites"""
        try:
            data = request.get_json()
            symbols = data.get('symbols', [])

            if not symbols:
                return error_response('No symbols provided', 400)

            service = get_favorites_service()
            result = run_async(service.bulk_add_favorites(symbols))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/favorites/<int:favorite_id>', methods=['PATCH', 'PUT'])
    def update_favorite(favorite_id):
        """PATCH/PUT /api/favorites/<id> - Update favorite"""
        try:
            data = request.get_json()

            service = get_favorites_service()
            success = run_async(service.update_favorite(
                favorite_id=favorite_id,
                tags=data.get('tags'),
                notes=data.get('notes'),
                priority=data.get('priority'),
                color=data.get('color')
            ))

            if success:
                return success_response(message='Favorite updated')
            else:
                return error_response('Favorite not found', 404)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/favorites/<int:favorite_id>', methods=['DELETE'])
    def delete_favorite(favorite_id):
        """DELETE /api/favorites/<id> - Delete favorite"""
        try:
            service = get_favorites_service()
            success = run_async(service.delete_favorite(favorite_id))

            if success:
                return success_response(message='Favorite deleted')
            else:
                return error_response('Favorite not found', 404)
        except Exception as e:
            return error_response(str(e), 500)
