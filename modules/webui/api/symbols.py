"""Symbols API endpoints"""
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response

def get_symbols_service():
    """Get symbols service from app context"""
    from flask import current_app
    return current_app.symbols_service

def register_routes(bp):
    """Register symbols routes"""

    @bp.route('/symbols/sync', methods=['POST'])
    def sync_symbols():
        """POST /api/symbols/sync - Sync symbols from exchange"""
        try:
            data = request.get_json() or {}
            market_type = data.get('market_type', 'both')
            force = data.get('force', False)

            service = get_symbols_service()
            result = run_async(service.sync_from_exchange(market_type, force))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/symbols/available')
    def available_symbols():
        """GET /api/symbols/available - Get available symbols"""
        try:
            market_type = request.args.get('market_type')
            quote_asset = request.args.get('quote_asset', 'USDT')
            search = request.args.get('search', '')
            limit = request.args.get('limit', 1000, type=int)

            service = get_symbols_service()
            result = run_async(service.get_available_symbols(
                market_type, quote_asset, search, limit
            ))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/symbols/pool')
    def symbols_pool():
        """GET /api/symbols/pool - Get all symbols (paginated)"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            search = request.args.get('search', '')

            service = get_symbols_service()
            result = run_async(service.get_symbols_pool(page, per_page, search))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/symbols/bulk-delete', methods=['DELETE'])
    def bulk_delete_symbols():
        """DELETE /api/symbols/bulk-delete - Delete multiple symbols"""
        try:
            data = request.get_json()
            symbol_ids = data.get('symbol_ids', [])

            if not symbol_ids:
                return error_response('No symbol IDs provided', 400)

            service = get_symbols_service()
            result = run_async(service.bulk_delete_symbols(symbol_ids))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)
