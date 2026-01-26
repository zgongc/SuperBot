"""
modules/webui/api/exchanges.py
SuperBot - Exchange Accounts API Endpoints
Author: SuperBot Team
Date: 2025-10-30
Versiyon: 1.0.0

Exchange account management REST API
"""

from flask import Blueprint, request, jsonify, current_app
from modules.webui.helpers.async_helper import run_async
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)

bp = Blueprint('exchanges_api', __name__)


def get_exchange_service():
    """Get exchange service from app context"""
    return current_app.exchange_service


def success_response(data=None, message=None):
    """Success response helper"""
    response = {'status': 'success'}
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response)


def error_response(message, status_code=400):
    """Error response helper"""
    return jsonify({
        'status': 'error',
        'message': message
    }), status_code


# ============================================
# Exchange Account CRUD
# ============================================

@bp.route('/exchanges', methods=['GET'])
def get_all_exchanges():
    """GET /api/exchanges - Get all exchange accounts"""
    try:
        service = get_exchange_service()
        accounts = run_async(service.get_all_exchange_accounts())
        return success_response({'accounts': accounts})
    except Exception as e:
        logger.error(f"❌ Get all exchanges error: {e}")
        return error_response(str(e), 500)


@bp.route('/exchanges/<int:account_id>', methods=['GET'])
def get_exchange(account_id):
    """GET /api/exchanges/:id - Get exchange account by ID"""
    try:
        service = get_exchange_service()
        account = run_async(service.get_exchange_account_by_id(account_id))

        if not account:
            return error_response('Exchange account not found', 404)

        return success_response({'account': account})
    except Exception as e:
        logger.error(f"❌ Get exchange error: {e}")
        return error_response(str(e), 500)


@bp.route('/exchanges', methods=['POST'])
def create_exchange():
    """POST /api/exchanges - Create new exchange account"""
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('name'):
            return error_response('Name is required', 400)

        if not data.get('exchange'):
            return error_response('Exchange is required', 400)

        # Validate exchange type
        valid_exchanges = ['binance', 'kucoin', 'gateio', 'manual']
        if data['exchange'] not in valid_exchanges:
            return error_response(f'Invalid exchange. Must be one of: {", ".join(valid_exchanges)}', 400)

        # Validate environment
        valid_environments = ['production', 'testnet']
        environment = data.get('environment', 'production')
        if environment not in valid_environments:
            return error_response(f'Invalid environment. Must be one of: {", ".join(valid_environments)}', 400)

        service = get_exchange_service()
        account_id = run_async(service.create_exchange_account(
            name=data['name'],
            exchange=data['exchange'],
            environment=environment,
            account_type=data.get('account_type', 'spot'),
            api_key=data.get('api_key'),
            api_secret=data.get('api_secret'),
            passphrase=data.get('passphrase'),
            settings=data.get('settings'),
            notes=data.get('notes')
        ))

        if account_id:
            # Auto-create portfolio for this exchange
            try:
                from .portfolio import get_portfolio_service
                portfolio_service = get_portfolio_service()
                portfolio_id = run_async(portfolio_service.create_portfolio(
                    name=data['name'],  # Same name as exchange
                    exchange_account_id=account_id
                ))
                logger.info(f"✅ Auto-created portfolio {portfolio_id} for exchange account {account_id}")
            except Exception as portfolio_err:
                logger.warning(f"⚠️ Failed to auto-create portfolio: {portfolio_err}")
                # Don't fail the exchange creation if portfolio creation fails

            return success_response(
                {'account_id': account_id},
                f'Exchange account "{data["name"]}" created successfully'
            ), 201
        else:
            return error_response('Failed to create exchange account', 500)

    except Exception as e:
        logger.error(f"❌ Create exchange error: {e}")
        return error_response(str(e), 500)


@bp.route('/exchanges/<int:account_id>', methods=['PUT'])
def update_exchange(account_id):
    """PUT /api/exchanges/:id - Update exchange account"""
    try:
        data = request.get_json()

        # Validate exchange if provided
        if 'exchange' in data:
            valid_exchanges = ['binance', 'bybit', 'okx', 'manual']
            if data['exchange'] not in valid_exchanges:
                return error_response(f'Invalid exchange. Must be one of: {", ".join(valid_exchanges)}', 400)

        # Validate environment if provided
        if 'environment' in data:
            valid_environments = ['production', 'testnet']
            if data['environment'] not in valid_environments:
                return error_response(f'Invalid environment. Must be one of: {", ".join(valid_environments)}', 400)

        # IMPORTANT: Remove empty/null API credentials from update data
        # If user doesn't provide new keys, we should keep the existing ones
        filtered_data = {}
        for key, value in data.items():
            # Skip empty API credentials
            if key in ['api_key', 'api_secret', 'passphrase']:
                if value and value.strip():  # Only include if non-empty
                    filtered_data[key] = value
                # If empty/null, don't include it (keep existing value in DB)
            else:
                filtered_data[key] = value

        service = get_exchange_service()
        success = run_async(service.update_exchange_account(account_id, **filtered_data))

        if success:
            return success_response(message='Exchange account updated successfully')
        else:
            return error_response('Failed to update exchange account', 500)

    except Exception as e:
        logger.error(f"❌ Update exchange error: {e}")
        return error_response(str(e), 500)


@bp.route('/exchanges/<int:account_id>', methods=['DELETE'])
def delete_exchange(account_id):
    """DELETE /api/exchanges/:id - Delete exchange account"""
    try:
        service = get_exchange_service()
        result = run_async(service.delete_exchange_account(account_id))

        if result['success']:
            return success_response(message='Exchange account deleted successfully')
        else:
            return error_response(
                result.get('error', 'Failed to delete exchange account'),
                400 if 'linked_portfolios' in result else 500
            )

    except Exception as e:
        logger.error(f"❌ Delete exchange error: {e}")
        return error_response(str(e), 500)


# ============================================
# Exchange Connection & Testing
# ============================================

@bp.route('/exchanges/<int:account_id>/test', methods=['POST'])
def test_exchange_connection(account_id):
    """POST /api/exchanges/:id/test - Test exchange connection"""
    try:
        service = get_exchange_service()
        result = run_async(service.test_connection(account_id))

        if result['success']:
            return success_response(
                {'connection_status': 'connected'},
                'Connection test successful'
            )
        else:
            return error_response(result.get('error', 'Connection test failed'), 400)

    except Exception as e:
        logger.error(f"❌ Test exchange connection error: {e}")
        return error_response(str(e), 500)


@bp.route('/exchanges/<int:account_id>/balance', methods=['GET'])
def get_exchange_balance(account_id):
    """GET /api/exchanges/:id/balance - Get exchange account balance"""
    try:
        service = get_exchange_service()
        result = run_async(service.get_exchange_balance(account_id))

        if result['success']:
            return success_response(result.get('data', {}))
        else:
            return error_response(result.get('error', 'Failed to fetch balance'), 400)

    except Exception as e:
        logger.error(f"❌ Get exchange balance error: {e}")
        return error_response(str(e), 500)


# ============================================
# Statistics & Analytics
# ============================================

@bp.route('/exchanges/summary', methods=['GET'])
def get_exchanges_summary():
    """GET /api/exchanges/summary - Get exchange accounts summary statistics"""
    try:
        service = get_exchange_service()
        summary = run_async(service.get_exchange_accounts_summary())
        return success_response({'summary': summary})

    except Exception as e:
        logger.error(f"❌ Get exchanges summary error: {e}")
        return error_response(str(e), 500)


# ============================================
# Import/Sync Operations (Phase 7.5)
# ============================================

@bp.route('/exchanges/<int:account_id>/sync/<int:portfolio_id>', methods=['POST'])
def sync_positions(account_id, portfolio_id):
    """POST /api/exchanges/:id/sync/:portfolio_id - Sync positions from exchange"""
    try:
        service = get_exchange_service()
        result = run_async(service.sync_positions(account_id, portfolio_id))

        if result['success']:
            return success_response(
                result.get('data', {}),
                'Positions synced successfully'
            )
        else:
            return error_response(result.get('error', 'Sync failed'), 400)

    except Exception as e:
        logger.error(f"❌ Sync positions error: {e}")
        return error_response(str(e), 500)
