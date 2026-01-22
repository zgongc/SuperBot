"""Account API endpoints"""
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response

def get_account_service():
    """Get account service from app context"""
    from flask import current_app
    return current_app.account_service

def register_routes(bp):
    """Register account routes"""

    @bp.route('/account/balance')
    def get_account_balance():
        """GET /api/account/balance - Get account balance from database"""
        try:
            account_type = request.args.get('account_type', 'SPOT')

            service = get_account_service()
            account_info = run_async(service.get_account_balance(account_type))

            if not account_info:
                return error_response(
                    'No account info available. Please sync first using /api/account/sync',
                    404
                )

            return success_response(account_info)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/account/sync', methods=['POST'])
    def sync_account():
        """POST /api/account/sync - Sync account balance from exchange"""
        try:
            data = request.get_json() or {}
            account_type = data.get('account_type', 'SPOT')

            service = get_account_service()
            success = run_async(service.sync_account_balance(account_type))

            if success:
                return success_response(
                    message=f'{account_type} account info synced successfully'
                )
            else:
                return error_response('Failed to save account info', 500)
        except Exception as e:
            return error_response(str(e), 500)
