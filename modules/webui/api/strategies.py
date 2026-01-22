"""
Strategy Management API Endpoints
Provides REST API for strategy CRUD operations
"""

from flask import Blueprint, jsonify, request
from ..helpers.async_helper import run_async
from ..services.strategy_ui_service import StrategyUIService
from ..helpers.response_helper import success_response, error_response

bp = Blueprint('api_strategies', __name__, url_prefix='/api/strategies')


def get_strategy_service():
    """Get StrategyUIService instance from Flask app context"""
    from flask import current_app
    return current_app.config['STRATEGY_UI_SERVICE']


@bp.route('/', methods=['GET'])
def get_strategies():
    """
    Get all strategies
    Query params:
        - status: filter by status (active/inactive)
    """
    try:
        service = get_strategy_service()
        status_filter = request.args.get('status')

        strategies = run_async(service.get_all_strategies(status_filter))

        return success_response(data={
            'strategies': strategies,
            'total': len(strategies)
        })
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>', methods=['GET'])
def get_strategy(strategy_id):
    """Get single strategy details"""
    try:
        service = get_strategy_service()
        strategy = run_async(service.get_strategy_by_id(strategy_id))

        if not strategy:
            return error_response('Strateji bulunamadı', 404)

        return success_response(data=strategy)
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/', methods=['POST'])
def create_strategy():
    """
    Create new strategy
    Body: JSON with strategy configuration
    """
    try:
        service = get_strategy_service()
        data = request.get_json()

        if not data:
            return error_response('Geçersiz veri', 400)

        result = run_async(service.create_strategy(data))

        return success_response(
            message='Strateji başarıyla oluşturuldu',
            data=result
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>', methods=['PUT'])
def update_strategy(strategy_id):
    """
    Update existing strategy
    Body: JSON with updated configuration
    """
    try:
        service = get_strategy_service()
        data = request.get_json()

        if not data:
            return error_response('Geçersiz veri', 400)

        result = run_async(service.update_strategy(strategy_id, data))

        if not result:
            return error_response('Strateji bulunamadı', 404)

        return success_response(
            message='Strateji başarıyla güncellendi',
            data=result
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete strategy"""
    try:
        service = get_strategy_service()

        success = run_async(service.delete_strategy(strategy_id))

        if not success:
            return error_response('Strateji bulunamadı', 404)

        return success_response(message='Strateji başarıyla silindi')
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/activate', methods=['POST'])
def activate_strategy(strategy_id):
    """Activate strategy"""
    try:
        service = get_strategy_service()

        result = run_async(service.set_strategy_status(strategy_id, True))

        if not result:
            return error_response('Strateji bulunamadı', 404)

        return success_response(
            message='Strateji aktifleştirildi',
            data=result
        )
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/deactivate', methods=['POST'])
def deactivate_strategy(strategy_id):
    """Deactivate strategy"""
    try:
        service = get_strategy_service()

        result = run_async(service.set_strategy_status(strategy_id, False))

        if not result:
            return error_response('Strateji bulunamadı', 404)

        return success_response(
            message='Strateji devre dışı bırakıldı',
            data=result
        )
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/duplicate', methods=['POST'])
def duplicate_strategy(strategy_id):
    """Duplicate strategy"""
    try:
        service = get_strategy_service()
        data = request.get_json() or {}
        new_name = data.get('name')

        result = run_async(service.duplicate_strategy(strategy_id, new_name))

        if not result:
            return error_response('Strateji bulunamadı', 404)

        return success_response(
            message='Strateji kopyalandı',
            data=result
        )
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/templates', methods=['GET'])
def get_templates():
    """Get all available strategy templates"""
    try:
        service = get_strategy_service()

        templates = run_async(service.get_available_templates())

        return success_response(data={
            'templates': templates,
            'total': len(templates)
        })
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/templates/basic', methods=['GET'])
def get_basic_templates():
    """Get basic starter templates from templates/basic/ folder"""
    try:
        service = get_strategy_service()

        templates = run_async(service.get_basic_templates())

        return success_response(data={
            'templates': templates,
            'total': len(templates)
        })
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/config', methods=['GET'])
def get_strategy_config(strategy_id):
    """Get strategy configuration (YAML format)"""
    try:
        service = get_strategy_service()

        config = run_async(service.get_strategy_config(strategy_id))

        if not config:
            return error_response('Strateji bulunamadı', 404)

        return success_response(data={'config': config})
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/validate', methods=['POST'])
def validate_strategy(strategy_id):
    """
    Validate strategy configuration
    Body: Optional JSON with configuration to validate
    """
    try:
        service = get_strategy_service()
        data = request.get_json()

        result = run_async(service.validate_strategy(strategy_id, data))

        return success_response(data=result)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/load', methods=['GET'])
def load_strategy(strategy_id):
    """
    Load strategy configuration for editing in UI

    Converts strategy object to UI-compatible JSON format with all parameters
    """
    try:
        service = get_strategy_service()

        result = run_async(service.load_strategy(strategy_id))

        if not result:
            return error_response('Strateji bulunamadı veya yüklenemedi', 404)

        return success_response(
            message='Strateji başarıyla yüklendi',
            data=result
        )
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/<strategy_id>/exists', methods=['GET'])
def check_strategy_exists(strategy_id):
    """
    Check if a strategy file already exists

    Used for preventing accidental overwrites when creating from template
    """
    try:
        service = get_strategy_service()

        exists = run_async(service.check_strategy_exists(strategy_id))

        return success_response(data={'exists': exists})
    except Exception as e:
        return error_response(str(e), 500)
