"""
Strategy Management Views
Handles all strategy-related pages (list, create, detail)
"""

from flask import Blueprint, render_template, request, redirect, url_for
from ..helpers.async_helper import run_async
from ..services.strategy_ui_service import StrategyUIService
from components.indicators import INDICATOR_REGISTRY

bp = Blueprint('strategies', __name__)


@bp.route('/strategies')
def strategies_list():
    """Strategy list page"""
    return render_template(
        'strategies.html',
        page='strategies',
        active_tab=request.args.get('tab', 'all')
    )


@bp.route('/strategies/create')
def strategy_create():
    """Strategy template selection page"""
    return render_template(
        'strategy_creator.html',
        page='strategies'
    )


@bp.route('/strategies/new/')
def strategy_new():
    """
    Create new strategy from template
    URL: /strategies/new/?load=template_name
    """
    template_name = request.args.get('load', 'empty_template')

    # Prepare simplified indicator registry for JavaScript
    indicator_registry_json = {}
    for key, data in INDICATOR_REGISTRY.items():
        indicator_registry_json[key] = {
            'description': data.get('description', ''),
            'default_params': data.get('default_params', {}),
            'output_keys': data.get('output_keys', []),
            'category': data.get('category', '').value if hasattr(data.get('category', ''), 'value') else str(data.get('category', ''))
        }

    return render_template(
        'strategy_edit_v2.html',
        page='strategies',
        strategy_id=template_name,
        is_new=True,
        indicator_registry_json=indicator_registry_json
    )


@bp.route('/strategies/<strategy_id>')
def strategy_detail(strategy_id):
    """Strategy detail page"""
    return render_template(
        'strategy_detail.html',
        page='strategies',
        strategy_id=strategy_id
    )


@bp.route('/strategies/edit/<strategy_id>')
def strategy_edit(strategy_id):
    """
    Edit existing strategy
    URL: /strategies/edit/strategy_name
    """
    # Prepare simplified indicator registry for JavaScript
    indicator_registry_json = {}
    for key, data in INDICATOR_REGISTRY.items():
        indicator_registry_json[key] = {
            'description': data.get('description', ''),
            'default_params': data.get('default_params', {}),
            'output_keys': data.get('output_keys', []),
            'category': data.get('category', '').value if hasattr(data.get('category', ''), 'value') else str(data.get('category', ''))
        }

    return render_template(
        'strategy_edit_v2.html',
        page='strategies',
        strategy_id=strategy_id,
        is_new=False,
        indicator_registry_json=indicator_registry_json
    )
