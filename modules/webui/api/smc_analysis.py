"""
SMC (Smart Money Concepts) Analysis API endpoints

Market structure analizi:
- BOS (Break of Structure)
- CHoCH (Change of Character)
- FVG (Fair Value Gap)
- Swing High/Low
"""

from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response
from ..services.smc_service import load_analysis_config


def get_smc_service():
    """Get SMC analysis service from app context"""
    from flask import current_app
    if not hasattr(current_app, 'smc_service'):
        from ..services.smc_service import SMCService
        current_app.smc_service = SMCService(
            parquets_engine=getattr(current_app, 'parquets_engine', None),
            logger=current_app.logger
        )
    return current_app.smc_service


def register_routes(bp):
    """Register SMC analysis routes"""

    @bp.route('/smc/config', methods=['GET'])
    def get_config():
        """
        GET /api/smc/config - Get analysis config (show values)

        Returns:
            {
                "bos": {"show": true},
                "choch": {"show": true},
                "swing": {"show": true},
                "fvg": {"show": true},
                "orderblocks": {"show": false},
                "liquidity": {"show": false},
                "qml": {"show": false},
                "levels": {"show": true}
            }
        """
        try:
            config = load_analysis_config()

            # Extract show values for each detector
            show_config = {}
            detectors = ['swing', 'bos', 'choch', 'fvg', 'gap', 'orderblocks', 'liquidity', 'qml', 'ftr', 'levels']

            for detector in detectors:
                detector_config = config.get(detector, {})
                show_config[detector] = {
                    'show': detector_config.get('show', True),
                    'enabled': detector_config.get('enabled', True)
                }

            return success_response(show_config)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/smc/analyze', methods=['POST'])
    def analyze_symbol():
        """
        POST /api/smc/analyze - Symbol analysis

        Body:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            limit: 500 (default)
            start_date: 2025-01-01 (optional)
            end_date: 2025-01-31 (optional)

        Returns:
            {
                "summary": {...},
                "formations": {...},
                "levels": {...},
                "annotations": [...]
            }
        """
        try:
            data = request.get_json() or {}

            symbol = data.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = data.get('timeframe', '5m')
            limit = data.get('limit', 500)
            start_date = data.get('start_date')
            end_date = data.get('end_date')

            service = get_smc_service()
            result = run_async(service.analyze(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                start_date=start_date,
                end_date=end_date
            ))

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/smc/formations', methods=['GET'])
    def get_formations():
        """
        GET /api/smc/formations - Formation list

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            type: bos, choch, fvg, swing (optional, all if not specified)
            active_only: true/false (default: false)
            limit: 100 (default)
        """
        try:
            symbol = request.args.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = request.args.get('timeframe', '5m')
            formation_type = request.args.get('type')
            active_only = request.args.get('active_only', 'false').lower() == 'true'
            limit = request.args.get('limit', 100, type=int)

            service = get_smc_service()
            result = run_async(service.get_formations(
                symbol=symbol,
                timeframe=timeframe,
                formation_type=formation_type,
                active_only=active_only,
                limit=limit
            ))

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/smc/annotations', methods=['GET'])
    def get_annotations():
        """
        GET /api/smc/annotations - Chart annotations (LightweightCharts format)

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            start: bar index start (default: 0)
            end: bar index end (default: all)

        Returns:
            {
                "markers": [...],  # BOS, CHoCH, Swing markers
                "zones": [...],    # FVG, OB rectangles
                "lines": [...]     # Swing levels
            }
        """
        try:
            symbol = request.args.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = request.args.get('timeframe', '5m')
            start = request.args.get('start', 0, type=int)
            end = request.args.get('end', type=int)

            service = get_smc_service()
            result = run_async(service.get_annotations(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            ))

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/smc/levels', methods=['GET'])
    def get_levels():
        """
        GET /api/smc/levels - Current swing levels

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
        """
        try:
            symbol = request.args.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = request.args.get('timeframe', '5m')

            service = get_smc_service()
            result = run_async(service.get_levels(
                symbol=symbol,
                timeframe=timeframe
            ))

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/smc/bar/<int:bar_index>', methods=['GET'])
    def get_bar_analysis(bar_index):
        """
        GET /api/smc/bar/{index} - Tek bar analizi

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
        """
        try:
            symbol = request.args.get('symbol')
            if not symbol:
                return error_response('symbol required', 400)

            timeframe = request.args.get('timeframe', '5m')

            service = get_smc_service()
            result = run_async(service.get_bar_analysis(
                symbol=symbol,
                timeframe=timeframe,
                bar_index=bar_index
            ))

            if not result:
                return error_response('Bar not found', 404)

            return success_response(result)

        except Exception as e:
            return error_response(str(e), 500)
