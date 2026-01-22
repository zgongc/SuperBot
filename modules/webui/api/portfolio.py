"""Portfolio API endpoints"""
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response
import logging

logger = logging.getLogger(__name__)

def get_portfolio_service():
    """Get portfolio service from app context"""
    from flask import current_app
    return current_app.portfolio_service

def register_routes(bp):
    """Register portfolio routes"""

    # ============================================
    # Portfolio Management
    # ============================================

    @bp.route('/portfolios', methods=['GET'])
    def get_portfolios():
        """GET /api/portfolios - Get all portfolios"""
        try:
            service = get_portfolio_service()
            portfolios = run_async(service.get_all_portfolios())
            return success_response({'portfolios': portfolios})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>', methods=['GET'])
    def get_portfolio_detail(portfolio_id):
        """GET /api/portfolios/:id - Get portfolio details"""
        try:
            service = get_portfolio_service()
            portfolio = run_async(service.get_portfolio_by_id(portfolio_id))

            if not portfolio:
                return error_response('Portfolio not found', 404)

            return success_response({'portfolio': portfolio})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios', methods=['POST'])
    def create_portfolio():
        """POST /api/portfolios - Create new portfolio"""
        try:
            data = request.get_json()

            # Validate required fields
            if not data.get('name'):
                return error_response('Name is required', 400)

            service = get_portfolio_service()
            portfolio_id = run_async(service.create_portfolio(
                name=data['name'],
                exchange_account_id=data.get('exchange_account_id'),  # NULL for manual portfolios
                notes=data.get('notes')
            ))

            if portfolio_id:
                return success_response({
                    'portfolio_id': portfolio_id,
                    'message': 'Portfolio created successfully'
                })
            else:
                return error_response('Failed to create portfolio', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>', methods=['PUT'])
    def update_portfolio(portfolio_id):
        """PUT /api/portfolios/:id - Update portfolio"""
        try:
            data = request.get_json()
            service = get_portfolio_service()

            success = run_async(service.update_portfolio(portfolio_id, **data))

            if success:
                return success_response({'message': 'Portfolio updated successfully'})
            else:
                return error_response('Failed to update portfolio', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>', methods=['DELETE'])
    def delete_portfolio(portfolio_id):
        """DELETE /api/portfolios/:id - Delete portfolio"""
        try:
            service = get_portfolio_service()
            success = run_async(service.delete_portfolio(portfolio_id))

            if success:
                return success_response({'message': 'Portfolio deleted successfully'})
            else:
                return error_response('Failed to delete portfolio', 500)

        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Portfolio Summary & Positions
    # ============================================

    @bp.route('/portfolios/<int:portfolio_id>/summary', methods=['GET'])
    def get_portfolio_summary(portfolio_id):
        """GET /api/portfolios/:id/summary - Get portfolio summary with positions and P&L"""
        try:
            is_open = request.args.get('is_open')
            if is_open is not None:
                is_open = is_open.lower() == 'true'

            service = get_portfolio_service()
            result = run_async(service.get_portfolio_summary(portfolio_id, is_open=is_open))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions', methods=['GET'])
    def get_portfolio_positions(portfolio_id):
        """GET /api/portfolios/:id/positions - Get portfolio positions"""
        try:
            is_open = request.args.get('is_open')
            if is_open is not None:
                is_open = is_open.lower() == 'true'

            service = get_portfolio_service()
            positions = run_async(service.get_positions(portfolio_id, is_open=is_open))

            return success_response({'positions': positions})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions', methods=['POST'])
    def add_manual_position(portfolio_id):
        """POST /api/portfolios/:id/positions - Add manual position"""
        try:
            data = request.get_json()

            # Validate required fields (symbol string VEYA symbol_id)
            required_fields = ['quantity', 'entry_price']
            for field in required_fields:
                if field not in data:
                    return error_response(f'{field} is required', 400)

            if not data.get('symbol') and not data.get('symbol_id'):
                return error_response('symbol or symbol_id is required', 400)

            service = get_portfolio_service()

            # symbol_id veya symbol parametresini gönder
            position_id = run_async(service.add_manual_position(
                portfolio_id=portfolio_id,
                symbol=data.get('symbol'),  # String (örn: "BTC/USDT")
                symbol_id=data.get('symbol_id'),  # Integer (opsiyonel)
                quantity=float(data['quantity']),
                entry_price=float(data['entry_price']),
                side=data.get('side', 'LONG'),
                position_type=data.get('position_type', 'SPOT'),
                opened_at=data.get('opened_at'),
                source=data.get('source', 'manual'),
                notes=data.get('notes')
            ))

            if position_id:
                return success_response({
                    'position_id': position_id,
                    'message': 'Position added successfully'
                })
            else:
                return error_response('Failed to add position', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>', methods=['PUT'])
    def update_position(portfolio_id, position_id):
        """PUT /api/portfolios/:id/positions/:pos_id - Update position"""
        try:
            data = request.get_json()

            from flask import current_app
            success = run_async(current_app.data_manager.update_position(
                position_id=position_id,
                quantity=data.get('quantity'),
                entry_price=data.get('entry_price'),
                opened_at=data.get('opened_at'),
                notes=data.get('notes'),
                is_open=data.get('is_open'),
                exit_price=data.get('exit_price')
            ))

            if success:
                return success_response({'message': 'Position updated successfully'})
            else:
                return error_response('Failed to update position', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>', methods=['DELETE'])
    def delete_position(portfolio_id, position_id):
        """DELETE /api/portfolios/:id/positions/:pos_id - Delete position"""
        try:
            service = get_portfolio_service()
            success = run_async(service.delete_position(position_id))

            if success:
                return success_response({'message': 'Position deleted successfully'})
            else:
                return error_response('Failed to delete position', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/close', methods=['POST'])
    def close_position(portfolio_id, position_id):
        """POST /api/portfolios/:id/positions/:pos_id/close - Close position"""
        try:
            data = request.get_json() or {}
            close_price = data.get('close_price')

            service = get_portfolio_service()
            success = run_async(service.close_position(position_id, close_price))

            if success:
                return success_response({'message': 'Position closed successfully'})
            else:
                return error_response('Failed to close position', 500)

        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Sync from Exchange
    # ============================================

    @bp.route('/portfolios/<int:portfolio_id>/sync', methods=['POST'])
    def sync_portfolio(portfolio_id):
        """POST /api/portfolios/:id/sync - Sync positions from exchange"""
        try:
            service = get_portfolio_service()
            result = run_async(service.sync_positions_from_exchange(portfolio_id))

            if result:
                return success_response({
                    'synced_count': result.get('synced_count', 0),
                    'message': f"{result.get('synced_count', 0)} pozisyon senkronize edildi"
                })
            else:
                return error_response('Senkronizasyon başarısız', 500)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/update-prices', methods=['POST'])
    def update_portfolio_prices(portfolio_id):
        """POST /api/portfolios/:id/update-prices - Update position prices from exchanges"""
        try:
            service = get_portfolio_service()
            result = run_async(service.update_positions_prices(portfolio_id))

            if result:
                return success_response({
                    'updated_count': result.get('updated_count', 0),
                    'total_positions': result.get('total_positions', 0),
                    'errors': result.get('errors', []),
                    'prices': result.get('prices', {}),
                    'message': result.get('message', 'Prices updated')
                })
            else:
                return error_response('Fiyat güncellemesi başarısız', 500)

        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Import from Exchange
    # ============================================

    @bp.route('/portfolios/<int:portfolio_id>/import-preview', methods=['GET'])
    def get_import_preview(portfolio_id):
        """GET /api/portfolios/:id/import-preview - Get import preview data"""
        try:
            service = get_portfolio_service()
            preview_data = run_async(service.prepare_import_preview(portfolio_id))
            return success_response(preview_data)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/import-from-exchange', methods=['POST'])
    def import_positions_from_exchange(portfolio_id):
        """POST /api/portfolios/:id/import-from-exchange - Import selected positions from exchange"""
        try:
            data = request.get_json()
            selected_assets = data.get('selected_assets', [])

            if not selected_assets:
                return error_response('No assets selected', 400)

            service = get_portfolio_service()
            result = run_async(service.import_selected_positions(
                portfolio_id=portfolio_id,
                selected_assets=selected_assets
            ))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Portfolio Analytics (3commas-style)
    # ============================================

    @bp.route('/portfolios/<int:portfolio_id>/statistics', methods=['GET'])
    def get_portfolio_statistics(portfolio_id):
        """GET /api/portfolios/:id/statistics - Get comprehensive portfolio statistics"""
        try:
            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            stats = run_async(analytics.get_portfolio_statistics(portfolio_id))

            return success_response(stats)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/allocation', methods=['GET'])
    def get_token_allocation(portfolio_id):
        """GET /api/portfolios/:id/allocation - Get token allocation for pie chart"""
        try:
            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            stats = run_async(analytics.get_portfolio_statistics(portfolio_id))

            return success_response({
                'allocation': stats.get('allocation', [])
            })
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/performance', methods=['GET'])
    def get_portfolio_performance(portfolio_id):
        """GET /api/portfolios/:id/performance - Get performance metrics"""
        try:
            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            stats = run_async(analytics.get_portfolio_statistics(portfolio_id))

            return success_response({
                'performance': stats.get('performance', {}),
                'risk_metrics': stats.get('risk_metrics', {})
            })
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions-detailed', methods=['GET'])
    def get_positions_detailed(portfolio_id):
        """GET /api/portfolios/:id/positions-detailed - Get positions with entries and exit targets"""
        try:
            is_open = request.args.get('is_open')
            if is_open is not None:
                is_open = is_open.lower() == 'true'

            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            positions = run_async(analytics.get_portfolio_positions_with_details(
                portfolio_id=portfolio_id,
                is_open=is_open
            ))

            return success_response({'positions': positions})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/details', methods=['GET'])
    def get_position_details(portfolio_id, position_id):
        """GET /api/portfolios/:id/positions/:pos_id/details - Get position details with entries/exits"""
        try:
            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            details = run_async(analytics.get_position_details(position_id))

            if not details:
                return error_response('Position not found', 404)

            return success_response(details)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions-multi-entry', methods=['POST'])
    def add_multi_entry_position(portfolio_id):
        """POST /api/portfolios/:id/positions-multi-entry - Create position with multiple entries"""
        try:
            data = request.get_json()

            # Validate required fields
            if not data.get('symbol'):
                return error_response('Symbol is required', 400)
            if not data.get('side'):
                return error_response('Side is required', 400)
            if not data.get('entries') or not isinstance(data['entries'], list):
                return error_response('Entries array is required', 400)

            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            position_id = run_async(analytics.add_position_with_entries(
                portfolio_id=portfolio_id,
                symbol=data['symbol'],
                side=data['side'],
                entries=data['entries'],
                exit_targets=data.get('exit_targets')
            ))

            if position_id:
                return success_response({
                    'position_id': position_id,
                    'message': 'Multi-entry position created successfully'
                })
            else:
                return error_response('Failed to create position', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/entries', methods=['POST'])
    def add_position_entry(portfolio_id, position_id):
        """POST /api/portfolios/:id/positions/:pos_id/entries - Add entry to existing position"""
        try:
            data = request.get_json()

            if not data.get('quantity') or not data.get('price'):
                return error_response('Quantity and price are required', 400)

            from flask import current_app
            entry_id = run_async(current_app.data_manager.add_position_entry(
                position_id=position_id,
                entry_number=data.get('entry_number', 1),
                quantity=float(data['quantity']),
                entry_price=float(data['price']),
                status=data.get('status', 'filled'),
                source=data.get('source', 'manual'),
                traded_at=data.get('traded_at')
            ))

            if entry_id:
                return success_response({
                    'entry_id': entry_id,
                    'message': 'Entry added successfully'
                })
            else:
                return error_response('Failed to add entry', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/entries', methods=['GET'])
    def get_position_entries(portfolio_id, position_id):
        """GET /api/portfolios/:id/positions/:pos_id/entries - Get all entries for a position"""
        try:
            from flask import current_app
            entries = run_async(current_app.data_manager.get_position_entries(position_id))

            return success_response({
                'entries': entries,
                'count': len(entries)
            })
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/entries/<int:entry_id>', methods=['PUT'])
    def update_position_entry(portfolio_id, position_id, entry_id):
        """PUT /api/portfolios/:id/positions/:pos_id/entries/:entry_id - Update entry"""
        try:
            data = request.get_json()

            from flask import current_app
            success = run_async(current_app.data_manager.update_position_entry(
                entry_id=entry_id,
                quantity=data.get('quantity'),
                entry_price=data.get('entry_price'),
                traded_at=data.get('traded_at'),
                notes=data.get('notes')
            ))

            if success:
                return success_response({
                    'message': 'Entry updated successfully'
                })
            else:
                return error_response('Failed to update entry', 500)
        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Import/Export
    # ============================================

    @bp.route('/portfolios/<int:portfolio_id>/export', methods=['GET'])
    def export_positions(portfolio_id):
        """GET /api/portfolios/:id/export - Export positions to JSON"""
        try:
            from flask import current_app
            from datetime import datetime, timezone

            service = get_portfolio_service()

            # Get portfolio info
            portfolio = run_async(service.get_portfolio_by_id(portfolio_id))
            if not portfolio:
                logger.warning(f"Export failed: Portfolio {portfolio_id} not found")
                return error_response('Portfolio not found', 404)

            # Get filter parameter
            filter_type = request.args.get('filter', 'all')  # all, open, closed
            is_open = None if filter_type == 'all' else (filter_type == 'open')
            logger.info(f"Exporting positions from portfolio '{portfolio['name']}' (filter={filter_type})")

            # Get positions
            positions = run_async(current_app.data_manager.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=is_open
            ))

            # Get entries and exit targets for each position
            for pos in positions:
                # Convert datetime fields to ISO format strings
                if pos.get('opened_at'):
                    from datetime import datetime as dt
                    if isinstance(pos['opened_at'], dt):
                        pos['opened_at'] = pos['opened_at'].isoformat()
                if pos.get('closed_at'):
                    from datetime import datetime as dt
                    if isinstance(pos['closed_at'], dt):
                        pos['closed_at'] = pos['closed_at'].isoformat()

                # Get entries
                entries = run_async(current_app.data_manager.get_position_entries(pos['id']))
                # Convert entry dates
                for entry in entries:
                    if entry.get('traded_at'):
                        from datetime import datetime as dt
                        if isinstance(entry['traded_at'], dt):
                            entry['traded_at'] = entry['traded_at'].isoformat()
                pos['entries'] = entries

                # Get exit targets
                exit_targets = run_async(current_app.data_manager.get_position_exit_targets(pos['id']))
                # Convert target dates
                for target in exit_targets:
                    if target.get('triggered_at'):
                        from datetime import datetime as dt
                        if isinstance(target['triggered_at'], dt):
                            target['triggered_at'] = target['triggered_at'].isoformat()
                pos['exit_targets'] = exit_targets

            # Calculate summary
            summary = run_async(service.get_portfolio_summary(portfolio_id))['summary']

            # Build export data
            export_data = {
                'version': '1.0',
                'export_date': datetime.now(timezone.utc).isoformat(),
                'portfolio': {
                    'id': portfolio['id'],
                    'name': portfolio['name'],
                    'exchange': portfolio.get('exchange'),
                    'account_type': portfolio.get('account_type')
                },
                'positions': positions,
                'summary': {
                    'total_positions': len(positions),
                    'open_positions': len([p for p in positions if p.get('is_open', True)]),
                    'closed_positions': len([p for p in positions if not p.get('is_open', True)]),
                    'total_realized_pnl': summary.get('total_realized_pnl', 0),
                    'total_unrealized_pnl': summary.get('total_unrealized_pnl', 0)
                }
            }

            logger.info(f"Export successful: {export_data['summary']['total_positions']} positions exported")
            return success_response(export_data)
        except Exception as e:
            logger.error(f"Export error for portfolio {portfolio_id}: {e}", exc_info=True)
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/check-duplicates', methods=['POST'])
    def check_duplicate_positions(portfolio_id):
        """POST /api/portfolios/:id/check-duplicates - Check which symbols already exist"""
        try:
            from flask import current_app

            data = request.get_json()
            symbols = data.get('symbols', [])

            if not symbols:
                return success_response({'duplicates': {}})

            # Get existing positions for this portfolio
            existing_positions = run_async(current_app.data_manager.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=None  # Check both open and closed
            ))

            # Build map of existing symbols with their details
            duplicates = {}
            for pos in existing_positions:
                if pos['symbol'] in symbols:
                    duplicates[pos['symbol']] = {
                        'id': pos['id'],
                        'quantity': pos['quantity'],
                        'entry_price': pos['entry_price'],
                        'is_open': pos.get('is_open', True),
                        'opened_at': pos.get('opened_at')
                    }

            logger.info(f"Duplicate check for portfolio {portfolio_id}: {len(duplicates)} duplicates found out of {len(symbols)} symbols")
            return success_response({'duplicates': duplicates})

        except Exception as e:
            logger.error(f"Duplicate check error for portfolio {portfolio_id}: {e}", exc_info=True)
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/import', methods=['POST'])
    def import_positions(portfolio_id):
        """POST /api/portfolios/:id/import - Import positions from JSON"""
        try:
            from flask import current_app

            data = request.get_json()

            # Validate JSON structure
            if not data.get('version'):
                return error_response('Invalid import data: missing version', 400)

            if not data.get('positions'):
                return error_response('Invalid import data: missing positions', 400)

            service = get_portfolio_service()

            # Check if portfolio exists
            portfolio = run_async(service.get_portfolio_by_id(portfolio_id))
            if not portfolio:
                return error_response('Portfolio not found', 404)

            imported_count = 0
            skipped_count = 0
            merged_count = 0
            replaced_count = 0
            errors = []

            for pos_data in data['positions']:
                try:
                    action = pos_data.get('action', 'create')  # create, skip, merge, replace
                    symbol = pos_data['symbol']

                    if action == 'skip':
                        skipped_count += 1
                        logger.info(f"Skipping {symbol} as requested")
                        continue

                    elif action == 'replace':
                        # Delete existing position first
                        duplicate_id = pos_data.get('duplicate_id')
                        if duplicate_id:
                            run_async(current_app.data_manager.delete_position(duplicate_id))
                            logger.info(f"Deleted existing position {duplicate_id} for {symbol}")
                        replaced_count += 1

                    elif action == 'merge':
                        # Add entries to existing position
                        duplicate_id = pos_data.get('duplicate_id')
                        if duplicate_id and pos_data.get('entries'):
                            for entry in pos_data['entries']:
                                run_async(current_app.data_manager.add_position_entry(
                                    position_id=duplicate_id,
                                    entry_number=entry.get('entry_number', 1),
                                    quantity=entry['quantity'],
                                    price=entry['price'],
                                    traded_at=entry.get('traded_at'),
                                    notes=entry.get('notes', 'Merged from import')
                                ))
                            logger.info(f"Merged {len(pos_data['entries'])} entries to existing position {duplicate_id}")
                            merged_count += 1
                            continue  # Don't create new position

                    # Create new position (for create, replace actions)
                    position_id = run_async(current_app.data_manager.create_position(
                        portfolio_id=portfolio_id,
                        symbol=symbol,
                        side=pos_data.get('side', 'LONG'),
                        quantity=pos_data['quantity'],
                        entry_price=pos_data['entry_price'],
                        opened_at=pos_data.get('opened_at'),
                        source=pos_data.get('source', 'import'),
                        notes=pos_data.get('notes')
                    ))

                    # Update current_price if available in export data
                    if pos_data.get('current_price'):
                        run_async(current_app.data_manager.update_position_price(
                            position_id=position_id,
                            current_price=pos_data['current_price']
                        ))
                        logger.info(f"Set current price for {symbol}: ${pos_data['current_price']}")

                    # If position is closed, update it
                    if not pos_data.get('is_open', True):
                        run_async(current_app.data_manager.close_position(
                            position_id=position_id,
                            exit_price=pos_data.get('exit_price'),
                            closed_at=pos_data.get('closed_at')
                        ))

                    # Import entries if present
                    if pos_data.get('entries'):
                        for entry in pos_data['entries']:
                            run_async(current_app.data_manager.add_position_entry(
                                position_id=position_id,
                                entry_number=entry.get('entry_number', 1),
                                quantity=entry['quantity'],
                                price=entry['price'],
                                traded_at=entry.get('traded_at'),
                                notes=entry.get('notes')
                            ))

                    # Import exit targets if present
                    if pos_data.get('exit_targets'):
                        for target in pos_data['exit_targets']:
                            run_async(current_app.data_manager.add_exit_target(
                                position_id=position_id,
                                target_type=target['target_type'],
                                target_number=target.get('target_number', 1),
                                target_price=target['target_price'],
                                quantity_percentage=target['quantity_percentage'],
                                status=target.get('status', 'pending')
                            ))

                    imported_count += 1
                except Exception as e:
                    errors.append(f"{pos_data.get('symbol', 'Unknown')}: {str(e)}")
                    skipped_count += 1
                    logger.error(f"Error importing {pos_data.get('symbol')}: {e}")

            logger.info(f"Import completed: {imported_count} imported, {merged_count} merged, {replaced_count} replaced, {skipped_count} skipped")

            return success_response({
                'message': 'Import completed',
                'imported': imported_count,
                'merged': merged_count,
                'replaced': replaced_count,
                'skipped': skipped_count,
                'errors': errors if errors else None
            })
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/entries/<int:entry_id>', methods=['DELETE'])
    def delete_position_entry(portfolio_id, position_id, entry_id):
        """DELETE /api/portfolios/:id/positions/:pos_id/entries/:entry_id - Delete entry"""
        try:
            from flask import current_app
            success = run_async(current_app.data_manager.delete_position_entry(entry_id))

            if success:
                return success_response({
                    'message': 'Entry deleted successfully'
                })
            else:
                return error_response('Entry not found or failed to delete', 404)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/exit-targets', methods=['POST'])
    def add_exit_target(portfolio_id, position_id):
        """POST /api/portfolios/:id/positions/:pos_id/exit-targets - Add TP/SL target"""
        try:
            data = request.get_json()

            required = ['target_type', 'target_number', 'target_price', 'quantity_percentage']
            for field in required:
                if field not in data:
                    return error_response(f'{field} is required', 400)

            from flask import current_app
            target_id = run_async(current_app.data_manager.add_exit_target(
                position_id=position_id,
                target_type=data['target_type'],
                target_number=int(data['target_number']),
                target_price=float(data['target_price']),
                quantity_percentage=float(data['quantity_percentage'])
            ))

            if target_id:
                return success_response({
                    'target_id': target_id,
                    'message': 'Exit target added successfully'
                })
            else:
                return error_response('Failed to add exit target', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/positions/<int:position_id>/exit-targets/<int:target_id>/trigger', methods=['POST'])
    def trigger_exit_target(portfolio_id, position_id, target_id):
        """POST /api/portfolios/:id/positions/:pos_id/exit-targets/:target_id/trigger - Trigger TP/SL"""
        try:
            data = request.get_json()

            required = ['triggered_price', 'quantity_closed', 'pnl']
            for field in required:
                if field not in data:
                    return error_response(f'{field} is required', 400)

            from flask import current_app
            success = run_async(current_app.data_manager.trigger_exit_target(
                target_id=target_id,
                triggered_price=float(data['triggered_price']),
                quantity_closed=float(data['quantity_closed']),
                pnl=float(data['pnl'])
            ))

            if success:
                return success_response({'message': 'Exit target triggered successfully'})
            else:
                return error_response('Failed to trigger exit target', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/portfolios/<int:portfolio_id>/metrics/update', methods=['POST'])
    def update_portfolio_metrics(portfolio_id):
        """POST /api/portfolios/:id/metrics/update - Recalculate all portfolio metrics"""
        try:
            from ..services.portfolio_analytics_service import PortfolioAnalyticsService
            from flask import current_app

            analytics = PortfolioAnalyticsService(current_app.data_manager)
            success = run_async(analytics.update_portfolio_metrics(portfolio_id))

            if success:
                return success_response({'message': 'Portfolio metrics updated successfully'})
            else:
                return error_response('Failed to update metrics', 500)
        except Exception as e:
            return error_response(str(e), 500)

    # ============================================
    # Legacy endpoint (for backward compatibility)
    # ============================================

    @bp.route('/portfolio', methods=['GET'])
    def get_portfolio_legacy():
        """GET /api/portfolio - Get portfolio summary (old endpoint)"""
        try:
            service = get_portfolio_service()
            result = run_async(service.get_portfolio_summary_legacy())
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)
