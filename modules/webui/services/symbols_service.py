"""Symbols business logic"""
from .base_service import BaseService

class SymbolsService(BaseService):
    """Symbols management service"""

    def __init__(self, data_manager, symbols_manager, logger):
        super().__init__(data_manager, logger)
        self.symbols_manager = symbols_manager

    async def sync_from_exchange(self, market_type='both', force=False):
        """Sync symbols from exchange"""
        self.logger.info(f"Syncing symbols - market: {market_type}, force: {force}")
        result = await self.symbols_manager.sync_from_exchange(
            market_type=market_type,
            force=force
        )
        return result

    async def get_available_symbols(self, market_type=None, quote_asset='USDT',
                                   search='', limit=1000):
        """Get available symbols from database"""
        symbols = await self.symbols_manager.get_available_symbols(
            market_type=market_type,
            quote_asset=quote_asset,
            search=search,
            limit=limit
        )
        return {
            'symbols': symbols,
            'total': len(symbols)
        }

    async def get_symbols_pool(self, page=1, per_page=50, search=''):
        """Get symbols pool with pagination"""
        symbols_data = await self.symbols_manager.get_symbols()
        base_assets = symbols_data.get('base_assets', [])
        quote_asset = symbols_data.get('quote_asset', 'USDT')

        # Filter by search
        if search:
            base_assets = [s for s in base_assets if search.upper() in s]

        # Pagination
        total = len(base_assets)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        paginated_assets = base_assets[start:end]

        # Build symbol list
        symbols_list = [
            {
                'symbol': f"{asset}{quote_asset}",
                'base_asset': asset,
                'quote_asset': quote_asset
            }
            for asset in paginated_assets
        ]

        return {
            'symbols': symbols_list,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }

    async def bulk_delete_symbols(self, symbol_ids):
        """Delete multiple symbols at once"""
        if not symbol_ids:
            raise ValueError("No symbol IDs provided")

        deleted_count = 0
        errors = []

        for symbol_id in symbol_ids:
            try:
                # Delete from database
                await self.data_manager.delete_symbol(symbol_id)
                deleted_count += 1
                self.logger.info(f"Deleted symbol ID: {symbol_id}")
            except Exception as e:
                error_msg = f"Failed to delete symbol {symbol_id}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        result = {
            'deleted_count': deleted_count,
            'total_requested': len(symbol_ids),
            'success': deleted_count == len(symbol_ids)
        }

        if errors:
            result['errors'] = errors

        return result
