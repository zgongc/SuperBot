"""Favorites business logic"""
from datetime import datetime
from .base_service import BaseService

class FavoritesService(BaseService):
    """Favorites management service"""

    async def get_favorites(self, sort_by='priority', tags=None):
        """Get user favorites"""
        favorites = await self.data_manager.get_favorites(
            sort_by=sort_by,
            tags=tags
        )
        return {
            'favorites': favorites,
            'total': len(favorites)
        }

    async def add_favorite(self, symbol, base_asset, quote_asset='USDT',
                          market_type='SPOT', tags=None, priority=5,
                          notes='', color=None):
        """Add symbol to favorites"""
        favorite_id = await self.data_manager.add_favorite(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            market_type=market_type,
            tags=tags or [],
            notes=notes,
            priority=priority,
            color=color
        )
        return favorite_id

    async def update_favorite(self, favorite_id, tags=None, notes=None,
                             priority=None, color=None):
        """Update favorite"""
        success = await self.data_manager.update_favorite(
            favorite_id=favorite_id,
            tags=tags,
            notes=notes,
            priority=priority,
            color=color
        )
        return success

    async def delete_favorite(self, favorite_id):
        """Delete favorite"""
        success = await self.data_manager.delete_favorite(favorite_id)
        return success

    async def bulk_add_favorites(self, symbols_list):
        """Add multiple symbols to favorites"""
        added_count = 0
        failed_count = 0
        errors = []

        for symbol_data in symbols_list:
            try:
                favorite_id = await self.add_favorite(
                    symbol=symbol_data['symbol'],
                    base_asset=symbol_data['base_asset'],
                    quote_asset=symbol_data.get('quote_asset', 'USDT'),
                    market_type=symbol_data.get('market_type', 'SPOT'),
                    tags=symbol_data.get('tags', []),
                    priority=symbol_data.get('priority', 5),
                    notes=symbol_data.get('notes', ''),
                    color=symbol_data.get('color')
                )

                if favorite_id:
                    added_count += 1
                else:
                    failed_count += 1
                    errors.append(f"{symbol_data['symbol']} already exists")

            except Exception as e:
                failed_count += 1
                errors.append(f"{symbol_data['symbol']}: {str(e)}")

        return {
            'added_count': added_count,
            'failed_count': failed_count,
            'total_requested': len(symbols_list),
            'errors': errors if errors else None
        }
