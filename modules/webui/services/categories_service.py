"""Categories business logic"""
from .base_service import BaseService

class CategoriesService(BaseService):
    """Categories management service"""

    async def get_categories(self):
        """Get all categories with symbol counts"""
        categories = await self.data_manager.get_categories()
        return {
            'categories': categories,
            'total': len(categories)
        }

    async def get_category(self, category_id):
        """Get single category by ID"""
        category = await self.data_manager.get_category(category_id)
        return category

    async def create_category(self, name, description=None, default_priority=5, default_color='#666666'):
        """Create new category"""
        category_id = await self.data_manager.create_category(
            name=name,
            description=description,
            default_priority=default_priority,
            default_color=default_color
        )
        return category_id

    async def update_category(self, category_id, name=None, description=None, default_priority=None, default_color=None):
        """Update category"""
        success = await self.data_manager.update_category(
            category_id=category_id,
            name=name,
            description=description,
            default_priority=default_priority,
            default_color=default_color
        )
        return success

    async def delete_category(self, category_id):
        """Delete category"""
        success = await self.data_manager.delete_category(category_id)
        return success

    async def get_category_symbols(self, category_id):
        """Get all symbols in a category"""
        symbols = await self.data_manager.get_category_symbols(category_id)
        return {
            'symbols': symbols,
            'total': len(symbols)
        }

    async def add_symbols_to_category(self, category_id, symbol_ids, priority=5, color='#666666'):
        """Add multiple symbols to category with priority and color"""
        success = await self.data_manager.add_symbols_to_category(
            category_id=category_id,
            symbol_ids=symbol_ids,
            priority=priority,
            color=color
        )
        return success

    async def remove_symbols_from_category(self, category_id, symbol_ids):
        """Remove multiple symbols from category"""
        success = await self.data_manager.remove_symbols_from_category(
            category_id=category_id,
            symbol_ids=symbol_ids
        )
        return success
