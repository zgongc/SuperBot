"""Categories API endpoints"""
from flask import request
from ..helpers.async_helper import run_async
from ..helpers.response_helper import success_response, error_response
from ..helpers.validation import validate_required_fields

def get_categories_service():
    """Get categories service from app context"""
    from flask import current_app
    return current_app.categories_service

def register_routes(bp):
    """Register categories routes"""

    @bp.route('/categories', methods=['GET'])
    def get_categories():
        """GET /api/categories - Get all categories"""
        try:
            service = get_categories_service()
            result = run_async(service.get_categories())

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>', methods=['GET'])
    def get_category(category_id):
        """GET /api/categories/<id> - Get single category"""
        try:
            service = get_categories_service()
            category = run_async(service.get_category(category_id))

            if category:
                return success_response(category)
            else:
                return error_response('Category not found', 404)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories', methods=['POST'])
    def create_category():
        """POST /api/categories - Create new category"""
        try:
            data = request.get_json()

            # Validate required fields
            is_valid, error_msg = validate_required_fields(data, ['name'])
            if not is_valid:
                return error_response(error_msg, 400)

            service = get_categories_service()
            category_id = run_async(service.create_category(
                name=data['name'],
                description=data.get('description'),
                default_priority=data.get('default_priority', 5),
                default_color=data.get('default_color', '#666666')
            ))

            if category_id:
                return success_response({
                    'category_id': category_id,
                    'name': data['name']
                }, message='Category created successfully')
            else:
                return error_response('Failed to create category', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>', methods=['PUT', 'PATCH'])
    def update_category(category_id):
        """PUT/PATCH /api/categories/<id> - Update category"""
        try:
            data = request.get_json()

            service = get_categories_service()
            success = run_async(service.update_category(
                category_id=category_id,
                name=data.get('name'),
                description=data.get('description'),
                default_priority=data.get('default_priority'),
                default_color=data.get('default_color')
            ))

            if success:
                return success_response(message='Category updated successfully')
            else:
                return error_response('Category not found', 404)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>', methods=['DELETE'])
    def delete_category(category_id):
        """DELETE /api/categories/<id> - Delete category"""
        try:
            service = get_categories_service()
            success = run_async(service.delete_category(category_id))

            if success:
                return success_response(message='Category deleted successfully')
            else:
                return error_response('Category not found', 404)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>/symbols', methods=['GET'])
    def get_category_symbols(category_id):
        """GET /api/categories/<id>/symbols - Get symbols in category"""
        try:
            service = get_categories_service()
            result = run_async(service.get_category_symbols(category_id))

            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>/symbols', methods=['POST'])
    def add_symbols_to_category(category_id):
        """POST /api/categories/<id>/symbols - Add symbols to category with priority and color"""
        try:
            data = request.get_json()
            symbol_ids = data.get('symbol_ids', [])
            priority = data.get('priority', 5)  # Default: 5
            color = data.get('color', '#666666')  # Default: grey

            if not symbol_ids:
                return error_response('No symbol_ids provided', 400)

            service = get_categories_service()
            success = run_async(service.add_symbols_to_category(
                category_id, symbol_ids, priority, color
            ))

            if success:
                return success_response(
                    message=f'Added {len(symbol_ids)} symbols to category'
                )
            else:
                return error_response('Failed to add symbols', 500)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/categories/<int:category_id>/symbols', methods=['DELETE'])
    def remove_symbols_from_category(category_id):
        """DELETE /api/categories/<id>/symbols - Remove symbols from category"""
        try:
            data = request.get_json()
            symbol_ids = data.get('symbol_ids', [])

            if not symbol_ids:
                return error_response('No symbol_ids provided', 400)

            service = get_categories_service()
            success = run_async(service.remove_symbols_from_category(
                category_id, symbol_ids
            ))

            if success:
                return success_response(
                    message=f'Removed {len(symbol_ids)} symbols from category'
                )
            else:
                return error_response('Failed to remove symbols', 500)
        except Exception as e:
            return error_response(str(e), 500)
