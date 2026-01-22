"""Base service class"""

class BaseService:
    """Base service with common dependencies"""

    def __init__(self, data_manager=None, logger=None):
        self.data_manager = data_manager
        self.logger = logger
