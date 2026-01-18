#!/usr/bin/env python3
"""
components/strategies/strategy_manager.py
SuperBot - Strategy Manager

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Strategy template loading and lifecycle management.
    
    - Load strategy from template file
    - Validate strategy config
    - Initialize indicators
    - Create StrategyExecutor

Usage:
    from components.strategies.strategy_manager import StrategyManager
    
    manager = StrategyManager(indicator_manager)
    strategy, executor = manager.load_strategy('templates/SMC_Volume.py')
"""

from typing import Dict, Any, Optional, Tuple, List
import importlib.util
import sys
from pathlib import Path

from components.strategies.base_strategy import BaseStrategy
from components.strategies.strategy_executor import StrategyExecutor
from components.strategies.helpers import validate_strategy, ValidationError


class StrategyManager:
    """
    Strategy lifecycle manager
    
    Template loading, validation, initialization
    """
    
    def __init__(
        self,
        indicator_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        logger: Any = None
    ):
        """
        Initialize StrategyManager
        
        Args:
            indicator_manager: IndicatorManager instance (optional)
            position_manager: PositionManager instance (optional)
            logger: Logger instance (optional)
        """
        self.indicator_manager = indicator_manager
        self.position_manager = position_manager
        self.logger = logger
        
        # Loaded strategies
        self.loaded_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_executors: Dict[str, StrategyExecutor] = {}
    
    # ========================================================================
    # STRATEGY LOADING
    # ========================================================================
    
    # Default template directory
    DEFAULT_TEMPLATE_DIR = "components/strategies/templates"

    def load_strategy(
        self,
        template_path: str,
        validate: bool = True
    ) -> Tuple[BaseStrategy, StrategyExecutor]:
        """
        Load the strategy template.

        Args:
            template_path: Template file path
                - "grok_scalp.py" → components/strategies/templates/grok_scalp.py
                - "grok_scalp" → components/strategies/templates/grok_scalp.py
                - "components/strategies/templates/grok_scalp.py" → as-is
                - "/full/path/to/strategy.py" → as-is
            validate: Should validation be performed?

        Returns:
            (strategy, executor): BaseStrategy and StrategyExecutor instances.

        Raises:
            FileNotFoundError: Template not found
            ValidationError: Validation failed
            Exception: Import error
        """
        # 0. Normalize path
        template_path = self._normalize_strategy_path(template_path)

        if self.logger:
            self.logger.info(f"📋 Loading strategy: {template_path}")

        # 1. Load template module
        strategy = self._load_template_module(template_path)
        
        # 2. Validate strategy (if enabled)
        if validate:
            try:
                validate_strategy(strategy)
                if self.logger:
                    self.logger.info(f"✅ Strategy validation passed: {strategy.strategy_name}")
            except ValidationError as e:
                if self.logger:
                    self.logger.error(f"❌ Strategy validation failed: {e}")
                raise
        
        # 3. Initialize indicators (if indicator_manager exists)
        if self.indicator_manager:
            self._initialize_indicators(strategy)
        
        # 4. Create StrategyExecutor
        executor = StrategyExecutor(
            strategy=strategy,
            position_manager=self.position_manager,
            indicator_manager=self.indicator_manager,
            logger=self.logger
        )
        
        # 5. Store
        strategy_id = strategy.strategy_name
        self.loaded_strategies[strategy_id] = strategy
        self.strategy_executors[strategy_id] = executor
        
        if self.logger:
            self.logger.info(
                f"✅ Strategy loaded: {strategy.strategy_name} v{strategy.strategy_version}"
            )
        
        return strategy, executor

    def _normalize_strategy_path(self, template_path: str) -> str:
        """
        Strategy path'i normalize et

        Args:
            template_path: Raw path input

        Returns:
            str: Normalized full path

        Examples:
            "grok_scalp" → "components/strategies/templates/grok_scalp.py"
            "grok_scalp.py" → "components/strategies/templates/grok_scalp.py"
            "components/strategies/templates/grok_scalp.py" → as-is
            "/full/path/strategy.py" → as-is
        """
        path = Path(template_path)

        # Add .py extension if missing
        if not path.suffix:
            path = path.with_suffix('.py')

        # If already absolute or contains directory, check if exists
        if path.is_absolute() or len(path.parts) > 1:
            if path.exists():
                return str(path)
            # Maybe relative to cwd
            if Path(template_path).exists():
                return template_path

        # Try default template directory
        full_path = Path(self.DEFAULT_TEMPLATE_DIR) / path.name
        if full_path.exists():
            return str(full_path)

        # Fallback: return original (will fail in _load_template_module with clear error)
        return str(path)

    def _list_available_templates(self) -> List[str]:
        """
        List the currently available strategy templates.

        Returns:
            List[str]: Template file names (without .py)
        """
        templates = []
        template_dir = Path(self.DEFAULT_TEMPLATE_DIR)

        if template_dir.exists():
            for f in template_dir.glob("*.py"):
                # Skip __init__, base files, old folder
                if f.name.startswith("_") or f.name.startswith("base"):
                    continue
                templates.append(f.stem)

        return sorted(templates)

    def _load_template_module(self, template_path: str) -> BaseStrategy:
        """
        Import the template file and create a strategy instance.

        Args:
            template_path: Template file path

        Returns:
            BaseStrategy: Strategy instance

        Raises:
            FileNotFoundError: Template not found
            Exception: Import error
        """
        path = Path(template_path)

        if not path.exists():
            # List available templates
            available = self._list_available_templates()
            error_msg = f"❌ Strategy not found: '{template_path}'"

            if available:
                error_msg += f"\n\n📋 Available strategy templates:\n"
                for t in available:
                    error_msg += f"   • {t}\n"
            else:
                error_msg += f"\n\n⚠️ No template found in the '{self.DEFAULT_TEMPLATE_DIR}' folder!"

            raise FileNotFoundError(error_msg)
        
        # Dynamic import
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        
        if spec is None or spec.loader is None:
            raise Exception(f"Failed to load spec for {template_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Find BaseStrategy subclass in module
        strategy_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, BaseStrategy) and 
                obj is not BaseStrategy):
                strategy_class = obj
                break
        
        if strategy_class is None:
            raise Exception(
                f"No BaseStrategy subclass found in {template_path}. "
                f"Template must define a class that inherits from BaseStrategy."
            )
        
        # Instantiate strategy
        strategy = strategy_class()
        
        return strategy
    
    def _initialize_indicators(self, strategy: BaseStrategy) -> None:
        """
        Initialize the indicators for the strategy.
        
        Args:
            strategy: BaseStrategy instance
        """
        if not self.indicator_manager:
            return
        
        indicators = strategy.technical_parameters.indicators
        
        if not indicators:
            if self.logger:
                self.logger.warning(
                    f"❌ Strategy {strategy.strategy_name} has no indicators defined"
                )
            return
        
        # Get all symbols
        symbols = strategy.get_all_symbols()
        
        # Get timeframes
        timeframes = strategy.mtf_timeframes
        
        if self.logger:
            self.logger.info(
                f"Initializing {len(indicators)} indicators for "
                f"{len(symbols)} symbols, {len(timeframes)} timeframes"
            )
        
        # Initialize via IndicatorManager
        # Note: This method should be adjusted according to the IndicatorManager's interface.
        if hasattr(self.indicator_manager, 'load_from_config'):
            self.indicator_manager.load_from_config(
                indicators,
                symbols,
                timeframes
            )
        
        if self.logger:
            self.logger.info("✅ Indicators initialized successfully")
    
    # ========================================================================
    # STRATEGY ACCESS
    # ========================================================================
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Returns the loaded strategy.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            BaseStrategy or None
        """
        return self.loaded_strategies.get(strategy_name)
    
    def get_executor(self, strategy_name: str) -> Optional[StrategyExecutor]:
        """
        Return the strategy executor.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            StrategyExecutor or None
        """
        return self.strategy_executors.get(strategy_name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Returns all loaded strategies"""
        return dict(self.loaded_strategies)
    
    def has_strategy(self, strategy_name: str) -> bool:
        """Is the strategy loaded?"""
        return strategy_name in self.loaded_strategies
    
    # ========================================================================
    # STRATEGY MANAGEMENT
    # ========================================================================
    
    def unload_strategy(self, strategy_name: str) -> bool:
        """
        Unload the strategy.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            bool: True if unloaded
        """
        if strategy_name not in self.loaded_strategies:
            return False
        
        # Remove from tracking
        del self.loaded_strategies[strategy_name]
        del self.strategy_executors[strategy_name]
        
        if self.logger:
            self.logger.info(f"✅ Strategy unloaded: {strategy_name}")
        
        return True
    
    def reload_strategy(
        self,
        strategy_name: str,
        template_path: str
    ) -> Tuple[BaseStrategy, StrategyExecutor]:
        """
        Reload the strategy.
        
        Args:
            strategy_name: Strategy name
            template_path: Template path
        
        Returns:
            (strategy, executor): New instances
        """
        # Unload if exists
        self.unload_strategy(strategy_name)
        
        # Load again
        return self.load_strategy(template_path)
    
    def clear_all(self) -> None:
        """Clear all strategies"""
        self.loaded_strategies.clear()
        self.strategy_executors.clear()
        
        if self.logger:
            self.logger.info("All strategies cleared")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def list_templates(self, templates_dir: str = 'components/strategies/templates') -> list[str]:
        """
        Lists all strategy templates in the template directory.
        
        Args:
            templates_dir: Templates directory path
        
        Returns:
            List[str]: Template file paths
        """
        path = Path(templates_dir)
        
        if not path.exists():
            return []
        
        # Find all .py files (excluding __init__.py)
        templates = [
            str(f) for f in path.glob('*.py')
            if f.name != '__init__.py'
        ]
        
        return sorted(templates)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Manager summary
        
        Returns:
            Dict: Summary info
        """
        return {
            'loaded_strategies': len(self.loaded_strategies),
            'strategy_names': list(self.loaded_strategies.keys()),
            'has_indicator_manager': self.indicator_manager is not None,
            'has_position_manager': self.position_manager is not None,
        }
    
    def __repr__(self) -> str:
        return (
            f"<StrategyManager "
            f"strategies={len(self.loaded_strategies)}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StrategyManager',
]

