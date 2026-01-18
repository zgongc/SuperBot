#!/usr/bin/env python3
"""
Strategy Validator - Validates strategy parameters.

It checks whether all strategy parameters (RiskManagement, ExitStrategy, entry/exit conditions, etc.)
are in the correct format and have sensible values.

This file is in the strategies/ folder because it is related to STRATEGY PARAMETERS.

Version: 2.0.0 (merged with all checks in helpers/validation.py)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import enums for validation
try:
    from components.strategies.base_strategy import (
        PositionSizeMethod,
        ExitMethod,
        StopLossMethod,
    )
except ImportError:
    # Fallback if imports fail
    PositionSizeMethod = None
    ExitMethod = None
    StopLossMethod = None


@dataclass
class ValidationError:
    """
    Validation error
    
    Attributes:
        field: Invalid field name
        value: Invalid value
        message: Error message (Turkish)
        severity: Error severity ('error', 'warning')
    """
    field: str
    value: Any
    message: str
    severity: str = 'error'  # 'error' or 'warning'


@dataclass
class StrategyValidationResult:
    """
    Strategy validation result.
    
    Attributes:
        valid: Is the strategy valid?
        errors: List of errors
        warnings: List of warnings
    """
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def __str__(self) -> str:
        """Human-readable format"""
        lines = []
        if self.valid:
            lines.append("Strategy is valid")
        else:
            lines.append("Strategy is invalid")
        
        if self.errors:
            lines.append(f"\nHatalar ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err.field}: {err.message}")
        
        if self.warnings:
            lines.append(f"\nUyarilar ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn.field}: {warn.message}")
        
        return "\n".join(lines)


class StrategyValidator:
    """
    Validates comprehensive strategy parameters.
    
    Checked Parameters:
    - Metadata: strategy_name, version, author
    - Symbols: Symbol configurations and format
    - Timeframes: MTF timeframes and primary timeframe
    - Indicators: Technical indicators and their parameters
    - RiskManagement: Risk management settings (ENUM + values)
    - ExitStrategy: Exit strategy settings (ENUM + values)
    - PositionManagement: Position management settings
    - entry_conditions: Entry conditions (syntax, operators, timeframes)
    - exit_conditions: Exit conditions (syntax, operators, timeframes)
    - custom_parameters: Custom parameters
    - optimizer_parameters: Optimization parameters
    - account_management: Account management
    - backtest_parameters: Backtest parameters
    
    Validation Types:
    - ✅ Required fields
    - ✅ Value ranges
    - ✅ Logical consistency (TP > SL vs.)
    - ✅ Format check (dict, list, vs.)
    - ✅ Enum validations
    - ✅ Operator validations
    - ✅ Timeframe validations
    - ✅ Symbol format validations
    - ✅ Condition syntax validations
    """
    
    # Required fields (minimum)
    REQUIRED_FIELDS = [
        'strategy_name',
        'strategy_version',
        'symbols',
        'technical_parameters',
        'entry_conditions',
        'exit_strategy',
        'risk_management',
        'position_management',
    ]
    
    # Valid timeframes
    VALID_TIMEFRAMES = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '12h',
        '1d', '3d', '1w', '1M'
    }
    
    # Valid operators
    VALID_OPERATORS = {
        '>', '<', '>=', '<=', '==', '!=',
        'crossover', 'crossunder', 'cross_over', 'cross_under',
        'crossabove', 'crossbelow',
        'rising', 'falling',
        'between', 'outside', 'near',
        'is', 'is_not',
        'above', 'below',
        'gte', 'lte', 'greater_equal', 'less_equal',
        'equals', 'not_equals',
    }
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize the validator.
        
        Args:
            logger: Optional logger
        """
        self.logger = logger
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def validate_strategy(self, strategy: Any) -> StrategyValidationResult:
        """
        Stratejiyi tamamen validate et (KAPSAMLI)

        Args:
            strategy: Strategy object (derived from BaseStrategy)

        Returns:
            StrategyValidationResult
        """
        self.errors = []
        self.warnings = []

        # Store current strategy for indicator checks
        self._current_strategy = strategy
        
        # 0. Required fields
        self._validate_required_fields(strategy)
        
        # 1. Metadata
        self._validate_metadata(strategy)
        
        # 2. Symbols
        if hasattr(strategy, 'symbols'):
            self._validate_symbols(strategy)
        
        # 3. Timeframes
        if hasattr(strategy, 'mtf_timeframes'):
            self._validate_timeframes(strategy)
        
        # 4. Indicators
        if hasattr(strategy, 'technical_parameters'):
            self._validate_indicators(strategy)
        
        # 5. RiskManagement validasyonu
        if hasattr(strategy, 'risk_management'):
            self._validate_risk_management(strategy.risk_management)
        else:
            self.errors.append(ValidationError(
                field='risk_management',
                value=None,
                message='RiskManagement parametresi eksik!'
            ))
        
        # 6. ExitStrategy validasyonu
        if hasattr(strategy, 'exit_strategy'):
            self._validate_exit_strategy(strategy.exit_strategy)
        else:
            self.errors.append(ValidationError(
                field='exit_strategy',
                value=None,
                message='ExitStrategy parametresi eksik!'
            ))
        
        # 7. PositionManagement validasyonu
        if hasattr(strategy, 'position_management'):
            self._validate_position_management(strategy.position_management)
        else:
            self.errors.append(ValidationError(
                field='position_management',
                value=None,
                message='PositionManagement parametresi eksik!'
            ))
        
        # 8. Entry/Exit Conditions validasyonu
        if hasattr(strategy, 'entry_conditions'):
            self._validate_conditions(strategy, 'entry_conditions')
        
        if hasattr(strategy, 'exit_conditions'):
            self._validate_conditions(strategy, 'exit_conditions')
        
        # 9. Custom Parameters validasyonu
        if hasattr(strategy, 'custom_parameters'):
            self._validate_custom_parameters(strategy.custom_parameters)
        
        # 10. Optimizer Parameters validasyonu
        if hasattr(strategy, 'optimizer_parameters'):
            self._validate_optimizer_parameters(strategy.optimizer_parameters)
        
        # 11. Account Management (optional)
        if hasattr(strategy, 'account_management') and strategy.account_management:
            self._validate_account_management(strategy.account_management)
        
        # 12. Backtest Parameters (optional)
        if hasattr(strategy, 'backtest_parameters') and strategy.backtest_parameters:
            self._validate_backtest_parameters(strategy)
        
        # Create the result
        result = StrategyValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings
        )
        
        if self.logger:
            if result.valid:
                self.logger.info(f"Strategy validation successful")
                if result.warnings:
                    self.logger.warning(f"{len(result.warnings)} warnings exist")
            else:
                self.logger.error(f"Strategy validation failed: {len(result.errors)} errors")
        
        return result
    
    def _validate_required_fields(self, strategy: Any):
        """Check for the existence of required fields"""
        missing = []
        
        for field in self.REQUIRED_FIELDS:
            if not hasattr(strategy, field):
                missing.append(field)
            elif getattr(strategy, field) is None:
                missing.append(field)
        
        if missing:
            self.errors.append(ValidationError(
                field='required_fields',
                value=missing,
                message=f"Eksik required field'lar: {missing}"
            ))
    
    def _validate_metadata(self, strategy: Any):
        """Validate metadata"""
        if hasattr(strategy, 'strategy_name'):
            if not strategy.strategy_name:
                self.errors.append(ValidationError(
                    field='strategy_name',
                    value=strategy.strategy_name,
                    message='strategy_name cannot be empty'
                ))
        
        if hasattr(strategy, 'strategy_version'):
            if not strategy.strategy_version:
                self.errors.append(ValidationError(
                    field='strategy_version',
                    value=strategy.strategy_version,
                    message='strategy_version cannot be empty'
                ))
        
        # Optional: author validation
        if hasattr(strategy, 'author') and strategy.author:
            if len(strategy.author) > 100:
                self.warnings.append(ValidationError(
                    field='author',
                    value=strategy.author,
                    message='author is too long (max 100 characters)',
                    severity='warning'
                ))
    
    def _validate_symbols(self, strategy: Any):
        """Validate symbol configurations"""
        if not strategy.symbols:
            self.errors.append(ValidationError(
                field='symbols',
                value=None,
                message='At least 1 symbol is required'
            ))
            return
        
        for i, symbol_config in enumerate(strategy.symbols):
            if not hasattr(symbol_config, 'symbol') or not symbol_config.symbol:
                self.errors.append(ValidationError(
                    field=f'symbols[{i}].symbol',
                    value=None,
                    message='Symbol list is empty'
                ))
            
            if not hasattr(symbol_config, 'quote') or not symbol_config.quote:
                self.errors.append(ValidationError(
                    field=f'symbols[{i}].quote',
                    value=None,
                    message='Quote bos'
                ))
            
            # Check symbol format
            if hasattr(symbol_config, 'symbol'):
                for sym in symbol_config.symbol:
                    if not sym or not isinstance(sym, str):
                        self.errors.append(ValidationError(
                            field=f'symbols[{i}].symbol',
                            value=sym,
                            message=f'Invalid symbol: {sym}'
                        ))
    
    def _validate_timeframes(self, strategy: Any):
        """Validate timeframes"""
        # MTF timeframes
        # MTF is optional - allow empty list
        if not strategy.mtf_timeframes:
            return  # Skip MTF validation if not used
        
        for tf in strategy.mtf_timeframes:
            if tf not in self.VALID_TIMEFRAMES:
                self.errors.append(ValidationError(
                    field='mtf_timeframes',
                    value=tf,
                    message=f"Invalid timeframe: '{tf}'. Valid: {sorted(self.VALID_TIMEFRAMES)}"
                ))
        
        # The primary timeframe must be within the mtf_timeframes.
        if hasattr(strategy, 'primary_timeframe'):
            if strategy.primary_timeframe not in strategy.mtf_timeframes:
                self.errors.append(ValidationError(
                    field='primary_timeframe',
                    value=strategy.primary_timeframe,
                    message=f"primary_timeframe ('{strategy.primary_timeframe}') must be within mtf_timeframes"
                ))
    
    def _validate_indicators(self, strategy: Any):
        """Validate indicator configurations"""
        indicators = strategy.technical_parameters.indicators

        if not indicators:
            self.errors.append(ValidationError(
                field='technical_parameters.indicators',
                value=None,
                message='At least 1 indicator is required'
            ))
            return

        for name, params in indicators.items():
            if not isinstance(params, dict):
                self.errors.append(ValidationError(
                    field=f'technical_parameters.indicators.{name}',
                    value=type(params).__name__,
                    message=f"Parameters must be a dict, {type(params).__name__} was given"
                ))

        # ATR_BASED check: Is the ATR indicator required?
        if hasattr(strategy, 'exit_strategy'):
            sl_method = getattr(strategy.exit_strategy, 'stop_loss_method', None)
            tp_method = getattr(strategy.exit_strategy, 'take_profit_method', None)

            sl_name = sl_method.name if hasattr(sl_method, 'name') else str(sl_method) if sl_method else None
            tp_name = tp_method.name if hasattr(tp_method, 'name') else str(tp_method) if tp_method else None

            # Is ATR_BASED selected?
            if sl_name == 'ATR_BASED' or tp_name == 'ATR_BASED':
                # Is ATR available?
                import re
                atr_pattern = re.compile(r'^atr(_\d+)?$')
                has_atr = any(atr_pattern.match(key) for key in indicators.keys())

                if not has_atr:
                    # Automatically add
                    indicators['atr_14'] = {'period': 14}
                    # Cache invalidate flag (backtest engine checks)
                    strategy._cache_invalidated = True
                    self.warnings.append(ValidationError(
                        field='technical_parameters.indicators',
                        value='atr_14',
                        message='ATR_BASED is selected, but ATR is not present -> Automatically added atr_14'
                    ))
    
    def _validate_risk_management(self, rm: Any):
        """Validate RiskManagement parameters (ENUM + values)"""
        # Enum validation (only if an enum is imported)
        if PositionSizeMethod is not None and hasattr(rm, 'sizing_method'):
            if not isinstance(rm.sizing_method, PositionSizeMethod):
                self.errors.append(ValidationError(
                    field='risk_management.sizing_method',
                    value=type(rm.sizing_method).__name__,
                    message='must be a PositionSizeMethod enum'
                ))
        
        # Validate position sizing parameters based on sizing_method
        if hasattr(rm, 'sizing_method'):
            if rm.sizing_method == PositionSizeMethod.FIXED_PERCENT:
                # FIXED_PERCENT requires position_percent_size
                if not hasattr(rm, 'position_percent_size') or rm.position_percent_size <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.position_percent_size',
                        value=getattr(rm, 'position_percent_size', None),
                        message='The position_percent_size must be positive for the FIXED_PERCENT method'
                    ))
            elif rm.sizing_method == PositionSizeMethod.FIXED_USD:
                # FIXED_USD requires position_usd_size
                if not hasattr(rm, 'position_usd_size') or rm.position_usd_size <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.position_usd_size',
                        value=getattr(rm, 'position_usd_size', None),
                        message='The position_usd_size must be positive for the FIXED_USD method'
                    ))
            elif rm.sizing_method == PositionSizeMethod.RISK_BASED:
                # RISK_BASED requires max_risk_per_trade
                if not hasattr(rm, 'max_risk_per_trade') or rm.max_risk_per_trade <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.max_risk_per_trade',
                        value=getattr(rm, 'max_risk_per_trade', None),
                        message='For the RISK_BASED method, max_risk_per_trade must be positive'
                    ))
                elif rm.max_risk_per_trade > 10:
                    self.warnings.append(ValidationError(
                        field='risk_management.max_risk_per_trade',
                        value=rm.max_risk_per_trade,
                        message='Trading carries a risk of over 10% - very high!',
                        severity='warning'
                    ))
        
        # Portfolio risk control
        if hasattr(rm, 'max_portfolio_risk'):
            if rm.max_portfolio_risk <= 0:
                self.errors.append(ValidationError(
                    field='risk_management.max_portfolio_risk',
                    value=rm.max_portfolio_risk,
                    message='max_portfolio_risk must be positive'
                ))
        
        # Maximum drawdown check
        if hasattr(rm, 'max_drawdown'):
            if rm.max_drawdown < 5:
                self.warnings.append(ValidationError(
                    field='risk_management.max_drawdown',
                    value=rm.max_drawdown,
                    message='Max drawdown is very low (<5%), there might be frequent stops!',
                    severity='warning'
                ))
    
    def _validate_exit_strategy(self, es: Any):
        """Validate ExitStrategy parameters (ENUM + values)"""
        # Enum validations (only if an enum is imported)
        if ExitMethod is not None and hasattr(es, 'take_profit_method'):
            if not isinstance(es.take_profit_method, ExitMethod):
                self.errors.append(ValidationError(
                    field='exit_strategy.take_profit_method',
                    value=type(es.take_profit_method).__name__,
                    message='take_profit_method must be an ExitMethod enum'
                ))
        
        if StopLossMethod is not None and hasattr(es, 'stop_loss_method'):
            if not isinstance(es.stop_loss_method, StopLossMethod):
                self.errors.append(ValidationError(
                    field='exit_strategy.stop_loss_method',
                    value=type(es.stop_loss_method).__name__,
                    message='stop_loss_method must be a StopLossMethod enum'
                ))

        # Take profit check
        if hasattr(es, 'take_profit_value'):
            if es.take_profit_value <= 0:
                self.errors.append(ValidationError(
                    field='exit_strategy.take_profit_value',
                    value=es.take_profit_value,
                    message='take_profit_value must be positive'
                ))
        
        # Stop loss control
        if hasattr(es, 'stop_loss_value'):
            if es.stop_loss_value <= 0:
                self.errors.append(ValidationError(
                    field='exit_strategy.stop_loss_value',
                    value=es.stop_loss_value,
                    message='stop_loss_value must be positive'
                ))
        
        # TP/SL ratio check (risk/reward)
        if hasattr(es, 'take_profit_value') and hasattr(es, 'stop_loss_value'):
            if es.take_profit_value > 0 and es.stop_loss_value > 0:
                ratio = es.take_profit_value / es.stop_loss_value
                if ratio < 1.0:
                    self.warnings.append(ValidationError(
                        field='exit_strategy',
                        value=f'TP/SL={ratio:.2f}',
                        message=f'TP/SL orani dusuk ({ratio:.2f}), risk/reward olumsuz!',
                        severity='warning'
                    ))
        
        # Trailing stop control
        if hasattr(es, 'trailing_stop_enabled') and es.trailing_stop_enabled:
            if not hasattr(es, 'trailing_activation_profit_percent'):
                self.errors.append(ValidationError(
                    field='exit_strategy.trailing_activation_profit_percent',
                    value=None,
                    message='Trailing stop is active, but the activation level is not set!'
                ))
        
        # Partial exit control
        if hasattr(es, 'partial_exit_enabled') and es.partial_exit_enabled:
            if not hasattr(es, 'partial_exit_levels') or not es.partial_exit_levels:
                self.errors.append(ValidationError(
                    field='exit_strategy.partial_exit_levels',
                    value=None,
                    message='partial_exit_enabled=True ama partial_exit_levels bos'
                ))
            
            if not hasattr(es, 'partial_exit_sizes') or not es.partial_exit_sizes:
                self.errors.append(ValidationError(
                    field='exit_strategy.partial_exit_sizes',
                    value=None,
                    message='partial_exit_enabled=True ama partial_exit_sizes bos'
                ))
            
            if hasattr(es, 'partial_exit_levels') and hasattr(es, 'partial_exit_sizes'):
                if len(es.partial_exit_levels) != len(es.partial_exit_sizes):
                    self.errors.append(ValidationError(
                        field='exit_strategy.partial_exit',
                        value=f'levels:{len(es.partial_exit_levels)}, sizes:{len(es.partial_exit_sizes)}',
                        message='partial_exit_levels and partial_exit_sizes must have the same length'
                    ))
    
    def _validate_position_management(self, pm: Any):
        """PositionManagement parametrelerini validate et"""
        # Position limit check
        if hasattr(pm, 'max_total_positions'):
            if pm.max_total_positions <= 0:
                self.errors.append(ValidationError(
                    field='position_management.max_total_positions',
                    value=pm.max_total_positions,
                    message='max_total_positions must be positive'
                ))
            elif pm.max_total_positions > 20:
                self.warnings.append(ValidationError(
                    field='position_management.max_total_positions',
                    value=pm.max_total_positions,
                    message='Too many concurrent positions (>20) can be risky!',
                    severity='warning'
                ))
        
        if hasattr(pm, 'max_positions_per_symbol'):
            if pm.max_positions_per_symbol <= 0:
                self.errors.append(ValidationError(
                    field='position_management.max_positions_per_symbol',
                    value=pm.max_positions_per_symbol,
                    message='max_positions_per_symbol must be positive'
                ))
        
        # Logical check
        if hasattr(pm, 'max_positions_per_symbol') and hasattr(pm, 'max_total_positions'):
            if pm.max_positions_per_symbol > pm.max_total_positions:
                self.errors.append(ValidationError(
                    field='position_management',
                    value=f'per_symbol:{pm.max_positions_per_symbol}, total:{pm.max_total_positions}',
                    message='max_positions_per_symbol cannot be greater than max_total_positions'
                ))
        
        # Pyramiding check
        if hasattr(pm, 'pyramiding_enabled') and pm.pyramiding_enabled:
            if not hasattr(pm, 'pyramiding_max_entries'):
                self.errors.append(ValidationError(
                    field='position_management.pyramiding_max_entries',
                    value=None,
                    message='Pyramiding is active, but max_entries is not defined!'
                ))
            elif pm.pyramiding_max_entries > 10:
                self.warnings.append(ValidationError(
                    field='position_management.pyramiding_max_entries',
                    value=pm.pyramiding_max_entries,
                    message='Too many pyramiding entries (>10), very risky!',
                    severity='warning'
                ))
        
        # Timeout check
        if hasattr(pm, 'position_timeout_enabled') and pm.position_timeout_enabled:
            if not hasattr(pm, 'position_timeout'):
                self.errors.append(ValidationError(
                    field='position_management.position_timeout',
                    value=None,
                    message='Timeout is active, but there is no timeout duration!'
                ))
    
    def _validate_conditions(self, strategy: Any, field_name: str):
        """Validate entry/exit conditions (DETAILED)"""
        conditions = getattr(strategy, field_name, {})
        
        if not isinstance(conditions, dict):
            self.errors.append(ValidationError(
                field=field_name,
                value=type(conditions).__name__,
                message=f'{field_name} must be a dictionary!'
            ))
            return
        
        # It should be at least long or short (for the entry)
        if field_name == 'entry_conditions':
            if 'long' not in conditions and 'short' not in conditions:
                self.errors.append(ValidationError(
                    field=field_name,
                    value=conditions.keys(),
                    message="entry_conditions must be at least 'long' or 'short'"
                ))
        
        # Check each condition group
        for side, cond_list in conditions.items():
            if not isinstance(cond_list, list):
                self.errors.append(ValidationError(
                    field=f'{field_name}.{side}',
                    value=type(cond_list).__name__,
                    message=f'{side} conditions must be a list!'
                ))
                continue
            
            # Check each condition
            for idx, condition in enumerate(cond_list):
                self._validate_single_condition(condition, f'{field_name}.{side}[{idx}]', strategy)
    
    def _validate_single_condition(self, condition: Any, context: str, strategy: Any):
        """Validate a single condition (DETAILED: syntax, operator, timeframe)"""
        if not isinstance(condition, (list, tuple)):
            self.errors.append(ValidationError(
                field=context,
                value=type(condition).__name__,
                message='The condition must be a list or tuple!'
            ))
            return
        
        if len(condition) < 3:
            self.errors.append(ValidationError(
                field=context,
                value=condition,
                message='The condition must contain at least 3 elements: [indicator, operator, value]'
            ))
            return
        
        if len(condition) > 4:
            self.errors.append(ValidationError(
                field=context,
                value=condition,
                message='The condition must contain a maximum of 4 elements: [indicator, operator, value, timeframe]'
            ))
            return
        
        # Operator check
        operator = condition[1]
        if operator not in self.VALID_OPERATORS:
            self.errors.append(ValidationError(
                field=f'{context}.operator',
                value=operator,
                message=f"Invalid operator: '{operator}'. Valid operators: {sorted(self.VALID_OPERATORS)}"
            ))
        
        # Timeframe check (if exists)
        if len(condition) == 4:
            timeframe = condition[3]
            if timeframe and timeframe not in self.VALID_TIMEFRAMES:
                self.errors.append(ValidationError(
                    field=f'{context}.timeframe',
                    value=timeframe,
                    message=f"Invalid timeframe: '{timeframe}'. Valid: {sorted(self.VALID_TIMEFRAMES)}"
                ))
            
            # Timeframe must be within the mtf_timeframes list.
            if hasattr(strategy, 'mtf_timeframes') and timeframe:
                if timeframe not in strategy.mtf_timeframes:
                    self.warnings.append(ValidationError(
                        field=f'{context}.timeframe',
                        value=timeframe,
                        message=f"Timeframe '{timeframe}' is not within mtf_timeframes!",
                        severity='warning'
                    ))
    
    def _validate_custom_parameters(self, params: Dict):
        """Validate custom parameters"""
        if not isinstance(params, dict):
            self.errors.append(ValidationError(
                field='custom_parameters',
                value=type(params).__name__,
                message='custom_parameters must be a dictionary!'
            ))
    
    def _validate_optimizer_parameters(self, params: Any):
        """Validate optimizer parameters"""
        # Optimizer parameters are optional, only check if they exist.
        if params is None:
            return
        
        if not isinstance(params, dict):
            self.errors.append(ValidationError(
                field='optimizer_parameters',
                value=type(params).__name__,
                message='optimizer_parameters must be a dictionary!'
            ))
    
    def _validate_account_management(self, acc_mgmt: Any):
        """Validate the account management configuration"""
        # Leverage validation
        if hasattr(acc_mgmt, 'leverage'):
            if acc_mgmt.leverage < 1 or acc_mgmt.leverage > 125:
                self.errors.append(ValidationError(
                    field='account_management.leverage',
                    value=acc_mgmt.leverage,
                    message='must be between 1 and 125'
                ))
        
        # Initial balance validation
        if hasattr(acc_mgmt, 'initial_balance'):
            if acc_mgmt.initial_balance <= 0:
                self.errors.append(ValidationError(
                    field='account_management.initial_balance',
                    value=acc_mgmt.initial_balance,
                    message='initial_balance must be positive'
                ))
    
    def _validate_backtest_parameters(self, strategy: Any):
        """Validates the backtest parameters configuration."""
        bt_params = strategy.backtest_parameters
        
        # Date validation
        if hasattr(strategy, 'backtest_start_date') and hasattr(strategy, 'backtest_end_date'):
            if strategy.backtest_start_date and strategy.backtest_end_date:
                if strategy.backtest_start_date >= strategy.backtest_end_date:
                    self.errors.append(ValidationError(
                        field='backtest_dates',
                        value=f'start:{strategy.backtest_start_date}, end:{strategy.backtest_end_date}',
                        message='The backtest start date must be before the backtest end date'
                    ))
        
        # Commission validation
        if hasattr(bt_params, 'commission'):
            if bt_params.commission < 0:
                self.errors.append(ValidationError(
                    field='backtest_parameters.commission',
                    value=bt_params.commission,
                    message='commission cannot be negative'
                ))
        
        # Slippage validation
        if hasattr(bt_params, 'max_slippage'):
            if bt_params.max_slippage < 0:
                self.errors.append(ValidationError(
                    field='backtest_parameters.max_slippage',
                    value=bt_params.max_slippage,
                    message='max_slippage cannot be negative'
                ))


def validate_strategy(strategy: Any, logger: Optional[Any] = None) -> StrategyValidationResult:
    """
    Stratejiyi validate et (helper function)
    
    Args:
        strategy: Strategy object
        logger: Optional logger
    
    Returns:
        StrategyValidationResult
    
    Example:
        >>> from components.strategies.templates.Golden_Cross_Trend import Strategy
        >>> strategy = Strategy()
        >>> result = validate_strategy(strategy)
        >>> if result.valid:
        >>>     print("Strategy is usable!")
        >>> else:
        >>>     for error in result.errors:
        >>>         print(f"Error: {error.message}")
    """
    validator = StrategyValidator(logger=logger)
    return validator.validate_strategy(strategy)


# For backward compatibility, the old exception-based function.
class ValidationException(Exception):
    """Validation exception (for backward compatibility)"""
    pass


def validate_strategy_strict(strategy: Any) -> None:
    """
    Validate the strategy and raise an exception if there is an error.
    
    (For backward compatibility: old behavior from helpers/validation.py)
    
    Args:
        strategy: Strategy object
    
    Raises:
        ValidationException: If there is a validation error.
    """
    result = validate_strategy(strategy)
    if not result.valid:
        error_messages = [f"{err.field}: {err.message}" for err in result.errors]
        raise ValidationException("\n".join(error_messages))


__all__ = [
    'StrategyValidator',
    'ValidationError',
    'StrategyValidationResult',
    'ValidationException',
    'validate_strategy',
    'validate_strategy_strict',
]
