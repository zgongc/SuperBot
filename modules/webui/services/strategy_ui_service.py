"""
Strategy UI Service
Business logic layer for strategy management in WebUI
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import shutil
from datetime import datetime

from .base_service import BaseService
from components.indicators import INDICATOR_REGISTRY


class StrategyUIService(BaseService):
    """Strategy management service for WebUI"""

    def __init__(self, data_manager, strategy_manager, logger):
        super().__init__(data_manager, logger)
        self.strategy_manager = strategy_manager
        self.template_path = Path("components/strategies/templates/")

    async def get_all_strategies(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all available strategy templates from filesystem

        Args:
            status_filter: Optional filter ('active' or 'inactive') - NOT USED (all templates shown)

        Returns:
            List of strategy info dictionaries
        """
        strategies = []

        # Scan template directory (same as get_available_templates but with more details)
        if self.template_path.exists():
            for file_path in self.template_path.glob("*.py"):
                if file_path.name.startswith('__'):
                    continue

                # Try to extract strategy metadata from file
                description = ""
                version = "1.0.0"
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # Extract description from docstring
                    if '"""' in content:
                        docstring_start = content.find('"""') + 3
                        docstring_end = content.find('"""', docstring_start)
                        if docstring_end > docstring_start:
                            description = content[docstring_start:docstring_end].strip()[:200]
                except Exception:
                    pass

                strategy_info = {
                    'id': file_path.stem,
                    'name': file_path.stem.replace('_', ' ').title(),
                    'version': version,
                    'description': description,
                    'status': 'inactive',  # All templates shown as inactive (not loaded in memory)
                    'symbols': [],
                    'symbol_count': 0,
                    'created_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }

                strategies.append(strategy_info)

        # Sort by name
        strategies.sort(key=lambda x: x['name'])

        return strategies

    async def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a single strategy

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy details dictionary or None if not found
        """
        # Try to load from StrategyManager first
        if strategy_id not in self.strategy_manager.loaded_strategies:
            # If not found, try to load from templates folder
            basic_path = self.template_path / "basic" / f"{strategy_id}.py"
            main_path = self.template_path / f"{strategy_id}.py"

            template_path = None
            if basic_path.exists():
                template_path = basic_path
            elif main_path.exists():
                template_path = main_path

            if template_path:
                # Load this strategy
                try:
                    strategy_obj, executor = self.strategy_manager.load_strategy(str(template_path))
                except Exception as e:
                    self.logger.error(f"❌ Failed to load strategy {strategy_id}: {e}")
                    return None
            else:
                return None
        else:
            strategy_obj = self.strategy_manager.loaded_strategies[strategy_id]

        # Extract symbols - handle both single and nested lists
        symbols = []
        if hasattr(strategy_obj, 'symbols'):
            for s in strategy_obj.symbols:
                if hasattr(s, 'symbol'):
                    symbol_val = s.symbol
                    if isinstance(symbol_val, list):
                        symbols.extend(symbol_val)
                    else:
                        symbols.append(symbol_val)
                else:
                    symbols.append(str(s))

        # Get detailed information
        strategy_detail = {
            'id': strategy_id,
            'name': strategy_obj.strategy_name if hasattr(strategy_obj, 'strategy_name') else strategy_id,
            'version': strategy_obj.strategy_version if hasattr(strategy_obj, 'strategy_version') else '1.0.0',
            'description': strategy_obj.strategy_description if hasattr(strategy_obj, 'strategy_description') else '',
            'status': 'active' if (hasattr(strategy_obj, 'enabled') and strategy_obj.enabled) else 'inactive',
            'symbols': symbols,
            'timeframes': getattr(strategy_obj, 'timeframes', None) or getattr(strategy_obj, 'mtf_timeframes', []),
            'exchange': strategy_obj.exchange if hasattr(strategy_obj, 'exchange') else None,
            'created_at': strategy_obj.created_at.isoformat() if hasattr(strategy_obj, 'created_at') and strategy_obj.created_at else None,
            'updated_at': strategy_obj.updated_at.isoformat() if hasattr(strategy_obj, 'updated_at') and strategy_obj.updated_at else None,

            # Performance metrics (if available)
            'performance': {
                'total_signals': getattr(strategy_obj, 'total_signals', 0),
                'successful_signals': getattr(strategy_obj, 'successful_signals', 0),
                'win_rate': getattr(strategy_obj, 'win_rate', 0),
                'avg_profit': getattr(strategy_obj, 'avg_profit', 0),
            }
        }

        return strategy_detail

    async def create_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new strategy from UI data

        Args:
            data: Strategy configuration dictionary from UI

        Returns:
            Created strategy info

        Raises:
            ValueError: If data is invalid
        """
        # Validate required fields
        if 'name' not in data:
            raise ValueError('Strategy name is required')

        strategy_name = data['name']

        # Check if strategy file already exists
        target_file = self.template_path / f"{strategy_name}.py"
        if target_file.exists():
            raise ValueError(f'A strategy file with the name "{strategy_name}" already exists')

        # Generate Python strategy code from UI data
        strategy_code = self._generate_strategy_code(strategy_name, data)

        # Save to file
        target_file.write_text(strategy_code, encoding='utf-8')

        self.logger.info(f"✅ New strategy created: {target_file}")

        # Reload strategies to include new one
        await self._reload_strategies()

        return await self.get_strategy_by_id(strategy_name)

    async def update_strategy(self, strategy_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update existing strategy configuration

        Args:
            strategy_id: Strategy identifier
            data: Updated configuration

        Returns:
            Updated strategy info or None if not found
        """
        # Check if template file exists (check both main and basic folders)
        main_file = self.template_path / f"{strategy_id}.py"
        basic_file = self.template_path / "basic" / f"{strategy_id}.py"

        if not main_file.exists() and not basic_file.exists():
            self.logger.error(f"❌ Strategy template not found: {strategy_id}")
            return None

        # Use existing file location or default to main folder
        target_file = main_file if main_file.exists() else basic_file

        # Generate Python strategy file
        strategy_code = self._generate_strategy_code(strategy_id, data)

        # Save to file (overwrite)
        target_file.write_text(strategy_code, encoding='utf-8')

        self.logger.info(f"✅ Strategy updated: {target_file}")

        # Reload the strategy in memory if it was loaded
        if strategy_id in self.strategy_manager.loaded_strategies:
            try:
                self.strategy_manager.reload_strategy(strategy_id, str(target_file))
                self.logger.info(f"♻️ Strategy reloaded in memory: {strategy_id}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not reload strategy in memory: {e}")

        return await self.get_strategy_by_id(strategy_id)

    def _generate_strategy_code(self, strategy_id: str, data: Dict[str, Any]) -> str:
        """
        Generate Python strategy code from UI data

        Args:
            strategy_id: Strategy identifier
            data: Strategy configuration from UI

        Returns:
            Python code as string
        """
        name = data.get('name', strategy_id)
        version = data.get('version', '1.0.0')
        description = data.get('description', '')
        author = data.get('author', 'WebUI Generated')
        warmup_period = data.get('warmup_period', 100)
        symbols = data.get('symbols', [])

        # Global Trading Config
        trading_side = data.get('trading_side', 'long_short')
        leverage = data.get('leverage', 1)
        symbol_source = data.get('symbol_source', 'strategy')

        # Timeframes
        timeframes = data.get('timeframes', ['15m'])
        primary_timeframe = data.get('primary_timeframe', '15m')

        # Margin Config
        set_default_leverage = data.get('set_default_leverage', False)
        hedge_mode = data.get('hedge_mode', False)
        set_margin_type = data.get('set_margin_type', False)
        margin_type = data.get('margin_type', 'isolated')

        # Backtest Config
        backtest_start = data.get('backtest_start_date', '2025-01-01')
        backtest_end = data.get('backtest_end_date', '2025-12-31')
        initial_balance = data.get('initial_balance', 10000.0)
        download_klines = data.get('download_klines', False)
        update_klines = data.get('update_klines', False)

        # Backtest Parameters
        commission = data.get('commission', 0.075)
        min_spread = data.get('min_spread', 0.01)
        max_slippage = data.get('max_slippage', 0.02)

        # Risk Management
        risk_mgmt = data.get('risk_management', {})

        # Position Management
        pos_mgmt = data.get('position_management', {})

        # Exit Strategy
        exit_strat = data.get('exit_strategy', {})

        # Indicators
        indicators_data = data.get('indicators', [])

        # Entry/Exit Conditions
        entry_conditions = data.get('entry_conditions', {"long": [], "short": []})
        exit_conditions = data.get('exit_conditions', {"long": [], "short": []})

        # Custom Parameters / Filters
        custom_params = data.get('custom_parameters', {})

        # Build indicators dict for template
        indicators_dict = {}
        for ind in indicators_data:
            key = ind['key']
            suffix = ind.get('suffix', '')
            params = ind.get('params', {})

            # Use suffix if provided, otherwise use key directly
            indicator_name = f"{key}_{suffix}" if suffix else key
            indicators_dict[indicator_name] = params

        # Generate code
        code = f'''#!/usr/bin/env python3
"""
components/strategies/templates/{strategy_id}_new.py
SuperBot - {name}
Generated by WebUI
Version: {version}

{description}
"""

import sys
from pathlib import Path

# SuperBot base directory'yi path'e ekle
base_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

from components.strategies.base_strategy import (
    BaseStrategy,
    SymbolConfig,
    TechnicalParameters,
    RiskManagement,
    ExitStrategy,
    PositionManagement,
    TradingSide,
    PositionSizeMethod,
    ExitMethod,
    StopLossMethod
)


class Strategy(BaseStrategy):
    """
    {name}

    {description}
    """

    def __init__(self):
        super().__init__()

        # STRATEGY METADATA
        self.strategy_name = "{name}"
        self.strategy_version = "{version}"
        self.description = "{description}"
        self.author = "{author}"
        self.created_date = "{datetime.now().strftime('%Y-%m-%d')}"
        self.warmup_period = {warmup_period}

        # BACKTEST CONFIGURATION
        self.backtesting_enabled = True
        self.backtest_start_date = "{backtest_start}"
        self.backtest_end_date = "{backtest_end}"
        self.initial_balance = {initial_balance}
        self.download_klines = {download_klines}
        self.update_klines = {update_klines}

        # BACKTEST PARAMETERS
        self.backtest_parameters = {{
            "min_spread": {min_spread},
            "commission": {commission},
            "max_slippage": {max_slippage}
        }}

'''

        # Add global trading config
        side_map = {'long_only': 'LONG', 'short_only': 'SHORT', 'long_short': 'BOTH', 'both': 'BOTH'}
        side_enum = side_map.get(trading_side.lower(), 'BOTH')

        code += f'''
        # TRADING CONFIGURATION
        self.side_method = TradingSide.{side_enum}
        self.leverage = {leverage}
        # Timeframe configuration
        self.primary_timeframe = "{primary_timeframe}"
        self.mtf_timeframes = {timeframes}

        self.set_default_leverage = {set_default_leverage}
        self.hedge_mode = {hedge_mode}
        self.set_margin_type = {set_margin_type}
        self.margin_type = "{margin_type}"

        # SYMBOLS
        self.symbol_source = "{symbol_source}"
        self.symbols = [
'''

        # Add single SymbolConfig with all symbols
        symbols_str = str(symbols) if symbols else '["BTC"]'
        code += f'''            SymbolConfig(
                symbol={symbols_str},
                quote="USDT",
                enabled=True
            )
        ]

        # RISK MANAGEMENT
        self.risk_management = RiskManagement(
'''
        # Convert sizing_method to correct enum value (frontend sends 'sizing_method', not 'position_size_type')
        size_type = risk_mgmt.get('sizing_method', risk_mgmt.get('position_size_type', 'fixed')).lower()
        size_type_map = {
            'fixed': 'FIXED_USD',
            'fixed_usd': 'FIXED_USD',
            'fixed_percent': 'FIXED_PERCENT',
            'risk_based': 'RISK_BASED',
            'risk': 'RISK_BASED',
            'kelly': 'KELLY_CRITERION',
            'kelly_criterion': 'KELLY_CRITERION',
            'ai': 'AI_DYNAMIC',
            'ai_dynamic': 'AI_DYNAMIC'
        }
        size_type_enum = size_type_map.get(size_type, 'FIXED_USD')

        # Get size values with fallbacks
        size_value = risk_mgmt.get('size_value', 10.0)

        code += f'''            sizing_method=PositionSizeMethod.{size_type_enum},
            position_percent_size={size_value if size_type_enum == 'FIXED_PERCENT' else 10.0},
            position_usd_size={size_value if size_type_enum == 'FIXED_USD' else 300.0},
            position_quantity_size={risk_mgmt.get('position_quantity_size', 2.0)},
            max_risk_per_trade={risk_mgmt.get('max_risk_per_trade', 2.5)},

            max_correlation={risk_mgmt.get('max_correlation', 0.6)},
            position_correlation_limit={risk_mgmt.get('position_correlation_limit', 0.7)},
            max_drawdown={risk_mgmt.get('max_drawdown', 100)},
            max_daily_trades={risk_mgmt.get('max_daily_trades', 300)},
            emergency_stop_enabled={risk_mgmt.get('emergency_stop_enabled', True)},
            ai_risk_enabled={risk_mgmt.get('ai_risk_enabled', False)},
            dynamic_position_sizing={risk_mgmt.get('dynamic_position_sizing', True)},
        )

        # POSITION MANAGEMENT
        self.position_management = PositionManagement(
            max_positions_per_symbol={pos_mgmt.get('max_positions_per_symbol', 1)},
            max_total_positions={pos_mgmt.get('max_total_positions', 400)},
            allow_hedging={pos_mgmt.get('allow_hedging', False)},
            position_timeout_enabled={pos_mgmt.get('position_timeout_enabled', True)},
            position_timeout={pos_mgmt.get('position_timeout', 1800)},
            pyramiding_enabled={pos_mgmt.get('pyramiding_enabled', False)},
            pyramiding_max_entries={pos_mgmt.get('pyramiding_max_entries', 3)},
            pyramiding_scale_factor={pos_mgmt.get('pyramiding_scale_factor', 0.5)}
        )

        # EXIT STRATEGY
        self.exit_strategy = ExitStrategy(
            # Take Profit
            take_profit_method=ExitMethod.{exit_strat.get('take_profit_method', 'FIXED_PERCENT').upper()},
            take_profit_percent={exit_strat.get('take_profit_percent', exit_strat.get('take_profit_value', 3.0))},
            take_profit_price={exit_strat.get('take_profit_price', 0.0)},
            take_profit_risk_reward_ratio={exit_strat.get('take_profit_risk_reward_ratio', 2.0)},
            take_profit_atr_multiplier={exit_strat.get('take_profit_atr_multiplier', 3.0)},
            take_profit_fib_level={exit_strat.get('take_profit_fib_level', 1.618)},
            take_profit_ai_level={exit_strat.get('take_profit_ai_level', 1)},

            # Stop Loss
            stop_loss_method=StopLossMethod.{exit_strat.get('stop_loss_method', 'FIXED_PERCENT').upper()},
            stop_loss_percent={exit_strat.get('stop_loss_percent', exit_strat.get('stop_loss_value', 1.5))},
            stop_loss_price={exit_strat.get('stop_loss_price', 0.0)},
            stop_loss_atr_multiplier={exit_strat.get('stop_loss_atr_multiplier', 2.0)},
            stop_loss_swing_lookback={exit_strat.get('stop_loss_swing_lookback', 10)},
            stop_loss_fib_level={exit_strat.get('stop_loss_fib_level', 0.382)},
            stop_loss_ai_level={exit_strat.get('stop_loss_ai_level', 1)},

            # Trailing Stop
            trailing_stop_enabled={exit_strat.get('trailing_stop_enabled', False)},
            trailing_activation_profit_percent={exit_strat.get('trailing_activation_profit_percent', exit_strat.get('trailing_activation_profit', 3.0))},
            trailing_callback_percent={exit_strat.get('trailing_callback_percent', 0.5)},
            trailing_take_profit={exit_strat.get('trailing_take_profit', False)},
            trailing_distance={exit_strat.get('trailing_distance', 0.2)},

            # Break-even
            break_even_enabled={exit_strat.get('break_even_enabled', False)},
            break_even_trigger_profit_percent={exit_strat.get('break_even_trigger_profit_percent', 1.2)},
            break_even_offset={exit_strat.get('break_even_offset', 0.3)},

            # Partial Exit
            partial_exit_enabled={exit_strat.get('partial_exit_enabled', False)},
            partial_exit_levels={exit_strat.get('partial_exit_levels', [5, 10, 25])},
            partial_exit_sizes={exit_strat.get('partial_exit_sizes', [0.40, 0.40, 0.20])},
        )

        # TECHNICAL INDICATORS
        self.technical_parameters = TechnicalParameters(
            indicators={{
'''

        # Add indicators
        for ind_name, params in indicators_dict.items():
            code += f'                "{ind_name}": {{\n'
            for param_name, param_value in params.items():
                code += f'                    "{param_name}": {param_value},\n'
            code += '                },\n'

        code += '''            }
        )

        # ENTRY CONDITIONS
        self.entry_conditions = {
'''

        # Add entry conditions
        for side, conditions in entry_conditions.items():
            code += f'            "{side}": [\n'
            for cond in conditions:
                code += f'                {cond},\n'
            code += '            ],\n'

        code += '''        }

        # EXIT CONDITIONS
        self.exit_conditions = {
'''

        # Add exit conditions
        for side, conditions in exit_conditions.items():
            code += f'            "{side}": [\n'
            for cond in conditions:
                code += f'                {cond},\n'
            code += '            ],\n'

        code += '''        }

        # CUSTOM PARAMETERS
        self.custom_parameters = {
'''

        # Add News Filter
        news_filter = custom_params.get('news_filter', False)
        code += f'''            "news_filter": {news_filter},
'''

        # Add Session Filter
        if custom_params.get('session_filter'):
            sf = custom_params['session_filter']
            code += '''
            # Session Filter
            "session_filter": {
'''
            code += f'                "enabled": {sf.get("enabled", False)},\n'
            code += f'                "sydney": {sf.get("sydney", True)},\n'
            code += f'                "tokyo": {sf.get("tokyo", True)},\n'
            code += f'                "london": {sf.get("london", True)},\n'
            code += f'                "new_york": {sf.get("new_york", True)},\n'
            code += f'                "london_ny_overlap": {sf.get("london_ny_overlap", True)},\n'
            code += '            },\n'

        # Add Time Filter
        if custom_params.get('time_filter'):
            tf = custom_params['time_filter']
            code += '''
            # Time Filter
            "time_filter": {
'''
            code += f'                "enabled": {tf.get("enabled", False)},\n'
            code += f'                "start_hour": {tf.get("start_hour", 8)},\n'
            code += f'                "end_hour": {tf.get("end_hour", 21)},\n'
            code += f'                "exclude_hours": {tf.get("exclude_hours", [])},\n'
            code += '            },\n'

        # Add Day Filter
        if custom_params.get('day_filter'):
            df = custom_params['day_filter']
            code += '''
            # Day Filter
            "day_filter": {
'''
            code += f'                "enabled": {df.get("enabled", False)},\n'
            code += f'                "monday": {df.get("monday", True)},\n'
            code += f'                "tuesday": {df.get("tuesday", True)},\n'
            code += f'                "wednesday": {df.get("wednesday", True)},\n'
            code += f'                "thursday": {df.get("thursday", True)},\n'
            code += f'                "friday": {df.get("friday", True)},\n'
            code += f'                "saturday": {df.get("saturday", True)},\n'
            code += f'                "sunday": {df.get("sunday", True)},\n'
            code += '            },\n'

        code += '''        }

        # OPTIMIZER PARAMETERS
        # Format: (min, max, step) for numeric or [choice1, choice2] for categorical
        #
        # MULTI-STAGE OPTIMIZATION STRATEGY:
        # Set 'enabled': False to skip a stage, True to activate it
        #
        # Stage 1: Risk Management (50-100 trials)
        #   - Optimize position sizing first (most critical)
        #   - indicators: enabled=False, exit_strategy: enabled=False, risk_management: enabled=True
        #
        # Stage 2: Exit Strategy (50-100 trials)
        #   - Use best risk params from Stage 1 (apply manually)
        #   - Optimize TP/SL/break-even/trailing
        #   - indicators: enabled=False, exit_strategy: enabled=True, risk_management: enabled=False
        #
        # Stage 3: Indicators (100-200 trials)
        #   - Use best risk + exit params from Stage 1+2 (apply manually)
        #   - Optimize indicator periods and thresholds
        #   - indicators: enabled=True, exit_strategy: enabled=False, risk_management: enabled=False
        #
        # Stage 4: Fine-tune (50 trials) - Optional
        #   - Set all enabled=True
        #   - Use small ranges around best values from previous stages
        #
        self.optimizer_parameters = {
            # ================================================================
            # STAGE 0: Main Strategy Parameters
            # ================================================================
            'main_strategy': {
                'enabled': False,
                #'side_method': ['BOTH', 'LONG', 'SHORT'],
                #'leverage': (1, 20, 1),
            },

            # ================================================================
            # STAGE 1: Risk Management (PRIORITY: HIGHEST)
            # ================================================================
            'risk_management': {
                'enabled': False,
                #'sizing_method': ['FIXED_PERCENT', 'RISK_BASED', 'FIXED_USD'],
                #'position_percent_size': (5.0, 20.0, 1),
                #'position_usd_size': (100, 1000, 100),
                #'max_risk_per_trade': (1.0, 5.0, 0.5),
            },

            # ================================================================
            # STAGE 2: Exit Strategy (PRIORITY: HIGH)
            # ================================================================
            'exit_strategy': {
                'enabled': False,
                #'stop_loss_method': ['FIXED_PERCENT', 'ATR_BASED', 'FIBONACCI'],
                #'stop_loss_percent': (0.8, 2.0, 0.1),
                #'take_profit_method': ['FIXED_PERCENT', 'RISK_REWARD', 'ATR_BASED'],
                #'take_profit_percent': (2.4, 5.2, 0.2),
                #'break_even_enabled': [True, False],
                #'break_even_trigger_profit_percent': (0.8, 2.0, 0.2),
                #'break_even_offset': (0.05, 0.30, 0.05),
                #'trailing_stop_enabled': [True, False],
                #'trailing_activation_profit_percent': (1.0, 3.0, 0.5),
                #'trailing_callback_percent': (0.3, 1.0, 0.1),
            },

            # ================================================================
            # STAGE 3: Indicators (PRIORITY: MEDIUM)
            # ================================================================
            'indicators': {
                'enabled': False,
'''

        # Add dynamic indicator optimizer params based on technical_parameters
        for ind in indicators_data:
            key = ind['key']
            params = ind.get('params', {})

            # Get base type (rsi, ema, adx, etc.)
            base_type = key.rsplit('_', 1)[0] if '_' in key else key

            code += f"                #'{base_type}': {{\n"

            # Add period if exists
            if 'period' in params:
                period = params['period']
                code += f"                #    'period': ({max(5, int(period*0.5))}, {int(period*2)}, 1),  # Period range (default: {period})\n"

            # RSI thresholds
            if 'rsi' in base_type.lower():
                code += f"                #    'overbought': (65, 80, 5),\n"
                code += f"                #    'oversold': (20, 35, 5),\n"

            # Bollinger std_dev
            elif 'bollinger' in base_type.lower() and 'std_dev' in params:
                std = params['std_dev']
                code += f"                #    'std_dev': ({std*0.75:.1f}, {std*1.5:.1f}, 0.25),\n"

            # ADX threshold
            elif 'adx' in base_type.lower():
                code += f"                #    'threshold': (17, 32, 5),\n"

            # MACD periods
            elif 'macd' in base_type.lower():
                if 'fast_period' in params:
                    code += f"                #    'fast_period': (8, 16, 2),\n"
                if 'slow_period' in params:
                    code += f"                #    'slow_period': (21, 31, 2),\n"
                if 'signal_period' in params:
                    code += f"                #    'signal_period': (7, 11, 1),\n"

            # BOS (Break of Structure)
            elif 'bos' in base_type.lower():
                left = params.get('left_bars', 5)
                right = params.get('right_bars', 5)
                levels = params.get('max_levels', 3)
                code += f"                #    'left_bars': ({max(2, left-2)}, {left+3}, 1),  # (default: {left})\n"
                code += f"                #    'right_bars': ({max(2, right-2)}, {right+3}, 1),  # (default: {right})\n"
                code += f"                #    'max_levels': ({max(1, levels-1)}, {levels+2}, 1),  # (default: {levels})\n"

            # CHoCH (Change of Character)
            elif 'choch' in base_type.lower():
                left = params.get('left_bars', 5)
                right = params.get('right_bars', 5)
                levels = params.get('max_levels', 3)
                strength = params.get('trend_strength', 3)
                code += f"                #    'left_bars': ({max(2, left-2)}, {left+3}, 1),  # (default: {left})\n"
                code += f"                #    'right_bars': ({max(2, right-2)}, {right+3}, 1),  # (default: {right})\n"
                code += f"                #    'max_levels': ({max(1, levels-1)}, {levels+2}, 1),  # (default: {levels})\n"
                code += f"                #    'trend_strength': ({max(1, strength-1)}, {strength+2}, 1),  # (default: {strength})\n"

            code += f"                #}},\n"

        code += '''            },

            # ================================================================
            # STAGE 4: Position Management (PRIORITY: LOW)
            # ================================================================
            'position_management': {
                'enabled': False,
                #'max_total_positions': (1, 5, 1),
                #'max_positions_per_symbol': (1, 3, 1),
                #'pyramiding_enabled': [True, False],
                #'pyramiding_max_entries': (2, 5, 1),
            },

            # ================================================================
            # STAGE 5: Market Filters (PRIORITY: LOW)
            # ================================================================
            'market_filters': {
                'enabled': False,
                #'day_filter_enabled': [True, False],
                #'monday_enabled': [True, False],
                #'tuesday_enabled': [True, False],
            },

            # ================================================================
            # GLOBAL CONSTRAINTS
            # ================================================================
            'constraints': {
                'max_combinations': 10000,
                'min_trades': 20,
                'min_sharpe': 0.5,
                'min_profit_factor': 1.0,
                'max_drawdown': 30.0,
                'timeout_per_backtest': 300,
            },
        }
'''

        return code

    async def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if deleted, False if not found
        """
        # Check if strategy file exists
        strategy_file = self.template_path / f"{strategy_id}.py"
        file_exists = strategy_file.exists()

        # Check if loaded in strategy manager
        in_manager = strategy_id in self.strategy_manager.loaded_strategies

        # If neither exists, return False
        if not file_exists and not in_manager:
            return False

        # Remove from strategy manager if loaded
        if in_manager:
            del self.strategy_manager.loaded_strategies[strategy_id]

        # Delete strategy file if it exists
        if file_exists:
            strategy_file.unlink()

        self.logger.info(f"Strategy deleted: {strategy_id}")
        return True

    async def set_strategy_status(self, strategy_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
        """
        Activate or deactivate a strategy

        Args:
            strategy_id: Strategy identifier
            enabled: True to activate, False to deactivate

        Returns:
            Updated strategy info or None if not found
        """
        if strategy_id not in self.strategy_manager.loaded_strategies:
            return None

        strategy_obj = self.strategy_manager.loaded_strategies[strategy_id]

        if hasattr(strategy_obj, 'enabled'):
            strategy_obj.enabled = enabled

        status = 'activated' if enabled else 'deactivated'
        self.logger.info(f"Strategy {strategy_id} {status}")

        return await self.get_strategy_by_id(strategy_id)

    async def duplicate_strategy(self, strategy_id: str, new_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Duplicate an existing strategy

        Args:
            strategy_id: Source strategy identifier
            new_name: Optional name for duplicated strategy

        Returns:
            Duplicated strategy info or None if source not found
        """
        if strategy_id not in self.strategy_manager.loaded_strategies:
            return None

        # Generate new name if not provided
        if not new_name:
            new_name = f"{strategy_id}_copy"
            counter = 1
            while new_name in self.strategy_manager.loaded_strategies:
                new_name = f"{strategy_id}_copy_{counter}"
                counter += 1

        # Check if new name already exists
        if new_name in self.strategy_manager.loaded_strategies:
            raise ValueError(f'A strategy with the name "{new_name}" already exists')

        # Copy strategy file
        source_file = self.template_path / f"{strategy_id}.py"
        target_file = self.template_path / f"{new_name}.py"

        if source_file.exists():
            shutil.copy2(source_file, target_file)

        # Reload strategies
        await self._reload_strategies()

        self.logger.info(f"Strategy duplicated: {strategy_id} -> {new_name}")

        return await self.get_strategy_by_id(new_name)

    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of available strategy templates

        Returns:
            List of template info dictionaries
        """
        templates = []

        # Scan template directory
        if self.template_path.exists():
            for file_path in self.template_path.glob("*.py"):
                if file_path.name.startswith('__'):
                    continue

                template_info = {
                    'id': file_path.stem,
                    'name': file_path.stem.replace('_', ' ').title(),
                    'file': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }

                templates.append(template_info)

        templates.sort(key=lambda x: x['name'])

        return templates

    async def get_basic_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of basic starter templates from templates/basic/ folder

        Returns:
            List of template info dictionaries with id, name, description
        """
        templates = []
        basic_path = self.template_path / "basic"

        # Scan basic templates directory
        if basic_path.exists():
            for file_path in basic_path.glob("*.py"):
                if file_path.name.startswith('__'):
                    continue

                # Try to extract description from the strategy
                description = ""
                try:
                    # Read the file to get description
                    content = file_path.read_text(encoding='utf-8')

                    # Look for description in docstring or self.description
                    import re

                    # Try to find description field in __init__
                    desc_match = re.search(r'self\.description\s*=\s*["\'](.+?)["\']', content)
                    if desc_match:
                        description = desc_match.group(1)
                    else:
                        # Try to find in docstring
                        doc_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
                        if doc_match:
                            # Get first line after header
                            lines = doc_match.group(1).strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('=') and not line.startswith('📊'):
                                    description = line
                                    break
                except Exception as e:
                    self.logger.debug(f"Could not extract description from {file_path.name}: {e}")

                template_info = {
                    'id': file_path.stem,
                    'name': file_path.stem.replace('_', ' ').title(),
                    'description': description or 'Strategy template',
                    'file': file_path.name,
                }

                templates.append(template_info)

        templates.sort(key=lambda x: x['name'])

        return templates

    async def check_strategy_exists(self, strategy_id: str) -> bool:
        """
        Check if a strategy file already exists

        Args:
            strategy_id: Strategy file name (without .py extension)

        Returns:
            True if file exists, False otherwise
        """
        strategy_file = self.template_path / f"{strategy_id}.py"
        return strategy_file.exists()

    async def get_strategy_config(self, strategy_id: str) -> Optional[str]:
        """
        Get strategy configuration in YAML format

        Args:
            strategy_id: Strategy identifier

        Returns:
            YAML configuration string or None if not found
        """
        if strategy_id not in self.strategy_manager.loaded_strategies:
            return None

        strategy_obj = self.strategy_manager.loaded_strategies[strategy_id]

        # Extract configuration from strategy object
        config = {
            'strategy_name': getattr(strategy_obj, 'strategy_name', strategy_id),
            'strategy_version': getattr(strategy_obj, 'strategy_version', '1.0.0'),
            'description': getattr(strategy_obj, 'strategy_description', ''),
            'symbols': [s.symbol for s in strategy_obj.symbols] if hasattr(strategy_obj, 'symbols') else [],
            'timeframes': getattr(strategy_obj, 'timeframes', None) or getattr(strategy_obj, 'mtf_timeframes', []),
            'exchange': getattr(strategy_obj, 'exchange', None),
            'entry_signals': getattr(strategy_obj, 'entry_signals', []),
            'exit_signals': getattr(strategy_obj, 'exit_signals', []),
            'filters': getattr(strategy_obj, 'filters', []),
            'risk_management': getattr(strategy_obj, 'risk_management', {}),
        }

        # Convert to YAML
        yaml_config = yaml.dump(config, default_flow_style=False, allow_unicode=True)

        return yaml_config

    async def validate_strategy(self, strategy_id: str, config_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate strategy configuration

        Args:
            strategy_id: Strategy identifier
            config_data: Optional configuration to validate (if not provided, validates current config)

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check if strategy exists
        if strategy_id not in self.strategy_manager.loaded_strategies:
            validation_result['valid'] = False
            validation_result['errors'].append('Strategy not found')
            return validation_result

        strategy_obj = self.strategy_manager.loaded_strategies[strategy_id]

        # Validate symbols
        if not hasattr(strategy_obj, 'symbols') or not strategy_obj.symbols:
            validation_result['warnings'].append('Symbol is not defined')

        # Validate entry signals
        if not hasattr(strategy_obj, 'entry_signals') or not strategy_obj.entry_signals:
            validation_result['warnings'].append('Input signal is not defined')

        # Validate exit signals
        if not hasattr(strategy_obj, 'exit_signals') or not strategy_obj.exit_signals:
            validation_result['warnings'].append('Output signal is not defined')

        # Validate risk management
        if not hasattr(strategy_obj, 'risk_management') or not strategy_obj.risk_management:
            validation_result['warnings'].append('Risk management is not defined')

        return validation_result

    async def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Load strategy configuration for editing in UI

        Converts strategy object to UI-compatible JSON format

        Args:
            strategy_id: Strategy identifier

        Returns:
            Dictionary with strategy configuration or None if not found
        """
        # Try to load from StrategyManager first
        if strategy_id not in self.strategy_manager.loaded_strategies:
            # If not found, try to load from basic templates folder first
            basic_path = self.template_path / "basic" / f"{strategy_id}.py"
            main_path = self.template_path / f"{strategy_id}.py"

            template_path = None
            if basic_path.exists():
                template_path = basic_path
                self.logger.info(f"📂 Loading basic template: {strategy_id}")
            elif main_path.exists():
                template_path = main_path
                self.logger.info(f"📂 Loading main template: {strategy_id}")

            if template_path:
                # Temporarily load this strategy (NEW: load_strategy returns tuple)
                try:
                    strategy_obj, executor = self.strategy_manager.load_strategy(str(template_path))
                except Exception as e:
                    self.logger.error(f"❌ Failed to load template: {e}")
                    return None
            else:
                self.logger.error(f"❌ Strategy not found: {strategy_id}")
                return None
        else:
            # Already loaded
            strategy_obj = self.strategy_manager.loaded_strategies[strategy_id]
        self.logger.info(f"📂 Loading strategy: {strategy_id}")

        try:
            # ════════════════════════════════════════════════════════════
            # BASIC INFORMATION
            # ════════════════════════════════════════════════════════════
            name = getattr(strategy_obj, 'strategy_name', strategy_id)
            version = getattr(strategy_obj, 'strategy_version', '1.0.0')
            description = getattr(strategy_obj, 'description', '')
            author = getattr(strategy_obj, 'author', 'Unknown')
            created_date = getattr(strategy_obj, 'created_date', None)
            display_info = getattr(strategy_obj, 'display_info', 30)
            warmup_period = getattr(strategy_obj, 'warmup_period', 100)

            # ════════════════════════════════════════════════════════════
            # TRADING CONFIGURATION
            # ════════════════════════════════════════════════════════════
            # Get side_method - prefer root level over symbol level
            side_method = 'long_short'  # default

            # Check root level first (preferred)
            if hasattr(strategy_obj, 'side_method'):
                side_method_val = getattr(strategy_obj, 'side_method', None)
                # Handle tuple case (trailing comma bug)
                if isinstance(side_method_val, tuple) and len(side_method_val) > 0:
                    side_method_val = side_method_val[0]
                if side_method_val:
                    side_method = str(side_method_val).replace('TradingSide.', '').lower()
            # Fallback to symbol level only if root level not set or is default
            elif hasattr(strategy_obj, 'symbols') and strategy_obj.symbols:
                first_symbol = strategy_obj.symbols[0]
                if hasattr(first_symbol, 'side_method'):
                    side_method_val = getattr(first_symbol, 'side_method', None)
                    if side_method_val:
                        side_method = str(side_method_val).replace('TradingSide.', '').lower()

            leverage = getattr(strategy_obj, 'leverage', 1)
            # Handle tuple case (trailing comma bug)
            if isinstance(leverage, tuple) and len(leverage) > 0:
                leverage = leverage[0]

            max_position_size = getattr(strategy_obj, 'max_position_size', 1000.0)
            # Handle tuple case (trailing comma bug)
            if isinstance(max_position_size, tuple) and len(max_position_size) > 0:
                max_position_size = max_position_size[0]

            # Margin Configuration
            set_default_leverage = getattr(strategy_obj, 'set_default_leverage', False)
            hedge_mode = getattr(strategy_obj, 'hedge_mode', False)
            set_margin_type = getattr(strategy_obj, 'set_margin_type', False)
            margin_type = getattr(strategy_obj, 'margin_type', 'isolated')

            # ════════════════════════════════════════════════════════════
            # TIMEFRAME CONFIGURATION
            # ════════════════════════════════════════════════════════════
            # Check root level first - support both 'timeframes' and 'mtf_timeframes'
            timeframes = getattr(strategy_obj, 'timeframes', None) or getattr(strategy_obj, 'mtf_timeframes', None)
            primary_timeframe = getattr(strategy_obj, 'primary_timeframe', None)

            # Handle tuple case for root level (trailing comma bug)
            if timeframes and isinstance(timeframes, tuple) and len(timeframes) > 0:
                timeframes = list(timeframes[0]) if isinstance(timeframes[0], list) else [timeframes[0]]
            if primary_timeframe and isinstance(primary_timeframe, tuple) and len(primary_timeframe) > 0:
                primary_timeframe = primary_timeframe[0]

            # Fallback to technical_parameters only if root level not set
            if not timeframes or not primary_timeframe:
                if hasattr(strategy_obj, 'technical_parameters'):
                    tp = strategy_obj.technical_parameters
                    if not timeframes:
                        # Try both 'timeframes' and 'mtf_timeframes'
                        tf_val = getattr(tp, 'timeframes', None) or getattr(tp, 'mtf_timeframes', None)
                        if tf_val:
                            # Handle tuple case
                            if isinstance(tf_val, tuple) and len(tf_val) > 0:
                                tf_val = list(tf_val[0]) if isinstance(tf_val[0], list) else [tf_val[0]]
                            timeframes = tf_val
                    if not primary_timeframe and hasattr(tp, 'primary_timeframe'):
                        ptf_val = getattr(tp, 'primary_timeframe', None)
                        if ptf_val:
                            # Handle tuple case
                            if isinstance(ptf_val, tuple) and len(ptf_val) > 0:
                                ptf_val = ptf_val[0]
                            primary_timeframe = ptf_val

            # Set defaults if still not found
            if not timeframes:
                timeframes = ['15m']
            if not primary_timeframe:
                primary_timeframe = '15m'

            # ════════════════════════════════════════════════════════════
            # BACKTEST CONFIGURATION
            # ════════════════════════════════════════════════════════════
            backtesting_enabled = getattr(strategy_obj, 'backtesting_enabled', True)
            backtest_start_date = getattr(strategy_obj, 'backtest_start_date', '2025-01-01 00:00')
            backtest_end_date = getattr(strategy_obj, 'backtest_end_date', '2025-12-31 00:00')
            initial_balance = getattr(strategy_obj, 'inital_balance', 10000)  # Note: typo in original
            download_klines = getattr(strategy_obj, 'download_klines', False)
            update_klines = getattr(strategy_obj, 'update_klines', False)

            # Backtest parameters
            backtest_params = getattr(strategy_obj, 'backtest_parameters', {})
            commission = backtest_params.get('commission', 0.075)
            min_spread = backtest_params.get('min_spread', 0.01)
            max_slippage = backtest_params.get('max_slippage', 0.02)

            # ════════════════════════════════════════════════════════════
            # SYMBOLS
            # ════════════════════════════════════════════════════════════
            symbols = []
            # Check root level symbol_source first
            symbol_source = getattr(strategy_obj, 'symbol_source', 'strategy')

            if hasattr(strategy_obj, 'symbols'):
                for s in strategy_obj.symbols:
                    if hasattr(s, 'symbol'):
                        symbol_val = s.symbol
                        if isinstance(symbol_val, list):
                            symbols.extend(symbol_val)
                        else:
                            symbols.append(symbol_val)

                        # Fallback: Get source from symbol object (deprecated)
                        if hasattr(s, 'source') and symbol_source == 'strategy':
                            symbol_source = getattr(s, 'source', 'strategy')
                    else:
                        symbols.append(str(s))

            # ════════════════════════════════════════════════════════════
            # RISK MANAGEMENT
            # ════════════════════════════════════════════════════════════
            risk_mgmt = {}
            if hasattr(strategy_obj, 'risk_management'):
                rm = strategy_obj.risk_management
                risk_mgmt = {
                    # Position Sizing
                    'position_size_type': str(getattr(rm, 'sizing_method', 'FIXED')).replace('PositionSizeMethod.', '').lower(),
                    'position_size_value': getattr(rm, 'size_value', 100.0),

                    # Risk Limits
                    'max_loss_per_trade': getattr(rm, 'max_risk_per_trade', 1.0),
                    'max_drawdown': getattr(rm, 'max_drawdown', 10.0),
                    'max_daily_trades': getattr(rm, 'max_daily_trades', 20),

                    # Correlation & Position Management
                    'max_correlation': getattr(rm, 'max_correlation', 0.6),
                    'position_correlation_limit': getattr(rm, 'position_correlation_limit', 0.7),
                    'max_open_positions': getattr(strategy_obj.position_management, 'max_total_positions', 3) if hasattr(strategy_obj, 'position_management') else 3,

                    # Advanced Features
                    'emergency_stop_enabled': getattr(rm, 'emergency_stop_enabled', True),
                    'ai_risk_enabled': getattr(rm, 'ai_risk_enabled', False),
                    'dynamic_sizing_enabled': getattr(rm, 'dynamic_position_sizing', False)
                }

            # ════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT
            # ════════════════════════════════════════════════════════════
            pos_mgmt = {}
            if hasattr(strategy_obj, 'position_management'):
                pm = strategy_obj.position_management
                # Don't overwrite leverage - it was already extracted above
                # (keeping this for backward compatibility with old strategies)
                pos_mgmt_leverage = leverage
                if hasattr(strategy_obj, 'symbols') and strategy_obj.symbols:
                    first_symbol = strategy_obj.symbols[0]
                    symbol_leverage = getattr(first_symbol, 'leverage', None)
                    if symbol_leverage and symbol_leverage != 1:  # Only use if explicitly set
                        pos_mgmt_leverage = symbol_leverage

                pos_mgmt = {
                    # Basic settings
                    'entry_order_type': 'market',
                    'exit_order_type': 'market',
                    'leverage': pos_mgmt_leverage,

                    # Position limits
                    'max_positions_per_symbol': getattr(pm, 'max_positions_per_symbol', 1),
                    'max_total_positions': getattr(pm, 'max_total_positions', 3),
                    'position_timeout': getattr(pm, 'position_timeout', 1800),

                    # Pyramiding
                    'pyramiding_enabled': getattr(pm, 'pyramiding_enabled', False),
                    'max_pyramid_entries': getattr(pm, 'max_pyramid_entries', 3),
                    'pyramid_scale_factor': getattr(pm, 'pyramid_scale_factor', 0.5)
                }

            # ════════════════════════════════════════════════════════════
            # EXIT STRATEGY
            # ════════════════════════════════════════════════════════════
            exit_strat = {}
            if hasattr(strategy_obj, 'exit_strategy'):
                es = strategy_obj.exit_strategy

                # Get TP/SL values (support both old and new attribute names)
                stop_loss_value = getattr(es, 'stop_loss_value', None) or getattr(es, 'sl_percent', 1.0)
                take_profit_value = getattr(es, 'take_profit_value', None) or getattr(es, 'tp_percent', 2.0)
                trailing_stop = getattr(es, 'trailing_stop_enabled', False)

                # Convert method enums to strings
                tp_method = getattr(es, 'take_profit_method', None)
                tp_method_str = str(tp_method).replace('ExitMethod.', '').lower() if tp_method else 'fixed_percent'

                sl_method = getattr(es, 'stop_loss_method', None)
                sl_method_str = str(sl_method).replace('StopLossMethod.', '').lower() if sl_method else 'fixed_percent'

                # Extract partial exit levels and sizes
                partial_exit_levels = getattr(es, 'partial_exit_levels', [1.0, 1.5, 2.0])
                partial_exit_sizes = getattr(es, 'partial_exit_sizes', [0.3, 0.3, 0.4])

                exit_strat = {
                    # TP/SL Methods and Values
                    'take_profit_method': tp_method_str,
                    'take_profit_value': take_profit_value,
                    'stop_loss_method': sl_method_str,
                    'stop_loss_value': stop_loss_value,

                    # Trailing Stop Loss
                    'trailing_stop_enabled': trailing_stop,
                    'trailing_callback_percent': getattr(es, 'trailing_callback_percent', 0.4),
                    'trailing_activation_profit': getattr(es, 'trailing_activation_profit_percent', 1.0),

                    # Trailing Take Profit
                    'trailing_take_profit': getattr(es, 'trailing_take_profit', False),
                    'trailing_distance': getattr(es, 'trailing_distance', 0.2),

                    # Break-Even
                    'break_even_enabled': getattr(es, 'break_even_enabled', False),
                    'break_even_trigger_profit_percent': getattr(es, 'break_even_trigger_profit_percent', 1.0),
                    'break_even_offset': getattr(es, 'break_even_offset', 0.1),

                    # Partial Exit
                    'partial_exit_enabled': getattr(es, 'partial_exit_enabled', False),
                    'partial_exit_levels': partial_exit_levels,
                    'partial_exit_sizes': partial_exit_sizes
                }

                # Legacy support: also put TP/SL in position_management
                pos_mgmt['stop_loss_value'] = stop_loss_value
                pos_mgmt['take_profit_value'] = take_profit_value
                pos_mgmt['trailing_stop_loss'] = trailing_stop

            # ════════════════════════════════════════════════════════════
            # TECHNICAL INDICATORS
            # ════════════════════════════════════════════════════════════
            indicators = []
            if hasattr(strategy_obj, 'technical_parameters'):
                tp = strategy_obj.technical_parameters
                if hasattr(tp, 'indicators') and isinstance(tp.indicators, dict):
                    for ind_name, ind_params in tp.indicators.items():
                        # Parse indicator name and suffix
                        # Examples: "rsi_14" -> key="rsi", suffix="14"
                        #           "ema_9" -> key="ema", suffix="9"
                        parts = ind_name.split('_', 1)
                        key = parts[0]
                        suffix = parts[1] if len(parts) > 1 else ''

                        indicators.append({
                            'key': key,
                            'suffix': suffix,
                            'params': ind_params if isinstance(ind_params, dict) else {}
                        })

            # ════════════════════════════════════════════════════════════
            # ENTRY/EXIT CONDITIONS
            # ════════════════════════════════════════════════════════════
            entry_conditions = {'long': [], 'short': []}
            if hasattr(strategy_obj, 'entry_conditions') and isinstance(strategy_obj.entry_conditions, dict):
                entry_conditions = strategy_obj.entry_conditions
                self.logger.debug(f"📊 Entry conditions - LONG: {len(entry_conditions.get('long', []))} conditions")
                self.logger.debug(f"📊 Entry conditions - SHORT: {len(entry_conditions.get('short', []))} conditions")
                for i, cond in enumerate(entry_conditions.get('long', [])):
                    self.logger.debug(f"  LONG[{i}]: {cond}")
                for i, cond in enumerate(entry_conditions.get('short', [])):
                    self.logger.debug(f"  SHORT[{i}]: {cond}")

            exit_conditions = {'long': [], 'short': [], 'stop_loss': [], 'take_profit': []}
            if hasattr(strategy_obj, 'exit_conditions') and isinstance(strategy_obj.exit_conditions, dict):
                exit_conditions = strategy_obj.exit_conditions

            # ════════════════════════════════════════════════════════════
            # CUSTOM PARAMETERS / FILTERS
            # ════════════════════════════════════════════════════════════
            custom_params = {}
            if hasattr(strategy_obj, 'custom_parameters') and isinstance(strategy_obj.custom_parameters, dict):
                cp = strategy_obj.custom_parameters

                # News filter (boolean field)
                custom_params['news_filter'] = cp.get('news_filter', False)

                # Session filter
                session_filter = cp.get('session_filter', {})
                custom_params['session_filter'] = {
                    'enabled': session_filter.get('enabled', False),
                    'sydney': session_filter.get('sydney', True),
                    'tokyo': session_filter.get('tokyo', True),
                    'london': session_filter.get('london', True),
                    'new_york': session_filter.get('new_york', True),
                    'london_ny_overlap': session_filter.get('london_ny_overlap', True)
                }

                # Time filter
                time_filter = cp.get('time_filter', {})
                custom_params['time_filter'] = {
                    'enabled': time_filter.get('enabled', False),
                    'start_hour': time_filter.get('start_hour', 8),
                    'end_hour': time_filter.get('end_hour', 21),
                    'exclude_hours': time_filter.get('exclude_hours', [])
                }

                # Day filter
                day_filter = cp.get('day_filter', {})
                custom_params['day_filter'] = {
                    'enabled': day_filter.get('enabled', False),
                    'monday': day_filter.get('monday', True),
                    'tuesday': day_filter.get('tuesday', True),
                    'wednesday': day_filter.get('wednesday', True),
                    'thursday': day_filter.get('thursday', True),
                    'friday': day_filter.get('friday', True),
                    'saturday': day_filter.get('saturday', True),
                    'sunday': day_filter.get('sunday', True)
                }

            # ════════════════════════════════════════════════════════════
            # BUILD RESULT
            # ════════════════════════════════════════════════════════════
            result = {
                'id': strategy_id,
                'name': name,
                'version': version,
                'description': description,
                'author': author,
                'created_date': created_date,
                'display_info': display_info,
                'warmup_period': warmup_period,
                'symbols': symbols,
                'symbol_source': symbol_source,
                'trading_side': side_method,
                'leverage': leverage,
                'max_position_size': max_position_size,
                'set_default_leverage': set_default_leverage,
                'hedge_mode': hedge_mode,
                'set_margin_type': set_margin_type,
                'margin_type': margin_type,
                'primary_timeframe': primary_timeframe,
                'timeframes': timeframes,
                'backtesting_enabled': backtesting_enabled,
                'backtest_start_date': backtest_start_date,
                'backtest_end_date': backtest_end_date,
                'initial_balance': initial_balance,
                'download_klines': download_klines,
                'update_klines': update_klines,
                'commission': commission,
                'min_spread': min_spread,
                'max_slippage': max_slippage,
                'risk_management': risk_mgmt,
                'position_management': pos_mgmt,
                'exit_strategy': exit_strat,
                'indicators': indicators,
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions,
                'custom_parameters': custom_params
            }

            self.logger.info(f"✅ Strategy loaded: {strategy_id}")
            self.logger.debug(f"📊 Loaded indicators: {len(indicators)}")
            self.logger.debug(f"📊 Symbols: {symbols}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error loading strategy: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    async def _reload_strategies(self):
        """Reload all strategies from template directory"""
        # This would trigger strategy manager to reload
        # Implementation depends on StrategyManager's reload mechanism
        if hasattr(self.strategy_manager, 'reload_strategies'):
            await self.strategy_manager.reload_strategies()
