#!/usr/bin/env python3
"""
components/strategies/templates/supertrend_scalp.py
SuperBot - SuperTrend Scalp Strategy
Yazar: SuperBot Team
Tarih: 2025-12-08
Versiyon: 1.0.0

SuperTrend + EMA + RSI Scalping Strategy

Strategy Logic:
- SuperTrend: Determines the main trend direction.
- EMA Cross (9/21): Detects rapid momentum changes.
- RSI: Overbought/oversold filter.
- ADX: Trend strength filter.

Entry:
- EMA crossover in the same direction as SuperTrend
- RSI is not in overbought regions
- ADX trend strength is sufficient.

Exit:
- SuperTrend direction change
- EMA crossunder/crossover
- Trailing stop with profit locking.

Expected Performance:
- High-frequency trading (scalping)
- Fast exits with low TP/SL
- High win rate with trend following
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
    SuperTrend Scalp Strategy

    ATR based SuperTrend for scalping
    Goal: Fast entry/exit, trend tracking
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "SuperTrend Scalp Strategy"
        self.strategy_version = "1.0.0"
        self.description = "Fast scalping strategy with SuperTrend + EMA + RSI"
        self.author = "SuperBot Team"
        self.created_date = "2025-12-08"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2025-01-05T00:00"
        self.backtest_end_date = "2025-03-30T00:00"
        self.initial_balance = 1000
        self.warmup_period = 200
        self.download_klines = False
        self.update_klines = False

        # ====================================================================
        # BACKTEST PARAMETERS (Realistic Binance Futures)
        # ====================================================================
        self.backtest_parameters = {
            "min_spread": 0.01,       # 0.01% spread
            "commission": 0.02,       # 0.02% Maker fee (Binance Futures) - FIXED!
            "max_slippage": 0.05      # 0.05% slippage
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH
        self.leverage = 1

        # Timeframe - for 1m/5m scalping
        self.mtf_timeframes = []
        self.primary_timeframe = "5m"

        # Exchange config
        self.set_default_leverage = False
        self.hedge_mode = False
        self.set_margin_type = False
        self.margin_type = "isolated"

        # ====================================================================
        # SYMBOLS
        # ====================================================================
        self.symbol_source = "strategy"
        self.symbols = [
            SymbolConfig(
                symbol=['BTC', 'ETH', 'BNB', 'SOL', 'AVAX', 'LINK', 'BCH', 'ZEC', 'LTC'],
                quote="USDT",
                enabled=True
            )
        ]

        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management = RiskManagement(
            sizing_method=PositionSizeMethod.FIXED_PERCENT,
            position_percent_size=10.0,
            position_usd_size=300.0,
            position_quantity_size=2.0,
            max_risk_per_trade=2.5,

            max_correlation=0.6,
            position_correlation_limit=0.7,
            max_drawdown=100,
            max_daily_trades=300,
            emergency_stop_enabled=True,
            ai_risk_enabled=False,
            dynamic_position_sizing=True,
        )

        # ====================================================================
        # POSITION MANAGEMENT
        # ====================================================================
        self.position_management = PositionManagement(
            max_positions_per_symbol=1,
            max_total_positions=400,
            allow_hedging=False,
            position_timeout_enabled=True,
            position_timeout=1800,
            pyramiding_enabled=False,
            pyramiding_max_entries=3,
            pyramiding_scale_factor=0.5
        )

        # ====================================================================
        # EXIT STRATEGY (Scalp - Dar SL/TP)
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=3.0,  # Narrow take profit for scalping
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=2.0,
            take_profit_atr_multiplier=2.0,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=2.0,                # Narrow stop loss for scalping
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=1.5,
            stop_loss_swing_lookback=5,
            stop_loss_fib_level=0.382,
            stop_loss_ai_level=1,

            # Trailing Stop - precise for scalping
            trailing_stop_enabled=True,
            trailing_activation_profit_percent=1.0,
            trailing_callback_percent=0.4,
            trailing_take_profit=False,
            trailing_distance=0.2,

            # Break-even - erken koruma
            break_even_enabled=True,
            break_even_trigger_profit_percent=1.2,
            break_even_offset=0.1,

            # Partial Exit - for quick profit taking in scalping
            partial_exit_enabled=True,
            partial_exit_levels=[0.8, 1.5, 2.0],
            partial_exit_sizes=[0.40, 0.35, 0.25],
        )

        # ====================================================================
        # INDICATORS (SuperTrend + Momentum)
        # ====================================================================
        self.technical_parameters = TechnicalParameters(
            indicators={
                # SuperTrend - Ana trend belirleme
                # output: supertrend, upper, lower, trend (1=UP, -1=DOWN)
                "supertrend": {"period": 10, "multiplier": 3.0},

                # Fast EMAs - for scalping
                "ema_9": {"period": 9},
                "ema_21": {"period": 21},
                "ema_50": {"period": 50},

                # RSI - Momentum filtresi
                "rsi_14": {"period": 14},

                # ADX - Trend strength
                "adx": {"period": 14},

                # ATR - Volatilite
                #"atr_14": {"period": 14},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS (SuperTrend + EMA Cross)
        # ====================================================================
        self.entry_conditions = {
            'long': [
                # 1. SUPERTREND: Bullish trend (close > supertrend)
                ['close', '>', 'supertrend'],

                # 2. EMA CROSS: Fast EMA crosses above slow EMA
                ['ema_9', 'crossover', 'ema_21'],

                # 3. TREND FILTER: Price above EMA 50
                ['close', '>', 'ema_50'],

                # 4. TREND STRENGTH: ADX > 20
                ['adx_adx', '>', 20],

                # 5. MOMENTUM: RSI not overbought
                ['rsi_14', 'between', [40,70]],
                #['rsi_14', '<', 70],
                #['rsi_14', '>', 40],
            ],
            'short': [
                # 1. SUPERTREND: Bearish trend (close < supertrend)
                ['close', '<', 'supertrend'],

                # 2. EMA CROSS: Fast EMA crosses below slow EMA
                ['ema_9', 'crossunder', 'ema_21'],

                # 3. TREND FILTER: Price below EMA 50
                ['close', '<', 'ema_50'],

                # 4. TREND STRENGTH: ADX > 20
                ['adx_adx', '>', 20],

                # 5. MOMENTUM: RSI not oversold
                ['rsi_14', 'between', [30,60]],
                #['rsi_14', '>', 30],
                #['rsi_14', '<', 60],
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS (SuperTrend Reversal)
        # ====================================================================
        self.exit_conditions = {
            'long': [
                # SuperTrend bearish flip
                ['close', '<', 'supertrend'],
                # EMA bearish cross
                ['ema_9', 'crossunder', 'ema_21'],
            ],
            'short': [
                # SuperTrend bullish flip
                ['close', '>', 'supertrend'],
                # EMA bullish cross
                ['ema_9', 'crossover', 'ema_21'],
            ],
            'stop_loss': [],
            'take_profit': []
        }

        # ====================================================================
        # CUSTOM PARAMETERS
        # ====================================================================
        self.custom_parameters = {
            # Market Filters
            "news_filter": False,

            # Session Filter
            "session_filter": {
                "enabled": False,
                "sydney": True,
                "tokyo": True,
                "london": True,
                "new_york": True,
                "london_ny_overlap": True,
            },

            # Time Filter
            "time_filter": {
                "enabled": False,
                "start_hour": 8,
                "end_hour": 21,
                "exclude_hours": [],
            },

            # Day Filter
            "day_filter": {
                "enabled": False,
                "monday": False,
                "tuesday": True,
                "wednesday": True,
                "thursday": True,
                "friday": True,
                "saturday": True,
                "sunday": True,
            },
        }

        # ====================================================================
        # OPTIMIZER PARAMETERS
        # ====================================================================
        self.optimizer_parameters = {
            # ================================================================
            # STAGE 0: Main Strategy Parameters
            # ================================================================
            'main_strategy': {
                'enabled': False,
                'side_method': ['BOTH', 'LONG', 'SHORT'],
                'leverage': (1, 20, 1),
            },

            # ================================================================
            # STAGE 1: Risk Management (PRIORITY: HIGHEST)
            # ================================================================
            'risk_management': {
                'enabled': False,
                'sizing_method': ['FIXED_PERCENT', 'RISK_BASED', 'FIXED_USD'],
                'position_percent_size': (5.0, 20.0, 1),
                #'position_usd_size': (100, 1000, 100),
                #'max_risk_per_trade': (1.0, 5.0, 0.5),
            },

            # ================================================================
            # STAGE 2: Exit Strategy (PRIORITY: HIGH)
            # ================================================================
            'exit_strategy': {
                'enabled': False,

                # Stop Loss
                #'stop_loss_method': ['PERCENTAGE', 'ATR_BASED', 'FIBONACCI'],
                'stop_loss_value': (0.5, 5, 0.1),  # Narrow SL range for scalping
                #'stop_loss_atr_multiplier': (1.0, 2.5, 0.25),
                #'stop_loss_fib_level': [0.236, 0.382, 0.5, 0.618],

                # Take Profit
                #'take_profit_method': ['PERCENTAGE', 'RISK_REWARD', 'ATR_BASED'],
                'take_profit_value': (1.0, 10.0, 0.2),  # Narrow TP range for scalping
                #'take_profit_risk_reward': (1.5, 3.0, 0.25),
                #'take_profit_atr_multiplier': (1.5, 3.0, 0.5),

                # Break-Even
                #'break_even_enabled': [True, False],
                #'break_even_trigger_profit_percent': (0.5, 1.5, 0.2),
                #'break_even_offset': (0.05, 0.20, 0.05),

                # Trailing Stop
                #'trailing_stop_enabled': [True, False],
                #'trailing_activation_profit_percent': (0.5, 2.0, 0.25),
                #'trailing_callback_percent': (0.2, 0.8, 0.1),

                #'trailing_take_profit': [True, False],

                # Partial Exit
                #'partial_exit_enabled': [True, False],
                #'partial_exit_level_1': (0.5, 1.5, 0.25),
                #'partial_exit_level_2': (1.0, 2.5, 0.5),
                #'partial_exit_level_3': (1.5, 3.0, 0.5),
                #'partial_exit_size_1': (0.20, 0.35, 0.05),
                #'partial_exit_size_2': (0.25, 0.40, 0.05),
                #'partial_exit_size_3': (0.30, 0.50, 0.05),
            },

            # ================================================================
            # STAGE 3: Indicators (PRIORITY: MEDIUM)
            # ================================================================
            'indicators': {
                'enabled': False,

                # SuperTrend
                'supertrend': {
                    'period': (7, 14, 1),
                    'multiplier': (2.0, 4.0, 0.5),
                },

                # RSI
                'rsi': {
                    'period': (7, 21, 2),
                    #'overbought': (65, 80, 5),
                    #'oversold': (20, 35, 5),
                },

                # EMA Fast
                #'ema_fast': {
                #    'period': (5, 15, 2),
                #},

                # EMA Slow
                #'ema_slow': {
                #    'period': (15, 30, 5),
                #},

                # ADX
                #'adx': {
                #    'period': (10, 20, 2),
                #    'threshold': (15, 30, 5),
                #},
            },

            # ================================================================
            # STAGE 4: Entry Conditions (PRIORITY: MEDIUM)
            # ================================================================
            'entry_conditions': {
                'enabled': False,
                #'volume_multiplier': (1.5, 3.0, 0.25),
                #'price_change_threshold': (0.5, 2.0, 0.25),
                #'require_all_conditions': [True, False],
                #'min_conditions_met': (1, 3, 1),
                #'confirmation_candles': (0, 3, 1),
            },

            # ================================================================
            # STAGE 5: Position Management (PRIORITY: LOW)
            # ================================================================
            'position_management': {
                'enabled': False,
                'max_total_positions': (1, 5, 1),
                #'max_positions_per_symbol': (1, 3, 1),
                #'allow_hedging': [True, False],
                #'pyramiding_enabled': [True, False],
                #'pyramiding_max_entries': (2, 5, 1),
                #'pyramiding_scale_factor': (0.5, 1.0, 0.1),
                #'position_timeout_enabled': [True, False],
                #'position_timeout_minutes': (60, 480, 60),
            },

            # ================================================================
            # STAGE 6: Market Filters (PRIORITY: LOW)
            # ================================================================
            'market_filters': {
                'enabled': False,
                'day_filter_enabled': [True, False],
                'monday_enabled': [True, False],
                'tuesday_enabled': [True, False],
                #'wednesday_enabled': [True, False],
                #'thursday_enabled': [True, False],
                #'friday_enabled': [True, False],
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


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    strategy = Strategy()

    print("\n" + "=" * 60)
    print(f"  {strategy.strategy_name} v{strategy.strategy_version}")
    print("=" * 60)

    # ---- METADATA ----
    print(f"\n  Description : {strategy.description}")
    print(f"  Author      : {strategy.author}")
    print(f"  Created     : {strategy.created_date}")

    # ---- TRADING CONFIG ----
    print(f"\n  TRADING CONFIG")
    print(f"  {'-' * 40}")
    print(f"  Timeframe   : {strategy.primary_timeframe} (MTF: {strategy.mtf_timeframes})")
    print(f"  Side        : {strategy.side_method.value}")
    print(f"  Leverage    : {strategy.leverage}x")

    # ---- SYMBOLS ----
    symbols = []
    for cfg in strategy.symbols:
        if cfg.enabled:
            symbols.extend([f"{s}{cfg.quote}" for s in cfg.symbol])
    print(f"  Symbols     : {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")

    # ---- RISK MANAGEMENT ----
    rm = strategy.risk_management
    print(f"\n  RISK MANAGEMENT")
    print(f"  {'-' * 40}")
    print(f"  Sizing      : {rm.sizing_method.value}")
    print(f"  Position %  : {rm.position_percent_size}%")
    print(f"  Max DD      : {rm.max_drawdown}%")
    print(f"  Daily Trades: {rm.max_daily_trades}")

    # ---- EXIT STRATEGY ----
    ex = strategy.exit_strategy
    rr = ex.take_profit_percent / ex.stop_loss_percent if ex.stop_loss_percent else 0
    print(f"\n  EXIT STRATEGY")
    print(f"  {'-' * 40}")
    print(f"  TP Method   : {ex.take_profit_method.value} ({ex.take_profit_percent}%)")
    print(f"  SL Method   : {ex.stop_loss_method.value} ({ex.stop_loss_percent}%)")
    print(f"  R/R Ratio   : 1:{rr:.2f}")
    print(f"  Trailing    : {'ON' if ex.trailing_stop_enabled else 'OFF'}")
    print(f"  Break-even  : {'ON' if ex.break_even_enabled else 'OFF'}")
    print(f"  Partial Exit: {'ON' if ex.partial_exit_enabled else 'OFF'}")

    # ---- INDICATORS ----
    indicators = list(strategy.technical_parameters.indicators.keys())
    print(f"\n  INDICATORS ({len(indicators)})")
    print(f"  {'-' * 40}")
    for ind in indicators:
        params = strategy.technical_parameters.indicators[ind]
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        print(f"  - {ind}: {param_str}")

    # ---- ENTRY CONDITIONS ----
    print(f"\n  ENTRY CONDITIONS")
    print(f"  {'-' * 40}")
    print(f"  LONG  ({len(strategy.entry_conditions.get('long', []))} conditions):")
    for cond in strategy.entry_conditions.get('long', []):
        print(f"    - {cond}")
    print(f"  SHORT ({len(strategy.entry_conditions.get('short', []))} conditions):")
    for cond in strategy.entry_conditions.get('short', []):
        print(f"    - {cond}")

    # ---- EXIT CONDITIONS ----
    print(f"\n  EXIT CONDITIONS")
    print(f"  {'-' * 40}")
    print(f"  LONG  : {len(strategy.exit_conditions.get('long', []))} conditions")
    print(f"  SHORT : {len(strategy.exit_conditions.get('short', []))} conditions")

    # ---- POSITION MANAGEMENT ----
    pm = strategy.position_management
    print(f"\n  POSITION MANAGEMENT")
    print(f"  {'-' * 40}")
    print(f"  Max/Symbol  : {pm.max_positions_per_symbol}")
    print(f"  Max Total   : {pm.max_total_positions}")
    print(f"  Timeout     : {pm.position_timeout} min")
    print(f"  Pyramiding  : {'ON' if pm.pyramiding_enabled else 'OFF'}")

    print("\n" + "=" * 60)
    print("  Strategy loaded successfully!")
    print("=" * 60 + "\n")
