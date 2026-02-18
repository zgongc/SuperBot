#!/usr/bin/env python3
"""
components/strategies/templates/turtleV2.py
SuperBot - Turtle Trading V2 (Original Turtle Rules)
Yazar: SuperBot Team
Tarih: 2026-02-17
Versiyon: 1.0.0

Pine Script Kaynak:
    Donchian Channel Strategy by zgongc (TradingView) - Pine Script v6
    - 20-period Donchian Channel entry (breakout)
    - 10-period Donchian Channel exit (faster exit)
    - No EMA filter (pure breakout system)
    - Uses previous bar data [1] for channel calculation

V1 vs V2 Farklar:
    - V1: EMA 200 trend filtresi VAR, entry/exit aynı kanal (20)
    - V2: EMA filtresi YOK, entry kanal=20, exit kanal=10 (orijinal Turtle)

Orijinal Turtle Trading System 1:
    Entry:
        - LONG:  close crossover 20-bar highest high (breakout)
        - SHORT: close crossunder 20-bar lowest low (breakdown)
    Exit:
        - LONG exit:  close crossunder 10-bar lowest low (faster exit)
        - SHORT exit: close crossover 10-bar highest high (faster exit)

Pine Script Notation:
    hh = ta.highest(high[1], 20)           -> 20-bar upper (shifted)
    ll = ta.lowest(low[1], 20)             -> 20-bar lower (shifted)
    exitLong = ta.lowest(low[1], 10)       -> 10-bar lower (exit)
    exitShort = ta.highest(high[1], 10)    -> 10-bar upper (exit)

SuperBot Adaptation:
    Pine [1] shift (excludes current bar) cannot be directly replicated.
    Entry: high >= donchian_20_upper (current bar is new 20-bar high)
    Exit: low <= donchian_10_lower (current bar is new 10-bar low)
    These approximate the crossover behavior when bar makes a new extreme.

Expected Performance:
    - Trend-following system (catches big moves)
    - More trades than V1 (no EMA filter)
    - Faster exits via 10-period channel (protects profits better)
    - More whipsaw risk without trend filter
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
    StopLossMethod,
    AIConfig
)


class Strategy(BaseStrategy):
    """
    Turtle Trading V2 - Original Turtle Rules

    20-period Donchian entry + 10-period Donchian exit.
    EMA filtresi yok, saf breakout sistemi.
    Pine Script'ten uyarlanmıştır.
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "turtleV2"
        self.strategy_version = "1.0.0"
        self.description = "Turtle Trading V2 - Original Rules (20-entry / 10-exit Donchian)"
        self.author = "SuperBot Team"
        self.created_date = "2026-02-17"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2026-02-01T00:00"
        self.backtest_end_date = "2026-02-28T00:00"
        self.initial_balance = 1000
        self.warmup_period = 200  # Donchian 20 + buffer (no EMA)
        self.download_klines = False
        self.update_klines = False

        # ====================================================================
        # BACKTEST PARAMETERS (Realistic Binance Futures)
        # ====================================================================
        self.backtest_parameters = {
            "min_spread": 0.01,       # 0.01% spread
            "commission": 0.02,       # 0.02% Maker fee (Binance Futures)
            "max_slippage": 0.05      # 0.05% slippage
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH  # LONG, SHORT, BOTH
        self.leverage = 3

        # V1 en iyi sonuç 1h'te -> V2 de 1h ile başla
        self.primary_timeframe = "5m"
        self.mtf_timeframes = []

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
                symbol=['BTC', 'BNB', 'ETH', 'SOL', 'AVAX', 'LINK', 'BCH', 'ZEC', 'DOT', 'ADA'],
                quote="USDT",
                enabled=True
            )
        ]

        # ====================================================================
        # AI CONFIG - Signal Intelligence
        # ====================================================================
        self.ai_config = AIConfig(
            ai_enabled=False,
            model_path="models/model.pkl",
            model_type="simple_train",

            # Entry Decision
            entry_decision=True,
            confidence_threshold=0.55,

            # TP/SL Optimization
            tp_optimization=False,
            sl_optimization=False,

            # Position Sizing
            position_sizing=False,

            # Exit Model
            exit_model_enabled=False,
            exit_model_path="models/exit_model.pkl",
            use_exit_model_tp=True,
            use_exit_model_sl=True,
            use_exit_model_trailing=True,
            use_exit_model_break_even=True,
            exit_model_blend_ratio=0.8,
        )

        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management = RiskManagement(
            sizing_method=PositionSizeMethod.FIXED_PERCENT,
            position_percent_size=5.0,
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
            max_positions_per_symbol=3,
            max_total_positions=400,
            allow_hedging=False,
            position_timeout_enabled=True,
            position_timeout=1800,
            pyramiding_enabled=True,
            pyramiding_max_entries=3,
            pyramiding_scale_factor=2.0
        )

        # ====================================================================
        # EXIT STRATEGY
        # ====================================================================
        # V2: Pure exit_strategy (no Donchian exit conditions)
        # Optimized: TP=6%, SL=4%, no trailing, no break-even
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=6.00,
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=2.0,
            take_profit_atr_multiplier=4.0,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=4.0,
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=2.0,
            stop_loss_swing_lookback=10,
            stop_loss_fib_level=0.382,
            stop_loss_ai_level=1,

            # Trailing Stop
            trailing_stop_enabled=False,
            trailing_activation_profit_percent=0.8,
            trailing_callback_percent=0.9,
            trailing_take_profit=False,
            trailing_distance=0.8,

            # Break-even
            break_even_enabled=False,
            break_even_trigger_profit_percent=0.8,
            break_even_offset=0.9,

            # Partial Exit
            partial_exit_enabled=True,
            partial_exit_levels=[2.8, 4.8, 7],
            partial_exit_sizes=[0.35, 0.40, 0.25],
        )

        # ====================================================================
        # INDICATORS
        # ====================================================================
        # Pine Script V2:
        #   input_long = 20, input_short = 10
        #   hh = ta.highest(high[1], input_long)   -> donchian_20 (entry)
        #   ll = ta.lowest(low[1], input_long)      -> donchian_20 (entry)
        #   exitLong = ta.lowest(low[1], input_short)   -> donchian_10 (exit)
        #   exitShort = ta.highest(high[1], input_short) -> donchian_10 (exit)
        #
        # V2 uses TWO Donchian channels:
        #   donchian_20 = entry channel (20-period)
        #   donchian_10 = exit channel (10-period, narrower, faster exit)
        self.technical_parameters = TechnicalParameters(
            indicators={
                "donchian_20": {"period": 30},   # Entry channel (optimized: 20→30)
                "donchian_10": {"period": 10},   # Disabled: exit_strategy performs better
                "ema_50": {"period": 50},
                "cmf_20": {'period': 20, 'buy_threshold': 0.05, 'sell_threshold': -0.05},   # Chaikin Money Flow
                "alma": {'period': 20, 'offset': 0.85, 'sigma': 6.0, 'source': 'close'},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS
        # ====================================================================
        # Pine Script V2:
        #   longCondition = ta.crossover(close, hh)
        #   shortCondition = ta.crossunder(close, ll)
        #
        # Pine uses [1] shift so close CAN exceed the channel.
        # In SuperBot, donchian includes current bar, so:
        #   close > donchian_20_upper is impossible (high >= close, upper >= high)
        #
        # Approximation:
        #   high >= donchian_20_upper = current bar IS the new 20-period high
        #   low <= donchian_20_lower  = current bar IS the new 20-period low
        #   This captures the same breakout events.
        #
        # V2 vs V1: No EMA filter! Pure Donchian breakout.
        self.entry_conditions = {
            'long': [
                #['close', '>', 'ema_50'],
                ['high', '>=', 'donchian_20_upper'],
                ["cmf_20", ">", 0.20],  # Güçlü para girişi
            ],
            'short': [
                #['close', '<', 'ema_50'],
                ['low', '<=', 'donchian_20_lower'],
                #["cmf_20", "<", -0.20],  # Güçlü para çıkışı
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS
        # ====================================================================
        # Pine Script V2:
        #   if (ta.crossunder(close, exitLong))   -> close crosses below 10-bar low
        #       strategy.close("Long")
        #   if (ta.crossover(close, exitShort))    -> close crosses above 10-bar high
        #       strategy.close("Short")
        #
        # V2 KEY FEATURE: 10-period exit channel (faster exit than 20-period)
        #
        # Approximation:
        #   low <= donchian_10_lower  = new 10-bar low -> exit long
        #   high >= donchian_10_upper = new 10-bar high -> exit short
        #
        # NOTE: On 1m timeframe this will overtrade (V1'den biliyoruz).
        #       On 1h+ it should work reasonably well.
        #       Comment out exit conditions and rely on trailing SL if overtrading occurs.
        self.exit_conditions = {
            'long': [
                #['low', '<=', 'donchian_10_lower'],  # Disabled: exit_strategy performs better
            ],
            'short': [
                #['high', '>=', 'donchian_10_upper'],  # Disabled: exit_strategy performs better
            ],
            'stop_loss': [],
            'take_profit': []
        }

        # ====================================================================
        # CUSTOM PARAMETERS
        # ====================================================================
        self.custom_parameters = {
            "news_filter": False,

            "session_filter": {
                "enabled": False,
                "sydney": False,
                "tokyo": False,
                "london": True,
                "new_york": True,
                "london_ny_overlap": True,
            },

            "time_filter": {
                "enabled": False,
                "start_hour": 8,
                "end_hour": 21,
                "exclude_hours": [],
            },

            "day_filter": {
                "enabled": False,
                "monday": False,
                "tuesday": False,
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
            'main_strategy': {
                'enabled': False,
            },

            'risk_management': {
                'enabled': False,
            },

            # ================================================================
            # STAGE 1: Exit Strategy - En kritik optimizasyon
            # Trailing + partial exit parametreleri
            # ================================================================
            'exit_strategy': {
                'enabled': False,

                'take_profit_percent': (3.0, 10.0, 1.0),
                'stop_loss_percent': (1.5, 5.0, 0.5),
                'trailing_stop_enabled': [True, False],
                'trailing_activation_profit_percent': (0.8, 3.0, 0.4),
                'trailing_callback_percent': (0.3, 1.5, 0.2),
                'trailing_distance': (0.2, 1.0, 0.2),
                'break_even_enabled': [True, False],
                'break_even_trigger_profit_percent': (0.8, 2.0, 0.3),
            },

            # ================================================================
            # STAGE 2: Indicators - Donchian periyotları
            # Entry(20) ve Exit(10) kanal genişliklerini optimize et
            # ================================================================
            'indicators': {
                'enabled': False,

                'donchian_20': {
                    'period': (10, 30, 5),
                },

                'donchian_10': {
                    'period': (5, 20, 5),
                },
            },

            'entry_conditions': {
                'enabled': False,
            },

            'position_management': {
                'enabled': False,
            },

            'market_filters': {
                'enabled': False,
            },

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
    from core.logger_engine import get_logger

    logger = get_logger(__name__)

    strategy = Strategy()

    logger.info(f"\n{'='*60}")
    logger.info(f"  {strategy.strategy_name} v{strategy.strategy_version}")
    logger.info(f"{'='*60}\n")

    logger.info(f"  Description: {strategy.description}")
    logger.info(f"  Author: {strategy.author}")
    logger.info(f"  Created: {strategy.created_date}\n")

    logger.info(f"  Configuration:")
    logger.info(f"   Leverage: {strategy.leverage}x")
    logger.info(f"   Position Size: {strategy.risk_management.size_value}%")
    logger.info(f"   Take Profit: {strategy.exit_strategy.take_profit_price}%")
    logger.info(f"   Stop Loss: {strategy.exit_strategy.stop_loss_price}%")
    rr_ratio = strategy.exit_strategy.take_profit_price / strategy.exit_strategy.stop_loss_price
    logger.info(f"   R/R Ratio: 1:{rr_ratio:.2f}\n")

    logger.info(f"  Indicators ({len(strategy.technical_parameters.indicators)}):")
    for name in strategy.technical_parameters.indicators.keys():
        logger.info(f"   - {name}")

    logger.info(f"\n  Entry Logic (Turtle Trading V2 - Original Rules):")
    logger.info(f"   LONG:  high >= Donchian(20) Upper (new 20-bar high breakout)")
    logger.info(f"   SHORT: low <= Donchian(20) Lower (new 20-bar low breakdown)")
    logger.info(f"   Exit LONG:  low <= Donchian(10) Lower (new 10-bar low)")
    logger.info(f"   Exit SHORT: high >= Donchian(10) Upper (new 10-bar high)")

    logger.info(f"\n  Strategy loaded successfully!")
