#!/usr/bin/env python3
"""
components/strategies/templates/turtleV1.py
SuperBot - Turtle Trading V1 (Donchian Channel Breakout)
Yazar: SuperBot Team
Tarih: 2026-02-17
Versiyon: 1.0.0

Pine Script Kaynak:
    Donchian Channel Strategy by pratyush_trades (TradingView)
    - 20-period Donchian Channel breakout
    - EMA 200 trend filter
    - Exit at opposite channel band

Turtle Trading Kuralları:
    Entry:
        - LONG:  close > EMA(200) AND high >= Donchian Upper (yeni 20-bar yüksek)
        - SHORT: close < EMA(200) AND low <= Donchian Lower (yeni 20-bar düşük)
    Exit:
        - LONG exit:  low <= Donchian Lower (yeni 20-bar düşük)
        - SHORT exit: high >= Donchian Upper (yeni 20-bar yüksek)

Expected Performance:
    - Trend-following system (catches big moves)
    - Moderate trade frequency (breakout-based)
    - Works best in trending markets
    - Whipsaw risk in ranging markets
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
    Turtle Trading V1 - Donchian Channel Breakout Strategy

    20-period Donchian Channel ile breakout tespit eder,
    EMA 200 ile trend yönünü filtreler.
    Pine Script'ten uyarlanmıştır.
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "turtleV1"
        self.strategy_version = "1.0.0"
        self.description = "Turtle Trading V1 - Donchian Channel Breakout with EMA 200 Filter"
        self.author = "SuperBot Team"
        self.created_date = "2026-02-17"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2026-02-01T00:00"
        self.backtest_end_date = "2026-02-28T00:00"
        self.initial_balance = 1000
        self.warmup_period = 220  # EMA 200 + buffer
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
        self.leverage = 1

        # Timeframe - Turtle Trading works best on higher timeframes
        self.primary_timeframe = "1h"
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
            max_positions_per_symbol=3,
            max_total_positions=400,
            allow_hedging=False,
            position_timeout_enabled=True,
            position_timeout=1800,
            pyramiding_enabled=True,
            pyramiding_max_entries=3,
            pyramiding_scale_factor=0.5
        )

        # ====================================================================
        # EXIT STRATEGY
        # ====================================================================
        # Pine Script'te exit: Long -> stop at Donchian Lower, Short -> stop at Donchian Upper
        # SuperBot'ta bu mantık exit_conditions ile sağlanır.
        # Aşağıdaki SL/TP güvenlik ağı olarak çalışır.
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=6.00,
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=2.0,
            take_profit_atr_multiplier=4.0,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=3.2,
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=2.0,
            stop_loss_swing_lookback=10,
            stop_loss_fib_level=0.382,
            stop_loss_ai_level=1,

            # Trailing Stop
            trailing_stop_enabled=True,
            trailing_activation_profit_percent=1.2,
            trailing_callback_percent=0.8,
            trailing_take_profit=False,
            trailing_distance=0.5,

            # Break-even
            break_even_enabled=True,
            break_even_trigger_profit_percent=1.4,
            break_even_offset=0.9,

            # Partial Exit
            partial_exit_enabled=True,
            partial_exit_levels=[3, 5, 7],
            partial_exit_sizes=[0.30, 0.40, 0.30],
        )

        # ====================================================================
        # INDICATORS
        # ====================================================================
        # Pine Script:
        #   length = 20
        #   hh = highest(high, length)   -> donchian_20 upper
        #   ll = lowest(low, length)     -> donchian_20 lower
        #   mid = (hh + ll) / 2          -> donchian_20 middle
        #   ema(close, 200)              -> ema_200
        self.technical_parameters = TechnicalParameters(
            indicators={
                "donchian_20": {"period": 18},
                "ema_200": {"period": 200},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS
        # ====================================================================
        # Pine Script:
        #   if (close > ema(close, 200))
        #       strategy.entry("Long", strategy.long, stop=hh)
        #   if (close < ema(close, 200))
        #       strategy.entry("Short", strategy.short, stop=ll)
        #
        # stop=hh -> buy stop order at upper band -> triggers when high >= upper
        # stop=ll -> sell stop order at lower band -> triggers when low <= lower
        #
        # high >= donchian_20_upper -> current bar makes new 20-period high (breakout)
        # low <= donchian_20_lower  -> current bar makes new 20-period low (breakdown)
        self.entry_conditions = {
            'long': [
                ['close', '>', 'ema_200'],
                ['high', '>=', 'donchian_20_upper'],
            ],
            'short': [
                ['close', '<', 'ema_200'],
                ['low', '<=', 'donchian_20_lower'],
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS
        # ====================================================================
        # Pine Script:
        #   if (strategy.position_size > 0)
        #       strategy.exit("Longs Exit", stop=ll)   -> exit long at lower band
        #   if (strategy.position_size < 0)
        #       strategy.exit("Shorts Exit", stop=hh)  -> exit short at upper band
        #
        # low <= donchian_20_lower  -> price hits lower band -> exit long
        # high >= donchian_20_upper -> price hits upper band -> exit short
        self.exit_conditions = {
            'long': [
                #['low', '<=', 'donchian_20_lower'],
                #['close', 'crossunder', 'donchian_20_middle'],
            ],
            'short': [
                #['high', '>=', 'donchian_20_upper'],
                #['close', 'crossover', 'donchian_20_middle'],
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

            'exit_strategy': {
                'enabled': False,

                'trailing_stop_enabled': [True, False],
                'trailing_activation_profit_percent': (1.0, 4.0, 0.5),
                'trailing_callback_percent': (0.2, 1.0, 0.1),
                'trailing_distance': (0.1, 0.5, 0.1),
            },

            'indicators': {
                'enabled': False,

                #'donchian_20': {
                #    'period': (10, 30, 5),
                #},

                #'ema_200': {
                #    'period': (100, 300, 50),
                #},
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
    logger.info(f"   Take Profit: {strategy.exit_strategy.c}%")
    logger.info(f"   Stop Loss: {strategy.exit_strategy.stop_loss_value}%")
    rr_ratio = strategy.exit_strategy.take_profit_value / strategy.exit_strategy.stop_loss_value
    logger.info(f"   R/R Ratio: 1:{rr_ratio:.2f}\n")

    logger.info(f"  Indicators ({len(strategy.technical_parameters.indicators)}):")
    for name in strategy.technical_parameters.indicators.keys():
        logger.info(f"   - {name}")

    logger.info(f"\n  Entry Logic (Turtle Trading V1):")
    logger.info(f"   LONG:  close > EMA(200) AND high >= Donchian Upper (20-bar breakout)")
    logger.info(f"   SHORT: close < EMA(200) AND low <= Donchian Lower (20-bar breakdown)")
    logger.info(f"   Exit:  Opposite Donchian band (channel trailing stop)")

    logger.info(f"\n  Strategy loaded successfully!")
