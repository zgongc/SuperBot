#!/usr/bin/env python3
"""
components/strategies/templates/simple_rsidivergence.py
SuperBot - RSI Divergence Reversal Strategy
Author: SuperBot Team
Date: 2026-01-29
Version: 1.0.0

Strategy Logic:
    RSI Divergence signals trend exhaustion and potential reversal.
    Waits for divergence + price confirmation (EMA crossover) for entry.

    LONG:  Bullish divergence (price ↓, RSI ↑) + price crosses above EMA50
           → Downtrend exhausted, reversal up confirmed

    SHORT: Bearish divergence (price ↑, RSI ↓) + price crosses below EMA50
           → Uptrend exhausted, reversal down confirmed

Expected Performance:
- Low trade frequency (divergence is rare)
- High win rate (strong reversal signals)
- Quick profits (reversals are fast)
- Requires tight trailing stop (momentum fades quickly)
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
    RSI Divergence + Confirmation Reversal Strategy

    Logic:
    1. Wait for RSI divergence (price and RSI move in opposite directions)
    2. Confirm with price action (EMA crossover)
    3. Filter with major trend (EMA200) and strength (ADX)
    4. Quick exit with trailing stop (reversals are short-lived)

    LONG:  Bullish divergence + price breaks above EMA50 (in uptrend context)
    SHORT: Bearish divergence + price breaks below EMA50 (in downtrend context)
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "simple_rsidivergence"
        self.strategy_version = "1.0.0"
        self.description = "RSI Divergence reversal with price confirmation"
        self.author = "SuperBot Team"
        self.created_date = "2026-01-29"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2026-01-01T00:00"
        self.backtest_end_date = "2026-01-30T00:00"
        self.initial_balance = 1000
        self.warmup_period = 200  # Sufficient for EMA 200
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
        self.side_method = TradingSide.BOTH  # LONG, SHORT, BOTH, FLAT
        self.leverage = 1  # 1x leverage (conservative for reversal trades)

        # Timeframe
        self.primary_timeframe = "1m"
        self.mtf_timeframes = ['15m']  # Primary timeframe MUST be in list

        # Exchange config
        self.set_default_leverage = False
        self.hedge_mode = False
        self.set_margin_type = False
        self.margin_type = "isolated"

        # ====================================================================
        # SYMBOLS
        # ====================================================================
        self.symbol_source = "strategy"  # file, strategy, exchange
        self.symbols = [
            SymbolConfig(
                symbol=['BTC', 'ETH', 'SOL', 'AVAX', 'LINK'],
                quote="USDT",
                enabled=True
            )
        ]

        # ====================================================================
        # RISK MANAGEMENT (Conservative for reversal trades)
        # ====================================================================
        self.risk_management = RiskManagement(
            sizing_method=PositionSizeMethod.FIXED_PERCENT,
            position_percent_size=8.0,          # 8% (conservative - reversals are risky)
            position_usd_size=300.0,            # For FIXED_USD (currently not used)
            position_quantity_size=2.0,         # FIXED_QUANTITY (currently not used)
            max_risk_per_trade=2.0,             # For RISK_BASED (currently not used)

            max_correlation=0.5,                # Strict (reversals sensitive to correlation)
            position_correlation_limit=0.6,
            max_drawdown=100,
            max_daily_trades=50,                # Low frequency expected
            emergency_stop_enabled=True,
            ai_risk_enabled=False,
            dynamic_position_sizing=True,
        )

        # ====================================================================
        # POSITION MANAGEMENT (Low frequency trades)
        # ====================================================================
        self.position_management = PositionManagement(
            max_positions_per_symbol=1,         # One position per symbol
            max_total_positions=5,              # Low total (divergence is rare)
            allow_hedging=False,                # No hedging
            position_timeout_enabled=True,
            position_timeout=1800,              # 30 hours timeout
            pyramiding_enabled=False,           # NO pyramiding (reversal trades)
            pyramiding_max_entries=1,
            pyramiding_scale_factor=0.5
        )

        # ====================================================================
        # EXIT STRATEGY (Aggressive - reversals are short-lived)
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=4.00,           # 4% TP (shorter than trend continuation)
            take_profit_price=110000.0,         # FIXED_PRICE (not used)
            take_profit_risk_reward_ratio=2.0,  # RISK_REWARD: 1:2
            take_profit_atr_multiplier=4.0,     # ATR_BASED (not used)
            take_profit_fib_level=1.618,        # FIBONACCI (not used)
            take_profit_ai_level=1,             # DYNAMIC_AI (not used)

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=2.00,             # 2% SL (tighter for quick exit on failure)
            stop_loss_price=95000.0,            # FIXED_PRICE (not used)
            stop_loss_atr_multiplier=2.0,       # ATR_BASED (not used)
            stop_loss_swing_lookback=10,        # SWING_POINTS (not used)
            stop_loss_fib_level=0.382,          # FIBONACCI (not used)
            stop_loss_ai_level=1,               # DYNAMIC_AI (not used)

            # Trailing Stop (CRITICAL for reversal trades!)
            trailing_stop_enabled=True,
            trailing_activation_profit_percent=0.8,   # Activate early (0.8%)
            trailing_callback_percent=0.4,            # Tight callback (0.4%)
            trailing_take_profit=False,
            trailing_distance=0.2,                    # Tight distance

            # Break-even (Early protection)
            break_even_enabled=True,
            break_even_trigger_profit_percent=0.8,    # Activate at 0.8% (early)
            break_even_offset=0.1,                    # Small offset

            # Partial Exit (Lock profits quickly)
            partial_exit_enabled=True,
            partial_exit_levels=[1.5, 2.5, 4.0],      # Graduated exits
            partial_exit_sizes=[0.40, 0.30, 0.30],    # 40% → 30% → 30%
        )

        # ====================================================================
        # INDICATORS
        # ====================================================================
        self.technical_parameters = TechnicalParameters(
            indicators={
                "rsidivergence": {
                    "rsi_period": 14,
                    "lookback": 5,
                    "min_strength": 30
                },
                "rsi_14": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "ema_50": {"period": 50},
                "ema_200": {"period": 200},
                "adx_14": {
                    "period": 14,
                    "adx_threshold": 25
                },
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS (Divergence + Confirmation)
        # ====================================================================
        self.entry_conditions = {
            'long': [
                # Divergence detected
                ['rsidivergence_bullish_divergence', '==', True],

                # RSI still in oversold area (confirming weakness before reversal)
                ['rsidivergence_rsi', '<', 40],

                # Price confirms reversal (breaks above EMA50)
                #['close', 'crossover', 'ema_50'],

                # Major trend is up (safer context)
                #['ema_50', '>', 'ema_200'],

                # Trend strength (not ranging)
                ['adx_14_adx', '>', 25],
            ],
            'short': [
                # Divergence detected
                ['rsidivergence_bearish_divergence', '==', True],

                # RSI still in overbought area
                ['rsidivergence_rsi', '>', 60],

                # Price confirms reversal (breaks below EMA50)
                #['close', 'crossunder', 'ema_50'],

                # Major trend is down (safer context)
                #['ema_50', '<', 'ema_200'],

                # Trend strength (not ranging)
                ['adx_14_adx', '>', 25],
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS (Let trailing stop handle exits)
        # ====================================================================
        self.exit_conditions = {
            'long': [
                # Optional: Exit on opposite divergence
                # ['rsidivergence_bearish_divergence', '==', True],
            ],
            'short': [
                # Optional: Exit on opposite divergence
                # ['rsidivergence_bullish_divergence', '==', True],
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
                "sydney": False,
                "tokyo": False,
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
                "monday": True,
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
        # Multi-stage optimization strategy for RSI Divergence
        #
        # Stage 1: Risk Management (50-100 trials)
        # Stage 2: Exit Strategy (100-150 trials) - CRITICAL for reversals
        # Stage 3: Indicators (100-200 trials)
        # Stage 4: Fine-tune (50 trials)
        # ====================================================================

        self.optimizer_parameters = {
            # ================================================================
            # STAGE 0: Main Strategy Parameters
            # ================================================================
            'main_strategy': {
                'enabled': False,
                #'side_method': ['BOTH', 'LONG', 'SHORT'],
                #'leverage': (1, 5, 1),  # Conservative leverage for reversals
            },

            # ================================================================
            # STAGE 1: Risk Management
            # ================================================================
            'risk_management': {
                'enabled': False,
                #'sizing_method': ['FIXED_PERCENT', 'RISK_BASED'],
                #'position_percent_size': (5.0, 15.0, 2.5),    # For FIXED_PERCENT
                #'max_risk_per_trade': (1.0, 3.0, 0.5),        # For RISK_BASED
            },

            # ================================================================
            # STAGE 2: Exit Strategy (MOST IMPORTANT for reversals!)
            # ================================================================
            'exit_strategy': {
                'enabled': False,

                # --- Stop Loss ---
                #'stop_loss_method': ['FIXED_PERCENT', 'ATR_BASED'],
                #'stop_loss_percent': (1.0, 3.0, 0.25),        # FIXED_PERCENT
                #'stop_loss_atr_multiplier': (1.5, 3.0, 0.5),  # ATR_BASED

                # --- Take Profit ---
                #'take_profit_method': ['FIXED_PERCENT', 'RISK_REWARD'],
                #'take_profit_percent': (2.0, 6.0, 0.5),               # FIXED_PERCENT
                #'take_profit_risk_reward_ratio': (1.5, 3.0, 0.5),     # RISK_REWARD

                # --- Break-Even (Critical for failed reversals) ---
                #'break_even_enabled': [True, False],
                #'break_even_trigger_profit_percent': (0.5, 1.5, 0.25),
                #'break_even_offset': (0.05, 0.3, 0.05),

                # --- Trailing Stop (MOST CRITICAL!) ---
                #'trailing_stop_enabled': [True, False],
                #'trailing_activation_profit_percent': (0.5, 2.0, 0.25),
                #'trailing_callback_percent': (0.2, 0.8, 0.1),
                #'trailing_distance': (0.1, 0.4, 0.05),

                # --- Partial Exit ---
                #'partial_exit_enabled': [True, False],
            },

            # ================================================================
            # STAGE 3: Indicators
            # ================================================================
            'indicators': {
                'enabled': False,

                # RSI Divergence
                #'rsidivergence': {
                #    'rsi_period': (10, 21, 7),
                #    'lookback': (3, 7, 2),
                #    'min_strength': (20, 50, 10),
                #},

                # RSI
                #'rsi_14': {
                #    'period': (10, 21, 7),
                #    'overbought': (65, 75, 5),
                #    'oversold': (25, 35, 5),
                #},

                # EMAs
                #'ema_50': {'period': (30, 70, 10)},
                #'ema_200': {'period': (150, 250, 25)},

                # ADX
                #'adx_14': {
                #    'period': (10, 21, 7),
                #    'adx_threshold': (20, 30, 5),
                #},
            },

            # ================================================================
            # STAGE 4: Entry Conditions
            # ================================================================
            'entry_conditions': {
                'enabled': False,
                #'rsi_threshold_long': (30, 50, 5),   # Max RSI for long entry
                #'rsi_threshold_short': (50, 70, 5),  # Min RSI for short entry
            },

            # ================================================================
            # STAGE 5: Position Management
            # ================================================================
            'position_management': {
                'enabled': False,
                #'max_positions_per_symbol': (1, 2, 1),
                #'max_total_positions': (3, 10, 1),
                #'position_timeout_enabled': [True, False],
                #'position_timeout': (120, 1440, 120),  # in minutes
            },

            # ================================================================
            # STAGE 6: Market Filters
            # ================================================================
            'market_filters': {
                'enabled': False,
                #'session_filter_enabled': [True, False],
                #'time_filter_enabled': [True, False],
            },

            # ================================================================
            # CONSTRAINTS (Optimizer global settings)
            # ================================================================
            'constraints': {
                'max_combinations': 10000,
                'min_trades': 10,           # Lower minimum (divergence is rare)
                'min_sharpe': 0.5,
                'min_profit_factor': 1.2,
                'max_drawdown': 25.0,
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
    logger.info(f"🎯 {strategy.strategy_name} v{strategy.strategy_version}")
    logger.info(f"{'='*60}\n")

    logger.info(f"📋 Description: {strategy.description}")
    logger.info(f"👤 Author: {strategy.author}")
    logger.info(f"📅 Created: {strategy.created_date}\n")

    logger.info(f"⚙️  Configuration:")
    logger.info(f"   Leverage: {strategy.leverage}x")
    logger.info(f"   Position Size: {strategy.risk_management.position_percent_size}%")
    logger.info(f"   Take Profit: {strategy.exit_strategy.take_profit_percent}%")
    logger.info(f"   Stop Loss: {strategy.exit_strategy.stop_loss_percent}%")
    rr_ratio = strategy.exit_strategy.take_profit_percent / strategy.exit_strategy.stop_loss_percent
    logger.info(f"   R/R Ratio: 1:{rr_ratio:.2f}\n")

    logger.info(f"📊 Indicators ({len(strategy.technical_parameters.indicators)}):")
    for name in strategy.technical_parameters.indicators.keys():
        logger.info(f"   ✓ {name}")

    logger.info(f"\n📈 Entry Logic:")
    logger.info(f"   LONG: Bullish divergence + price breaks above EMA50")
    logger.info(f"   SHORT: Bearish divergence + price breaks below EMA50")
    logger.info(f"   Exit: Trailing stop (reversals fade quickly)")

    logger.info(f"\n✅ Strategy loaded successfully!")
