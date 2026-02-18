#!/usr/bin/env python3
"""
components/strategies/templates/almaV1.py
SuperBot - ALMA SD Bands Strategy V1
Yazar: SuperBot Team
Tarih: 2026-02-18
Versiyon: 1.0.0

Pine Script Kaynak:
    "ALMA SD Bands | RakoQuant" by RakoQuant (TradingView) - Pine Script v6
    - ALMA (Arnaud Legoux Moving Average) basis line
    - Standard Deviation bands (smoothed with ALMA)
    - Deadband regime detection (bull/bear/neutral)

Strateji Mantığı:
    Entry:
        - LONG:  close > alma_sd_bands_upper (fiyat üst bandın üstünde kapattı)
        - SHORT: close < alma_sd_bands_lower (fiyat alt bandın altında kapattı)
    Exit:
        - TP/SL based exit strategy (no indicator exit conditions)

Pine Script Orijinal Parametreler (Daily/Swing Mode):
    - ALMA Period: 111
    - ALMA Offset: 0.85
    - ALMA Sigma: 6.0
    - Band Multiplier: 0.4σ
    - Vol Smooth: 75
    - Deadband: 0.35σ
    - Source: high

Intraday Mode:
    - ALMA Period: 21
    - Vol Smooth: 10
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
    ALMA SD Bands Strategy V1

    ALMA Standard Deviation Bands ile trend takip sistemi.
    Fiyat üst bandı aşarsa LONG, alt bandı kırarsa SHORT.
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "almaV1"
        self.strategy_version = "1.0.0"
        self.description = "ALMA SD Bands Strategy - Volatility Band Breakout"
        self.author = "SuperBot Team"
        self.created_date = "2026-02-18"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2026-02-01T00:00"
        self.backtest_end_date = "2026-02-28T00:00"
        self.initial_balance = 1000
        self.warmup_period = 250  # ALMA 111 + vol_smooth 75 + buffer
        self.download_klines = False
        self.update_klines = False

        # ====================================================================
        # BACKTEST PARAMETERS (Realistic Binance Futures)
        # ====================================================================
        self.backtest_parameters = {
            "min_spread": 0.01,
            "commission": 0.02,
            "max_slippage": 0.05
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH
        self.leverage = 1

        self.primary_timeframe = "1m"
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
        # AI CONFIG
        # ====================================================================
        self.ai_config = AIConfig(
            ai_enabled=False,
            model_path="models/model.pkl",
            model_type="simple_train",
            entry_decision=True,
            confidence_threshold=0.55,
            tp_optimization=False,
            sl_optimization=False,
            position_sizing=False,
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
            pyramiding_scale_factor=1.0
        )

        # ====================================================================
        # EXIT STRATEGY
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.RISK_REWARD,
            take_profit_percent=6.00,
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=1.0,
            take_profit_atr_multiplier=4.0,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=4.0,
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=2.0,
            stop_loss_swing_lookback=10,
            stop_loss_fib_level=0.382,
            stop_loss_dynamic_source_long="alma_sd_bands_lower",   # LONG SL = alt band
            stop_loss_dynamic_source_short="alma_sd_bands_upper",  # SHORT SL = üst band
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
            partial_exit_levels=[3.5, 5, 7],
            partial_exit_sizes=[0.30, 0.40, 0.30],
        )

        # ====================================================================
        # INDICATORS
        # ====================================================================
        # ALMA SD Bands (Pine Script Daily/Swing preset)
        #   basis = ta.alma(src, 111, 0.85, 6.0)
        #   vol   = ta.alma(ta.stdev(src, 111), 75, 0.85, 6.0)
        #   upper = basis + 0.4 * vol
        #   lower = basis - 0.4 * vol
        self.technical_parameters = TechnicalParameters(
            indicators={
                "alma_sd_bands": {
                    'period': 111,
                    'offset': 0.85,
                    'sigma': 6.0,
                    'mult': 0.4,
                    'source': 'high',
                    'vol_smooth': 75,
                    'deadband_mult': 0.35
                },
                "cmf_20": {'period': 20, 'buy_threshold': 0.05, 'sell_threshold': -0.05},
                "ema_20":{'period': 20},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS
        # ====================================================================
        # ALMA SD Bands:
        #   close > upper band → LONG (fiyat üst bandı aştı, bull breakout)
        #   close < lower band → SHORT (fiyat alt bandı kırdı, bear breakdown)
        #
        # CMF filtresi (turtleV2'den kanıtlanmış):
        #   Long'da CMF > 0.20 → güçlü para girişi
        self.entry_conditions = {
            'long': [
                ['close', '>', 'alma_sd_bands_upper'],
                #["cmf_20", ">", 0.20],
                ['close', '>', 'ema_20'],
            ],
            'short': [
                ['close', '<', 'alma_sd_bands_lower'],
                ['close', '<', 'ema_20'],
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS
        # ====================================================================
        # Pure exit_strategy (TP/SL based)
        self.exit_conditions = {
            'long': [],
            'short': [],
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

            # STAGE 1: Exit Strategy
            'exit_strategy': {
                'enabled': True,

                'take_profit_percent': (3.0, 10.0, 1.0),
                'stop_loss_percent': (1.5, 5.0, 0.5),
                'trailing_stop_enabled': [True, False],
                'trailing_activation_profit_percent': (0.8, 3.0, 0.4),
                'trailing_callback_percent': (0.3, 1.5, 0.2),
                'trailing_distance': (0.2, 1.0, 0.2),
                'break_even_enabled': [True, False],
                'break_even_trigger_profit_percent': (0.8, 2.0, 0.3),
            },

            # STAGE 2: ALMA SD Bands parameters
            'indicators': {
                'enabled': False,

                'alma_sd_bands': {
                    'period': (21, 150, 15),
                    'mult': (0.2, 1.0, 0.2),
                    'vol_smooth': (10, 100, 15),
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
    logger.info(f"  Indicators: {list(strategy.technical_parameters.indicators.keys())}")

    logger.info(f"\n  Entry Logic (ALMA SD Bands):")
    logger.info(f"   LONG:  close > alma_sd_bands_upper (price above upper band)")
    logger.info(f"   SHORT: close < alma_sd_bands_lower (price below lower band)")

    logger.info(f"\n  Strategy loaded successfully!")
