#!/usr/bin/env python3
"""
components/strategies/templates/simple_pmax.py
SuperBot - PMax + EMA200 Trend Following Strategy
Author: SuperBot Team
Date: 2026-01-28
Version: 1.0.0

PMax + EMA200 Trend Following Strategy

Strategy Logic:
- Uses PMax indicator for dynamic support/resistance
- EMA200 as major trend filter
- Only trades in the direction of both PMax and EMA200

LONG Entry:
  - Close > EMA200 (major uptrend)
  - PMax trend_direction == 1 (uptrend confirmed)
  - Price respects dynamic support

SHORT Entry:
  - Close < EMA200 (major downtrend)
  - PMax trend_direction == -1 (downtrend confirmed)
  - Price respects dynamic resistance

Expected Performance:
- Medium frequency trades (PMax changes regularly)
- High win rate (dual trend confirmation)
- Strong trends only
"""

import sys
from pathlib import Path

# Add SuperBot base directory to path
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
    PMax + EMA200 Trend Following Strategy

    Logic: Ride strong trends with dual confirmation from PMax and EMA200.

    LONG:  Price > EMA200 + PMax uptrend (trend_direction == 1)
           → Both indicators confirm bullish momentum

    SHORT: Price < EMA200 + PMax downtrend (trend_direction == -1)
           → Both indicators confirm bearish momentum
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "simple_pmax"
        self.strategy_version = "1.0.0"
        self.description = "PMax + EMA200 trend following strategy"
        self.author = "SuperBot Team"
        self.created_date = "2026-01-28"

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
        # NOTE: Values are in PERCENTAGE format (not decimal)
        # Example: 0.02 = 0.02%, NOT 2%
        self.backtest_parameters = {
            "min_spread": 0.01,       # 0.01% spread
            "commission": 0.02,       # 0.02% Maker fee (Binance Futures) - FIXED!
            "max_slippage": 0.05      # 0.05% slippage
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH  # LONG, SHORT, BOTH, FLAT
        self.leverage = 1  # 5x leverage (recommended for 88% win rate)

        # Timeframe
        self.primary_timeframe = "15m"
        self.mtf_timeframes = []  # Primary timeframe MUST be in list

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
                symbol=['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'LINK', 'DOT', 'ADA'],
                quote="USDT",
                enabled=True
            )
        ]

        self.ai_config = AIConfig(
            # ═══════════════════════════════════════════════════════════
            # GENERAL
            # ═══════════════════════════════════════════════════════════
            ai_enabled=True,        # ✅ AI aktif
            model_path="models/model.pkl",     # XGBoost model dosyası
            model_type="simple_train",      # RL model (PPO)

            # ═══════════════════════════════════════════════════════════
            # ENTRY DECISION (Sinyal Filtreleme)
            # RL model strateji sinyalini onaylıyor mu?
            # Strateji: "LONG aç" → RL: "LONG %70" → Onay ✓
            # Strateji: "LONG aç" → RL: "SHORT %80" → Red ✗
            # ═══════════════════════════════════════════════════════════
            entry_decision=True,                          # AI sinyal filtrelemede kullanılsın
            confidence_threshold=0.55,                     # %50+ güven eşiği (RL için daha düşük)

            # ═══════════════════════════════════════════════════════════
            # TP/SL OPTIMIZATION (Opsiyonel - RL desteklemiyor)
            # ═══════════════════════════════════════════════════════════
            tp_optimization=False,                        # RL model TP önermiyor
            sl_optimization=False,                        # RL model SL önermiyor

            # ═══════════════════════════════════════════════════════════
            # POSITION SIZING (Opsiyonel)
            # ═══════════════════════════════════════════════════════════
            position_sizing=False,                        # AI pozisyon boyutu ayarlamasın
            
            # ════════════════════════════════════════════════════════════
            # EXIT MODEL (Dynamic Exit Optimization) - TEST
            # ═══════════════════════════════════════════════════════════
            exit_model_enabled=False,                      # ✅ Exit Model aktif (TEST)
            exit_model_path="models/exit_model.pkl",
            use_exit_model_tp=True,                       # TP'yi Exit Model optimize etsin
            use_exit_model_sl=True,                       # SL'yi Exit Model optimize etsin
            use_exit_model_trailing=True,                 # Trailing kararını Exit Model alsın
            use_exit_model_break_even=True,               # BE kararını Exit Model alsın
            exit_model_blend_ratio=0.8,                   # %80 Exit Model, %20 Strategy
        )

        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management = RiskManagement(
            sizing_method=PositionSizeMethod.FIXED_PERCENT,
            position_percent_size=10.0,         # 10% of portfolio × leverage
            position_usd_size=300.0,            # For FIXED_USD (not used)
            position_quantity_size=2.0,         # For FIXED_QUANTITY (not used)
            max_risk_per_trade=2.5,             # For RISK_BASED (not used)

            max_correlation=0.6,
            position_correlation_limit=0.7,
            max_drawdown=100,
            max_daily_trades=800,
            emergency_stop_enabled=True,
            ai_risk_enabled=False,
            dynamic_position_sizing=True,
        )

        # ====================================================================
        # POSITION MANAGEMENT
        # ====================================================================
        self.position_management = PositionManagement(
            max_positions_per_symbol=1,         # Max 1 position per symbol
            max_total_positions=400,            # Max total positions
            allow_hedging=False,                # No opposite positions
            position_timeout_enabled=True,      # Enable timeout check
            position_timeout=1800,              # 1800 min = 30 hours
            pyramiding_enabled=False,           # No pyramiding
            pyramiding_max_entries=3,
            pyramiding_scale_factor=0.5
        )

        # ====================================================================
        # EXIT STRATEGY
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=8.00,                       # 8% TP
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=2.0,
            take_profit_atr_multiplier=4.0,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=4.00,                         # 4% SL
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=2.0,
            stop_loss_swing_lookback=10,
            stop_loss_fib_level=0.382,
            stop_loss_ai_level=1,

            # Trailing Stop - CONSERVATIVE (activate late, wide callback)
            trailing_stop_enabled=True,
            trailing_activation_profit_percent=4.0,         # Activate at 4% profit
            trailing_callback_percent=1.5,                  # Wide 1.5% callback
            trailing_take_profit=False,
            trailing_distance=0.5,

            # Break-even - ENABLED for profit protection
            break_even_enabled=True,
            break_even_trigger_profit_percent=1.2,          # Activate at 2% profit
            break_even_offset=0.2,                          # 0.2% above entry

            # Partial Exit - ENABLED for gradual profit taking
            partial_exit_enabled=True,
            partial_exit_levels=[3, 5, 7],                  # 3%, 5%, 7%
            partial_exit_sizes=[0.30, 0.40, 0.30],          # 30%, 40%, 30%
        )

        # ====================================================================
        # INDICATORS
        # ====================================================================
        self.technical_parameters = TechnicalParameters(
            indicators={
                "ema_200": {"period": 200},                 # Major trend filter
                "pmax": {                                    # PMax indicator
                    "atr_period": 10,
                    "atr_multiplier": 3.0,  # TradingView default
                    "ma_period": 10
                },
                "adx_14": {"period": 14, "adx_threshold": 25},  # Trend strength filter
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS
        # ====================================================================
        self.entry_conditions = {
            'long': [
                ['close', '>', 'ema_200'],                  # Price above EMA200
                ['adx_14_adx', '>', 20],                    # Strong trend (not ranging)
                ['pmax_mavg', 'crossover', 'pmax_pmax'],    # EMA crosses above PMax
                #['pmax_trend_direction', '==', 1]
            ],
            'short': [
                ['close', '<', 'ema_200'],                  # Price below EMA200
                ['adx_14_adx', '>', 20],                    # Strong trend (not ranging)
                ['pmax_mavg', 'crossunder', 'pmax_pmax'],   # EMA crosses below PMax
                #['pmax_trend_direction', '==', -1]
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS
        # ====================================================================
        # NOTE: Exit conditions disabled - let BE/PE/Trailing do the work
        # Exit condition was too aggressive, closing on every minor reversal
        self.exit_conditions = {
            'long': [
                # DISABLED: Too aggressive
                # ['pmax_mavg', 'crossunder', 'pmax_pmax'],

                # Alternative: Only exit when trend weakens significantly
                # ['adx_14_adx', '<', 20],  # Exit when ADX drops below 20
            ],
            'short': [
                # DISABLED: Too aggressive
                # ['pmax_mavg', 'crossover', 'pmax_pmax'],

                # Alternative: Only exit when trend weakens significantly
                # ['adx_14_adx', '<', 20],  # Exit when ADX drops below 20
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
            # ================================================================
            # STAGE 0: Main Strategy Parameters
            # ================================================================
            'main_strategy': {
                'enabled': False,
                #'side_method': ['BOTH', 'LONG', 'SHORT'],
                #'leverage': (1, 20, 1),
            },

            # ================================================================
            # STAGE 1: Risk Management
            # ================================================================
            'risk_management': {
                'enabled': False,
                #'sizing_method': ['FIXED_PERCENT', 'RISK_BASED', 'FIXED_USD'],
                #'position_percent_size': (5.0, 25.0, 2.5),
                #'position_usd_size': (100, 1000, 100),
                #'max_risk_per_trade': (1.0, 5.0, 0.5),
            },

            # ================================================================
            # STAGE 2: Exit Strategy
            # ================================================================
            'exit_strategy': {
                'enabled': False,

                # Stop Loss
                #'stop_loss_method': ['FIXED_PERCENT', 'ATR_BASED'],
                #'stop_loss_percent': (0.5, 6.0, 0.25),
                #'stop_loss_atr_multiplier': (1.0, 3.0, 0.5),

                # Take Profit
                #'take_profit_method': ['FIXED_PERCENT', 'RISK_REWARD'],
                #'take_profit_percent': (1.0, 12.0, 0.5),
                #'take_profit_risk_reward_ratio': (1.5, 4.0, 0.5),

                # Break-Even
                'break_even_enabled': [True, False],
                'break_even_trigger_profit_percent': (0.5, 2.5, 0.25),

                # Trailing Stop
                'trailing_stop_enabled': [True, False],
                'trailing_activation_profit_percent': (1.0, 4.0, 0.5),
                'trailing_callback_percent': (0.2, 1.0, 0.1),
            },

            # ================================================================
            # STAGE 3: Indicators
            # ================================================================
            'indicators': {
                'enabled': False,

                # EMA
                #'ema_200': {'period': (100, 200, 20)},

                # PMax
                #'pmax': {
                #    'atr_period': (7, 14, 1),
                #    'atr_multiplier': (2.0, 4.0, 0.5),
                #    'ma_period': (7, 14, 1),
                #},
            },

            # ================================================================
            # STAGE 4: Entry Conditions
            # ================================================================
            'entry_conditions': {
                'enabled': False,
            },

            # ================================================================
            # STAGE 5: Position Management
            # ================================================================
            'position_management': {
                'enabled': False,
                #'max_positions_per_symbol': (1, 3, 1),
                #'max_total_positions': (1, 10, 1),
            },

            # ================================================================
            # STAGE 6: Market Filters
            # ================================================================
            'market_filters': {
                'enabled': False,
            },

            # ================================================================
            # CONSTRAINTS
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
    logger.info(f"   LONG: Close > EMA200 + ADX > 25 + PMax crossover")
    logger.info(f"   SHORT: Close < EMA200 + ADX > 25 + PMax crossunder")
    logger.info(f"\n🚪 Exit Logic:")
    logger.info(f"   TP/SL: 8%/4% (R/R = 1:2)")
    logger.info(f"   Break-even: Activate at 2%, move SL to +0.2%")
    logger.info(f"   Partial Exit: 30% @ 3%, 40% @ 5%, 30% @ 7%")
    logger.info(f"   Trailing: Activate at 4%, callback 1.5%")
    logger.info(f"   Exit condition: DISABLED (too aggressive)")

    logger.info(f"\n✅ Strategy loaded successfully!")
