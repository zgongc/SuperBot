#!/usr/bin/env python3
"""
components/strategies/templates/base_template.py
SuperBot - TradingView Multi-Indicator Dashboard Strategy
Yazar: SuperBot Team
Tarih: 2025-11-15
Versiyon: 1.0.0


Expected Performance:
- Very few trades (strict filtering)
- High win rate (all indicators aligned)
- Strong trends only
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
    RSI + EMA Trend Continuation Strategy

    Logic: Don't catch falling knives. Wait for RSI to exit extreme zones,
    then trade in the direction of the trend (EMA filter).

    LONG:  Uptrend (close > EMA50) + RSI crosses down from overbought (75)
           → Pullback complete, trend resumes up

    SHORT: Downtrend (close < EMA50) + RSI crosses up from oversold (20)
           → Dead cat bounce over, trend resumes down
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "simple_rsi"
        self.strategy_version = "1.0.0"
        self.description = "simple_rsi_strategy - trend continuation"
        self.author = "SuperBot Team"
        self.created_date = "2025-11-15"

        # ====================================================================
        # BACKTEST CONFIGURATION
        # ====================================================================
        self.backtesting_enabled = True
        self.backtest_start_date = "2025-01-05T00:00"
        self.backtest_end_date = "2025-03-30T00:00"
        self.initial_balance = 1000
        self.warmup_period = 200  # SMA 200 için yeterli
        self.download_klines = False
        self.update_klines = False

        # ====================================================================
        # BACKTEST PARAMETERS (Realistic Binance Futures)
        # ====================================================================
        self.backtest_parameters = {
            "min_spread": 0.0001,     # %0.01 spread
            "commission": 0.0002,     # %0.02 Maker fee (Binance Futures)
            "max_slippage": 0.0005    # %0.05 slippage
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH  # LONG, SHORT, BOTH, FLAT
        self.leverage = 1  # 10x leverage (max_portfolio_risk auto-calculated: 10 × 100 = 1000)

        # Timeframe
        self.primary_timeframe = "5m"
        self.mtf_timeframes = ['5m','15m']  # Primary timeframe MUST be in list

        # Exchange config
        self.set_default_leverage = False
        self.hedge_mode = False
        self.set_margin_type = False
        self.margin_type = "isolated"

        # ====================================================================
        # SYMBOLS
        # ====================================================================
        self.symbol_source = "strategy" # file, strategy, exchange
        self.symbols = [
            SymbolConfig(
                symbol=['BTC','BNB','ETH','SOL','AVAX','LINK','BCH','ZEC','DOT','ADA'],
                #symbol=['BTC','SOL'],
                quote="USDT",
                enabled=True
            )
        ]

        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management = RiskManagement(
            sizing_method=PositionSizeMethod.FIXED_PERCENT,
            position_percent_size=10.0,         # 10.0 FIXED_PERCENT için (AKTİF) - Portfolio'nun %10'u × leverage
            position_usd_size=300.0,           # FIXED_USD için (şu an kullanılmaz)
            position_quantity_size=2.0,         # FIXED_QUANTITY = "FIXED_QUANTITY" # Sabit miktar (0.01 BTC)
            max_risk_per_trade=2.5,             # RISK_BASED için (şu an kullanılmaz)

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
            max_positions_per_symbol=1,  # Her sembol için max pozisyon sayısı
            max_total_positions=400,  # Toplam max pozisyon sayısı (tüm semboller)
            allow_hedging=False,  # Karşıt yönde pozisyon açma izni (LONG+SHORT birlikte)
            position_timeout_enabled=True,  # Pozisyon timeout kontrolü (dakika cinsinden)
            position_timeout=1800,  # Timeout süresi: 1800 dakika = 30 saat
            pyramiding_enabled=False,  # Aynı yönde kademeli giriş izni
            pyramiding_max_entries=3,  # Max pyramiding entry sayısı (aynı yön)
            pyramiding_scale_factor=0.5  # Her yeni entry'de boyut çarpanı (0.5 = yarı yarıya)
        )

        # ====================================================================
        # EXIT STRATEGY
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=6.00,             # FIXED_PERCENT - %6 TP
            take_profit_price=110000.0,          # FIXED_PRICE - $110,000'da TP
            take_profit_risk_reward_ratio=2.0,   # RISK_REWARD - 1:2.5 R/R → SL %1.5 ise, TP %3.75 (1.5 × 2.5)
            take_profit_atr_multiplier=4.0,      # ATR_BASED - 3× ATR kadar yukarıda TP
            take_profit_fib_level=1.618,         # FIBONACCI - Fib extension 1.618
            take_profit_ai_level=1,              # DYNAMIC_AI

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=3.2,               # FIXED_PERCENT - %1.5 SL
            stop_loss_price=95000.0,             # FIXED_PRICE - $95,000'da SL
            stop_loss_atr_multiplier=2.0,        # ATR_BASED - 2× ATR kadar aşağıda SL
            stop_loss_swing_lookback=10,          # SWING_POINTS - Son 3 mum'un swing low/high'ı
            stop_loss_fib_level=0.382,           # FIBONACCI - Fib retracement 0.382
            stop_loss_ai_level=1,                # DYNAMIC_AI

            # Trailing Stop (lock profits)
            trailing_stop_enabled=True,
            trailing_activation_profit_percent=1.0,  # 2.5%'te aktif
            trailing_callback_percent=0.3,           # 0.6% geri çekilince close
            trailing_take_profit=False,              # TP'ye ulaştığında trailing devam etsin mi?
            trailing_distance=0.2,                   # Trailing stop distance (%)

            # Break-even (early protection)
            break_even_enabled=True,  # TEST: Break-even + partial exit together
            break_even_trigger_profit_percent=1.4,  # 1.2%'de aktif
            break_even_offset=0.9,

            # Partial Exit (Kısmi Kar Al) - RECOMMENDED: 3 levels with graduated weights
            partial_exit_enabled=True,  # TEST: Partial exit enabled
            partial_exit_levels=[3, 4, 10],           # %2, %5, %8 (good spacing)
            partial_exit_sizes=[0.40, 0.40, 0.20],   # Graduated: 25% → 35% → 40% = 100%
        )

        # ====================================================================
        # INDICATORS (11 indicators for dashboard)
        # ====================================================================
        self.technical_parameters = TechnicalParameters(
            indicators={
                "ema_50": {"period": 50},
                "adx_14": {'period': 14, 'adx_threshold': 25},
                "rsi_14": {"period": 14, 'overbought': 75, 'oversold': 20},
                #"atr_14": {"period": 14},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS (CROSSOVER = EARLY ENTRY!)
        # ====================================================================
        self.entry_conditions = {
            'long': [
                ['close', '>', 'ema_50'],
                #['adx_14_adx', '>', 20],          # Strong trend (not ranging)
                ['rsi_14', 'crossunder', 75],  # No '5m' for now
            ],
            'short': [
                ['close', '<', 'ema_50'],
                #['adx_14_adx', '>', 20],          # Strong trend (not ranging)
                ['rsi_14', 'crossover', 20],  # No '5m' for now
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS (Let profits run!)
        # ====================================================================
        self.exit_conditions = {
            'long': [
            ],
            'short': [  
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
                "enabled": False,  # Monday/Tuesday disabled (better performance)
                "monday": False,
                "tuesday": False,
                "wednesday": True,
                "thursday": True,
                "friday": True,
                "saturday": True,
                "sunday": True,
            },
        }


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
        # ====================================================================
        # OPTIMIZER PARAMETERS
        # ====================================================================
        # Bu parametreler backtest/trade sırasında kullanılmaz.
        # Sadece optimizer tarafından okunur ve optimize edilir.
        #
        # Format:
        #   - Tuple (min, max, step): Numeric range optimization
        #   - List [val1, val2, ...]: Categorical/enum optimization
        #   - Comment (#) ile kapatılan parametreler optimize edilmez
        #
        # WebUI üzerinden template edit edilirken bu parametreler
        # otomatik gösterilir ve değiştirilebilir.
        #
        # Optimizer Stage Flow:
        #   Stage 1 (Risk Management): Position sizing, leverage
        #   Stage 2 (Exit Strategy): SL, TP, trailing, break-even
        #   Stage 3 (Indicators): Indicator periods and thresholds
        #   Stage 4 (Entry Conditions): Entry logic thresholds
        #   Stage 5 (Position Management): Max positions, pyramiding
        #   Stage 6 (Market Filters): Session, time, day filters
        # ====================================================================

        # ====================================================================
        # OPTIMIZER PARAMETERS
        # ====================================================================
        # Bu parametreler backtest/trade sırasında kullanılmaz.
        # Sadece optimizer tarafından okunur ve optimize edilir.
        #
        # Format:
        #   - Tuple (min, max, step): Numeric range → [min, min+step, ..., max]
        #   - List [val1, val2, ...]: Categorical/enum seçenekleri
        #
        # Kullanım:
        #   python modules/optimizer/cli.py --strategy <path> --stage <stage_name> --trials 50
        #
        # ÖNEMLİ: Attribute isimleri base_strategy.py'deki dataclass'larla eşleşmeli!
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
            # Attribute names: RiskManagement dataclass
            'risk_management': {
                'enabled': False,
                # sizing_method kullanıyorsa conditional logic optimizer'da var
                #'sizing_method': ['FIXED_PERCENT', 'RISK_BASED', 'FIXED_USD'],
                #'position_percent_size': (5.0, 25.0, 2.5),    # FIXED_PERCENT için
                #'position_usd_size': (100, 1000, 100),        # FIXED_USD için
                #'max_risk_per_trade': (1.0, 5.0, 0.5),        # RISK_BASED için
            },

            # ================================================================
            # STAGE 2: Exit Strategy
            # ================================================================
            # Attribute names: ExitStrategy dataclass
            # Önerilen: Aşamalı optimize edin (SL/TP → Break-even → Trailing)
            'exit_strategy': {
                'enabled': False,

                # --- Stop Loss ---
                #'stop_loss_method': ['FIXED_PERCENT', 'ATR_BASED', 'SWING_POINTS', 'FIBONACCI'],
                #'stop_loss_percent': (0.5, 6.0, 0.25),        # FIXED_PERCENT
                #'stop_loss_atr_multiplier': (1.0, 3.0, 0.5),  # ATR_BASED
                #'stop_loss_swing_lookback': (5, 20, 5),       # SWING_POINTS
                #'stop_loss_fib_level': [0.236, 0.382, 0.5, 0.618],  # FIBONACCI

                # --- Take Profit ---
                #'take_profit_method': ['FIXED_PERCENT', 'RISK_REWARD', 'ATR_BASED', 'FIBONACCI'],
                #'take_profit_percent': (1.0, 12.0, 0.5),              # FIXED_PERCENT
                #'take_profit_risk_reward_ratio': (1.5, 4.0, 0.5),     # RISK_REWARD
                #'take_profit_atr_multiplier': (2.0, 6.0, 1.0),        # ATR_BASED
                #'take_profit_fib_level': [1.0, 1.272, 1.618, 2.0],    # FIBONACCI

                # --- Break-Even ---
                #'break_even_enabled': [True, False],
                #'break_even_trigger_profit_percent': (0.5, 2.5, 0.25),
                #'break_even_offset': (0.05, 0.5, 0.05),

                # --- Trailing Stop ---
                'trailing_stop_enabled': [True, False],
                'trailing_activation_profit_percent': (1.0, 4.0, 0.5),
                'trailing_callback_percent': (0.2, 1.0, 0.1),
                'trailing_distance': (0.1, 0.5, 0.1),

                # --- Partial Exit ---
                # NOT: Liste parametreleri henüz desteklenmiyor
                #'partial_exit_enabled': [True, False],
            },

            # ================================================================
            # STAGE 3: Indicators
            # ================================================================
            # Nested dict format: {'indicator_name': {'param': range}}
            # NOT: Henüz tam desteklenmiyor, flat parametreler kullanın
            'indicators': {
                'enabled': False,

                # RSI
                #'rsi_14': {
                #    'period': (7, 28, 7),
                #    'overbought': (65, 80, 5),
                #    'oversold': (20, 35, 5),
                #},

                # EMA
                #'ema_fast': {'period': (5, 21, 2)},
                #'ema_slow': {'period': (21, 100, 10)},

                # Bollinger Bands
                #'bollinger': {
                #    'period': (14, 26, 4),
                #    'std_dev': (1.5, 2.5, 0.5),
                #},

                # ADX
                #'adx_14': {
                #    'period': (10, 28, 4),
                #    'adx_threshold': (20, 30, 5),
                #},

                # ATR
                #'atr_14': {'period': (7, 28, 7)},

                # MACD
                #'macd': {
                #    'fast_period': (8, 16, 2),
                #    'slow_period': (20, 30, 2),
                #    'signal_period': (7, 12, 1),
                #},

                # SuperTrend
                #'supertrend': {
                #    'period': (7, 14, 1),
                #    'multiplier': (2.0, 4.0, 0.5),
                #},

                # Stochastic
                #'stochastic': {
                #    'k_period': (5, 21, 4),
                #    'd_period': (3, 7, 2),
                #},
            },

            # ================================================================
            # STAGE 4: Entry Conditions
            # ================================================================
            # NOT: Henüz implement edilmedi
            'entry_conditions': {
                'enabled': False,
                #'min_conditions_met': (1, 5, 1),
                #'confirmation_candles': (0, 3, 1),
            },

            # ================================================================
            # STAGE 5: Position Management
            # ================================================================
            # Attribute names: PositionManagement dataclass
            'position_management': {
                'enabled': False,
                #'max_positions_per_symbol': (1, 3, 1),
                #'max_total_positions': (1, 10, 1),
                #'allow_hedging': [True, False],
                #'pyramiding_enabled': [True, False],
                #'pyramiding_max_entries': (2, 5, 1),
                #'pyramiding_scale_factor': (0.3, 1.0, 0.1),
                #'position_timeout_enabled': [True, False],
                #'position_timeout': (60, 1440, 60),  # dakika cinsinden
            },

            # ================================================================
            # STAGE 6: Market Filters
            # ================================================================
            # NOT: Henüz implement edilmedi
            'market_filters': {
                'enabled': False,
                #'session_filter_enabled': [True, False],
                #'day_filter_enabled': [True, False],
                #'time_filter_enabled': [True, False],
            },

            # ================================================================
            # CONSTRAINTS (Optimizer global ayarları)
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
    logger.info(f"   Position Size: {strategy.risk_management.size_value}%")
    logger.info(f"   Take Profit: {strategy.exit_strategy.take_profit_value}%")
    logger.info(f"   Stop Loss: {strategy.exit_strategy.stop_loss_value}%")
    rr_ratio = strategy.exit_strategy.take_profit_value / strategy.exit_strategy.stop_loss_value
    logger.info(f"   R/R Ratio: 1:{rr_ratio:.2f}\n")

    logger.info(f"📊 Indicators ({len(strategy.technical_parameters.indicators)}):")
    for name in strategy.technical_parameters.indicators.keys():
        logger.info(f"   ✓ {name}")

    logger.info(f"\n📈 Entry Logic:")
    logger.info(f"   LONG: 5 SMART filters (MACD + EMA + SMA200 + RSI + Price)")
    logger.info(f"   SHORT: 5 SMART filters (less is more!)")
    logger.info(f"   Exit: Only on trend reversal (let profits run)")

    logger.info(f"\n✅ Strategy loaded successfully!")
