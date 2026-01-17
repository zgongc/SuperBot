#!/usr/bin/env python3
"""
components/strategies/templates/smc_v2.py
SuperBot - Smart Money Concept (SMC) Strategy v2
Yazar: SuperBot Team
Tarih: 2025-12-25
Versiyon: 2.0.0

Smart Money Concept (SMC) / ICT Strategy v2

v2 Degisiklikler:
- Entry: BOS VEYA CHoCH ile giris (herhangi bir structure break)
- Exit: Ters yonde structure break (BOS veya CHoCH)
- Daha dogru BOS/CHoCH sinyal kullanimi

Strateji Mantigi:
- BOS (Break of Structure): Trend devam sinyali
- CHoCH (Change of Character): Trend donus sinyali
- Her ikisi de structure break, farkli market koşullarinda

Entry:
- Long: Bullish BOS (trend devam) veya Bullish CHoCH (trend donus)
- Short: Bearish BOS (trend devam) veya Bearish CHoCH (trend donus)

Exit:
- Long: Bearish structure break (BOS veya CHoCH)
- Short: Bullish structure break (BOS veya CHoCH)
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
    Smart Money Concept (SMC) Strategy v2

    BOS ve CHoCH'u birlikte kullanan strateji
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # STRATEGY METADATA
        # ====================================================================
        self.strategy_name = "Smart Money Concept Strategy v2"
        self.strategy_version = "2.0.0"
        self.description = "SMC with combined BOS/CHoCH entry signals"
        self.author = "SuperBot Team"
        self.created_date = "2025-12-25"

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
            "min_spread": 0.0001,
            "commission": 0.0002,
            "max_slippage": 0.0005
        }

        # ====================================================================
        # TRADING CONFIGURATION
        # ====================================================================
        self.side_method = TradingSide.BOTH
        self.leverage = 10

        self.mtf_timeframes = ['5m']
        self.primary_timeframe = "5m"

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
        # EXIT STRATEGY
        # ====================================================================
        self.exit_strategy = ExitStrategy(
            take_profit_method=ExitMethod.FIXED_PERCENT,
            take_profit_percent=3.0,
            take_profit_price=110000.0,
            take_profit_risk_reward_ratio=2.0,
            take_profit_atr_multiplier=2.5,
            take_profit_fib_level=1.618,
            take_profit_ai_level=1,

            stop_loss_method=StopLossMethod.FIXED_PERCENT,
            stop_loss_percent=1.5,
            stop_loss_price=95000.0,
            stop_loss_atr_multiplier=1.5,
            stop_loss_swing_lookback=5,
            stop_loss_fib_level=0.382,
            stop_loss_ai_level=1,

            trailing_stop_enabled=True,
            trailing_activation_profit_percent=1.5,
            trailing_callback_percent=0.5,
            trailing_take_profit=False,
            trailing_distance=0.3,

            break_even_enabled=True,
            break_even_trigger_profit_percent=1.2,
            break_even_offset=0.1,

            partial_exit_enabled=True,
            partial_exit_levels=[1.2, 2.0, 3.0],
            partial_exit_sizes=[0.40, 0.30, 0.30],
        )

        # ====================================================================
        # INDICATORS (SMC Detection)
        # ====================================================================
        self.technical_parameters = TechnicalParameters(
            indicators={
                # BOS (Break of Structure) - Trend continuation signal
                "bos": {"left_bars": 5, "right_bars": 5, "max_levels": 3, "trend_strength": 3},

                # CHoCH (Change of Character) - Trend reversal signal
                "choch": {"left_bars": 5, "right_bars": 5, "max_levels": 3, "trend_strength": 3},

                # Trend filters
                "ema_21": {"period": 21},
                "ema_50": {"period": 50},
                "ema_200": {"period": 200},

                # Momentum
                "rsi_14": {"period": 14},

                # ADX for trend strength
                "adx": {"period": 14},

                # Volatility
                "atr_14": {"period": 14},
            }
        )

        # ====================================================================
        # ENTRY CONDITIONS (SMC v2 - CHoCH ile entry)
        # ====================================================================
        # SMC Mantigi:
        # - CHoCH = Trend donus sinyali (daha guvenilir entry)
        # - BOS = Trend devam sinyali (pyramiding icin kullanilabilir)
        #
        # v2'de CHoCH ile entry yapiyoruz cunku:
        # - CHoCH trend donusunu yakalar (dip/tepe)
        # - BOS zaten trend yonunde, daha gec sinyal
        self.entry_conditions = {
            'long': [
                # CHoCH bullish = Downtrend'den uptrend'e donus
                ['choch', '==', 1],

                # TREND STRENGTH: ADX > 20 (trending market)
                ['adx_adx', '>', 20],

                # MOMENTUM: RSI not overbought (room to go up)
                ['rsi_14', '<', 60],
            ],
            'short': [
                # CHoCH bearish = Uptrend'den downtrend'e donus
                ['choch', '==', -1],

                # TREND STRENGTH: ADX > 20 (trending market)
                ['adx_adx', '>', 20],

                # MOMENTUM: RSI not oversold (room to go down)
                ['rsi_14', '>', 40],
            ]
        }

        # ====================================================================
        # EXIT CONDITIONS (Ters Structure Break)
        # ====================================================================
        self.exit_conditions = {
            'long': [
                # Bearish structure break (CHoCH veya BOS)
                ['choch', '==', -1],
                ['close', 'crossunder', 'ema_50'],
            ],
            'short': [
                # Bullish structure break
                ['choch', '==', 1],
                ['close', 'crossover', 'ema_50'],
            ],
            'stop_loss': [],
            'take_profit': []
        }

        # ====================================================================
        # CUSTOM PARAMETERS
        # ====================================================================
        self.custom_parameters = {
            "news_filter": False,
            "session_filter": {"enabled": False},
            "time_filter": {"enabled": False},
            "day_filter": {"enabled": False},
        }


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    strategy = Strategy()

    print("\n" + "=" * 60)
    print(f"  {strategy.strategy_name} v{strategy.strategy_version}")
    print("=" * 60)

    print(f"\n  Description : {strategy.description}")
    print(f"  Author      : {strategy.author}")
    print(f"  Created     : {strategy.created_date}")

    print(f"\n  ENTRY CONDITIONS")
    print(f"  {'-' * 40}")
    print(f"  LONG  ({len(strategy.entry_conditions.get('long', []))} conditions):")
    for cond in strategy.entry_conditions.get('long', []):
        print(f"    - {cond}")
    print(f"  LONG_CHOCH ({len(strategy.entry_conditions.get('long_choch', []))} conditions):")
    for cond in strategy.entry_conditions.get('long_choch', []):
        print(f"    - {cond}")

    print("\n" + "=" * 60)
    print("  Strategy loaded successfully!")
    print("=" * 60 + "\n")
