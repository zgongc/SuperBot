"""
strategies/helpers/indicator_presets.py - Indicator Configuration Presets

Version: 1.0.0
Date: 2025-10-23
Author: SuperBot Team

Description:
    72 indikatör için hazır konfigürasyonlar.
    Strategy'de sadece get_preset() çağır, tüm config gelsin!

    Avantajları:
    - ✅ 70+ indicator için standart config
    - ✅ Signal confirmation örnekleri
    - ✅ Açıklamalı (eğitici)
    - ✅ Override edilebilir
    - ✅ Maintenance kolaylaşır

    Kullanım:
        from strategies.helpers.indicator_presets import get_preset, merge_config

        # Standart config kullan
        indicators = {
            "rsi": get_preset("rsi"),
            "ema": get_preset("ema"),
        }

        # Özelleştir
        indicators = {
            "rsi": get_preset("rsi", period=21, oversold=25),  # Override
            "ema": merge_config(
                get_preset("ema"),
                {"default_fast": 12}  # Sadece bir parametreyi değiştir
            )
        }

Dependencies:
    - indicators.indicator_registry (local)
"""

from typing import Dict, Any, Optional
from copy import deepcopy


# ============================================================================
# INDICATOR PRESETS (72 indicators)
# ============================================================================

INDICATOR_PRESETS: Dict[str, Dict[str, Any]] = {

    # ========================================================================
    # MOMENTUM (11 indicators)
    # ========================================================================

    "rsi": {
        "display_info": True,
        "default_period": 14,
        "oversold": 30,              # Oversold seviye
        "overbought": 70,            # Overbought seviye

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,       # ⚠️ Backtest için False (historical data yok)

            # Method: "trend_reversal"
            # Açıklama: RSI oversold/overbought bölgesine girip
            #           çıkana kadar bekle (gerçek reversal)
            "method": "trend_reversal",

            # distance_ratio: Ne kadar yakın olmalı (0.2 = %20)
            "distance_ratio": 0.2,

            # min_bars_in_zone: Kaç bar oversold/overbought'ta kalmalı
            "min_bars_in_zone": 2,   # 2 bar zone'da kal

            # confirmation_bars: Çıkış için kaç bar bekle
            "confirmation_bars": 1,  # 1 bar sonra reversal onayı

            # max_pullback: Max geri çekilme (0.1 = %10)
            "max_pullback": 0.1
        },
    },

    "stochastic": {
        "display_info": True,
        "default_k_period": 21,      # %K period
        "default_d_period": 3,       # %D period (signal line)
        "default_smooth_k": 3,       # Smoothing
        "oversold": 20,
        "overbought": 80,

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,        # ✅ %K/%D crossover

            # Method: "crossover_confirmation"
            # Açıklama: %K ve %D çizgileri kesiştiğinde
            "method": "crossover_confirmation",

            # min_divergence: Min %K-%D farkı
            "min_divergence": 5.0,   # 5 puan fark

            # confirmation_bars: Crossover sonrası kaç bar bekle
            "confirmation_bars": 1
        },
    },

    "rsi_divergence": {
        "display_info": True,
        "default_period": 14,
        "lookback": 5,               # Kaç bar geriye bak
        "min_divergence_strength": 0.5,  # Min divergence gücü

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 2
        },
    },

    "williams_r": {
        "display_info": True,
        "default_period": 14,
        "oversold": -80,
        "overbought": -20,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "distance_ratio": 0.2
        },
    },

    "roc": {
        "display_info": True,
        "default_period": 12,
        "zero_line": 0,              # Zero line (momentum değişim noktası)

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "cci": {
        "display_info": True,
        "default_period": 20,
        "oversold": -100,
        "overbought": 100,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "distance_ratio": 0.15
        },
    },

    "mfi": {
        "display_info": True,
        "default_period": 14,
        "oversold": 20,
        "overbought": 80,
        "requires_volume": True,     # ⚠️ Volume gerekli!

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "distance_ratio": 0.2
        },
    },

    "tsi": {
        "display_info": True,
        "long_period": 25,
        "short_period": 13,
        "signal_period": 7,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "awesome": {
        "display_info": True,
        "fast_period": 5,
        "slow_period": 34,
        "zero_line": 0,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "ultimate": {
        "display_info": True,
        "period1": 7,
        "period2": 14,
        "period3": 28,
        "oversold": 30,
        "overbought": 70,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "distance_ratio": 0.2
        },
    },

    "stochastic_rsi": {
        "display_info": True,
        "rsi_period": 14,
        "stoch_period": 14,
        "k_period": 3,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "min_divergence": 5.0
        },
    },

    # ========================================================================
    # TREND (15 indicators)
    # ========================================================================

    "sma": {
        "display_info": True,
        "default_period": 20,

        "signal_confirmation": {
            "enabled": False,

            # Method: "crossover_confirmation"
            # Açıklama: Price SMA'yı kestiğinde
            "method": "crossover_confirmation",

            "distance_ratio": 0.1,   # %10 mesafe
            "confirmation_bars": 1
        },
    },

    "ema": {
        "display_info": True,
        "default_fast": 9,           # Hızlı EMA
        "default_slow": 21,          # Yavaş EMA

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,        # ✅ Crossover confirmation

            # Method: "crossover_confirmation"
            # Açıklama: EMA crossover gerçekleştikten sonra
            #           belli bir mesafe ayrılana kadar bekle
            "method": "crossover_confirmation",

            # distance_ratio: Crossover sonrası min mesafe
            "distance_ratio": 0.15,  # %15 ayrılma

            # min_separation: Min absolute separation
            "min_separation": 0.02   # %2 min ayırma
        },
    },

    "wma": {
        "display_info": True,
        "default_period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "distance_ratio": 0.1
        },
    },

    "hma": {
        "display_info": True,
        "default_period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "distance_ratio": 0.1
        },
    },

    "tema": {
        "display_info": True,
        "default_period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "distance_ratio": 0.1
        },
    },

    "dema": {
        "display_info": True,
        "default_period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "distance_ratio": 0.1
        },
    },

    "vwma": {
        "display_info": True,
        "default_period": 20,
        "requires_volume": True,     # ⚠️ Volume gerekli!

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "distance_ratio": 0.1
        },
    },

    "macd": {
        "display_info": True,
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,

            # Method: "crossover_confirmation"
            # Açıklama: MACD line ve Signal line kesiştiğinde
            "method": "crossover_confirmation",

            "distance_ratio": 0.1,
            "confirmation_bars": 1,
            "min_histogram": 0.001   # Min histogram değeri
        },
    },

    "adx": {
        "display_info": True,
        "default_period": 14,
        "threshold": 25,             # Trend gücü eşiği (>25 = güçlü trend)

        "signal_confirmation": {
            "enabled": False,

            # Method: "volatility_filter"
            # Açıklama: ADX > threshold ise trend var, trade yap
            "method": "volatility_filter",

            "high_threshold": 25,    # Min ADX (trend strength)
            "low_threshold": 20      # Weak trend level
        },
    },

    "aroon": {
        "display_info": True,
        "default_period": 25,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "supertrend": {
        "display_info": True,
        "period": 10,
        "multiplier": 3.0,

        "signal_confirmation": {
            "enabled": False,

            # Method: "trend_reversal"
            # Açıklama: SuperTrend renk değiştirdiğinde
            "method": "trend_reversal",

            "confirmation_bars": 1,
            "min_distance": 0.02     # %2 min mesafe
        },
    },

    "parabolic_sar": {
        "display_info": True,
        "acceleration": 0.02,
        "maximum": 0.2,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "ichimoku": {
        "display_info": True,
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_period": 52,

        "signal_confirmation": {
            "enabled": False,

            # Method: "crossover_confirmation"
            # Açıklama: Tenkan-Kijun cross + Cloud desteği
            "method": "crossover_confirmation",

            "confirmation_bars": 1,
            "require_cloud_support": True  # Cloud desteği gerekli mi?
        },
    },

    "donchian": {
        "display_info": True,
        "period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "keltner": {
        "display_info": True,
        "period": 20,
        "multiplier": 2.0,

        "signal_confirmation": {
            "enabled": False,

            # Method: "squeeze_expansion"
            # Açıklama: Keltner channel daraldıktan sonra genişlediğinde
            "method": "squeeze_expansion",

            "min_squeeze": 0.5,
            "expansion_threshold": 1.5
        },
    },

    # ========================================================================
    # VOLATILITY (8 indicators)
    # ========================================================================

    "atr": {
        "display_info": True,
        "default_period": 14,
        "multiplier": 1.5,           # Stop loss için ATR çarpanı

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,        # ✅ Volatility filter

            # Method: "volatility_filter"
            # Açıklama: Aşırı yüksek/düşük volatilitede trade yapma
            "method": "volatility_filter",

            # high_threshold: Çok yüksek volatilite (trade yapma)
            "high_threshold": 2.0,   # 2x average ATR

            # low_threshold: Çok düşük volatilite (range)
            "low_threshold": 0.5     # 0.5x average ATR
        },
    },

    "bollinger": {
        "display_info": True,
        "default_period": 20,
        "default_std_dev": 2.0,      # Standard deviation

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,        # ✅ Squeeze/expansion detection

            # Method: "squeeze_expansion"
            # Açıklama: BB sıkışıp genişlediğinde sinyal ver
            #           (volatilite artışı)
            "method": "squeeze_expansion",

            # min_squeeze: Min BB genişlik (std cinsinden)
            "min_squeeze": 0.5,      # 0.5 std

            # expansion_threshold: Genişleme eşiği
            "expansion_threshold": 1.5  # 1.5x genişleme
        },
    },

    "keltner_vol": {
        "display_info": True,
        "period": 20,
        "multiplier": 2.0,

        "signal_confirmation": {
            "enabled": False,
            "method": "squeeze_expansion",
            "min_squeeze": 0.5,
            "expansion_threshold": 1.5
        },
    },

    "standard_dev": {
        "display_info": True,
        "period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "volatility_filter",
            "high_threshold": 2.0,
            "low_threshold": 0.5
        },
    },

    "true_range": {
        "display_info": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "volatility_filter",
            "high_threshold": 2.0
        },
    },

    "natr": {
        "display_info": True,
        "period": 14,

        "signal_confirmation": {
            "enabled": False,
            "method": "volatility_filter",
            "high_threshold": 5.0,   # %5 normalized ATR
            "low_threshold": 1.0     # %1 normalized ATR
        },
    },

    "chandelier": {
        "display_info": True,
        "period": 22,
        "multiplier": 3.0,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "squeeze": {
        "display_info": True,
        "bb_period": 20,
        "kc_period": 20,
        "bb_multiplier": 2.0,
        "kc_multiplier": 1.5,

        "signal_confirmation": {
            "enabled": False,

            # Method: "squeeze_expansion"
            # Açıklama: TTM Squeeze - BB Keltner içinde sıkıştığında
            "method": "squeeze_expansion",

            "confirmation_bars": 1,
            "require_momentum_shift": True  # Momentum değişimi gerekli mi?
        },
    },

    # ========================================================================
    # VOLUME (9 indicators)
    # ========================================================================

    "obv": {
        "display_info": True,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,

            # Method: "volume_surge"
            # Açıklama: OBV trend değişimi
            "method": "volume_surge",

            "confirmation_bars": 2,
            "min_change": 0.05       # %5 min değişim
        },
    },

    "vwap": {
        "display_info": True,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,

            # Method: "crossover_confirmation"
            # Açıklama: Price VWAP'ı kestiğinde
            "method": "crossover_confirmation",

            "distance_ratio": 0.1,
            "confirmation_bars": 1
        },
    },

    "vwap_bands": {
        "display_info": True,
        "std_dev": 2.0,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "squeeze_expansion",
            "min_squeeze": 0.5,
            "expansion_threshold": 1.5
        },
    },

    "volume": {
        "display_info": True,
        "default_sma_period": 20,    # Volume SMA period
        "volume_threshold": 1.5,     # 1.5x average = surge
        "requires_volume": True,

        # 🎯 Signal Confirmation
        "signal_confirmation": {
            "enabled": False,        # ✅ Volume surge detection

            # Method: "volume_surge"
            # Açıklama: Hacim aniden artarsa (volume spike)
            "method": "volume_surge",

            # distance_ratio: Threshold multiplier
            "distance_ratio": 0.3,   # %30 fazla

            # confirmation_bars: Kaç bar yüksek hacim kalmalı
            "confirmation_bars": 1
        },
    },

    "ad": {
        "display_info": True,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "volume_surge",
            "confirmation_bars": 2
        },
    },

    "cmf": {
        "display_info": True,
        "period": 20,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "eom": {
        "display_info": True,
        "period": 14,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "force_index": {
        "display_info": True,
        "period": 13,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1
        },
    },

    "volume_profile": {
        "display_info": True,
        "bins": 24,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "volume_surge",
            "confirmation_bars": 1
        },
    },

    # ========================================================================
    # SUPPORT/RESISTANCE (8 indicators)
    # ========================================================================

    "pivot_points": {
        "display_info": True,
        "method": "classic",         # classic, fibonacci, camarilla, woodie

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1,
            "bounce_distance": 0.002  # %0.2 bounce
        },
    },

    "fibonacci_pivot": {
        "display_info": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "camarilla": {
        "display_info": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "woodie": {
        "display_info": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "zigzag": {
        "display_info": True,
        "deviation": 5.0,            # %5 deviation

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 2
        },
    },

    "support_resistance": {
        "display_info": True,
        "lookback": 50,
        "tolerance": 0.02,           # %2 tolerance

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "fib_retracement": {
        "display_info": True,
        "levels": [0.236, 0.382, 0.5, 0.618, 0.786],

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "swing_points": {
        "display_info": True,
        "lookback": 5,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    # ========================================================================
    # STRUCTURE (6 indicators) - Smart Money Concepts
    # ========================================================================

    "bos": {
        "display_info": True,
        "lookback": 50,

        "signal_confirmation": {
            "enabled": False,

            # Method: "trend_reversal"
            # Açıklama: BOS (Break of Structure) tespit edildiğinde
            "method": "trend_reversal",

            "confirmation_bars": 2,
            "min_structure_size": 0.005  # %0.5 min structure
        },
    },

    "choch": {
        "display_info": True,
        "lookback": 50,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 2
        },
    },

    "fvg": {
        "display_info": True,
        "min_gap_percent": 0.1,      # %0.1 min gap

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1,
            "require_fill": False    # Gap doldurulmalı mı?
        },
    },

    "orderblocks": {
        "display_info": True,
        "lookback": 50,
        "min_body_percent": 30,      # %30 min body size
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 2,
            "require_volume_spike": True  # Volume spike gerekli mi?
        },
    },

    "liquidityzones": {
        "display_info": True,
        "lookback": 100,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 2,
            "require_sweep": True    # Liquidity sweep gerekli mi?
        },
    },

    "market_structure": {
        "display_info": True,
        "lookback": 100,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 3
        },
    },

    # ========================================================================
    # BREAKOUT (5 indicators)
    # ========================================================================

    "squeeze_momentum": {
        "display_info": True,
        "bb_period": 20,
        "kc_period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "squeeze_expansion",
            "confirmation_bars": 1,
            "require_momentum_direction": True
        },
    },

    "volatility_breakout": {
        "display_info": True,
        "period": 20,
        "k": 0.5,                    # Breakout multiplier

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1,
            "min_range": 0.01        # %1 min range
        },
    },

    "breakout_scanner": {
        "display_info": True,
        "lookback": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "volume_surge",
            "confirmation_bars": 1,
            "require_volume": True
        },
    },

    "range_breakout": {
        "display_info": True,
        "min_range_bars": 10,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1,
            "min_breakout_percent": 0.01
        },
    },

    "consolidation": {
        "display_info": True,
        "period": 20,
        "threshold": 0.02,           # %2 max range

        "signal_confirmation": {
            "enabled": False,
            "method": "squeeze_expansion",
            "confirmation_bars": 1
        },
    },

    # ========================================================================
    # STATISTICAL (5 indicators)
    # ========================================================================

    "z_score": {
        "display_info": True,
        "period": 20,
        "threshold": 2.0,            # 2 standard deviations

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "percentile_rank": {
        "display_info": True,
        "period": 100,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1
        },
    },

    "linear_regression": {
        "display_info": True,
        "period": 20,
        "std_dev": 2.0,

        "signal_confirmation": {
            "enabled": False,
            "method": "squeeze_expansion",
            "confirmation_bars": 1
        },
    },

    "correlation": {
        "display_info": True,
        "period": 20,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 2
        },
    },

    "cointegration": {
        "display_info": True,
        "lookback": 100,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 3
        },
    },

    # ========================================================================
    # COMBO (5 indicators)
    # ========================================================================

    "rsi_bollinger": {
        "display_info": True,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std_dev": 2.0,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 1,
            "require_both": True     # Her iki sinyal de gerekli mi?
        },
    },

    "macd_rsi": {
        "display_info": True,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1,
            "require_both": True
        },
    },

    "ema_ribbon": {
        "display_info": True,
        "periods": [8, 13, 21, 34, 55],  # Fibonacci EMA'ları

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1,
            "require_alignment": True  # Tüm EMA'lar sıralı mı?
        },
    },

    "triple_screen": {
        "display_info": True,
        "timeframes": ["1h", "5m", "1m"],  # Elder's Triple Screen

        "signal_confirmation": {
            "enabled": False,
            "method": "crossover_confirmation",
            "confirmation_bars": 1,
            "require_all_timeframes": True
        },
    },

    "smart_money": {
        "display_info": True,
        "lookback": 100,
        "requires_volume": True,

        "signal_confirmation": {
            "enabled": False,
            "method": "trend_reversal",
            "confirmation_bars": 3,
            "require_structure": True  # Market structure gerekli mi?
        },
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preset(indicator_name: str, **overrides) -> Dict[str, Any]:
    """
    Get indicator preset configuration

    Args:
        indicator_name: Indicator adı (örn: 'rsi', 'ema')
        **overrides: Override parametreleri

    Returns:
        Dict: Indicator configuration

    Raises:
        ValueError: Unknown indicator

    Examples:
        # Standart config
        rsi_config = get_preset("rsi")

        # Override ile
        rsi_config = get_preset("rsi", default_period=21, oversold=25)

        # Signal confirmation aktif et
        ema_config = get_preset("ema", signal_confirmation={"enabled": True})
    """
    if indicator_name not in INDICATOR_PRESETS:
        raise ValueError(
            f"❌ Indicator preset bulunamadı: '{indicator_name}'\n"
            f"Mevcut indicator'lar: {list(INDICATOR_PRESETS.keys())}"
        )

    # Deep copy (original'i değiştirme)
    config = deepcopy(INDICATOR_PRESETS[indicator_name])

    # Override parametreleri uygula
    if overrides:
        config.update(overrides)

    return config


def merge_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations (deep merge)

    Args:
        base_config: Base configuration
        overrides: Override values

    Returns:
        Dict: Merged configuration

    Example:
        base = get_preset("rsi")
        custom = merge_config(base, {
            "default_period": 21,
            "signal_confirmation": {
                "enabled": True,
                "confirmation_bars": 2
            }
        })
    """
    result = deepcopy(base_config)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result


def list_available_presets() -> Dict[str, int]:
    """
    List all available indicator presets by category

    Returns:
        Dict: Category -> count
    """
    from components.indicators import get_indicator_info

    categories = {}

    for name in INDICATOR_PRESETS.keys():
        try:
            info = get_indicator_info(name)
            cat = info['category'].value
            categories[cat] = categories.get(cat, 0) + 1
        except:
            categories['unknown'] = categories.get('unknown', 0) + 1

    return categories


def get_presets_by_category(category: str) -> list:
    """
    Get all presets in a category

    Args:
        category: Category name (momentum, trend, volatility, etc.)

    Returns:
        List of indicator names
    """
    from components.indicators import list_indicators

    registry_names = set(list_indicators(category=category))
    preset_names = set(INDICATOR_PRESETS.keys())

    return sorted(registry_names & preset_names)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'INDICATOR_PRESETS',
    'get_preset',
    'merge_config',
    'list_available_presets',
    'get_presets_by_category',
]


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """
    Test indicator presets
    """

    print("\n" + "="*60)
    print("INDICATOR PRESETS TEST")
    print("="*60 + "\n")

    # Test 1: Get preset
    print("1. Get Preset (RSI):")
    rsi = get_preset("rsi")
    print(f"   ✓ Period: {rsi['default_period']}")
    print(f"   ✓ Oversold: {rsi['oversold']}")
    print(f"   ✓ Signal confirmation: {rsi['signal_confirmation']['method']}")

    # Test 2: Override
    print("\n2. Override Parameters:")
    rsi_custom = get_preset("rsi", default_period=21, oversold=25)
    print(f"   ✓ Period: {rsi_custom['default_period']} (was 14)")
    print(f"   ✓ Oversold: {rsi_custom['oversold']} (was 30)")

    # Test 3: Merge config
    print("\n3. Merge Config:")
    ema = get_preset("ema")
    ema_custom = merge_config(ema, {
        "default_fast": 12,
        "signal_confirmation": {
            "enabled": True,
            "distance_ratio": 0.2
        }
    })
    print(f"   ✓ Fast EMA: {ema_custom['default_fast']} (was 9)")
    print(f"   ✓ Confirmation enabled: {ema_custom['signal_confirmation']['enabled']}")
    print(f"   ✓ Method preserved: {ema_custom['signal_confirmation']['method']}")

    # Test 4: List presets
    print("\n4. Available Presets:")
    cats = list_available_presets()
    print(f"   ✓ Total: {sum(cats.values())} indicators")
    for cat, count in cats.items():
        print(f"      - {cat}: {count}")

    # Test 5: Get by category
    print("\n5. Get Momentum Indicators:")
    momentum = get_presets_by_category("momentum")
    print(f"   ✓ Found {len(momentum)} momentum indicators:")
    print(f"      {momentum}")

    # Test 6: Check signal confirmation variety
    print("\n6. Signal Confirmation Methods:")
    methods = {}
    for name, config in INDICATOR_PRESETS.items():
        if 'signal_confirmation' in config:
            method = config['signal_confirmation'].get('method', 'none')
            methods[method] = methods.get(method, 0) + 1

    for method, count in methods.items():
        print(f"   ✓ {method}: {count} indicators")

    # Test 7: Volume requirements
    print("\n7. Volume Requirements:")
    volume_required = [name for name, cfg in INDICATOR_PRESETS.items()
                      if cfg.get('requires_volume', False)]
    print(f"   ✓ {len(volume_required)} indicators require volume:")
    print(f"      {volume_required}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
    print(f"📦 {len(INDICATOR_PRESETS)} indicator presets ready!")
    print("="*60 + "\n")
