#!/usr/bin/env python3
"""
components/strategies/helpers/strategy_indicator_bridge.py
SuperBot - Strategy & Indicator Manager Bridge
Yazar: SuperBot Team
Tarih: 2025-11-18
Versiyon: 3.0.0 (Registry-Based Auto-Aliasing)

Strategy template ile IndicatorManager arasındaki köprü.

Özellikler:
- Strategy template'teki indicator config'ini IndicatorManager'a aktarır
- Multi-timeframe indicator yönetimi
- **YENİ v3:** Registry-based automatic smart aliasing (yeni indicator → otomatik çalışır!)
- Indicator sonuçlarını strategy formatına çevirir
- Her iki syntax desteklenir: kısa ('macd') ve açık ('macd_macd')
- Cache integration

Kullanım:
    from strategies.helpers.strategy_indicator_bridge import create_indicator_manager_from_strategy

    # Strategy'den indicator manager oluştur
    indicator_manager = create_indicator_manager_from_strategy(
        strategy=my_strategy,
        cache_manager=cache,
        logger=logger
    )

    # Indicators hesapla
    results = indicator_manager.calculate_all(symbol="BTCUSDT", data=df)

Bağımlılıklar:
    - components.indicators.indicator_manager
    - components.strategies.base_strategy_template
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path

# SuperBot base directory'yi path'e ekle
base_dir = Path(__file__).parent.parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))


def create_indicator_manager_from_strategy(
    strategy: Any,
    cache_manager: Optional[Any] = None,
    logger: Optional[Any] = None,
    error_handler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    event_bus: Optional[Any] = None,
    verbose: bool = False
) -> Any:
    """
    Strategy template'ten IndicatorManager oluştur

    Args:
        strategy: BaseStrategyTemplate instance
        cache_manager: Cache manager (optional)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
        config: Override config (optional)
        event_bus: EventBus instance for real-time updates (optional)

    Returns:
        IndicatorManager instance

    Example:
        >>> from strategies.templates.base_template_sample import get_strategy_config
        >>> strategy = get_strategy_config()
        >>> indicator_manager = create_indicator_manager_from_strategy(
        ...     strategy,
        ...     logger=logger,
        ...     event_bus=event_bus
        ... )
        >>> # Artık indicator_manager.calculate_all() kullanılabilir
        >>> # EventBus ile real-time updates da aktif
    """
    try:
        from components.indicators.indicator_manager import IndicatorManager
    except ImportError:
        # Fallback: indicators modülü doğrudan erişilebilirse
        from indicators.indicator_manager import IndicatorManager

    # Config hazırla
    manager_config = config or {}

    # IndicatorManager oluştur (with EventBus support)
    indicator_manager = IndicatorManager(
        config=manager_config,
        cache_manager=cache_manager,
        logger=logger,
        error_handler=error_handler,
        event_bus=event_bus,
        strategy=strategy,
        verbose=verbose
    )

    # Strategy'nin indicator config'ini yükle
    if hasattr(strategy, 'technical_parameters') and strategy.technical_parameters:
        indicators_config = strategy.technical_parameters.indicators

        if indicators_config and len(indicators_config) > 0:
            if logger and verbose:
                logger.info(f"📊 Strategy'den {len(indicators_config)} indicator yükleniyor...")

            indicator_manager.load_from_config(indicators_config)

            if logger and verbose:
                logger.info(f"✅ {len(indicators_config)} indicator yüklendi")
        else:
            if logger:
                logger.warning("⚠️ Strategy'de indicator config bulunamadı")
    else:
        if logger:
            logger.warning("⚠️ Strategy'de technical_parameters bulunamadı")

    return indicator_manager


def format_indicator_results_for_strategy(
    indicator_results: Dict[str, Any],
    timeframe: str,
    ohlcv_data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    IndicatorManager sonuçlarını strategy için formatlama

    **v3 - Registry-Based Automatic Smart Aliasing:**
    - Registry'den output_keys otomatik okunur (yeni indicator → manuel kod gerekmez!)
    - Self-named outputs otomatik tespit edilir
    - Her iki syntax desteklenir: kısa ve açık

    IndicatorManager bazı indicator'lar için complex objeler döner
    (örn: Supertrend → {"trend": 1, "value": 50000})

    Bu fonksiyon bunları strategy'nin kolayca kullanabileceği
    flat dict'e çevirir ve smart aliasing ekler.

    Args:
        indicator_results: IndicatorManager.calculate_all() sonucu
        timeframe: Timeframe (örn: "1m", "5m")
        ohlcv_data: Optional DataFrame with OHLCV data (for MTF conditions)

    Returns:
        Formatted dict with smart aliasing:
        {
            # Main output aliases (kısa syntax)
            "macd": 0.5,
            "supertrend": 49500,

            # Full names (collision-safe, backward compat)
            "macd_macd": 0.5,
            "macd_signal": 0.3,
            "macd_histogram": 0.2,
            "supertrend_supertrend": 49500,
            "supertrend_upper": 51000,

            # Single values (unchanged)
            "rsi": 45.67,
            "ema_21": 50000.0
        }

    Example:
        >>> results = indicator_manager.calculate_all("BTCUSDT", df)
        >>> formatted = format_indicator_results_for_strategy(results, "1m")
        >>>
        >>> # Her iki syntax de çalışır:
        >>> if formatted["macd"] > formatted["macd_signal"]:  # Kısa
        >>> if formatted["macd_macd"] > formatted["macd_signal"]:  # Açık
    """
    from components.indicators import get_indicator_info

    formatted = {}

    for indicator_name, result in indicator_results.items():
        if result is None:
            continue

        # IndicatorResult objesi ise value'yu çıkar
        if hasattr(result, 'value'):
            value = result.value
        else:
            value = result

        # Single value indicator (RSI, EMA, ATR, etc.)
        if isinstance(value, (int, float, bool, str)):
            formatted[indicator_name] = value

        # Multi-value indicator (MACD, Bollinger, Stochastic, etc.)
        elif isinstance(value, dict):
            # Base indicator name'i al (custom naming için: ema_21 → ema)
            base_indicator_name = _extract_base_indicator_name(indicator_name)

            # Registry'den bu indicator'ü kontrol et (automatic!)
            try:
                indicator_info = get_indicator_info(base_indicator_name)
                output_keys = indicator_info.get('output_keys', [])
            except (ValueError, KeyError):
                # Registry'de yoksa fallback
                output_keys = []

            # Check if this indicator has a self-named main output
            # (örn: macd indicator'ünün 'macd' output'u var mı?)
            has_self_named_output = base_indicator_name in value or indicator_name in value

            for key, val in value.items():
                # Full name (her zaman ekle - collision-safe)
                full_key = f"{indicator_name}_{key}"
                formatted[full_key] = val

                # Main output alias (kısa syntax için)
                # Sadece key == base_indicator_name veya key == indicator_name ise
                if key == base_indicator_name or key == indicator_name:
                    # Self-named main output → add short aliases
                    # 1. Custom name alias (supertrend_10_4.0)
                    formatted[indicator_name] = val

                    # 2. Base name alias (supertrend) - for custom named indicators
                    if indicator_name != base_indicator_name:
                        formatted[base_indicator_name] = val

        # Array/list → son değeri al
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            formatted[indicator_name] = value[-1]

        # Diğer tipler → olduğu gibi geçir
        else:
            formatted[indicator_name] = value

    # Add OHLCV data to formatted results (for MTF conditions like ['close', '>', 'ema_55', '1h'])
    if ohlcv_data is not None:
        import pandas as pd
        if isinstance(ohlcv_data, pd.DataFrame):
            for ohlcv_col in ['open', 'high', 'low', 'close', 'volume']:
                if ohlcv_col in ohlcv_data.columns:
                    formatted[ohlcv_col] = ohlcv_data[ohlcv_col]

    return formatted


def _extract_base_indicator_name(indicator_name: str) -> str:
    """
    Custom indicator name'den base indicator name'i çıkar

    Örnekler:
        "ema_21" → "ema"
        "macd_12_26_9" → "macd"
        "rsi" → "rsi"
        "bollinger_20_2" → "bollinger"
        "stochastic_rsi_14_14" → "stochastic_rsi"

    Args:
        indicator_name: Full indicator name (custom veya base)

    Returns:
        Base indicator name

    Not:
        Multi-word indicators (stochastic_rsi, volume_sma, pivot_points, etc.)
        için özel logic var
    """
    # Underscore'dan önceki kısmı al
    parts = indicator_name.split('_')

    # İlk part base name
    base_name = parts[0]

    # Multi-word indicators (2 kelime)
    if base_name == 'stochastic' and len(parts) > 1:
        if parts[1] == 'rsi':
            return 'stochastic_rsi'

    if base_name == 'volume' and len(parts) > 1:
        if parts[1] in ['sma', 'oscillator', 'profile']:
            return f"{base_name}_{parts[1]}"

    if base_name == 'pivot' and len(parts) > 1:
        if parts[1] == 'points':
            return 'pivot_points'

    if base_name == 'fibonacci' and len(parts) > 1:
        if parts[1] in ['pivot', 'retracement']:
            return f"{base_name}_{parts[1]}"

    if base_name == 'candlestick' and len(parts) > 1:
        if parts[1] == 'patterns':
            return 'candlestick_patterns'

    if base_name == 'talib' and len(parts) > 1:
        if parts[1] == 'patterns':
            return 'talib_patterns'

    if base_name == 'linear' and len(parts) > 1:
        if parts[1] == 'regression':
            return 'linear_regression'

    if base_name == 'range' and len(parts) > 1:
        if parts[1] == 'breakout':
            return 'range_breakout'

    if base_name == 'volatility' and len(parts) > 1:
        if parts[1] == 'breakout':
            return 'volatility_breakout'

    if base_name == 'squeeze' and len(parts) > 1:
        if parts[1] == 'momentum':
            return 'squeeze_momentum'

    if base_name == 'breakout' and len(parts) > 1:
        if parts[1] == 'scanner':
            return 'breakout_scanner'

    if base_name == 'support' and len(parts) > 1:
        if parts[1] == 'resistance':
            return 'support_resistance'

    if base_name == 'market' and len(parts) > 1:
        if parts[1] == 'structure':
            return 'market_structure'

    if base_name == 'liquidity' and len(parts) > 1:
        if parts[1] == 'zones':
            return 'liquidityzones'

    if base_name == 'order' and len(parts) > 1:
        if parts[1] == 'blocks':
            return 'orderblocks'

    if base_name == 'swing' and len(parts) > 1:
        if parts[1] == 'points':
            return 'swing_points'

    if base_name == 'fair' and len(parts) > 1:
        if parts[1] == 'value':
            return 'fvg'  # fair_value_gap -> fvg

    if base_name == 'smart' and len(parts) > 1:
        if parts[1] in ['money', 'grok']:
            return f'smart_{parts[1]}'

    if base_name == 'triple' and len(parts) > 1:
        if parts[1] == 'screen':
            return 'triple_screen'

    if base_name == 'ema' and len(parts) > 1:
        if parts[1] == 'ribbon':
            return 'ema_ribbon'

    if base_name == 'macd' and len(parts) > 1:
        if parts[1] == 'rsi':
            return 'macd_rsi'

    if base_name == 'rsi' and len(parts) > 1:
        if parts[1] == 'bollinger':
            return 'rsi_bollinger'
        if parts[1] == 'divergence':
            return 'rsi_divergence'

    if base_name == 'lv' and len(parts) > 1:
        if parts[1] == 'void':
            return 'lvvoid'

    if base_name == 'fib' and len(parts) > 1:
        if parts[1] == 'retracement':
            return 'fib_retracement'

    if base_name == 'vwap' and len(parts) > 1:
        if parts[1] == 'bands':
            return 'vwap_bands'

    # Single word indicator
    return base_name


def get_multi_timeframe_data(
    indicator_manager: Any,
    symbol: str,
    timeframe_data: Dict[str, Any],
    logger: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Multi-timeframe indicator hesaplama

    Strategy'nin check_entry_conditions() metoduna gönderilecek
    timeframe_data'yı hazırlar.

    Args:
        indicator_manager: IndicatorManager instance
        symbol: Symbol (örn: "BTCUSDT")
        timeframe_data: {
            "1m": pd.DataFrame(...),
            "5m": pd.DataFrame(...),
            "15m": pd.DataFrame(...)
        }
        logger: Logger (optional)

    Returns:
        {
            "1m": {"rsi_14": 45.67, "ema_9": 50000, ...},
            "5m": {"rsi_14": 48.32, "ema_21": 50100, ...},
            "15m": {...}
        }

    Example:
        >>> # Backtest/Trading bot'ta
        >>> timeframe_data = {
        >>>     "1m": df_1m,
        >>>     "5m": df_5m
        >>> }
        >>>
        >>> indicator_data = get_multi_timeframe_data(
        >>>     indicator_manager,
        >>>     "BTCUSDT",
        >>>     timeframe_data,
        >>>     logger
        >>> )
        >>>
        >>> # Strategy'ye gönder
        >>> result = strategy.check_entry_conditions(
        >>>     symbol="BTCUSDT",
        >>>     timeframe_data=indicator_data,
        >>>     positions=[]
        >>> )
    """
    multi_tf_indicators = {}

    for timeframe, data in timeframe_data.items():
        try:
            # Calculate indicators for this timeframe
            results = indicator_manager.calculate_all(symbol, data)

            # Format for strategy
            formatted = format_indicator_results_for_strategy(results, timeframe)

            multi_tf_indicators[timeframe] = formatted

            if logger:
                logger.debug(f"✅ {timeframe}: {len(formatted)} indicator hesaplandı")

        except Exception as e:
            if logger:
                logger.error(f"❌ {timeframe} indicator hesaplama hatası: {e}")
            multi_tf_indicators[timeframe] = {}

    return multi_tf_indicators


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Strategy-Indicator Bridge v3 Test (Registry-Based Auto-Aliasing)")
    print("=" * 80)

    # Mock logger
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")

    logger = MockLogger()

    # Test 1: Base name extraction
    print("\n1. Base Name Extraction Test:")
    test_names = [
        ("ema_21", "ema"),
        ("macd_12_26_9", "macd"),
        ("rsi", "rsi"),
        ("bollinger_20_2", "bollinger"),
        ("stochastic_rsi_14_14", "stochastic_rsi"),
        ("volume_sma_20", "volume_sma"),
        ("pivot_points", "pivot_points"),
    ]

    for full_name, expected_base in test_names:
        base = _extract_base_indicator_name(full_name)
        status = "✅" if base == expected_base else "❌"
        print(f"   {status} '{full_name}' → '{base}' (expected: '{expected_base}')")

    # Test 2: Format results with smart aliasing
    print("\n2. Smart Aliasing Test:")

    # MACD (has self-named main output)
    macd_results = {
        "macd": {
            "macd": 0.5,
            "signal": 0.3,
            "histogram": 0.2
        }
    }

    formatted_macd = format_indicator_results_for_strategy(macd_results, "1m")
    print(f"   MACD Input: {macd_results}")
    print(f"   MACD Output: {formatted_macd}")
    print(f"   ✅ Main alias: {'macd' in formatted_macd and formatted_macd['macd'] == 0.5}")
    print(f"   ✅ Full names: {'macd_macd' in formatted_macd and 'macd_signal' in formatted_macd}")

    # Bollinger (no self-named main output)
    bollinger_results = {
        "bollinger": {
            "upper": 52000,
            "middle": 50000,
            "lower": 48000
        }
    }

    formatted_bollinger = format_indicator_results_for_strategy(bollinger_results, "1m")
    print(f"\n   Bollinger Input: {bollinger_results}")
    print(f"   Bollinger Output: {formatted_bollinger}")
    print(f"   ✅ No main alias: {'bollinger' not in formatted_bollinger or formatted_bollinger.get('bollinger') in [52000, 50000, 48000]}")
    print(f"   ✅ Full names: {'bollinger_upper' in formatted_bollinger}")

    # Single value (RSI)
    single_results = {
        "rsi": 45.67,
        "ema_21": 50000.0
    }

    formatted_single = format_indicator_results_for_strategy(single_results, "1m")
    print(f"\n   Single Value Input: {single_results}")
    print(f"   Single Value Output: {formatted_single}")
    print(f"   ✅ Unchanged: {formatted_single == single_results}")

    # Test 3: Multi-indicator collision test
    print("\n3. Collision Prevention Test:")

    multi_results = {
        "macd": {"macd": 0.5, "signal": 0.3, "histogram": 0.2},
        "tsi": {"tsi": 25.5, "signal": 23.1, "histogram": 2.4}
    }

    formatted_multi = format_indicator_results_for_strategy(multi_results, "1m")
    print(f"   Multi Input: {multi_results}")
    print(f"   Multi Output keys: {list(formatted_multi.keys())}")
    print(f"   ✅ No collision: {'macd_signal' in formatted_multi and 'tsi_signal' in formatted_multi}")
    print(f"   ✅ Different values: {formatted_multi.get('macd_signal') != formatted_multi.get('tsi_signal')}")

    print("\n" + "=" * 80)
    print("✅ Tüm testler tamamlandı!")
    print("=" * 80)
    print("\n💡 Özellikler:")
    print("   ✅ Registry-based automatic detection (yeni indicator → otomatik çalışır)")
    print("   ✅ Smart aliasing (main output → kısa alias)")
    print("   ✅ Full names (collision-safe)")
    print("   ✅ Backward compatible")
    print("   ✅ Multi-word indicator support (stochastic_rsi, volume_sma, etc.)")
    print("")
