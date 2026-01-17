#!/usr/bin/env python3
"""
modules/backtest/backtest_config.py
SuperBot - Backtest Config Builder
Yazar: SuperBot Team
Tarih: 2025-11-16
Versiyon: 3.0.0

Strategy object'inden BacktestConfig oluşturur.

Özellikler:
- Strategy → BacktestConfig conversion
- Multi-timeframe support
- Multi-symbol support
- Validation & defaults

Kullanım:
    from modules.backtest.backtest_config import build_config

    config = build_config(strategy)

Bağımlılıklar:
    - python>=3.10
    - modules.backtest.backtest_types
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from datetime import datetime
from typing import TYPE_CHECKING

from modules.backtest.backtest_types import BacktestConfig

if TYPE_CHECKING:
    from components.strategies.base_strategy import Strategy


# ============================================================================
# CONFIG BUILDER
# ============================================================================

def build_config(strategy: Strategy) -> BacktestConfig:
    """
    Strategy object'inden BacktestConfig oluştur

    Args:
        strategy: Strategy instance

    Returns:
        BacktestConfig: Backtest konfigürasyonu

    Raises:
        ValueError: Geçersiz strategy parametreleri
    """
    # Backtest parametrelerini al (eski: dict, yeni: BacktestParameters object)
    backtest_params = strategy.backtest_parameters

    # Dict mi object mi kontrol et
    is_dict = isinstance(backtest_params, dict)

    # Sembol - strategy attribute'larından al
    if hasattr(strategy, 'symbols') and strategy.symbols:
        # Yeni format: symbols attribute var
        if isinstance(strategy.symbols, list):
            # Liste içinde SymbolConfig object'leri var
            symbols = []
            for sym_config in strategy.symbols:
                if hasattr(sym_config, 'symbol') and hasattr(sym_config, 'quote'):
                    # SymbolConfig object
                    for s in sym_config.symbol:
                        symbols.append(f"{s}{sym_config.quote}")
                else:
                    # Direkt string
                    symbols.append(str(sym_config))
        elif hasattr(strategy.symbols, 'symbol'):
            # Tek SymbolConfig object
            symbols = [f"{s}{strategy.symbols.quote}" for s in strategy.symbols.symbol]
        else:
            # Direkt string veya başka format
            symbols = [str(strategy.symbols)]
    else:
        # Eski format: backtest_parameters'dan al
        if is_dict:
            # backtest_parameters bir dict ise, symbol yok - default BTCUSDT
            symbols = ["BTCUSDT"]
        else:
            # BacktestParameters object
            symbols = [backtest_params.symbol]

    # Timeframe
    primary_timeframe = strategy.primary_timeframe if hasattr(strategy, 'primary_timeframe') else "15m"
    mtf_timeframes = strategy.mtf_timeframes if hasattr(strategy, 'mtf_timeframes') else [primary_timeframe]

    # Eğer mtf_timeframes boşsa, sadece primary kullan
    if not mtf_timeframes:
        mtf_timeframes = [primary_timeframe]

    # Tarih aralığı
    if hasattr(strategy, 'backtest_start_date'):
        # Yeni format
        start_date = _parse_date(strategy.backtest_start_date)
        end_date = _parse_date(strategy.backtest_end_date)
    elif is_dict:
        # Eski format - dict - default tarihler
        start_date = datetime(2025, 1, 5)
        end_date = datetime(2025, 2, 10)
    else:
        # BacktestParameters object
        start_date = _parse_date(backtest_params.start_date)
        end_date = _parse_date(backtest_params.end_date)

    # Portfolio
    initial_balance = strategy.initial_balance if hasattr(strategy, 'initial_balance') else 10000.0

    # Data loading
    warmup_period = strategy.warmup_period if hasattr(strategy, 'warmup_period') else 200

    # Maliyetler
    if is_dict:
        commission_pct = backtest_params.get('commission', 0.04)
        slippage_pct = backtest_params.get('max_slippage', 0.01)
        spread_pct = backtest_params.get('min_spread', 0.01)
    else:
        commission_pct = backtest_params.commission_pct if hasattr(backtest_params, 'commission_pct') else 0.04
        slippage_pct = backtest_params.slippage_pct if hasattr(backtest_params, 'slippage_pct') else 0.01
        spread_pct = backtest_params.spread_pct if hasattr(backtest_params, 'spread_pct') else 0.01

    # Config oluştur
    config = BacktestConfig(
        symbols=symbols,
        primary_timeframe=primary_timeframe,
        mtf_timeframes=mtf_timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        warmup_period=warmup_period,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        spread_pct=spread_pct,
        strategy_name=strategy.strategy_name,
        strategy_version=strategy.strategy_version,
    )

    return config


def _parse_date(date_str: str) -> datetime:
    """
    Tarih string'ini datetime'a çevir

    Desteklenen formatlar:
    - '2025-01-05T00:00' (ISO format with T)
    - '2025-01-05 00:00' (ISO format with space)
    - '2025-01-05'       (Date only, time 00:00)

    Args:
        date_str: Tarih string'i

    Returns:
        datetime: Parse edilmiş tarih

    Raises:
        ValueError: Geçersiz tarih formatı
    """
    if not date_str:
        raise ValueError("Tarih string'i boş olamaz")

    # Önce ISO format dene (T ile)
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    # Space ile dene
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    # Sadece tarih (time 00:00)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(
        f"Geçersiz tarih formatı: '{date_str}'. "
        f"Desteklenen formatlar: '2025-01-05T00:00', '2025-01-05 00:00', '2025-01-05'"
    )


def validate_config(config: BacktestConfig) -> None:
    """
    Config validation

    Args:
        config: BacktestConfig instance

    Raises:
        ValueError: Validation hatası
    """
    # Sembol kontrolü
    if not config.symbols:
        raise ValueError("En az 1 sembol belirtilmeli")

    # Timeframe kontrolü
    if not config.primary_timeframe:
        raise ValueError("Primary timeframe belirtilmeli")

    if config.primary_timeframe not in config.mtf_timeframes:
        raise ValueError(
            f"Primary timeframe ({config.primary_timeframe}) "
            f"mtf_timeframes içinde olmalı ({config.mtf_timeframes})"
        )

    # Tarih kontrolü
    if not config.start_date or not config.end_date:
        raise ValueError("start_date ve end_date belirtilmeli")

    if config.start_date >= config.end_date:
        raise ValueError(
            f"start_date ({config.start_date}) end_date'den ({config.end_date}) önce olmalı"
        )

    # Balance kontrolü
    if config.initial_balance <= 0:
        raise ValueError(f"initial_balance pozitif olmalı (mevcut: {config.initial_balance})")

    # Warmup kontrolü
    if config.warmup_period < 0:
        raise ValueError(f"warmup_period negatif olamaz (mevcut: {config.warmup_period})")

    # Maliyet kontrolü
    if config.commission_pct < 0 or config.commission_pct > 10:
        raise ValueError(
            f"commission_pct 0-10 arasında olmalı (mevcut: {config.commission_pct})"
        )

    if config.slippage_pct < 0 or config.slippage_pct > 10:
        raise ValueError(
            f"slippage_pct 0-10 arasında olmalı (mevcut: {config.slippage_pct})"
        )


def get_cache_key(config: BacktestConfig) -> str:
    """
    Config için cache key oluştur

    Aynı config parametreleri → aynı cache key
    Data caching için kullanılır.

    Args:
        config: BacktestConfig instance

    Returns:
        str: Cache key (örn: 'BTCUSDT_15m_20250105_20250210')
    """
    # Semboller (alfabetik sıralı)
    symbols_str = "_".join(sorted(config.symbols))

    # Timeframe'ler (alfabetik sıralı)
    timeframes_str = "_".join(sorted(config.mtf_timeframes))

    # Tarihler (YYYYMMDD formatında)
    start_str = config.start_date.strftime("%Y%m%d")
    end_str = config.end_date.strftime("%Y%m%d")

    # Cache key
    cache_key = f"{symbols_str}_{timeframes_str}_{start_str}_{end_str}"

    return cache_key


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Backtest Config Builder Test")
    print("=" * 60)

    # Mock Strategy class for testing
    class MockBacktestParams:
        def __init__(self):
            self.symbol = "BTCUSDT"
            self.timeframe = "15m"
            self.start_date = "2025-01-05T00:00"
            self.end_date = "2025-02-10T00:00"
            self.warmup_period = 200
            self.commission_pct = 0.04
            self.slippage_pct = 0.01

    class MockStrategy:
        def __init__(self):
            self.strategy_name = "TestStrategy"
            self.strategy_version = "1.0.0"
            self.initial_balance = 10000.0
            self.mtf_timeframes = ["15m", "1h", "4h"]
            self.backtest_parameters = MockBacktestParams()

    # Test 1: Config oluşturma
    print("\n📋 Test 1: Config oluşturma")
    strategy = MockStrategy()
    config = build_config(strategy)
    print(f"   ✅ Config oluşturuldu")
    print(f"   ✅ Sembol: {config.symbols[0]}")
    print(f"   ✅ Primary TF: {config.primary_timeframe}")
    print(f"   ✅ MTF TFs: {config.mtf_timeframes}")
    print(f"   ✅ Tarih: {config.start_date} → {config.end_date}")
    print(f"   ✅ Balance: ${config.initial_balance:,.0f}")

    # Test 2: Date parsing
    print("\n📅 Test 2: Date parsing")
    date1 = _parse_date("2025-01-05T00:00")
    print(f"   ✅ ISO format (T): {date1}")

    date2 = _parse_date("2025-01-05 00:00")
    print(f"   ✅ ISO format (space): {date2}")

    date3 = _parse_date("2025-01-05")
    print(f"   ✅ Date only: {date3}")

    # Test 3: Validation
    print("\n✅ Test 3: Validation")
    try:
        validate_config(config)
        print("   ✅ Config geçerli")
    except ValueError as e:
        print(f"   ❌ Validation hatası: {e}")

    # Test 4: Geçersiz config
    print("\n❌ Test 4: Geçersiz config")
    try:
        bad_config = BacktestConfig(
            symbols=[],  # Boş sembol listesi
            primary_timeframe="15m",
            mtf_timeframes=["15m"],
            initial_balance=10000,
        )
        print("   ❌ Config oluşturulmamalıydı!")
    except ValueError as e:
        print(f"   ✅ Beklenen hata yakalandı: {e}")

    # Test 5: Cache key
    print("\n🔑 Test 5: Cache key")
    cache_key = get_cache_key(config)
    print(f"   ✅ Cache key: {cache_key}")

    # İki farklı config, farklı cache key olmalı
    config2 = BacktestConfig(
        symbols=['ETHUSDT'],
        primary_timeframe='1h',
        mtf_timeframes=['1h'],
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 1),
        initial_balance=10000,
    )
    cache_key2 = get_cache_key(config2)
    print(f"   ✅ Cache key 2: {cache_key2}")
    print(f"   ✅ Farklı mı? {cache_key != cache_key2}")

    # Test 6: Config serialization
    print("\n💾 Test 6: Config serialization")
    config_dict = config.to_dict()
    print(f"   ✅ Config dict'e çevrildi")
    print(f"   ✅ Keys: {list(config_dict.keys())[:5]}...")

    print("\n✅ Tüm testler başarılı!")
    print("=" * 60)
