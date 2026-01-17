"""
indicators/statistical/__init__.py - Statistical Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    İstatistiksel analiz indikatörleri paketi

    İçindekiler:
        - ZScore: Fiyatın standart sapma cinsinden konumu
        - PercentileRank: Fiyatın yüzdelik dilimi (0-100)
        - LinearRegression: Doğrusal regresyon analizi (slope, r², forecast)
        - Correlation: İki varlık arasında korelasyon (-1 ile +1)
        - Cointegration: Eş-bütünleşme analizi (spread, zscore, pairs trading)

Kategoriler:
    - STATISTICAL: İstatistiksel analiz ve matematiksel modelleme

Kullanım:
    from indicators.statistical import ZScore, PercentileRank, LinearRegression
    from indicators.statistical import Correlation, Cointegration

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scipy>=1.10.0
    - statsmodels>=0.14.0 (opsiyonel, cointegration ADF testi için)
"""

from components.indicators.statistical.zscore import ZScore
from components.indicators.statistical.percentile_rank import PercentileRank
from components.indicators.statistical.linear_regression import LinearRegression
from components.indicators.statistical.correlation import Correlation
from components.indicators.statistical.cointegration import Cointegration

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # İstatistiksel İndikatörler
    'ZScore',
    'PercentileRank',
    'LinearRegression',
    'Correlation',
    'Cointegration',
]

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__category__ = 'statistical'

# İndikatör bilgileri
INDICATORS_INFO = {
    'ZScore': {
        'name': 'Z-Score',
        'type': 'SINGLE_VALUE',
        'description': 'Fiyatın standart sapma cinsinden konumu',
        'parameters': ['period', 'overbought', 'oversold'],
        'default_period': 20,
        'use_cases': ['Mean reversion', 'Aşırı alım/satım', 'Outlier detection']
    },
    'PercentileRank': {
        'name': 'Percentile Rank',
        'type': 'SINGLE_VALUE',
        'description': 'Fiyatın yüzdelik dilimi (0-100)',
        'parameters': ['period', 'overbought', 'oversold'],
        'default_period': 20,
        'use_cases': ['Göreceli güç', 'Aşırı alım/satım', 'Ranking']
    },
    'LinearRegression': {
        'name': 'Linear Regression',
        'type': 'MULTIPLE_VALUES',
        'description': 'Doğrusal regresyon analizi',
        'parameters': ['period', 'forecast_periods', 'min_r_squared'],
        'default_period': 20,
        'outputs': ['slope', 'intercept', 'r_squared', 'forecast'],
        'use_cases': ['Trend analizi', 'Fiyat tahmini', 'Trend gücü']
    },
    'Correlation': {
        'name': 'Correlation',
        'type': 'SINGLE_VALUE',
        'description': 'İki varlık arasında korelasyon',
        'parameters': ['period', 'reference_data', 'high_correlation', 'low_correlation'],
        'default_period': 20,
        'range': '(-1, 1)',
        'use_cases': ['Pairs trading', 'Portföy diversifikasyonu', 'Risk yönetimi']
    },
    'Cointegration': {
        'name': 'Cointegration',
        'type': 'MULTIPLE_VALUES',
        'description': 'Eş-bütünleşme analizi',
        'parameters': ['period', 'reference_data', 'entry_threshold', 'exit_threshold'],
        'default_period': 50,
        'outputs': ['spread', 'zscore', 'is_cointegrated'],
        'use_cases': ['Pairs trading', 'Statistical arbitrage', 'Mean reversion'],
        'requires': 'statsmodels (opsiyonel)'
    }
}


def get_indicator_info(indicator_name: str) -> dict:
    """
    İndikatör bilgilerini getir

    Args:
        indicator_name: İndikatör adı

    Returns:
        dict: İndikatör bilgileri
    """
    return INDICATORS_INFO.get(indicator_name, {})


def list_indicators() -> list:
    """
    Tüm mevcut indikatörleri listele

    Returns:
        list: İndikatör adları
    """
    return list(INDICATORS_INFO.keys())


# ============================================================================
# KULLANIM ÖRNEKLERİ
# ============================================================================

if __name__ == "__main__":
    """Statistical indicators package test"""

    print("\n" + "="*70)
    print("STATISTICAL INDICATORS PACKAGE TEST")
    print("="*70 + "\n")

    print("1. Paket Bilgileri:")
    print(f"   [OK] Versiyon: {__version__}")
    print(f"   [OK] Kategori: {__category__}")
    print(f"   [OK] Toplam indikatör: {len(__all__)}")

    print("\n2. Mevcut İndikatörler:")
    for idx, indicator_name in enumerate(__all__, 1):
        info = get_indicator_info(indicator_name)
        print(f"   [{idx}] {info.get('name', indicator_name)}")
        print(f"       - Tip: {info.get('type', 'N/A')}")
        print(f"       - Açıklama: {info.get('description', 'N/A')}")
        if 'outputs' in info:
            print(f"       - Çıktılar: {', '.join(info['outputs'])}")
        if 'range' in info:
            print(f"       - Aralık: {info['range']}")
        print(f"       - Kullanım: {', '.join(info.get('use_cases', []))}")
        if 'requires' in info:
            print(f"       - Gereksinim: {info['requires']}")

    print("\n3. Import Testleri:")
    import numpy as np
    import pandas as pd

    # Test verisi oluştur
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]
    prices = [100]
    for _ in range(99):
        prices.append(prices[-1] + np.random.randn() * 1.0 + 0.05)

    test_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    # ZScore testi
    print("\n   a) Z-Score:")
    zscore = ZScore(period=20)
    result = zscore(test_data)
    print(f"      [OK] Import başarılı: {zscore.name}")
    print(f"      [OK] Değer: {result.value}")
    print(f"      [OK] Sinyal: {result.signal.value}")

    # PercentileRank testi
    print("\n   b) Percentile Rank:")
    prank = PercentileRank(period=20)
    result = prank(test_data)
    print(f"      [OK] Import başarılı: {prank.name}")
    print(f"      [OK] Değer: {result.value}%")
    print(f"      [OK] Sinyal: {result.signal.value}")

    # LinearRegression testi
    print("\n   c) Linear Regression:")
    linreg = LinearRegression(period=20)
    result = linreg(test_data)
    print(f"      [OK] Import başarılı: {linreg.name}")
    print(f"      [OK] Slope: {result.value['slope']}")
    print(f"      [OK] R²: {result.value['r_squared']}")
    print(f"      [OK] Forecast: {result.value['forecast']:.2f}")

    # Correlation testi
    print("\n   d) Correlation:")
    corr = Correlation(period=20)
    result = corr(test_data)
    print(f"      [OK] Import başarılı: {corr.name}")
    print(f"      [OK] Autocorrelation: {result.value}")
    print(f"      [OK] İlişki: {result.metadata['relationship']}")

    # Cointegration testi
    print("\n   e) Cointegration:")
    # İkinci varlık oluştur
    prices2 = [110]
    common_trend = np.cumsum(np.random.randn(100) * 0.05)
    for i in range(99):
        prices2.append(110 + common_trend[i] + np.random.randn() * 0.5)

    test_data2 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices2,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices2],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices2],
        'close': prices2,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices2]
    })

    coint = Cointegration(period=50, reference_data=test_data2)
    result = coint(test_data)
    print(f"      [OK] Import başarılı: {coint.name}")
    print(f"      [OK] Spread: {result.value['spread']:.4f}")
    print(f"      [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"      [OK] Eş-bütünleşme: {result.value['is_cointegrated']}")

    print("\n4. İndikatör Listesi:")
    indicators = list_indicators()
    print(f"   [OK] Toplam: {len(indicators)} indikatör")
    for indicator in indicators:
        print(f"   [OK] {indicator}")

    print("\n" + "="*70)
    print("[BAŞARILI] TÜM PAKET TESTLERİ BAŞARILI!")
    print("="*70 + "\n")
