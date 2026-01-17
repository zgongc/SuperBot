"""
indicators/community - TradingView Community Indicators

Version: 1.0.0
Date: 2025-12-12

Description:
    TradingView'dan port edilen topluluk indikatörleri.

    Bu klasördeki indikatörler:
    - Orijinal PineScript kodundan Python'a çevrilmiştir
    - SuperBot indicator sistemine entegre edilmiştir
    - calculate(), calculate_batch() ve update() methodları uyumludur

Available Indicators:
    - MavilimW: Kivanc Ozbilgic'in WMA tabanlı trend indikatörü
"""

from .mavilimw import MavilimW

__all__ = [
    'MavilimW',
]
