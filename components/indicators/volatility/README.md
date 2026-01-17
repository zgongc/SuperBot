# Volatility Indicators

Volatilite indikatörleri - Fiyat hareketlerinin genişliğini ve dalgalanmasını ölçen indikatörler.

## İndikatörler

| İndikatör | Dosya | Tip | Açıklama |
|-----------|-------|-----|----------|
| **ATR** | `atr.py` | SINGLE_VALUE | Average True Range - Ortalama gerçek aralık |
| **Bollinger Bands** | `bollinger.py` | BANDS | SMA + StdDev volatilite bantları |
| **Keltner Channel** | `keltner.py` | BANDS | EMA + ATR volatilite kanalları |
| **Standard Deviation** | `standard_dev.py` | SINGLE_VALUE | Standart sapma |
| **True Range** | `true_range.py` | SINGLE_VALUE | Gerçek fiyat aralığı (tek mum) |
| **NATR** | `natr.py` | SINGLE_VALUE | Normalleştirilmiş ATR (yüzde) |
| **Chandelier Exit** | `chandelier.py` | BANDS | ATR tabanlı trailing stop |
| **TTM Squeeze** | `squeeze.py` | SINGLE_VALUE | Volatilite sıkışması göstergesi |

**Toplam:** 8 indikatör

## Hızlı Kullanım

```python
from indicators.volatility import (
    ATR,
    BollingerBands,
    KeltnerChannel,
    StandardDeviation,
    TrueRange,
    NATR,
    ChandelierExit,
    TTMSqueeze
)

# 1. ATR - Volatilite ölçümü
atr = ATR(period=14)
result = atr(data)
print(f"ATR: {result.value}, Volatilite: {result.metadata['volatility_pct']:.2f}%")

# 2. Bollinger Bands - Aşırı alım/satım
bb = BollingerBands(period=20, std_dev=2.0)
result = bb(data)
print(f"BB: Upper={result.value['upper']:.2f}, Lower={result.value['lower']:.2f}")
print(f"%B: {result.metadata['percent_b']:.4f}")

# 3. Keltner Channel - ATR tabanlı bantlar
kc = KeltnerChannel(ema_period=20, atr_period=10, multiplier=2.0)
result = kc(data)
print(f"KC: Upper={result.value['upper']:.2f}, Lower={result.value['lower']:.2f}")

# 4. Standard Deviation - Volatilite ölçümü
std = StandardDeviation(period=20)
result = std(data)
print(f"StdDev: {result.value:.4f}, CV: {result.metadata['cv']:.2f}%")

# 5. True Range - Tek mum volatilite
tr = TrueRange()
result = tr(data)
print(f"TR: {result.value:.4f}, Gap: {result.metadata['gap_direction']}")

# 6. NATR - Normalize volatilite
natr = NATR(period=14)
result = natr(data)
print(f"NATR: {result.value:.2f}%")

# 7. Chandelier Exit - Trailing stop
ce = ChandelierExit(period=22, multiplier=3.0)
result = ce(data)
print(f"Long Stop: {result.value['long_stop']:.2f}")
print(f"Short Stop: {result.value['short_stop']:.2f}")

# 8. TTM Squeeze - Volatilite sıkışması
squeeze = TTMSqueeze()
result = squeeze(data)
print(f"Momentum: {result.value:.4f}")
print(f"Squeeze Aktif: {result.metadata['squeeze_on']}")
```

## Kullanım Senaryoları

### 1. Risk Yönetimi
```python
# ATR ile position sizing
atr = ATR(period=14)
result = atr(data)

risk_amount = 1000  # $1000 risk
stop_distance = result.value * 2  # 2 ATR stop
position_size = risk_amount / stop_distance
```

### 2. Aşırı Alım/Satım Tespiti
```python
# Bollinger Bands ile
bb = BollingerBands(period=20, std_dev=2.0)
result = bb(data)

if result.metadata['percent_b'] > 1:
    print("Aşırı alım - üst bantta")
elif result.metadata['percent_b'] < 0:
    print("Aşırı satım - alt bantta")
```

### 3. Breakout Beklentisi
```python
# TTM Squeeze ile
squeeze = TTMSqueeze()
result = squeeze(data)

if result.metadata['squeeze_on']:
    print("Squeeze aktif - büyük hareket yaklaşıyor")
elif result.value > 0:
    print("Squeeze bitti - yukarı yönlü hareket")
else:
    print("Squeeze bitti - aşağı yönlü hareket")
```

### 4. Trailing Stop-Loss
```python
# Chandelier Exit ile
ce = ChandelierExit(period=22, multiplier=3.0)
result = ce(data)

# Long pozisyon için
if data['close'].iloc[-1] < result.value['long_stop']:
    print("Long pozisyondan çık!")

# Short pozisyon için
if data['close'].iloc[-1] > result.value['short_stop']:
    print("Short pozisyondan çık!")
```

## Özellikler

✓ **Türkçe Commentler:** Tüm kod Türkçe açıklamalarla
✓ **CCI Template:** CCI template yapısını takip eder
✓ **Test Blokları:** Her dosyada kapsamlı test kodu
✓ **Type Safety:** Doğru IndicatorType kullanımı
✓ **Metadata:** Detaylı metadata bilgileri
✓ **Export:** __all__ ile düzgün export

## Test

```bash
# Tüm volatility indikatörlerini test et
python test_volatility_indicators.py

# Tek bir indikatörü test et
cd indicators/volatility
python -c "import sys; sys.path.insert(0, 'd:/Python/SuperBot'); exec(open('atr.py').read())"
```

## Dokümantasyon

Detaylı dokümantasyon için:
- `docs/volatility_indicators.md` - Tam dokümantasyon

## Versiyon

- **Version:** 2.0.0
- **Date:** 2025-10-14
- **Author:** SuperBot Team
