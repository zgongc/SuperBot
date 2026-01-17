# Community Indicators

TradingView'dan port edilen topluluk indikatörleri.

## MavilimW

**Kaynak:** Kivanc Ozbilgic (@mavilim0732)
**Tür:** Trend Following

### Açıklama

6 kademeli WMA (Weighted Moving Average) zincirleme hesaplaması yaparak gürültüyü filtreler ve temiz trend sinyalleri üretir. Fibonacci benzeri periyot artışı kullanır.

### Formül

```
fmal=3, smal=5 için:
M1 = WMA(close, 3)
M2 = WMA(M1, 5)
M3 = WMA(M2, 8)      # 3+5
M4 = WMA(M3, 13)     # 5+8
M5 = WMA(M4, 21)     # 8+13
MAVW = WMA(M5, 34)   # 13+21
```

### Parametreler

| Parametre | Default | Açıklama |
|-----------|---------|----------|
| `fmal` | 3 | First MA length |
| `smal` | 5 | Second MA length |

### Output

| Kolon | Tip | Açıklama |
|-------|-----|----------|
| `mavw` | float | MavilimW değeri (fiyat overlay) |
| `trend_direction` | int | 1=yukarı (mavi), -1=aşağı (kırmızı), 0=nötr |

### Kullanılan Dahili İndikatörler

- `indicators.trend.wma.WMA` - 6 kademeli WMA zinciri için

### Strateji Kullanımı

```python
# indicators
"mavilimw": {"fmal": 3, "smal": 5}

# entry_conditions
"long": [
    ['close', '>', 'mavilimw_mavw'],           # Fiyat MAVW üstünde
    ['mavilimw_trend_direction', '==', 1],     # Mavi (yükseliş)
],
"short": [
    ['close', '<', 'mavilimw_mavw'],           # Fiyat MAVW altında
    ['mavilimw_trend_direction', '==', -1],   # Kırmızı (düşüş)
],
```

### Sinyaller

- **Long:** Fiyat MAVW üstüne çıktığında ve MAVW yükseliyorken (mavi)
- **Short:** Fiyat MAVW altına indiğinde ve MAVW düşerken (kırmızı)
- **Exit Long:** `trend_direction` -1 olduğunda
- **Exit Short:** `trend_direction` 1 olduğunda
