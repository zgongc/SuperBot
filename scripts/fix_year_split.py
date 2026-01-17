#!/usr/bin/env python3
"""
fix_year_split.py
SuperBot - Year-based File Split Fixer
Yazar: SuperBot Team
Tarih: 2025-01-06
Versiyon: 2.0.0

Yıl bazlı dosyaları temizler ve yıl sınırlarını zorlar.

Sorun:
- 2024 dosyası 2025 verilerini içeriyor (UTC+3 timezone hatası)
- 2025 dosyası 2024 verilerini içeriyor
- Duplicate veriler var
- Year boundaries karışık

Çözüm:
- Her dosyayı oku
- Epoch time bazında yıl sınırlarını kontrol et (timezone safe)
- Sadece ilgili yıla ait verileri tut
- Duplicate'leri temizle
- Düzgün kaydet

Kullanım:
    python fix_year_split.py

Bağımlılıklar:
    - python>=3.10
    - pandas>=2.0
    - pyarrow>=10.0
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import sys


def fix_year_split(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1m',
    data_dir: str = 'data/parquets'
):
    """
    Fix year-based file split

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        data_dir: Data directory (base dir, will use {data_dir}/{symbol}/)
    """
    print("=" * 70)
    print(f"🔧 YEAR-SPLIT FIXER: {symbol} {timeframe}")
    print("=" * 70)

    # Yeni format: data/parquets/{symbol}/
    data_path = Path(data_dir) / symbol

    if not data_path.exists():
        print(f"❌ Sembol dizini bulunamadı: {data_path}")
        return

    # Find all year files
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = sorted(data_path.glob(pattern))

    if not files:
        print(f"❌ No files found matching: {pattern}")
        return

    print(f"\n📂 Found {len(files)} file(s):")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)

    # Process each file
    for filepath in files:
        # Extract year from filename: BTCUSDT_1m_2024.parquet
        year_str = filepath.stem.split('_')[-1]

        try:
            year = int(year_str)
        except ValueError:
            print(f"⚠️  Skipping {filepath.name} - invalid year: {year_str}")
            continue

        print(f"\n📅 Processing: {filepath.name} (Year {year})")
        print("-" * 70)

        # 1. Load file
        df = pd.read_parquet(filepath)
        initial_rows = len(df)
        print(f"   Initial: {initial_rows:,} rows")

        # Get time column
        time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'

        # Show date range
        date_min = df[time_col].min()
        date_max = df[time_col].max()
        print(f"   📅 Tarih aralığı: {date_min} -> {date_max}")

        # 2. Epoch time bazında yıl filtresi (timezone-safe)
        # Binance epoch time kullanıyor, timezone sorunu yok
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

        # Epoch time bazında yıl sınırları (UTC)
        # 2024-01-01 00:00:00 UTC = 1704067200000 ms epoch
        # 2024-12-31 23:59:59 UTC = 1735689599000 ms epoch
        year_start_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        year_end_utc = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Convert to pandas Timestamp for comparison
        year_start_ts = pd.Timestamp(year_start_utc)
        year_end_ts = pd.Timestamp(year_end_utc)

        # Filter: only keep data within this year (UTC epoch time bazında)
        df_year = df[
            (df[time_col] >= year_start_ts) &
            (df[time_col] <= year_end_ts)
        ].copy()

        year_rows = len(df_year)

        rows_removed = initial_rows - year_rows
        if rows_removed > 0:
            print(f"   🧹 Filtrelendi: {rows_removed:,} satır diğer yıllardan silindi")
        else:
            print(f"   ✅ Tüm satırlar {year} yılına ait")

        # 3. Duplicate'leri temizle
        df_year = df_year.drop_duplicates(subset=[time_col], keep='last')
        df_year = df_year.sort_values(time_col).reset_index(drop=True)

        after_dedup = len(df_year)
        duplicates = year_rows - after_dedup

        if duplicates > 0:
            print(f"   🧹 Temizlendi: {duplicates:,} duplicate satır")

        # 4. Beklenen tarih aralığını kontrol et (UTC epoch bazında)
        expected_start = year_start_ts
        expected_end = pd.Timestamp(datetime(year, 12, 31, 23, 59, 0, tzinfo=timezone.utc))

        actual_start = df_year[time_col].min()
        actual_end = df_year[time_col].max()

        print(f"\n   Beklenen (UTC): {expected_start} -> {expected_end}")
        print(f"   Gerçek (UTC):   {actual_start} -> {actual_end}")

        # Eksik veri kontrolü (başlangıç çok geç veya son çok erken)
        needs_download = False

        if actual_start > expected_start:
            missing_days = (actual_start - expected_start).days
            if missing_days > 0:
                print(f"   ⚠️  Eksik: Başlangıçta {missing_days} gün eksik")
                needs_download = True

        # Sadece geçmiş yıllar için son tarih kontrolü (current year için değil)
        current_year_utc = datetime.now(timezone.utc).year
        if year < current_year_utc and actual_end < expected_end:
            missing_days = (expected_end - actual_end).days
            if missing_days > 0:
                print(f"   ⚠️  Eksik: Sonda {missing_days} gün eksik")
                needs_download = True

        if needs_download:
            print(f"   💡 İpucu: Eksik aralıkları doldurmak için data_downloader çalıştır")

        # 5. Temizlenmiş dosyayı kaydet
        print(f"\n   💾 Temizlenmiş dosya kaydediliyor...")
        df_year.to_parquet(filepath, index=False, compression='snappy')

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   ✅ Kaydedildi: {after_dedup:,} satır ({size_mb:.2f} MB)")

        # Memory cleanup
        del df
        del df_year

    print("\n" + "=" * 70)
    print("✅ YIL-SPLIT DÜZELTMESİ TAMAMLANDI!")
    print("=" * 70)

    # Final summary
    print("\n📊 Son Dosya Özeti:")
    for filepath in files:
        df_check = pd.read_parquet(filepath)
        time_col = 'open_time' if 'open_time' in df_check.columns else 'timestamp'
        date_min = df_check[time_col].min()
        date_max = df_check[time_col].max()
        size_mb = filepath.stat().st_size / (1024 * 1024)

        print(f"\n   {filepath.name}:")
        print(f"      Satır: {len(df_check):,}")
        print(f"      Aralık: {date_min} -> {date_max}")
        print(f"      Boyut: {size_mb:.2f} MB")

        del df_check

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Windows UTF-8 fix (emoji display için)
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            # Fallback for older Python or IDLE
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    fix_year_split(
        symbol='BTCUSDT',
        timeframe='1m',
        data_dir='data/parquets'
    )
