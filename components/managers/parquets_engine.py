#!/usr/bin/env python3
"""
components/managers/parquets_engine.py
SuperBot - Parquet Historical Data Manager
Yazar: SuperBot Team
Tarih: 2025-11-11
Versiyon: 1.0.0

Parquet dosyalarından historical data yönetimi

Özellikler:
- Multi-year support (2023, 2024, 2025 dosyalarını otomatik birleştir)
- Timezone conversion (UTC ↔ Local)
- Warmup period desteği (başlangıçtan önce N mum)
- Akıllı dosya bulma (eksik yılları atla)
- Memory efficient (lazy loading)
- TODO: MTF resample (1m → 5m, 15m, 1h...) Volume için sum, OHLC için resample

Kullanım:
    from components.managers.parquets_engine import ParquetsEngine

    engine = ParquetsEngine(data_path='data/parquets', logger_engine=logger)

    # Historical data al
    df = await engine.get_historical_data(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date='2023-01-01T00:00',
        end_date='2025-01-03T00:00',
        warmup_candles=200,
        utc_offset=3
    )

Bağımlılıklar:
    - python>=3.10
    - pandas
    - pyarrow (parquet okuma için)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class ParquetsEngine:
    """
    Parquet dosyalarından historical data yönetimi

    Multi-year desteği, timezone conversion, warmup period
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        config_engine: Any = None,
        logger_engine: Any = None
    ):
        """
        Initialize ParquetsEngine

        Args:
            data_path: Parquet dosyalarının bulunduğu klasör (opsiyonel - config'den okunur)
            config_engine: ConfigEngine instance (opsiyonel - config'den path okur)
            logger_engine: LoggerEngine instance (opsiyonel)
        """
        self.config_engine = config_engine
        self.logger_engine = logger_engine
        self.logger = logger_engine.get_logger(__name__) if logger_engine else None

        # Data path - config'den oku veya fallback
        if data_path:
            self.data_path = Path(data_path)
        elif config_engine:
            # Config'den oku
            parquet_config = config_engine.get('parquet', {})
            path = parquet_config.get('path', 'data/parquets')
            self.data_path = Path(path)
            if self.logger:
                self.logger.info(f"📂 ParquetsEngine: Config'den path okundu: {self.data_path}")
        else:
            # Fallback
            self.data_path = Path('data/parquets')
            if self.logger:
                self.logger.warning(f"⚠️  ParquetsEngine: Config yok, default path kullanılıyor: {self.data_path}")

        # Cache için (aynı dosya tekrar okunmasın)
        self._file_cache: Dict[str, pd.DataFrame] = {}

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        warmup_candles: int = 0,
        utc_offset: int = 0
    ) -> pd.DataFrame:
        """
        Tarih aralığındaki historical data'yı getir

        Args:
            symbol: Trading pair (örn: BTCUSDT)
            timeframe: Timeframe (örn: 1m, 5m, 15m, 1h)
            start_date: Başlangıç tarihi (local time, ISO format)
            end_date: Bitiş tarihi (local time, ISO format)
            warmup_candles: Başlangıçtan önce kaç mum lazım (warmup için)
            utc_offset: UTC offset (saat) - örn: 3 = UTC+3

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume, timestamp

        Raises:
            FileNotFoundError: Gerekli dosya bulunamadı
            RuntimeError: Yetersiz data
        """
        if self.logger:
            self.logger.info(f"📂 ParquetsEngine: Historical data yükleniyor")
            self.logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}")
            self.logger.info(f"   Başlangıç (local): {start_date}")
            self.logger.info(f"   Bitiş (local): {end_date}")
            self.logger.info(f"   Warmup: {warmup_candles} mum")
            self.logger.info(f"   Timezone: UTC{utc_offset:+d}")

        # Local time'ı UTC'ye çevir
        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)

        start_utc = (dt_start - pd.Timedelta(hours=utc_offset)).tz_localize('UTC')
        end_utc = (dt_end - pd.Timedelta(hours=utc_offset)).tz_localize('UTC')

        if self.logger:
            self.logger.info(f"   Başlangıç (UTC): {start_utc}")
            self.logger.info(f"   Bitiş (UTC): {end_utc}")

        # Hangi yıl dosyaları lazım? (start_utc - warmup'dan end_utc'ye kadar)
        # Warmup için başlangıçtan önce de data lazım
        years = self._get_required_years(start_utc, end_utc, warmup_candles, timeframe)

        if self.logger:
            self.logger.info(f"   Gerekli yıllar: {years}")

        # Multi-year dosyaları oku ve birleştir
        df_list = []
        for year in years:
            df_year = self._read_parquet_file(symbol, timeframe, year)
            if df_year is not None:
                df_list.append(df_year)

        if len(df_list) == 0:
            raise FileNotFoundError(f"Hiç parquet dosya bulunamadı: {symbol}_{timeframe}")

        # Birleştir
        df = pd.concat(df_list, ignore_index=True)

        # CRITICAL: Normalize open_time to UTC-aware for consistent comparison
        # Some files might be tz-naive (old downloads) and some tz-aware (new downloads)
        if 'open_time' in df.columns:
            # Ensure open_time is datetime type (concat might lose dtype)
            if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
                df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
            else:
                # Convert to UTC-aware if not already
                if df['open_time'].dt.tz is None:
                    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
                else:
                    df['open_time'] = df['open_time'].dt.tz_convert('UTC')

            # Timestamp kolonu ekle (int64 ms)
            # Remove timezone info before converting to int64
            df['timestamp'] = df['open_time'].dt.tz_localize(None).astype('int64') // 10**6

        if self.logger:
            self.logger.info(f"   ✅ Toplam {len(df)} satır yüklendi (birleştirilmiş)")

        # Warmup için başlangıçtan ÖNCE warmup_candles kadar data lazım
        if warmup_candles > 0:
            df_before_start = df[df['open_time'] < start_utc]

            if len(df_before_start) < warmup_candles:
                # UYARI: Yetersiz warmup, ama devam et
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Yetersiz warmup data! "
                        f"Başlangıçtan ({start_utc}) önce {warmup_candles} mum gerekli, "
                        f"ancak sadece {len(df_before_start)} mum var. "
                        f"İlk {warmup_candles - len(df_before_start)} mum için indicator değerleri eksik olabilir."
                    )

                # Var olanı kullan
                if len(df_before_start) > 0:
                    warmup_start = df_before_start.iloc[0]['open_time']
                    if self.logger:
                        self.logger.info(f"   📊 Kısmi warmup başlangıcı: {warmup_start} ({len(df_before_start)} mum)")
                    df = df[(df['open_time'] >= warmup_start) & (df['open_time'] <= end_utc)].copy()
                else:
                    # Hiç warmup yok, start_utc'den başla
                    if self.logger:
                        self.logger.warning(f"⚠️  Hiç warmup data yok, başlangıçtan ({start_utc}) başlıyor")
                    df = df[(df['open_time'] >= start_utc) & (df['open_time'] <= end_utc)].copy()
            else:
                # Yeterli warmup var
                warmup_start = df_before_start.iloc[-warmup_candles]['open_time']

                if self.logger:
                    self.logger.info(f"   📊 Warmup başlangıcı: {warmup_start}")

                # Warmup başlangıcından end_utc'ye kadar filtrele
                df = df[(df['open_time'] >= warmup_start) & (df['open_time'] <= end_utc)].copy()
        else:
            # Warmup yok, sadece start_utc - end_utc aralığı
            df = df[(df['open_time'] >= start_utc) & (df['open_time'] <= end_utc)].copy()

        if self.logger:
            self.logger.info(f"   ✅ Filtre sonrası: {len(df)} satır")
            self.logger.info(f"   📅 Tarih aralığı: {df.iloc[0]['open_time']} - {df.iloc[-1]['open_time']}")

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def _get_required_years(
        self,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        warmup_candles: int,
        timeframe: str
    ) -> List[int]:
        """
        Gerekli yıl dosyalarını belirle

        Warmup için başlangıçtan önce de data lazım, o yüzden daha eski yıllar da gerekebilir.
        """
        # Başlangıç ve bitiş yılları
        start_year = start_utc.year
        end_year = end_utc.year

        # Warmup için kaç gün geriye gitmek lazım?
        if warmup_candles > 0:
            # Timeframe'i dakikaya çevir
            tf_minutes = self._parse_timeframe_to_minutes(timeframe)

            # Warmup için gereken toplam süre (dakika)
            warmup_minutes = warmup_candles * tf_minutes

            # Dakikayı güne çevir
            warmup_days = warmup_minutes / (60 * 24)

            # Warmup başlangıç tarihi
            warmup_start = start_utc - pd.Timedelta(days=warmup_days)

            # Warmup başlangıç yılı
            warmup_start_year = warmup_start.year

            if self.logger:
                self.logger.debug(f"   Warmup hesaplama: {warmup_candles} × {tf_minutes}min = {warmup_days:.1f} gün")
                self.logger.debug(f"   Warmup başlangıç yılı: {warmup_start_year}")

            # start_year'ı güncelle
            start_year = min(start_year, warmup_start_year)

        # Yıl listesi oluştur
        years = list(range(start_year, end_year + 1))

        return years

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Timeframe string'ini dakikaya çevir (örn: '15m' -> 15, '1h' -> 60)"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        else:
            # Default: 1m
            return 1

    def _read_parquet_file(
        self,
        symbol: str,
        timeframe: str,
        year: int
    ) -> Optional[pd.DataFrame]:
        """
        Tek bir parquet dosyayı oku

        Cache kullanır, dosya yoksa None döner (hata vermez)
        """
        # Windows case-insensitive fix: 1M (month) → 1MO
        # (1m minute ile karışmasın diye)
        file_timeframe = "1MO" if timeframe == "1M" else timeframe

        # Yeni format: data/parquets/{symbol}/{symbol}_{timeframe}_{year}.parquet
        filename = f"{symbol}_{file_timeframe}_{year}.parquet"
        symbol_dir = self.data_path / symbol
        filepath = symbol_dir / filename

        # Cache'de var mı?
        cache_key = str(filepath)
        if cache_key in self._file_cache:
            if self.logger:
                self.logger.debug(f"   📦 Cache'den okundu: {filename}")
            return self._file_cache[cache_key]

        # Dosya var mı?
        if not filepath.exists():
            if self.logger:
                self.logger.warning(f"   ⚠️  Dosya bulunamadı (atlanıyor): {filename}")
            return None

        # Oku
        try:
            df = pd.read_parquet(filepath)

            if self.logger:
                self.logger.info(f"   ✅ Okundu: {filename} ({len(df)} satır)")

            # Cache'e ekle
            self._file_cache[cache_key] = df

            return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"   ❌ Okuma hatası: {filename} - {e}")
            return None

    def clear_cache(self):
        """Cache'i temizle"""
        self._file_cache.clear()
        if self.logger:
            self.logger.info("🧹 ParquetsEngine cache temizlendi")

    # ========================================================================
    # TODO: MTF RESAMPLE SUPPORT
    # ========================================================================

    async def resample_timeframe(
        self,
        df: pd.DataFrame,
        source_tf: str,
        target_tf: str
    ) -> pd.DataFrame:
        """
        TODO: Bir timeframe'den diğerine resample

        Örnek: 1m → 5m, 15m, 1h

        Rules:
        - OHLC: first, max, min, last
        - Volume: sum
        - open_time: first

        Args:
            df: Source DataFrame
            source_tf: Kaynak timeframe (örn: 1m)
            target_tf: Hedef timeframe (örn: 5m)

        Returns:
            Resampled DataFrame
        """
        raise NotImplementedError("MTF resample henüz implement edilmedi - TODO")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from core.logger_engine import LoggerEngine
    from core.config_engine import ConfigEngine

    async def test_parquets_engine():
        """ParquetsEngine test"""
        print("=" * 80)
        print("ParquetsEngine Test")
        print("=" * 80)

        # Config & Logger
        config_engine = ConfigEngine(config_path='config/main.yaml')
        logger_engine = LoggerEngine()
        logger = logger_engine.get_logger(__name__)

        # Engine - config'den path oku
        engine = ParquetsEngine(
            config_engine=config_engine,
            logger_engine=logger_engine
        )

        # Test: 2025-01-01 - 2025-01-03 (200 warmup)
        logger.info("\n📊 Test 1: Multi-year warmup (2024 + 2025)")

        try:
            df = await engine.get_historical_data(
                symbol='BTCUSDT',
                timeframe='15m',
                start_date='2025-01-01T00:00',
                end_date='2025-01-03T00:00',
                warmup_candles=200,
                utc_offset=3
            )

            logger.info(f"\n✅ Test 1 BAŞARILI!")
            logger.info(f"   Toplam satır: {len(df)}")
            logger.info(f"   İlk mum: {df.iloc[0]['open_time']}")
            logger.info(f"   Son mum: {df.iloc[-1]['open_time']}")

        except Exception as e:
            logger.error(f"❌ Test 1 BAŞARISIZ: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST TAMAMLANDI!")
        logger.info("=" * 80)

    asyncio.run(test_parquets_engine())
