"""Replay Trading API endpoints"""
from flask import request
from ..helpers.response_helper import success_response, error_response
from ..helpers.async_helper import run_async


def get_replay_service():
    """Get replay service from app context"""
    from flask import current_app
    return current_app.replay_service


def register_routes(bp):
    """Register replay API routes"""

    @bp.route('/replay/strategies', methods=['GET'])
    def get_strategies():
        """GET /api/replay/strategies - Strateji listesi"""
        try:
            service = get_replay_service()
            strategies = run_async(service.get_strategies())
            return success_response({'strategies': strategies})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions', methods=['POST'])
    def create_session():
        """POST /api/replay/sessions - Yeni replay session oluştur

        Body:
            strategy_id: Strateji ID (required)
            calculate_indicators: true/false - Indicator hesaplama (default: false)
        """
        try:
            data = request.get_json()
            if not data:
                return error_response('Request body boş olamaz', 400)

            strategy_id = data.get('strategy_id')
            if not strategy_id:
                return error_response('strategy_id gerekli', 400)

            # Performance: default False - indicator hesaplamayı atla
            calculate_indicators = data.get('calculate_indicators', False)

            service = get_replay_service()
            session = run_async(service.create_session(
                strategy_id,
                calculate_indicators=calculate_indicators
            ))
            return success_response({'session': session})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>', methods=['GET'])
    def get_session(session_id):
        """GET /api/replay/sessions/{id} - Session detayları"""
        try:
            service = get_replay_service()
            session = run_async(service.get_session(session_id))
            if not session:
                return error_response('Session bulunamadı', 404)
            return success_response({'session': session})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/candles', methods=['GET'])
    def get_candles(session_id):
        """GET /api/replay/sessions/{id}/candles - Chart verisi"""
        try:
            start = request.args.get('start', 0, type=int)
            limit = request.args.get('limit', 100, type=int)

            service = get_replay_service()
            result = run_async(service.get_candles(session_id, start, limit))
            if not result:
                return error_response('Session bulunamadı', 404)
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/play', methods=['POST'])
    def play_session(session_id):
        """POST /api/replay/sessions/{id}/play - Replay başlat"""
        try:
            service = get_replay_service()
            result = run_async(service.play(session_id))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/pause', methods=['POST'])
    def pause_session(session_id):
        """POST /api/replay/sessions/{id}/pause - Replay duraklat"""
        try:
            service = get_replay_service()
            result = run_async(service.pause(session_id))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/step', methods=['POST'])
    def step_session(session_id):
        """POST /api/replay/sessions/{id}/step - Bir bar ileri/geri"""
        try:
            data = request.get_json() or {}
            direction = data.get('direction', 1)  # 1 = ileri, -1 = geri

            service = get_replay_service()
            result = run_async(service.step(session_id, direction))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/seek', methods=['POST'])
    def seek_session(session_id):
        """POST /api/replay/sessions/{id}/seek - Belirli pozisyona git"""
        try:
            data = request.get_json() or {}
            position = data.get('position')
            if position is None:
                return error_response('position gerekli', 400)

            service = get_replay_service()
            result = run_async(service.seek(session_id, position))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/speed', methods=['POST'])
    def set_speed(session_id):
        """POST /api/replay/sessions/{id}/speed - Hız ayarla"""
        try:
            data = request.get_json() or {}
            speed = data.get('speed', 1.0)

            service = get_replay_service()
            result = run_async(service.set_speed(session_id, speed))
            return success_response(result)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/state', methods=['GET'])
    def get_state(session_id):
        """GET /api/replay/sessions/{id}/state - Mevcut state"""
        try:
            service = get_replay_service()
            state = run_async(service.get_state(session_id))
            if not state:
                return error_response('Session bulunamadı', 404)
            return success_response({'state': state})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>/trades', methods=['GET'])
    def get_trades(session_id):
        """GET /api/replay/sessions/{id}/trades - Trade listesi"""
        try:
            service = get_replay_service()
            trades = run_async(service.get_trades(session_id))
            return success_response({'trades': trades})
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/replay/sessions/<session_id>', methods=['DELETE'])
    def delete_session(session_id):
        """DELETE /api/replay/sessions/{id} - Session sil"""
        try:
            service = get_replay_service()
            run_async(service.delete_session(session_id))
            return success_response(message='Session silindi')
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/backtest/trades', methods=['GET'])
    def get_backtest_trades():
        """GET /api/backtest/trades - Backtest trade'lerini ve bar index'lerini getir

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (required)
        """
        try:
            import pandas as pd
            from pathlib import Path

            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe')

            if not symbol or not timeframe:
                return error_response('symbol ve timeframe gerekli', 400)

            # Backtest parquet dosyasını bul
            backtest_dir = Path("data/ai/features/backtest")
            if not backtest_dir.exists():
                return error_response('Backtest dizini bulunamadı', 404)

            pattern = f"backtest_{symbol}_{timeframe}_*.parquet"
            files = sorted(backtest_dir.glob(pattern), reverse=True)

            if not files:
                return error_response(f'Backtest dosyası bulunamadı: {pattern}', 404)

            # En son backtest dosyasını yükle
            df = pd.read_parquet(files[0])

            if df.empty:
                return success_response({'trades': [], 'trade_bar_indices': []})

            # Candle verilerini yükle (timestamp -> bar index mapping için)
            parquet_dir = Path("data/parquets") / symbol
            candle_pattern = f"{symbol}_{timeframe}_*.parquet"
            candle_files = sorted(parquet_dir.glob(candle_pattern))

            if not candle_files:
                return error_response('Candle dosyası bulunamadı', 404)

            candle_df = pd.read_parquet(candle_files[-1])

            # Timestamp -> bar index mapping
            ts_to_index = {}
            for idx, row in candle_df.iterrows():
                ts = row.get('timestamp', row.get('open_time', 0))
                if isinstance(ts, (int, float)):
                    ts_sec = int(ts / 1000) if ts > 1e12 else int(ts)
                elif hasattr(ts, 'timestamp'):
                    ts_sec = int(ts.timestamp())
                else:
                    ts_sec = int(pd.to_datetime(ts).timestamp())
                ts_to_index[ts_sec] = idx

            # Trade bar index'lerini hesapla
            trade_bar_indices = []
            trades = []

            for _, trade in df.iterrows():
                entry_ts = trade.get('entry_time')
                if not pd.notna(entry_ts):
                    continue

                # Entry timestamp'i saniyeye çevir
                if isinstance(entry_ts, (int, float)):
                    entry_ts_sec = int(entry_ts / 1000) if entry_ts > 1e12 else int(entry_ts)
                else:
                    entry_ts_sec = int(entry_ts)

                # En yakın candle'ı bul
                closest_idx = 0
                min_diff = float('inf')
                for ts, idx in ts_to_index.items():
                    diff = abs(ts - entry_ts_sec)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = idx

                trade_bar_indices.append(closest_idx)
                trades.append({
                    'entry_time': entry_ts_sec,
                    'bar_index': closest_idx,
                    'side': str(trade.get('side', '')),
                    'pnl': float(trade.get('pnl', 0)) if pd.notna(trade.get('pnl')) else 0
                })

            return success_response({
                'trades': trades,
                'trade_bar_indices': sorted(trade_bar_indices),
                'total': len(trades)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_response(str(e), 500)

    @bp.route('/replay/candles', methods=['GET'])
    def get_direct_candles():
        """GET /api/replay/candles - Dogrudan parquet'ten candle cek

        Query params:
            symbol: BTCUSDT (required)
            timeframe: 5m (default)
            limit: 500 (default)
        """
        try:
            import pandas as pd
            from pathlib import Path

            symbol = request.args.get('symbol')
            if not symbol:
                return error_response('symbol gerekli', 400)

            timeframe = request.args.get('timeframe', '5m')
            limit = request.args.get('limit', 500, type=int)

            # Dogrudan parquet dosyasindan oku - yeni format: data/parquets/{symbol}/
            parquet_dir = Path("data/parquets")
            symbol_dir = parquet_dir / symbol
            if not symbol_dir.exists():
                return error_response(f'Sembol dizini bulunamadi: {symbol}', 404)

            pattern = f"{symbol}_{timeframe}_*.parquet"
            files = list(symbol_dir.glob(pattern))

            if not files:
                return error_response(f'Parquet dosyasi bulunamadi: {pattern}', 404)

            parquet_file = sorted(files)[-1]
            df = pd.read_parquet(parquet_file)

            if df is None or len(df) == 0:
                return error_response('Veri bulunamadi', 404)

            # Limit uygula
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Chart formatina cevir
            candles = []
            for _, row in df.iterrows():
                # Timestamp al
                ts = row.get('timestamp', row.get('open_time', 0))

                # Timestamp tipini int saniyeye cevir
                if pd.api.types.is_datetime64_any_dtype(type(ts)) or hasattr(ts, 'timestamp'):
                    ts = int(ts.timestamp())
                elif isinstance(ts, (int, float)):
                    ts = float(ts)
                    if ts > 1e12:
                        ts = int(ts / 1000)
                    else:
                        ts = int(ts)
                else:
                    # String veya baska tip
                    ts = int(pd.to_datetime(ts).timestamp())

                candles.append({
                    'time': ts,
                    'timestamp': ts * 1000,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0))
                })

            return success_response({
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': candles,
                'total': len(candles)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_response(str(e), 500)

    @bp.route('/data/symbols', methods=['GET'])
    def get_data_symbols():
        """GET /api/data/symbols - data/parquets klasöründeki sembolleri listele

        Returns:
            {
                "symbols": ["BTCUSDT", "ETHUSDT", ...],
                "total": 10
            }
        """
        try:
            from pathlib import Path

            parquet_dir = Path("data/parquets")
            if not parquet_dir.exists():
                return success_response({'symbols': [], 'total': 0})

            # Her alt klasör bir sembol
            symbols = []
            for item in sorted(parquet_dir.iterdir()):
                if item.is_dir():
                    # Klasör adı sembol adı
                    symbol = item.name.upper()
                    # En az bir parquet dosyası var mı kontrol et
                    parquet_files = list(item.glob("*.parquet"))
                    if parquet_files:
                        symbols.append(symbol)

            return success_response({
                'symbols': symbols,
                'total': len(symbols)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_response(str(e), 500)
