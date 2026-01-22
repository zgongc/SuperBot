"""Analysis business logic"""
from .base_service import BaseService

class AnalysisService(BaseService):
    """Analysis management service"""

    async def queue_analysis(self, symbol_ids=None, symbols=None, timeframe='1h',
                            strategy_name=None):
        """Queue symbols for analysis"""
        # If symbol_ids provided, fetch symbols from favorites
        if symbol_ids and not symbols:
            favorites = await self.data_manager.get_favorites()
            symbols = [f['symbol'] for f in favorites if f['id'] in symbol_ids]

        # Queue each symbol
        queue_ids = []
        for symbol in symbols:
            queue_id = await self.data_manager.queue_analysis(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name
            )
            if queue_id:
                queue_ids.append(queue_id)

        return {
            'analysis_ids': queue_ids,
            'queued_count': len(queue_ids)
        }

    async def get_analysis_queue(self, status=None, limit=100):
        """Get analysis queue"""
        queue = await self.data_manager.get_analysis_queue(
            status=status,
            limit=limit
        )

        # Count by status
        status_counts = {
            'pending': len([q for q in queue if q['status'] == 'pending']),
            'analyzing': len([q for q in queue if q['status'] == 'analyzing']),
            'completed': len([q for q in queue if q['status'] == 'completed']),
            'failed': len([q for q in queue if q['status'] == 'failed'])
        }

        return {
            'queue': queue,
            'total': len(queue),
            'status_counts': status_counts
        }

    async def get_analysis_results(self, symbol=None, timeframe=None, signal=None,
                                  min_confidence=None, limit=100):
        """Get analysis results"""
        results = await self.data_manager.get_analysis_results(
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            min_confidence=min_confidence,
            limit=limit
        )

        # Group by signal
        signal_counts = {
            'BUY': len([r for r in results if r['signal'] == 'BUY']),
            'SELL': len([r for r in results if r['signal'] == 'SELL']),
            'HOLD': len([r for r in results if r['signal'] == 'HOLD']),
            'NEUTRAL': len([r for r in results if r['signal'] == 'NEUTRAL'])
        }

        return {
            'results': results,
            'total': len(results),
            'signal_counts': signal_counts
        }
