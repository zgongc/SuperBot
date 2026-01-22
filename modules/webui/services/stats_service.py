"""Stats business logic"""
from .base_service import BaseService

class StatsService(BaseService):
    """Statistics service"""

    def __init__(self, data_manager, symbols_manager, logger):
        super().__init__(data_manager, logger)
        self.symbols_manager = symbols_manager

    async def get_overview_stats(self):
        """Get dashboard overview statistics"""
        # Get symbols count
        symbols_data = await self.symbols_manager.get_symbols() if self.symbols_manager else {}
        symbols_count = len(symbols_data.get('base_assets', []))

        # Get favorites count
        favorites = await self.data_manager.get_favorites()
        favorites_count = len(favorites)

        # Get analysis stats
        queue = await self.data_manager.get_analysis_queue(limit=1000)
        analysis_results = await self.data_manager.get_analysis_results(limit=1000)

        # Get portfolio stats
        holdings = await self.data_manager.get_portfolio_holdings()
        total_portfolio_value = sum(h['current_value'] or 0 for h in holdings)
        total_pnl = sum(h['unrealized_pnl'] or 0 for h in holdings) + sum(h['realized_pnl'] or 0 for h in holdings)

        # Signal counts
        signal_counts = {
            'buy': len([r for r in analysis_results if r['signal'] == 'BUY']),
            'sell': len([r for r in analysis_results if r['signal'] == 'SELL']),
            'hold': len([r for r in analysis_results if r['signal'] == 'HOLD'])
        }

        return {
            'symbols': {
                'total': symbols_count,
                'favorites': favorites_count
            },
            'analysis': {
                'pending': len([q for q in queue if q['status'] == 'pending']),
                'completed_total': len(analysis_results),
                'signals': signal_counts
            },
            'portfolio': {
                'total_value': total_portfolio_value,
                'total_pnl': total_pnl,
                'holdings_count': len(holdings)
            }
        }
