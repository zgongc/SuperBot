"""Portfolio business logic"""
from typing import Dict, List, Any, Optional
from .base_service import BaseService
from modules.webui.managers.portfolio_calculator import PortfolioCalculator
from components.exchanges.binance_api import BinanceAPI
# TODO: Multi-exchange support - add other exchanges when needed
# from components.exchanges.gateio_api import GateioAPI

class PortfolioService(BaseService):
    """Portfolio management service"""

    # ============================================
    # Portfolio Management
    # ============================================

    async def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolios with calculated metrics"""
        portfolios = await self.data_manager.get_all_portfolios()

        # Add position count and calculated metrics to each portfolio
        for portfolio in portfolios:
            positions = await self.data_manager.get_portfolio_positions(
                portfolio['id'],
                is_open=True
            )
            portfolio['position_count'] = len(positions)
            portfolio['total_positions'] = len(await self.data_manager.get_portfolio_positions(portfolio['id']))

            # Calculate portfolio metrics
            portfolio['total_value'] = PortfolioCalculator.calculate_total_value(positions)
            portfolio['total_cost'] = PortfolioCalculator.calculate_total_cost(positions)
            pnl, pnl_pct = PortfolioCalculator.calculate_pnl(positions)
            portfolio['unrealized_pnl'] = pnl
            portfolio['unrealized_pnl_pct'] = pnl_pct

        return portfolios

    async def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Get portfolio details"""
        return await self.data_manager.get_portfolio_by_id(portfolio_id)

    async def create_portfolio(
        self,
        name: str,
        exchange_account_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Optional[int]:
        """Create new portfolio

        Args:
            name: Portfolio name
            exchange_account_id: Foreign key to exchange_accounts (NULL for manual portfolio)
            notes: Optional notes
        """
        portfolio_id = await self.data_manager.create_portfolio(
            name=name,
            exchange_account_id=exchange_account_id,
            notes=notes
        )

        if portfolio_id:
            self.logger.info(f"Portfolio created: {name} (ID: {portfolio_id})")

        return portfolio_id

    async def update_portfolio(
        self,
        portfolio_id: int,
        **kwargs
    ) -> bool:
        """Update portfolio"""
        return await self.data_manager.update_portfolio(portfolio_id, **kwargs)

    async def delete_portfolio(self, portfolio_id: int) -> bool:
        """Delete portfolio"""
        return await self.data_manager.delete_portfolio(portfolio_id)

    # ============================================
    # Portfolio Summary & Analytics
    # ============================================

    async def get_portfolio_summary(self, portfolio_id: int, is_open: Optional[bool] = None) -> Dict[str, Any]:
        """Get portfolio summary with positions and P&L"""
        # Get positions (filtered by is_open if specified)
        positions = await self.data_manager.get_portfolio_positions(
            portfolio_id=portfolio_id,
            is_open=is_open
        )

        # Calculate summary
        total_value = 0
        total_cost = 0
        total_unrealized_pnl = 0

        for pos in positions:
            cost = pos['quantity'] * pos['entry_price']

            # Only calculate unrealized P&L for OPEN positions
            if pos.get('is_open', True):
                value = pos['quantity'] * (pos['current_price'] or pos['entry_price'])
                pnl = value - cost

                total_cost += cost
                total_value += value
                total_unrealized_pnl += pnl

        unrealized_pnl_pct = (total_unrealized_pnl / total_cost * 100) if total_cost > 0 else 0

        # Get closed positions for realized P&L
        closed_positions = await self.data_manager.get_portfolio_positions(
            portfolio_id=portfolio_id,
            is_open=False
        )
        total_realized_pnl = sum(p.get('realized_pnl') or 0 for p in closed_positions)

        # Calculate cost for closed positions
        closed_cost = sum(p['quantity'] * p['entry_price'] for p in closed_positions)
        realized_pnl_pct = (total_realized_pnl / closed_cost * 100) if closed_cost > 0 else 0

        # Total cost includes both open and closed positions
        total_all_cost = total_cost + closed_cost

        summary = {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'total_realized_pnl': total_realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_pnl_pct': ((total_unrealized_pnl + total_realized_pnl) / total_all_cost * 100) if total_all_cost > 0 else 0,
            'position_count': len(positions),
            'closed_position_count': len(closed_positions)
        }

        return {
            'positions': positions,
            'summary': summary
        }

    # ============================================
    # Position Management
    # ============================================

    async def get_positions(self, portfolio_id: int, is_open: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get portfolio positions"""
        return await self.data_manager.get_portfolio_positions(
            portfolio_id=portfolio_id,
            is_open=is_open
        )

    async def add_manual_position(
        self,
        portfolio_id: int,
        symbol: Optional[str] = None,
        symbol_id: Optional[int] = None,
        quantity: float = 0,
        entry_price: float = 0,
        side: str = 'LONG',
        position_type: str = 'SPOT',
        opened_at: Optional[str] = None,
        source: str = 'manual',
        notes: Optional[str] = None
    ) -> Optional[int]:
        """Add manual position

        Args:
            portfolio_id: Portfolio ID
            symbol: Symbol string (e.g., "BTC/USDT") - user manual input
            symbol_id: Symbol ID from exchange_symbols table (optional)
            quantity: Position quantity
            entry_price: Entry price
            side: LONG or SHORT (default: LONG)
            position_type: SPOT or FUTURES (default: SPOT)
            opened_at: ISO datetime string (default: now)
            source: Position source (default: manual)
            notes: Optional notes
        """
        position_id = await self.data_manager.create_position(
            portfolio_id=portfolio_id,
            symbol=symbol,
            symbol_id=symbol_id,
            quantity=quantity,
            entry_price=entry_price,
            side=side,
            position_type=position_type,
            opened_at=opened_at,
            source=source,
            notes=notes
        )

        if position_id:
            symbol_info = symbol or f"symbol_id={symbol_id}"
            self.logger.info(f"📊 Manually added position: {symbol_info}, quantity={quantity}, price={entry_price}")

        return position_id

    async def close_position(self, position_id: int, close_price: Optional[float] = None) -> bool:
        """Close a position"""
        return await self.data_manager.close_position(position_id, close_price)

    async def delete_position(self, position_id: int) -> bool:
        """Delete a position"""
        return await self.data_manager.delete_position(position_id)

    async def update_positions_prices(self, portfolio_id: int) -> Dict[str, Any]:
        """
        Update all position prices from exchanges

        Returns:
            {
                'updated_count': int,
                'errors': List[str],
                'prices': Dict[str, float]
            }
        """
        try:
            # Get all open positions
            positions = await self.data_manager.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=True
            )

            if not positions:
                return {
                    'updated_count': 0,
                    'errors': [],
                    'prices': {},
                    'message': 'No open positions to update'
                }

            updated_count = 0
            errors = []
            prices = {}

            # Group symbols by exchange for batch requests
            symbols_by_exchange = {}
            for pos in positions:
                exchange = pos.get('exchange', 'binance')
                symbol = pos.get('symbol', '')
                base_asset = pos.get('base_asset', '')
                quote_asset = pos.get('quote_asset', '')

                if exchange not in symbols_by_exchange:
                    symbols_by_exchange[exchange] = []
                symbols_by_exchange[exchange].append({
                    'position_id': pos['id'],
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'quote_asset': quote_asset
                })

            self.logger.info(f"📊 Price update: {len(positions)} positions, {len(symbols_by_exchange)} exchanges")

            # Fetch prices from each exchange
            for exchange, symbol_list in symbols_by_exchange.items():
                try:
                    client = None

                    # Initialize appropriate client based on exchange
                    if exchange == 'binance':
                        client = BinanceAPI(config={'testnet': False})
                    # elif exchange == 'gateio':
                    #     client = GateioAPI(config={'testnet': False})
                    # elif exchange == 'kucoin':
                    #     client = KucoinAPI(config={'testnet': False})
                    # elif exchange == 'bybit':
                    #     client = BybitAPI(config={'testnet': False})
                    # elif exchange == 'okx':
                    #     client = OKXAPI(config={'testnet': False})
                    else:
                        self.logger.warning(f"⚠️  {exchange} desteklenmiyor")
                        for sym_data in symbol_list:
                            errors.append(f"{sym_data['symbol']}: {exchange} not supported")
                        continue

                    self.logger.info(f"📡 Connection to {exchange} established")

                    # Fetch prices for each symbol
                    for sym_data in symbol_list:
                        try:
                            # Get ticker for symbol
                            # Binance uses BTCUSDT format (no slash), CCXT uses BTC/USDT format
                            if exchange == 'binance':
                                # Binance: use symbol as-is (remove slash if present)
                                symbol_for_api = sym_data['symbol'].replace('/', '')
                            else:
                                # CCXT exchanges: need BASE/QUOTE format with slash
                                base = sym_data.get('base_asset', '')
                                quote = sym_data.get('quote_asset', '')
                                if base and quote:
                                    symbol_for_api = f"{base}/{quote}"
                                else:
                                    # Fallback: if slash exists use as-is, otherwise add slash before last 4 chars (USDT)
                                    if '/' in sym_data['symbol']:
                                        symbol_for_api = sym_data['symbol']
                                    else:
                                        # Assume USDT as quote asset for now
                                        symbol_for_api = sym_data['symbol'][:-4] + '/' + sym_data['symbol'][-4:]

                            ticker = await client.get_ticker(symbol_for_api)
                            current_price = float(ticker.get('lastPrice', 0))

                            if current_price > 0:
                                # Update position price
                                success = await self.data_manager.update_position_price(
                                    sym_data['position_id'],
                                    current_price
                                )

                                if success:
                                    updated_count += 1
                                    prices[sym_data['symbol']] = current_price
                                    self.logger.info(f"✅ {sym_data['symbol']}: ${current_price}")
                                else:
                                    errors.append(f"Failed to update {sym_data['symbol']} in database")
                            else:
                                errors.append(f"Invalid price for {sym_data['symbol']}: {current_price}")

                        except Exception as e:
                            error_msg = f"Error fetching {sym_data['symbol']} from {exchange}: {str(e)}"
                            errors.append(error_msg)
                            self.logger.error(f"❌ {error_msg}")

                    # Close CCXT client connections
                    if hasattr(client, 'close'):
                        await client.close()

                except Exception as e:
                    error_msg = f"Error connecting to {exchange}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")

            return {
                'updated_count': updated_count,
                'total_positions': len(positions),
                'errors': errors,
                'prices': prices,
                'message': f'{updated_count}/{len(positions)} positions updated'
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"❌ Price update error: {e}")
            self.logger.error(f"Traceback:\n{error_details}")
            return {
                'updated_count': 0,
                'errors': [f"{str(e)} - Check server logs for details"],
                'prices': {},
                'message': 'Price update failed'
            }

    # ============================================
    # Import from Exchange (Placeholder)
    # ============================================

    async def sync_positions_from_exchange(self, portfolio_id: int) -> Dict[str, Any]:
        """
        Synchronize positions from the stock market (SPOT balances)

        Args:
            portfolio_id: Portfolio ID

        Returns:
            {'synced_count': int, 'message': str}
        """
        try:
            # Portfolio bilgilerini al
            portfolio = await self.data_manager.get_portfolio_by_id(portfolio_id)
            if not portfolio or not portfolio.get('exchange_account_id'):
                self.logger.warning(f"⚠️  Portfolio {portfolio_id} does not have an exchange account")
                return {'synced_count': 0, 'message': 'Exchange account not found'}

            # Exchange hesap bilgilerini al
            exchange_account = await self.data_manager.get_exchange_account_by_id(
                portfolio['exchange_account_id']
            )
            if not exchange_account:
                self.logger.warning(f"⚠️ Exchange account not found: {portfolio['exchange_account_id']}")
                return {'synced_count': 0, 'message': 'Exchange account not found'}

            # Synchronization is not possible for manually created portfolios.
            if exchange_account['exchange'] == 'manual':
                self.logger.info(f"📝 Manual portfolio synchronization is not possible: {portfolio['name']}")
                return {'synced_count': 0, 'message': 'Sync cannot be performed for manually created portfolios'}

            # TODO: Exchange connector integration
            # Currently a placeholder - an exchange connector will be used in the future.
            self.logger.info(f"🔄 Starting sync: {portfolio['name']} - {exchange_account['exchange']}")

            # Placeholder: An exchange connector is required for the actual implementation.
            # from components.connectors.exchange_connector_engine import ExchangeConnectorEngine
            # connector = ExchangeConnectorEngine(...)
            # balances = await connector.get_account_balance(...)

            self.logger.warning(f"⚠️ Exchange connector has not yet been integrated")

            return {
                'synced_count': 0,
                'message': 'Exchange connector entegrasyonu bekleniyor'
            }

        except Exception as e:
            self.logger.error(f"❌ Sync error: {e}")
            raise

    async def prepare_import_preview(self, portfolio_id: int) -> Dict[str, Any]:
        """
        Prepare import preview by fetching exchange balances and comparing with DB

        Returns:
            {
                'assets': [
                    {
                        'asset': 'BTC',
                        'symbol': 'BTCUSDT',
                        'exchange_quantity': 1.5,
                        'db_quantity': 1.5,
                        'difference': 0,
                        'status': 'already_imported',  # new, already_imported, quantity_increased, quantity_decreased
                        'current_price': 43000,
                        'checked': False
                    },
                    ...
                ]
            }
        """
        try:
            # Get portfolio with exchange account info
            portfolio = await self.data_manager.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                return {
                    'assets': [],
                    'message': 'Portfolio not found'
                }

            exchange_account_id = portfolio.get('exchange_account_id')
            if not exchange_account_id:
                return {
                    'assets': [],
                    'message': 'This portfolio is not linked to a brokerage account. Use the JSON import for a manual portfolio.'
                }

            # Get exchange account credentials
            exchange_account = await self.data_manager.get_exchange_account_by_id(exchange_account_id)
            if not exchange_account:
                return {
                    'assets': [],
                    'message': 'Exchange account not found'
                }

            # Check if API credentials exist
            api_key = exchange_account.get('api_key')
            api_secret = exchange_account.get('api_secret')
            if not api_key or not api_secret:
                return {
                    'assets': [],
                    'message': 'An API key has not been defined for this exchange account. Please add an API key from the Exchange Accounts page.'
                }

            # Initialize exchange client based on exchange type
            exchange_type = exchange_account.get('exchange', 'binance').lower()
            environment = exchange_account.get('environment', 'production')

            if exchange_type != 'binance':
                return {
                    'assets': [],
                    'message': f'{exchange_type} is not yet supported. Currently, only Binance is supported.'
                }

            # Create client
            testnet = environment == 'testnet'
            client = BinanceAPI(config={
                'testnet': testnet,
                'api_key': api_key,
                'api_secret': api_secret
            })

            # Fetch account balance
            self.logger.info(f"🔄 Fetching balances from {exchange_type} ({environment})...")
            account_data = await client.get_account_balance()

            # Save to account_spot.json for later use (cache for homepage)
            try:
                import json
                import os
                json_path = os.path.join('data', 'json', 'account_spot.json')
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(account_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 Saved account data to {json_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not save account_spot.json: {e}")

            # Extract non-zero balances
            balances = account_data.get('balances', [])
            non_zero_balances = [
                b for b in balances
                if float(b.get('free', 0)) + float(b.get('locked', 0)) > 0
            ]

            self.logger.info(f"📊 Found {len(non_zero_balances)} assets with non-zero balance")

            # Filter out stablecoins first
            filtered_balances = [
                b for b in non_zero_balances
                if b['asset'] not in ['USDT', 'BUSD', 'USDC', 'DAI', 'FDUSD', 'TUSD']
            ]

            self.logger.info(f"📊 {len(filtered_balances)} tradeable assets (after filtering stablecoins)")

            # Fetch ALL prices in ONE call (much faster!)
            self.logger.info(f"💰 Fetching all ticker prices in one call...")
            all_tickers = await client.get_all_tickers()

            # Create price map for fast lookup
            price_map = {ticker['symbol']: float(ticker['price']) for ticker in all_tickers}
            self.logger.info(f"✅ Got prices for {len(price_map)} symbols")

            # Get existing positions from DB
            existing_positions = await self.data_manager.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=True
            )

            # Create a map of existing positions by base asset
            existing_map = {}
            for pos in existing_positions:
                symbol = pos.get('symbol', '')
                # Extract base asset (e.g., BTCUSDT -> BTC)
                if symbol.endswith('USDT'):
                    base_asset = symbol.replace('USDT', '')
                elif symbol.endswith('BUSD'):
                    base_asset = symbol.replace('BUSD', '')
                elif '/' in symbol:
                    base_asset = symbol.split('/')[0]
                else:
                    base_asset = symbol

                if base_asset not in existing_map:
                    existing_map[base_asset] = pos

            # Build asset list with comparison
            assets = []
            for balance in filtered_balances:
                asset = balance['asset']
                exchange_quantity = float(balance['free']) + float(balance['locked'])

                # Construct symbol (assume USDT pairs)
                symbol = f"{asset}USDT"

                # Get current price from price map (already fetched in one call!)
                current_price = price_map.get(symbol, 0)

                if current_price == 0:
                    # Try BUSD pair if USDT not available
                    symbol_busd = f"{asset}BUSD"
                    current_price = price_map.get(symbol_busd, 0)
                    if current_price > 0:
                        symbol = symbol_busd

                # Calculate USD value
                usd_value = exchange_quantity * current_price

                # Skip assets with value less than $1 (same as Binance "Hide assets <1 USD" filter)
                if usd_value < 1.0:
                    self.logger.debug(f"⏭️ Skipping {asset}: ${usd_value:.4f} < $1")
                    continue

                # Compare with DB
                db_position = existing_map.get(asset)
                if db_position:
                    db_quantity = db_position.get('quantity', 0)
                    difference = exchange_quantity - db_quantity

                    if abs(difference) < 0.00000001:  # Essentially equal
                        status = 'already_imported'
                    elif difference > 0:
                        status = 'quantity_increased'
                    else:
                        status = 'quantity_decreased'
                else:
                    db_quantity = 0
                    difference = exchange_quantity
                    status = 'new'

                assets.append({
                    'asset': asset,
                    'symbol': symbol,
                    'exchange_quantity': exchange_quantity,
                    'db_quantity': db_quantity,
                    'difference': difference,
                    'status': status,
                    'current_price': current_price,
                    'checked': status in ['new', 'quantity_increased', 'quantity_decreased']
                })

            # Sort: new first, then increased/decreased, then already_imported
            status_order = {'new': 0, 'quantity_increased': 1, 'quantity_decreased': 2, 'already_imported': 3}
            assets.sort(key=lambda x: status_order.get(x['status'], 4))

            self.logger.info(f"✅ Import preview prepared: {len(assets)} assets")
            return {'assets': assets}

        except Exception as e:
            self.logger.error(f"❌ Prepare import preview error: {e}", exc_info=True)
            return {
                'assets': [],
                'message': f'Error: {str(e)}'
            }

    async def import_selected_positions(
        self,
        portfolio_id: int,
        selected_assets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Import selected positions from exchange

        Args:
            selected_assets: List of assets with quantity and price
            [
                {'asset': 'BTC', 'symbol': 'BTCUSDT', 'quantity': 4.0, 'price': 43200},
                {'asset': 'SOL', 'symbol': 'SOLUSDT', 'quantity': 100.0, 'price': 145.5}
            ]

        Returns:
            {
                'imported_count': 2,
                'skipped_count': 0,
                'errors': []
            }
        """
        imported_count = 0
        skipped_count = 0
        errors = []

        for asset_data in selected_assets:
            try:
                position_id = await self.data_manager.create_position(
                    portfolio_id=portfolio_id,
                    symbol=asset_data['symbol'],
                    quantity=asset_data['quantity'],
                    entry_price=asset_data['price'],
                    side='LONG',
                    position_type='SPOT',
                    source='binance_import',
                    notes=f"Imported from exchange on {asset_data.get('import_date', 'unknown')}"
                )

                if position_id:
                    imported_count += 1
                    self.logger.info(f"Imported: {asset_data['symbol']} x{asset_data['quantity']}")
                else:
                    skipped_count += 1
                    errors.append(f"Failed to import {asset_data['symbol']}")

            except Exception as e:
                skipped_count += 1
                errors.append(f"Error importing {asset_data.get('symbol', 'unknown')}: {str(e)}")
                self.logger.error(f"Import error: {e}")

        # Update sync time
        if imported_count > 0:
            await self.data_manager.update_portfolio_sync_time(portfolio_id)

        return {
            'imported_count': imported_count,
            'skipped_count': skipped_count,
            'errors': errors
        }

    # ============================================
    # Legacy Method (for backward compatibility)
    # ============================================

    async def get_portfolio_summary_legacy(self):
        """Get portfolio summary with holdings (old method)"""
        # Get holdings
        holdings = await self.data_manager.get_portfolio_holdings()

        # Calculate summary
        total_value = sum(h['current_value'] or 0 for h in holdings)
        total_cost = sum(h['cost_basis'] or 0 for h in holdings)
        total_unrealized_pnl = sum(h['unrealized_pnl'] or 0 for h in holdings)
        total_realized_pnl = sum(h['realized_pnl'] or 0 for h in holdings)

        summary = {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_pnl_pct': ((total_unrealized_pnl + total_realized_pnl) / total_cost * 100) if total_cost > 0 else 0,
            'total_holdings': len(holdings)
        }

        return {
            'holdings': holdings,
            'summary': summary
        }
