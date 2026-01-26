#!/usr/bin/env python3
"""
components/managers/symbols_manager.py
SuperBot - Symbols Manager
Author: SuperBot Team
Date: 2025-10-28
Versiyon: 5.0.0

UPGRADE v5.0: Tier system removed - Use SymbolFavorite for priority management

Symbols Manager - Symbol loading, filtering, exchange sync, and storage

Features:
- Multi-exchange support (binance, bybit, okx)
- Exchange symbol synchronization (SPOT + FUTURES)
- Database storage (ExchangeSymbol, SymbolFavorite)
- Base + Quote asset logic
- Blacklist/whitelist (wildcard patterns)
- Cache integration
- Priority management via SymbolFavorite
- Error Handler integration
- Dynamic exchange filter registry

Symbol Management:
- Exchange symbols stored in database (exchange_symbols table)
- User favorites with priority 1-10 (symbol_favorites table)
- SPOT/FUTURES market type separation
- Tags, notes, color coding per favorite

Usage:
    sm = SymbolsManager(
        exchange_name='binance',
        config=config,
        logger=logger,
        error_handler=error_handler,
        cache_manager=cache,
        exchange_client=binance_client,
        data_manager=data_manager
    )

    await sm.initialize()

    # Sync symbols from exchange
    result = await sm.sync_from_exchange(market_type='both')

    # Get symbols
    symbols = await sm.get_available_symbols(market_type='SPOT')

    # Get favorites
    favorites = await data_manager.get_favorites()

Dependencies:
    - fnmatch (built-in)
"""

import asyncio
import fnmatch
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Callable, Tuple, Any

from core.logger_engine import LoggerEngine


class SymbolsManager:
    """
    Multi-source, Multi-exchange Symbol Manager with Database Integration

    Features:
    - Exchange symbol synchronization (SPOT + FUTURES)
    - Database storage via DataManager
    - User-based favorites with priority system
    - Market type separation (SPOT/FUTURES)
    - Blacklist/whitelist filtering
    - Cache integration

    Exchanges:
    - binance (default)
    - bybit, okx, kraken (future)
    """
    
    def __init__(
        self,
        exchange_name: str = 'binance',
        config: Optional[Dict] = None,
        logger: Optional[Any] = None,
        error_handler: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        exchange_client: Optional[Any] = None,
        data_manager: Optional[Any] = None,
        strategy: Optional[Any] = None,
        strategy_file: Optional[str] = None
    ):
        """
        Args:
            exchange_name: Exchange name (binance, bybit, okx)
            config: Config engine
            logger: Logger instance
            error_handler: ErrorHandler instance
            cache_manager: CacheManager instance
            exchange_client: Exchange client
            data_manager: DataManager instance
            strategy: Strategy instance (for source=strategy)
            strategy_file: Strategy file name (e.g., grok_scalp_king_5m_v2.py)
        """
        self.exchange_name = exchange_name
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.cache = cache_manager
        self.exchange_client = exchange_client
        self.data_manager = data_manager
        self.strategy = strategy
        self.strategy_file = strategy_file

        # State
        self.base_assets: Set[str] = set()
        self.quote_asset: str = "USDT"
        self.active_symbols: Set[str] = set()
        self.source: Optional[str] = None
        self.last_refresh: Optional[datetime] = None

        # Config
        self.cache_duration = 600  # 10 min

        # Exchange filter registry
        self.filter_methods: Dict[str, Callable] = {
            'top_volume': self._filter_top_volume,
            'top_gainers': self._filter_top_gainers,
            'top_losers': self._filter_top_losers,
            'all': self._filter_all,
        }

        if self.logger:
            self.logger.info(f"💱 SymbolManager initialized: {exchange_name}")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize(self):
        """Initialize Symbol Manager"""
        if self.logger:
            self.logger.info(f"🔧 Initializing SymbolsManager for {self.exchange_name}...")

        # Get config
        if self.config:
            # Try symbols.quote_asset first, fallback to exchange.trading.quote_asset
            self.quote_asset = self.config.get("symbols.quote_asset") or \
                             self.config.get("exchange.trading.quote_asset", "USDT")
            self.cache_duration = self.config.get("symbols.cache_duration", 600)

        if self.logger:
            self.logger.info(f"💱 Quote asset: {self.quote_asset}")

        # Load symbols (backward compatibility - will use database in future)
        await self.load_symbols()

        if self.logger:
            self.logger.info(
                f"✅ SymbolsManager ready: {len(self.base_assets)} base assets → "
                f"{len(self.active_symbols)} trading pairs"
            )
    
    # ========================================================================
    # CORE - LOAD SYMBOLS
    # ========================================================================
    
    async def load_symbols(self, force_refresh: bool = False):
        """
        Load symbols from configured source
        
        Priority determined by config: symbols.source
        """
        # Cache check
        if not force_refresh and await self._is_cache_valid():
            if self.logger:
                self.logger.debug("📦 Using cached symbols")
            return
        
        # Get source from config
        source = self.config.get("symbols.source", "config") if self.config else "config"
        
        if self.logger:
            self.logger.info(f"📊 Loading symbols from source: {source}")
        
        try:
            if source == "config":
                await self._load_from_config()

            elif source == "exchange":
                if self.exchange_client:
                    await self._load_from_exchange()
                else:
                    if self.logger:
                        self.logger.warning("⚠️ No exchange client, fallback to config")
                    await self._load_from_config()

            elif source == "file":
                await self._load_from_file()

            elif source == "strategy":
                success = await self._load_from_strategy()
                if not success:
                    if self.logger:
                        self.logger.warning("⚠️ Strategy source failed, fallback to config")
                    await self._load_from_config()

            else:
                if self.logger:
                    self.logger.error(f"❌ Unknown source: {source}, using config")
                await self._load_from_config()
            
            # Build trading pairs
            self._build_trading_pairs()
            
            # Apply blacklist
            self._apply_exclusions()
            
            # Update cache
            self.last_refresh = datetime.now()
            if self.cache:
                await self._cache_symbols()
            
            # Auto-add to watchlist (if data_manager available)
            if self.data_manager:
                await self._update_watchlist()

            # Save to storage (JSON or Database)
            await self.save_to_storage()

            if self.logger:
                self.logger.info(
                    f"✅ Loaded {len(self.base_assets)} base assets → "
                    f"{len(self.active_symbols)} pairs"
                )
                self.logger.debug(f"📊 Trading pairs: {sorted(list(self.active_symbols))}")
        
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, context={'operation': 'load_symbols'})
            elif self.logger:
                self.logger.error(f"❌ Symbol loading failed: {e}")
            
            # Emergency fallback
            self.base_assets = {"BTC", "ETH"}
            self._build_trading_pairs()
            self.source = "fallback"
    
    # ========================================================================
    # SOURCE: CONFIG
    # ========================================================================
    
    async def _load_from_config(self):
        """Load base assets from config"""
        config_list = self.config.get("symbols.config_list", []) if self.config else []
        
        if not config_list:
            # Fallback
            fallback = ["BTC", "ETH"]
            if self.logger:
                self.logger.warning(f"⚠️ No config symbols, using fallback: {fallback}")
            config_list = fallback
        
        # Extract base assets
        base_assets = []
        for item in config_list:
            base = self._extract_base_from_symbol(item)
            base_assets.append(base)
        
        self.base_assets = set(base_assets)
        self.source = "config"
        
        if self.logger:
            self.logger.info(f"📋 Config base assets: {sorted(list(self.base_assets))}")
    
    # ========================================================================
    # SOURCE: EXCHANGE
    # ========================================================================
    
    async def _load_from_exchange(self):
        """Load base assets from exchange"""
        if not self.exchange_client:
            raise Exception("Exchange client required")
        
        if not self.config:
            await self._load_from_config()
            return
        
        exchange_filter = self.config.get("symbols.exchange_filter", {})
        method = exchange_filter.get("method", "top_volume")
        limit = exchange_filter.get("limit", 20)
        
        if self.logger:
            self.logger.info(f"🔄 Fetching from exchange: method={method}, limit={limit}")
        
        try:
            # Get filter function from registry
            if method in self.filter_methods:
                filter_func = self.filter_methods[method]
                result = await filter_func(limit, exchange_filter)
                
                # Extract base assets
                base_assets = result[0] if isinstance(result, tuple) else result
            else:
                if self.logger:
                    self.logger.error(f"❌ Unknown filter method: {method}")
                result = await self._filter_top_volume(limit, exchange_filter)
                base_assets = result[0] if isinstance(result, tuple) else result
            
            self.base_assets = set(base_assets)
            self.source = "exchange"
            
            if self.logger:
                self.logger.info(
                    f"📊 Exchange base assets ({method}): {sorted(list(self.base_assets))}"
                )
        
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, context={'operation': 'load_from_exchange'})
            elif self.logger:
                self.logger.error(f"❌ Exchange fetch failed: {e}")
            
            # Fallback to config
            await self._load_from_config()
    
    async def _filter_top_volume(self, limit: int, filters: Dict) -> Tuple[List[str], Dict[str, float]]:
        """Filter: Top volume - Returns (base_assets, volumes)"""
        tickers = await self.exchange_client.get_24h_tickers()
        min_volume = filters.get("min_volume_24h", 0)
        
        candidates = []
        volumes = {}
        
        for ticker in tickers:
            symbol = ticker['symbol']
            
            # Must end with our quote asset
            if not symbol.endswith(self.quote_asset):
                continue
            
            base = self._extract_base_from_symbol(symbol)
            volume = float(ticker.get('quoteVolume', 0))
            
            if volume >= min_volume:
                candidates.append({'base': base, 'volume': volume})
                volumes[f"{base}{self.quote_asset}"] = volume
        
        # Sort by volume
        candidates.sort(key=lambda x: x['volume'], reverse=True)
        
        base_assets = [c['base'] for c in candidates[:limit]]
        return base_assets, volumes
    
    async def _filter_top_gainers(self, limit: int, filters: Dict) -> List[str]:
        """Filter: Top gainers (24h)"""
        tickers = await self.exchange_client.get_24h_tickers()
        min_volume = filters.get("min_volume_24h", 0)
        
        candidates = []
        for ticker in tickers:
            symbol = ticker['symbol']
            
            if not symbol.endswith(self.quote_asset):
                continue
            
            base = self._extract_base_from_symbol(symbol)
            volume = float(ticker.get('quoteVolume', 0))
            price_change = float(ticker.get('priceChangePercent', 0))
            
            if volume >= min_volume and price_change > 0:
                candidates.append({'base': base, 'change': price_change, 'volume': volume})
        
        # Sort by price change
        candidates.sort(key=lambda x: x['change'], reverse=True)
        
        return [c['base'] for c in candidates[:limit]]
    
    async def _filter_top_losers(self, limit: int, filters: Dict) -> List[str]:
        """Filter: Top losers (24h)"""
        tickers = await self.exchange_client.get_24h_tickers()
        min_volume = filters.get("min_volume_24h", 0)
        
        candidates = []
        for ticker in tickers:
            symbol = ticker['symbol']
            
            if not symbol.endswith(self.quote_asset):
                continue
            
            base = self._extract_base_from_symbol(symbol)
            volume = float(ticker.get('quoteVolume', 0))
            price_change = float(ticker.get('priceChangePercent', 0))
            
            if volume >= min_volume and price_change < 0:
                candidates.append({'base': base, 'change': price_change, 'volume': volume})
        
        # Sort by price change (ascending)
        candidates.sort(key=lambda x: x['change'])
        
        return [c['base'] for c in candidates[:limit]]
    
    async def _filter_all(self, limit: int, filters: Dict) -> List[str]:
        """Filter: All trading pairs"""
        tickers = await self.exchange_client.get_24h_tickers()
        min_volume = filters.get("min_volume_24h", 0)
        
        candidates = []
        for ticker in tickers:
            symbol = ticker['symbol']
            
            if not symbol.endswith(self.quote_asset):
                continue
            
            base = self._extract_base_from_symbol(symbol)
            volume = float(ticker.get('quoteVolume', 0))
            
            if volume >= min_volume:
                candidates.append({'base': base, 'volume': volume})
        
        # Sort by volume
        candidates.sort(key=lambda x: x['volume'], reverse=True)
        
        return [c['base'] for c in candidates[:limit]]
    
    # ========================================================================
    # SOURCE: STRATEGY
    # ========================================================================

    async def _load_from_strategy(self) -> bool:
        """
        Load base assets from strategy

        Priority:
        1. self.strategy (injected instance) - Comes from TradingEngine
        2. Load strategy file from Config (fallback)
        """
        try:
            strategy_instance = None

            # Priority 1: Use injected strategy instance
            if self.strategy:
                strategy_instance = self.strategy
                if self.logger:
                    self.logger.debug("📋 Using injected strategy instance")

            # Priority 2: Load from file (fallback)
            elif self.config:
                strategy_instance = await self._load_strategy_from_file()

            if not strategy_instance:
                return False

            # Check for quote asset override
            if hasattr(strategy_instance, 'quote_asset'):
                self.quote_asset = strategy_instance.quote_asset
                if self.logger:
                    self.logger.info(f"💱 Strategy quote override: {self.quote_asset}")

            # Extract base assets
            base_assets = self._extract_base_assets_from_strategy(strategy_instance)

            if base_assets:
                self.base_assets = set(base_assets)
                self.source = "strategy"

                # Get strategy identifier (prefer file name)
                strategy_id = (
                    self.strategy_file or
                    getattr(strategy_instance, 'strategy_name', None) or
                    getattr(strategy_instance, 'name', None) or
                    strategy_instance.__class__.__name__
                )
                if self.logger:
                    self.logger.info(
                        f"📋 Strategy '{strategy_id}' -> {len(base_assets)} symbols: "
                        f"{sorted(list(self.base_assets))}"
                    )
                return True

            return False

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, context={'operation': 'load_from_strategy'})
            elif self.logger:
                self.logger.error(f"❌ Strategy loading error: {e}")
            return False

    async def _load_strategy_from_file(self) -> Optional[Any]:
        """Load strategy from file (fallback when no instance injected)"""
        if not self.config:
            return None

        strategy_name = self.config.get("strategies.active_strategy")
        if not strategy_name:
            return None

        strategy_folder = Path(
            self.config.get("strategies.config_folder", "strategies/configs")
        )
        strategy_file = strategy_folder / f"{strategy_name}.py"

        if not strategy_file.exists():
            if self.logger:
                self.logger.warning(f"⚠️ Strategy file not found: {strategy_file}")
            return None

        # Import strategy
        spec = importlib.util.spec_from_file_location(
            f"strategy_{strategy_name}",
            strategy_file
        )
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get strategy config
        if hasattr(module, 'get_strategy_config'):
            return module.get_strategy_config()

        return None
    
    def _extract_base_assets_from_strategy(self, strategy_instance) -> List[str]:
        """
        Extract base assets from strategy instance

        Supports multiple SymbolConfig formats:
        - symbol: List[str] → ['BTC', 'ETH', 'SOL']
        - symbol: str → 'BTCUSDT'
        - asset: str → 'BTC'
        """
        base_assets = []

        # Method 1: symbols attribute (List[SymbolConfig])
        if hasattr(strategy_instance, 'symbols'):
            for symbol_config in strategy_instance.symbols:
                # Skip disabled
                if hasattr(symbol_config, 'enabled') and not symbol_config.enabled:
                    continue

                # Case A: asset attribute (direct base)
                if hasattr(symbol_config, 'asset'):
                    base_assets.append(symbol_config.asset)

                # Case B: symbol attribute
                elif hasattr(symbol_config, 'symbol'):
                    symbols = symbol_config.symbol

                    # symbol can be list or string
                    if isinstance(symbols, list):
                        # ['BTC', 'ETH', 'SOL'] → each is base
                        for sym in symbols:
                            base = self._extract_base_from_symbol(sym)
                            base_assets.append(base)
                    else:
                        # 'BTCUSDT' or 'BTC' → extract base
                        base = self._extract_base_from_symbol(symbols)
                        base_assets.append(base)

        # Method 2: get_symbols_to_trade()
        elif hasattr(strategy_instance, 'get_symbols_to_trade'):
            symbols = strategy_instance.get_symbols_to_trade()
            for symbol in symbols:
                base = self._extract_base_from_symbol(symbol)
                base_assets.append(base)

        # Method 3: Direct assets list
        elif hasattr(strategy_instance, 'assets'):
            base_assets = list(strategy_instance.assets)

        return base_assets

    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _extract_base_from_symbol(self, symbol: str) -> str:
        """Extract base asset from symbol"""
        # If already base (short), return as is
        if len(symbol) <= 5:
            return symbol.upper()

        # Try to extract base by removing known quote assets
        known_quotes = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "EUR", "GBP"]

        for quote in known_quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:
                    return base.upper()

        # Fallback: first 3-4 chars
        return symbol[:4].upper()

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to standard format

        Handles:
        - ETH → ETHUSDT (base asset → full symbol)
        - ETHUSDT → ETHUSDT (already correct)
        - ETHUSDTUSDT → ETHUSDT (double quote fix)

        Args:
            symbol: Raw symbol input

        Returns:
            str: Normalized symbol (e.g., ETHUSDT)
        """
        symbol = symbol.upper().strip()
        quote = self.quote_asset

        # Case 1: Double quote fix (ETHUSDTUSDT → ETHUSDT)
        double_quote = f"{quote}{quote}"
        if symbol.endswith(double_quote):
            symbol = symbol[:-len(quote)]
            if self.logger:
                self.logger.debug(f"🔧 Double quote corrected: {symbol}{quote} -> {symbol}")

        # Case 2: Already has quote (ETHUSDT → ETHUSDT)
        if symbol.endswith(quote):
            return symbol

        # Case 3: Base only (ETH → ETHUSDT)
        return f"{symbol}{quote}"
    
    def _build_trading_pairs(self):
        """Build trading pairs from base assets (normalized)"""
        pairs = set()

        for base in self.base_assets:
            # Skip if base = quote (can't trade BTC/BTC)
            if base == self.quote_asset:
                if self.logger:
                    self.logger.debug(f"⏭️ Skipping {base} (same as quote asset)")
                continue

            # Build and normalize pair
            pair = self._normalize_symbol(base)
            pairs.add(pair)

        self.active_symbols = pairs

        if not pairs and self.logger:
            self.logger.warning("⚠️ No valid trading pairs! Check base/quote assets")
    
    def _apply_exclusions(self):
        """Apply blacklist"""
        if not self.config:
            return
        
        exclusion_list = self.config.get("symbols.exclusion_list", [])
        
        if not exclusion_list:
            return
        
        excluded = set()
        
        for symbol in list(self.active_symbols):
            for pattern in exclusion_list:
                if fnmatch.fnmatch(symbol, pattern):
                    excluded.add(symbol)
                    self.active_symbols.discard(symbol)

                    # Also remove from base assets
                    base = self._extract_base_from_symbol(symbol)
                    self.base_assets.discard(base)
                    break
        
        if excluded and self.logger:
            self.logger.info(f"🚫 Excluded {len(excluded)} symbols: {sorted(list(excluded))}")
    
    # ========================================================================
    # CACHE
    # ========================================================================
    
    async def _is_cache_valid(self) -> bool:
        """Check if cache is valid"""
        if not self.cache or not self.last_refresh:
            return False

        # Check cache
        cached = self.cache.get(f'symbols:{self.exchange_name}:{self.source}')

        if cached:
            self.base_assets = set(cached['base_assets'])
            self.active_symbols = set(cached['active_symbols'])
            self.quote_asset = cached['quote_asset']
            self.source = cached['source']
            return True

        return False
    
    async def _cache_symbols(self):
        """Cache symbols"""
        if not self.cache:
            return

        self.cache.set(
            f'symbols:{self.exchange_name}:{self.source}',
            {
                'base_assets': list(self.base_assets),
                'active_symbols': list(self.active_symbols),
                'quote_asset': self.quote_asset,
                'source': self.source,
            },
            ttl=self.cache_duration
        )
    
    # ========================================================================
    # WATCHLIST
    # ========================================================================
    
    async def _update_watchlist(self):
        """Auto-add symbols to watchlist"""
        if not self.data_manager:
            return

        for symbol in self.active_symbols:
            await self.data_manager.add_to_watchlist(
                symbol=symbol,
                is_trading=True,
                is_favorite=False,  # User sets favorites via WebUI
                notes=None,
                tags=None
            )
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def get_active_symbols(self) -> List[str]:
        """Get active trading pairs

        Returns:
            List of symbols
        """
        return sorted(list(self.active_symbols))
    
    def get_base_assets(self) -> List[str]:
        """Get base assets"""
        return sorted(list(self.base_assets))
    
    def get_quote_asset(self) -> str:
        """Get quote asset"""
        return self.quote_asset
    
    def change_quote_asset(self, new_quote: str):
        """Change quote asset and rebuild pairs"""
        old_quote = self.quote_asset
        self.quote_asset = new_quote.upper()
        
        if self.logger:
            self.logger.info(f"💱 Changing quote: {old_quote} → {self.quote_asset}")

        self._build_trading_pairs()
        self._apply_exclusions()
        
        if self.logger:
            self.logger.info(f"✅ New pairs: {sorted(list(self.active_symbols))}")
    
    def is_symbol_active(self, symbol: str) -> bool:
        """Check if symbol is active"""
        return symbol in self.active_symbols
    
    def get_source(self) -> str:
        """Get symbol source"""
        return self.source or "unknown"
    
    def get_stats(self) -> Dict:
        """Get symbol manager statistics"""
        return {
            'exchange': self.exchange_name,
            'source': self.source,
            'quote_asset': self.quote_asset,
            'base_assets': len(self.base_assets),
            'active_symbols': len(self.active_symbols),
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None
        }
    
    async def refresh(self):
        """Force refresh symbols"""
        await self.load_symbols(force_refresh=True)

        # Save to storage after refresh
        await self.save_to_storage()

    # ========================================================================
    # STORAGE: Save to JSON or Database
    # ========================================================================

    async def save_to_storage(self):
        """Save symbols to configured storage (JSON or Database)"""
        if not self.config:
            return

        storage_type = self.config.get("symbols.storage.type", "json")

        try:
            if storage_type == "json":
                await self._save_to_json()
            elif storage_type == "database":
                await self._save_to_database()
            else:
                if self.logger:
                    self.logger.warning(f"⚠️ Unknown storage type: {storage_type}, skipping save")
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, context={'operation': 'save_to_storage'})
            elif self.logger:
                self.logger.error(f"❌ Failed to save symbols: {e}")

    async def _save_to_json(self):
        """Save symbols to JSON file"""
        import json
        from datetime import datetime

        output_file = self.config.get("symbols.storage.json.output_file", "data/json/active_symbols.json")

        data = {
            "asset": sorted(list(self.base_assets)),  # Base assets only
            "quote_asset": self.quote_asset,
            "timestamp": int(datetime.now().timestamp()),
            "source": self.source or "unknown",
            "method": self.config.get("symbols.exchange_filter.method", ""),
            "count": len(self.base_assets),
            "exclusions_applied": bool(self.config.get("symbols.exclusion_list")),
            "last_update": datetime.utcnow().isoformat() + "Z"
        }

        # Write to file
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        if self.logger:
            self.logger.info(f"💾 Saved {len(self.base_assets)} symbols to {output_file}")

    async def _save_to_database(self):
        """Save symbols to database (uses infrastructure.yaml)"""
        if self.logger:
            self.logger.warning("⚠️ Database storage not implemented yet")
        # TODO: Implement database storage
        # table = self.config.get("symbols.storage.database.table", "active_symbols")
        # self.db.execute(f"DELETE FROM {table}")
        # for asset in self.base_assets:
        #     self.db.execute(INSERT INTO...)

    # ========================================================================
    # SOURCE: FILE (NEW)
    # ========================================================================

    async def _load_from_file(self):
        """Load symbols from JSON file"""
        import json

        file_path = self.config.get("symbols.file_source", "data/json/basic_symbols.json")

        if self.logger:
            self.logger.info(f"📂 Loading symbols from file: {file_path}")

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Extract assets
            assets = data.get("asset", [])
            quote = data.get("quote_asset", "USDT")

            self.base_assets = set(assets)
            self.quote_asset = quote
            self.source = "file"

            if self.logger:
                self.logger.info(f"✅ Loaded {len(self.base_assets)} symbols from file")

        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"❌ File not found: {file_path}")
            # Fallback to config
            await self._load_from_config()
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to load from file: {e}")
            await self._load_from_config()

    # ========================================================================
    # EXCHANGE SYNC - Download and save symbols to database
    # ========================================================================

    async def sync_from_exchange(
        self,
        market_type: str = 'both',  # 'spot', 'futures', 'both'
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Sync symbols from exchange and save to database

        Process:
        1. Download from exchange API
        2. Save to JSON files (data/json/exchange_spot.json, data/json/exchange_futures.json)
        3. Load from JSON and save to database

        Args:
            market_type: Which market to sync ('spot', 'futures', 'both')
            force: Force sync even if recently synced

        Returns:
            dict: Sync result with status and counts
        """
        import json
        from pathlib import Path

        try:
            if self.logger:
                self.logger.info(f"🔄 Starting exchange sync - market: {market_type}, force: {force}")

            # Check if sync needed (unless forced)
            if not force and self.data_manager:
                if market_type in ['spot', 'both']:
                    last_sync = await self.data_manager.get_last_sync_time('SPOT')
                    if last_sync and (datetime.now() - last_sync).total_seconds() < 86400:  # 24 hours
                        if self.logger:
                            self.logger.info("⏭️  SPOT already synced today, skipping...")
                        if market_type == 'spot':
                            return {'status': 'skipped', 'reason': 'Already synced today', 'market_type': 'SPOT'}

                if market_type in ['futures', 'both']:
                    last_sync = await self.data_manager.get_last_sync_time('FUTURES')
                    if last_sync and (datetime.now() - last_sync).total_seconds() < 86400:
                        if self.logger:
                            self.logger.info("⏭️  FUTURES already synced today, skipping...")
                        if market_type == 'futures':
                            return {'status': 'skipped', 'reason': 'Already synced today', 'market_type': 'FUTURES'}

            # Ensure data/json directory exists
            json_dir = Path(__file__).parent.parent.parent / 'data' / 'json'
            json_dir.mkdir(parents=True, exist_ok=True)

            all_symbols = []

            # Sync SPOT
            if market_type in ['spot', 'both']:
                if self.logger:
                    self.logger.info("📥 Syncing SPOT symbols from exchange...")
                spot_symbols = await self._sync_spot_symbols()

                # Save to JSON
                spot_json_path = json_dir / 'exchange_spot.json'
                with open(spot_json_path, 'w', encoding='utf-8') as f:
                    json.dump(spot_symbols, f, indent=2, ensure_ascii=False)
                if self.logger:
                    self.logger.info(f"💾 Saved {len(spot_symbols)} SPOT symbols to {spot_json_path}")

                all_symbols.extend(spot_symbols)

            # Sync FUTURES
            if market_type in ['futures', 'both']:
                if self.logger:
                    self.logger.info("📥 Syncing FUTURES symbols from exchange...")
                futures_symbols = await self._sync_futures_symbols()

                # Save to JSON
                futures_json_path = json_dir / 'exchange_futures.json'
                with open(futures_json_path, 'w', encoding='utf-8') as f:
                    json.dump(futures_symbols, f, indent=2, ensure_ascii=False)
                if self.logger:
                    self.logger.info(f"💾 Saved {len(futures_symbols)} FUTURES symbols to {futures_json_path}")

                all_symbols.extend(futures_symbols)

            # Save to database
            if self.data_manager and all_symbols:
                saved_count = await self.data_manager.save_exchange_symbols(all_symbols)
                if self.logger:
                    self.logger.info(f"✅ Saved {saved_count} symbols to database")

                return {
                    'status': 'success',
                    'symbols_count': len(all_symbols),
                    'saved_count': saved_count,
                    'spot': len([s for s in all_symbols if s['market_type'] == 'SPOT']),
                    'futures': len([s for s in all_symbols if s['market_type'] == 'FUTURES'])
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No data_manager available or no symbols found'
                }

        except Exception as e:
            print(f"Sync error: {e}")
            if self.logger:
                self.logger.error(f"Exchange sync error: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _sync_spot_symbols(self) -> List[Dict[str, Any]]:
        """Sync SPOT symbols from Binance"""
        try:
            if not self.exchange_client:
                print("⚠️  No exchange client available")
                return []

            # Get exchange info
            # ConnectorEngine uses python-binance client internally
            if hasattr(self.exchange_client, 'get_exchange_info'):
                exchange_info = await self.exchange_client.get_exchange_info()
            elif hasattr(self.exchange_client, 'client'):
                # Direct access to binance client
                exchange_info = self.exchange_client.client.get_exchange_info()
            else:
                print("❌ Exchange client doesn't have get_exchange_info method")
                return []

            symbols_data = exchange_info.get('symbols', [])

            processed_symbols = []
            for symbol_info in symbols_data:
                # Only SPOT and TRADING status
                if symbol_info.get('status') != 'TRADING':
                    continue

                # Extract filters
                filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
                price_filter = filters.get('PRICE_FILTER', {})
                lot_size = filters.get('LOT_SIZE', {})
                min_notional = filters.get('NOTIONAL', {}) or filters.get('MIN_NOTIONAL', {})

                processed_symbols.append({
                    'symbol': symbol_info['symbol'],
                    'base_asset': symbol_info['baseAsset'],
                    'quote_asset': symbol_info['quoteAsset'],
                    'market_type': 'SPOT',
                    'status': symbol_info['status'],
                    'is_active': symbol_info['status'] == 'TRADING',
                    'spot_enabled': True,
                    'spot_trading': symbol_info.get('isSpotTradingAllowed', False),
                    'futures_enabled': False,
                    'contract_type': None,
                    'price_precision': symbol_info.get('quotePrecision'),
                    'quantity_precision': symbol_info.get('baseAssetPrecision'),
                    'base_asset_precision': symbol_info.get('baseAssetPrecision'),
                    'quote_precision': symbol_info.get('quotePrecision'),
                    'min_price': float(price_filter.get('minPrice', 0)) if price_filter else None,
                    'max_price': float(price_filter.get('maxPrice', 0)) if price_filter else None,
                    'tick_size': float(price_filter.get('tickSize', 0)) if price_filter else None,
                    'min_qty': float(lot_size.get('minQty', 0)) if lot_size else None,
                    'max_qty': float(lot_size.get('maxQty', 0)) if lot_size else None,
                    'step_size': float(lot_size.get('stepSize', 0)) if lot_size else None,
                    'min_notional': float(min_notional.get('minNotional', 0)) if min_notional else None,
                    'extra_data': {'permissions': symbol_info.get('permissions', [])}
                })

            return processed_symbols

        except Exception as e:
            print(f"❌ SPOT sync error: {e}")
            if self.logger:
                self.logger.error(f"❌ SPOT symbols sync error: {e}")
            return []

    async def _sync_futures_symbols(self) -> List[Dict[str, Any]]:
        """Sync FUTURES symbols from Binance"""
        try:
            if not self.exchange_client:
                print("⚠️  No exchange client available")
                return []

            # Get futures exchange info using python-binance client
            if hasattr(self.exchange_client, 'get_futures_exchange_info'):
                futures_info = await self.exchange_client.get_futures_exchange_info()
            elif hasattr(self.exchange_client, 'client'):
                try:
                    futures_info = self.exchange_client.client.futures_exchange_info()
                except Exception as e:
                    print(f"❌ Futures API error: {e}")
                    print("💡 Note: Futures might need separate API keys or testnet doesn't support it")
                    return []
            else:
                print("❌ Exchange client doesn't have futures methods")
                return []

            symbols_data = futures_info.get('symbols', [])
            processed_symbols = []

            for symbol_info in symbols_data:
                if symbol_info.get('status') != 'TRADING':
                    continue

                filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
                price_filter = filters.get('PRICE_FILTER', {})
                lot_size = filters.get('LOT_SIZE', {})
                min_notional = filters.get('MIN_NOTIONAL', {})

                processed_symbols.append({
                    'symbol': symbol_info['symbol'],
                    'base_asset': symbol_info.get('baseAsset', ''),
                    'quote_asset': symbol_info.get('quoteAsset', ''),
                    'market_type': 'FUTURES',
                    'status': symbol_info['status'],
                    'is_active': symbol_info['status'] == 'TRADING',
                    'spot_enabled': False,
                    'spot_trading': False,
                    'futures_enabled': True,
                    'contract_type': symbol_info.get('contractType', 'PERPETUAL'),
                    'price_precision': symbol_info.get('pricePrecision'),
                    'quantity_precision': symbol_info.get('quantityPrecision'),
                    'base_asset_precision': symbol_info.get('baseAssetPrecision'),
                    'quote_precision': symbol_info.get('quotePrecision'),
                    'min_price': float(price_filter.get('minPrice', 0)) if price_filter else None,
                    'max_price': float(price_filter.get('maxPrice', 0)) if price_filter else None,
                    'tick_size': float(price_filter.get('tickSize', 0)) if price_filter else None,
                    'min_qty': float(lot_size.get('minQty', 0)) if lot_size else None,
                    'max_qty': float(lot_size.get('maxQty', 0)) if lot_size else None,
                    'step_size': float(lot_size.get('stepSize', 0)) if lot_size else None,
                    'min_notional': float(min_notional.get('notional', 0)) if min_notional else None,
                    'extra_data': {
                        'contract_type': symbol_info.get('contractType'),
                        'delivery_date': symbol_info.get('deliveryDate'),
                        'onboard_date': symbol_info.get('onboardDate')
                    }
                })

            return processed_symbols

        except Exception as e:
            print(f"❌ FUTURES sync error: {e}")
            if self.logger:
                self.logger.error(f"❌ FUTURES symbols sync error: {e}")
            return []

    async def get_available_symbols(
        self,
        market_type: Optional[str] = None,
        quote_asset: str = 'USDT',
        search: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get available symbols from database

        Args:
            market_type: Filter by market type
            quote_asset: Filter by quote asset
            search: Search in symbol name
            limit: Max results

        Returns:
            List of symbols with metadata
        """
        try:
            if not self.data_manager:
                return []

            symbols = await self.data_manager.get_exchange_symbols(
                market_type=market_type,
                quote_asset=quote_asset,
                is_active=True,
                search=search,
                limit=limit
            )

            return symbols

        except Exception as e:
            print(f"❌ Error getting available symbols: {e}")
            if self.logger:
                self.logger.error(f"❌ Get available symbols error: {e}")
            return []


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test code removed in v5.0.0 (tier system removed)
    # Use WebUI dashboard or pytest for testing
    print("ℹ️  SymbolsManager v5.0.0")
    print("Use WebUI dashboard or pytest for testing")
    print("Tier system removed - Use SymbolFavorite for priority management")