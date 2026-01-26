#!/usr/bin/env python3
"""
modules/trading/trade_logger.py
SuperBot - Trade Logger
Author: SuperBot Team
Date: 2025-12-07
Versiyon: 1.0.0

Saves the trade history in JSON format.

Ozellikler:
- Entry/Exit kaydi
- Partial exit tracking
- Break-even, trailing stop olaylari
- Session bazli dosyalar (data/trades/YYYY-MM-DD_session.json)
- Daily and total summary

Usage:
    from modules.trading.trade_logger import TradeLogger

    logger = TradeLogger(strategy_name="simple_rsi", mode="paper")

    # Entry
    logger.log_entry(trade_id=1, symbol="BTCUSDT", side="LONG",
                     entry_price=95000, quantity=0.1, sl=94000, tp=97000)

    # Exit
    logger.log_exit(trade_id=1, exit_price=96000, exit_reason="TP",
                    pnl=100, pnl_pct=1.05)

    # Partial Exit
    logger.log_partial_exit(trade_id=1, level=1, exit_price=96000,
                            size_pct=40, pnl=40)

File Format:
    data/trades/simple_rsi_paper_2025-12-07.json

Bagimliliklar:
    - python>=3.12
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from core.logger_engine import get_logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeEvent:
    """A trade event (entry, exit, partial, break-even, trailing)"""
    event_type: str  # ENTRY, EXIT, PARTIAL_EXIT, BREAK_EVEN, TRAILING_UPDATE
    timestamp: str   # ISO format

    # Ortak alanlar
    trade_id: int
    symbol: str
    side: str  # LONG/SHORT

    # Price information
    price: float

    # For entry
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # For exit
    exit_reason: Optional[str] = None  # TP, SL, SIGNAL, TIMEOUT, MANUAL
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fee: Optional[float] = None

    # For partial exit
    partial_level: Optional[int] = None
    partial_size_pct: Optional[float] = None
    remaining_pct: Optional[float] = None

    # For trailing/break-even
    old_sl: Optional[float] = None
    new_sl: Optional[float] = None

    # Metadata
    notes: Optional[str] = None

    # AI Training: Indicator snapshot (entry/exit anindaki degerler)
    indicators: Optional[Dict[str, Any]] = None

    # AI Training: Market context
    market_context: Optional[Dict[str, Any]] = None


@dataclass
class TradeSummary:
    """A summary of a trade (from entry to exit)"""
    trade_id: int
    symbol: str
    side: str

    entry_time: str
    entry_price: float
    quantity: float
    initial_sl: Optional[float] = None
    initial_tp: Optional[float] = None

    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    final_sl: Optional[float] = None

    # PnL
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    net_pnl_pct: Optional[float] = None
    total_fee: Optional[float] = None

    # Partial exits
    partial_exits: List[Dict] = field(default_factory=list)
    partial_pnl: float = 0.0

    # Trailing/Break-even
    break_even_triggered: bool = False
    trailing_updates: int = 0
    max_profit_pct: Optional[float] = None

    # Recovery: Extreme prices (for trailing stop)
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None

    # Duration
    duration_minutes: Optional[int] = None

    # Status
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL

    # AI Training: Entry anindaki indicator snapshot
    entry_indicators: Optional[Dict[str, Any]] = None

    # AI Training: Exit anindaki indicator snapshot
    exit_indicators: Optional[Dict[str, Any]] = None

    # AI Training: Market context (volatility, trend, etc.)
    entry_market_context: Optional[Dict[str, Any]] = None
    exit_market_context: Optional[Dict[str, Any]] = None


# ============================================================================
# TRADE LOGGER
# ============================================================================

class TradeLogger:
    """
    Trade Logger - Trade record in JSON format.

    File structure:
        data/trades/
            simple_rsi_paper_2025-12-07.json    (daily file)
            simple_rsi_paper_summary.json        (summary of all trades)
    """

    def __init__(
        self,
        strategy_name: str = "unknown",
        mode: str = "paper",
        output_dir: Optional[Path] = None,
        logger: Any = None
    ):
        """
        Args:
            strategy_name: Strategy name (used in the filename)
            mode: paper/live
            output_dir: Cikti dizini (default: data/trades/)
            logger: Logger instance
        """
        self.strategy_name = strategy_name
        self.mode = mode
        self.logger = logger or get_logger("modules.trading.trade_logger")

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "data" / "trades"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Session file (gunluk)
        today = datetime.now().strftime("%Y-%m-%d")
        self.session_file = self.output_dir / f"{strategy_name}_{mode}_{today}.json"

        # Summary file (all trades)
        self.summary_file = self.output_dir / f"{strategy_name}_{mode}_summary.json"

        # In-memory storage
        self.events: List[TradeEvent] = []
        self.trades: Dict[int, TradeSummary] = {}  # trade_id -> summary

        # Load existing session
        self._load_session()

        self.logger.info(f"TradeLogger initialized: {self.session_file}")

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def log_entry(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        notes: Optional[str] = None,
        indicators: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record event.

        Args:
            trade_id: Trade ID
            symbol: Symbol (e.g., BTCUSDT)
            side: LONG or SHORT
            entry_price: Entry fiyati
            quantity: Quantity
            stop_loss: SL fiyati
            take_profit: TP fiyati
            notes: Ek notlar
            indicators: AI Training - entry anindaki indicator degerleri
                        {"rsi": 35.2, "ema_50": 95000, "atr": 1500, ...}
            market_context: AI Training - market status
                           {"volatility": "high", "trend": "bullish", "volume_ratio": 1.5, ...}
        """
        event = TradeEvent(
            event_type="ENTRY",
            timestamp=datetime.now().isoformat(),
            trade_id=trade_id,
            symbol=symbol,
            side=side.upper(),
            price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notes=notes,
            indicators=indicators,
            market_context=market_context
        )

        self.events.append(event)

        # Create trade summary
        self.trades[trade_id] = TradeSummary(
            trade_id=trade_id,
            symbol=symbol,
            side=side.upper(),
            entry_time=event.timestamp,
            entry_price=entry_price,
            quantity=quantity,
            initial_sl=stop_loss,
            initial_tp=take_profit,
            final_sl=stop_loss,
            status="OPEN",
            highest_price=entry_price,  # Initial value for recovery
            lowest_price=entry_price,   # Initial value for recovery
            entry_indicators=indicators,
            entry_market_context=market_context
        )

        self._save_session()
        self.logger.debug(f"Trade #{trade_id} ENTRY logged: {symbol} {side} @ ${entry_price:,.2f}")

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        fee: float = 0.0,
        notes: Optional[str] = None,
        indicators: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save exit event.

        Args:
            trade_id: Trade ID
            exit_price: Exit fiyati
            exit_reason: Exit nedeni (TP, SL, SIGNAL, TIMEOUT, MANUAL)
            pnl: Net PnL
            pnl_pct: Net PnL yuzdesi
            fee: Total fee
            notes: Ek notlar
            indicators: AI Training - exit anindaki indicator degerleri
            market_context: AI Training - market status
        """
        trade = self.trades.get(trade_id)
        if not trade:
            self.logger.warning(f"Trade #{trade_id} not found for EXIT")
            return

        event = TradeEvent(
            event_type="EXIT",
            timestamp=datetime.now().isoformat(),
            trade_id=trade_id,
            symbol=trade.symbol,
            side=trade.side,
            price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=fee,
            notes=notes,
            indicators=indicators,
            market_context=market_context
        )

        self.events.append(event)

        # Update trade summary
        trade.exit_time = event.timestamp
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.net_pnl = pnl
        trade.net_pnl_pct = pnl_pct
        trade.total_fee = fee
        trade.status = "CLOSED"
        trade.exit_indicators = indicators
        trade.exit_market_context = market_context

        # Duration hesapla
        try:
            entry_dt = datetime.fromisoformat(trade.entry_time)
            exit_dt = datetime.fromisoformat(trade.exit_time)
            trade.duration_minutes = int((exit_dt - entry_dt).total_seconds() / 60)
        except:
            pass

        self._save_session()
        self._save_summary()

        pnl_emoji = "+" if pnl >= 0 else ""
        self.logger.info(f"Trade #{trade_id} EXIT logged: {exit_reason} @ ${exit_price:,.2f} | PnL: {pnl_emoji}${pnl:,.2f} ({pnl_pct:+.2f}%)")

    def log_partial_exit(
        self,
        trade_id: int,
        level: int,
        exit_price: float,
        size_pct: float,
        remaining_pct: float,
        pnl: float,
        fee: float = 0.0,
        notes: Optional[str] = None
    ) -> None:
        """
        Record partial exit event.
        """
        trade = self.trades.get(trade_id)
        if not trade:
            self.logger.warning(f"Trade #{trade_id} not found for PARTIAL_EXIT")
            return

        event = TradeEvent(
            event_type="PARTIAL_EXIT",
            timestamp=datetime.now().isoformat(),
            trade_id=trade_id,
            symbol=trade.symbol,
            side=trade.side,
            price=exit_price,
            partial_level=level,
            partial_size_pct=size_pct,
            remaining_pct=remaining_pct,
            pnl=pnl,
            fee=fee,
            notes=notes
        )

        self.events.append(event)

        # Update trade summary
        trade.partial_exits.append({
            "level": level,
            "price": exit_price,
            "size_pct": size_pct,
            "pnl": pnl,
            "time": event.timestamp
        })
        trade.partial_pnl += pnl
        trade.status = "PARTIAL"

        self._save_session()
        self.logger.debug(f"Trade #{trade_id} PARTIAL_EXIT #{level} logged: {size_pct:.0f}% @ ${exit_price:,.2f}")

    def log_break_even(
        self,
        trade_id: int,
        old_sl: float,
        new_sl: float,
        current_price: float,
        notes: Optional[str] = None
    ) -> None:
        """
        Save break-even SL update.
        """
        trade = self.trades.get(trade_id)
        if not trade:
            return

        event = TradeEvent(
            event_type="BREAK_EVEN",
            timestamp=datetime.now().isoformat(),
            trade_id=trade_id,
            symbol=trade.symbol,
            side=trade.side,
            price=current_price,
            old_sl=old_sl,
            new_sl=new_sl,
            notes=notes
        )

        self.events.append(event)

        trade.break_even_triggered = True
        trade.final_sl = new_sl

        self._save_session()
        self.logger.debug(f"Trade #{trade_id} BREAK_EVEN: SL ${old_sl:,.2f} -> ${new_sl:,.2f}")

    def log_trailing_update(
        self,
        trade_id: int,
        old_sl: float,
        new_sl: float,
        current_price: float,
        notes: Optional[str] = None
    ) -> None:
        """
        Save trailing stop update
        """
        trade = self.trades.get(trade_id)
        if not trade:
            return

        event = TradeEvent(
            event_type="TRAILING_UPDATE",
            timestamp=datetime.now().isoformat(),
            trade_id=trade_id,
            symbol=trade.symbol,
            side=trade.side,
            price=current_price,
            old_sl=old_sl,
            new_sl=new_sl,
            notes=notes
        )

        self.events.append(event)

        trade.trailing_updates += 1
        trade.final_sl = new_sl

        self._save_session()
        self.logger.debug(f"Trade #{trade_id} TRAILING: SL ${old_sl:,.2f} -> ${new_sl:,.2f}")

    def update_extreme_prices(
        self,
        trade_id: int,
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None
    ) -> None:
        """
        Update the trade's highest/lowest price (for recovery).

        TradingEngine'de PositionManager.update_extreme_prices() cagrildiktan
        then this method is called so that the correct values are used in recovery.

        Args:
            trade_id: Trade ID
            highest_price: New highest price (for LONG)
            lowest_price: New lowest price (for SHORT)
        """
        trade = self.trades.get(trade_id)
        if not trade:
            return

        updated = False
        if highest_price is not None and (trade.highest_price is None or highest_price > trade.highest_price):
            trade.highest_price = highest_price
            updated = True
        if lowest_price is not None and (trade.lowest_price is None or lowest_price < trade.lowest_price):
            trade.lowest_price = lowest_price
            updated = True

        if updated:
            self._save_session()

    # ========================================================================
    # STATS & SUMMARY
    # ========================================================================

    def get_session_stats(self) -> Dict[str, Any]:
        """Gunluk session istatistikleri"""
        closed_trades = [t for t in self.trades.values() if t.status == "CLOSED"]

        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0
            }

        wins = [t for t in closed_trades if (t.net_pnl or 0) > 0]
        total_pnl = sum(t.net_pnl or 0 for t in closed_trades)

        return {
            "total_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(closed_trades) - len(wins),
            "win_rate": len(wins) / len(closed_trades) * 100,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed_trades),
            "total_partial_pnl": sum(t.partial_pnl for t in closed_trades)
        }

    def get_open_trades(self) -> List[TradeSummary]:
        """Freeze open trades"""
        return [t for t in self.trades.values() if t.status in ("OPEN", "PARTIAL")]

    def has_open_trades(self) -> bool:
        """Is there an open trade?"""
        return any(t.status in ("OPEN", "PARTIAL") for t in self.trades.values())

    def get_open_trades_for_recovery(self) -> List[Dict[str, Any]]:
        """
        Returns open trades for recovery in position dictionary format.

        When the TradingEngine restarts, the positions obtained from this method
        are loaded into the _positions dictionary, and trading continues.

        Returns:
            List[Dict]: List of position dictionaries (in TradingEngine format)

        Example:
            >>> logger = TradeLogger("test", "paper")
            >>> open_positions = logger.get_open_trades_for_recovery()
            >>> for pos in open_positions:
            ...     engine._positions[pos['symbol']] = pos
        """
        positions = []

        for trade in self.trades.values():
            if trade.status not in ("OPEN", "PARTIAL"):
                continue

            # TradingEngine position formatina cevir
            # Convert entry_time to datetime (it comes as a string from JSON)
            try:
                entry_time = datetime.fromisoformat(trade.entry_time)
            except:
                entry_time = trade.entry_time

            position = {
                'id': trade.trade_id,
                'position_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time': entry_time,
                'entry_price': trade.entry_price,
                'quantity': trade.quantity,
                'original_quantity': trade.quantity,
                'sl_price': trade.final_sl,
                'tp_price': trade.initial_tp,
                'stop_loss': trade.final_sl,
                'take_profit': trade.initial_tp,
                # Extreme prices: saved values or entry_price
                'highest_price': trade.highest_price or trade.entry_price,
                'lowest_price': trade.lowest_price or trade.entry_price,
                'completed_partial_exits': len(trade.partial_exits),
                # Recovery metadata
                '_recovered': True,
                '_recovered_at': datetime.now().isoformat(),
            }

            positions.append(position)

        return positions

    def get_last_trade_id(self) -> int:
        """
        Freeze the last trade ID (counter for the new trade)

        To initialize the trade_counter with the correct value during recovery.

        Returns:
            int: Last trade ID (0 if no trade exists)
        """
        if not self.trades:
            return 0
        return max(self.trades.keys())

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _save_session(self) -> None:
        """Save the session file"""
        try:
            data = {
                "strategy": self.strategy_name,
                "mode": self.mode,
                "last_updated": datetime.now().isoformat(),
                "events": [asdict(e) for e in self.events],
                "trades": {str(k): asdict(v) for k, v in self.trades.items()}
            }

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Session save error: {e}")

    def _load_session(self) -> None:
        """Load session file (if it exists)"""
        if not self.session_file.exists():
            return

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Events
            for e in data.get("events", []):
                self.events.append(TradeEvent(**e))

            # Trades
            for tid, t in data.get("trades", {}).items():
                self.trades[int(tid)] = TradeSummary(**t)

            self.logger.info(f"Loaded {len(self.events)} events, {len(self.trades)} trades from session")

        except Exception as e:
            self.logger.warning(f"Session load error: {e}")

    def _save_summary(self) -> None:
        """Update the summary file (all closed trades)"""
        try:
            # Load the current summary
            existing = []
            if self.summary_file.exists():
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)

            # Add the closed trades in this session.
            closed = [asdict(t) for t in self.trades.values() if t.status == "CLOSED"]

            # Duplicate kontrol (trade_id bazli)
            existing_ids = {t.get("trade_id") for t in existing}
            new_trades = [t for t in closed if t.get("trade_id") not in existing_ids]

            if new_trades:
                existing.extend(new_trades)

                with open(self.summary_file, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)

                self.logger.debug(f"Summary updated: {len(new_trades)} new trades")

        except Exception as e:
            self.logger.error(f"Summary save error: {e}")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TradeLogger Test")
    print("=" * 60)

    # Test instance
    logger = TradeLogger(strategy_name="test", mode="paper")

    # Entry
    print("\n1. Log Entry:")
    logger.log_entry(
        trade_id=1,
        symbol="BTCUSDT",
        side="LONG",
        entry_price=95000,
        quantity=0.1,
        stop_loss=94000,
        take_profit=97000
    )
    print(f"   Trade #1 created")

    # Break-even
    print("\n2. Log Break-even:")
    logger.log_break_even(
        trade_id=1,
        old_sl=94000,
        new_sl=95100,
        current_price=96000
    )
    print(f"   Break-even triggered")

    # Trailing
    print("\n3. Log Trailing Update:")
    logger.log_trailing_update(
        trade_id=1,
        old_sl=95100,
        new_sl=95500,
        current_price=96500
    )
    print(f"   Trailing SL updated")

    # Partial Exit
    print("\n4. Log Partial Exit:")
    logger.log_partial_exit(
        trade_id=1,
        level=1,
        exit_price=96500,
        size_pct=40,
        remaining_pct=60,
        pnl=60
    )
    print(f"   Partial exit #1")

    # Full Exit
    print("\n5. Log Exit:")
    logger.log_exit(
        trade_id=1,
        exit_price=97000,
        exit_reason="TP",
        pnl=120,
        pnl_pct=1.26,
        fee=5
    )

    # Stats
    print("\n6. Session Stats:")
    stats = logger.get_session_stats()
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']:.1f}%")
    print(f"   Total PnL: ${stats['total_pnl']:.2f}")

    print(f"\n   Session file: {logger.session_file}")
    print(f"   Summary file: {logger.summary_file}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
