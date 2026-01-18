#!/usr/bin/env python3
"""
components/strategies/signal_manager.py
SuperBot - Signal Manager

Version: 1.0.0
Date: 2025-11-17
Author: SuperBot Team

Description:
    Signal generation, logging, and persistence management.

    - Signal generation (entry/exit)
    - Signal logging and persistence
    - Signal history tracking
    - Signal filtering
    - Signal statistics

Usage:
    from components.strategies.signal_manager import SignalManager

    manager = SignalManager(logger=logger)

    # Generate signal
    signal = manager.generate_signal(
        symbol='BTCUSDT',
        signal_type='LONG',
        score=0.85,
        metadata={'strategy': 'EMA_Cross', 'conditions_met': [...]}
    )

    # Signal'i kaydet
    manager.save_signal(signal)

    # Signal history'yi al
    history = manager.get_signal_history(symbol='BTCUSDT', limit=100)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.logger_engine import LoggerEngine


class SignalType(str, Enum):
    """Signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NONE = "NONE"


class SignalSource(str, Enum):
    """Signal source"""
    STRATEGY = "STRATEGY"
    AI = "AI"
    FUSION = "FUSION"
    MANUAL = "MANUAL"


@dataclass
class Signal:
    """
    Signal data structure

    It contains all the information about a trading signal.
    """
    # Core fields
    signal_id: str
    symbol: str
    signal_type: SignalType
    timestamp: datetime

    # Signal quality
    score: float  # 0.0 - 1.0 (confidence)
    source: SignalSource = SignalSource.STRATEGY

    # Price info
    price: Optional[float] = None
    timeframe: Optional[str] = None

    # Strategy metadata
    strategy_name: Optional[str] = None
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    is_active: bool = True
    is_executed: bool = False
    execution_time: Optional[datetime] = None
    execution_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the signal to a dictionary (for JSON serialization)"""
        data = asdict(self)
        # Convert datetimes to strings
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        data['execution_time'] = self.execution_time.isoformat() if self.execution_time else None
        # Convert enums to string
        data['signal_type'] = self.signal_type.value
        data['source'] = self.source.value
        return data


class SignalManager:
    """
    Signal Manager - Signal generation, logging, and tracking.

    Responsibilities:
    1. Signal generation (entry/exit)
    2. Signal validation and filtering
    3. Signal persistence (JSON/Database)
    4. Signal history tracking
    5. Signal statistics
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        data_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the SignalManager.

        Args:
            logger: Logger instance
            data_dir: Signal data directory (default: data/signals/)
        """
        # Logger
        if logger:
            self.logger = logger
        else:
            logger_engine = LoggerEngine()
            self.logger = logger_engine.get_logger(__name__)

        # Data directory
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = Path("data/signals")

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory signal storage
        self.active_signals: Dict[str, List[Signal]] = {}  # symbol -> [signals]
        self.signal_history: List[Signal] = []

        # Statistics
        self.total_signals = 0
        self.signals_by_type: Dict[str, int] = {
            'LONG': 0,
            'SHORT': 0,
            'EXIT_LONG': 0,
            'EXIT_SHORT': 0,
            'NONE': 0
        }
        self.signals_by_source: Dict[str, int] = {
            'STRATEGY': 0,
            'AI': 0,
            'FUSION': 0,
            'MANUAL': 0
        }

        self.logger.info("✅ SignalManager started")
        self.logger.info(f"   Data directory: {self.data_dir}")

    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================

    def generate_signal(
        self,
        symbol: str,
        signal_type: str | SignalType,
        score: float,
        price: Optional[float] = None,
        timeframe: Optional[str] = None,
        strategy_name: Optional[str] = None,
        source: str | SignalSource = SignalSource.STRATEGY,
        conditions_met: Optional[List[str]] = None,
        conditions_failed: Optional[List[str]] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Signal:
        """
        Generate a new signal.

        Args:
            symbol: Trading symbol
            signal_type: Signal type (LONG/SHORT/EXIT_LONG/EXIT_SHORT)
            score: Signal confidence (0.0 - 1.0)
            price: Current price
            timeframe: Timeframe
            strategy_name: Strategy name
            source: Signal source
            conditions_met: Met conditions
            conditions_failed: Failed conditions
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size
            metadata: Additional metadata

        Returns:
            Signal: The generated signal.
        """
        # Type conversions
        if isinstance(signal_type, str):
            signal_type = SignalType(signal_type.upper())
        if isinstance(source, str):
            source = SignalSource(source.upper())

        # Generate signal ID
        signal_id = f"{symbol}_{signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Create signal
        signal = Signal(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            timestamp=datetime.now(),
            score=score,
            source=source,
            price=price,
            timeframe=timeframe,
            strategy_name=strategy_name,
            conditions_met=conditions_met or [],
            conditions_failed=conditions_failed or [],
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            metadata=metadata or {}
        )

        # Update statistics
        self.total_signals += 1
        self.signals_by_type[signal_type.value] += 1
        self.signals_by_source[source.value] += 1

        # Store in active signals
        if symbol not in self.active_signals:
            self.active_signals[symbol] = []
        self.active_signals[symbol].append(signal)

        # Add to history
        self.signal_history.append(signal)

        self.logger.info(
            f"📊 Signal generated: {symbol} {signal_type.value} "
            f"(score: {score:.2%}, source: {source.value})"
        )

        return signal

    # ========================================================================
    # SIGNAL PERSISTENCE
    # ========================================================================

    def save_signal(
        self,
        signal: Signal,
        to_file: bool = True
    ) -> None:
        """
        Signal'i kaydet (JSON file)

        Args:
            signal: Signal object
            to_file: File'a kaydet mi?
        """
        if not to_file:
            return

        try:
            # Daily signal file
            date_str = signal.timestamp.strftime('%Y%m%d')
            signal_file = self.data_dir / f"signals_{date_str}.json"

            # Load existing signals
            if signal_file.exists():
                with open(signal_file, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
            else:
                signals = []

            # Append new signal
            signals.append(signal.to_dict())

            # Save
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signals, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"💾 Signal kaydedildi: {signal_file}")

        except Exception as e:
            self.logger.error(f"❌ Error saving signal: {e}")

    def save_all_signals(self) -> None:
        """Save all active signals"""
        for symbol, signals in self.active_signals.items():
            for signal in signals:
                if signal.is_active:
                    self.save_signal(signal)

    # ========================================================================
    # SIGNAL HISTORY & FILTERING
    # ========================================================================

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str | SignalType] = None,
        source: Optional[str | SignalSource] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Signal]:
        """
        Signal history'yi al (filtered)

        Args:
            symbol: Symbol filter (None = all)
            signal_type: Signal type filter
            source: Source filter
            since: Start date filter
            limit: Max signal count

        Returns:
            List[Signal]: Filtered signals
        """
        # Type conversions
        if isinstance(signal_type, str):
            signal_type = SignalType(signal_type.upper())
        if isinstance(source, str):
            source = SignalSource(source.upper())

        # Filter
        filtered = self.signal_history.copy()

        if symbol:
            filtered = [s for s in filtered if s.symbol == symbol]

        if signal_type:
            filtered = [s for s in filtered if s.signal_type == signal_type]

        if source:
            filtered = [s for s in filtered if s.source == source]

        if since:
            filtered = [s for s in filtered if s.timestamp >= since]

        # Sort by timestamp (newest first)
        filtered.sort(key=lambda s: s.timestamp, reverse=True)

        # Limit
        if limit:
            filtered = filtered[:limit]

        return filtered

    def get_active_signals(
        self,
        symbol: Optional[str] = None
    ) -> List[Signal]:
        """
        Receive active signals.

        Args:
            symbol: Symbol filter (None = all)

        Returns:
            List[Signal]: Active signals
        """
        if symbol:
            return [s for s in self.active_signals.get(symbol, []) if s.is_active]
        else:
            active = []
            for signals in self.active_signals.values():
                active.extend([s for s in signals if s.is_active])
            return active

    def mark_signal_executed(
        self,
        signal_id: str,
        execution_price: Optional[float] = None
    ) -> bool:
        """
        Mark the signal as executed.

        Args:
            signal_id: Signal ID
            execution_price: Execution price

        Returns:
            bool: Success
        """
        for signals in self.active_signals.values():
            for signal in signals:
                if signal.signal_id == signal_id:
                    signal.is_executed = True
                    signal.execution_time = datetime.now()
                    signal.execution_price = execution_price

                    self.logger.info(
                        f"✅ Signal executed: {signal.symbol} {signal.signal_type.value} "
                        f"@ ${execution_price:.2f}"
                    )
                    return True

        self.logger.warning(f"⚠️ Signal not found: {signal_id}")
        return False

    def deactivate_signal(self, signal_id: str) -> bool:
        """
        Signal'i deactivate et

        Args:
            signal_id: Signal ID

        Returns:
            bool: Success
        """
        for signals in self.active_signals.values():
            for signal in signals:
                if signal.signal_id == signal_id:
                    signal.is_active = False
                    self.logger.info(f"⏸️  Signal deactivated: {signal_id}")
                    return True

        return False

    # ========================================================================
    # SIGNAL STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Signal istatistiklerini al

        Returns:
            dict: Statistics
        """
        # Active signals count
        active_count = sum(
            len([s for s in signals if s.is_active])
            for signals in self.active_signals.values()
        )

        # Executed signals count
        executed_count = len([s for s in self.signal_history if s.is_executed])

        # Recent signals (last 24h)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_signals = [s for s in self.signal_history if s.timestamp >= recent_cutoff]

        return {
            'total_signals': self.total_signals,
            'active_signals': active_count,
            'executed_signals': executed_count,
            'execution_rate': (executed_count / self.total_signals * 100) if self.total_signals > 0 else 0,
            'signals_by_type': self.signals_by_type.copy(),
            'signals_by_source': self.signals_by_source.copy(),
            'recent_24h': len(recent_signals),
            'symbols_tracked': len(self.active_signals)
        }

    def print_statistics(self) -> None:
        """Prints signal statistics"""
        stats = self.get_statistics()

        self.logger.info("📊 Signal Statistics:")
        self.logger.info(f"   Total Signals: {stats['total_signals']}")
        self.logger.info(f"   Active: {stats['active_signals']}")
        self.logger.info(f"   Executed: {stats['executed_signals']} ({stats['execution_rate']:.1f}%)")
        self.logger.info(f"   Recent 24h: {stats['recent_24h']}")
        self.logger.info(f"   Symbols Tracked: {stats['symbols_tracked']}")

        self.logger.info(f"   By Type:")
        for signal_type, count in stats['signals_by_type'].items():
            self.logger.info(f"      {signal_type}: {count}")

        self.logger.info(f"   By Source:")
        for source, count in stats['signals_by_source'].items():
            self.logger.info(f"      {source}: {count}")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 SignalManager Test")
    print("=" * 60)

    # Create manager
    manager = SignalManager()

    print("\n1️⃣ Signal generation testi:")
    signal1 = manager.generate_signal(
        symbol="BTCUSDT",
        signal_type="LONG",
        score=0.85,
        price=95000.0,
        timeframe="15m",
        strategy_name="EMA_Cross",
        conditions_met=["EMA21 > EMA55", "RSI > 50"],
        stop_loss=94000.0,
        take_profit=97000.0
    )
    print(f"   ✅ Signal created: {signal1.signal_id}")

    print("\n2️⃣ Signal persistence testi:")
    manager.save_signal(signal1)
    print("   ✅ Signal saved")

    print("\n3️⃣ Multiple signals testi:")
    for i in range(5):
        manager.generate_signal(
            symbol="ETHUSDT" if i % 2 == 0 else "BTCUSDT",
            signal_type="SHORT" if i % 2 == 0 else "LONG",
            score=0.7 + (i * 0.05),
            price=3500.0 if i % 2 == 0 else 95000.0,
            source="AI" if i < 2 else "STRATEGY"
        )
    print(f"   ✅ {manager.total_signals} signals generated")

    print("\n4️⃣ Signal history testi:")
    history = manager.get_signal_history(symbol="BTCUSDT", limit=3)
    print(f"   Found {len(history)} BTCUSDT signals")

    print("\n5️⃣ Signal execution testi:")
    manager.mark_signal_executed(signal1.signal_id, execution_price=95100.0)
    print("   ✅ Signal marked as executed")

    print("\n6️⃣ Statistics testi:")
    manager.print_statistics()

    print("\n✅ All tests completed!")
    print("=" * 60)
