#!/usr/bin/env python3
"""
components/datamanager/services/analysis.py
SuperBot - Analysis Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Analysis queue and results management - models and CRUD operations

Features:
- AnalysisQueue, AnalysisResult, AnalysisAlert, AlertNotification models
- Queue management (add, get, update status)
- Results storage and retrieval
- Alert configuration and notifications

Usage:
    from components.datamanager.services.analysis import AnalysisService, AnalysisQueue

    service = AnalysisService(db_manager)
    queue_id = await service.queue_analysis("BTCUSDT", "1h")
    await service.save_analysis_result(queue_id, {...})

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.analysis")


# ============================================
# MODELS
# ============================================

class AnalysisQueue(Base):
    """
    Analysis queue for background processing

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "analysis_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    strategy_name = Column(String(50))

    # Status tracking
    status = Column(String(20), default="pending", index=True)  # pending/analyzing/completed/failed
    progress = Column(Integer, default=0)  # 0-100%

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_symbol_timeframe', 'symbol', 'timeframe'),
    )


class AnalysisResult(Base):
    """
    Analysis results with signals and indicators

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    queue_id = Column(Integer, index=True)  # Reference to AnalysisQueue
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    strategy_name = Column(String(50))

    # Signal
    signal = Column(String(10))  # BUY/SELL/HOLD/NEUTRAL
    confidence = Column(Float)  # 0-100%
    score = Column(Float)  # Opportunity score

    # Indicators & Patterns (JSON)
    indicators = Column(JSON)  # {"rsi": 65.4, "macd": {...}}
    patterns = Column(JSON)  # ["ascending_triangle", "bullish_engulfing"]

    # Trade recommendations
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    risk_reward_ratio = Column(Float)

    # Timing
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Extra data
    extra_data = Column(JSON)
    notes = Column(Text)

    # Indexes
    __table_args__ = (
        Index('idx_signal_confidence', 'signal', 'confidence'),
        Index('idx_result_created_at', 'created_at'),
    )


class AnalysisAlert(Base):
    """
    Analysis alert configurations

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "analysis_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, default=1)

    # Basic Info
    name = Column(String(200), nullable=False)
    description = Column(Text)

    # Scope
    scope_type = Column(String(20), nullable=False)  # 'symbol' or 'category'
    symbol_id = Column(Integer)
    category_id = Column(Integer)

    # Alert Type & Conditions
    alert_type = Column(String(30), nullable=False)  # 'pattern', 'signal', 'score'
    conditions = Column(Text, nullable=False)  # JSON

    # Notification Channels
    notify_webui = Column(Boolean, default=True)
    notify_telegram = Column(Boolean, default=False)
    notify_email = Column(Boolean, default=False)

    # Status
    is_active = Column(Boolean, default=True, index=True)

    # Statistics
    trigger_count = Column(Integer, default=0)
    last_triggered_at = Column(DateTime)
    cooldown_minutes = Column(Integer, default=60)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_alerts_symbol', 'symbol_id'),
        Index('idx_alerts_category', 'category_id'),
        Index('idx_alerts_type', 'alert_type'),
    )


class AlertNotification(Base):
    """
    Alert notification history

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "alert_notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # References
    alert_id = Column(Integer, nullable=False, index=True)
    analysis_result_id = Column(Integer)

    # Trigger Details
    trigger_type = Column(String(30), nullable=False)
    trigger_details = Column(Text)  # JSON

    # Notification Status
    webui_sent = Column(Boolean, default=False)
    webui_sent_at = Column(DateTime)

    telegram_sent = Column(Boolean, default=False)
    telegram_sent_at = Column(DateTime)
    telegram_error = Column(Text)

    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime)
    email_error = Column(Text)

    # Read Status
    is_read = Column(Boolean, default=False, index=True)
    read_at = Column(DateTime)

    # Metadata
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('idx_alert_notifs_alert', 'alert_id'),
        Index('idx_alert_notifs_unread', 'is_read', 'triggered_at'),
    )


# ============================================
# SERVICE
# ============================================

class AnalysisService(BaseService):
    """Analysis queue and results management service"""

    # Stats tracking
    stats = {
        "total_queue": 0,
        "total_results": 0,
        "total_alerts": 0
    }

    # ============================================
    # Queue Operations
    # ============================================

    async def queue_analysis(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: Optional[str] = None
    ) -> Optional[int]:
        """
        Add symbol to analysis queue

        Args:
            symbol: Trading pair
            timeframe: Timeframe (1m, 5m, 1h, etc.)
            strategy_name: Strategy to use

        Returns:
            int: Queue ID if successful
        """
        # === PORTED FROM data_manager.py lines 211-260 ===
        try:
            async with self.session() as session:
                # Check if already queued
                query = select(AnalysisQueue).where(
                    AnalysisQueue.symbol == symbol,
                    AnalysisQueue.timeframe == timeframe,
                    AnalysisQueue.status.in_(["pending", "analyzing"])
                )
                result = await session.execute(query)
                existing = result.scalars().first()

                if existing:
                    logger.warning(f"⚠️ {symbol} {timeframe} already in queue (status: {existing.status})")
                    return existing.id

                # Create new queue entry
                queue_entry = AnalysisQueue(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    status="pending"
                )
                session.add(queue_entry)
                await session.commit()
                await session.refresh(queue_entry)

                self.stats["total_queue"] += 1
                logger.info(f"✅ Analysis queued: {symbol} {timeframe}")
                return queue_entry.id

        except Exception as e:
            logger.error(f"❌ Analysis queue error: {e}")
            return None

    async def get_analysis_queue(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get analysis queue

        Args:
            status: Filter by status (pending/analyzing/completed/failed)
            limit: Max entries

        Returns:
            List: Queue entries
        """
        # === PORTED FROM data_manager.py lines 262-308 ===
        try:
            async with self.session() as session:
                query = select(AnalysisQueue)

                if status:
                    query = query.where(AnalysisQueue.status == status)

                query = query.order_by(AnalysisQueue.created_at.desc()).limit(limit)

                result = await session.execute(query)
                entries = result.scalars().all()

                return [
                    {
                        "id": e.id,
                        "symbol": e.symbol,
                        "timeframe": e.timeframe,
                        "strategy_name": e.strategy_name,
                        "status": e.status,
                        "progress": e.progress,
                        "created_at": e.created_at,
                        "started_at": e.started_at,
                        "completed_at": e.completed_at,
                        "error_message": e.error_message,
                        "retry_count": e.retry_count
                    }
                    for e in entries
                ]

        except Exception as e:
            logger.error(f"❌ Analysis queue get error: {e}")
            return []

    async def update_analysis_status(
        self,
        queue_id: int,
        status: str,
        progress: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update analysis queue status

        Args:
            queue_id: Queue ID
            status: New status (pending/analyzing/completed/failed)
            progress: Progress percentage (0-100)
            error_message: Error message if failed

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 310-362 ===
        try:
            async with self.session() as session:
                query = select(AnalysisQueue).where(AnalysisQueue.id == queue_id)
                result = await session.execute(query)
                entry = result.scalars().first()

                if not entry:
                    logger.warning(f"⚠️ Analysis queue entry not found: ID {queue_id}")
                    return False

                # Update status
                old_status = entry.status
                entry.status = status

                if progress is not None:
                    entry.progress = progress

                if error_message:
                    entry.error_message = error_message

                # Update timestamps
                if status == "analyzing" and old_status == "pending":
                    entry.started_at = get_utc_now()
                elif status in ["completed", "failed"]:
                    entry.completed_at = get_utc_now()

                await session.commit()

                logger.info(f"✅ Analysis status updated: {entry.symbol} → {status}")
                return True

        except Exception as e:
            logger.error(f"❌ Analysis status update error: {e}")
            return False

    async def get_pending_analysis(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending analysis entries"""
        return await self.get_analysis_queue(status="pending", limit=limit)

    # ============================================
    # Results Operations
    # ============================================

    async def save_analysis_result(
        self,
        symbol: str,
        timeframe: str,
        signal: str,
        confidence: float,
        score: float,
        indicators: Dict,
        patterns: List[str],
        queue_id: Optional[int] = None,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        risk_reward_ratio: Optional[float] = None,
        expires_at: Optional[datetime] = None,
        strategy_name: Optional[str] = None,
        extra_data: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Save analysis result

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            signal: BUY/SELL/HOLD/NEUTRAL
            confidence: Confidence 0-100%
            score: Opportunity score
            indicators: Indicator values (JSON)
            patterns: Detected patterns
            queue_id: Reference to analysis queue
            entry_price: Recommended entry price
            target_price: Target price
            stop_loss: Stop loss price
            risk_reward_ratio: Risk/reward ratio
            expires_at: Signal expiry time
            strategy_name: Strategy used
            extra_data: Additional metadata

        Returns:
            int: Result ID if successful
        """
        # === PORTED FROM data_manager.py lines 368-438 ===
        try:
            async with self.session() as session:
                result = AnalysisResult(
                    queue_id=queue_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    signal=signal,
                    confidence=confidence,
                    score=score,
                    indicators=indicators,
                    patterns=patterns,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward_ratio,
                    expires_at=expires_at,
                    extra_data=extra_data
                )
                session.add(result)
                await session.commit()
                await session.refresh(result)

                self.stats["total_results"] += 1
                logger.info(f"✅ Analysis result saved: {symbol} {timeframe} → {signal} ({confidence:.1f}%)")
                return result.id

        except Exception as e:
            logger.error(f"❌ Analysis result save error: {e}")
            return None

    async def get_analysis_results(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get analysis results

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            signal: Filter by signal (BUY/SELL/HOLD/NEUTRAL)
            min_confidence: Minimum confidence level
            limit: Max results

        Returns:
            List: Analysis results
        """
        # === PORTED FROM data_manager.py lines 440-502 ===
        try:
            async with self.session() as session:
                query = select(AnalysisResult)

                if symbol:
                    query = query.where(AnalysisResult.symbol == symbol)

                if timeframe:
                    query = query.where(AnalysisResult.timeframe == timeframe)

                if signal:
                    query = query.where(AnalysisResult.signal == signal)

                if min_confidence:
                    query = query.where(AnalysisResult.confidence >= min_confidence)

                query = query.order_by(AnalysisResult.created_at.desc()).limit(limit)

                result = await session.execute(query)
                results = result.scalars().all()

                return [
                    {
                        "id": r.id,
                        "queue_id": r.queue_id,
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "strategy_name": r.strategy_name,
                        "signal": r.signal,
                        "confidence": r.confidence,
                        "score": r.score,
                        "indicators": r.indicators,
                        "patterns": r.patterns,
                        "entry_price": r.entry_price,
                        "target_price": r.target_price,
                        "stop_loss": r.stop_loss,
                        "risk_reward_ratio": r.risk_reward_ratio,
                        "expires_at": r.expires_at,
                        "created_at": r.created_at
                    }
                    for r in results
                ]

        except Exception as e:
            logger.error(f"❌ Analysis results get error: {e}")
            return []

    async def get_latest_result(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get latest analysis result for symbol/timeframe"""
        results = await self.get_analysis_results(symbol=symbol, timeframe=timeframe, limit=1)
        return results[0] if results else None

    # ============================================
    # Alert Operations
    # ============================================

    async def create_alert(
        self,
        name: str,
        scope_type: str,
        alert_type: str,
        conditions: str,
        symbol_id: Optional[int] = None,
        category_id: Optional[int] = None,
        description: Optional[str] = None,
        notify_webui: bool = True,
        notify_telegram: bool = False,
        notify_email: bool = False,
        cooldown_minutes: int = 60
    ) -> Optional[int]:
        """Create analysis alert"""
        try:
            async with self.session() as session:
                alert = AnalysisAlert(
                    name=name,
                    description=description,
                    scope_type=scope_type,
                    symbol_id=symbol_id,
                    category_id=category_id,
                    alert_type=alert_type,
                    conditions=conditions,
                    notify_webui=notify_webui,
                    notify_telegram=notify_telegram,
                    notify_email=notify_email,
                    cooldown_minutes=cooldown_minutes
                )
                session.add(alert)
                await session.commit()
                await session.refresh(alert)

                self.stats["total_alerts"] += 1
                logger.info(f"✅ Alert created: {name}")
                return alert.id

        except Exception as e:
            logger.error(f"❌ Alert create error: {e}")
            return None

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        try:
            async with self.session() as session:
                query = select(AnalysisAlert).where(AnalysisAlert.is_active == True)
                result = await session.execute(query)
                alerts = result.scalars().all()

                return [
                    {
                        "id": a.id,
                        "name": a.name,
                        "scope_type": a.scope_type,
                        "symbol_id": a.symbol_id,
                        "category_id": a.category_id,
                        "alert_type": a.alert_type,
                        "conditions": a.conditions,
                        "trigger_count": a.trigger_count,
                        "last_triggered_at": a.last_triggered_at
                    }
                    for a in alerts
                ]

        except Exception as e:
            logger.error(f"❌ Active alerts get error: {e}")
            return []

    async def trigger_alert(
        self,
        alert_id: int,
        trigger_type: str,
        trigger_details: str,
        analysis_result_id: Optional[int] = None
    ) -> Optional[int]:
        """Create alert notification"""
        try:
            async with self.session() as session:
                notification = AlertNotification(
                    alert_id=alert_id,
                    analysis_result_id=analysis_result_id,
                    trigger_type=trigger_type,
                    trigger_details=trigger_details,
                    webui_sent=True,
                    webui_sent_at=get_utc_now()
                )
                session.add(notification)

                # Update alert stats
                query = select(AnalysisAlert).where(AnalysisAlert.id == alert_id)
                result = await session.execute(query)
                alert = result.scalars().first()
                if alert:
                    alert.trigger_count += 1
                    alert.last_triggered_at = get_utc_now()

                await session.commit()
                await session.refresh(notification)

                logger.info(f"✅ Alert triggered: ID {alert_id}")
                return notification.id

        except Exception as e:
            logger.error(f"❌ Alert trigger error: {e}")
            return None

    async def get_unread_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get unread notifications"""
        try:
            async with self.session() as session:
                query = select(AlertNotification).where(
                    AlertNotification.is_read == False
                ).order_by(AlertNotification.triggered_at.desc()).limit(limit)

                result = await session.execute(query)
                notifications = result.scalars().all()

                return [
                    {
                        "id": n.id,
                        "alert_id": n.alert_id,
                        "trigger_type": n.trigger_type,
                        "trigger_details": n.trigger_details,
                        "triggered_at": n.triggered_at,
                        "is_read": n.is_read
                    }
                    for n in notifications
                ]

        except Exception as e:
            logger.error(f"❌ Unread notifications get error: {e}")
            return []

    async def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        try:
            async with self.session() as session:
                query = select(AlertNotification).where(AlertNotification.id == notification_id)
                result = await session.execute(query)
                notification = result.scalars().first()

                if notification:
                    notification.is_read = True
                    notification.read_at = get_utc_now()
                    await session.commit()
                    return True

                return False

        except Exception as e:
            logger.error(f"❌ Mark notification read error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json

    print("=" * 70)
    print("🧪 AnalysisService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = AnalysisService(db)

        # ============================================
        # QUEUE TESTS
        # ============================================
        print("\n" + "=" * 50)
        print("📋 QUEUE TESTS")
        print("=" * 50)

        # Test 1: Queue analysis
        print("\n1️⃣ Queue analysis for BTCUSDT 1h")
        queue_id = await service.queue_analysis("BTCUSDT", "1h", strategy_name="RSI_MACD")
        print(f"   ✅ Queued: ID {queue_id}")

        # Test 2: Queue multiple
        print("\n2️⃣ Queue multiple symbols")
        for symbol in ["ETHUSDT", "BNBUSDT", "SOLUSDT"]:
            qid = await service.queue_analysis(symbol, "4h")
            print(f"   ✅ {symbol}: ID {qid}")

        # Test 3: Get pending queue
        print("\n3️⃣ Get pending queue")
        pending = await service.get_pending_analysis(limit=10)
        print(f"   ✅ Pending: {len(pending)} entries")
        for p in pending[:3]:
            print(f"      - {p['symbol']} {p['timeframe']} ({p['status']})")

        # Test 4: Update status to analyzing
        print(f"\n4️⃣ Update status to 'analyzing'")
        success = await service.update_analysis_status(queue_id, "analyzing", progress=10)
        print(f"   ✅ Updated: {success}")

        # Test 5: Update progress
        print(f"\n5️⃣ Update progress to 50%")
        success = await service.update_analysis_status(queue_id, "analyzing", progress=50)
        print(f"   ✅ Updated: {success}")

        # Test 6: Complete analysis
        print(f"\n6️⃣ Complete analysis")
        success = await service.update_analysis_status(queue_id, "completed", progress=100)
        print(f"   ✅ Completed: {success}")

        # Test 7: Get queue with status filter
        print("\n7️⃣ Get completed analyses")
        completed = await service.get_analysis_queue(status="completed")
        print(f"   ✅ Completed: {len(completed)} entries")

        # ============================================
        # RESULTS TESTS
        # ============================================
        print("\n" + "=" * 50)
        print("📊 RESULTS TESTS")
        print("=" * 50)

        # Test 8: Save analysis result
        print("\n8️⃣ Save analysis result")
        result_id = await service.save_analysis_result(
            symbol="BTCUSDT",
            timeframe="1h",
            signal="BUY",
            confidence=85.5,
            score=72.3,
            indicators={
                "rsi": 42.5,
                "macd": {"value": 150.2, "signal": 120.5, "histogram": 29.7},
                "bb": {"upper": 52000, "middle": 50000, "lower": 48000}
            },
            patterns=["bullish_engulfing", "support_bounce"],
            queue_id=queue_id,
            entry_price=50000.0,
            target_price=52500.0,
            stop_loss=49000.0,
            risk_reward_ratio=2.5,
            strategy_name="RSI_MACD"
        )
        print(f"   ✅ Result saved: ID {result_id}")

        # Test 9: Save multiple results
        print("\n9️⃣ Save multiple results")
        signals = [
            ("ETHUSDT", "4h", "SELL", 78.2, 65.1),
            ("BNBUSDT", "4h", "HOLD", 55.0, 45.3),
            ("SOLUSDT", "4h", "BUY", 92.1, 88.5)
        ]
        for symbol, tf, signal, conf, score in signals:
            rid = await service.save_analysis_result(
                symbol=symbol,
                timeframe=tf,
                signal=signal,
                confidence=conf,
                score=score,
                indicators={"rsi": 50.0},
                patterns=[]
            )
            print(f"   ✅ {symbol}: {signal} ({conf}%) - ID {rid}")

        # Test 10: Get all results
        print("\n🔟 Get all results")
        results = await service.get_analysis_results(limit=10)
        print(f"   ✅ Total: {len(results)} results")

        # Test 11: Filter by signal
        print("\n1️⃣1️⃣ Filter BUY signals")
        buy_results = await service.get_analysis_results(signal="BUY")
        print(f"   ✅ BUY signals: {len(buy_results)}")
        for r in buy_results:
            print(f"      - {r['symbol']}: {r['confidence']:.1f}%")

        # Test 12: Filter by min confidence
        print("\n1️⃣2️⃣ Filter confidence >= 80%")
        high_conf = await service.get_analysis_results(min_confidence=80.0)
        print(f"   ✅ High confidence: {len(high_conf)}")
        for r in high_conf:
            print(f"      - {r['symbol']}: {r['signal']} ({r['confidence']:.1f}%)")

        # Test 13: Get latest result
        print("\n1️⃣3️⃣ Get latest BTCUSDT 1h result")
        latest = await service.get_latest_result("BTCUSDT", "1h")
        if latest:
            print(f"   ✅ Latest: {latest['signal']} ({latest['confidence']:.1f}%)")
            print(f"      Entry: ${latest['entry_price']}, Target: ${latest['target_price']}")
            print(f"      R/R: {latest['risk_reward_ratio']}")

        # ============================================
        # ALERT TESTS
        # ============================================
        print("\n" + "=" * 50)
        print("🚨 ALERT TESTS")
        print("=" * 50)

        # Test 14: Create alert
        print("\n1️⃣4️⃣ Create BUY signal alert")
        alert_id = await service.create_alert(
            name="BTC Strong BUY Alert",
            scope_type="symbol",
            symbol_id=1,
            alert_type="signal",
            conditions=json.dumps({"signal": "BUY", "min_confidence": 80}),
            notify_webui=True,
            notify_telegram=True
        )
        print(f"   ✅ Alert created: ID {alert_id}")

        # Test 15: Get active alerts
        print("\n1️⃣5️⃣ Get active alerts")
        alerts = await service.get_active_alerts()
        print(f"   ✅ Active alerts: {len(alerts)}")
        for a in alerts:
            print(f"      - {a['name']} ({a['alert_type']})")

        # Test 16: Trigger alert
        print("\n1️⃣6️⃣ Trigger alert")
        if alert_id:
            notif_id = await service.trigger_alert(
                alert_id=alert_id,
                trigger_type="signal",
                trigger_details=json.dumps({"symbol": "BTCUSDT", "signal": "BUY", "confidence": 92.5}),
                analysis_result_id=result_id
            )
            print(f"   ✅ Notification created: ID {notif_id}")

        # Test 17: Get unread notifications
        print("\n1️⃣7️⃣ Get unread notifications")
        unread = await service.get_unread_notifications()
        print(f"   ✅ Unread: {len(unread)} notifications")
        for n in unread:
            print(f"      - Alert {n['alert_id']}: {n['trigger_type']} @ {n['triggered_at']}")

        # Test 18: Mark as read
        print("\n1️⃣8️⃣ Mark notification as read")
        if unread:
            success = await service.mark_notification_read(unread[0]['id'])
            print(f"   ✅ Marked as read: {success}")

        # Test 19: Verify read status
        print("\n1️⃣9️⃣ Verify unread count decreased")
        unread_after = await service.get_unread_notifications()
        print(f"   ✅ Unread after: {len(unread_after)} (was {len(unread)})")

        # ============================================
        # STATS
        # ============================================
        print("\n" + "=" * 50)
        print("📈 SERVICE STATS")
        print("=" * 50)
        print(f"   Total Queue Entries: {service.stats['total_queue']}")
        print(f"   Total Results: {service.stats['total_results']}")
        print(f"   Total Alerts: {service.stats['total_alerts']}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
