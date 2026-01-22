"""
Alert Evaluator - Check if analysis results trigger any alerts
"""

import json
from core.logger_engine import LoggerEngine
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class AlertEvaluator:
    """Evaluate alerts against analysis results"""

    def __init__(self, data_manager):
        """
        Initialize evaluator

        Args:
            data_manager: DataManager instance for database access
        """
        self.data_manager = data_manager
        # LoggerEngine setup
        logger_engine = LoggerEngine()
        self.logger = logger_engine.get_logger(__name__)

    async def evaluate_alerts(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all active alerts for this symbol/category

        Args:
            analysis_result: Analysis result object with all data

        Returns:
            List of triggered alert objects
        """
        symbol_id = analysis_result.get('symbol_id')
        category_id = await self.get_category_id(symbol_id)

        # Get relevant alerts
        alerts = await self.get_relevant_alerts(symbol_id, category_id)

        if not alerts:
            return []

        triggered_alerts = []

        for alert in alerts:
            # Skip inactive
            if not alert['is_active']:
                continue

            # Check cooldown
            if self.is_in_cooldown(alert):
                self.logger.debug(f"Alert {alert['id']} is in cooldown, skipping")
                continue

            # Evaluate based on type
            if await self.should_trigger(alert, analysis_result):
                self.logger.info(f"Alert {alert['id']} ({alert['name']}) triggered!")
                triggered_alerts.append(alert)

        return triggered_alerts

    async def get_relevant_alerts(self, symbol_id: Optional[int], category_id: Optional[int]) -> List[Dict[str, Any]]:
        """Get all alerts that apply to this symbol or category"""

        query = """
            SELECT *
            FROM analysis_alerts
            WHERE is_active = 1
              AND (
                  (scope_type = 'symbol' AND symbol_id = ?)
                  OR (scope_type = 'category' AND category_id = ?)
              )
            ORDER BY created_at
        """

        alerts = await self.data_manager.fetch_all(query, (symbol_id, category_id))
        return alerts or []

    async def get_category_id(self, symbol_id: int) -> Optional[int]:
        """Get category ID for a symbol (if assigned)"""
        # For now, return None - this would need category_symbols table
        # TODO: Implement when category-symbol relationship is established
        return None

    async def should_trigger(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Check if alert conditions are met

        Args:
            alert: Alert configuration
            result: Analysis result

        Returns:
            True if alert should trigger
        """
        alert_type = alert['alert_type']

        if alert_type == 'pattern':
            return self.check_pattern_alert(alert, result)
        elif alert_type == 'signal':
            return self.check_signal_alert(alert, result)
        elif alert_type == 'score':
            return self.check_score_alert(alert, result)
        elif alert_type == 'trend_change':
            return await self.check_trend_alert(alert, result)
        elif alert_type == 'category_aggregate':
            return await self.check_category_aggregate_alert(alert, result)

        self.logger.warning(f"Unknown alert type: {alert_type}")
        return False

    def check_pattern_alert(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if pattern conditions are met"""
        try:
            conditions = json.loads(alert['conditions'])

            # Get detected patterns from result
            detected_patterns = self.extract_patterns(result)

            if not detected_patterns:
                return False

            # Filter by confidence
            min_conf = conditions.get('min_confidence', 0.0)
            filtered = [p for p in detected_patterns if p.get('confidence', 0) >= min_conf]

            # Check if any required pattern is present
            pattern_names = conditions.get('pattern_names', [])
            matching = [p for p in filtered if p.get('name') in pattern_names]

            notify_when = conditions.get('notify_when', 'any')
            if notify_when == 'any':
                return len(matching) > 0
            else:  # 'all'
                return len(matching) == len(pattern_names)

        except Exception as e:
            self.logger.error(f"Pattern alert check failed: {e}")
            return False

    def check_signal_alert(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if signal conditions are met"""
        try:
            conditions = json.loads(alert['conditions'])

            # Get signals from result
            signals = self.extract_signals(result)

            if not signals:
                return False

            # Filter by type
            signal_type = conditions.get('signal_type', 'BOTH')
            if signal_type != 'BOTH':
                signals = [s for s in signals if s.get('type') == signal_type]

            # Filter by strategy (if specified)
            strategies = conditions.get('strategies', [])
            if strategies:
                signals = [s for s in signals if s.get('strategy') in strategies]

            # Filter by strength
            min_strength = conditions.get('min_strength', 0.0)
            signals = [s for s in signals if s.get('strength', 0) >= min_strength]

            return len(signals) > 0

        except Exception as e:
            self.logger.error(f"Signal alert check failed: {e}")
            return False

    def check_score_alert(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if score threshold is met"""
        try:
            conditions = json.loads(alert['conditions'])

            score = result.get('overall_score', 0.0)
            threshold = conditions.get('threshold', 0)
            condition = conditions.get('condition', 'above')

            if condition == 'above':
                return score > threshold
            else:  # 'below'
                return score < threshold

        except Exception as e:
            self.logger.error(f"Score alert check failed: {e}")
            return False

    async def check_trend_alert(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if trend changed"""
        try:
            conditions = json.loads(alert['conditions'])

            current_trend = result.get('trend')
            if not current_trend:
                return False

            # Get previous trend (from previous analysis)
            previous_trend = await self.get_previous_trend(result.get('symbol_id'))

            from_trend = conditions.get('from_trend')
            to_trend = conditions.get('to_trend')

            # Check if transition matches
            if from_trend and previous_trend != from_trend:
                return False

            return current_trend == to_trend

        except Exception as e:
            self.logger.error(f"Trend alert check failed: {e}")
            return False

    async def check_category_aggregate_alert(self, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if category aggregate conditions are met"""
        try:
            conditions = json.loads(alert['conditions'])

            # This type of alert checks multiple symbols in a category
            # For now, return False as it requires special handling
            # TODO: Implement category aggregate logic
            self.logger.warning("Category aggregate alerts not yet implemented")
            return False

        except Exception as e:
            self.logger.error(f"Category aggregate alert check failed: {e}")
            return False

    def extract_patterns(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract detected patterns from analysis result"""
        # Parse patterns from result
        patterns_data = result.get('patterns', '[]')

        if isinstance(patterns_data, str):
            try:
                patterns = json.loads(patterns_data)
            except:
                patterns = []
        else:
            patterns = patterns_data

        return patterns if isinstance(patterns, list) else []

    def extract_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract signals from analysis result"""
        # Parse signals from result
        signals_data = result.get('signals', '[]')

        if isinstance(signals_data, str):
            try:
                signals = json.loads(signals_data)
            except:
                signals = []
        else:
            signals = signals_data

        return signals if isinstance(signals, list) else []

    async def get_previous_trend(self, symbol_id: int) -> Optional[str]:
        """Get trend from previous analysis of this symbol"""
        query = """
            SELECT trend
            FROM analysis_results
            WHERE symbol_id = ?
            ORDER BY created_at DESC
            LIMIT 1 OFFSET 1
        """

        result = await self.data_manager.fetch_one(query, (symbol_id,))
        return result['trend'] if result else None

    def is_in_cooldown(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is in cooldown period"""
        last_triggered = alert.get('last_triggered_at')

        if not last_triggered:
            return False

        try:
            if isinstance(last_triggered, str):
                last_triggered_dt = datetime.fromisoformat(last_triggered)
            else:
                last_triggered_dt = last_triggered

            cooldown_minutes = alert.get('cooldown_minutes', 60)
            cooldown = timedelta(minutes=cooldown_minutes)
            elapsed = datetime.now() - last_triggered_dt

            return elapsed < cooldown

        except Exception as e:
            self.logger.error(f"Cooldown check failed: {e}")
            return False
