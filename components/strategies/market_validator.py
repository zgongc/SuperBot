#!/usr/bin/env python3
"""
components/strategies/market_validator.py
SuperBot - Market Validator

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Market filtreleri:
    - Trading session filters (London, NY, Tokyo, Sydney)
    - Time filters (start_hour, end_hour)
    - Day filters (Monday-Sunday)
    - Market tradeable check

Kullanım:
    from components.strategies.market_validator import MarketValidator

    validator = MarketValidator(strategy)
    is_tradeable = validator.is_market_tradeable(timestamp)
"""

from typing import Optional, List, Any
from datetime import datetime, time
import pandas as pd
import pytz


class MarketValidator:
    """
    Market koşulları validator'ı

    Trading session, time, day filtrelerini yönetir
    """
    
    # Session saatleri (UTC)
    SESSIONS = {
        'sydney': (time(22, 0), time(7, 0)),    # Sydney: 22:00-07:00 UTC
        'tokyo': (time(0, 0), time(9, 0)),      # Tokyo: 00:00-09:00 UTC
        'london': (time(7, 0), time(16, 0)),    # London: 07:00-16:00 UTC
        'new_york': (time(12, 0), time(21, 0)), # NY: 12:00-21:00 UTC
    }
    
    # Day mapping
    DAY_NAMES = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6,
    }
    
    def __init__(
        self,
        strategy: Any,
        logger: Any = None
    ):
        """
        Initialize MarketValidator

        Args:
            strategy: BaseStrategy instance (with custom_parameters)
            logger: Logger instance (optional)
        """
        self.strategy = strategy
        self.logger = logger
        
        # Extract custom_parameters
        self.custom_params = getattr(strategy, 'custom_parameters', {})
        
        # Parse filters from strategy.custom_parameters
        self.session_filter_enabled = self._is_session_filter_enabled()
        self.enabled_sessions = self._parse_sessions()
        
        self.time_filter_enabled = self._is_time_filter_enabled()
        self.start_hour = self._parse_start_hour()
        self.end_hour = self._parse_end_hour()
        self.exclude_hours = self._parse_exclude_hours()
        
        self.day_filter_enabled = self._is_day_filter_enabled()
        self.enabled_days = self._parse_days()
    
    # ========================================================================
    # MARKET TRADEABLE CHECK
    # ========================================================================
    
    def is_market_tradeable(
        self,
        timestamp: Optional[datetime] = None,
        timezone: str = 'UTC'
    ) -> bool:
        """
        Market trade edilebilir mi?
        
        Kontroller (sırasıyla):
        1. Session Filter (eğer enabled)
        2. Time Filter (eğer enabled)
        3. Day Filter (eğer enabled)
        
        Args:
            timestamp: Timestamp (None = now)
            timezone: Timezone string
        
        Returns:
            bool: True if tradeable
        """
        # Default: now
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        # Ensure timezone aware
        if timestamp.tzinfo is None:
            timestamp = pytz.timezone(timezone).localize(timestamp)
        
        # Convert to UTC for session checks
        timestamp_utc = timestamp.astimezone(pytz.UTC)
        
        # Check session filter (if enabled)
        if self.session_filter_enabled and self.enabled_sessions:
            if not self._is_session_active(timestamp_utc):
                if self.logger:
                    self.logger.debug(f"Session filter failed at {timestamp_utc}")
                return False
        
        # Check time filter (if enabled)
        if self.time_filter_enabled:
            if not self._is_time_allowed(timestamp_utc):
                if self.logger:
                    self.logger.debug(f"Time filter failed at {timestamp_utc}")
                return False
        
        # Check day filter (if enabled)
        if self.day_filter_enabled and self.enabled_days:
            if not self._is_day_allowed(timestamp_utc):
                if self.logger:
                    self.logger.debug(f"Day filter failed at {timestamp_utc}")
                return False
        
        return True
    
    # ========================================================================
    # SESSION FILTERS
    # ========================================================================
    
    def _is_session_active(self, timestamp_utc: datetime) -> bool:
        """
        Trading session aktif mi?
        
        Args:
            timestamp_utc: UTC timestamp
        
        Returns:
            True if any enabled session is active
        """
        current_time = timestamp_utc.time()
        
        for session_name in self.enabled_sessions:
            session_start, session_end = self.SESSIONS.get(session_name, (None, None))
            
            if session_start is None:
                continue
            
            # Handle session crossing midnight
            if session_start <= session_end:
                # Normal session (e.g., London 07:00-16:00)
                if session_start <= current_time <= session_end:
                    return True
            else:
                # Midnight crossing (e.g., Sydney 22:00-07:00)
                if current_time >= session_start or current_time <= session_end:
                    return True
        
        return False
    
    def _is_session_filter_enabled(self) -> bool:
        """Check if session filter is enabled"""
        session_filter = self.custom_params.get('session_filter', {})
        return session_filter.get('enabled', False)
    
    def _parse_sessions(self) -> List[str]:
        """
        Parse enabled sessions from strategy.custom_parameters
        
        Format:
            custom_parameters = {
                "session_filter": {
                    "enabled": True,
                    "sydney": False,
                    "tokyo": False,
                    "london": True,
                    "new_york": True,
                    "london_ny_overlap": False,  # TODO: Implement overlap
                }
            }
        
        Returns:
            List of enabled session names: ['london', 'new_york']
        """
        session_filter = self.custom_params.get('session_filter', {})
        
        if not session_filter:
            return []
        
        enabled = []
        
        # Check each session
        for session_name in ['sydney', 'tokyo', 'london', 'new_york']:
            if session_filter.get(session_name, False):
                enabled.append(session_name)
        
        return enabled
    
    # ========================================================================
    # TIME FILTERS
    # ========================================================================
    
    def _is_time_filter_enabled(self) -> bool:
        """Check if time filter is enabled"""
        time_filter = self.custom_params.get('time_filter', {})
        return time_filter.get('enabled', False)
    
    def _is_time_allowed(self, timestamp_utc: datetime) -> bool:
        """
        Time range içinde mi?
        
        Args:
            timestamp_utc: UTC timestamp
        
        Returns:
            True if within allowed time range
        """
        current_hour = timestamp_utc.hour
        
        # Check exclude_hours first
        if self.exclude_hours and current_hour in self.exclude_hours:
            return False
        
        # No start/end restrictions
        if self.start_hour is None and self.end_hour is None:
            return True
        
        # Only start_hour restriction
        if self.start_hour is not None and self.end_hour is None:
            return current_hour >= self.start_hour
        
        # Only end_hour restriction
        if self.start_hour is None and self.end_hour is not None:
            return current_hour <= self.end_hour
        
        # Both restrictions
        if self.start_hour <= self.end_hour:
            # Normal range (e.g., 9-17)
            return self.start_hour <= current_hour <= self.end_hour
        else:
            # Midnight crossing (e.g., 22-6)
            return current_hour >= self.start_hour or current_hour <= self.end_hour
    
    def _parse_start_hour(self) -> Optional[int]:
        """
        Parse start hour from strategy.custom_parameters
        
        Format:
            custom_parameters = {
                "time_filter": {
                    "enabled": True,
                    "start_hour": 8,
                    "end_hour": 21,
                }
            }
        """
        time_filter = self.custom_params.get('time_filter', {})
        start_hour = time_filter.get('start_hour')
        
        if start_hour is not None:
            return int(start_hour)
        return None
    
    def _parse_end_hour(self) -> Optional[int]:
        """Parse end hour from strategy.custom_parameters"""
        time_filter = self.custom_params.get('time_filter', {})
        end_hour = time_filter.get('end_hour')
        
        if end_hour is not None:
            return int(end_hour)
        return None
    
    def _parse_exclude_hours(self) -> List[int]:
        """Parse exclude_hours from strategy.custom_parameters"""
        time_filter = self.custom_params.get('time_filter', {})
        exclude_hours = time_filter.get('exclude_hours', [])
        
        if not exclude_hours:
            return []
        
        # Convert to int list
        return [int(h) for h in exclude_hours if isinstance(h, (int, str))]
    
    # ========================================================================
    # DAY FILTERS
    # ========================================================================
    
    def _is_day_filter_enabled(self) -> bool:
        """Check if day filter is enabled"""
        day_filter = self.custom_params.get('day_filter', {})
        return day_filter.get('enabled', False)
    
    def _is_day_allowed(self, timestamp_utc: datetime) -> bool:
        """
        Gün trade edilebilir mi?
        
        Args:
            timestamp_utc: UTC timestamp
        
        Returns:
            True if day is allowed
        """
        weekday = timestamp_utc.weekday()  # 0=Monday, 6=Sunday
        
        return weekday in self.enabled_days
    
    def _parse_days(self) -> List[int]:
        """
        Parse enabled days from strategy.custom_parameters
        
        Format:
            custom_parameters = {
                "day_filter": {
                    "enabled": True,
                    "monday": True,
                    "tuesday": True,
                    "wednesday": True,
                    "thursday": True,
                    "friday": True,
                    "saturday": False,
                    "sunday": False,
                }
            }
        
        Returns:
            List of enabled weekday numbers: [0, 1, 2, 3, 4] (Mon-Fri)
        """
        day_filter = self.custom_params.get('day_filter', {})
        
        if not day_filter:
            return []
        
        enabled = []
        
        # Check each day
        for day_name, weekday_num in self.DAY_NAMES.items():
            if day_filter.get(day_name, False):
                enabled.append(weekday_num)
        
        return enabled
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_next_tradeable_time(
        self,
        timestamp: Optional[datetime] = None,
        timezone: str = 'UTC'
    ) -> Optional[datetime]:
        """
        Sonraki trade edilebilir zaman
        
        Args:
            timestamp: Starting timestamp (None = now)
            timezone: Timezone string
        
        Returns:
            Next tradeable datetime or None
        
        Note:
            Bu basitleştirilmiş bir implementasyon.
            Production'da daha sophisticated olmalı.
        """
        # Default: now
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        # Ensure timezone aware
        if timestamp.tzinfo is None:
            timestamp = pytz.timezone(timezone).localize(timestamp)
        
        # Max 7 gün ileriye bak
        for days_ahead in range(7):
            check_time = timestamp + pd.Timedelta(days=days_ahead, hours=1)
            
            if self.is_market_tradeable(check_time, timezone):
                return check_time
        
        return None
    
    def has_filters(self) -> bool:
        """
        Market filtresi var mı?
        
        Returns:
            True if any filters are enabled
        """
        return bool(
            (self.session_filter_enabled and self.enabled_sessions) or
            (self.time_filter_enabled and (self.start_hour is not None or self.end_hour is not None or self.exclude_hours)) or
            (self.day_filter_enabled and self.enabled_days)
        )
    
    def __repr__(self) -> str:
        return (
            f"<MarketValidator "
            f"sessions={len(self.enabled_sessions)} "
            f"days={len(self.enabled_days)} "
            f"hours={self.start_hour}-{self.end_hour}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MarketValidator',
]

