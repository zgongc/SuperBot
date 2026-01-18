#!/usr/bin/env python3

"""
core/timezone_utils.py

SuperBot - Timezone Utilities
Author: SuperBot Team
Date: 2025-11-12
Version: 1.0.0

Helper functions for managing timezone and datetime conversions.

Features:
- UTC and local time conversions.
- Auto timezone selection based on configuration.
- Timestamp formatting helpers.

Usage:
    from core.timezone_utils import TimezoneUtils
    now = TimezoneUtils.now()

Dependencies:
    - python>=3.12
    - pytz>=2023.3
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import pytz

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config_engine import get_config


def get_utc_now() -> datetime:
    """Return UTC timezone-aware current time."""
    return datetime.now(timezone.utc)


def get_utc_now_with_offset(hours: int = 0, minutes: int = 0) -> datetime:
    """Return current time with UTC offset."""
    offset = timedelta(hours=hours, minutes=minutes)
    return datetime.now(timezone(offset))


def timestamp_to_utc(timestamp: int, unit: str = "ms") -> datetime:
    """Convert Unix timestamp to UTC datetime."""
    if unit == "ms":
        timestamp = timestamp / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def utc_to_timestamp(dt: datetime, unit: str = "ms") -> int:
    """Convert UTC datetime to Unix timestamp."""
    ts = dt.astimezone(timezone.utc).timestamp()
    if unit == "ms":
        return int(ts * 1000)
    return int(ts)


def get_config_timezone_now(config: Optional[dict] = None) -> datetime:
    """
    Return time according to timezone setting in config.

    Args:
        config: Config dict (if None, global config is used)
    """

    config = config or get_config().get("system", {})
    utc_offset = config.get("utc_offset", 0)
    use_local = config.get("use_local_time", False)

    if use_local and config.get("timezone"):
        return datetime.now(pytz.timezone(config["timezone"]))
    return get_utc_now_with_offset(hours=utc_offset)


class TimezoneUtils:
    """
    Timezone helper class.

    Attributes:
        _tz: Active timezone object
    """

    _tz: Optional[pytz.timezone] = None

    @classmethod
    def _get_timezone(cls) -> pytz.timezone:
        if cls._tz is None:
            cfg = get_config()
            system_cfg = cfg.get("system", {})
            tz_name = system_cfg.get("timezone", "UTC")
            utc_offset = system_cfg.get("utc_offset", 0)
            # Create timezone with UTC offset
            if utc_offset != 0:
                cls._tz = pytz.timezone(f'Etc/GMT{-utc_offset:+d}')  # Note: signs are reversed in Etc/GMT
            else:
                cls._tz = pytz.timezone(tz_name)
        return cls._tz

    @classmethod
    def now(cls) -> datetime:
        """Return current time according to active timezone."""
        cfg = get_config()
        system_cfg = cfg.get("system", {})
        use_local = system_cfg.get("use_local_time", False)
        if use_local:
            return datetime.now(cls._get_timezone())
        return get_config_timezone_now(system_cfg)

    @classmethod
    def to_timezone(cls, dt: datetime, tz_name: Optional[str] = None) -> datetime:
        """Convert datetime object to specified timezone."""
        timezone = pytz.timezone(tz_name) if tz_name else cls._get_timezone()
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(timezone)

    @classmethod
    def format(cls, dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Convert datetime object to string according to format."""
        return cls.to_timezone(dt).strftime(fmt)


# ============================================================================
# TEST
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TimezoneUtils Test")
    print("=" * 60)

    now = TimezoneUtils.now()
    print(f"Now: {now}")
    formatted = TimezoneUtils.format(now)
    print(f"Formatted: {formatted}")
    print("   ✅ Test successful")

    print("\n✅ All tests completed!")
    print("=" * 60)


