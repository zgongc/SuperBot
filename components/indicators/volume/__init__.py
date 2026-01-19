"""
indicators/volume/__init__.py - Volume Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Volume category indicators.
    Total of 9 indicators: OBV, VWAP, VWAP Bands, A/D, CMF, EOM,
    Force Index, Volume Profile, Volume Oscillator

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from indicators.volume.obv import OBV
from indicators.volume.vwap import VWAP
from indicators.volume.vwap_bands import VWAPBands
from indicators.volume.ad import AD
from indicators.volume.cmf import CMF
from indicators.volume.eom import EOM
from indicators.volume.force_index import ForceIndex
from indicators.volume.volume_profile import VolumeProfile
from indicators.volume.volume_oscillator import VolumeOscillator


__all__ = [
    # Volume Indicators (9 total)
    'OBV',              # On Balance Volume
    'VWAP',             # Volume Weighted Average Price
    'VWAPBands',        # VWAP Bands
    'AD',               # Accumulation/Distribution
    'CMF',              # Chaikin Money Flow
    'EOM',              # Ease of Movement
    'ForceIndex',       # Force Index
    'VolumeProfile',    # Volume Profile (POC, VAH, VAL)
    'VolumeOscillator', # Volume Oscillator
]


# Kategori bilgisi
CATEGORY = 'volume'
INDICATOR_COUNT = 9

# The required volume status for each indicator.
REQUIRES_VOLUME = {
    'obv': True,
    'vwap': True,
    'vwap_bands': True,
    'ad': True,
    'cmf': True,
    'eom': True,
    'force_index': True,
    'volume_profile': True,
    'volume_oscillator': True,
}


def get_all_indicators():
    """
    Returns all volume indicators.

    Returns:
        dict: A dictionary of indicators in the format {name: class}
    """
    return {
        'obv': OBV,
        'vwap': VWAP,
        'vwap_bands': VWAPBands,
        'ad': AD,
        'cmf': CMF,
        'eom': EOM,
        'force_index': ForceIndex,
        'volume_profile': VolumeProfile,
        'volume_oscillator': VolumeOscillator,
    }


def get_indicator_info():
    """
    Returns indicator information.

    Returns:
        dict: Indicator information
    """
    return {
        'category': CATEGORY,
        'count': INDICATOR_COUNT,
        'indicators': list(get_all_indicators().keys()),
        'requires_volume': REQUIRES_VOLUME
    }
