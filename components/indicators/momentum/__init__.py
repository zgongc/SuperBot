"""
indicators/momentum/__init__.py - Momentum Indicators

Version: 2.0.0
Date: 2025-10-14

Description:
    Momentum oscillators and indicators.
    11 indicators in this category.

Available Indicators:
    - RSI: Relative Strength Index
    - Stochastic: Stochastic Oscillator
    - CCI: Commodity Channel Index
    - MFI: Money Flow Index
    - Williams %R: Williams Percent Range
    - ROC: Rate of Change
    - TSI: True Strength Index
    - Awesome Oscillator: Bill Williams' AO
    - Ultimate Oscillator: Larry Williams' UO
    - RSI Divergence: RSI Divergence Detection
    - Stochastic RSI: Stochastic applied to RSI
"""

from .rsi import RSI, calculate_rsi_values
from .stochastic import Stochastic
from .cci import CCI
from .mfi import MFI
from .williams_r import WilliamsR
from .roc import ROC
from .tsi import TSI
from .awesome import AwesomeOscillator
from .ultimate import UltimateOscillator
from .rsidivergence import RSIDivergence
from .stochasticrsi import StochasticRSI

__all__ = [
    'RSI',
    'calculate_rsi_values',
    'Stochastic',
    'CCI',
    'MFI',
    'WilliamsR',
    'ROC',
    'TSI',
    'AwesomeOscillator',
    'UltimateOscillator',
    'RSIDivergence',
    'StochasticRSI',
]

# Category metadata
CATEGORY_INFO = {
    'name': 'momentum',
    'description': 'Momentum oscillators and indicators',
    'indicators': __all__,
    'total': 11,  # Target total
    'implemented': 11  # Currently implemented
}
