#!/usr/bin/env python3
"""
components/strategies/base_strategy.py
SuperBot - Base Strategy Class & Config Types

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Base strategy class and all config types.
    Each strategy template inherits from this class.

Usage:
    from components.strategies.base_strategy import BaseStrategy, TradingSide
    
    class MyStrategy(BaseStrategy):
        def __init__(self):
            super().__init__()
            self.strategy_name = "My Strategy"
            self.symbols = [...]
            self.entry_conditions = {...}
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS
# ============================================================================

class TradingSide(str, Enum):
    """Trading direction - The strategy can open positions in which direction"""
    LONG = "LONG"      # Open only LONG position
    SHORT = "SHORT"    # Open only SHORT position
    BOTH = "BOTH"      # Can open both LONG and SHORT
    FLAT = "FLAT"  # No position opening (pause)


class PositionSizeMethod(str, Enum):
    """Position size calculation methods"""
    FIXED_USD = "FIXED_USD"                  # Sabit dolar ($100)
    FIXED_PERCENT = "FIXED_PERCENT"          # Fixed percentage (%5 of capital)
    FIXED_QUANTITY = "FIXED_QUANTITY"        # Fixed quantity (0.01 BTC)
    RISK_BASED = "RISK_BASED"                # Risk-based (based on stop loss)
    KELLY_CRITERION = "KELLY_CRITERION"  # Kelly formula
    VOLATILITY_SCALED = "VOLATILITY_SCALED"  # Based on ATR/volatility
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based dynamic sizing


class ExitMethod(str, Enum):
    """Take profit methods"""
    FIXED_PERCENT = "FIXED_PERCENT"          # Fixed percentage (%2)
    FIXED_PRICE = "FIXED_PRICE"              # Sabit fiyat ($45000)
    RISK_REWARD = "RISK_REWARD"              # Risk/Reward ratio (1:2)
    ATR_BASED = "ATR_BASED"                  # ATR factor (2 x ATR)
    FIBONACCI = "FIBONACCI"                  # Fibonacci seviyeleri
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based dynamic exit


class StopLossMethod(str, Enum):
    """Stop loss methods"""
    FIXED_PERCENT = "FIXED_PERCENT"          # Fixed percentage (%1)
    FIXED_PRICE = "FIXED_PRICE"              # Sabit fiyat ($95000)
    ATR_BASED = "ATR_BASED"                  # ATR factor (1.5 x ATR)
    SWING_POINTS = "SWING_POINTS"            # Swing low/high
    FIBONACCI = "FIBONACCI"                  # Fibonacci retracement
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based adaptive SL


# ============================================================================
# CONFIG DATACLASSES
# ============================================================================

@dataclass
class SymbolConfig:
    """
    Symbol configuration
    
    Attributes:
        symbol: List of base assets (e.g., ['BTC', 'ETH'])
        quote: Quote currency (e.g., 'USDT')
        enabled: Bu semboller trade edilsin mi?
    """
    symbol: List[str]
    quote: str
    enabled: bool = True


@dataclass
class TechnicalParameters:
    """
    Technical indicator parameters
    
    Attributes:
        indicators: Dictionary of indicators.
            Format: {
                "rsi_14": {"period": 14, "overbought": 70, "oversold": 30},
                "ema_50": {"period": 50},
                ...
            }
    """
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RiskManagement:
    """
    Risk management parameters

    POSITION SIZING METHODS (only set the leverage, the rest is automatic!):

    1. FIXED_PERCENT (RECOMMENDED - Simplest):
       - size_value: What percentage (%) of the portfolio (multiplied by leverage)
       - max_risk_per_trade: NOT USED (will be ignored)
       - Example: size_value=10, leverage=5 -> Each trade is %50 of the position (10% x 5 x)

    2. FIXED_USD:
       - size_value: How many $ per trade (leverage NOT APPLIED!)
       - max_risk_per_trade: DO NOT USE
       - Example: size_value=1000 -> Each trade is a $1000 position.

    3. RISK_BASED (COMPLEX - Use with caution!):
       - size_value: DO NOT USE
       - max_risk_per_trade: The percentage you are willing to lose (divided by the stop_loss)
       - Formula: Position = (Portfolio x max_risk_per_trade) / stop_loss_distance
       - Example: max_risk=2%, stop_loss=2%, leverage=1 -> 100% position
       - ⚠️ WARNING: If max_risk/stop_loss > leverage, a LIMIT EXCEEDS!

    Attributes:
        sizing_method: Position sizing calculation method (one of the three above)
        size_value: Position size value (varies depending on the method - see above)
        max_risk_per_trade: Maximum risk per trade (used only with RISK_BASED)
        max_correlation: Maximum correlation limit
        position_correlation_limit: Position correlation limit
        max_drawdown: Maximum drawdown (%)
        max_daily_trades: Maximum number of trades per day
        emergency_stop_enabled: Is the emergency stop enabled?
        ai_risk_enabled: Is AI risk management enabled? (not yet implemented)
        dynamic_position_sizing: Is dynamic position sizing enabled?

    Note:
        max_portfolio_risk is automatically calculated: strategy.leverage x 100
        Example: leverage=5 -> max_portfolio_risk=500% (With a leverage of 5 x, the maximum position is 500%)
    """
    sizing_method: PositionSizeMethod

    # Position sizing parameters (each method uses its own parameters)
    position_percent_size: float = 10.0      # For FIXED_PERCENT: What percentage of the portfolio?
    position_usd_size: float = 1000.0        # For FIXED_USD: How many dollars
    position_quantity_size: float = 0.01     # For FIXED_QUANTITY: How many units (e.g., 0.01 BTC)
    max_risk_per_trade: float = 2.0          # For RISK_BASED: Percentage of capital you are willing to lose.

    # Backward compatibility (deprecated - do not use!)
    size_value: float = 0.0                  # DEPRECATED: Use the new parameters (position_*_size)

    # max_portfolio_risk: REMOVED - Now auto-calculated from strategy.leverage
    max_correlation: float = 0.7
    position_correlation_limit: float = 0.7
    max_drawdown: float = 20.0
    max_daily_trades: int = 100
    emergency_stop_enabled: bool = False
    ai_risk_enabled: bool = False
    dynamic_position_sizing: bool = False


@dataclass
class PositionManagement:
    """
    Position management parameters
    
    Attributes:
        max_positions_per_symbol: Maximum number of positions per symbol
        max_total_positions: Total maximum number of positions
        allow_hedging: Allows hedging.
        position_timeout_enabled: Is the timeout control active?
        position_timeout: Position timeout duration (minutes)
        pyramiding_enabled: Is pyramiding enabled?
        pyramiding_max_entries: Maximum number of pyramiding entries
        pyramiding_scale_factor: Size factor for each entry
    """
    max_positions_per_symbol: int
    max_total_positions: int
    allow_hedging: bool = False
    position_timeout_enabled: bool = False
    position_timeout: int = 1800  # minute (default: 30 hours)
    pyramiding_enabled: bool = False
    pyramiding_max_entries: int = 3
    pyramiding_scale_factor: float = 0.5


@dataclass
class ExitStrategy:
    """
    Exit strategy parameters

    Attributes:
        # Take Profit Methods
        take_profit_method: Method for calculating the take profit.
        take_profit_percent: %TP for FIXED_PERCENT
        take_profit_price: price for FIXED_PRICE
        take_profit_risk_reward_ratio: R/R ratio for RISK_REWARD
        take_profit_atr_multiplier: ATR multiplier for ATR_BASED
        take_profit_fib_level: extension level for FIBONACCI
        take_profit_ai_level: level for DYNAMIC_AI

        # Stop Loss Methods
        stop_loss_method: Method for calculating the stop loss.
        stop_loss_percent: %SL for FIXED_PERCENT
        stop_loss_price: price for FIXED_PRICE
        stop_loss_atr_multiplier: ATR multiplier for ATR_BASED
        stop_loss_swing_lookback: lookback period for SWING_POINTS
        stop_loss_fib_level: retracement level for FIBONACCI
        stop_loss_ai_level: level for DYNAMIC_AI

        # Trailing Stop
        trailing_stop_enabled: Is the trailing stop active?
        trailing_activation_profit_percent: Trailing will activate at this profit percentage.
        trailing_callback_percent: Callback percentage for trailing.
        trailing_take_profit: Start trailing when the take profit is reached.
        trailing_distance: Distance from the take profit.

        # Break Even
        break_even_enabled: Is break-even enabled?
        break_even_trigger_profit_percent: Trigger profit percentage
        break_even_offset: Entry'den offset

        # Partial Exit
        partial_exit_enabled: Is partial exit enabled?
        partial_exit_levels: Snow levels (%)
        partial_exit_sizes: The amount to exit at each level.
    """
    # Take Profit (method is required, all parameters optional)
    take_profit_method: ExitMethod = ExitMethod.FIXED_PERCENT
    take_profit_percent: float = 0.0          # FIXED_PERCENT
    take_profit_price: float = 0.0            # FIXED_PRICE
    take_profit_risk_reward_ratio: float = 0.0  # RISK_REWARD
    take_profit_atr_multiplier: float = 0.0   # ATR_BASED
    take_profit_fib_level: float = 0.0        # FIBONACCI
    take_profit_ai_level: int = 0             # DYNAMIC_AI

    # Stop Loss (method is required, all parameters optional)
    stop_loss_method: StopLossMethod = StopLossMethod.FIXED_PERCENT
    stop_loss_percent: float = 0.0            # FIXED_PERCENT
    stop_loss_price: float = 0.0              # FIXED_PRICE
    stop_loss_atr_multiplier: float = 0.0     # ATR_BASED
    stop_loss_swing_lookback: int = 0         # SWING_POINTS
    stop_loss_fib_level: float = 0.0          # FIBONACCI
    stop_loss_ai_level: int = 0               # DYNAMIC_AI

    # Trailing Stop
    trailing_stop_enabled: bool = False
    trailing_activation_profit_percent: float = 1.0
    trailing_callback_percent: float = 0.5
    trailing_take_profit: bool = False
    trailing_distance: float = 0.5
    
    # Break Even
    break_even_enabled: bool = False
    break_even_trigger_profit_percent: float = 1.0
    break_even_offset: float = 0.1
    
    # Partial Exit
    partial_exit_enabled: bool = False
    partial_exit_levels: List[float] = field(default_factory=list)
    partial_exit_sizes: List[float] = field(default_factory=list)


@dataclass
class AIConfig:
    """
    AI Model Configuration

    AI model helps with strategy decisions:
    - Entry Decision: Evaluate the quality of the signal
    - TP/SL Optimization: Suggest optimal TP/SL values
    - Position Sizing: Risk-based position size
    - Exit Timing: Exit timing

    Attributes:
        # General
        ai_enabled: Is AI enabled?
        model_path: Model checkpoint file
        model_type: Model type (rl_model, signal_model, lstm, transformer)

        # Entry Decision
        entry_decision: Should be used in the AI entry decision?
        confidence_threshold: Minimum confidence threshold (0.5-0.9)

        # TP/SL Optimization
        tp_optimization: Should the AI suggest an optimal Take Profit (TP)?
        sl_optimization: Should the AI suggest an optimal Stop Loss (SL)?
        use_ai_tp: Use AI TP directly (override strategy TP)
        use_ai_sl: Use AI SL directly (override strategy SL)

        # Position Sizing
        position_sizing: Should the AI suggest position sizes?
        risk_assessment: Should the AI perform risk assessment?
        max_ai_position_mult: AI maximum position multiplier (1.0 = normal)

        # Exit Timing
        exit_timing: Should the AI suggest an exit timing?
        early_exit_enabled: Is early exit enabled?
        early_exit_threshold: Early exit loss threshold.

        # Break Even & Trailing
        ai_break_even: AI break-even tetiklemesi
        ai_trailing: AI trailing stop management

        # Lookback/Forward Config (per timeframe)
        lookback_bars: How many bars to look back (feature extraction)
        forward_bars: How many bars to look forward (default)
        forward_bars_1m: Forward bars for 1 minute
        forward_bars_5m: Forward bars for 5 minutes
        forward_bars_15m: Forward bars for 15 minutes
        forward_bars_30m: Forward bars for 30 minutes
        forward_bars_1h: Forward bars for 1 hour
        forward_bars_4h: Forward bars for 4 hours
        forward_bars_1d: Forward bars for 1 day
        forward_bars_1w: Forward bars for 1 week

    Example:
        ai_config = AIConfig(
            ai_enabled=True,
            model_path="data/checkpoints/ai/global_best.pt",
            entry_decision=True,
            confidence_threshold=0.6,
            tp_optimization=True,
            sl_optimization=True,
        )
    """
    # ═══════════════════════════════════════════════════════════════
    # GENERAL
    # ═══════════════════════════════════════════════════════════════
    ai_enabled: bool = False
    model_path: str = "models/global_best.pt"
    model_type: str = "rl_model"  # rl_model (PPO), signal_model (legacy), lstm, transformer

    # ═══════════════════════════════════════════════════════════════
    # ENTRY DECISION
    # ═══════════════════════════════════════════════════════════════
    entry_decision: bool = True           # Should be used in the AI entry decision?
    confidence_threshold: float = 0.6     # Min confidence (LONG > threshold, SHORT < 1-threshold)

    # ═══════════════════════════════════════════════════════════════
    # TP/SL OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════
    tp_optimization: bool = False         # Should the AI suggest the optimal TP?
    sl_optimization: bool = False         # Should the AI suggest an optimal stop loss?
    use_ai_tp: bool = False               # Use AI TP directly (override the strategy TP)
    use_ai_sl: bool = False               # Use AI SL directly (overrides the strategy SL)
    tp_blend_ratio: float = 0.5           # AI/Strategy TP blend ratio (0=strategy, 1=AI)
    sl_blend_ratio: float = 0.5           # AI/Strategy stop loss blend ratio (0=strategy, 1=AI)

    # ═══════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    position_sizing: bool = False         # Should the AI suggest position size?
    risk_assessment: bool = False         # Should it perform an AI risk assessment?
    max_ai_position_mult: float = 1.5     # AI max position multiplier (increase if confidence is high)
    min_ai_position_mult: float = 0.5     # AI minimum position multiplier (reduce if confidence is low)

    # ═══════════════════════════════════════════════════════════════
    # EXIT TIMING
    # ═══════════════════════════════════════════════════════════════
    exit_timing: bool = False             # Should the AI suggest exit timing?
    early_exit_enabled: bool = False      # Is early exit enabled?
    early_exit_profit_threshold: float = 0.5   # Minimum profit percentage for early exit
    early_exit_loss_threshold: float = -1.0    # Max loss percentage for early exit

    # ═══════════════════════════════════════════════════════════════
    # BREAK EVEN & TRAILING
    # ═══════════════════════════════════════════════════════════════
    ai_break_even: bool = False           # AI break-even tetiklemesi
    ai_trailing: bool = False             # AI trailing stop management
    ai_partial_exit: bool = False         # AI partial exit decision

    # ═══════════════════════════════════════════════════════════════
    # EXIT MODEL (Dynamic Exit Optimization)
    # ═══════════════════════════════════════════════════════════════
    exit_model_enabled: bool = False      # Is the Exit Model active?
    exit_model_path: str = "data/ai/checkpoints/exit_model.pkl"
    use_exit_model_tp: bool = False       # Should this override the Exit Model TP?
    use_exit_model_sl: bool = False       # Should this override the Exit Model SL?
    use_exit_model_trailing: bool = False # Should it make the Exit Model trailing decision?
    use_exit_model_break_even: bool = False # Should the Exit Model break-even decision be made?
    exit_model_blend_ratio: float = 1.0  # Exit Model blend ratio (0=strategy, 1=AI)

    # ═══════════════════════════════════════════════════════════════
    # LOOKBACK/FORWARD CONFIG
    # ═══════════════════════════════════════════════════════════════
    lookback_bars: int = 200              # Lookback period for feature extraction

    # Forward bars (prediction horizon) - time-based
    forward_bars: int = 24                # Default forward bars
    forward_bars_1m: int = 24             # 1m: 24 bar = 24 minutes
    forward_bars_5m: int = 24             # 5m: 24 bar = 2 hours
    forward_bars_15m: int = 24            # 15m: 24 bar = 6 hours
    forward_bars_30m: int = 24            # 30m: 24 bar = 12 hours
    forward_bars_1h: int = 24             # 1h: 24 bar = 24 hours
    forward_bars_4h: int = 24             # 4h: 24 bar = 4 days
    forward_bars_1d: int = 12             # 1d: 12 bar = 12 days
    forward_bars_1w: int = 5              # 1w: 5 bar = 5 weeks

    def get_forward_bars(self, timeframe: str) -> int:
        """Get forward bars for specific timeframe."""
        tf_map = {
            '1m': self.forward_bars_1m,
            '5m': self.forward_bars_5m,
            '15m': self.forward_bars_15m,
            '30m': self.forward_bars_30m,
            '1h': self.forward_bars_1h,
            '4h': self.forward_bars_4h,
            '1d': self.forward_bars_1d,
            '1w': self.forward_bars_1w,
        }
        return tf_map.get(timeframe, self.forward_bars)


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class BaseStrategy(ABC):
    """
    Base strategy class
    
    All strategy templates inherit from this class.
    
    Example:
        class MyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()
                
                # Metadata
                self.strategy_name = "My Strategy"
                self.strategy_version = "1.0.0"
                
                # Config
                self.symbols = [SymbolConfig(...)]
                self.risk_management = RiskManagement(...)
                self.entry_conditions = {...}
    """
    
    def __init__(self):
        """Initialize base strategy"""
        
        # ====================================================================
        # METADATA
        # ====================================================================
        self.strategy_name: str = "Unnamed Strategy"
        self.strategy_version: str = "1.0.0"
        self.description: str = ""
        self.author: str = "Unknown"
        self.created_date: str = ""
        
        # ====================================================================
        # DATA MANAGEMENT
        # ====================================================================
        self.backtesting_enabled: bool = False
        self.backtest_start_date: Optional[str] = None
        self.backtest_end_date: Optional[str] = None
        self.initial_balance: float = 10000.0
        self.download_klines: bool = False
        self.update_klines: bool = False
        self.warmup_period: int = 200
        
        # Backtest parameters
        self.backtest_parameters: Dict[str, Any] = {
            "min_spread": 0.0,
            "commission": 0.0,
            "max_slippage": 0.0
        }
        
        # ====================================================================
        # SYMBOL MANAGEMENT
        # ====================================================================
        self.symbol_source: str = "strategy"
        self.symbols: List[SymbolConfig] = []
        
        # ====================================================================
        # ACCOUNT MANAGEMENT
        # ====================================================================
        self.side_method: TradingSide = TradingSide.BOTH
        self.leverage: int = 1
        self.set_default_leverage: bool = False
        self.hedge_mode: bool = False
        self.set_margin_type: bool = False
        self.margin_type: str = "isolated"
        
        # ====================================================================
        # INDICATOR MANAGEMENT
        # ====================================================================
        self.mtf_timeframes: List[str] = ["5m"]
        self.primary_timeframe: str = "5m"
        self.technical_parameters: TechnicalParameters = TechnicalParameters()
        
        # ====================================================================
        # SIGNAL MANAGEMENT
        # ====================================================================
        self.entry_conditions: Dict[str, List] = {
            "long": [],
            "short": []
        }
        
        # ====================================================================
        # EXIT MANAGEMENT
        # ====================================================================
        self.exit_strategy: Optional[ExitStrategy] = None
        self.exit_conditions: Dict[str, List] = {
            "long": [],
            "short": [],
            "stop_loss": [],
            "take_profit": []
        }
        
        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management: Optional[RiskManagement] = None
        
        # ====================================================================
        # POSITION MANAGEMENT
        # ====================================================================
        self.position_management: Optional[PositionManagement] = None
        
        # ====================================================================
        # MARKET MANAGEMENT
        # ====================================================================
        self.custom_parameters: Dict[str, Any] = {
            "news_filter": False,
            "session_filter": {
                "enabled": False,
                "sydney": False,
                "tokyo": False,
                "london": False,
                "new_york": False,
                "london_ny_overlap": False,
            },
            "time_filter": {
                "enabled": False,
                "start_hour": 0,
                "end_hour": 24,
                "exclude_hours": [],
            },
            "day_filter": {
                "enabled": False,
                "monday": True,
                "tuesday": True,
                "wednesday": True,
                "thursday": True,
                "friday": True,
                "saturday": True,
                "sunday": True,
            },
        }
        
        # ====================================================================
        # OPTIMIZER MANAGEMENT (for backtesting, not used in trading)
        # ====================================================================
        self.optimizer_parameters: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Called when a subclass is created.
        Ensures that the primary TF is within the MTF.
        """
        super().__init_subclass__(**kwargs)

        # Wrap the original __init__
        original_init = cls.__init__

        def wrapped_init(self, *args, **kw):
            original_init(self, *args, **kw)
            self._ensure_primary_tf_in_mtf()

        cls.__init__ = wrapped_init

    def _ensure_primary_tf_in_mtf(self) -> None:
        """
        Ensure that the primary timeframe is present in the MTF list.

        In some strategies, the primary_timeframe may not be within the mtf_timeframes.
        This security check always ensures that the primary timeframe is within the MTF.
        """
        if not hasattr(self, 'primary_timeframe') or not self.primary_timeframe:
            return

        if not hasattr(self, 'mtf_timeframes'):
            self.mtf_timeframes = []

        if self.primary_timeframe not in self.mtf_timeframes:
            self.mtf_timeframes.append(self.primary_timeframe)

    # ========================================================================
    # OPTIONAL OVERRIDE METHODS
    # ========================================================================
    
    def on_init(self) -> None:
        """
        Strategy initialization hook (optional override)
        
        Args:
            symbol: Trading symbol
        """
        pass
    
    def on_bar_close(self, symbol: str, timeframe: str, data: Any) -> None:
        """
        Called on every bar close (optional override)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: Market data (DataFrame or dict)
        """
        pass
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_config(self) -> bool:
        """
        Config validation (will be implemented by validation.py)
        
        Returns:
            True if valid, raises exception otherwise
        """
        # This method will be implemented by helpers/validation.py
        # Placeholder for now
        return True
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_all_timeframes(self) -> List[str]:
        """Return all used timeframes"""
        return self.mtf_timeframes
    
    def get_indicator_names(self) -> List[str]:
        """Returns all indicator names"""
        return list(self.technical_parameters.indicators.keys())
    
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.strategy_name}' "
            f"version='{self.strategy_version}'>"
        )
    
    @property
    def max_portfolio_risk(self) -> float:
        """
        Auto-calculated maximum portfolio risk based on leverage

        Formula: max_portfolio_risk = leverage × 100

        Examples:
            leverage=1  → max_portfolio_risk=100  (no leverage, max 100% exposure)
            leverage=5  → max_portfolio_risk=500  (5x leverage, max 500% notional)
            leverage=10 → max_portfolio_risk=1000 (10x leverage, max 1000% notional)
            leverage=20 → max_portfolio_risk=2000 (20x leverage, max 2000% notional)

        Returns:
            Maximum portfolio risk percentage (auto-calculated)
        """
        return self.leverage * 100.0

    @property
    def symbol(self) -> str:
        """
        Get first symbol for backtesting (helper property)

        Returns:
            First symbol from symbols list (e.g., 'BTCUSDT')
            Defaults to 'BTCUSDT' if no symbols configured
        """
        if self.symbols and len(self.symbols) > 0:
            first_symbol_config = self.symbols[0]
            if first_symbol_config.symbol and len(first_symbol_config.symbol) > 0:
                # symbols = ['BTC', 'ETH'], quote = 'USDT' -> 'BTCUSDT'
                return f"{first_symbol_config.symbol[0]}{first_symbol_config.quote}"
        return "BTCUSDT"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'TradingSide',
    'PositionSizeMethod',
    'ExitMethod',
    'StopLossMethod',
    
    # Config dataclasses
    'SymbolConfig',
    'TechnicalParameters',
    'RiskManagement',
    'PositionManagement',
    'ExitStrategy',
    
    # Base class
    'BaseStrategy',
]
