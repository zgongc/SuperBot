#!/usr/bin/env python3
"""
components/strategies/exit_manager.py
SuperBot - Exit Manager

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Exit strategy management:
    - Stop Loss calculation (all methods)
    - Take Profit calculation (all methods)
    - Trailing stop logic
    - Break-even logic
    - Partial exits

Usage:
    from components.strategies.exit_manager import ExitManager
    
    manager = ExitManager(strategy)
    sl_price = manager.calculate_stop_loss(entry_price, side, data)
    tp_price = manager.calculate_take_profit(entry_price, side, data)
"""

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import pandas as pd

from components.strategies.base_strategy import (
    BaseStrategy,
    ExitMethod,
    StopLossMethod,
)

# Lazy import for AI predictor (avoid circular imports)
if TYPE_CHECKING:
    from modules.ai.inference.predictor import RLPredictor


class ExitManager:
    """
    Exit strategy manager
    
    SL, TP, trailing, break-even logic is managed.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        logger: Any = None,
        ai_predictor: Optional["RLPredictor"] = None
    ):
        """
        Initialize ExitManager

        Args:
            strategy: BaseStrategy instance
            logger: Logger instance (optional)
            ai_predictor: RLPredictor instance for DYNAMIC_AI exit methods
        """
        self.strategy = strategy
        self.logger = logger
        self.exit_strategy = strategy.exit_strategy
        self.ai_predictor = ai_predictor

        if not self.exit_strategy:
            raise ValueError("Strategy exit_strategy is required")
    
    # ========================================================================
    # STOP LOSS CALCULATION
    # ========================================================================
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        data: Optional[pd.DataFrame] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate the stop loss price.

        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            data: Market data (optional, for swing/fibonacci)
            atr_value: ATR value (optional, for ATR-based)

        Returns:
            Stop loss price or None (disabled)
        """
        method = self.exit_strategy.stop_loss_method

        # FIXED_PERCENT
        if method == StopLossMethod.FIXED_PERCENT:
            percent = self.exit_strategy.stop_loss_percent
            if percent == 0:
                return None
            return self._calculate_sl_fixed_percent(entry_price, percent, side)

        # FIXED_PRICE
        elif method == StopLossMethod.FIXED_PRICE:
            price = self.exit_strategy.stop_loss_price
            if price == 0:
                return None
            return price

        # ATR_BASED
        elif method == StopLossMethod.ATR_BASED:
            multiplier = self.exit_strategy.stop_loss_atr_multiplier
            if multiplier == 0:
                return None
            if atr_value is None:
                if self.logger:
                    self.logger.warning("ATR_BASED SL but ATR value is missing")
                return None
            return self._calculate_sl_atr_based(entry_price, atr_value, multiplier, side)

        # SWING_POINTS
        elif method == StopLossMethod.SWING_POINTS:
            lookback = self.exit_strategy.stop_loss_swing_lookback
            if lookback == 0:
                return None
            if data is None or data.empty:
                if self.logger:
                    self.logger.warning("SWING_POINTS SL but data is missing")
                return None
            return self._calculate_sl_swing_point(entry_price, data, side, lookback)

        # FIBONACCI
        elif method == StopLossMethod.FIBONACCI:
            fib_level = self.exit_strategy.stop_loss_fib_level
            if fib_level == 0:
                return None
            if data is None or data.empty:
                if self.logger:
                    self.logger.warning("FIBONACCI SL but no data")
                return None
            return self._calculate_sl_fibonacci(entry_price, data, fib_level, side)

        # DYNAMIC_AI
        elif method == StopLossMethod.DYNAMIC_AI:
            ai_level = self.exit_strategy.stop_loss_ai_level
            if data is None or data.empty:
                if self.logger:
                    self.logger.warning("DYNAMIC_AI SL but no data")
                return None
            return self._calculate_sl_dynamic_ai(entry_price, data, side, ai_level, atr_value)

        # Default
        else:
            return None
    
    def _calculate_sl_fixed_percent(
        self,
        entry_price: float,
        percent: float,
        side: str
    ) -> float:
        """FIXED_PERCENT stop loss"""
        if side.upper() == 'LONG':
            return entry_price * (1 - percent / 100.0)
        else:  # SHORT
            return entry_price * (1 + percent / 100.0)
    
    def _calculate_sl_atr_based(
        self,
        entry_price: float,
        atr_value: float,
        multiplier: float,
        side: str
    ) -> float:
        """ATR_BASED stop loss"""
        distance = atr_value * multiplier
        
        if side.upper() == 'LONG':
            return entry_price - distance
        else:  # SHORT
            return entry_price + distance
    
    def _calculate_sl_swing_point(
        self,
        entry_price: float,
        data: pd.DataFrame,
        side: str,
        lookback: int = 20
    ) -> Optional[float]:
        """
        SWING_POINTS stop loss - Simple lookback min/max

        lookback=3 -> Use the lowest/highest price of the last 3 candles.
        """
        try:
            # Need enough data
            if len(data) < lookback:
                if self.logger:
                    self.logger.warning(f"SWING_POINTS: Not enough data ({len(data)} < {lookback})")
                return None

            # Get last N bars
            recent_data = data.tail(lookback)

            if side.upper() == 'LONG':
                # LONG: The lowest price (dip) of the last N candles.
                swing_low = recent_data['low'].min()
                # Add small buffer below swing low
                return swing_low * 0.998
            else:  # SHORT
                # SHORT: The highest price (peak) of the last N candles.
                swing_high = recent_data['high'].max()
                # Add small buffer above swing high
                return swing_high * 1.002

        except Exception as e:
            if self.logger:
                self.logger.warning(f"SWING_POINTS SL calculation error: {e}")
            return None
    
    def _calculate_sl_support_resistance(
        self,
        entry_price: float,
        data: pd.DataFrame,
        side: str
    ) -> float:
        """SUPPORT_RESISTANCE stop loss"""
        # Simplified: same as swing points (real SR detection is needed)
        return self._calculate_sl_swing_point(entry_price, data, side, lookback=20)

    def _calculate_sl_fibonacci(
        self,
        entry_price: float,
        data: pd.DataFrame,
        fib_level: float,
        side: str
    ) -> Optional[float]:
        """FIBONACCI stop loss (uses FibonacciRetracement indicator)"""
        try:
            from components.indicators.support_resistance.fib_retracement import FibonacciRetracement

            # Use 50-bar lookback for fibonacci calculation
            fib_indicator = FibonacciRetracement(lookback=50)

            # Need enough data
            if len(data) < fib_indicator.get_required_periods():
                if self.logger:
                    self.logger.warning(f"FIBONACCI: Not enough data ({len(data)} < {fib_indicator.get_required_periods()})")
                return None

            result = fib_indicator.calculate(data)
            levels = result.value  # Dict of fibonacci levels

            # For SL, use retracement levels (0.382, 0.5, 0.618, 0.786)
            # Map fib_level parameter to closest fibonacci level
            fib_key = None
            if abs(fib_level - 0.382) < 0.01:
                fib_key = 'Fib_38.2'
            elif abs(fib_level - 0.5) < 0.01:
                fib_key = 'Fib_50.0'
            elif abs(fib_level - 0.618) < 0.01:
                fib_key = 'Fib_61.8'
            elif abs(fib_level - 0.786) < 0.01:
                fib_key = 'Fib_78.6'
            else:
                # Default to 50% retracement
                fib_key = 'Fib_50.0'

            sl_price = levels.get(fib_key)
            if sl_price is None:
                if self.logger:
                    self.logger.warning(f"FIBONACCI: Level {fib_key} not found")
                return None

            # For LONG: SL should be below entry (use lower fib levels)
            # For SHORT: SL should be above entry (use higher fib levels)
            if side.upper() == 'LONG':
                # Ensure SL is below entry
                if sl_price >= entry_price:
                    sl_price = entry_price * 0.985  # Fallback to 1.5% below
            else:  # SHORT
                # Ensure SL is above entry
                if sl_price <= entry_price:
                    sl_price = entry_price * 1.015  # Fallback to 1.5% above

            return sl_price

        except Exception as e:
            if self.logger:
                self.logger.warning(f"FIBONACCI SL calculation error: {e}")
            return None
    
    def _calculate_sl_volatility_based(
        self,
        entry_price: float,
        data: pd.DataFrame,
        multiplier: float,
        side: str
    ) -> float:
        """VOLATILITY_BASED stop loss (std dev based)"""
        lookback = 20
        
        if 'close' in data.columns:
            # Standard deviation hesapla
            returns = data['close'].pct_change().tail(lookback)
            std_dev = returns.std()
            distance = entry_price * std_dev * multiplier
            
            if side.upper() == 'LONG':
                return entry_price - distance
            else:  # SHORT
                return entry_price + distance
        else:
            # Fallback
            return self._calculate_sl_fixed_percent(entry_price, multiplier, side)

    def _calculate_sl_dynamic_ai(
        self,
        entry_price: float,
        data: pd.DataFrame,
        side: str,
        ai_level: int = 1,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        """
        DYNAMIC_AI stop loss - Uses AI predictor for optimal SL.

        Args:
            entry_price: Entry price
            data: OHLCV DataFrame
            side: 'LONG' or 'SHORT'
            ai_level: AI aggressiveness level (1=conservative, 2=moderate, 3=aggressive)
            atr_value: ATR value (optional, calculated if not provided)

        Returns:
            Stop loss price or None if AI unavailable
        """
        if self.ai_predictor is None:
            if self.logger:
                self.logger.warning("DYNAMIC_AI SL: AI predictor not available, using ATR fallback")
            # Fallback to ATR-based if no AI predictor
            if atr_value is not None:
                multiplier = 1.5 + (ai_level - 1) * 0.5  # level 1=1.5x, 2=2x, 3=2.5x
                return self._calculate_sl_atr_based(entry_price, atr_value, multiplier, side)
            return None

        try:
            # Use AI predictor's optimize_tp_sl method
            # Strategy SL is set to 0 to let AI fully decide
            result = self.ai_predictor.optimize_tp_sl(
                df=data,
                side=side.upper(),
                strategy_tp=0.0,  # Not used for SL
                strategy_sl=2.0,  # Default fallback
                entry_price=entry_price,
                force_tp=False,
                force_sl=True  # Force AI SL calculation
            )

            sl_percent = result.get('sl_percent', 2.0)

            # Apply ai_level modifier (higher level = wider SL)
            # Level 1: 0.8x (tighter), Level 2: 1.0x (normal), Level 3: 1.2x (wider)
            level_multiplier = 0.8 + (ai_level - 1) * 0.2
            sl_percent *= level_multiplier

            # Calculate SL price
            if side.upper() == 'LONG':
                return entry_price * (1 - sl_percent / 100.0)
            else:  # SHORT
                return entry_price * (1 + sl_percent / 100.0)

        except Exception as e:
            if self.logger:
                self.logger.warning(f"DYNAMIC_AI SL calculation error: {e}")
            return None

    # ========================================================================
    # TAKE PROFIT CALCULATION
    # ========================================================================
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        stop_loss_price: Optional[float] = None,
        atr_value: Optional[float] = None,
        data: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """
        Calculate the take profit price.

        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            stop_loss_price: SL price (for RISK_REWARD)
            atr_value: ATR value (for ATR_BASED)
            data: Market data (for FIBONACCI)

        Returns:
            Take profit price or None (disabled)
        """
        method = self.exit_strategy.take_profit_method

        # FIXED_PERCENT
        if method == ExitMethod.FIXED_PERCENT:
            percent = self.exit_strategy.take_profit_percent
            if percent == 0:
                return None
            return self._calculate_tp_fixed_percent(entry_price, percent, side)

        # FIXED_PRICE
        elif method == ExitMethod.FIXED_PRICE:
            price = self.exit_strategy.take_profit_price
            if price == 0:
                return None
            return price

        # RISK_REWARD
        elif method == ExitMethod.RISK_REWARD:
            ratio = self.exit_strategy.take_profit_risk_reward_ratio
            if ratio == 0:
                return None
            if stop_loss_price is None:
                if self.logger:
                    self.logger.warning("RISK_REWARD TP but no SL")
                return None
            return self._calculate_tp_risk_reward(entry_price, stop_loss_price, ratio, side)

        # ATR_BASED
        elif method == ExitMethod.ATR_BASED:
            multiplier = self.exit_strategy.take_profit_atr_multiplier
            if multiplier == 0:
                return None
            if atr_value is None:
                if self.logger:
                    self.logger.warning("ATR_BASED TP but ATR is not available")
                return None
            return self._calculate_tp_atr_based(entry_price, atr_value, multiplier, side)

        # FIBONACCI
        elif method == ExitMethod.FIBONACCI:
            fib_level = self.exit_strategy.take_profit_fib_level
            if fib_level == 0:
                return None
            if data is None or data.empty:
                if self.logger:
                    self.logger.warning("FIBONACCI TP but no data")
                return None
            return self._calculate_tp_fibonacci(entry_price, data, fib_level, side)

        # DYNAMIC_AI
        elif method == ExitMethod.DYNAMIC_AI:
            ai_level = self.exit_strategy.take_profit_ai_level
            if data is None or data.empty:
                if self.logger:
                    self.logger.warning("DYNAMIC_AI TP but no data")
                return None
            return self._calculate_tp_dynamic_ai(entry_price, data, side, ai_level, stop_loss_price, atr_value)

        # Default
        else:
            return None
    
    def _calculate_tp_fixed_percent(
        self,
        entry_price: float,
        percent: float,
        side: str
    ) -> float:
        """FIXED_PERCENT take profit"""
        if side.upper() == 'LONG':
            return entry_price * (1 + percent / 100.0)
        else:  # SHORT
            return entry_price * (1 - percent / 100.0)
    
    def _calculate_tp_risk_reward(
        self,
        entry_price: float,
        stop_loss_price: float,
        ratio: float,
        side: str
    ) -> float:
        """RISK_REWARD take profit"""
        risk = abs(entry_price - stop_loss_price)
        reward = risk * ratio
        
        if side.upper() == 'LONG':
            return entry_price + reward
        else:  # SHORT
            return entry_price - reward
    
    def _calculate_tp_atr_based(
        self,
        entry_price: float,
        atr_value: float,
        multiplier: float,
        side: str
    ) -> float:
        """ATR_BASED take profit"""
        distance = atr_value * multiplier

        if side.upper() == 'LONG':
            return entry_price + distance
        else:  # SHORT
            return entry_price - distance

    def _calculate_tp_fibonacci(
        self,
        entry_price: float,
        data: pd.DataFrame,
        fib_level: float,
        side: str
    ) -> Optional[float]:
        """FIBONACCI take profit (uses Fibonacci extensions: 1.272, 1.618, 2.0, 2.618)"""
        try:
            from components.indicators.support_resistance.fib_retracement import FibonacciRetracement

            # Use 50-bar lookback for fibonacci calculation
            fib_indicator = FibonacciRetracement(lookback=50)

            # Need enough data
            if len(data) < fib_indicator.get_required_periods():
                if self.logger:
                    self.logger.warning(f"FIBONACCI: Not enough data ({len(data)} < {fib_indicator.get_required_periods()})")
                return None

            result = fib_indicator.calculate(data)
            high = result.metadata.get('high')
            low = result.metadata.get('low')
            is_uptrend = result.metadata.get('trend') == 'uptrend'

            if high is None or low is None:
                return None

            # Calculate Fibonacci extensions for TP
            # Extensions: 1.272, 1.618, 2.0, 2.618
            range_hl = high - low

            if is_uptrend:
                # In uptrend: extend above high
                # Map fib_level to extension multiplier
                if abs(fib_level - 1.272) < 0.01:
                    tp_price = high + (range_hl * 0.272)
                elif abs(fib_level - 1.618) < 0.01:
                    tp_price = high + (range_hl * 0.618)
                elif abs(fib_level - 2.0) < 0.01:
                    tp_price = high + (range_hl * 1.0)
                elif abs(fib_level - 2.618) < 0.01:
                    tp_price = high + (range_hl * 1.618)
                else:
                    # Default to 1.618 extension
                    tp_price = high + (range_hl * 0.618)
            else:
                # In downtrend: extend below low
                if abs(fib_level - 1.272) < 0.01:
                    tp_price = low - (range_hl * 0.272)
                elif abs(fib_level - 1.618) < 0.01:
                    tp_price = low - (range_hl * 0.618)
                elif abs(fib_level - 2.0) < 0.01:
                    tp_price = low - (range_hl * 1.0)
                elif abs(fib_level - 2.618) < 0.01:
                    tp_price = low - (range_hl * 1.618)
                else:
                    # Default to 1.618 extension
                    tp_price = low - (range_hl * 0.618)

            # Validate TP is in correct direction
            if side.upper() == 'LONG':
                # TP should be above entry
                if tp_price <= entry_price:
                    tp_price = entry_price * 1.05  # Fallback to 5% above
            else:  # SHORT
                # TP should be below entry
                if tp_price >= entry_price:
                    tp_price = entry_price * 0.95  # Fallback to 5% below

            return tp_price

        except Exception as e:
            if self.logger:
                self.logger.warning(f"FIBONACCI TP calculation error: {e}")
            return None

    def _calculate_tp_dynamic_ai(
        self,
        entry_price: float,
        data: pd.DataFrame,
        side: str,
        ai_level: int = 1,
        stop_loss_price: Optional[float] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        """
        DYNAMIC_AI take profit - AI decides when to exit based on prediction.

        DYNAMIC_AI TP logic:
        - Wait for the AI to give a CLOSE or reverse signal instead of the TP price.
        - This method only calculates fallback TP values (if AI is not available).
        - The actual exit decision is made by check_ai_exit().

        Args:
            entry_price: Entry price
            data: OHLCV DataFrame
            side: 'LONG' or 'SHORT'
            ai_level: AI aggressiveness level (1=conservative, 2=moderate, 3=aggressive)
            stop_loss_price: Stop loss price (for R:R calculation fallback)
            atr_value: ATR value (optional, calculated if not provided)

        Returns:
            Fallback TP price (AI exit decision is checked separately)
        """
        # In DYNAMIC_AI mode, the TP price is kept very far away.
        # Because the actual exit decision is made based on AI prediction.
        # This only works as a "maximum TP" limit.

        if self.ai_predictor is None:
            if self.logger:
                self.logger.warning("DYNAMIC_AI TP: AI predictor not available, using ATR fallback")
            # AI yoksa ATR-based fallback
            if atr_value is not None:
                multiplier = 2.0 + (ai_level - 1) * 1.0
                return self._calculate_tp_atr_based(entry_price, atr_value, multiplier, side)
            elif stop_loss_price is not None:
                ratio = 1.5 + (ai_level - 1) * 0.5
                return self._calculate_tp_risk_reward(entry_price, stop_loss_price, ratio, side)
            return None

        # If AI exists, set a very wide TP limit (until the AI makes a decision)
        # The AI exit signal is checked using the check_ai_exit() method.
        max_tp_percent = 10.0 + (ai_level - 1) * 5.0  # level 1=10%, 2=15%, 3=20%

        if side.upper() == 'LONG':
            return entry_price * (1 + max_tp_percent / 100.0)
        else:  # SHORT
            return entry_price * (1 - max_tp_percent / 100.0)

    def check_ai_exit(
        self,
        data: pd.DataFrame,
        side: str,
        current_pnl_percent: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check if the AI has sent an exit signal.

        DYNAMIC_AI exit logic:
        - AI CLOSE sinyali verirse → Exit
        - AI ters sinyal verirse (LONG'dayken SHORT) → Exit
        - If AI confidence is low -> Hold (do not exit)

        Args:
            data: OHLCV DataFrame
            side: Current position side ('LONG' or 'SHORT')
            current_pnl_percent: Current unrealized PnL %

        Returns:
            {
                'should_exit': bool,
                'reason': str,
                'ai_action': str,
                'confidence': float
            }
        """
        if self.ai_predictor is None:
            return {
                'should_exit': False,
                'reason': 'AI predictor not available',
                'ai_action': None,
                'confidence': 0.0
            }

        try:
            prediction = self.ai_predictor.predict(data)

            if 'error' in prediction:
                return {
                    'should_exit': False,
                    'reason': f"Prediction error: {prediction.get('error')}",
                    'ai_action': None,
                    'confidence': 0.0
                }

            ai_action = prediction.get('action', 'HOLD')
            confidence = prediction.get('confidence', 0.0)
            probs = prediction.get('probabilities', {})

            # Exit conditions:
            # 1. AI CLOSE sinyali verdi
            if ai_action == 'CLOSE':
                return {
                    'should_exit': True,
                    'reason': f"AI signals CLOSE with {confidence:.1%} confidence",
                    'ai_action': ai_action,
                    'confidence': confidence
                }

            # 2. AI gave the reverse signal (with high confidence)
            if side.upper() == 'LONG' and ai_action == 'SHORT':
                short_prob = probs.get('SHORT', 0)
                if short_prob > 0.6:  # Probability of SHORT is 60%+
                    return {
                        'should_exit': True,
                        'reason': f"AI signals SHORT ({short_prob:.1%}) while in LONG",
                        'ai_action': ai_action,
                        'confidence': confidence
                    }

            if side.upper() == 'SHORT' and ai_action == 'LONG':
                long_prob = probs.get('LONG', 0)
                if long_prob > 0.6:  # If the probability of LONG is 60% or higher
                    return {
                        'should_exit': True,
                        'reason': f"AI signals LONG ({long_prob:.1%}) while in SHORT",
                        'ai_action': ai_action,
                        'confidence': confidence
                    }

            # 3. It says "snow" and "AI HOLD" but with low confidence.
            if current_pnl_percent > 1.0 and confidence < 0.4:
                return {
                    'should_exit': True,
                    'reason': f"Low AI confidence ({confidence:.1%}) with profit, taking profit",
                    'ai_action': ai_action,
                    'confidence': confidence
                }

            # No exit, continue
            return {
                'should_exit': False,
                'reason': f"AI says {ai_action} with {confidence:.1%} confidence",
                'ai_action': ai_action,
                'confidence': confidence
            }

        except Exception as e:
            if self.logger:
                self.logger.warning(f"AI exit check error: {e}")
            return {
                'should_exit': False,
                'reason': f"Error: {e}",
                'ai_action': None,
                'confidence': 0.0
            }

    # ========================================================================
    # TRAILING STOP
    # ========================================================================
    
    def should_update_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        side: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Should the trailing stop be updated?
        
        Args:
            entry_price: Entry price
            current_price: Current price
            current_sl: Current SL price
            side: 'LONG' or 'SHORT'
        
        Returns:
            (should_update, new_sl_price)
        """
        if not self.exit_strategy.trailing_stop_enabled:
            return False, None
        
        # Trailing activation check
        activation_percent = self.exit_strategy.trailing_activation_profit_percent
        current_profit_percent = self._calculate_profit_percent(
            entry_price,
            current_price,
            side
        )
        
        if current_profit_percent < activation_percent:
            # Not active yet
            return False, None
        
        # Trailing callback percentage
        callback_percent = self.exit_strategy.trailing_callback_percent

        # If no SL set, cannot trail
        if current_sl is None:
            return False, None

        # Yeni SL hesapla
        if side.upper() == 'LONG':
            new_sl = current_price * (1 - callback_percent / 100.0)

            # SL should move upwards (trailing)
            if new_sl > current_sl:
                return True, new_sl
        else:  # SHORT
            new_sl = current_price * (1 + callback_percent / 100.0)

            # SL should move down (trailing)
            if new_sl < current_sl:
                return True, new_sl

        return False, None
    
    # ========================================================================
    # BREAK-EVEN
    # ========================================================================
    
    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        side: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Should the SL be adjusted to the break-even point?
        
        Args:
            entry_price: Entry price
            current_price: Current price
            current_sl: Current SL price
            side: 'LONG' or 'SHORT'
        
        Returns:
            (should_move, breakeven_price)
        """
        if not self.exit_strategy.break_even_enabled:
            return False, None
        
        # Trigger profit control
        trigger_percent = self.exit_strategy.break_even_trigger_profit_percent
        current_profit_percent = self._calculate_profit_percent(
            entry_price,
            current_price,
            side
        )
        
        if current_profit_percent < trigger_percent:
            # Not yet triggered
            return False, None
        
        # Break-even offset
        offset_percent = self.exit_strategy.break_even_offset

        # If no SL set, cannot move to break-even
        if current_sl is None:
            return False, None

        if side.upper() == 'LONG':
            breakeven_price = entry_price * (1 + offset_percent / 100.0)

            # If the current SL is below the break-even point, update it.
            if current_sl < breakeven_price:
                return True, breakeven_price
        else:  # SHORT
            breakeven_price = entry_price * (1 - offset_percent / 100.0)

            # If it is above the current SL break-even, update it.
            if current_sl > breakeven_price:
                return True, breakeven_price

        return False, None
    
    # ========================================================================
    # PARTIAL EXITS
    # ========================================================================
    
    def get_partial_exit_size(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        completed_exits: int = 0
    ) -> Tuple[bool, float]:
        """
        Should a partial exit be performed? How many lots?
        
        Args:
            entry_price: Entry price
            current_price: Current price
            side: 'LONG' or 'SHORT'
            completed_exits: Number of partial exits performed
        
        Returns:
            (should_exit, exit_size_percent)
        """
        if not self.exit_strategy.partial_exit_enabled:
            return False, 0.0
        
        levels = self.exit_strategy.partial_exit_levels
        sizes = self.exit_strategy.partial_exit_sizes
        
        if completed_exits >= len(levels):
            # All partial exits are completed
            return False, 0.0
        
        # Current profit
        current_profit_percent = self._calculate_profit_percent(
            entry_price,
            current_price,
            side
        )
        
        # Check the next level
        next_level = levels[completed_exits]
        
        if current_profit_percent >= next_level:
            exit_size = sizes[completed_exits]
            return True, exit_size
        
        return False, 0.0
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_profit_percent(
        self,
        entry_price: float,
        current_price: float,
        side: str
    ) -> float:
        """
        Calculate the percentage of snow.
        
        Returns:
            Profit percent (positive = profit, negative = loss)
        """
        if side.upper() == 'LONG':
            return ((current_price - entry_price) / entry_price) * 100.0
        else:  # SHORT
            return ((entry_price - current_price) / entry_price) * 100.0
    
    # ========================================================================
    # COMPREHENSIVE EXIT CHECK (BACKTEST INTEGRATION)
    # ========================================================================
    
    def check_exit(
        self,
        position: Any,  # Position object with entry_price, side, entry_time, stop_loss, take_profit
        current_price: float,
        current_timestamp: Any,
        current_idx: int,
        position_management: Any = None,  # PositionManagement config
        timeframe_data: Optional[Dict] = None,  # MTF data for exit conditions
        current_bar_data: Optional[pd.DataFrame] = None  # Current bar for ATR, etc.
    ) -> Optional[Dict[str, Any]]:
        """
        Comprehensive exit control - For backtest integration.
        
        Check order:
        1. Position Timeout (if enabled)
        2. Stop Loss (basic or updated with trailing/break-even)
        3. Take Profit
        4. Trailing stop update
        5. Break-even update
        6. Partial Exit
        7. Exit Conditions (strategy defined)
        
        Args:
            position: Position object (entry_price, side, entry_time, stop_loss, take_profit)
            current_price: Current price (close)
            current_timestamp: Current timestamp (bar time)
            current_idx: Current bar index
            position_management: PositionManagement config (for timeout)
            timeframe_data: MTF data dict (for exit conditions)
            current_bar_data: Current bar DataFrame (for ATR, indicators)
        
        Returns:
            {
                'should_exit': bool,
                'reason': str,  # 'stop_loss', 'take_profit', 'timeout', 'exit_condition'
                'exit_price': float,
                'exit_timestamp': Any,
                'exit_idx': int,
                'update_sl': Optional[float],  # New SL for trailing or break-even
                'partial_exit_size': Optional[float]  # Partial exit varsa %
            }
            or None (no exit)
        """
        if current_idx <= getattr(position, 'entry_idx', 0):
            return None
        
        entry_price = position.entry_price
        side = position.side.lower()
        
        # Calculate current PnL %
        pnl_pct = self._calculate_profit_percent(entry_price, current_price, side)
        
        # ====================================================================
        # 1. POSITION TIMEOUT CHECK
        # ====================================================================
        if position_management and getattr(position_management, 'position_timeout_enabled', False):
            timeout_minutes = getattr(position_management, 'position_timeout', 1800)
            
            # Calculate time in position (in minutes)
            entry_ts = position.entry_time if isinstance(position.entry_time, (int, float)) else pd.Timestamp(position.entry_time).timestamp() * 1000
            current_ts = current_timestamp if isinstance(current_timestamp, (int, float)) else pd.Timestamp(current_timestamp).timestamp() * 1000
            time_in_position_minutes = (current_ts - entry_ts) / 1000 / 60
            
            if time_in_position_minutes >= timeout_minutes:
                return {
                    'should_exit': True,
                    'reason': f'timeout ({time_in_position_minutes:.0f}m/{timeout_minutes}m)',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
        
        # ====================================================================
        # 2. STOP LOSS CHECK
        # ====================================================================
        current_sl = getattr(position, 'stop_loss', None)
        if current_sl is not None:
            # Long: price <= SL
            # Short: price >= SL
            if side == 'long' and current_price <= current_sl:
                return {
                    'should_exit': True,
                    'reason': 'stop_loss',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
            elif side == 'short' and current_price >= current_sl:
                return {
                    'should_exit': True,
                    'reason': 'stop_loss',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
        
        # ====================================================================
        # 2.5 DYNAMIC_AI EXIT CHECK (before TP check)
        # ====================================================================
        tp_method = self.exit_strategy.take_profit_method
        if tp_method == ExitMethod.DYNAMIC_AI and current_bar_data is not None:
            ai_exit = self.check_ai_exit(
                data=current_bar_data,
                side=side.upper(),
                current_pnl_percent=pnl_pct
            )
            if ai_exit.get('should_exit', False):
                return {
                    'should_exit': True,
                    'reason': f"ai_exit: {ai_exit.get('reason', 'AI signal')}",
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }

        # ====================================================================
        # 3. TAKE PROFIT CHECK
        # ====================================================================
        current_tp = getattr(position, 'take_profit', None)
        if current_tp is not None:
            # Long: price >= TP
            # Short: price <= TP
            if side == 'long' and current_price >= current_tp:
                return {
                    'should_exit': True,
                    'reason': 'take_profit',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
            elif side == 'short' and current_price <= current_tp:
                return {
                    'should_exit': True,
                    'reason': 'take_profit',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
        
        # ====================================================================
        # 4. TRAILING STOP UPDATE CHECK
        # ====================================================================
        if self.exit_strategy.trailing_stop_enabled and current_sl is not None:
            should_update, new_sl = self.should_update_trailing_stop(
                entry_price, current_price, current_sl, side
            )
            
            if should_update and new_sl is not None:
                # Return update signal (not exit, just update SL)
                return {
                    'should_exit': False,
                    'reason': 'trailing_update',
                    'exit_price': None,
                    'exit_timestamp': None,
                    'exit_idx': None,
                    'update_sl': new_sl,
                    'partial_exit_size': None
                }
        
        # ====================================================================
        # 5. BREAK-EVEN UPDATE CHECK
        # ====================================================================
        if self.exit_strategy.break_even_enabled and current_sl is not None:
            should_move, breakeven_price = self.should_move_to_breakeven(
                entry_price, current_price, current_sl, side
            )
            
            if should_move and breakeven_price is not None:
                # Return update signal (not exit, just move SL to break-even)
                return {
                    'should_exit': False,
                    'reason': 'breakeven_update',
                    'exit_price': None,
                    'exit_timestamp': None,
                    'exit_idx': None,
                    'update_sl': breakeven_price,
                    'partial_exit_size': None
                }
        
        # ====================================================================
        # 6. PARTIAL EXIT CHECK
        # ====================================================================
        if self.exit_strategy.partial_exit_enabled:
            completed_exits = getattr(position, 'completed_partial_exits', 0)
            should_exit_partial, exit_size = self.get_partial_exit_size(
                entry_price, current_price, side, completed_exits
            )
            
            if should_exit_partial:
                # Return partial exit signal
                return {
                    'should_exit': True,
                    'reason': f'partial_exit_{completed_exits + 1}',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': exit_size
                }
        
        # ====================================================================
        # 7. EXIT CONDITIONS CHECK (Strategy-defined)
        # ====================================================================
        if hasattr(self.strategy, 'exit_conditions') and timeframe_data:
            # Get exit conditions for this side
            exit_conditions = self.strategy.exit_conditions.get(side, [])
            
            if exit_conditions and self._check_exit_conditions(exit_conditions, timeframe_data):
                return {
                    'should_exit': True,
                    'reason': 'exit_condition',
                    'exit_price': current_price,
                    'exit_timestamp': current_timestamp,
                    'exit_idx': current_idx,
                    'update_sl': None,
                    'partial_exit_size': None
                }
        
        # No exit triggered
        return None
    
    def _check_exit_conditions(
        self,
        conditions: list,
        timeframe_data: Dict[str, Dict]
    ) -> bool:
        """
        Check exit conditions.
        
        Args:
            conditions: Exit conditions list
                Format: [['rsi_14', '>', 75, '15m'], ...]
            timeframe_data: MTF data dict
                Format: {'5m': {...}, '15m': {...}}
        
        Returns:
            True if ALL conditions pass
        """
        for condition in conditions:
            if len(condition) < 3:
                continue
            
            indicator = condition[0]      # "rsi_14", "ema_50"
            operator = condition[1]        # ">", "<", "crossover", etc.
            value = condition[2]           # 75, "ema_200"
            
            # Timeframe (optional, default to first available)
            if len(condition) >= 4:
                timeframe = condition[3]
            else:
                # Use first available timeframe
                timeframe = list(timeframe_data.keys())[0] if timeframe_data else None
            
            if not timeframe or timeframe not in timeframe_data:
                return False
            
            tf_data = timeframe_data[timeframe]
            
            # Get indicator value
            if indicator not in tf_data:
                return False
            
            indicator_value = tf_data[indicator]
            
            # Evaluate condition
            if not self._evaluate_condition(indicator_value, operator, value, tf_data, indicator):
                return False
        
        # All conditions passed
        return True
    
    def _evaluate_condition(
        self,
        indicator_value: Any,
        operator: str,
        value: Any,
        data: Dict[str, Any],
        indicator_name: str
    ) -> bool:
        """
        Evaluate a single condition.
        
        Args:
            indicator_value: Current indicator value
            operator: Comparison operator (">", "<", ">=", etc.)
            value: Threshold or comparison value
            data: Timeframe data dict
            indicator_name: Indicator name for array lookup
        
        Returns:
            True if condition passes
        """
        try:
            # Handle indicator-to-indicator comparison
            if isinstance(value, str) and value in data:
                compare_value = data[value]
            else:
                compare_value = value
            
            # Standard comparison operators
            if operator in [">", "above"]:
                return indicator_value > compare_value
            elif operator in ["<", "below"]:
                return indicator_value < compare_value
            elif operator in [">=", "gte", "greater_equal"]:
                return indicator_value >= compare_value
            elif operator in ["<=", "lte", "less_equal"]:
                return indicator_value <= compare_value
            elif operator in ["==", "equals"]:
                return indicator_value == compare_value
            elif operator in ["!=", "not_equals"]:
                return indicator_value != compare_value
            
            # ARRAY-BASED OPERATORS (crossover, crossunder, rising, falling)
            elif operator in ["crossover", "cross_over"]:
                # Requires array data for previous values
                arrays = data.get('_arrays')
                if not arrays or indicator_name not in arrays:
                    return False
                
                indicator_array = arrays[indicator_name]
                if len(indicator_array) < 2:
                    return False
                
                # Handle indicator-to-indicator crossover
                if isinstance(value, str) and value in arrays:
                    compare_array = arrays[value]
                    if len(compare_array) < 2:
                        return False
                    # Current: ind > comp, Previous: ind <= comp
                    return indicator_array[-1] > compare_array[-1] and indicator_array[-2] <= compare_array[-2]
                else:
                    # Crossover threshold
                    return indicator_array[-1] > compare_value and indicator_array[-2] <= compare_value
            
            elif operator in ["crossunder", "cross_under"]:
                # Requires array data for previous values
                arrays = data.get('_arrays')
                if not arrays or indicator_name not in arrays:
                    return False
                
                indicator_array = arrays[indicator_name]
                if len(indicator_array) < 2:
                    return False
                
                # Handle indicator-to-indicator crossunder
                if isinstance(value, str) and value in arrays:
                    compare_array = arrays[value]
                    if len(compare_array) < 2:
                        return False
                    # Current: ind < comp, Previous: ind >= comp
                    return indicator_array[-1] < compare_array[-1] and indicator_array[-2] >= compare_array[-2]
                else:
                    # Crossunder threshold
                    return indicator_array[-1] < compare_value and indicator_array[-2] >= compare_value
            
            elif operator in ["rising", "rising_n"]:
                arrays = data.get('_arrays')
                if not arrays or indicator_name not in arrays:
                    return False
                
                indicator_array = arrays[indicator_name]
                threshold = int(value) if isinstance(value, (int, float)) else 2
                
                if len(indicator_array) < (threshold + 1):
                    return False
                
                # Check if rising: each bar higher than previous
                for i in range(1, threshold + 1):
                    if indicator_array[-i] <= indicator_array[-i-1]:
                        return False
                return True
            
            elif operator in ["falling", "falling_n"]:
                arrays = data.get('_arrays')
                if not arrays or indicator_name not in arrays:
                    return False
                
                indicator_array = arrays[indicator_name]
                threshold = int(value) if isinstance(value, (int, float)) else 2
                
                if len(indicator_array) < (threshold + 1):
                    return False
                
                # Check if falling: each bar lower than previous
                for i in range(1, threshold + 1):
                    if indicator_array[-i] >= indicator_array[-i-1]:
                        return False
                return True
            
            elif operator in ["between"]:
                # Value must be [min, max]
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    return False
                return value[0] <= indicator_value <= value[1]
            
            elif operator in ["outside"]:
                # Value must be [min, max]
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    return False
                return indicator_value < value[0] or indicator_value > value[1]
            
            else:
                if self.logger:
                    self.logger.warning(f"Unknown operator in exit condition: {operator}")
                return False
        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Exit condition evaluation error: {e}")
            return False
    
    def __repr__(self) -> str:
        return (
            f"<ExitManager "
            f"sl={self.exit_strategy.stop_loss_method.value} "
            f"tp={self.exit_strategy.take_profit_method.value} "
            f"trailing={self.exit_strategy.trailing_stop_enabled} "
            f"breakeven={self.exit_strategy.break_even_enabled}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ExitManager',
]

