#!/usr/bin/env python3
"""
components/strategies/strategy_executor.py
SuperBot - Strategy Executor

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Strategy execution orchestrator.
    
    Coordinates all managers:
    - SignalValidator (entry/exit conditions)
    - ExitManager (SL/TP/trailing)
    - MarketValidator (session/time filters)
    - PatternGenerator (pattern detection)
    - PortfolioCoordinator (multi-symbol coordination)

Usage:
    from components.strategies.strategy_executor import StrategyExecutor
    
    executor = StrategyExecutor(strategy)
    result = executor.evaluate(symbol, data)
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd

from components.strategies.base_strategy import BaseStrategy
from components.strategies.signal_validator import SignalValidator
from components.strategies.exit_manager import ExitManager
from components.strategies.market_validator import MarketValidator
from components.strategies.pattern_generator import PatternGenerator
from components.strategies.portfolio_coordinator import PortfolioCoordinator


class StrategyExecutor:
    """
    Strategy Executor
    
    This class executes the strategy logic and coordinates all managers.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        position_manager: Optional[Any] = None,
        indicator_manager: Optional[Any] = None,
        logger: Any = None,
        ai_predictor: Optional[Any] = None
    ):
        """
        Initialize StrategyExecutor

        Args:
            strategy: BaseStrategy instance
            position_manager: PositionManager instance (optional)
            indicator_manager: IndicatorManager instance (optional)
            logger: Logger instance (optional)
            ai_predictor: RLPredictor instance for AI-based exits (optional)
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.indicator_manager = indicator_manager
        self.logger = logger
        self.ai_predictor = ai_predictor

        # Initialize managers
        self.signal_manager = SignalValidator(strategy, logger)
        self.exit_manager = ExitManager(strategy, logger, ai_predictor=ai_predictor)
        self.market_manager = MarketValidator(strategy, logger)
        self.pattern_manager = PatternGenerator(strategy, logger)
        self.portfolio_manager = PortfolioCoordinator(strategy, position_manager, logger)
        
        # Call strategy init hook
        strategy.on_init()
        
        if self.logger:
            self.logger.info(f"✅ StrategyExecutor initialized for {strategy.strategy_name}")
    
    # ========================================================================
    # MAIN EVALUATION
    # ========================================================================
    
    def evaluate(
        self,
        symbol: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_time: Optional[Any] = None,
        _verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the strategy (unified interface for engines)

        Args:
            symbol: Trading symbol
            data: Market data (with indicators calculated)
                - Single timeframe: pd.DataFrame
                - Multi-timeframe: {'5m': df, '15m': df, ...}
            current_time: Current timestamp (optional)
            _verbose: Temporary verbose flag (for debug status logging)

        Returns:
            Dict: Evaluation result
                {
                    'symbol': str,
                    'timestamp': ...,
                    'market_tradeable': bool,
                    'entry_signal': {...},
                    'exit_signals': {...},
                    'patterns': {...},
                    'can_open_position': bool,
                    'recommendation': 'BUY' | 'SELL' | 'HOLD' | 'CLOSE'
                }
        """
        result = {
            'symbol': symbol,
            'timestamp': current_time,
            'market_tradeable': False,
            'entry_signal': None,
            'exit_signals': {},
            'patterns': {},
            'can_open_position': False,
            'recommendation': 'HOLD',
            'reason': None,
        }

        # 1. Market tradeable check
        if current_time:
            result['market_tradeable'] = self.market_manager.is_market_tradeable(current_time)
        else:
            result['market_tradeable'] = True  # No time filter

        if not result['market_tradeable']:
            result['reason'] = 'market_not_tradeable'
            return result

        # 2. Pattern detection (if enabled)
        result['patterns'] = self.pattern_manager.detect_patterns(
            self._get_primary_data(data)
        )

        # 3. Entry signal evaluation
        result['entry_signal'] = self.signal_manager.evaluate_entry(
            symbol=symbol,
            data=data,
            _verbose=_verbose
        )
        
        # 4. Portfolio position check
        can_open, reason = self.portfolio_manager.can_open_position(symbol)
        result['can_open_position'] = can_open
        
        if not can_open:
            result['reason'] = reason
        
        # 5. Determine recommendation
        result['recommendation'] = self._determine_recommendation(result)

        # 6. Calculate stop_loss/take_profit if signal is valid
        if result['recommendation'] in ('BUY', 'SELL'):
            primary_data = self._get_primary_data(data)
            if primary_data is not None and len(primary_data) > 0:
                current_price = float(primary_data.iloc[-1]['close'])

                # Stop Loss
                if hasattr(self.strategy, 'exit_strategy') and self.strategy.exit_strategy:
                    if hasattr(self.strategy.exit_strategy, 'stop_loss_percent') and self.strategy.exit_strategy.stop_loss_percent:
                        sl_percent = self.strategy.exit_strategy.stop_loss_percent / 100.0
                        if result['recommendation'] == 'BUY':
                            result['stop_loss'] = current_price * (1 - sl_percent)
                        else:
                            result['stop_loss'] = current_price * (1 + sl_percent)

                    # Take Profit
                    if hasattr(self.strategy.exit_strategy, 'take_profit_percent') and self.strategy.exit_strategy.take_profit_percent:
                        tp_percent = self.strategy.exit_strategy.take_profit_percent / 100.0
                        if result['recommendation'] == 'BUY':
                            result['take_profit'] = current_price * (1 + tp_percent)
                        else:
                            result['take_profit'] = current_price * (1 - tp_percent)

            # Call strategy hook (if exists)
            if hasattr(self.strategy, 'on_signal') and callable(self.strategy.on_signal):
                self.strategy.on_signal({
                    'symbol': symbol,
                    'side': result['recommendation'],
                    'data': result
                })

        return result
    
    def evaluate_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Evaluate exit conditions.
        
        Args:
            symbol: Trading symbol
            position: Position data
                {
                    'id': str,
                    'side': 'LONG' | 'SHORT',
                    'entry_price': float,
                    'stop_loss': float,
                    'take_profit': float,
                    ...
                }
            data: Market data (with indicators)
            current_price: Current market price
        
        Returns:
            Dict: Exit evaluation result
                {
                    'should_exit': bool,
                    'exit_type': 'SL' | 'TP' | 'TRAILING' | 'SIGNAL' | None,
                    'reason': str,
                    'updated_sl': float (if trailing),
                    'updated_tp': float (if applicable),
                }
        """
        result = {
            'should_exit': False,
            'exit_type': None,
            'reason': None,
            'updated_sl': position.get('stop_loss'),
            'updated_tp': position.get('take_profit'),
        }

        side = position['side']
        entry_price = position['entry_price']
        current_sl = position.get('stop_loss')
        current_tp = position.get('take_profit')

        # 0. Check DYNAMIC_AI exit FIRST (before SL/TP checks)
        # AI decides when to exit - SL is just a safety net
        from components.strategies.base_strategy import ExitMethod
        tp_method = self.exit_manager.exit_strategy.take_profit_method
        if tp_method == ExitMethod.DYNAMIC_AI and self.ai_predictor is not None:
            # Calculate current PnL percentage
            if side.upper() == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # Check AI exit signal
            ai_exit = self.exit_manager.check_ai_exit(
                data=data,
                side=side,
                current_pnl_percent=pnl_pct
            )

            if ai_exit.get('should_exit', False):
                result['should_exit'] = True
                result['exit_type'] = 'AI_EXIT'
                result['reason'] = ai_exit.get('reason', 'ai_exit_signal')
                if self.logger:
                    self.logger.debug(f"🤖 AI Exit: {ai_exit.get('reason')} (PnL: {pnl_pct:.2f}%)")
                return result

        # 1. Check SL hit
        if current_sl:
            if self._is_sl_hit(current_price, current_sl, side):
                result['should_exit'] = True
                result['exit_type'] = 'SL'
                result['reason'] = 'stop_loss_hit'
                return result
        
        # 2. Check TP hit
        if current_tp:
            tp_hit = self._is_tp_hit(current_price, current_tp, side)

            # Check if trailing_take_profit is enabled
            if tp_hit and self.exit_manager.exit_strategy.trailing_take_profit:
                # TP hit but don't exit - activate TP trailing instead
                if not position.get('tp_trailing_active'):
                    # First time TP hit - activate TP trailing
                    position['tp_trailing_active'] = True

                    # Calculate initial trailing SL from TP level
                    trailing_distance = self.exit_manager.exit_strategy.trailing_distance
                    if side.upper() == 'LONG':
                        tp_trailing_sl = current_tp * (1 - trailing_distance / 100.0)
                    else:  # SHORT
                        tp_trailing_sl = current_tp * (1 + trailing_distance / 100.0)

                    result['updated_sl'] = tp_trailing_sl
                    result['tp_trailing_activated'] = True

                    if self.logger:
                        self.logger.debug(f"TP trailing activated: SL = {tp_trailing_sl} (from TP {current_tp})")

                # Update TP trailing SL if price continues
                else:
                    # Calculate new trailing SL from current price
                    trailing_distance = self.exit_manager.exit_strategy.trailing_distance
                    if side.upper() == 'LONG':
                        new_tp_trailing_sl = current_price * (1 - trailing_distance / 100.0)
                        # Only update if SL moves up
                        if new_tp_trailing_sl > current_sl:
                            result['updated_sl'] = new_tp_trailing_sl
                            if self.logger:
                                self.logger.debug(f"TP trailing updated: {current_sl} -> {new_tp_trailing_sl}")
                    else:  # SHORT
                        new_tp_trailing_sl = current_price * (1 + trailing_distance / 100.0)
                        # Only update if SL moves down
                        if new_tp_trailing_sl < current_sl:
                            result['updated_sl'] = new_tp_trailing_sl
                            if self.logger:
                                self.logger.debug(f"TP trailing updated: {current_sl} -> {new_tp_trailing_sl}")

            elif tp_hit:
                # Normal TP behavior - exit immediately
                result['should_exit'] = True
                result['exit_type'] = 'TP'
                result['reason'] = 'take_profit_hit'
                return result
        
        # 3. Check trailing stop update
        should_update_sl, new_sl = self.exit_manager.should_update_trailing_stop(
            entry_price, current_price, current_sl, side
        )
        if should_update_sl and new_sl:
            result['updated_sl'] = new_sl
            if self.logger:
                self.logger.debug(f"Trailing SL updated: {current_sl} -> {new_sl}")
        
        # 4. Check break-even move
        should_move_be, be_price = self.exit_manager.should_move_to_breakeven(
            entry_price, current_price, current_sl, side
        )
        if should_move_be and be_price:
            result['updated_sl'] = be_price
            result['break_even_moved'] = True  # Flag for debug logging

        # 5. Check partial exit
        if self.exit_manager.exit_strategy.partial_exit_enabled:
            completed_exits = position.get('completed_partial_exits', 0)
            should_exit_partial, exit_size = self.exit_manager.get_partial_exit_size(
                entry_price, current_price, side, completed_exits
            )

            if should_exit_partial:
                result['should_exit'] = True
                result['exit_type'] = 'PARTIAL'
                result['partial_exit_size'] = exit_size  # 0.25, 0.35, 0.40
                result['partial_exit_level'] = completed_exits + 1  # 1, 2, 3
                result['reason'] = f'partial_exit_level_{completed_exits + 1}'

                if self.logger:
                    level = completed_exits + 1
                    profit_pct = self.exit_manager.exit_strategy.partial_exit_levels[completed_exits]
                    self.logger.debug(f"Partial exit {level}: Close {exit_size*100:.0f}% at +{profit_pct}%")

                return result

        # 6. Check signal-based exit
        signal_exit = self.signal_manager.evaluate_exit(
            symbol, side, data, position
        )
        if signal_exit['should_exit']:
            result['should_exit'] = True
            result['exit_type'] = 'SIGNAL'
            result['reason'] = signal_exit['reason']
            return result

        return result
    
    # ========================================================================
    # ORDER CREATION
    # ========================================================================

    def create_order(
        self,
        symbol: str,
        recommendation: str,
        current_price: float,
        current_balance: float
    ) -> Optional[Any]:
        """
        Create an Order based on strategy rules

        This method is responsible for:
        - Calculating position size based on risk management
        - Determining order side and type
        - Calculating SL/TP prices
        - Creating the Order object

        Args:
            symbol: Trading symbol
            recommendation: 'BUY' or 'SELL'
            current_price: Current market price
            current_balance: Available balance

        Returns:
            Order object or None if order cannot be created
        """
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            # 1. Calculate position size based on strategy risk management
            position_size = self._calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                current_balance=current_balance
            )

            if position_size is None or position_size <= 0:
                if self.logger:
                    self.logger.warning(f"⚠️  Invalid position size for {symbol}: {position_size}")
                return None

            # 2. Determine order side
            if recommendation == 'BUY':
                order_side = OrderSide.BUY
            elif recommendation == 'SELL':
                order_side = OrderSide.SELL
            else:
                if self.logger:
                    self.logger.error(f"❌ Invalid recommendation: {recommendation}")
                return None

            # 3. Calculate SL/TP prices
            stop_loss = None
            take_profit = None

            # Get SL/TP from exit_strategy (not risk_management)
            if hasattr(self.strategy, 'exit_strategy') and self.strategy.exit_strategy:
                if hasattr(self.strategy.exit_strategy, 'stop_loss_percent') and self.strategy.exit_strategy.stop_loss_percent:
                    sl_percent = self.strategy.exit_strategy.stop_loss_percent / 100.0
                    if order_side == OrderSide.BUY:
                        stop_loss = current_price * (1 - sl_percent)
                    else:  # SELL
                        stop_loss = current_price * (1 + sl_percent)

                if hasattr(self.strategy.exit_strategy, 'take_profit_percent') and self.strategy.exit_strategy.take_profit_percent:
                    tp_percent = self.strategy.exit_strategy.take_profit_percent / 100.0
                    if order_side == OrderSide.BUY:
                        take_profit = current_price * (1 + tp_percent)
                    else:  # SELL
                        take_profit = current_price * (1 - tp_percent)

            # 4. Create Order object
            # Note: MARKET orders should not include price field (validation requirement)
            # Paper mode will use current market price automatically
            order = Order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=position_size,
                price=None,  # MARKET orders don't specify price
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if self.logger:
                sl_str = f"{stop_loss:.2f}" if stop_loss else "None"
                tp_str = f"{take_profit:.2f}" if take_profit else "None"
                self.logger.debug(
                    f"Order created: {symbol} {order_side.value} "
                    f"qty={position_size:.6f} @ market "
                    f"(SL={sl_str}, TP={tp_str})"
                )

            return order

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Order creation failed for {symbol}: {e}")
            return None

    def create_exit_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Optional[Any]:
        """
        Create an exit Order

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL' (opposite of position side)
            quantity: Quantity to close

        Returns:
            Order object or None if order cannot be created
        """
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            order_side = OrderSide.BUY if side == 'BUY' else OrderSide.SELL

            order = Order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None  # MARKET order
            )

            if self.logger:
                self.logger.debug(
                    f"Exit order created: {symbol} {order_side.value} "
                    f"qty={quantity:.6f} @ market"
                )

            return order

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Exit order creation failed for {symbol}: {e}")
            return None

    def _calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        current_balance: float
    ) -> Optional[float]:
        """
        Calculate position size based on strategy risk management

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_balance: Available balance

        Returns:
            Position size (quantity) or None if invalid
        """
        try:
            sizing_method = self.strategy.risk_management.sizing_method

            if sizing_method == "FIXED_PERCENT":
                # Position size as percentage of balance
                position_percent = self.strategy.risk_management.position_percent_size / 100.0
                position_value = current_balance * position_percent

                # Calculate quantity from position value
                quantity = position_value / current_price

                if self.logger:
                    self.logger.debug(
                        f"Position sizing: {symbol} "
                        f"balance=${current_balance:.2f} × {position_percent*100:.1f}% "
                        f"= ${position_value:.2f} ÷ ${current_price:.2f} "
                        f"= {quantity:.6f} units"
                    )

                return quantity

            elif sizing_method == "FIXED_AMOUNT":
                # Fixed dollar amount per trade
                fixed_amount = self.strategy.risk_management.get("fixed_amount", 1000.0)
                quantity = fixed_amount / current_price

                if self.logger:
                    self.logger.debug(
                        f"Position sizing: {symbol} "
                        f"fixed=${fixed_amount:.2f} ÷ ${current_price:.2f} "
                        f"= {quantity:.6f} units"
                    )

                return quantity

            elif sizing_method == "RISK_BASED":
                # Size based on risk (account risk % and stop loss distance)
                account_risk_percent = self.strategy.risk_management.get("account_risk_percent", 1.0) / 100.0
                stop_loss_percent = self.strategy.risk_management.stop_loss_percent / 100.0

                # Risk amount = account balance × account risk %
                risk_amount = current_balance * account_risk_percent

                # Position size = risk amount / (price × stop loss %)
                quantity = risk_amount / (current_price * stop_loss_percent)

                if self.logger:
                    self.logger.debug(
                        f"Position sizing: {symbol} "
                        f"risk=${risk_amount:.2f} ÷ (${current_price:.2f} × {stop_loss_percent*100:.1f}%) "
                        f"= {quantity:.6f} units"
                    )

                return quantity

            else:
                if self.logger:
                    self.logger.warning(f"⚠️  Unknown sizing method: {sizing_method}")
                return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Position size calculation failed for {symbol}: {e}")
            return None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _determine_recommendation(self, eval_result: Dict[str, Any]) -> str:
        """
        Evaluation sonucundan recommendation belirle
        
        Args:
            eval_result: evaluate() partial result
        
        Returns:
            'BUY' | 'SELL' | 'HOLD'
        """
        # Market not tradeable
        if not eval_result['market_tradeable']:
            return 'HOLD'
        
        # Cannot open position (portfolio limits)
        if not eval_result['can_open_position']:
            return 'HOLD'
        
        # Check entry signal
        entry_signal = eval_result.get('entry_signal')
        if not entry_signal:
            return 'HOLD'
        
        signal = entry_signal.get('signal')
        
        if signal == 'LONG':
            return 'BUY'
        elif signal == 'SHORT':
            return 'SELL'
        else:
            return 'HOLD'
    
    def _get_primary_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Returns the primary timeframe data"""
        if isinstance(data, pd.DataFrame):
            return data
        
        # Multi-timeframe dict
        primary_tf = self.strategy.primary_timeframe
        return data.get(primary_tf, pd.DataFrame())
    
    def _is_sl_hit(self, current_price: float, sl_price: float, side: str) -> bool:
        """Stop loss hit oldu mu?"""
        if side.upper() == 'LONG':
            return current_price <= sl_price
        else:  # SHORT
            return current_price >= sl_price
    
    def _is_tp_hit(self, current_price: float, tp_price: float, side: str) -> bool:
        """Take profit hit oldu mu?"""
        if side.upper() == 'LONG':
            return current_price >= tp_price
        else:  # SHORT
            return current_price <= tp_price
    
    # ========================================================================
    # POSITION LIFECYCLE HOOKS
    # ========================================================================
    
    def on_position_opened(self, position: Dict[str, Any]) -> None:
        """
        Called when a position is opened.
        
        Args:
            position: Position data
        """
        symbol = position.get('symbol')
        if symbol:
            self.portfolio_manager.register_position(symbol, position)
        
        # Strategy hook
        self.strategy.on_order_filled(position)
    
    def on_position_closed(self, position: Dict[str, Any]) -> None:
        """
        Called when the position is closed.
        
        Args:
            position: Position data
        """
        symbol = position.get('symbol')
        position_id = position.get('id')
        
        if symbol and position_id:
            self.portfolio_manager.unregister_position(symbol, position_id)
        
        # Strategy hook
        self.strategy.on_position_closed(position)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_required_indicators(self) -> List[str]:
        """
        Returns all the indicators required for the strategy.
        
        Returns:
            List[str]: Indicator names
        """
        # Indicators defined in the Strategy.
        strategy_indicators = self.strategy.get_indicator_names()
        
        # Indicators used in signal conditions
        signal_indicators = self.signal_manager.get_required_indicators()
        
        # Combine
        all_indicators = set(strategy_indicators + signal_indicators)
        
        return sorted(all_indicators)
    
    def sync_portfolio_from_position_manager(self) -> None:
        """Synchronize positions from PositionManager"""
        self.portfolio_manager.update_positions_from_position_manager()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns the portfolio summary"""
        return self.portfolio_manager.get_summary()
    
    def __repr__(self) -> str:
        return (
            f"<StrategyExecutor "
            f"strategy='{self.strategy.strategy_name}' "
            f"positions={self.portfolio_manager.get_total_position_count()}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StrategyExecutor',
]

