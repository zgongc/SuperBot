#!/usr/bin/env python3
"""
engines/risk_manager.py
SuperBot - Risk Manager
Author: SuperBot Team
Date: 2025-10-16
Version: 1.0.0

Risk Manager - Risk control and validation

Features:
- Position size validation
- Max open positions limit
- Daily loss limit
- Max drawdown check
- Exposure management
- Leverage limits
- Risk/Reward ratio validation
- Portfolio heat check
- Emergency stop mechanism

Risk Parameters:
    max_position_size: 10%      # Portfolio'nun max %10'u
    max_open_positions: 5       # Maximum 5 positions at the same time
    daily_loss_limit: -5%       # Maximum %5 loss per day
    max_drawdown: -15%          # Max drawdown %15
    max_portfolio_heat: 20%     # Max risk exposure
    min_risk_reward: 1.5        # Min 1:1.5 risk/reward
    max_leverage: 10            # Max leverage

Usage:
    rm = RiskManager(
        config=config,
        logger=logger,
        data_manager=data_manager
    )
    
    await rm.initialize()
    
    # Validate position size
    is_valid, size = rm.calculate_position_size(
        symbol='BTCUSDT',
        entry_price=50000,
        stop_loss=49000,
        portfolio_value=10000
    )
    
    # Check if can open position
    can_open = rm.can_open_position('BTCUSDT')
    
    # Emergency stop check
    if rm.is_emergency_stop():
        print("Trading halted!")
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskManager:
    """
    Risk Manager - Trading risk control and validation
    
    Risk Checks:
    1. Position size (portfolio %)
    2. Max open positions
    3. Daily loss limit
    4. Max drawdown
    5. Portfolio heat (exposure)
    6. Leverage limits
    7. Risk/Reward ratio
    8. Emergency stop
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        logger: Optional[any] = None,
        data_manager: Optional[any] = None
    ):
        """
        Args:
            config: Config engine
            logger: Logger instance
            data_manager: DataManager instance
        """
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        
        # Risk parameters (defaults)
        self.max_position_size_pct = 10.0  # %10 of portfolio
        self.max_open_positions = 5
        self.daily_loss_limit_pct = -5.0  # -5%
        self.max_drawdown_pct = -15.0  # -15%
        self.max_portfolio_heat_pct = 20.0  # %20
        self.min_risk_reward = 1.5  # 1:1.5
        self.max_leverage = 10
        
        # State tracking
        self.emergency_stop = False
        self.daily_start_balance = 0.0
        self.peak_balance = 0.0
        self.last_reset = datetime.now()
        
        # Daily tracking (strategy-driven)
        self.daily_trades_count = 0
        self.daily_trades_limit = 999  # Will be overridden by strategy
        self.last_trade_date = None
        
        # Tracking
        self.rejected_trades = []
        self.risk_violations = []
        
        self.logger.info("🛡️ RiskManager started")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize(self):
        """Initialize Risk Manager"""
        # Load config
        if self.config:
            risk_config = self.config.get("risk", {})
            
            self.max_position_size_pct = risk_config.get("max_position_size", 10.0)
            self.max_open_positions = risk_config.get("max_open_positions", 5)
            self.daily_loss_limit_pct = risk_config.get("daily_loss_limit", -5.0)
            self.max_drawdown_pct = risk_config.get("max_drawdown", -15.0)
            self.max_portfolio_heat_pct = risk_config.get("max_portfolio_heat", 20.0)
            self.min_risk_reward = risk_config.get("min_risk_reward", 1.5)
            self.max_leverage = risk_config.get("max_leverage", 10)
        
        # Get current balance
        if self.data_manager:
            try:
                balance = await self.data_manager.get_latest_balance()
                if balance:
                    self.daily_start_balance = balance.get("total_equity", 0.0)
                    self.peak_balance = self.daily_start_balance
                else:
                    # No balance in DB yet, use default
                    self.daily_start_balance = 10000.0
                    self.peak_balance = 10000.0
            except Exception as e:
                # Default balance on error
                self.daily_start_balance = 10000.0
                self.peak_balance = 10000.0
                if self.logger:
                    self.logger.warning(f"⚠️ Could not get balance, using default: {e}")
        else:
            # Default balance if no data_manager
            self.daily_start_balance = 10000.0
            self.peak_balance = 10000.0
        
        if self.logger:
            self.logger.info(f"✅ RiskManager ready")
            self.logger.info(f"   Max Position Size: {self.max_position_size_pct}%")
            self.logger.info(f"   Max Open Positions: {self.max_open_positions}")
            self.logger.info(f"   Daily Loss Limit: {self.daily_loss_limit_pct}%")
            self.logger.info(f"   Max Drawdown: {self.max_drawdown_pct}%")
            self.logger.info(f"   Start Balance: ${self.daily_start_balance:,.2f}")
    
    # ========================================================================
    # POSITION SIZE CALCULATION
    # ========================================================================
    
    def calculate_position_size_from_strategy(
        self,
        strategy: any,
        risk_management: any,
        entry_price: float,
        portfolio_value: float,
        stop_loss_price: float = None
    ) -> float:
        """
        Position size hesapla (Strategy RiskManagement config'inden)

        BaseStrategy.RiskManagement config'ini kullanarak position size hesaplar.
        Applies ALL risk parameters from the strategy:

        Desteklenen metodlar:
        - FIXED_PERCENT: Portfolio'nun %X'i
        - FIXED_USD: Fixed $X value.
        - FIXED_QUANTITY: Sabit X adet
        - RISK_BASED: Based on stop loss (uses max_risk_per_trade).

        Risk Limitleri (stratejiden):
        - max_risk_per_trade: Maximum risk percentage per trade.
        - max_portfolio_risk: Auto-calculated from strategy.leverage (leverage × 100)

        Args:
            strategy: Strategy instance (for accessing max_portfolio_risk property)
            risk_management: Strategy.risk_management object (RiskManagement dataclass)
            entry_price: Entry price
            portfolio_value: Current portfolio value (balance)
            stop_loss_price: Stop loss price (required for RISK_BASED)

        Returns:
            Position size (quantity)
        
        Example:
            >>> rm = RiskManager()
            >>> size = rm.calculate_position_size_from_strategy(
            ...     risk_management=strategy.risk_management,
            ...     entry_price=100000,
            ...     portfolio_value=10000,
            ...     stop_loss_price=98500
            ... )
        """
        # === OVERRIDE STRATEGY PARAMETERS ===
        # The strategy defines its own risk rules!
        if hasattr(risk_management, 'max_risk_per_trade'):
            strategy_max_risk = risk_management.max_risk_per_trade
        else:
            strategy_max_risk = self.max_position_size_pct  # Fallback
        
        # Get sizing method (can be an enum or a string)
        sizing_method = getattr(risk_management.sizing_method, 'value', risk_management.sizing_method)
        sizing_method = str(sizing_method).lower()
        
        if self.logger:
            self.logger.debug(f"Position sizing: method={sizing_method}, value={risk_management.size_value}, price={entry_price}")
            self.logger.debug(f"  Strategy limits: max_risk_per_trade={strategy_max_risk}%")
        
        # Get leverage from strategy (default 1x if not set)
        leverage = getattr(strategy, 'leverage', 1)

        # FIXED_PERCENT: Portfolio'nun %X'i × leverage
        if sizing_method == 'fixed_percent':
            # Use new parameter (position_percent_size), fallback to old (size_value) for backward compatibility
            percent_value = risk_management.position_percent_size if hasattr(risk_management, 'position_percent_size') else risk_management.size_value

            # Validate percent_value
            if percent_value is None or percent_value <= 0:
                if self.logger:
                    self.logger.warning(f"⚠️ Invalid position_percent_size: {percent_value}, using 2% fallback")
                percent_value = 2.0

            risk_pct = percent_value / 100
            size = (portfolio_value * risk_pct * leverage) / entry_price
            if self.logger:
                self.logger.debug(f"  FIXED_PERCENT: {percent_value}% of ${portfolio_value:,.2f} × {leverage}x leverage = {size:.6f}")

        # FIXED_USD: Position with a fixed $X value x leverage
        elif sizing_method == 'fixed_usd':
            # Use new parameter (position_usd_size), fallback to old (size_value) for backward compatibility
            usd_value = risk_management.position_usd_size if hasattr(risk_management, 'position_usd_size') else risk_management.size_value

            # Validate usd_value
            if usd_value is None or usd_value <= 0:
                if self.logger:
                    self.logger.warning(f"⚠️ Invalid position_usd_size: {usd_value}, using $100 fallback")
                usd_value = 100.0

            size = (usd_value * leverage) / entry_price
            if self.logger:
                self.logger.debug(f"  FIXED_USD: ${usd_value} × {leverage}x leverage / ${entry_price:,.2f} = {size:.6f}")
        
        # FIXED_QUANTITY: Sabit X adet
        elif sizing_method == 'fixed_quantity':
            quantity_value = risk_management.position_quantity_size if hasattr(risk_management, 'position_quantity_size') else risk_management.size_value
            size = quantity_value
            if self.logger:
                self.logger.debug(f"  FIXED_QUANTITY: {size:.6f}")
        
        # RISK_BASED: Based on stop loss (uses max_risk_per_trade from the strategy) x leverage
        elif sizing_method == 'risk_based':
            if stop_loss_price is None:
                if self.logger:
                    self.logger.warning(f"RISK_BASED requires stop_loss_price, using 2% fallback")
                risk_pct = 0.02
                size = (portfolio_value * risk_pct * leverage) / entry_price
            else:
                # Risk per trade (USD)
                risk_amount = portfolio_value * (strategy_max_risk / 100)

                # Price difference (risk per unit)
                price_diff = abs(entry_price - stop_loss_price)

                if price_diff == 0:
                    if self.logger:
                        self.logger.warning(f"Stop loss equal to entry price, using 2% fallback")
                    risk_pct = 0.02
                    size = (portfolio_value * risk_pct * leverage) / entry_price
                else:
                    # Risk per unit percentage
                    risk_per_unit_pct = (price_diff / entry_price) * 100

                    # Position size (quantity) with leverage
                    position_value = risk_amount / (risk_per_unit_pct / 100)
                    size = (position_value * leverage) / entry_price
                    
                    if self.logger:
                        self.logger.debug(f"  RISK_BASED: risk_amount=${risk_amount:.2f}, price_diff=${price_diff:.2f}, size={size:.6f}")
        
        # Unknown method: fallback to 2%
        else:
            if self.logger:
                self.logger.warning(f"Unknown sizing method: '{sizing_method}', using 2% fallback")
            risk_pct = 0.02
            size = (portfolio_value * risk_pct) / entry_price
        
        # === VALIDATE: max_risk_per_trade limit (from the strategy) ===
        # NOTE: max_risk_per_trade is only used for RISK_BASED.
        # The global_max check (leverage x 100) for FIXED_PERCENT and FIXED_USD is sufficient.
        position_value = size * entry_price
        position_pct = (position_value / portfolio_value) * 100

        # REMOVED: max_risk_per_trade check for FIXED_PERCENT/FIXED_USD
        # These methods should only be limited by global_max (leverage × 100)
        
        # === VALIDATE: Global limit from strategy (auto-calculated: leverage × 100) ===
        try:
            global_max = strategy.max_portfolio_risk  # Property: leverage × 100
        except AttributeError:
            global_max = self.max_position_size_pct  # Fallback: 100%

        if position_pct > global_max:
            if self.logger:
                self.logger.debug(
                    f"Position size exceeds global limit ({position_pct:.2f}% > {global_max}%), "
                    f"reducing to global max (leverage={strategy.leverage}x)"
                )
            max_position_value = portfolio_value * (global_max / 100)
            size = max_position_value / entry_price
        
        # Round quantity (crypto: 8 decimals)
        size = round(size, 8)

        # Validate: size must be positive
        if size <= 0:
            if self.logger:
                self.logger.warning(
                    f"⚠️ Calculated size is non-positive ({size}), "
                    f"method={sizing_method}, portfolio=${portfolio_value:.2f}, price=${entry_price:.2f}"
                )
            # Return minimum viable size
            return 0.0

        if self.logger:
            final_value = size * entry_price
            final_pct = (final_value / portfolio_value) * 100
            self.logger.debug(f"  FINAL: size={size:.6f}, value=${final_value:.2f} ({final_pct:.2f}%)")

        return size
    
    def validate_can_open_position(
        self,
        strategy: any,
        risk_management: any,
        portfolio_value: float,
        current_balance: float,
        open_positions: Dict[str, any],
        timestamp: any = None
    ) -> Tuple[bool, str]:
        """
        Check if the position can be opened (COMPREHENSIVE - Apply ALL rules from the strategy)

        Kontroller (stratejiden):
        1. max_portfolio_risk: Total risk of open positions (auto: leverage x 100)
        2. max_drawdown: Drawdown from peak balance
        3. max_daily_trades: Daily trade limit
        4. emergency_stop: Emergency stop
        
        Args:
            risk_management: Strategy.risk_management object
            portfolio_value: Current portfolio value
            current_balance: Current balance
            open_positions: Open positions dictionary (symbol -> List of Positions)
            timestamp: Current timestamp (for daily trades)
        
        Returns:
            (can_open: bool, reason: str)
        
        Example:
            >>> can_open, reason = rm.validate_can_open_position(
            ...     risk_management=strategy.risk_management,
            ...     portfolio_value=10000,
            ...     current_balance=9800,
            ...     open_positions={'BTCUSDT': [pos1, pos2]},
            ...     timestamp=datetime.now()
            ... )
        """
        # === 1. EMERGENCY STOP CHECK ===
        if self.emergency_stop:
            return False, "Emergency stop is active!"
        
        # === 2. MAX PORTFOLIO RISK CHECK (auto-calculated from leverage) ===
        if hasattr(strategy, 'max_portfolio_risk'):
            max_portfolio_risk = strategy.max_portfolio_risk  # leverage × 100

            # Total value of open positions
            total_open_value = 0.0
            for side, positions in open_positions.items():
                if isinstance(positions, list):
                    for pos in positions:
                        if hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                            total_open_value += pos.size * pos.entry_price
                elif hasattr(positions, 'size') and hasattr(positions, 'entry_price'):
                    total_open_value += positions.size * positions.entry_price

            # Portfolio risk %
            portfolio_risk_pct = (total_open_value / portfolio_value) * 100 if portfolio_value > 0 else 0

            if portfolio_risk_pct >= max_portfolio_risk:
                reason = f"Max portfolio risk exceeded ({portfolio_risk_pct:.2f}% >= {max_portfolio_risk}% [leverage={strategy.leverage}x])"
                if self.logger:
                    self.logger.warning(f"❌ {reason}")
                return False, reason
        
        # === 3. MAX DRAWDOWN CHECK (stratejiden) ===
        if hasattr(risk_management, 'max_drawdown'):
            max_drawdown = risk_management.max_drawdown
            
            # Update the peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # Drawdown hesapla
            if self.peak_balance > 0:
                drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance) * 100
                
                if drawdown_pct >= max_drawdown:
                    reason = f"Max drawdown exceeded ({drawdown_pct:.2f}% >= {max_drawdown}%)"
                    if self.logger:
                        self.logger.error(f"🚨 {reason} - EMERGENCY STOP!")
                    
                    # Emergency stop aktive et
                    self.emergency_stop = True
                    return False, reason
        
        # === 4. MAX DAILY TRADES CHECK (stratejiden) ===
        if hasattr(risk_management, 'max_daily_trades'):
            max_daily_trades = risk_management.max_daily_trades
            
            # Check the daily trade counter
            if timestamp:
                can_trade, reason = self._check_daily_trades_limit(max_daily_trades, timestamp)
                if not can_trade:
                    return False, reason
        
        # === ALL CHECKS PASSED ===
        return True, "OK"
    
    def _check_daily_trades_limit(self, max_daily_trades: int, timestamp: any) -> Tuple[bool, str]:
        """
        Check the daily trade limit.
        
        Args:
            max_daily_trades: Maximum number of daily trades
            timestamp: Current timestamp
        
        Returns:
            (can_trade: bool, reason: str)
        """
        # Convert timestamp to date
        if isinstance(timestamp, (int, float)):
            import pandas as pd
            current_date = pd.Timestamp(timestamp, unit='ms').date()
        else:
            current_date = timestamp.date() if hasattr(timestamp, 'date') else datetime.now().date()
        
        # Is it a new day?
        if self.last_trade_date is None or current_date != self.last_trade_date:
            # New day - reset
            self.daily_trades_count = 0
            self.last_trade_date = current_date
        
        # Limit check
        if self.daily_trades_count >= max_daily_trades:
            reason = f"Daily trade limit exceeded ({self.daily_trades_count}/{max_daily_trades})"
            if self.logger:
                self.logger.warning(f"❌ {reason}")
            return False, reason
        
        return True, "OK"
    
    def increment_daily_trades(self):
        """
        Increment the daily trade counter (called when a trade is opened).
        """
        self.daily_trades_count += 1
        if self.logger:
            self.logger.debug(f"Daily trades: {self.daily_trades_count}/{self.daily_trades_limit}")
    
    def reset_emergency_stop(self):
        """
        Reset the emergency stop (use with caution!)
        """
        self.emergency_stop = False
        if self.logger:
            self.logger.info("Emergency stop resetlendi")
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        risk_per_trade_pct: float = 1.0,
        leverage: int = 1
    ) -> Tuple[bool, float, str]:
        """
        Position size hesapla (Risk-based - LEGACY)
        
        DEPRECATED: Use calculate_position_size_from_strategy() instead
        
        Args:
            symbol: Symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            portfolio_value: Portfolio value
            risk_per_trade_pct: Risk per trade (%)
            leverage: Leverage
        
        Returns:
            (is_valid, quantity, reason)
        """
        try:
            # Risk per trade (USD)
            risk_amount = portfolio_value * (risk_per_trade_pct / 100)
            
            # Price difference (risk per unit)
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff == 0:
                return False, 0.0, "Stop loss price cannot be equal to the entry price"
            
            # Risk per unit percentage
            risk_per_unit_pct = (price_diff / entry_price) * 100
            
            # Position size (quantity)
            position_value = risk_amount / (risk_per_unit_pct / 100)
            quantity = position_value / entry_price
            
            # Adjust for leverage
            position_value_with_leverage = position_value / leverage
            
            # Check max position size
            position_pct = (position_value_with_leverage / portfolio_value) * 100
            
            if position_pct > self.max_position_size_pct:
                # Reduce to max allowed
                max_position_value = portfolio_value * (self.max_position_size_pct / 100)
                quantity = (max_position_value * leverage) / entry_price
                
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Position size reduced: {position_pct:.2f}% → "
                        f"{self.max_position_size_pct}%"
                    )
            
            # Round quantity (crypto: 8 decimals)
            quantity = round(quantity, 8)
            
            if quantity <= 0:
                return False, 0.0, "The calculated amount is zero or negative"
            
            if self.logger:
                self.logger.info(
                    f"📊 Position Size: {symbol} - Quantity: {quantity}, "
                    f"Risk: ${risk_amount:.2f} ({risk_per_trade_pct}%)"
                )
            
            return True, quantity, "OK"
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error calculating position size: {e}")
            return False, 0.0, str(e)
    
    # ========================================================================
    # VALIDATION CHECKS
    # ========================================================================
    
    async def can_open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1
    ) -> Tuple[bool, str]:
        """
        Checks if the position can be opened.
        
        Returns:
            (can_open, reason)
        """
        try:
            # 1. Emergency stop check
            if self.emergency_stop:
                return False, "Emergency stop is active"
            
            # 2. Max open positions check
            if not await self._check_max_positions():
                return False, f"Max open position limit ({self.max_open_positions})"
            
            # 3. Daily loss limit check
            if not await self._check_daily_loss():
                return False, f"Daily loss limit exceeded ({self.daily_loss_limit_pct}%)"
            
            # 4. Max drawdown check
            if not await self._check_max_drawdown():
                return False, f"Maximum drawdown limit exceeded ({self.max_drawdown_pct}%)"
            
            # 5. Portfolio heat check
            position_value = quantity * entry_price / leverage
            if not await self._check_portfolio_heat(position_value):
                return False, f"Portfolio heat limit exceeded ({self.max_portfolio_heat_pct}%)"
            
            # 6. Leverage check
            if leverage > self.max_leverage:
                return False, f"Maximum leverage limit ({self.max_leverage}x)"
            
            # 7. Risk/Reward check
            if stop_loss and take_profit:
                if not self._check_risk_reward(entry_price, stop_loss, take_profit):
                    return False, f"Risk/Reward ratio is insufficient (min {self.min_risk_reward})"
            
            return True, "OK"
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Position validation error: {e}")
            return False, str(e)
    
    async def _check_max_positions(self) -> bool:
        """Maximum open position check"""
        if not self.data_manager:
            return True
        
        open_positions = await self.data_manager.get_open_positions()
        current_count = len(open_positions)
        
        if current_count >= self.max_open_positions:
            if self.logger:
                self.logger.warning(
                    f"⚠️ Maximum position limit: {current_count}/{self.max_open_positions}"
                )
            return False
        
        return True
    
    async def _check_daily_loss(self) -> bool:
        """Daily loss limit check"""
        if not self.data_manager:
            return True

        # Reset if new day
        await self._reset_daily_if_needed()

        # Get current balance
        try:
            balance = await self.data_manager.get_latest_balance()
            if not balance:
                return True
            current_equity = balance.get("total_equity", 0.0)
        except Exception:
            return True
        
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_equity
            return True
        
        # Calculate daily PnL %
        daily_pnl_pct = ((current_equity - self.daily_start_balance) / 
                         self.daily_start_balance * 100)
        
        if daily_pnl_pct <= self.daily_loss_limit_pct:
            if self.logger:
                self.logger.error(
                    f"🚨 Daily loss limit exceeded: {daily_pnl_pct:.2f}% "
                    f"(limit: {self.daily_loss_limit_pct}%)"
                )
            
            # Trigger emergency stop
            self.emergency_stop = True
            self._log_risk_violation("DAILY_LOSS_LIMIT", daily_pnl_pct)
            
            return False
        
        return True
    
    async def _check_max_drawdown(self) -> bool:
        """Maximum drawdown check"""
        if not self.data_manager:
            return True

        # Get current balance
        try:
            balance = await self.data_manager.get_latest_balance()
            if not balance:
                return True
        except Exception:
            return True
        
        current_equity = balance.get("total_equity", 0.0)
        
        # Update peak
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        
        if self.peak_balance == 0:
            return True
        
        # Calculate drawdown %
        drawdown_pct = ((current_equity - self.peak_balance) / 
                        self.peak_balance * 100)
        
        if drawdown_pct <= self.max_drawdown_pct:
            if self.logger:
                self.logger.error(
                    f"🚨 Maximum drawdown limit exceeded: {drawdown_pct:.2f}% "
                    f"(limit: {self.max_drawdown_pct}%)"
                )
            
            # Trigger emergency stop
            self.emergency_stop = True
            self._log_risk_violation("MAX_DRAWDOWN", drawdown_pct)
            
            return False
        
        return True
    
    async def _check_portfolio_heat(self, new_position_value: float) -> bool:
        """Portfolio heat (exposure) control"""
        if not self.data_manager:
            return True
        
        # Get open positions
        open_positions = await self.data_manager.get_open_positions()
        
        # Calculate current exposure
        total_exposure = sum(
            pos['quantity'] * pos['entry_price'] / pos.get('leverage', 1)
            for pos in open_positions
        )
        
        # Add new position
        total_exposure += new_position_value
        
        # Get portfolio value
        balance = await self.data_manager.get_latest_balance()
        if not balance:
            return True
        
        portfolio_value = balance.get("total_equity", 1.0)
        
        # Calculate heat %
        heat_pct = (total_exposure / portfolio_value) * 100
        
        if heat_pct > self.max_portfolio_heat_pct:
            if self.logger:
                self.logger.warning(
                    f"⚠️ Portfolio heat limit exceeded: {heat_pct:.2f}% "
                    f"(limit: {self.max_portfolio_heat_pct}%)"
                )
            return False
        
        return True
    
    def _check_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """Risk/Reward ratio check"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return False
        
        rr_ratio = reward / risk
        
        if rr_ratio < self.min_risk_reward:
            if self.logger:
                self.logger.warning(
                    f"⚠️ Risk/Reward yetersiz: {rr_ratio:.2f} "
                    f"(min: {self.min_risk_reward})"
                )
            return False
        
        return True
    
    # ========================================================================
    # EMERGENCY STOP
    # ========================================================================
    
    def is_emergency_stop(self) -> bool:
        """Emergency stop control"""
        return self.emergency_stop
    
    def trigger_emergency_stop(self, reason: str = "Manual"):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self._log_risk_violation("EMERGENCY_STOP", 0, reason)
        
        if self.logger:
            self.logger.error(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        
        if self.logger:
            self.logger.info("✅ Emergency stop reset")
    
    # ========================================================================
    # RISK METRICS
    # ========================================================================
    
    async def get_risk_metrics(self) -> Dict:
        """Risk metriklerini getir"""
        metrics = {
            "emergency_stop": self.emergency_stop,
            "max_position_size_pct": self.max_position_size_pct,
            "max_open_positions": self.max_open_positions,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_portfolio_heat_pct": self.max_portfolio_heat_pct,
            "min_risk_reward": self.min_risk_reward,
            "max_leverage": self.max_leverage
        }
        
        if self.data_manager:
            # Current positions
            open_positions = await self.data_manager.get_open_positions()
            metrics["current_positions"] = len(open_positions)
            
            # Current balance
            balance = await self.data_manager.get_latest_balance()
            if balance:
                current_equity = balance.get("total_equity", 0.0)
                metrics["current_equity"] = current_equity
                
                # Daily PnL
                if self.daily_start_balance > 0:
                    daily_pnl_pct = ((current_equity - self.daily_start_balance) / 
                                     self.daily_start_balance * 100)
                    metrics["daily_pnl_pct"] = daily_pnl_pct
                
                # Drawdown
                if self.peak_balance > 0:
                    drawdown_pct = ((current_equity - self.peak_balance) / 
                                    self.peak_balance * 100)
                    metrics["current_drawdown_pct"] = drawdown_pct
                
                # Portfolio heat
                total_exposure = sum(
                    pos['quantity'] * pos['entry_price'] / pos.get('leverage', 1)
                    for pos in open_positions
                )
                heat_pct = (total_exposure / current_equity) * 100 if current_equity > 0 else 0
                metrics["portfolio_heat_pct"] = heat_pct
        
        return metrics
    
    def get_risk_level(self) -> RiskLevel:
        """Gets the current risk level"""
        if self.emergency_stop:
            return RiskLevel.CRITICAL
        
        # Simple risk level calculation
        # TODO: Implement more advanced calculations
        return RiskLevel.LOW
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    async def _reset_daily_if_needed(self):
        """New day check and reset"""
        now = datetime.now()
        
        # Check if new day
        if now.date() > self.last_reset.date():
            if self.logger:
                self.logger.info("🔄 New day - Risk metrics reset")
            
            # Get current balance as new start
            if self.data_manager:
                balance = await self.data_manager.get_latest_balance()
                if balance:
                    self.daily_start_balance = balance.get("total_equity", 0.0)
            
            self.last_reset = now
            
            # Reset emergency stop if daily loss
            if self.emergency_stop:
                self.emergency_stop = False
                if self.logger:
                    self.logger.info("✅ Emergency stop reset (new day)")
    
    def _log_risk_violation(self, violation_type: str, value: float, reason: str = ""):
        """Risk ihlalini kaydet"""
        violation = {
            "timestamp": datetime.now(),
            "type": violation_type,
            "value": value,
            "reason": reason
        }
        
        self.risk_violations.append(violation)
        
        # Keep last 100
        if len(self.risk_violations) > 100:
            self.risk_violations = self.risk_violations[-100:]


if __name__ == "__main__":
    import asyncio
    
    async def test():
        """Test Risk Manager"""
        print("=" * 60)
        print("RiskManager v1.0 - Test")
        print("=" * 60)
        
        # Mock config
        config = {
            "risk": {
                "max_position_size": 10.0,
                "max_open_positions": 5,
                "daily_loss_limit": -5.0,
                "max_drawdown": -15.0,
                "max_portfolio_heat": 20.0,
                "min_risk_reward": 1.5,
                "max_leverage": 10
            }
        }
        
        # Create manager
        rm = RiskManager(config=config)
        await rm.initialize()
        
        print("\n1. Calculate Position Size:")
        is_valid, quantity, reason = rm.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=50000,
            stop_loss=49000,
            portfolio_value=10000,
            risk_per_trade_pct=1.0
        )
        print(f"   Valid: {is_valid}")
        print(f"   Quantity: {quantity}")
        print(f"   Reason: {reason}")
        
        print("\n2. Can Open Position:")
        can_open, reason = await rm.can_open_position(
            symbol="BTCUSDT",
            quantity=0.02,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )
        print(f"   Can Open: {can_open}")
        print(f"   Reason: {reason}")
        
        print("\n3. Risk Metrics:")
        metrics = await rm.get_risk_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        print("\n4. Emergency Stop Test:")
        rm.trigger_emergency_stop("Test")
        print(f"   Emergency Stop: {rm.is_emergency_stop()}")
        
        rm.reset_emergency_stop()
        print(f"   After Reset: {rm.is_emergency_stop()}")
        
        print("\n" + "=" * 60)
        print("✅ Test completed!")
        print("=" * 60)
    
    asyncio.run(test())
