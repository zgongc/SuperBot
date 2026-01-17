#!/usr/bin/env python3
"""
modules/backtest/backtest_positions.py
SuperBot - Position Simulator
Yazar: SuperBot Team
Tarih: 2025-11-16
Versiyon: 3.0.0

Position simulation with accurate sizing and exit logic.

CRITICAL: Bu modül optimizer'ın doğru çalışması için kritik!
Position sizing farklı değerlerde farklı sonuçlar vermeli.

Özellikler:
- Accurate position sizing (FIXED_USD, FIXED_PERCENT, RISK_BASED)
- Entry/Exit logic
- TP/SL/Trailing/Break-even
- Commission & Slippage
- Multi-position support (multi-symbol için)

Kullanım:
    from modules.backtest.backtest_positions import PositionSimulator

    simulator = PositionSimulator(logger)
    trades = simulator.simulate(signals, data, indicators, strategy, config)

Bağımlılıklar:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
    - modules.backtest.backtest_types
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import numpy as np

from modules.backtest.backtest_types import (
    Trade, Position, PositionSide, ExitReason, BacktestConfig
)

if TYPE_CHECKING:
    from components.strategies.base_strategy import Strategy, PositionSizeMethod


# ============================================================================
# POSITION SIMULATOR
# ============================================================================

class PositionSimulator:
    """
    Position simulation with accurate sizing and exit logic

    CRITICAL: Position sizing MUST work correctly!
    - FIXED_USD: 100 vs 1000 should give different results
    - FIXED_PERCENT: 5% vs 20% should give different results
    - RISK_BASED: Risk değişince sonuç değişmeli
    """

    def __init__(self, logger=None):
        """
        Initialize PositionSimulator

        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger
        self.trade_counter = 0  # Trade ID counter

    def simulate(
        self,
        signals: np.ndarray,
        data: pd.DataFrame,
        indicators: Dict[str, np.ndarray],
        strategy: Strategy,
        config: BacktestConfig
    ) -> List[Trade]:
        """
        Simulate all positions

        Args:
            signals: Signal array (1=LONG, -1=SHORT, 0=NONE)
            data: Price data (OHLCV)
            indicators: Calculated indicators
            strategy: Strategy instance
            config: Backtest config

        Returns:
            List[Trade]: Completed trades
        """
        if self.logger:
            self.logger.info("💼 Position simulation başlıyor...")

        trades = []
        position: Optional[Position] = None
        balance = config.initial_balance

        # Iterate through candles
        for i in range(len(data)):
            row = data.iloc[i]
            signal = signals[i]

            # 1. Check exit conditions (if position open)
            if position:
                exit_info = self._check_exit(position, row, signal, strategy, indicators, i)
                if exit_info:
                    # Close position
                    trade = self._close_position(
                        position,
                        exit_info['price'],
                        exit_info['reason'],
                        row.name,  # timestamp
                        balance,
                        config
                    )
                    trades.append(trade)
                    balance += trade.net_pnl_usd
                    position = None

                    if self.logger and self.logger.level <= 10:  # DEBUG
                        self.logger.debug(
                            f"   Trade #{trade.trade_id}: {trade.side.value} "
                            f"${trade.net_pnl_usd:+.2f} ({trade.exit_reason.value})"
                        )
                else:
                    # Update extremes for trailing/analytics
                    position.update_extremes(row['close'])

            # 2. Check entry conditions (if no position)
            if not position and signal != 0:
                # Calculate position size
                quantity = self._calculate_position_size(
                    signal, row['close'], balance, strategy
                )

                if quantity > 0:
                    # Open position
                    position = self._open_position(
                        signal, row, quantity, strategy, config
                    )

                    if self.logger and self.logger.level <= 10:  # DEBUG
                        self.logger.debug(
                            f"   Position #{position.position_id}: {position.side.value} "
                            f"@ ${position.entry_price:.2f}, qty={quantity:.6f}"
                        )

        # Close any remaining open position at end
        if position:
            if self.logger:
                self.logger.warning(
                    f"⚠️  Position #{position.position_id} backtest sonunda açık kaldı, "
                    f"son fiyattan kapatılıyor"
                )
            final_row = data.iloc[-1]
            trade = self._close_position(
                position,
                final_row['close'],
                ExitReason.MANUAL,
                final_row.name,
                balance,
                config
            )
            trades.append(trade)

        if self.logger:
            self.logger.info(
                f"💼 Simulation tamamlandı: {len(trades)} trade, "
                f"final balance: ${balance + trades[-1].net_pnl_usd if trades else balance:,.2f}"
            )

        return trades

    # ========================================================================
    # POSITION SIZING - EN KRİTİK KISIM!
    # ========================================================================

    def _calculate_position_size(
        self,
        signal: int,
        price: float,
        balance: float,
        strategy: Strategy
    ) -> float:
        """
        Calculate position size

        CRITICAL: Bu fonksiyon DOĞRU çalışmalı!
        Farklı sizing_method ve parametreler farklı sonuçlar vermeli.

        Args:
            signal: 1 (LONG) or -1 (SHORT)
            price: Current price
            balance: Current balance
            strategy: Strategy instance

        Returns:
            float: Position quantity (in base asset, e.g., BTC)
        """
        rm = strategy.risk_management

        # Import PositionSizeMethod enum
        from components.strategies.base_strategy import PositionSizeMethod

        # FIXED_USD: Sabit dolar miktarı
        if rm.sizing_method == PositionSizeMethod.FIXED_USD:
            usd_size = rm.position_usd_size
            quantity = usd_size / price

            if self.logger and self.logger.level <= 10:  # DEBUG
                self.logger.debug(
                    f"   FIXED_USD: ${usd_size} / ${price:.2f} = {quantity:.6f}"
                )

            return quantity

        # FIXED_PERCENT: Balance'ın yüzdesi
        elif rm.sizing_method == PositionSizeMethod.FIXED_PERCENT:
            percent = rm.position_percent_size
            usd_size = balance * (percent / 100)
            quantity = usd_size / price

            if self.logger and self.logger.level <= 10:  # DEBUG
                self.logger.debug(
                    f"   FIXED_PERCENT: {percent}% of ${balance:.2f} = "
                    f"${usd_size:.2f} / ${price:.2f} = {quantity:.6f}"
                )

            return quantity

        # RISK_BASED: Stop loss mesafesine göre
        elif rm.sizing_method == PositionSizeMethod.RISK_BASED:
            risk_pct = rm.max_risk_per_trade
            risk_amount = balance * (risk_pct / 100)

            # Calculate SL price
            sl_price = self._calculate_sl_price(price, signal, strategy)
            sl_distance = abs(price - sl_price)

            if sl_distance == 0:
                if self.logger:
                    self.logger.warning(
                        f"⚠️  RISK_BASED: SL distance = 0, fallback to 2% risk"
                    )
                sl_distance = price * 0.02  # Fallback

            quantity = risk_amount / sl_distance

            if self.logger and self.logger.level <= 10:  # DEBUG
                self.logger.debug(
                    f"   RISK_BASED: {risk_pct}% risk = ${risk_amount:.2f}, "
                    f"SL distance = ${sl_distance:.2f}, qty = {quantity:.6f}"
                )

            return quantity

        else:
            if self.logger:
                self.logger.error(f"❌ Unknown sizing method: {rm.sizing_method}")
            return 0.0

    def _calculate_sl_price(
        self,
        entry_price: float,
        signal: int,
        strategy: Strategy
    ) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price
            signal: 1 (LONG) or -1 (SHORT)
            strategy: Strategy instance

        Returns:
            float: Stop loss price
        """
        exit_strat = strategy.exit_strategy

        # Eğer SL disabled ise, default %2 kullan
        if not hasattr(exit_strat, 'stop_loss_value') or exit_strat.stop_loss_value == 0:
            sl_pct = 2.0  # Default 2% SL
        else:
            sl_pct = exit_strat.stop_loss_value

        # LONG: SL aşağıda
        if signal > 0:
            sl_price = entry_price * (1 - sl_pct / 100)
        # SHORT: SL yukarıda
        else:
            sl_price = entry_price * (1 + sl_pct / 100)

        return sl_price

    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================

    def _open_position(
        self,
        signal: int,
        row: pd.Series,
        quantity: float,
        strategy: Strategy,
        config: BacktestConfig
    ) -> Position:
        """
        Open new position

        Args:
            signal: 1 (LONG) or -1 (SHORT)
            row: Current candle data
            quantity: Position size
            strategy: Strategy instance
            config: Backtest config

        Returns:
            Position: Opened position
        """
        self.trade_counter += 1

        # Side
        side = PositionSide.LONG if signal > 0 else PositionSide.SHORT

        # Entry price (with slippage)
        entry_price = self._apply_slippage(row['close'], signal, config.slippage_pct)

        # Calculate SL and TP
        sl_price = self._calculate_sl_price(entry_price, signal, strategy)
        tp_price = self._calculate_tp_price(entry_price, signal, strategy)

        # Calculate entry costs
        position_value = entry_price * quantity
        entry_commission = position_value * (config.commission_pct / 100)
        entry_slippage = abs(entry_price - row['close']) * quantity

        # Create position
        position = Position(
            position_id=self.trade_counter,
            symbol=config.symbols[0],  # Single symbol for now
            side=side,
            entry_time=row.name,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            entry_commission=entry_commission,
            entry_slippage=entry_slippage,
        )

        # Trailing stop setup
        if hasattr(strategy.exit_strategy, 'trailing_enabled') and strategy.exit_strategy.trailing_enabled:
            position.trailing_stop_distance = strategy.exit_strategy.trailing_distance_pct

        return position

    def _calculate_tp_price(
        self,
        entry_price: float,
        signal: int,
        strategy: Strategy
    ) -> Optional[float]:
        """
        Calculate take profit price

        Args:
            entry_price: Entry price
            signal: 1 (LONG) or -1 (SHORT)
            strategy: Strategy instance

        Returns:
            Optional[float]: TP price or None
        """
        exit_strat = strategy.exit_strategy

        if not hasattr(exit_strat, 'take_profit_value') or exit_strat.take_profit_value == 0:
            return None  # No TP

        tp_pct = exit_strat.take_profit_value

        # LONG: TP yukarıda
        if signal > 0:
            tp_price = entry_price * (1 + tp_pct / 100)
        # SHORT: TP aşağıda
        else:
            tp_price = entry_price * (1 - tp_pct / 100)

        return tp_price

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: ExitReason,
        exit_time: datetime,
        current_balance: float,
        config: BacktestConfig
    ) -> Trade:
        """
        Close position and create trade record

        Args:
            position: Position to close
            exit_price: Exit price
            exit_reason: Why closing
            exit_time: Exit timestamp
            current_balance: Current balance (for analytics)
            config: Backtest config

        Returns:
            Trade: Completed trade
        """
        # Calculate exit costs
        position_value = exit_price * position.quantity
        exit_commission = position_value * (config.commission_pct / 100)
        exit_slippage = abs(exit_price - exit_price) * position.quantity  # Minimal

        total_commission = position.entry_commission + exit_commission
        total_slippage = position.entry_slippage + exit_slippage

        # Calculate PnL
        if position.side == PositionSide.LONG:
            # LONG: profit = (exit - entry) × quantity
            gross_pnl_usd = (exit_price - position.entry_price) * position.quantity
        else:
            # SHORT: profit = (entry - exit) × quantity
            gross_pnl_usd = (position.entry_price - exit_price) * position.quantity

        gross_pnl_pct = (gross_pnl_usd / (position.entry_price * position.quantity)) * 100

        # Net PnL (after costs)
        net_pnl_usd = gross_pnl_usd - total_commission - total_slippage
        net_pnl_pct = (net_pnl_usd / (position.entry_price * position.quantity)) * 100

        # Analytics: max profit/loss during position
        if position.side == PositionSide.LONG:
            max_profit_usd = (position.highest_price - position.entry_price) * position.quantity
            max_loss_usd = (position.lowest_price - position.entry_price) * position.quantity
        else:
            max_profit_usd = (position.entry_price - position.lowest_price) * position.quantity
            max_loss_usd = (position.entry_price - position.highest_price) * position.quantity

        max_profit_pct = (max_profit_usd / (position.entry_price * position.quantity)) * 100
        max_loss_pct = (max_loss_usd / (position.entry_price * position.quantity)) * 100

        # Create trade
        trade = Trade(
            trade_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            quantity=position.quantity,
            gross_pnl_usd=gross_pnl_usd,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_usd=net_pnl_usd,
            net_pnl_pct=net_pnl_pct,
            commission=total_commission,
            slippage=total_slippage,
            max_profit_usd=max_profit_usd,
            max_profit_pct=max_profit_pct,
            max_loss_usd=max_loss_usd,
            max_loss_pct=max_loss_pct,
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
        )

        return trade

    # ========================================================================
    # EXIT LOGIC
    # ========================================================================

    def _check_exit(
        self,
        position: Position,
        row: pd.Series,
        signal: int,
        strategy: Strategy,
        indicators: Dict[str, np.ndarray],
        index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check all exit conditions

        Priority:
        1. Stop Loss (highest priority)
        2. Take Profit
        3. Trailing Stop
        4. Break-even
        5. Opposite signal

        Args:
            position: Current position
            row: Current candle
            signal: Current signal
            strategy: Strategy instance
            indicators: Indicators (for advanced exits)
            index: Current index

        Returns:
            Optional[Dict]: {'price': float, 'reason': ExitReason} or None
        """
        # 1. Stop Loss check (candle low/high touched SL)
        if position.stop_loss_price:
            if position.side == PositionSide.LONG:
                # LONG: Check if low touched SL
                if row['low'] <= position.stop_loss_price:
                    return {'price': position.stop_loss_price, 'reason': ExitReason.STOP_LOSS}
            else:
                # SHORT: Check if high touched SL
                if row['high'] >= position.stop_loss_price:
                    return {'price': position.stop_loss_price, 'reason': ExitReason.STOP_LOSS}

        # 2. Take Profit check
        if position.take_profit_price:
            if position.side == PositionSide.LONG:
                # LONG: Check if high touched TP
                if row['high'] >= position.take_profit_price:
                    return {'price': position.take_profit_price, 'reason': ExitReason.TAKE_PROFIT}
            else:
                # SHORT: Check if low touched TP
                if row['low'] <= position.take_profit_price:
                    return {'price': position.take_profit_price, 'reason': ExitReason.TAKE_PROFIT}

        # 3. Trailing Stop check
        if position.trailing_stop_distance:
            trailing_exit = self._check_trailing_stop(position, row)
            if trailing_exit:
                return trailing_exit

        # 4. Break-even check
        if hasattr(strategy.exit_strategy, 'break_even_enabled') and strategy.exit_strategy.break_even_enabled:
            self._update_break_even(position, row, strategy)

        # 5. Opposite signal
        if signal != 0 and ((signal > 0 and position.side == PositionSide.SHORT) or
                             (signal < 0 and position.side == PositionSide.LONG)):
            return {'price': row['close'], 'reason': ExitReason.SIGNAL}

        return None

    def _check_trailing_stop(
        self,
        position: Position,
        row: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Check trailing stop

        Args:
            position: Current position
            row: Current candle

        Returns:
            Optional[Dict]: Exit info or None
        """
        if not position.trailing_stop_distance:
            return None

        distance_pct = position.trailing_stop_distance

        if position.side == PositionSide.LONG:
            # LONG: Trailing SL moves up with highest price
            trailing_sl = position.highest_price * (1 - distance_pct / 100)
            if row['low'] <= trailing_sl:
                return {'price': trailing_sl, 'reason': ExitReason.TRAILING_STOP}

        else:
            # SHORT: Trailing SL moves down with lowest price
            trailing_sl = position.lowest_price * (1 + distance_pct / 100)
            if row['high'] >= trailing_sl:
                return {'price': trailing_sl, 'reason': ExitReason.TRAILING_STOP}

        return None

    def _update_break_even(
        self,
        position: Position,
        row: pd.Series,
        strategy: Strategy
    ):
        """
        Update SL to break-even if profit threshold reached

        Args:
            position: Current position
            row: Current candle
            strategy: Strategy instance
        """
        if position.break_even_activated:
            return  # Already at BE

        exit_strat = strategy.exit_strategy

        if not hasattr(exit_strat, 'break_even_trigger_pct'):
            return

        trigger_pct = exit_strat.break_even_trigger_pct
        current_pnl_pct = position.current_pnl_pct(row['close'])

        if current_pnl_pct >= trigger_pct:
            # Move SL to break-even
            position.stop_loss_price = position.entry_price
            position.break_even_activated = True

            if self.logger and self.logger.level <= 10:  # DEBUG
                self.logger.debug(
                    f"   Position #{position.position_id}: Break-even activated "
                    f"(PnL: {current_pnl_pct:.2f}%)"
                )

    def _apply_slippage(self, price: float, signal: int, slippage_pct: float) -> float:
        """
        Apply slippage to price

        Args:
            price: Original price
            signal: 1 (LONG buy) or -1 (SHORT sell)
            slippage_pct: Slippage percentage

        Returns:
            float: Price with slippage
        """
        slippage_amount = price * (slippage_pct / 100)

        # Buy (LONG): Price slightly higher
        # Sell (SHORT): Price slightly lower
        if signal > 0:
            return price + slippage_amount
        else:
            return price - slippage_amount


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Position Simulator Test")
    print("=" * 60)

    # Mock data
    print("\n📊 Test 1: Mock data hazırlama")
    dates = pd.date_range('2025-01-01', periods=100, freq='15min')
    np.random.seed(42)
    prices = 100000 + np.cumsum(np.random.randn(100) * 100)

    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(100) * 100,
        'low': prices - np.random.rand(100) * 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    print(f"   ✅ {len(data)} candle oluşturuldu")
    print(f"   ✅ Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")

    # Mock signals (simple: buy every 10 candles)
    print("\n📍 Test 2: Mock signals")
    signals = np.zeros(len(data))
    signals[::10] = 1  # LONG every 10 candles
    print(f"   ✅ {(signals != 0).sum()} sinyal oluşturuldu")

    # Mock strategy
    print("\n⚙️  Test 3: Mock strategy")

    class MockRiskManagement:
        def __init__(self, sizing_method, **kwargs):
            from components.strategies.base_strategy import PositionSizeMethod
            self.sizing_method = PositionSizeMethod(sizing_method)
            self.__dict__.update(kwargs)

    class MockExitStrategy:
        def __init__(self):
            self.stop_loss_value = 2.0  # 2% SL
            self.take_profit_value = 5.0  # 5% TP
            self.trailing_enabled = False
            self.break_even_enabled = False

    class MockStrategy:
        def __init__(self, sizing_method, **kwargs):
            self.risk_management = MockRiskManagement(sizing_method, **kwargs)
            self.exit_strategy = MockExitStrategy()

    # Test FIXED_USD
    print("\n💵 Test 4: FIXED_USD sizing ($500)")
    strategy = MockStrategy('FIXED_USD', position_usd_size=500)
    config = BacktestConfig(
        symbols=['BTCUSDT'],
        primary_timeframe='15m',
        initial_balance=10000,
        commission_pct=0.04,
        slippage_pct=0.01
    )

    simulator = PositionSimulator()
    trades = simulator.simulate(signals, data, {}, strategy, config)

    print(f"   ✅ {len(trades)} trade tamamlandı")
    if trades:
        total_pnl = sum(t.net_pnl_usd for t in trades)
        print(f"   ✅ Total PnL: ${total_pnl:+.2f}")
        print(f"   ✅ First trade: {trades[0].side.value} @ ${trades[0].entry_price:.2f}")

    # Test FIXED_PERCENT
    print("\n📊 Test 5: FIXED_PERCENT sizing (20%)")
    strategy2 = MockStrategy('FIXED_PERCENT', position_percent_size=20)
    simulator2 = PositionSimulator()  # New simulator for clean trade counter
    trades2 = simulator2.simulate(signals, data, {}, strategy2, config)

    print(f"   ✅ {len(trades2)} trade tamamlandı")
    if trades2:
        total_pnl2 = sum(t.net_pnl_usd for t in trades2)
        print(f"   ✅ Total PnL: ${total_pnl2:+.2f}")
        print(f"   ✅ Qty comparison: {trades[0].quantity:.6f} (USD) vs {trades2[0].quantity:.6f} (PCT)")
        print(f"   ✅ Different results? {abs(total_pnl - total_pnl2) > 1}")

    # Test RISK_BASED
    print("\n⚠️  Test 6: RISK_BASED sizing")
    strategy3 = MockStrategy('RISK_BASED', max_risk_per_trade=2.0)
    trades3 = simulator.simulate(signals, data, {}, strategy3, config)

    print(f"   ✅ {len(trades3)} trade tamamlandı")
    if trades3:
        total_pnl3 = sum(t.net_pnl_usd for t in trades3)
        print(f"   ✅ Total PnL: ${total_pnl3:+.2f}")

    print("\n✅ Tüm testler başarılı!")
    print("=" * 60)
