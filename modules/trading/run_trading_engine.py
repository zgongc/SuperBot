#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/trading/run_trading_engine.py

SuperBot - Trading Engine CLI
Date: 2025-11-30
Versiyon: 1.1.0

CLI entry point for Trading Engine V4.

Usage:
    # Paper mode (default)
    python modules/trading/run_trading_engine.py --mode paper --strategy ema5_bb_adx

    # Live mode
    python modules/trading/run_trading_engine.py --mode live --strategy ema5_bb_adx

    # Replay mode (historical data playback)
    python modules/trading/run_trading_engine.py --mode replay --strategy bb_stochrsi

    # Verbose mode
    python modules/trading/run_trading_engine.py --mode paper --strategy ema5_bb_adx --verbose

    # Full strategy path
    python modules/trading/run_trading_engine.py --strategy components/strategies/templates/ema5_bb_adx.py

Replay Mode Controls:
    +/=     : Increase speed (0.25x → 0.5x → 1x → 2x → 4x → 10x)
    -/_     : Decrease speed
    SPACE   : Pause/Resume
    q       : Quit
"""

from __future__ import annotations

import sys
import signal
import asyncio
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.logger_engine import get_logger
from modules.trading.trading_engine import TradingEngine


# ============================================================================
# KEYBOARD INPUT HANDLER (for replay mode)
# ============================================================================

class KeyboardHandler:
    """
    Non-blocking keyboard input handler for replay mode controls.

    Windows: Uses msvcrt
    Unix/Linux: Uses select + termios
    """

    def __init__(self):
        self._is_windows = sys.platform == 'win32'
        self._original_settings = None

        if self._is_windows:
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios
            import tty
            import select
            self._termios = termios
            self._tty = tty
            self._select = select

    def setup(self):
        """Setup terminal for non-blocking input"""
        if not self._is_windows:
            import sys
            self._original_settings = self._termios.tcgetattr(sys.stdin)
            self._tty.setcbreak(sys.stdin.fileno())

    def cleanup(self):
        """Restore terminal settings"""
        if not self._is_windows and self._original_settings:
            import sys
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._original_settings)

    def get_key(self) -> str | None:
        """
        Non-blocking key read.

        Returns:
            Key character or None if no key pressed
        """
        if self._is_windows:
            if self._msvcrt.kbhit():
                key = self._msvcrt.getch()
                # Handle special keys
                if key == b'\xe0':  # Arrow keys prefix
                    self._msvcrt.getch()  # Consume arrow key
                    return None
                return key.decode('utf-8', errors='ignore')
            return None
        else:
            import sys
            if self._select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None


def handle_replay_input(key: str, mode) -> bool:
    """
    Handle keyboard input for replay mode.

    Args:
        key: Pressed key character
        mode: ReplayMode instance

    Returns:
        True if should quit, False otherwise
    """
    if not mode:
        print(f"[DEBUG] Key '{key}' pressed but mode is None")
        return False

    # Speed levels
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]

    if key in ('+', '='):
        # Increase speed
        current = mode._speed
        current_idx = -1
        for i, s in enumerate(SPEEDS):
            if abs(current - s) < 0.01:
                current_idx = i
                break

        if current_idx < len(SPEEDS) - 1:
            new_speed = SPEEDS[current_idx + 1]
            mode.set_speed(new_speed)
            print(f"\n⏩ Speed: {new_speed}x")
        return False

    elif key in ('-', '_'):
        # Decrease speed
        current = mode._speed
        current_idx = -1
        for i, s in enumerate(SPEEDS):
            if abs(current - s) < 0.01:
                current_idx = i
                break

        if current_idx > 0:
            new_speed = SPEEDS[current_idx - 1]
            mode.set_speed(new_speed)
            print(f"\n⏪ Speed: {new_speed}x")
        return False

    elif key == ' ':
        # Toggle pause/resume
        if mode._paused:
            mode.resume()
            print("\n▶️ Resumed")
        else:
            mode.pause()
            print("\n⏸️ Paused")
        return False

    elif key in ('q', 'Q'):
        # Quit
        print("\n🛑 Quitting...")
        mode.stop()
        return True

    return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SuperBot Trading Engine V4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python modules/trading/run_trading_engine.py --mode paper --strategy ema5_bb_adx
  python modules/trading/run_trading_engine.py --mode live --strategy ema5_bb_adx --verbose
  python modules/trading/run_trading_engine.py --strategy components/strategies/templates/ema5_bb_adx.py
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['paper', 'live', 'replay'],
        default='paper',
        help='Trading mode: paper, live, or replay (default: paper)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name or full path (e.g., ema5_bb_adx or components/strategies/templates/ema5_bb_adx.py)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose mode (detailed trading info)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Custom config file path'
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    logger = get_logger("run_trading_engine")

    is_replay = args.mode == 'replay'

    # Print banner
    print("=" * 60)
    print("🤖 SuperBot Trading Engine V4")
    print("=" * 60)
    print(f"   Mode:     {args.mode.upper()}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Verbose:  {args.verbose}")

    if is_replay:
        print("-" * 60)
        print("🎮 Replay Controls:")
        print("   +/=     : Increase speed")
        print("   -/_     : Decrease speed")
        print("   SPACE   : Pause/Resume")
        print("   q       : Quit")

    print("=" * 60)

    # Create engine
    engine = TradingEngine(
        mode=args.mode,
        strategy_path=args.strategy,
        verbose=args.verbose
    )

    # Shutdown flag
    shutdown_event = asyncio.Event()

    # Keyboard handler for replay mode (init early for signal handler access)
    keyboard = None
    if is_replay:
        keyboard = KeyboardHandler()

    # Track Ctrl+C count for replay mode
    ctrl_c_count = [0]

    def signal_handler(sig, _frame):
        ctrl_c_count[0] += 1

        if is_replay:
            if ctrl_c_count[0] == 1:
                # First Ctrl+C - just warn, don't exit
                print("\n⚠️ Ctrl+C pressed - use 'q' to quit or SPACE to pause")
                return  # Don't set shutdown_event
            else:
                # Second Ctrl+C - force exit
                print("\n⚠️ Force shutdown...")
                if keyboard:
                    keyboard.cleanup()
                sys.exit(1)
        else:
            logger.info(f"⚠️ Signal {sig} received, initiating shutdown...")
            shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start engine
        await engine.initialize()

        # Run engine.start() as background task so we can handle keyboard
        engine_task = asyncio.create_task(engine.start())

        # Wait for engine to actually start (race condition fix)
        for _ in range(50):  # Max 2.5 seconds
            if engine.is_running:
                break
            await asyncio.sleep(0.05)

        # Run until stopped
        if is_replay:
            logger.info("🎬 Replay Mode running. Use +/- for speed, SPACE for pause, q to quit.")
            keyboard.setup()
        else:
            logger.info("📈 Trading Engine running. Press Ctrl+C to stop.")

        # Wait for shutdown signal or engine stop
        while (engine.is_running or not engine_task.done()) and not shutdown_event.is_set():
            # Handle keyboard input for replay mode
            if is_replay and keyboard:
                try:
                    key = keyboard.get_key()
                    if key:
                        # Get replay mode from engine (current_mode attribute)
                        mode = getattr(engine, 'current_mode', None)
                        if handle_replay_input(key, mode):
                            shutdown_event.set()
                except Exception as e:
                    logger.debug(f"Keyboard error: {e}")

            # Check if engine task completed
            if engine_task.done():
                break

            await asyncio.sleep(0.05)  # 50ms polling for responsive keyboard

        # Cancel engine task if still running
        if not engine_task.done():
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info("⚠️ Keyboard interrupt received")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)

    finally:
        # Cleanup keyboard handler
        if keyboard:
            keyboard.cleanup()

        await engine.stop()

    # Print final stats
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)

    # Replay mode stats
    if is_replay and hasattr(engine, 'current_mode') and engine.current_mode:
        mode = engine.current_mode
        stats = mode.get_statistics() if hasattr(mode, 'get_statistics') else {}
        print(f"   Processed:     {stats.get('processed_candles', 0)} candles")
        print(f"   Total Trades:  {stats.get('trades', 0)}")
        print(f"   PnL:           ${stats.get('pnl', 0):+,.2f}")
    # Normal mode stats (paper/live)
    elif hasattr(engine, '_trade_logger') and engine._trade_logger:
        stats = engine._trade_logger.get_statistics() if hasattr(engine._trade_logger, 'get_statistics') else {}
        print(f"   Total Trades:  {stats.get('total_trades', len(engine._positions))}")
        print(f"   Open Positions: {len(engine._positions)}")

    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
