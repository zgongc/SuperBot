"""Async helper utilities for Flask"""
import asyncio
import threading

# Persistent event loop for Flask async calls
_loop = None
_loop_thread = None
_loop_lock = threading.Lock()
_loop_started = False


def _start_background_loop(loop):
    """Start event loop in background thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_persistent_loop():
    """Get or create a persistent event loop"""
    global _loop, _loop_thread, _loop_started
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_started = False
        return _loop


def _ensure_loop_running():
    """Ensure the background loop thread is started"""
    global _loop_thread, _loop_started
    with _loop_lock:
        if not _loop_started and _loop is not None:
            _loop_thread = threading.Thread(target=_start_background_loop, args=(_loop,), daemon=True)
            _loop_thread.start()
            _loop_started = True


def get_event_loop():
    """
    Get event loop for initialization (run_until_complete style).
    This returns the loop before it starts running in background.
    """
    return get_persistent_loop()


def run_async(coro):
    """
    Run async coroutine in Flask context using a persistent event loop.

    This avoids the "Lock bound to different event loop" error by
    always using the same event loop for all async operations.
    """
    global _loop_started
    loop = get_persistent_loop()

    # If loop is not yet running in background, use run_until_complete
    if not _loop_started:
        return loop.run_until_complete(coro)

    # Otherwise, use run_coroutine_threadsafe
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=60)


def start_background_loop():
    """
    Start the background event loop thread.
    Call this after initialization is complete.
    """
    _ensure_loop_running()
