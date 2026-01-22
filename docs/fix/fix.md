# Fix Documentation

## 1. Windows Ctrl+C and WinError 10038 Issue

### Problem Description
When running Flask (Werkzeug) on Windows with SQLAlchemy's `Base.metadata.create_all` operation (`run_sync`) executed through the `aiosqlite` async engine, this operation creates a "thread-lock" that blocks Windows signal handling (Ctrl+C). When attempting to close the application, it would hang and produce the following error:
`OSError: [WinError 10038] An operation was attempted on something that is not a socket`

### Solution: Hybrid Startup Approach
The root cause of the deadlock is the heavy sync-async bridge (`run_sync`) operation performed in the middle of the async engine lifecycle. The solution is to perform table creation with a **temporary synchronous engine** before entering the async event loop.

**Implementation example (`base.py`):**
```python
# Create tables with sync engine without polluting the async engine
from sqlalchemy import create_engine
sync_url = self.database_url.replace("+aiosqlite", "")
sync_engine = create_engine(sync_url)
Base.metadata.create_all(sync_engine)
sync_engine.dispose()
```

### Why This Works
1. Table creation is a one-time startup operation
2. Using a separate sync engine avoids async/sync bridge conflicts
3. The sync engine is disposed immediately after use
4. The async engine remains clean for runtime operations
5. Windows signal handling (Ctrl+C) works correctly
