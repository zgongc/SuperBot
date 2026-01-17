#!/usr/bin/env python3

"""
docs/master/rules.md

SuperBot - Development Rules and Standards
Author: SuperBot Team
Date: 2025-11-12
Version: 1.0.0

This document defines the rules, coding standards, and process expectations
to be followed when working on the SuperBot project. The goal is to ensure
consistency across modules, ease of maintenance, and quality assurance.
"""

# 1. General Principles

- **Plan-First**: Before starting new development, the relevant sprint or
  architecture plan under `docs/plans/` must be updated.
- **Backtest-First**: Every strategy to be deployed in live environment must
  have passed success criteria in the Backtest module.
- **Modularity**: `core/` services and `components/` are shared resources;
  modules should be designed to be loosely coupled.
- **Observability**: Logging and metrics should be considered from day one,
  even at a minimal level.

# 2. File Structure Standards

- Every Python module must include a header and a test section (footer).
- **Header template**:

```
#!/usr/bin/env python3

"""
path/to/file.py

SuperBot - Module Name
Author: SuperBot Team
Date: YYYY-MM-DD
Version: X.Y.Z

Module description (brief and concise)

Features:
- Feature 1
- Feature 2
- Feature 3

Usage:
    from module import Class
    instance = Class()
    result = instance.method()

Dependencies:
    - python>=3.10
    - package1>=1.0.0
    - package2 (optional)
"""
```

- **Footer template**:

```
# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ModuleName Test")
    print("=" * 60)
    # Test 1: Basic functionality
    print("Basic function test:")
    # Test code here
    print("   ✅ Test successful")
    # Test 2: Another test
    print("Second test:")
    # Test code here
    print("   ✅ Test successful")
    print("\n✅ All tests completed!")
    print("=" * 60)
```

- The test section is optional only for CLI/daemon-like scripts; it is
  mandatory for library files.

# 3. Coding Standards

- **Language**: Python 3.12. All code must be `black` format compliant.
- **Types**: Use `from __future__ import annotations`; write complete type hints.
  Target Pyright/ruff compatibility.
- **Naming**:
  - Files: `snake_case.py`
  - Classes: `CapWords`
  - Variables/Functions: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Docstring**: Write Google-style docstrings for every module, class, and
  complex function. Add a brief summary per module.
- **TODO**: If needed, use `# TODO(username): description` format; reference
  the related backlog item.
- **Class Structure**: The docstring should list what the class does and its
  attributes. `__init__`, `initialize`, `process` examples follow this template:

```
class MyManager:
    """
    Manager description

    This manager does the following:
    - Task 1
    - Task 2

    Attributes:
        config: Config engine instance
        logger: Logger instance
    """

    def __init__(self, config, logger):
        """Initialize the manager"""
        self.config = config
        self.logger = logger
        self._initialized = False

    def initialize(self):
        """Start the manager"""
        self.logger.info("🚀 Manager starting...")
        # Initialization code
        self._initialized = True
        self.logger.info("✅ Manager started")

    def process(self, data):
        """
        Process data

        Args:
            data: Data to process

        Returns:
            dict: Processed result
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized")

        # Processing code
        return result
```

- **Error Handling**: Context should be provided with Turkish message + emoji;
  example template:

```
try:
    result = risky_operation()
except ConnectionError as e:
    self.logger.error(f"❌ Connection error: {e}")
    raise
except ValueError as e:
    self.logger.warning(f"⚠️  Invalid value: {e}")
    return None
except Exception as e:
    self.logger.critical(f"🚨 Unexpected error: {e}")
    raise
```

# 4. File and Directory Rules

- Every new component must adhere to the directory structure specified in the
  plan document.
- Files under `components/` should only import necessary functions when
  importing core services.
- Temporary scripts or notebooks should be kept in a local `sandbox/` directory
  and should not enter the repository.
- Configuration changes should be made through `config/main.yaml` and
  `config/infrastructure.yaml`; default values should not be embedded in code.
- Every file needing `config` access must load config through
  `core/config_engine.py`; direct YAML reading is prohibited.
- Files using logging or `print` must create loggers through
  `core/logger_engine.py`; deviating from standardization is not allowed.

# 5. Logging, Emoji, and Language Standards

- The logger provided by `core/logger_engine.py` must be used; `print` is
  prohibited (except for CLI/daemon entry point test outputs only).
- Log levels:
  - `debug`: Developer-focused detail
  - `info`: Workflow steps
  - `warning`: Unexpected but tolerable condition
  - `error`: Recoverable error
  - `critical`: Error threatening system stability
- Each log should include context tags like `strategy`, `symbol`, `timeframe`,
  `request_id` where possible.
- Metrics should be kept Prometheus-compatible; when adding new metrics, add
  notes to `docs/plans/`.
- **Log Messages**: 100% Turkish, indicate level with emoji; incorrect examples
  are not accepted.
- **Emoji Preservation**: No emoji in code should be deleted or modified.
  Console not displaying emojis is a cosmetic issue; on Windows, the
  `PYTHONIOENCODING` and `PYTHONLEGACYWINDOWSSTDIO` environment variables
  can be set to `utf-8`.

```
# ✅ Correct
logger.debug(f"🔍 Debug data: {variable}")
logger.info(f"📊 Statistics updated: {count} records")
logger.warning(f"⚠️  Limit exceeded: {warning_detail}")
logger.error(f"❌ Risk limit violation: {error_message}")
logger.critical(f"🚨 System error: {critical_issue}")

# ❌ Wrong
logger.debug("Debug data")
logger.info("Stats updated")
logger.warning("Warning")
```

- **Emoji Guide**:
  - `🔍` debug/search
  - `✅` success
  - `📊` statistics
  - `🚀` startup
  - `⚠️` warning
  - `❌` error
  - `🚨` critical error
  - `🛑` stop
  - `🔄` restart
  - `💾` data save
  - `📝` log entry
  - `🌐` network
  - `🔐` security
  - `⏱️` timing
  - `💰` capital
  - `📂` file
  - `🎯` target
- **Comments and Exception Messages**: Written 100% in Turkish.

```
# ✅ Correct
# Start the engine and perform health check
raise ValueError("Invalid config parameter")

# ❌ Wrong
# Start the engine and perform health check
raise ValueError("Invalid config parameter")
```

# 6. Test Policy

- pytest-based tests are mandatory for all new code. Code without accompanying
  tests will be held in PR.
- Backtest scenarios should run as regression tests; failed tests will not be
  merged without resolution.
- Use fixture-based realistic data when possible instead of mocks.
- Tests must be deterministic; seed must be fixed for random components.

# 7. Security and Configuration

- Secret keys are managed through `.env` or secret manager; they are never
  committed to the repository in plaintext.
- Write a rollback plan before updating `security_engine` master key.
- Configuration changes should be made through `config_engine`; manually
  written config files must pass schema validation.

# 8. Dependency Management

- Open a discussion before adding new dependencies; check license and
  compatibility.
- When updating `requirements.txt`, pin the full version number.
- For dependencies requiring system services, add an installation guide under
  `docs/guides/`.

# 9. Development Processes

- **Branching**: `main` is protected. For feature development, open branches
  in `feature/<module>/<feature>` format.
- **Commit Message**: `type(scope): description` (e.g., `feat(trading): add live monitor`).
  `type` set: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `build`.
- **Code Review**: Run tests before opening PR; add summary and checklist for
  reviewer.
- **CI/CD**: If pipeline fails, priority is fixing it; pipeline is not left
  broken.
- **Python Environment**: Use `conda activate superbot` command to use the
  `superbot` environment in all development and test processes. Alternative
  environments must switch back to this environment before opening PR.

# 10. AI and Automation Usage

- AI-assisted tools (e.g., strategy optimization) must be defined in plans
  under `docs/plans/` before producing results.
- AI outputs must undergo manual validation; source and justification of
  automatically generated code is added to PR description.
- FastAPI-based AI services versioning and model registration policy is kept
  consistent with `docs/plans/superbot-architecture.md`.

# 11. Violation and Revision

- If non-compliance with these rules is detected, the relevant developer is
  warned; in recurring cases, the code review process is tightened.
- To maintain document currency, it is reviewed at the end of each sprint;
  if revision is needed, version number is incremented.

----

These rules are designed to ensure the SuperBot project develops in a
sustainable and scalable manner. Every team member is responsible for
working in accordance with this guide.
