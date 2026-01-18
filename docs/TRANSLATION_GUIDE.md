# Translation Guide for SuperBot

This document describes the translation process from Turkish to English for the SuperBot codebase.

## Scope

Translation covers **comments, docstrings, and log messages only**. Code logic remains untouched.

## What Gets Translated

| Element | Example Before | Example After |
|---------|---------------|---------------|
| Module docstrings | `"""SuperBot - Config Yönetim Sistemi"""` | `"""SuperBot - Config Management System"""` |
| Class docstrings | `"""Merkezi loglama sistemi"""` | `"""Central logging system"""` |
| Method docstrings | `"""Config değerini al"""` | `"""Get config value"""` |
| Inline comments | `# Dosyayı yükle` | `# Load file` |
| Log messages | `logger.info("Bot başlatıldı")` | `logger.info("Bot started")` |
| Error messages | `raise ValueError("Geçersiz değer")` | `raise ValueError("Invalid value")` |
| Print statements | `print("Test tamamlandı")` | `print("Test completed")` |

## What Does NOT Get Translated

- Variable names
- Function names
- Class names
- File names
- Configuration keys
- Any code logic

## Translation Rules

1. **Keep technical terms in English**: API, JSON, YAML, Redis, Pydantic, etc.
2. **Preserve formatting**: Keep the same indentation, line breaks, and structure
3. **Maintain docstring style**: If original uses Args/Returns format, keep it
4. **Keep emojis**: Emojis in log messages should remain unchanged
5. **Preserve placeholders**: `{variable}`, `${ENV_VAR}`, f-string expressions stay as-is

## Example Translation

### Before (Turkish)
```python
def load(self, filename: str) -> bool:
    """
    Tek bir config dosyasını yükle

    Args:
        filename: Config dosya adı (örn: main.yaml)

    Returns:
        bool: Başarılı ise True
    """
    if not file_path.exists():
        logger.error(f"Config dosyası bulunamadı: {file_path}")
        return False
```

### After (English)
```python
def load(self, filename: str) -> bool:
    """
    Load a single config file.

    Args:
        filename: Config file name (e.g., main.yaml)

    Returns:
        bool: True if successful
    """
    if not file_path.exists():
        logger.error(f"Config file not found: {file_path}")
        return False
```

## Files to Translate

### Core Module (`core/`)
- [x] `logger_engine.py` - Central logging system
- [ ] `config_engine.py` - Config management system
- [x] `cache_manager.py` - Cache management (translated earlier)

### Memory Module (`memory/`)
- [ ] `captain_memory.py` - Session memory system

### Components
- [ ] `components/strategies/templates/*.py` - Strategy templates

### Documentation (`docs/`)
- Already has both TR and EN versions in `docs/master/` and `docs/master/tr/`

## Prompt for AI Translation

When asking an AI to translate, use this prompt:

```
Translate the Turkish content in this Python file to English.

Rules:
1. Only translate comments, docstrings, and string literals (log messages, error messages)
2. Do NOT modify any code logic, variable names, function names, or class names
3. Keep technical terms in English (API, JSON, config, etc.)
4. Preserve all formatting, indentation, and structure
5. Keep emojis unchanged
6. Maintain the same docstring style (Args/Returns format)

File: [paste file content here]
```

## Verification

After translation, verify:
1. Code still runs without syntax errors
2. All tests pass
3. No Turkish words remain in comments/docstrings (use IDE spell checker)
4. Log messages make sense in English context
