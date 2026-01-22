#!/usr/bin/env python3
"""
Translation Scanner - Find and Translate Turkish content in Python/HTML/JS files

Scans Python, HTML, and JavaScript files for Turkish content
and can auto-translate using Ollama (local LLM).

Usage:
    # Python files (default)
    python components/tools/translate_scanner.py                    # Scan all .py files
    python components/tools/translate_scanner.py core/              # Scan specific folder
    python components/tools/translate_scanner.py --translate        # Auto-translate with Ollama

    # HTML files
    python components/tools/translate_scanner.py modules/webui/templates/ --html --translate

    # JavaScript files
    python components/tools/translate_scanner.py modules/webui/static/js/ --js --translate

    # Combined flags
    python components/tools/translate_scanner.py modules/webui/ --html --js --translate --dry-run

    # With options
    python components/tools/translate_scanner.py --translate --ollama-url http://192.168.1.195:11434 --model translategemma:12b
"""

import re
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Ollama API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Turkish-specific characters
TURKISH_CHARS = re.compile(r'[çğıöşüÇĞİÖŞÜ]')

# Common Turkish words (for detection even without special chars)
TURKISH_WORDS = [
    # Conjunctions & prepositions
    r'\b(ve|veya|icin|ile|olan|olarak|gibi|kadar|sonra|once)\b',
    r'\b(degil|var|yok|evet|hayir|tamam|lutfen)\b',
    # File/folder terms
    r'\b(dosya|klasor|dizin|ayar|yapilandirma)\b',
    # Status words
    r'\b(basarili|basarisiz|hata|uyari|bilgi)\b',
    r'\b(yukleniyor|kaydediliyor|siliniyor|guncelleniyor)\b',
    r'\b(bulunamadi|zaten|mevcut|gerekli|zorunlu)\b',
    # Common technical terms in Turkish
    r'\b(aktif|pasif|etkin|devre)\b',
    r'\b(hesaplama|metodu?|parametreler?i?)\b',
    r'\b(korelasyon|limiti|seviyes?i)\b',  # limiti (not limit), seviye/seviyesi
    r'\b(dakika|saat|gun|hafta)\b',
    r'\b(maksimum|toplam)\b',  # removed 'minimum' - same in English
    r'\b(acil|durum|durdurucu)\b',
    r'\b(dinamik|boyut|boyutu)\b',
    r'\b(tetikleme|tetikle)\b',
    r'\b(cikis|giris|kar|zarar)\b',
    r'\b(ornek|kullanim|kullanma|kullan)\b',
    r'\b(KULLANILMAZ|edilir|etsin)\b',
    r'\b(izni|izin|kontrol)\b',
    # More common Turkish words without special chars
    r'\b(sadece|ayarla|gerisi|otomatik)\b',
    r'\b(tipi|tipinde|olacak|olmaz)\b',
    r'\b(tespit|edildi|sonucu?)\b',
    r'\b(miktar|fazla|riskli|cok)\b',
    r'\b(koordinasyonu?|kosullari?)\b',
    r'\b(olmali|listesi?)\b',
    # More validation/error message words
    r'\b(kosul|eleman|icermeli)\b',
    r'\b(gecersiz|operator|negatif|olamaz)\b',
    r'\b(strateji|objesi|opsiyonel)\b',
    r'\b(ikisi|gelir|sahip|kendi)\b',
    # Performance/trading report words
    r'\b(periyod|sembol|performans)\b',
    r'\b(toplam|trade|getiri)\b',
    r'\b(detaylar|kazanan|kaybeden)\b',
    r'\b(komisyon|slippage|spread)\b',
    r'\b(ort|execution)\b',
    # More trading/strategy words
    r'\b(strateji|sinyal|pozisyon)\b',
    r'\b(baslangic|bitis|sure)\b',
    r'\b(sonuc|rapor|ozet)\b',
    r'\b(islem|adet|oran)\b',
    r'\b(kar|zarar|bakiye)\b',
    r'\b(satis|alis|fiyat)\b',
    r'\b(acik|kapali|beklemede)\b',
    r'\b(hacim|lot|pip)\b',
    r'\b(giren|cikan|kalan)\b',
    r'\b(basari|kayip|elde)\b',
    # Headers/Metadata
    r'\b(yazar|tarih|sistemi|yetersiz)\b',
    r'\b(sorumluluk|uzun|vadeli|daha|gelecek)\b', 
    r'\b(son|devam)\b', 
    
    # Question endings
    r'\b\w+\s+mi\?\b',
    # Common log words/verbs
    r'\b(okundu|temizle|temizlendi|tamamlandi)\b',
]
TURKISH_WORD_PATTERN = re.compile('|'.join(TURKISH_WORDS), re.IGNORECASE)


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    base_url: str = "http://192.168.1.195:11434"
    model: str = "translategemma:12b"  # or qwen2.5, mistral, etc.
    timeout: int = 120


class OllamaTranslator:
    """Translate text using Ollama API."""

    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session() if REQUESTS_AVAILABLE else None

    def translate_line(self, turkish_text: str, context: str = "") -> Optional[str]:
        """
        Translate a single line from Turkish to English.

        Args:
            turkish_text: Turkish text to translate
            context: Additional context (e.g., "docstring", "comment")

        Returns:
            English translation or None if failed
        """
        if not self.session:
            print("[ERROR] requests library not installed")
            return None

        prompt = f"""Translate this Turkish text to English.

CRITICAL RULES:
- This is Python source code translation
- Do NOT use markdown code blocks (```) in your response
- Do NOT add any formatting like ```text or ```python
- NEVER add triple quotes (''' or \"\"\") - they break the Python file structure!
- Return ONLY the translated text, preserving the exact same line count
- Only translate Turkish comments and strings to English
- Keep ALL Python syntax exactly as it is (indentation, quotes, operators, etc.)
- Keep variable names, function names, class names UNCHANGED
- Keep technical terms in English (API, JSON, config, etc.)
- Keep emojis unchanged
- If it's a docstring, maintain the exact Args/Returns/Raises format
- The input has {turkish_text.count(chr(10)) + 1} lines - output must have EXACTLY the same number of lines

Context: {context}

Turkish text:
{turkish_text}

English translation (same line count, no triple quotes, no markdown):"""

        try:
            # Try Open WebUI OpenAI-compatible endpoint
            response = self.session.post(
                f"{self.config.base_url}/ollama/api/chat",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are a translator. Translate Turkish to English. Only output the translation, nothing else."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    translated = data["message"].get("content", "").strip()
                    translated = self._clean_translation(translated, context)
                    return translated
                if "choices" in data:
                    translated = data["choices"][0]["message"]["content"].strip()
                    translated = self._clean_translation(translated, context)
                    return translated

            # Try native Ollama API (via Open WebUI proxy)
            response = self.session.post(
                f"{self.config.base_url}/ollama/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                translated = data.get("response", "").strip()
                translated = self._clean_translation(translated, context)
                return translated

            # Try direct Ollama API (if running separately)
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                translated = data.get("response", "").strip()
                translated = self._clean_translation(translated, context)
                return translated

            print(f"[ERROR] Ollama API error: {response.status_code}")
            return None

        except requests.exceptions.Timeout:
            print("[ERROR] Ollama API timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Ollama API error: {e}")
            return None

    def _clean_translation(self, translated: str, context: str = "") -> str:
        """
        Clean up translated text, removing artifacts that break Python syntax.

        Args:
            translated: Raw translation from LLM
            context: Content type (docstring, comment, etc.)

        Returns:
            Cleaned translation safe for Python source
        """
        # Remove markdown code blocks
        translated = re.sub(r'^```\w*\n', '', translated)
        translated = re.sub(r'\n```$', '', translated)
        translated = translated.replace('```text', '').replace('```python', '')

        # Remove standalone ``` lines
        translated = '\n'.join(line for line in translated.split('\n') if line.strip() != '```')

        # CRITICAL: Remove triple quotes that LLM may have added
        # These break docstring structure when inserted into the middle of docstrings
        if context in ('docstring', 'docstring_block'):
            # Remove standalone triple quotes on their own line
            lines = translated.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines that are only triple quotes
                if stripped in ('"""', "'''"):
                    continue
                # Remove triple quotes at start/end of lines within docstrings
                # But preserve lines that legitimately contain code with strings
                if stripped.startswith('"""') and not '=' in stripped:
                    line = line.replace('"""', '', 1)
                if stripped.endswith('"""') and not '=' in stripped:
                    line = line[::-1].replace('"""'[::-1], '', 1)[::-1]
                if stripped.startswith("'''") and not '=' in stripped:
                    line = line.replace("'''", '', 1)
                if stripped.endswith("'''") and not '=' in stripped:
                    line = line[::-1].replace("'''"[::-1], '', 1)[::-1]
                cleaned_lines.append(line)
            translated = '\n'.join(cleaned_lines)

        # Normalize Unicode to ASCII
        translated = self._normalize_unicode(translated)

        return translated

    def _normalize_unicode(self, text: str) -> str:
        """Normalize problematic Unicode characters to ASCII equivalents."""
        # Common problematic Unicode characters that cause Python syntax errors
        replacements = {
            '\u00d7': 'x',      # Multiplication sign (U+00D7) -> x
            '\u2192': '->',     # Rightwards arrow (U+2192) -> ->
            '\u2264': '<=',     # Less-than or equal to (U+2264) -> <=
            '\u2265': '>=',     # Greater-than or equal to (U+2265) -> >=
            '\u2260': '!=',     # Not equal to (U+2260) -> !=
            '\u2212': '-',      # Minus sign (U+2212) -> -
            '\u00b7': '*',      # Middle dot (U+00B7) -> *
            '\u00f7': '/',      # Division sign (U+00F7) -> /
        }
        
        for unicode_char, ascii_equiv in replacements.items():
            text = text.replace(unicode_char, ascii_equiv)
        
        # Fix invalid Python syntax: number followed by 'x' (e.g., "5x" -> "5 x")
        # This happens when docstrings are malformed and content ends up outside quotes
        import re
        text = re.sub(r'(\d+)x\b', r'\1 x', text)  # 5x -> 5 x
        text = re.sub(r'(\d+)x\)', r'\1 x)', text)  # 5x) -> 5 x)
        
        return text

    def translate_file_content(self, lines: List[str], findings: List[Tuple[int, str, str]],
                                progress_callback=None) -> Dict[int, str]:
        """
        Translate multiple lines in a file.
        
        Groups consecutive docstring lines to preserve context during translation.

        Args:
            lines: All lines in the file
            findings: List of (line_number, content_type, text) to translate
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping line_number to translated text
        """
        translations = {}
        total = len(findings)
        
        # Group consecutive docstring lines together
        grouped_findings = self._group_docstring_lines(findings)
        
        processed = 0
        for group in grouped_findings:
            if len(group) == 1:
                # Single line - translate normally
                line_num, content_type, text = group[0]
                processed += 1
                
                if progress_callback:
                    progress_callback(processed, total, text[:50])
                
                translated = self.translate_line(text, context=content_type)
                if translated:
                    translations[line_num] = translated
            else:
                # Multiple consecutive docstring lines - translate as a block
                first_line_num = group[0][0]
                last_line_num = group[-1][0]
                
                # Combine lines while preserving indentation structure
                combined_lines = [item[2] for item in group]
                combined_text = '\n'.join(combined_lines)
                
                if progress_callback:
                    processed += len(group)
                    progress_callback(processed, total, f"[DOCSTRING BLOCK L{first_line_num}-{last_line_num}]")
                
                # Translate the entire block
                translated_block = self.translate_line(combined_text, context="docstring_block")
                
                if translated_block:
                    # Split back into individual lines
                    translated_lines = translated_block.split('\n')
                    
                    # Ensure we have the same number of lines
                    if len(translated_lines) == len(group):
                        for idx, (line_num, _, _) in enumerate(group):
                            translations[line_num] = translated_lines[idx]
                    else:
                        # Fallback: if line count mismatch, translate individually
                        print(f"  [WARN] Block translation line count mismatch, falling back to individual")
                        for line_num, content_type, text in group:
                            translated = self.translate_line(text, context=content_type)
                            if translated:
                                translations[line_num] = translated

        return translations
    
    def _group_docstring_lines(self, findings: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
        """
        Group consecutive docstring lines together.
        
        Args:
            findings: List of (line_number, content_type, text)
            
        Returns:
            List of groups, where each group is a list of consecutive docstring findings
        """
        if not findings:
            return []
        
        groups = []
        current_group = []
        last_line_num = None
        
        for finding in findings:
            line_num, content_type, text = finding
            
            # Check if this is a docstring and consecutive to the last line
            if content_type == 'docstring':
                if current_group and last_line_num is not None and line_num == last_line_num + 1:
                    # Add to current docstring group
                    current_group.append(finding)
                else:
                    # Start new group
                    if current_group:
                        groups.append(current_group)
                    current_group = [finding]
                last_line_num = line_num
            else:
                # Not a docstring - close current group and add as single item
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([finding])
                last_line_num = None
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups

    def check_connection(self) -> bool:
        """Check if Ollama API is accessible."""
        if not self.session:
            return False

        try:
            # Try models endpoint
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                return True

            # Try OpenAI-compatible endpoint
            response = self.session.get(
                f"{self.config.base_url}/api/models",
                timeout=5
            )
            return response.status_code == 200

        except Exception:
            return False


# ============================================================================
# HTML Scanner
# ============================================================================

class HTMLScanner:
    """Scan HTML/Jinja2 template files for Turkish content."""

    # Patterns for extractable text in HTML
    # These are SAFE to translate
    SAFE_PATTERNS = [
        # Text inside common tags (added small, strong, em, b, i)
        (r'<(h[1-6]|p|span|div|label|button|a|td|th|li|option|title|small|strong|em|b|i)([^>]*)>([^<]+)</\1>', 'tag_content'),
        # Placeholder attributes
        (r'placeholder="([^"]+)"', 'placeholder'),
        (r"placeholder='([^']+)'", 'placeholder'),
        # Title attributes
        (r'title="([^"]+)"', 'title'),
        (r"title='([^']+)'", 'title'),
        # Alt attributes for images
        (r'alt="([^"]+)"', 'alt'),
        # HTML comments (but not Jinja comments)
        (r'<!--\s*([^-]+)\s*-->', 'html_comment'),
    ]

    # Patterns to SKIP (dangerous to translate)
    SKIP_PATTERNS = [
        r'\$\{.*?\}',           # Template literals
        r'\{\{.*?\}\}',         # Jinja2 variables
        r'\{%.*?%\}',           # Jinja2 blocks
        r'onclick=',            # Event handlers
        r'onchange=',
        r'onsubmit=',
        r'oninput=',
        r'data-[a-z]+=',        # Data attributes
        r'class=',              # CSS classes
        r'id=',                 # IDs
        r'href=',               # Links
        r'src=',                # Sources
        r'style=',              # Inline styles
    ]

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results: Dict[str, List[Tuple[int, str, str, str]]] = {}  # line_num, type, original, extracted
        self.skip_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv'}

    def is_turkish(self, text: str) -> bool:
        """Check if text contains Turkish content."""
        if TURKISH_CHARS.search(text):
            return True
        if TURKISH_WORD_PATTERN.search(text):
            return True
        return False

    def should_skip_line(self, line: str) -> bool:
        """Check if line contains patterns that shouldn't be translated."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if there's Turkish content outside these patterns
                cleaned = re.sub(pattern, '', line, flags=re.IGNORECASE)
                if not self.is_turkish(cleaned):
                    return True
        return False

    def extract_translatable_text(self, line: str, line_num: int) -> List[Tuple[int, str, str, str]]:
        """
        Extract translatable text segments from an HTML line.

        Returns:
            List of (line_num, content_type, full_match, text_to_translate)
        """
        findings = []

        for pattern, content_type in self.SAFE_PATTERNS:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                # Get the captured group (text content)
                if content_type == 'tag_content':
                    text = match.group(3).strip()
                else:
                    text = match.group(1).strip()

                if text and self.is_turkish(text):
                    # Skip if it's a Jinja2 expression
                    if '{{' in text or '{%' in text:
                        continue
                    # Skip if it's mostly code
                    if text.count('(') > 0 or text.count('{') > 0:
                        continue
                    findings.append((line_num, content_type, match.group(0), text))

        return findings

    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str, str]]:
        """Scan a single HTML file for Turkish content."""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except (UnicodeDecodeError, IOError):
            return findings

        # First, remove script and style blocks from content for whole-file scanning
        clean_content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)

        # Scan whole file for multi-line tag content
        for pattern, content_type in self.SAFE_PATTERNS:
            # Use DOTALL for multi-line matching
            for match in re.finditer(pattern, clean_content, re.IGNORECASE | re.DOTALL):
                if content_type == 'tag_content':
                    text = match.group(3).strip()
                else:
                    text = match.group(1).strip()

                if text and self.is_turkish(text):
                    # Skip Jinja2 expressions
                    if '{{' in text or '{%' in text:
                        continue
                    # Skip if mostly code
                    if text.count('(') > 0 or text.count('{') > 0:
                        continue

                    # Find line number
                    pos = match.start()
                    line_num = content[:pos].count('\n') + 1

                    findings.append((line_num, content_type, match.group(0), text))

        # Deduplicate findings (same text might match multiple patterns)
        seen = set()
        unique_findings = []
        for f in findings:
            key = (f[0], f[3])  # line_num and text
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)

        return unique_findings

    def scan_directory(self, directory: Optional[Path] = None) -> Dict[str, List[Tuple[int, str, str, str]]]:
        """Scan all HTML files in directory."""
        if directory is None:
            directory = self.root_path

        self.results = {}

        for html_file in directory.rglob('*.html'):
            if any(skip in html_file.parts for skip in self.skip_dirs):
                continue

            findings = self.scan_file(html_file)
            if findings:
                rel_path = str(html_file.relative_to(self.root_path))
                self.results[rel_path] = findings

        return self.results

    def translate_file(self, file_path: str, translator: OllamaTranslator,
                       dry_run: bool = False) -> bool:
        """Translate a single HTML file."""
        if file_path not in self.results:
            print(f"[SKIP] No Turkish content in {file_path}")
            return True

        full_path = self.root_path / file_path
        findings = self.results[file_path]

        print(f"\n[TRANSLATE HTML] {file_path} ({len(findings)} items)", flush=True)

        # Read file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate each finding
        translations = {}
        for i, (line_num, content_type, full_match, text) in enumerate(findings, 1):
            print(f"  [{i}/{len(findings)}] {text[:40]}...", flush=True)

            translated = translator.translate_line(text, context=f"HTML {content_type}")
            if translated:
                translations[full_match] = full_match.replace(text, translated.strip())

        if dry_run:
            print(f"\n[DRY-RUN] Would modify {file_path}:")
            for original, new in translations.items():
                print(f"  {original[:50]}")
                print(f"  -> {new[:50]}")
        else:
            # Backup
            backup_path = full_path.with_suffix('.html.bak')
            shutil.copy(full_path, backup_path)

            # Apply translations
            for original, new in translations.items():
                content = content.replace(original, new)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"[OK] Translated {file_path}")

        return True

    def print_summary(self):
        """Print summary of HTML findings."""
        if not self.results:
            print("[OK] No Turkish content found in HTML files!")
            return

        total = sum(len(f) for f in self.results.values())
        print(f"\n[HTML SCAN] Found {total} Turkish items in {len(self.results)} files")

        for file_path, findings in sorted(self.results.items()):
            print(f"\n  {file_path} ({len(findings)} items)")
            for line_num, content_type, full_match, text in findings[:3]:
                print(f"    L{line_num} [{content_type}]: {text[:50]}")
            if len(findings) > 3:
                print(f"    ... and {len(findings) - 3} more")


# ============================================================================
# JavaScript Scanner
# ============================================================================

class JSScanner:
    """Scan JavaScript files for Turkish content."""

    # Patterns for extractable text in JS
    SAFE_PATTERNS = [
        # Single-quoted strings (not in template literals)
        (r"'([^'\\]*(?:\\.[^'\\]*)*)'", 'string_single'),
        # Double-quoted strings
        (r'"([^"\\]*(?:\\.[^"\\]*)*)"', 'string_double'),
        # Single-line comments
        (r'//\s*(.+)$', 'comment'),
        # Multi-line comments
        (r'/\*\s*(.*?)\s*\*/', 'block_comment'),
    ]

    # Patterns to SKIP
    SKIP_PATTERNS = [
        r'\$\{.*?\}',           # Template literal expressions
        r'console\.',           # Console calls
        r'fetch\(',             # Fetch calls
        r'getElementById',      # DOM methods
        r'querySelector',
        r'addEventListener',
        r'\.classList\.',       # Class manipulation
        r'\.style\.',           # Style manipulation
        r'JSON\.',              # JSON operations
        r'localStorage',        # Storage
        r'sessionStorage',
        r'window\.',            # Window object
        r'document\.',          # Document object
        r'function\s+\w+',      # Function declarations
        r'const\s+\w+\s*=',     # Const declarations
        r'let\s+\w+\s*=',       # Let declarations
        r'var\s+\w+\s*=',       # Var declarations
        r'=>',                  # Arrow functions
        r'import\s+',           # Imports
        r'export\s+',           # Exports
    ]

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results: Dict[str, List[Tuple[int, str, str, str]]] = {}
        self.skip_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv', 'dist', 'build'}

    def is_turkish(self, text: str) -> bool:
        """Check if text contains Turkish content."""
        if TURKISH_CHARS.search(text):
            return True
        if TURKISH_WORD_PATTERN.search(text):
            return True
        return False

    def is_code_string(self, text: str) -> bool:
        """Check if the string looks like code rather than user-facing text."""
        code_indicators = [
            r'^[a-z_][a-z0-9_]*$',   # Variable names
            r'^[A-Z_]+$',            # Constants
            r'^\.',                   # Starts with dot
            r'^#[a-fA-F0-9]+$',      # Hex colors
            r'^https?://',           # URLs
            r'^/',                    # Paths
            r'\.(js|css|html|json)$', # File extensions
            r'^\d+$',                 # Numbers only
        ]
        for pattern in code_indicators:
            if re.match(pattern, text.strip()):
                return True
        return False

    def extract_translatable_text(self, line: str, line_num: int) -> List[Tuple[int, str, str, str]]:
        """Extract translatable text segments from a JS line."""
        findings = []

        # Skip lines with dangerous patterns
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, line):
                return findings

        for pattern, content_type in self.SAFE_PATTERNS:
            for match in re.finditer(pattern, line):
                text = match.group(1).strip()

                if not text:
                    continue

                # Skip code-like strings
                if self.is_code_string(text):
                    continue

                # Skip template literal expressions
                if '${' in text:
                    continue

                if self.is_turkish(text):
                    findings.append((line_num, content_type, match.group(0), text))

        return findings

    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str, str]]:
        """Scan a single JS file for Turkish content."""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, IOError):
            return findings

        in_template_literal = False

        for i, line in enumerate(lines, 1):
            # Track template literals (backticks)
            backtick_count = line.count('`') - line.count('\\`')
            if backtick_count % 2 == 1:
                in_template_literal = not in_template_literal

            # Inside template literals, be more careful
            if in_template_literal:
                # Only extract obvious Turkish text, not in ${} expressions
                clean_line = re.sub(r'\$\{[^}]+\}', '', line)
                if self.is_turkish(clean_line):
                    # Find the Turkish text
                    for match in re.finditer(r'>([^<$]+)<', line):
                        text = match.group(1).strip()
                        if text and self.is_turkish(text):
                            findings.append((i, 'template_literal', match.group(0), text))
                continue

            # Regular line scanning
            line_findings = self.extract_translatable_text(line, i)
            findings.extend(line_findings)

        return findings

    def scan_directory(self, directory: Optional[Path] = None) -> Dict[str, List[Tuple[int, str, str, str]]]:
        """Scan all JS files in directory."""
        if directory is None:
            directory = self.root_path

        self.results = {}

        for js_file in directory.rglob('*.js'):
            if any(skip in js_file.parts for skip in self.skip_dirs):
                continue

            findings = self.scan_file(js_file)
            if findings:
                rel_path = str(js_file.relative_to(self.root_path))
                self.results[rel_path] = findings

        return self.results

    def translate_file(self, file_path: str, translator: OllamaTranslator,
                       dry_run: bool = False, check_syntax: bool = False) -> bool:
        """Translate a single JS file."""
        if file_path not in self.results:
            print(f"[SKIP] No Turkish content in {file_path}")
            return True

        full_path = self.root_path / file_path
        findings = self.results[file_path]

        print(f"\n[TRANSLATE JS] {file_path} ({len(findings)} items)", flush=True)

        # Read file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate each finding
        translations = {}
        for i, (line_num, content_type, full_match, text) in enumerate(findings, 1):
            print(f"  [{i}/{len(findings)}] {text[:40]}...", flush=True)

            translated = translator.translate_line(text, context=f"JavaScript {content_type}")
            if translated:
                # CRITICAL: Remove newlines from JS string translations
                # JS single/double quoted strings cannot contain unescaped newlines
                translated_clean = translated.strip().replace('\n', ' ').replace('\r', '')
                # Collapse multiple spaces
                while '  ' in translated_clean:
                    translated_clean = translated_clean.replace('  ', ' ')

                # Preserve the original quote type
                new_match = full_match.replace(text, translated_clean)
                translations[full_match] = new_match

        if dry_run:
            print(f"\n[DRY-RUN] Would modify {file_path}:")
            for original, new in translations.items():
                print(f"  {original[:50]}")
                print(f"  -> {new[:50]}")
        else:
            # Backup
            backup_path = full_path.with_suffix('.js.bak')
            shutil.copy(full_path, backup_path)

            # Apply translations
            for original, new in translations.items():
                content = content.replace(original, new)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Run syntax check if requested
            if check_syntax:
                import subprocess
                try:
                    result = subprocess.run(
                        ['node', '--check', str(full_path)],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        print(f"[SYNTAX ERROR] {file_path}")
                        print(f"  {result.stderr[:200]}")
                        # Restore from backup
                        shutil.copy(backup_path, full_path)
                        print(f"[RESTORE] Restored from backup")
                        return False
                    else:
                        print(f"[OK] Syntax check passed")
                except FileNotFoundError:
                    print(f"[WARN] Node.js not found, skipping syntax check")

            print(f"[OK] Translated {file_path}")

        return True

    def print_summary(self):
        """Print summary of JS findings."""
        if not self.results:
            print("[OK] No Turkish content found in JS files!")
            return

        total = sum(len(f) for f in self.results.values())
        print(f"\n[JS SCAN] Found {total} Turkish items in {len(self.results)} files")

        for file_path, findings in sorted(self.results.items()):
            print(f"\n  {file_path} ({len(findings)} items)")
            for line_num, content_type, full_match, text in findings[:3]:
                print(f"    L{line_num} [{content_type}]: {text[:50]}")
            if len(findings) > 3:
                print(f"    ... and {len(findings) - 3} more")


# ============================================================================
# Python Scanner (Original)
# ============================================================================

class TranslationScanner:
    """Scan Python files for Turkish content."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results: Dict[str, List[Tuple[int, str, str]]] = {}

        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules',
            '.pytest_cache', '.mypy_cache', 'dist', 'build', 'eggs',
            '.eggs', '*.egg-info', 'data', 'logs'
        }

    def is_turkish(self, text: str) -> bool:
        """Check if text contains Turkish content."""
        # Check for Turkish-specific characters
        if TURKISH_CHARS.search(text):
            return True
        # Check for common Turkish words
        if TURKISH_WORD_PATTERN.search(text):
            return True
        return False

    def get_content_type(self, line: str) -> Optional[str]:
        """Determine the type of translatable content."""
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            return None

        # Comment
        if stripped.startswith('#'):
            return 'comment'

        # Docstring (triple quotes)
        if '"""' in stripped or "'''" in stripped:
            return 'docstring'

        # String literals with Turkish content
        if ("'" in stripped or '"' in stripped) and self.is_turkish(stripped):
            # Check if it's a log message, error message, etc.
            if any(kw in stripped.lower() for kw in ['logger.', 'print(', 'raise ', 'error', 'warning']):
                return 'message'
            return 'string'

        return None

    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """
        Scan a single file for Turkish content.

        Returns:
            List of (line_number, content_type, line_text)
        """
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, IOError):
            return findings

        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines, 1):
            # Track docstring state
            if '"""' in line or "'''" in line:
                quote = '"""' if '"""' in line else "'''"
                count = line.count(quote)

                if not in_docstring:
                    in_docstring = True
                    docstring_char = quote
                    if count >= 2:  # Single line docstring
                        in_docstring = False
                elif quote == docstring_char:
                    in_docstring = False

            # Check for Turkish content
            if self.is_turkish(line):
                if in_docstring:
                    content_type = 'docstring'
                else:
                    content_type = self.get_content_type(line) or 'other'

                findings.append((i, content_type, line.rstrip()))

        return findings

    def scan_directory(self, directory: Optional[Path] = None) -> Dict[str, List[Tuple[int, str, str]]]:
        """Scan all Python files in directory."""
        if directory is None:
            directory = self.root_path

        self.results = {}

        for py_file in directory.rglob('*.py'):
            # Skip excluded directories
            if any(skip in py_file.parts for skip in self.skip_dirs):
                continue

            findings = self.scan_file(py_file)
            if findings:
                rel_path = str(py_file.relative_to(self.root_path))
                self.results[rel_path] = findings

        return self.results

    def _apply_translation_preserving_structure(self, original: str, translated: str) -> str:
        """
        Apply translation while preserving the original line's Python structure.

        This preserves:
        - Indentation
        - Docstring delimiters (triple quotes)
        - Comment markers (#)
        - Newline at end

        Args:
            original: Original line from file
            translated: Translated content

        Returns:
            New line with translation applied but structure preserved
        """
        # Get indentation
        indent = len(original) - len(original.lstrip())
        indent_str = original[:indent]

        # Check for newline
        has_newline = original.endswith('\n')
        original_stripped = original.rstrip('\n')
        translated_stripped = translated.rstrip('\n')

        # Get the content part (without indent)
        original_content = original_stripped[indent:]

        # Detect structure elements
        triple_double = '"""'
        triple_single = chr(39)*3
        starts_with_triple_double = original_content.startswith(triple_double)
        starts_with_triple_single = original_content.startswith(triple_single)
        ends_with_triple_double = original_content.rstrip().endswith(triple_double)
        ends_with_triple_single = original_content.rstrip().endswith(triple_single)
        starts_with_comment = original_content.lstrip().startswith('#')

        # Clean translated content (remove any accidental delimiters LLM added)
        trans_clean = translated_stripped.lstrip()

        # Remove triple quotes that LLM might have added
        if trans_clean.startswith(triple_double):
            trans_clean = trans_clean[3:].lstrip()
        if trans_clean.startswith(triple_single):
            trans_clean = trans_clean[3:].lstrip()
        if trans_clean.rstrip().endswith(triple_double):
            trans_clean = trans_clean.rstrip()[:-3].rstrip()
        if trans_clean.rstrip().endswith(triple_single):
            trans_clean = trans_clean.rstrip()[:-3].rstrip()

        # Build new line preserving original structure
        new_content = trans_clean

        # Restore docstring delimiters if original had them
        if starts_with_triple_double:
            new_content = triple_double + new_content
        elif starts_with_triple_single:
            new_content = triple_single + new_content

        if ends_with_triple_double and not new_content.rstrip().endswith(triple_double):
            new_content = new_content.rstrip() + triple_double
        elif ends_with_triple_single and not new_content.rstrip().endswith(triple_single):
            new_content = new_content.rstrip() + triple_single

        # Restore comment marker if original was a comment
        if starts_with_comment and not new_content.lstrip().startswith('#'):
            # Find if there was space after # in original
            comment_match = re.match(r'#(\s*)', original_content.lstrip())
            comment_prefix = '#' + (comment_match.group(1) if comment_match else ' ')
            new_content = comment_prefix + new_content.lstrip()

        # Combine with indent and newline
        result = indent_str + new_content
        if has_newline:
            result = result.rstrip() + '\n'

        return result

    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """
        Check Python syntax using py_compile.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (success, error_message)
        """
        import py_compile
        try:
            py_compile.compile(str(file_path), doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)

    def translate_file(self, file_path: str, translator: OllamaTranslator,
                       dry_run: bool = False, check_syntax: bool = False) -> bool:
        """
        Translate a single file using Ollama.

        Args:
            file_path: Relative path to file
            translator: OllamaTranslator instance
            dry_run: If True, only show what would be changed
            check_syntax: If True, verify Python syntax after translation

        Returns:
            True if successful
        """
        if file_path not in self.results:
            print(f"[SKIP] No Turkish content in {file_path}")
            return True

        full_path = self.root_path / file_path
        findings = self.results[file_path]

        print(f"\n[TRANSLATE] {file_path} ({len(findings)} lines)", flush=True)

        # Read file
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Translate
        def progress(current, total, text):
            # Transliterate Turkish characters to ASCII for clean console output
            turkish_map = {
                'ç': 'c', 'Ç': 'C',
                'ğ': 'g', 'Ğ': 'G',
                'ı': 'i', 'İ': 'I',
                'ö': 'o', 'Ö': 'O',
                'ş': 's', 'Ş': 'S',
                'ü': 'u', 'Ü': 'U'
            }
            short_text = text[:40]
            for tr_char, ascii_char in turkish_map.items():
                short_text = short_text.replace(tr_char, ascii_char)
            # Remove any remaining non-ASCII
            short_text = short_text.encode('ascii', 'ignore').decode('ascii')
            if len(text) > 40:
                short_text += "..."
            print(f"  [{current}/{total}] {short_text}", flush=True)

        translations = translator.translate_file_content(lines, findings, progress)

        if not translations:
            print(f"[ERROR] No translations received for {file_path}")
            return False

        # Apply translations
        modified_lines = lines.copy()
        for line_num, translated in translations.items():
            idx = line_num - 1
            original = lines[idx]

            # Preserve original line structure (docstring delimiters, etc.)
            new_line = self._apply_translation_preserving_structure(original, translated)
            modified_lines[idx] = new_line

        if dry_run:
            print(f"\n[DRY-RUN] Would modify {file_path}:")
            for line_num, translated in translations.items():
                print(f"  L{line_num}: {lines[line_num-1].strip()[:50]}")
                print(f"     -> {translated.strip()[:50]}")
        else:
            # Backup original
            backup_path = full_path.with_suffix('.py.bak')
            shutil.copy(full_path, backup_path)

            # Write translated
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)

            print(f"[OK] Translated {file_path} (backup: {backup_path.name})")

            # Syntax check if requested
            if check_syntax:
                syntax_ok, error_msg = self.check_syntax(full_path)
                if syntax_ok:
                    print(f"[SYNTAX OK] {file_path}")
                else:
                    print(f"[SYNTAX ERROR] {file_path}")
                    print(f"  {error_msg}")
                    # Don't restore - keep the translated file for manual fix
                    return False

        return True

    def translate_all(self, translator: OllamaTranslator, dry_run: bool = False,
                      limit: int = None, check_syntax: bool = False) -> Dict[str, bool]:
        """
        Translate all files with Turkish content.

        Args:
            translator: OllamaTranslator instance
            dry_run: If True, only show what would be changed
            limit: Max number of files to translate (None = all)
            check_syntax: If True, verify Python syntax after each translation

        Returns:
            Dict mapping file_path to success status
        """
        results = {}
        files = list(self.results.keys())

        if limit:
            files = files[:limit]

        total = len(files)
        print(f"\n[START] Translating {total} files...", flush=True)
        if check_syntax:
            print("[INFO] Syntax checking enabled (--check)", flush=True)

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{total}] Processing {file_path}", flush=True)
            success = self.translate_file(file_path, translator, dry_run, check_syntax)
            results[file_path] = success

            if not success:
                print(f"[WARN] Failed to translate {file_path}, continuing...")

        # Summary
        success_count = sum(1 for v in results.values() if v)
        failed_files = [f for f, success in results.items() if not success]

        print(f"\n{'='*60}")
        print(f"[DONE] Translated {success_count}/{total} files")

        if failed_files:
            print(f"\n[SYNTAX ERRORS] {len(failed_files)} files have syntax errors:")
            for f in failed_files:
                full_path = self.root_path / f
                print(f"  - {f}")
                # Show the error again
                syntax_ok, error_msg = self.check_syntax(full_path)
                if not syntax_ok:
                    # Extract line number from error
                    print(f"    {error_msg}")
            print(f"\n[TIP] Backup files (.py.bak) are available for manual restore if needed")

        return results

    def generate_report(self) -> str:
        """Generate markdown report of findings."""
        if not self.results:
            return "No Turkish content found."

        lines = [
            "# Translation Report",
            "",
            f"**Files with Turkish content:** {len(self.results)}",
            f"**Total lines to translate:** {sum(len(f) for f in self.results.values())}",
            "",
            "---",
            ""
        ]

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| File | Lines | Types |")
        lines.append("|------|-------|-------|")

        for file_path, findings in sorted(self.results.items()):
            types = set(f[1] for f in findings)
            types_str = ", ".join(sorted(types))
            lines.append(f"| `{file_path}` | {len(findings)} | {types_str} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Detailed findings
        lines.append("## Detailed Findings")
        lines.append("")

        for file_path, findings in sorted(self.results.items()):
            lines.append(f"### {file_path}")
            lines.append("")
            lines.append("```")
            for line_num, content_type, text in findings:
                # Truncate long lines
                display_text = text[:100] + "..." if len(text) > 100 else text
                lines.append(f"L{line_num:4d} [{content_type:8s}] {display_text}")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def generate_translation_prompt(self, file_path: str) -> str:
        """Generate a translation prompt for a specific file."""
        if file_path not in self.results:
            return f"No Turkish content found in {file_path}"

        findings = self.results[file_path]

        lines = [
            f"# Translation Task: {file_path}",
            "",
            "Translate the following Turkish content to English:",
            "",
            "## Lines to translate:",
            ""
        ]

        for line_num, content_type, text in findings:
            lines.append(f"**Line {line_num}** ({content_type}):")
            lines.append(f"```")
            lines.append(text)
            lines.append(f"```")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Rules:",
            "1. Only translate comments, docstrings, and string literals",
            "2. Keep code logic, variable names, function names unchanged",
            "3. Keep technical terms in English (API, JSON, config, etc.)",
            "4. Preserve emojis",
            "5. Maintain docstring format (Args/Returns)",
        ])

        return "\n".join(lines)

    def print_summary(self):
        """Print summary to console."""
        if not self.results:
            print("[OK] No Turkish content found!")
            return

        total_lines = sum(len(f) for f in self.results.values())

        print(f"\n{'='*60}")
        print(f"[SCAN] Translation Scanner Results")
        print(f"{'='*60}")
        print(f"\n[FILES] Files with Turkish content: {len(self.results)}")
        print(f"[LINES] Total lines to translate: {total_lines}")
        print(f"\n{'-'*60}")

        for file_path, findings in sorted(self.results.items()):
            types = set(f[1] for f in findings)
            print(f"\n[FILE] {file_path}")
            print(f"   Lines: {len(findings)} | Types: {', '.join(sorted(types))}")

            # Show first 3 examples
            for line_num, content_type, text in findings[:3]:
                short_text = text.strip()[:60]
                if len(text.strip()) > 60:
                    short_text += "..."
                print(f"   L{line_num}: {short_text}")

            if len(findings) > 3:
                print(f"   ... and {len(findings) - 3} more lines")

        print(f"\n{'='*60}")


def main():
    """Main entry point."""
    import argparse
    import sys
    import io

    # Force UTF-8 output for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Scan and translate Turkish content in Python/HTML/JS files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan Python files (default)
  python translate_scanner.py modules/

  # Scan HTML files
  python translate_scanner.py modules/webui/templates/ --html

  # Scan JavaScript files
  python translate_scanner.py modules/webui/static/js/ --js

  # Translate HTML files with dry-run
  python translate_scanner.py modules/webui/templates/ --html --translate --dry-run

  # Translate both HTML and JS
  python translate_scanner.py modules/webui/ --html --js --translate
        """
    )
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--prompt", help="Generate translation prompt for specific file")

    # File type flags
    parser.add_argument("--html", action="store_true", help="Scan HTML/Jinja2 template files")
    parser.add_argument("--js", action="store_true", help="Scan JavaScript files")
    parser.add_argument("--py", action="store_true", help="Scan Python files (default if no flag specified)")

    # Translation options
    parser.add_argument("--translate", action="store_true", help="Auto-translate using Ollama")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be translated without making changes")
    parser.add_argument("--ollama-url", default="http://192.168.1.195:11434", help="Ollama API URL")
    parser.add_argument("--model", default="translategemma:12b", help="Ollama model to use")
    parser.add_argument("--limit", type=int, help="Limit number of files to translate")
    parser.add_argument("--file", help="Translate only this specific file")
    parser.add_argument("--check", action="store_true", help="Run py_compile to verify Python syntax after translation")

    args = parser.parse_args()

    # If no file type flag specified, default to Python
    if not args.html and not args.js and not args.py:
        args.py = True

    # Setup translator if needed
    translator = None
    if args.translate:
        if not REQUESTS_AVAILABLE:
            print("[ERROR] requests library required for translation. Install: pip install requests")
            sys.exit(1)

        config = OllamaConfig(
            base_url=args.ollama_url,
            model=args.model
        )
        translator = OllamaTranslator(config)

        print(f"[CHECK] Connecting to Ollama at {config.base_url}...")
        if not translator.check_connection():
            print(f"[ERROR] Cannot connect to Ollama at {config.base_url}")
            print("Make sure Ollama is running and accessible.")
            sys.exit(1)
        print(f"[OK] Connected to Ollama (model: {config.model})")

    # =========================================================================
    # HTML Mode
    # =========================================================================
    if args.html:
        print(f"\n{'='*60}")
        print("[HTML] Scanning HTML/Jinja2 templates...")
        print(f"{'='*60}")

        if Path(args.path).is_file() and args.path.endswith('.html'):
            # Single file mode - use parent as root
            file_path = Path(args.path)
            html_scanner = HTMLScanner(file_path.parent)
            findings = html_scanner.scan_file(file_path)
            if findings:
                html_scanner.results[file_path.name] = findings
        else:
            # Directory mode
            html_scanner = HTMLScanner(args.path)
            html_scanner.scan_directory()

        if args.translate and translator and html_scanner.results:
            total = len(html_scanner.results)
            files = list(html_scanner.results.keys())
            if args.limit:
                files = files[:args.limit]

            print(f"\n[START] Translating {len(files)} HTML files...", flush=True)
            for i, file_path in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}] {file_path}", flush=True)
                html_scanner.translate_file(file_path, translator, args.dry_run)
        else:
            html_scanner.print_summary()

    # =========================================================================
    # JavaScript Mode
    # =========================================================================
    if args.js:
        print(f"\n{'='*60}")
        print("[JS] Scanning JavaScript files...")
        print(f"{'='*60}")

        if Path(args.path).is_file() and args.path.endswith('.js'):
            # Single file mode - use parent as root
            file_path = Path(args.path)
            js_scanner = JSScanner(file_path.parent)
            findings = js_scanner.scan_file(file_path)
            if findings:
                js_scanner.results[file_path.name] = findings
        else:
            # Directory mode
            js_scanner = JSScanner(args.path)
            js_scanner.scan_directory()

        if args.translate and translator and js_scanner.results:
            total = len(js_scanner.results)
            files = list(js_scanner.results.keys())
            if args.limit:
                files = files[:args.limit]

            print(f"\n[START] Translating {len(files)} JS files...", flush=True)
            for i, file_path in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}] {file_path}", flush=True)
                js_scanner.translate_file(file_path, translator, args.dry_run, args.check)
        else:
            js_scanner.print_summary()

    # =========================================================================
    # Python Mode (Original)
    # =========================================================================
    if args.py:
        print(f"\n{'='*60}")
        print("[PY] Scanning Python files...")
        print(f"{'='*60}")

        # Original Python scanning logic
        if args.file:
            file_path = Path(args.file)
            if file_path.is_absolute():
                scanner = TranslationScanner(file_path.parent)
                findings = scanner.scan_file(file_path)
                if findings:
                    scanner.results[file_path.name] = findings
            else:
                scanner = TranslationScanner(args.path)
                full_path = Path(args.path) / file_path
                findings = scanner.scan_file(full_path)
                if findings:
                    scanner.results[str(file_path)] = findings
        elif Path(args.path).is_file():
            file_path = Path(args.path)
            scanner = TranslationScanner(file_path.parent)
            findings = scanner.scan_file(file_path)
            if findings:
                scanner.results[file_path.name] = findings
        else:
            scanner = TranslationScanner(args.path)
            scanner.scan_directory()

        if args.translate and translator:
            if scanner.results:
                scanner.translate_all(translator, args.dry_run, args.limit, args.check)
            else:
                print("[INFO] No Turkish content found in Python files")
        elif args.prompt:
            print(scanner.generate_translation_prompt(args.prompt))
        elif args.report:
            report = scanner.generate_report()
            if args.output:
                Path(args.output).write_text(report, encoding='utf-8')
                print(f"[OK] Report saved to {args.output}")
            else:
                print(report)
        else:
            scanner.print_summary()

    print(f"\n{'='*60}")
    print("[DONE] Scan complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
