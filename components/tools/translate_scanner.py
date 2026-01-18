#!/usr/bin/env python3
"""
Translation Scanner - Find and Translate Turkish content in Python files

Scans Python files for Turkish content (comments, docstrings, strings)
and can auto-translate using Ollama (local LLM).

Usage:
    python components/tools/translate_scanner.py                    # Scan all
    python components/tools/translate_scanner.py core/              # Scan specific folder
    python components/tools/translate_scanner.py --report           # Generate markdown report
    python components/tools/translate_scanner.py --translate        # Auto-translate with Ollama
    python components/tools/translate_scanner.py --translate --dry-run  # Preview changes
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
    r'\b(ve|veya|için|ile|olan|olarak|gibi|kadar|sonra|önce)\b',
    r'\b(değil|var|yok|evet|hayır|tamam|lütfen)\b',
    r'\b(dosya|klasör|dizin|ayar|yapılandırma)\b',
    r'\b(başarılı|başarısız|hata|uyarı|bilgi)\b',
    r'\b(yükleniyor|kaydediliyor|siliniyor|güncelleniyor)\b',
    r'\b(bulunamadı|zaten|mevcut|gerekli|zorunlu)\b',
]
TURKISH_WORD_PATTERN = re.compile('|'.join(TURKISH_WORDS), re.IGNORECASE)


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    base_url: str = "http://192.168.1.195:11434"
    model: str = "llama3.2"  # or qwen2.5, mistral, etc.
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
Rules:
- Only translate the Turkish parts
- Keep code syntax unchanged (variable names, function names, etc.)
- Keep emojis unchanged
- Keep technical terms in English (API, JSON, config, WebSocket, etc.)
- If it's a docstring, maintain Args/Returns format
- Return ONLY the translated text, nothing else

Context: {context}

Turkish text:
{turkish_text}

English translation:"""

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
                    return data["message"].get("content", "").strip()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()

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
                return data.get("response", "").strip()

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
                return data.get("response", "").strip()

            print(f"[ERROR] Ollama API error: {response.status_code}")
            return None

        except requests.exceptions.Timeout:
            print("[ERROR] Ollama API timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Ollama API error: {e}")
            return None

    def translate_file_content(self, lines: List[str], findings: List[Tuple[int, str, str]],
                                progress_callback=None) -> Dict[int, str]:
        """
        Translate multiple lines in a file.

        Args:
            lines: All lines in the file
            findings: List of (line_number, content_type, text) to translate
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping line_number to translated text
        """
        translations = {}
        total = len(findings)

        for i, (line_num, content_type, text) in enumerate(findings, 1):
            if progress_callback:
                progress_callback(i, total, text[:50])

            translated = self.translate_line(text, context=content_type)
            if translated:
                translations[line_num] = translated

            # Small delay to not overload the API
            time.sleep(0.1)

        return translations

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

    def translate_file(self, file_path: str, translator: OllamaTranslator,
                       dry_run: bool = False) -> bool:
        """
        Translate a single file using Ollama.

        Args:
            file_path: Relative path to file
            translator: OllamaTranslator instance
            dry_run: If True, only show what would be changed

        Returns:
            True if successful
        """
        if file_path not in self.results:
            print(f"[SKIP] No Turkish content in {file_path}")
            return True

        full_path = self.root_path / file_path
        findings = self.results[file_path]

        print(f"\n[TRANSLATE] {file_path} ({len(findings)} lines)")

        # Read file
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Translate
        def progress(current, total, text):
            # Remove non-ASCII for console output
            short_text = text[:40].encode('ascii', 'replace').decode('ascii')
            if len(text) > 40:
                short_text += "..."
            print(f"  [{current}/{total}] {short_text}")

        translations = translator.translate_file_content(lines, findings, progress)

        if not translations:
            print(f"[ERROR] No translations received for {file_path}")
            return False

        # Apply translations
        modified_lines = lines.copy()
        for line_num, translated in translations.items():
            idx = line_num - 1
            original = lines[idx]

            # Preserve indentation
            indent = len(original) - len(original.lstrip())
            indent_str = original[:indent]

            # Add newline if original had one
            if original.endswith('\n'):
                translated = translated.rstrip() + '\n'

            modified_lines[idx] = indent_str + translated.lstrip()

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

        return True

    def translate_all(self, translator: OllamaTranslator, dry_run: bool = False,
                      limit: int = None) -> Dict[str, bool]:
        """
        Translate all files with Turkish content.

        Args:
            translator: OllamaTranslator instance
            dry_run: If True, only show what would be changed
            limit: Max number of files to translate (None = all)

        Returns:
            Dict mapping file_path to success status
        """
        results = {}
        files = list(self.results.keys())

        if limit:
            files = files[:limit]

        total = len(files)
        print(f"\n[START] Translating {total} files...")

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{total}] Processing {file_path}")
            success = self.translate_file(file_path, translator, dry_run)
            results[file_path] = success

            if not success:
                print(f"[WARN] Failed to translate {file_path}, continuing...")

        # Summary
        success_count = sum(1 for v in results.values() if v)
        print(f"\n[DONE] Translated {success_count}/{total} files")

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
        import sys
        import io

        # Force UTF-8 output
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

    parser = argparse.ArgumentParser(description="Scan and translate Turkish content in Python files")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--prompt", help="Generate translation prompt for specific file")

    # Translation options
    parser.add_argument("--translate", action="store_true", help="Auto-translate using Ollama")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be translated without making changes")
    parser.add_argument("--ollama-url", default="http://192.168.1.195:11434", help="Ollama API URL")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--limit", type=int, help="Limit number of files to translate")
    parser.add_argument("--file", help="Translate only this specific file")

    args = parser.parse_args()

    # Determine root path
    if Path(args.path).is_file():
        scanner = TranslationScanner(Path(args.path).parent)
        scanner.results[args.path] = scanner.scan_file(Path(args.path))
    else:
        scanner = TranslationScanner(args.path)
        scanner.scan_directory()

    # Translation mode
    if args.translate:
        if not REQUESTS_AVAILABLE:
            print("[ERROR] requests library required for translation. Install: pip install requests")
            sys.exit(1)

        config = OllamaConfig(
            base_url=args.ollama_url,
            model=args.model
        )
        translator = OllamaTranslator(config)

        # Check connection
        print(f"[CHECK] Connecting to Ollama at {config.base_url}...")
        if not translator.check_connection():
            print(f"[ERROR] Cannot connect to Ollama at {config.base_url}")
            print("Make sure Ollama is running and accessible.")
            sys.exit(1)
        print(f"[OK] Connected to Ollama (model: {config.model})")

        if args.file:
            # Translate single file
            scanner.translate_file(args.file, translator, args.dry_run)
        else:
            # Translate all
            scanner.translate_all(translator, args.dry_run, args.limit)

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


if __name__ == "__main__":
    main()
