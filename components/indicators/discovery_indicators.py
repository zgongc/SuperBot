#!/usr/bin/env python3
"""
components/indicators/discovery_indicators.py
SuperBot - Indicator Discovery & Registry Generator

Görev:
    Indicator dosyalarını tara ve components/indicators/__init__.py oluştur

Auto-detect:
    1. Class name
    2. __init__ params → default_params
    3. calculate_batch() var mı → has_calculate_batch
    4. calculate() return → output_type, output_keys
    5. requires_volume

Kullanım:
    python components/indicators/discovery_indicators.py

Output:
    components/indicators/__init__.py (INDICATOR_REGISTRY)
"""

import re
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

CATEGORY_TO_ENUM = {
    'momentum': 'IndicatorCategory.MOMENTUM',
    'trend': 'IndicatorCategory.TREND',
    'volatility': 'IndicatorCategory.VOLATILITY',
    'volume': 'IndicatorCategory.VOLUME',
    'support_resistance': 'IndicatorCategory.SUPPORT_RESISTANCE',
    'structure': 'IndicatorCategory.STRUCTURE',
    'breakout': 'IndicatorCategory.BREAKOUT',
    'statistical': 'IndicatorCategory.STATISTICAL',
    'combo': 'IndicatorCategory.COMBO',
    'patterns': 'IndicatorCategory.PATTERNS',
}


def detect_output_keys(content: str) -> list:
    """
    Output keys'leri detect et (return value içindeki dict keys)

    Strategy:
    1. calculate_batch() return DataFrame'deki column names (PRIMARY)
    2. calculate() return IndicatorResult value dict keys (FALLBACK)

    Returns:
        List of output key names (örn: ['upper', 'middle', 'lower'])
    """
    keys = []

    # ÖNCE calculate_batch() return DataFrame/Series column names'i al (EN DOĞRU)
    if 'calculate_batch' in content:
        # Simple but robust: Split by function definitions, find calculate_batch section
        # Split by 'def ' to get function blocks
        func_blocks = re.split(r'\n    def ', content)
        calculate_batch_block = None
        for block in func_blocks:
            if block.startswith('calculate_batch('):
                calculate_batch_block = block
                break

        if calculate_batch_block:
            # Case 1: return pd.DataFrame({ ... }) - Multiple columns
            in_dataframe = False
            dataframe_lines = []
            for line in calculate_batch_block.split('\n'):
                if 'return pd.DataFrame' in line and '{' in line:
                    in_dataframe = True
                if in_dataframe:
                    dataframe_lines.append(line)
                    if '}' in line and ('index=' in line or line.strip().endswith('})') or line.strip().endswith('})')):
                        break

            if dataframe_lines:
                df_section = '\n'.join(dataframe_lines)
                # Find all 'key': value or "key": value patterns
                col_matches = re.findall(r"['\"]([A-Za-z0-9_.]+)['\"]:\s*", df_section)
                keys.extend(col_matches)

            # Case 2: result_df['column'] or df_result['column'] assignment pattern
            if not keys:
                # Find patterns like: result_df['upper'] = ... or df_result['ms_bos_value'] = ...
                df_assignment_matches = re.findall(r"(?:result_df|df_result)\[\'([A-Za-z0-9_.]+)\'\]", calculate_batch_block)
                if df_assignment_matches:
                    keys.extend(df_assignment_matches)

            # Case 2b: Dynamic f-string pattern like result_df[f'ema_{period}']
            if not keys:
                # Find pattern: result_df[f'prefix_{variable}'] with for loop
                fstring_match = re.search(r"result_df\[f['\"]([A-Za-z0-9_]+)_\{([A-Za-z0-9_]+)\}", calculate_batch_block)
                if fstring_match:
                    prefix = fstring_match.group(1)  # e.g., 'ema'
                    var_name = fstring_match.group(2)  # e.g., 'period'

                    # Find the loop variable source: for period in self.xxx_periods
                    loop_match = re.search(rf'for\s+{var_name}\s+in\s+self\.([A-Za-z0-9_]+)', calculate_batch_block)
                    if loop_match:
                        list_attr = loop_match.group(1)  # e.g., 'ema_periods'

                        # Try to find default value in __init__
                        init_pattern = rf'{list_attr}\s*=\s*\[([0-9,\s]+)\]'
                        init_match = re.search(init_pattern, content)
                        if init_match:
                            # Extract numbers from list
                            numbers = [int(n.strip()) for n in init_match.group(1).split(',')]
                            # Generate column names
                            for num in numbers:
                                keys.append(f'{prefix}_{num}')

            # Case 3: return pd.Series(..., name='xxx') - Single value
            if not keys and 'return pd.Series' in calculate_batch_block:
                series_match = re.search(r'return\s+pd\.Series\([^)]*name=[\'"]([A-Za-z0-9_.]+)[\'"]', calculate_batch_block)
                if series_match:
                    keys.append(series_match.group(1))
                # If Series has no name, it's likely a single-value indicator
                # We'll leave keys empty here and handle it via fallback

    # FALLBACK: calculate() return IndicatorResult(value={...})
    if not keys:
        # return IndicatorResult(value={...}) içindeki keys'leri bul
        value_dict_matches = re.finditer(r'value=\s*\{([^}]+)\}', content, re.DOTALL)
        for value_match in value_dict_matches:
            value_content = value_match.group(1)
            # 'upper': ..., 'middle': ..., 'lower': ... pattern'i bul
            key_matches = re.findall(r"['\"]([A-Za-z0-9_.]+)['\"]:\s*", value_content)
            keys.extend(key_matches)

    # Duplicate'leri kaldır, sırala
    return sorted(list(set(keys))) if keys else []


def detect_indicator_type(class_name: str, content: str) -> str:
    """
    Indicator type'ını otomatik detect et

    Returns:
        IndicatorType enum string (örn: 'IndicatorType.BANDS')
    """
    name_lower = class_name.lower()

    # BANDS: Bollinger, Keltner, Donchian
    if any(x in name_lower for x in ['bollinger', 'keltner', 'donchian', 'envelope']):
        return 'IndicatorType.BANDS'

    # LINES: Ichimoku, Moving Averages
    if 'ichimoku' in name_lower:
        return 'IndicatorType.LINES'

    # LEVELS: Pivot Points, Fibonacci
    if any(x in name_lower for x in ['pivot', 'fibonacci', 'camarilla']):
        return 'IndicatorType.LEVELS'

    # ZONES: FVG, Order Blocks, Liquidity
    if any(x in name_lower for x in ['fvg', 'orderblock', 'liquidity', 'zone']):
        return 'IndicatorType.ZONES'

    # MULTIPLE_VALUES: MACD, Stochastic, gibi birden fazla output
    # Check if indicator returns dict with multiple keys
    if any(x in name_lower for x in ['macd', 'stochastic', 'stoch', 'adx', 'aroon', 'ppo', 'cci_dvg']):
        return 'IndicatorType.MULTIPLE_VALUES'

    # Check return statement içinde dict var mı
    if 'values={' in content or 'values = {' in content:
        return 'IndicatorType.MULTIPLE_VALUES'

    # Default: SINGLE_VALUE (RSI, ATR, EMA, etc.)
    return 'IndicatorType.SINGLE_VALUE'


def parse_indicator_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Indicator dosyasını parse et ve metadata çıkar

    Returns:
        {
            'class_name': 'BollingerBands',
            'default_params': {'period': 20, 'std_dev': 2.0},
            'has_calculate_batch': True,
            'requires_volume': False,
            'description': 'Bollinger Bands',
        }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            'class_name': None,
            'default_params': {},
            'has_calculate_batch': False,
            'requires_volume': False,
            'description': '',
            'indicator_type': None,
            'output_keys': [],
        }

        # 1. Class name bul
        class_match = re.search(r'class\s+(\w+)\(BaseIndicator\)', content)
        if not class_match:
            return None
        metadata['class_name'] = class_match.group(1)

        # 2. Description bul (docstring'den)
        desc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if desc_match:
            desc_lines = desc_match.group(1).strip().split('\n')
            # İlk satır veya ikinci satırı al (genelde açıklama)
            if desc_lines:
                first_line = desc_lines[0].strip()

                # Format 1: "indicators/path/file.py - Description" (eski format)
                if ' - ' in first_line and first_line.startswith('indicators/'):
                    desc = first_line.split(' - ', 1)[1].strip()

                # Format 2: Çok satırlı header (yeni format)
                # Satır 1: indicators/path/file.py
                # Satır 2: SuperBot - X (Full Name)
                elif first_line.startswith('indicators/') and len(desc_lines) > 1:
                    second_line = desc_lines[1].strip()
                    # "SuperBot - APO (Absolute Price Oscillator)" -> "APO (Absolute Price Oscillator)"
                    if second_line.startswith('SuperBot -'):
                        desc = second_line.replace('SuperBot - ', '').strip()
                    else:
                        # Fallback: ilk satırdan çıkar
                        desc = first_line.split('/')[-1].replace('.py', '').replace('_', ' ').title()

                # Format 3: Direkt açıklama (standart format)
                else:
                    desc = first_line

                metadata['description'] = desc

        # 3. __init__ params bul (multi-line support)
        init_match = re.search(
            r'def\s+__init__\s*\(\s*self\s*,\s*(.*?)\)\s*:',
            content,
            re.DOTALL
        )
        if init_match:
            params_str = init_match.group(1)
            # Clean up whitespace and newlines
            params_str = ' '.join(params_str.split())

            # Parse params: period: int = 14, std_dev: float = 2.0
            for param in params_str.split(','):
                param = param.strip()
                if '=' in param:
                    # Extract: period: int = 14 → ('period', 14)
                    parts = param.split('=')
                    param_name = parts[0].split(':')[0].strip()
                    param_value = parts[1].strip()

                    # Skip special params
                    if param_name in ['logger', 'error_handler', 'self']:
                        continue

                    # Parse value
                    try:
                        # Try eval for simple values
                        metadata['default_params'][param_name] = ast.literal_eval(param_value)
                    except:
                        # String fallback
                        metadata['default_params'][param_name] = param_value.strip("'\"")

        # 4. calculate_batch() var mı?
        metadata['has_calculate_batch'] = 'def calculate_batch(' in content

        # 5. requires_volume check
        # İlk olarak _requires_volume() methodunu kontrol et
        requires_match = re.search(r'def\s+_requires_volume\(.*?\).*?return\s+(True|False)', content, re.DOTALL)
        if requires_match:
            metadata['requires_volume'] = requires_match.group(1) == 'True'
        else:
            # Fallback: volume keyword'ü var mı? (data['volume'] gibi)
            metadata['requires_volume'] = "data['volume']" in content or 'data["volume"]' in content

        # 6. Indicator type detect et
        metadata['indicator_type'] = detect_indicator_type(metadata['class_name'], content)

        # 7. Output keys detect et (return value içindeki dict keys)
        metadata['output_keys'] = detect_output_keys(content)

        return metadata

    except Exception as e:
        print(f"   HATA {file_path.name}: {e}")
        return None


def discover_all_indicators(indicators_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Tüm indicator'ları tara

    Returns:
        {
            'rsi': {
                'module': 'momentum.rsi',
                'class': 'RSI',
                'category': 'momentum',
                'description': 'Relative Strength Index',
                'default_params': {'period': 14},
                'has_calculate_batch': True,
                'requires_volume': False,
            },
            ...
        }
    """
    registry = {}

    # Kategorileri tara
    categories = [d for d in indicators_path.iterdir()
                  if d.is_dir() and not d.name.startswith('_')]

    print(f"\n[*] {len(categories)} kategori bulundu")

    for category_path in sorted(categories):
        category = category_path.name
        print(f"\n[{category}] taranıyor...")

        count = 0
        for file_path in sorted(category_path.glob('*.py')):
            if file_path.name.startswith('_'):
                continue

            name = file_path.stem

            # Parse file
            metadata = parse_indicator_file(file_path)
            if not metadata or not metadata['class_name']:
                print(f"   [!] {name}: Class bulunamadı")
                continue

            # Registry'ye ekle (enum format)
            registry[name] = {
                'module': f'{category}.{name}',
                'class': metadata['class_name'],
                'category': CATEGORY_TO_ENUM.get(category, f"'{category}'"),
                'type': metadata['indicator_type'],
                'description': metadata['description'] or name.replace('_', ' ').title(),
                'default_params': metadata['default_params'],
                'requires_volume': metadata['requires_volume'],
                'has_calculate_batch': metadata['has_calculate_batch'],
                'output_keys': metadata['output_keys'],
            }

            batch_icon = "[BATCH]" if metadata['has_calculate_batch'] else "      "
            volume_icon = "[VOL]" if metadata['requires_volume'] else "     "
            print(f"   {batch_icon} {volume_icon} {name:20s} -> {metadata['class_name']}")
            count += 1

        print(f"   [OK] {count} indicator")

    return registry


def generate_init_file(registry: Dict[str, Dict[str, Any]], output_path: Path):
    """components/indicators/__init__.py oluştur"""

    # Kategoriye göre grupla
    by_category = {}
    for name, info in registry.items():
        cat = info['category']
        if cat not in by_category:
            by_category[cat] = {}
        by_category[cat][name] = info

    # Generate content
    lines = [
        '#!/usr/bin/env python3',
        '"""',
        'components/indicators/__init__.py',
        'SuperBot - Indicator Registry',
        '',
        'AUTO-GENERATED by discovery_indicators.py - DO NOT EDIT MANUALLY',
        '',
        f'Total Indicators: {len(registry)}',
        f'Categories: {len(by_category)}',
        '',
        'Kullanım:',
        '    from components.indicators import INDICATOR_REGISTRY, get_indicator_class',
        '    ',
        '    # Get indicator class',
        '    RSI = get_indicator_class("rsi")',
        '    rsi = RSI(period=14)',
        '"""',
        '',
        'from typing import Dict, Any, List, Type',
        'import importlib',
        'from pathlib import Path',
        '',
        '# Import enums',
        'from components.indicators.indicator_types import IndicatorCategory, IndicatorType',
        '',
        '',
        '# ============================================================================',
        '# INDICATOR REGISTRY',
        '# ============================================================================',
        '',
        f'INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {{',
        '',
    ]

    # Her kategori
    for category in sorted(by_category.keys()):
        indicators = by_category[category]

        lines.append(f'    # {category.upper()} ({len(indicators)} indicators)')
        lines.append('')

        for name in sorted(indicators.keys()):
            info = indicators[name]

            # Escape single quotes in description
            description = info['description'].replace("'", "\\'")

            lines.append(f"    '{name}': {{")
            lines.append(f"        'module': '{info['module']}',")
            lines.append(f"        'class': '{info['class']}',")
            lines.append(f"        'category': {info['category']},")
            lines.append(f"        'type': {info['type']},")
            lines.append(f"        'description': '{description}',")
            lines.append(f"        'default_params': {info['default_params']},")
            lines.append(f"        'requires_volume': {info['requires_volume']},")
            lines.append(f"        'has_calculate_batch': {info['has_calculate_batch']},")
            lines.append(f"        'output_keys': {info['output_keys']}")
            lines.append(f"    }},")
            lines.append('')

    lines.append('}')
    lines.append('')
    lines.append('')

    # Helper functions
    lines.extend([
        '# ============================================================================',
        '# HELPER FUNCTIONS',
        '# ============================================================================',
        '',
        'def get_indicator_class(name: str) -> Type:',
        '    """',
        '    Indicator class\'ını import et',
        '    ',
        '    Args:',
        '        name: Indicator adı (örn: \'rsi\', \'bollinger\')',
        '    ',
        '    Returns:',
        '        Indicator class',
        '    """',
        '    if name not in INDICATOR_REGISTRY:',
        '        available = list(INDICATOR_REGISTRY.keys())',
        '        raise ValueError(',
        '            f"Indicator \'{name}\' bulunamadı!\\n"',
        '            f"Mevcut: {len(available)} indicator"',
        '        )',
        '    ',
        '    info = INDICATOR_REGISTRY[name]',
        '    ',
        '    # sys.modules hack - Fix "from indicators.X import" errors',
        '    import sys',
        '    import components.indicators',
        '    import components.indicators.base_indicator',
        '    import components.indicators.indicator_types',
        '    sys.modules[\'indicators\'] = sys.modules[\'components.indicators\']',
        '    sys.modules[\'indicators.base_indicator\'] = sys.modules[\'components.indicators.base_indicator\']',
        '    sys.modules[\'indicators.indicator_types\'] = sys.modules[\'components.indicators.indicator_types\']',
        '    ',
        '    # Import module',
        '    module_path = f"components.indicators.{info[\'module\']}"',
        '    module = importlib.import_module(module_path)',
        '    ',
        '    return getattr(module, info[\'class\'])',
        '',
        '',
        'def list_indicators(category: str = None) -> List[str]:',
        '    """Indicator listesi"""',
        '    if category:',
        '        return [',
        '            name for name, info in INDICATOR_REGISTRY.items()',
        '            if info[\'category\'] == category',
        '        ]',
        '    return list(INDICATOR_REGISTRY.keys())',
        '',
        '',
        'def get_batch_capable_indicators() -> List[str]:',
        '    """calculate_batch() olan indicator\'lar"""',
        '    return [',
        '        name for name, info in INDICATOR_REGISTRY.items()',
        '        if info[\'has_calculate_batch\']',
        '    ]',
        '',
        '',
        'def get_categories() -> List[str]:',
        '    """Kategoriler"""',
        '    categories = set(str(info[\'category\']).split(\'.\')[-1] for info in INDICATOR_REGISTRY.values())',
        '    return sorted(categories)',
        '',
        '',
        'def get_indicator_info(name: str) -> Dict[str, Any]:',
        '    """',
        '    Get indicator information',
        '    ',
        '    Args:',
        '        name: Indicator name',
        '    ',
        '    Returns:',
        '        Dict with indicator info',
        '    """',
        '    if name not in INDICATOR_REGISTRY:',
        '        raise ValueError(f"Unknown indicator: \'{name}\'")',
        '    ',
        '    return INDICATOR_REGISTRY[name].copy()',
        '',
        '',
        'def get_registry_statistics() -> Dict[str, Any]:',
        '    """',
        '    Get registry statistics',
        '    ',
        '    Returns:',
        '        Dict with statistics',
        '    """',
        '    total = len(INDICATOR_REGISTRY)',
        '    by_category = {}',
        '    requires_volume = 0',
        '    batch_capable = 0',
        '    ',
        '    for info in INDICATOR_REGISTRY.values():',
        '        # Category count',
        '        cat = info[\'category\']',
        '        cat_str = str(cat).split(\'.\')[-1]  # IndicatorCategory.MOMENTUM -> MOMENTUM',
        '        by_category[cat_str] = by_category.get(cat_str, 0) + 1',
        '        ',
        '        # Volume requirement',
        '        if info[\'requires_volume\']:',
        '            requires_volume += 1',
        '        ',
        '        # Batch capable',
        '        if info.get(\'has_calculate_batch\', False):',
        '            batch_capable += 1',
        '    ',
        '    return {',
        '        \'total_indicators\': total,',
        '        \'by_category\': by_category,',
        '        \'requires_volume\': requires_volume,',
        '        \'batch_capable\': batch_capable,',
        '        \'categories\': get_categories()',
        '    }',
        '',
        '',
        '__all__ = [',
        '    \'INDICATOR_REGISTRY\',',
        '    \'get_indicator_class\',',
        '    \'list_indicators\',',
        '    \'get_batch_capable_indicators\',',
        '    \'get_categories\',',
        '    \'get_indicator_info\',',
        '    \'get_registry_statistics\',',
        ']',
        '',
    ])

    # Yaz
    content = '\n'.join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n[OK] {output_path} oluşturuldu!")
    print(f"   [*] {len(registry)} indicator")
    print(f"   [*] {len(by_category)} kategori")

    # Stats
    batch_count = sum(1 for info in registry.values() if info['has_calculate_batch'])
    volume_count = sum(1 for info in registry.values() if info['requires_volume'])

    print(f"   [BATCH] {batch_count} calculate_batch() var")
    print(f"   [VOL] {volume_count} volume gerekli")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INDICATOR DISCOVERY & REGISTRY GENERATOR")
    print("=" * 70)

    # Paths
    base_path = Path(__file__).parent
    indicators_path = base_path
    output_path = base_path / '__init__.py'

    print(f"\n[PATH] Indicators: {indicators_path}")
    print(f"[OUT] Output: {output_path}")

    # Discover
    registry = discover_all_indicators(indicators_path)

    # Generate
    print("\n" + "=" * 70)
    print("GENERATING __init__.py")
    print("=" * 70)

    generate_init_file(registry, output_path)

    print("\n" + "=" * 70)
    print("[OK] TAMAMLANDI!")
    print("=" * 70)
    print("\n[INFO] Kullanım:")
    print("   from components.indicators import INDICATOR_REGISTRY")
    print("   from components.indicators import get_indicator_class")
    print("   ")
    print("   RSI = get_indicator_class('rsi')")
    print("   rsi = RSI(period=14)")
    print("")
