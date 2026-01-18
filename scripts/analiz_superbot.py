#!/usr/bin/env python3
"""
SuperBot Proje Analiz Scripti
Analyzes all files and directories.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def analyze_directory(root_path, output_file="superbot_analysis.md"):
    """
    SuperBot projesini analiz et
    """
    
    Parent directories
    main_folders = ['components', 'core', 'config', 'modules']
    
    # Analysis Results
    analysis = {
        'total_files': 0,
        'total_lines': 0,
        'file_types': {},
        'modules_detail': {},
        'recent_files': [],
        'large_files': []
    }
    
    output = []
    output.append("# 🔍 SuperBot Proje Analizi")
    output.append(f"\nAnaliz Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("\n" + "="*80)
    
    # For all main directories
    for folder in main_folders:
        folder_path = Path(root_path) / folder
        if not folder_path.exists():
            continue
            
        ## {folder} FOLDER
        output.append("-" * 40)
        
        # Look at subfolders
        subfolders = {}
        for root, dirs, files in os.walk(folder_path):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_path = Path(root).relative_to(folder_path)
            level = len(rel_path.parts)
            
            # List files
            py_files = [f for f in files if f.endswith('.py')]
            yaml_files = [f for f in files if f.endswith(('.yaml', '.yml'))]
            json_files = [f for f in files if f.endswith('.json')]
            
            if py_files or yaml_files or json_files:
                indent = "  " * level
                subfolder_name = str(rel_path) if str(rel_path) != "." else ""
                
                if subfolder_name:
                    output.append(f"\n{indent}### {subfolder_name}/")
                
                # Python files
                if py_files:
                    **Python files ({len(py_files)})**:
                    for f in sorted(py_files)[:10]:  # First 10 files
                        file_path = Path(root) / f
                        try:
                            # File size and row count
                            size = file_path.stat().st_size / 1024  # KB
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                                lines = len(fp.readlines())
                            output.append(f"{indent}- `f` ({lines} lines, {size:.1f} KB)")
                            
                            # First docstring
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                                content = fp.read()
                                if '"""' in content:
                                    docstring = content.split('"""')[1].strip()
                                    first_line = docstring.split('\n')[0][:60]
                                    if first_line:
                                        output.append(f"{indent}  └─ {first_line}")
                        except:
                            output.append(f"{indent}- `{f}`")
                    
                    if len(py_files) > 10:
                        "... and {len(py_files)-10} more files"
                
                # Configuration files
                if yaml_files:
                    **Config files ({}):**
                    for f in sorted(yaml_files):
                        output.append(f"{indent}- `{f}`")
                
                # JSON Files
                if json_files:
                    **Data files ({}):**
                    for f in sorted(json_files)[:5]:
                        output.append(f"{indent}- `{f}`")
                    if len(json_files) > 5:
                        "... and {len(json_files)-5} more files"
    
    # Private directories check
    Other Important Classes
    output.append("-" * 40)
    
    other_folders = ['data', 'indicators', 'strategies', 'scripts', 'tests', 'docs']
    for folder in other_folders:
        folder_path = Path(root_path) / folder
        if folder_path.exists():
            file_count = sum(1 for _ in folder_path.rglob('*.py'))
            if file_count > 0:
                output.append(f"\n### {folder}/")
                output.append(f"- Python files: {file_count}")
                
                # List the first few files
                files = list(folder_path.rglob('*.py'))[:5]
                for f in files:
                    rel_path = f.relative_to(folder_path)
                    output.append(f"  - `{rel_path}`")
    
    # Check sprint status
    ## Sprint Status Analysis
    output.append("-" * 40)
    
    status_files = list(Path(root_path).glob('STATUS_Sprint_*.md'))
    for status_file in sorted(status_files):
        output.append(f"\n### {status_file.name}")
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]  # First 20 lines
                for line in lines:
                    if 'Completed' in line or 'COMPLETED' in line or '%' in line:
                        output.append(f"  {line.strip()}")
        except:
            pass
    
    # Last modified files found
    ## Last Modified Files
    output.append("-" * 40)
    
    recent_files = []
    for folder in main_folders:
        folder_path = Path(root_path) / folder
        if folder_path.exists():
            for f in folder_path.rglob('*.py'):
                try:
                    mtime = f.stat().st_mtime
                    recent_files.append((mtime, f))
                except:
                    pass
    
    recent_files.sort(reverse=True)
    "Last modified files:"
    for mtime, f in recent_files[:10]:
        rel_path = f.relative_to(root_path)
        date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        output.append(f"- {date}: `{rel_path}`")
    
    # Summary statistics
    ## Summary Statistics
    output.append("-" * 40)
    
    total_py_files = sum(1 for _ in Path(root_path).rglob('*.py'))
    total_yaml_files = sum(1 for _ in Path(root_path).rglob('*.yaml'))
    total_json_files = sum(1 for _ in Path(root_path).rglob('*.json'))
    
    "- **Total Python files:** {total_py_files}"
    "output.append(f"- **Total config file:** {total_yaml_files}")
    "- **Total JSON files:** {total_json_files}"
    
    # Counting files related to AI/ML
    ai_files = []
    for f in Path(root_path).rglob('*.py'):
        name = f.name.lower()
        if any(keyword in name for keyword in ['ai', 'ml', 'model', 'train', 'predict', 'backtest']):
            ai_files.append(f)
    
    if ai_files:
        ### AI/ML Related Files ({})

### 🤖
        for f in ai_files[:15]:
            rel_path = f.relative_to(root_path)
            output.append(f"- `{rel_path}`")
    
    # Sonucu yaz
    output_content = "\n".join(output)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"✅ Analysis completed: {output_file}")
    print(f"Total {total_py_files} Python files were analyzed")
    
    return output_content

if __name__ == "__main__":
    # SuperBot root directory path
    # Windows for example: "D:/Python/SuperBot"
    # Linux example: "/home/user/SuperBot"
    
    project_path = input("SuperBot project path (e.g. D:/Python/SuperBot): ").strip()
    
    if not Path(project_path).exists():
        ❌ Folder not found: {project_path}
    else:
        result = analyze_directory(project_path)
        print("\n" + "="*50)
        print("Analysis result saved to superbot_analysis.md file")
        print("="*50)