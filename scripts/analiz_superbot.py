#!/usr/bin/env python3
"""
SuperBot Proje Analiz Scripti
Tüm klasör ve dosyaları analiz eder
"""

import os
import json
from pathlib import Path
from datetime import datetime

def analyze_directory(root_path, output_file="superbot_analysis.md"):
    """
    SuperBot projesini analiz et
    """
    
    # Ana klasörler
    main_folders = ['components', 'core', 'config', 'modules']
    
    # Analiz sonuçları
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
    
    # Her ana klasör için
    for folder in main_folders:
        folder_path = Path(root_path) / folder
        if not folder_path.exists():
            continue
            
        output.append(f"\n## 📁 {folder.upper()} KLASÖRİ")
        output.append("-" * 40)
        
        # Alt klasörleri tara
        subfolders = {}
        for root, dirs, files in os.walk(folder_path):
            # __pycache__ ve .git gibi klasörleri atla
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_path = Path(root).relative_to(folder_path)
            level = len(rel_path.parts)
            
            # Dosyaları listele
            py_files = [f for f in files if f.endswith('.py')]
            yaml_files = [f for f in files if f.endswith(('.yaml', '.yml'))]
            json_files = [f for f in files if f.endswith('.json')]
            
            if py_files or yaml_files or json_files:
                indent = "  " * level
                subfolder_name = str(rel_path) if str(rel_path) != "." else ""
                
                if subfolder_name:
                    output.append(f"\n{indent}### {subfolder_name}/")
                
                # Python dosyaları
                if py_files:
                    output.append(f"{indent}**Python dosyaları ({len(py_files)}):**")
                    for f in sorted(py_files)[:10]:  # İlk 10 dosya
                        file_path = Path(root) / f
                        try:
                            # Dosya boyutu ve satır sayısı
                            size = file_path.stat().st_size / 1024  # KB
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                                lines = len(fp.readlines())
                            output.append(f"{indent}- `{f}` ({lines} satır, {size:.1f}KB)")
                            
                            # İlk docstring'i oku
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
                        output.append(f"{indent}... ve {len(py_files)-10} dosya daha")
                
                # Config dosyaları
                if yaml_files:
                    output.append(f"{indent}**Config dosyaları ({len(yaml_files)}):**")
                    for f in sorted(yaml_files):
                        output.append(f"{indent}- `{f}`")
                
                # JSON dosyaları
                if json_files:
                    output.append(f"{indent}**Data dosyaları ({len(json_files)}):**")
                    for f in sorted(json_files)[:5]:
                        output.append(f"{indent}- `{f}`")
                    if len(json_files) > 5:
                        output.append(f"{indent}... ve {len(json_files)-5} dosya daha")
    
    # Özel klasörleri kontrol et
    output.append("\n## 🔧 DİĞER ÖNEMLİ KLASÖRLER")
    output.append("-" * 40)
    
    other_folders = ['data', 'indicators', 'strategies', 'scripts', 'tests', 'docs']
    for folder in other_folders:
        folder_path = Path(root_path) / folder
        if folder_path.exists():
            file_count = sum(1 for _ in folder_path.rglob('*.py'))
            if file_count > 0:
                output.append(f"\n### {folder}/")
                output.append(f"- Python dosyaları: {file_count}")
                
                # İlk birkaç dosyayı listele
                files = list(folder_path.rglob('*.py'))[:5]
                for f in files:
                    rel_path = f.relative_to(folder_path)
                    output.append(f"  - `{rel_path}`")
    
    # Sprint durumlarını kontrol et
    output.append("\n## 📊 SPRINT DURUM ANALİZİ")
    output.append("-" * 40)
    
    status_files = list(Path(root_path).glob('STATUS_Sprint_*.md'))
    for status_file in sorted(status_files):
        output.append(f"\n### {status_file.name}")
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]  # İlk 20 satır
                for line in lines:
                    if 'Tamamlandı' in line or 'COMPLETED' in line or '%' in line:
                        output.append(f"  {line.strip()}")
        except:
            pass
    
    # Son değiştirilen dosyaları bul
    output.append("\n## 🕐 SON DEĞİŞTİRİLEN DOSYALAR")
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
    output.append("\nEn son değiştirilen 10 dosya:")
    for mtime, f in recent_files[:10]:
        rel_path = f.relative_to(root_path)
        date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        output.append(f"- {date}: `{rel_path}`")
    
    # Özet istatistikler
    output.append("\n## 📈 ÖZET İSTATİSTİKLER")
    output.append("-" * 40)
    
    total_py_files = sum(1 for _ in Path(root_path).rglob('*.py'))
    total_yaml_files = sum(1 for _ in Path(root_path).rglob('*.yaml'))
    total_json_files = sum(1 for _ in Path(root_path).rglob('*.json'))
    
    output.append(f"\n- **Toplam Python dosyası:** {total_py_files}")
    output.append(f"- **Toplam Config dosyası:** {total_yaml_files}")
    output.append(f"- **Toplam JSON dosyası:** {total_json_files}")
    
    # AI/ML ile ilgili dosyaları say
    ai_files = []
    for f in Path(root_path).rglob('*.py'):
        name = f.name.lower()
        if any(keyword in name for keyword in ['ai', 'ml', 'model', 'train', 'predict', 'backtest']):
            ai_files.append(f)
    
    if ai_files:
        output.append(f"\n### 🤖 AI/ML İlgili Dosyalar ({len(ai_files)})")
        for f in ai_files[:15]:
            rel_path = f.relative_to(root_path)
            output.append(f"- `{rel_path}`")
    
    # Sonucu yaz
    output_content = "\n".join(output)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"✅ Analiz tamamlandı: {output_file}")
    print(f"📊 Toplam {total_py_files} Python dosyası analiz edildi")
    
    return output_content

if __name__ == "__main__":
    # SuperBot ana klasör yolu
    # Windows için örnek: "D:/Python/SuperBot"
    # Linux için örnek: "/home/user/SuperBot"
    
    project_path = input("SuperBot proje klasör yolu (örn: D:/Python/SuperBot): ").strip()
    
    if not Path(project_path).exists():
        print(f"❌ Klasör bulunamadı: {project_path}")
    else:
        result = analyze_directory(project_path)
        print("\n" + "="*50)
        print("Analiz sonucu superbot_analysis.md dosyasına kaydedildi")
        print("="*50)