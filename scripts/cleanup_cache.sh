#!/bin/bash
# scripts/cleanup_cache.sh
# SuperBot - Cache Temizleme Script (Linux/Mac)
# Python cache, log, temporary dosyalari temizler

set -e

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "   SuperBot Cache Temizleme"
echo "========================================"
echo ""

# Ana klasör (proje kökü)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Sayaçlar
total_deleted=0

echo "Temizleniyor: $PROJECT_ROOT"
echo ""

# ==========================================
# 1. __pycache__ klasörlerini sil
# ==========================================
echo "[1/6] __pycache__ klasorleri siliniyor..."
pycache_count=$(find "$PROJECT_ROOT" -type d -name "__pycache__" | wc -l)
if [ "$pycache_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec echo "  - {}" \; -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}[OK]${NC} $pycache_count __pycache__ klasoru temizlendi"
else
    echo "[SKIP] __pycache__ klasoru bulunamadi"
fi
echo ""

# ==========================================
# 2. .pyc dosyalarını sil
# ==========================================
echo "[2/6] .pyc dosyalari siliniyor..."
pyc_count=$(find "$PROJECT_ROOT" -type f -name "*.pyc" | wc -l)
if [ "$pyc_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*.pyc" -exec echo "  - {}" \; -delete
    echo -e "${GREEN}[OK]${NC} $pyc_count .pyc dosyasi temizlendi"
else
    echo "[SKIP] .pyc dosyasi bulunamadi"
fi
echo ""

# ==========================================
# 3. .pyo dosyalarını sil
# ==========================================
echo "[3/6] .pyo dosyalari siliniyor..."
pyo_count=$(find "$PROJECT_ROOT" -type f -name "*.pyo" | wc -l)
if [ "$pyo_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*.pyo" -exec echo "  - {}" \; -delete
    echo -e "${GREEN}[OK]${NC} $pyo_count .pyo dosyasi temizlendi"
else
    echo "[SKIP] .pyo dosyasi bulunamadi"
fi
echo ""

# ==========================================
# 4. data/cache/* temizle
# ==========================================
echo "[4/6] data/cache/* temizleniyor..."
if [ -d "$PROJECT_ROOT/data/cache" ]; then
    cache_count=$(find "$PROJECT_ROOT/data/cache" -type f | wc -l)
    if [ "$cache_count" -gt 0 ]; then
        rm -rf "$PROJECT_ROOT/data/cache/"*
        echo -e "${GREEN}[OK]${NC} $cache_count cache dosyasi temizlendi"
    else
        echo "[SKIP] Cache klasoru bos"
    fi
else
    echo "[SKIP] Cache klasoru bulunamadi"
fi
echo ""

# ==========================================
# 5. .log dosyalarını sil (data/logs/)
# ==========================================
echo "[5/6] Log dosyalari siliniyor..."
if [ -d "$PROJECT_ROOT/data/logs" ]; then
    log_count=$(find "$PROJECT_ROOT/data/logs" -type f -name "*.log" | wc -l)
    if [ "$log_count" -gt 0 ]; then
        find "$PROJECT_ROOT/data/logs" -type f -name "*.log" -delete
        echo -e "${GREEN}[OK]${NC} $log_count log dosyasi temizlendi"
    else
        echo "[SKIP] Log dosyasi bulunamadi"
    fi
else
    echo "[SKIP] Logs klasoru bulunamadi"
fi
echo ""

# ==========================================
# 6. Temporary dosyaları sil
# ==========================================
echo "[6/6] Temporary dosyalar siliniyor..."

# .tmp dosyaları
tmp_count=$(find "$PROJECT_ROOT" -type f -name "*.tmp" | wc -l)
if [ "$tmp_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*.tmp" -exec echo "  - {}" \; -delete
fi

# .bak dosyaları
bak_count=$(find "$PROJECT_ROOT" -type f -name "*.bak" | wc -l)
if [ "$bak_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*.bak" -exec echo "  - {}" \; -delete
fi

# .backup dosyaları
backup_count=$(find "$PROJECT_ROOT" -type f -name "*.backup" | wc -l)
if [ "$backup_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*.backup" -exec echo "  - {}" \; -delete
fi

# ~ ile biten dosyalar
tilde_count=$(find "$PROJECT_ROOT" -type f -name "*~" | wc -l)
if [ "$tilde_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type f -name "*~" -exec echo "  - {}" \; -delete
fi

temp_total=$((tmp_count + bak_count + backup_count + tilde_count))
if [ "$temp_total" -gt 0 ]; then
    echo -e "${GREEN}[OK]${NC} $temp_total temporary dosya temizlendi"
else
    echo "[SKIP] Temporary dosya bulunamadi"
fi
echo ""

# ==========================================
# Özet
# ==========================================
echo "========================================"
echo "   Temizlik Tamamlandi!"
echo "========================================"
echo ""
echo "Temizlenen ogeler:"
echo "  - __pycache__ klasorleri"
echo "  - .pyc, .pyo dosyalari"
echo "  - data/cache/* dosyalari"
echo "  - Log dosyalari"
echo "  - Temporary dosyalar (.tmp, .bak, .backup, ~)"
echo ""
echo "Proje yolu: $PROJECT_ROOT"
echo ""
