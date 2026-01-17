#!/bin/bash
# scripts/restart_ai_server.sh
# AI Server restart script (Linux)

echo "🔄 AI Server restart başlatılıyor..."

# 1. Mevcut AI Server process'ini bul ve durdur
echo "1️⃣ Mevcut AI Server process'i durduruluyor..."
pkill -f "uvicorn.*modules.ai.server.main:app" && echo "   ✅ Process durduruldu" || echo "   ⚠️  Process bulunamadı"

# 2. 2 saniye bekle
sleep 2

# 3. Yeni AI Server başlat
echo "2️⃣ Yeni AI Server başlatılıyor..."
cd /opt/superbot-ai

# Conda environment path (miniconda3)
CONDA_PATH="/opt/zgongc/miniconda3"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate superbot

# Nohup ile background'da çalıştır
nohup uvicorn modules.ai.server.main:app --host 0.0.0.0 --port 8001 --workers 1 --log-level info > ai_server.log 2>&1 &

# 4. 3 saniye bekle server'ın başlaması için
sleep 3

# 5. Health check
echo "3️⃣ Health check..."
if curl -s http://localhost:8001/health | grep -q "ok"; then
    echo "   ✅ AI Server başarıyla başlatıldı!"
    echo "   📊 Process bilgisi:"
    ps aux | grep "uvicorn.*modules.ai.server.main:app" | grep -v grep
else
    echo "   ❌ AI Server başlatılamadı!"
    echo "   📋 Son 20 satır log:"
    tail -20 ai_server.log
    exit 1
fi

echo ""
echo "✅ AI Server restart tamamlandı!"
echo "📝 Log dosyası: /opt/superbot-ai/ai_server.log"
echo "🔍 Log takibi: tail -f /opt/superbot-ai/ai_server.log"
