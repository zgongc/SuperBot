@echo off
REM scripts/cleanup_cache.bat
REM SuperBot - Cache Temizleme Script
REM Python cache, log, temporary dosyalari temizler

setlocal

echo.
echo ========================================
echo   SuperBot Cache Temizleme
echo ========================================
echo.

REM Ana klasör (proje kökü)
set "project_root=%cd%"

REM Toplam silinen dosya/klasör sayısı
set /a total_deleted=0

echo Temizleniyor: %project_root%
echo.

REM ==========================================
REM 1. __pycache__ klasörlerini sil
REM ==========================================
echo [1/6] __pycache__ klasorleri siliniyor...
for /r "%project_root%" %%d in (.) do (
    if exist "%%d\__pycache__" (
        echo   - %%d\__pycache__
        rd /s /q "%%d\__pycache__" 2>nul
        set /a total_deleted+=1
    )
)
echo [OK] __pycache__ temizlendi
echo.

REM ==========================================
REM 2. .pyc dosyalarını sil
REM ==========================================
echo [2/6] .pyc dosyalari siliniyor...
for /r "%project_root%" %%f in (*.pyc) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)
echo [OK] .pyc dosyalari temizlendi
echo.

REM ==========================================
REM 3. .pyo dosyalarını sil
REM ==========================================
echo [3/6] .pyo dosyalari siliniyor...
for /r "%project_root%" %%f in (*.pyo) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)
echo [OK] .pyo dosyalari temizlendi
echo.

REM ==========================================
REM 4. data/cache/* temizle
REM ==========================================
echo [4/6] data/cache/* temizleniyor...
if exist "%project_root%\data\cache\*" (
    del /f /q "%project_root%\data\cache\*" 2>nul
    echo [OK] Cache klasoru temizlendi
) else (
    echo [SKIP] Cache klasoru bulunamadi
)
echo.

REM ==========================================
REM 5. .log dosyalarını sil (data/logs/)
REM ==========================================
echo [5/6] Log dosyalari siliniyor...
if exist "%project_root%\data\logs\*.log" (
    del /f /q "%project_root%\data\logs\*.log" 2>nul
    echo [OK] Log dosyalari temizlendi
) else (
    echo [SKIP] Log dosyalari bulunamadi
)
echo.

REM ==========================================
REM 6. Temporary dosyaları sil
REM ==========================================
echo [6/6] Temporary dosyalar siliniyor...

REM .tmp dosyaları
for /r "%project_root%" %%f in (*.tmp) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)

REM .bak dosyaları
for /r "%project_root%" %%f in (*.bak) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)

REM .backup dosyaları
for /r "%project_root%" %%f in (*.backup) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)

REM ~ ile biten dosyalar (gedit/vim backup)
for /r "%project_root%" %%f in (*~) do (
    echo   - %%f
    del /f /q "%%f" 2>nul
    set /a total_deleted+=1
)

echo [OK] Temporary dosyalar temizlendi
echo.

REM ==========================================
REM Özet
REM ==========================================
echo ========================================
echo   Temizlik Tamamlandi!
echo ========================================
echo.
echo Temizlenen ogeler:
echo   - __pycache__ klasorleri
echo   - .pyc, .pyo dosyalari
echo   - data/cache/* dosyalari
echo   - Log dosyalari
echo   - Temporary dosyalar (.tmp, .bak, .backup, ~)
echo.
echo Proje yolu: %project_root%
echo.

REM pause
endlocal
