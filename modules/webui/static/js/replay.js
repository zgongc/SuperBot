/**
 * Replay Trading JavaScript
 * Chart, kontroller ve state yönetimi
 */

// ============================================
// STATE
// ============================================
const ReplayState = {
    sessionId: null,
    strategies: [],
    isPlaying: false,
    speed: 1,
    currentBar: 0,
    totalBars: 0,
    playInterval: null,

    // Chart objects
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    indicators: {},         // Overlay indicators (EMA, SMA, BB)
    markers: [],

    // Trade visualization
    tradeLines: [],         // SL/TP price lines
    tradeAreas: [],         // TP/SL area series

    // Indicator visibility state
    indicatorVisibility: {}  // {name: true/false}
};

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    initReplay();
});

async function initReplay() {
    console.log('🎮 Replay Trading başlatılıyor...');

    // Event listeners
    setupEventListeners();

    // Load strategies
    await loadStrategies();
}

function setupEventListeners() {
    // Strategy select
    document.getElementById('strategy-select').addEventListener('change', onStrategyChange);

    // Start/Reset buttons
    document.getElementById('btn-start-replay').addEventListener('click', startReplay);
    document.getElementById('btn-reset-replay').addEventListener('click', resetReplay);

    // Playback controls
    document.getElementById('btn-play-pause').addEventListener('click', togglePlayPause);
    document.getElementById('btn-step-back').addEventListener('click', () => jumpToTrade(-1));  // Önceki trade
    document.getElementById('btn-step-forward').addEventListener('click', () => jumpToTrade(1)); // Sonraki trade
    document.getElementById('btn-prev').addEventListener('click', () => step(-10));
    document.getElementById('btn-next').addEventListener('click', () => step(10));

    // Timeline slider
    document.getElementById('timeline-slider').addEventListener('input', onTimelineChange);

    // Speed select
    document.getElementById('speed-select').addEventListener('change', onSpeedChange);
}

// ============================================
// STRATEGIES
// ============================================
async function loadStrategies() {
    try {
        const response = await fetch('/api/replay/strategies');
        const data = await response.json();

        if (data.status === 'success') {
            ReplayState.strategies = data.data.strategies;
            populateStrategySelect(data.data.strategies);
        }
    } catch (error) {
        console.error('Strateji yükleme hatası:', error);
        showError('Strategies could not be loaded.');
    }
}

function populateStrategySelect(strategies) {
    const select = document.getElementById('strategy-select');
    select.innerHTML = '<option value="">Select strategy...</option>';

    strategies.forEach(s => {
        const option = document.createElement('option');
        option.value = s.id;
        option.textContent = `${s.name} (${s.symbol} ${s.timeframe})`;
        option.dataset.symbol = s.symbol;
        option.dataset.timeframe = s.timeframe;
        option.dataset.startDate = s.start_date;
        option.dataset.endDate = s.end_date;
        select.appendChild(option);
    });
}

function onStrategyChange(e) {
    const option = e.target.selectedOptions[0];
    const btn = document.getElementById('btn-start-replay');

    if (option && option.value) {
        // Show strategy information.
        document.getElementById('symbol-display').value = option.dataset.symbol || '-';
        document.getElementById('timeframe-display').value = option.dataset.timeframe || '-';

        const startDate = option.dataset.startDate || '';
        const endDate = option.dataset.endDate || '';
        document.getElementById('date-range-display').value =
            startDate && endDate ? `${startDate} → ${endDate}` : '-';

        btn.disabled = false;
    } else {
        document.getElementById('symbol-display').value = '-';
        document.getElementById('timeframe-display').value = '-';
        document.getElementById('date-range-display').value = '-';
        btn.disabled = true;
    }
}

// ============================================
// SESSION MANAGEMENT
// ============================================
async function startReplay() {
    const strategyId = document.getElementById('strategy-select').value;
    if (!strategyId) return;

    try {
        showLoading('Replay is starting...');

        const response = await fetch('/api/replay/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy_id: strategyId })
        });

        const data = await response.json();
        hideLoading();

        if (data.status === 'success') {
            const session = data.data.session;
            ReplayState.sessionId = session.id;
            ReplayState.totalBars = session.total_bars;
            ReplayState.currentBar = 0;

            // Update the UI.
            showReplayUI(session);

            // Wait for the DOM to repaint, then create the chart.
            await new Promise(resolve => requestAnimationFrame(resolve));

            // Create the chart.
            createChart();

            // Load the initial data.
            await loadCandles();

            showSuccess('Replay started!');
        } else {
            showError(data.message || 'Replay could not be started.');
        }
    } catch (error) {
        hideLoading();
        console.error('Replay başlatma hatası:', error);
        showError('Replay could not be started: ' + error.message);
    }
}

function showReplayUI(session) {
    // Setup panelini gizle
    document.getElementById('setup-panel').style.display = 'none';

    // Show other panels.
    document.getElementById('chart-panel').style.display = 'block';
    document.getElementById('controls-panel').style.display = 'block';
    document.getElementById('stats-panel').style.display = 'grid';

    // Update the chart title.
    document.getElementById('chart-title').textContent = session.strategy_name;
    document.getElementById('chart-symbol').textContent = session.symbol;
    document.getElementById('chart-timeframe').textContent = session.timeframe;

    // Update the timeline.
    document.getElementById('timeline-slider').max = session.total_bars - 1;
    document.getElementById('total-bars').textContent = `/ ${session.total_bars}`;

    // Reset butonunu aktifle
    document.getElementById('btn-reset-replay').disabled = false;
}

async function resetReplay() {
    // Delete the session.
    if (ReplayState.sessionId) {
        try {
            await fetch(`/api/replay/sessions/${ReplayState.sessionId}`, {
                method: 'DELETE'
            });
        } catch (e) {
            console.error('Session silme hatası:', e);
        }
    }

    // Reset the state.
    stopPlayback();
    ReplayState.sessionId = null;
    ReplayState.currentBar = 0;
    ReplayState.totalBars = 0;

    // Clear the chart.
    if (ReplayState.chart) {
        ReplayState.chart.remove();
        ReplayState.chart = null;
        ReplayState.candleSeries = null;
        ReplayState.volumeSeries = null;
        ReplayState.indicators = {};
    }

    // Reset the UI.
    document.getElementById('setup-panel').style.display = 'block';
    document.getElementById('chart-panel').style.display = 'none';
    document.getElementById('controls-panel').style.display = 'none';
    document.getElementById('stats-panel').style.display = 'none';

    document.getElementById('btn-reset-replay').disabled = true;
    document.getElementById('strategy-select').value = '';
    onStrategyChange({ target: document.getElementById('strategy-select') });
}

// ============================================
// CHART
// ============================================

// Indicator color palette
const INDICATOR_COLORS = [
    '#f1c40f', // yellow
    '#9b59b6', // mor
    '#3498db', // mavi
    '#e74c3c', // red
    '#2ecc71', // green
    '#1abc9c', // turkuaz
    '#e67e22', // turuncu
    '#95a5a6', // gri
    '#00bcd4', // cyan
    '#ff5722'  // deep orange
];

// Session için indikatör isimleri (from the backend)
let sessionIndicatorNames = [];

function createChart() {
    const container = document.getElementById('chart-container');
    container.innerHTML = '';

    // Chart theme
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    const chartOptions = {
        layout: {
            background: { type: 'solid', color: isDark ? '#0f1419' : '#ffffff' },
            textColor: isDark ? '#e7e9ea' : '#0f1419'
        },
        grid: {
            vertLines: { color: isDark ? '#2f3542' : '#e1e4e8' },
            horzLines: { color: isDark ? '#2f3542' : '#e1e4e8' }
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: isDark ? '#2f3542' : '#e1e4e8' },
        timeScale: {
            borderColor: isDark ? '#2f3542' : '#e1e4e8',
            timeVisible: true,
            secondsVisible: false
        }
    };

    // Ana Chart (Candlestick + Overlay indicators)
    ReplayState.chart = LightweightCharts.createChart(container, {
        ...chartOptions,
        width: container.clientWidth,
        height: container.clientHeight || 400
    });

    // Candlestick series
    ReplayState.candleSeries = ReplayState.chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a'
    });

    // Volume series
    ReplayState.volumeSeries = ReplayState.chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        scaleMargins: { top: 0.85, bottom: 0 }
    });

    // Indicators will be created dynamically (after loadCandles).
    ReplayState.indicators = {};

    // Resize handler (debounced)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (ReplayState.chart) {
                ReplayState.chart.applyOptions({ width: container.clientWidth });
            }
        }, 100);
    });
}

// Check if the indicator is an overlay (EMA, SMA, BB = overlay, RSI/ATR/ADX = oscillator - skipping for now)
function isOverlayIndicator(name) {
    const nameLower = name.toLowerCase();
    return nameLower.includes('ema') ||
           nameLower.includes('sma') ||
           nameLower.includes('bollinger') ||
           nameLower.includes('ma_');  // ma_20 like
}

function createIndicatorSeries(indicatorNames) {
    if (!ReplayState.chart || !indicatorNames) return;

    // Filter only overlay indicators.
    const overlayNames = indicatorNames.filter(name => isOverlayIndicator(name));
    console.log('📊 Overlay indikatörler:', overlayNames);

    // Clear the existing indicators.
    for (const [name, series] of Object.entries(ReplayState.indicators)) {
        try {
            ReplayState.chart.removeSeries(series);
        } catch (e) {
            console.warn('İndikatör kaldırılamadı:', name);
        }
    }
    ReplayState.indicators = {};

    // Store indicator names.
    sessionIndicatorNames = overlayNames;

    // Create a series for each overlay indicator.
    overlayNames.forEach((name, index) => {
        const color = INDICATOR_COLORS[index % INDICATOR_COLORS.length];

        // Default visibility (yoksa true)
        if (ReplayState.indicatorVisibility[name] === undefined) {
            ReplayState.indicatorVisibility[name] = true;
        }
        const isVisible = ReplayState.indicatorVisibility[name];

        // Add as an overlay to the main chart.
        ReplayState.indicators[name] = ReplayState.chart.addLineSeries({
            color: color,
            lineWidth: 2,
            title: name.toUpperCase(),
            visible: isVisible,
            lastValueVisible: false,  // Hide the label on the right side (reduces flickering).
            priceLineVisible: false   // Hide the price line.
        });
    });

    // Update the indicator panel (overlays only).
    updateIndicatorPanel(overlayNames);
}

function updateIndicatorPanel(indicatorNames) {
    const panel = document.getElementById('indicators-panel');
    if (!panel) return;

    // Clear and recreate the panel content.
    panel.innerHTML = '';

    indicatorNames.forEach((name, index) => {
        const color = INDICATOR_COLORS[index % INDICATOR_COLORS.length];

        // Default: visible
        if (ReplayState.indicatorVisibility[name] === undefined) {
            ReplayState.indicatorVisibility[name] = true;
        }
        const isVisible = ReplayState.indicatorVisibility[name];

        const div = document.createElement('div');
        div.className = 'stat-item indicator-toggle-item';
        div.innerHTML = `
            <label class="indicator-toggle">
                <input type="checkbox"
                       id="toggle-${name}"
                       ${isVisible ? 'checked' : ''}
                       onchange="toggleIndicator('${name}')">
                <span class="indicator-label" style="color: ${color};">${name.toUpperCase()}</span>
            </label>
            <span class="stat-value" id="ind-${name}">-</span>
        `;
        panel.appendChild(div);
    });

    // Toggle all button event
    const toggleAllBtn = document.getElementById('btn-toggle-all-indicators');
    if (toggleAllBtn) {
        toggleAllBtn.onclick = toggleAllIndicators;
    }
}

function toggleIndicator(name) {
    const checkbox = document.getElementById(`toggle-${name}`);
    const isVisible = checkbox ? checkbox.checked : false;

    ReplayState.indicatorVisibility[name] = isVisible;

    const series = ReplayState.indicators[name];
    if (series) {
        series.applyOptions({ visible: isVisible });
    }
}

function toggleAllIndicators() {
    // If any are open, close all; otherwise, open all.
    const anyVisible = Object.values(ReplayState.indicatorVisibility).some(v => v);
    const newState = !anyVisible;

    for (const name of Object.keys(ReplayState.indicatorVisibility)) {
        ReplayState.indicatorVisibility[name] = newState;

        // Update the checkbox.
        const checkbox = document.getElementById(`toggle-${name}`);
        if (checkbox) checkbox.checked = newState;

        // Update the series.
        if (ReplayState.indicators[name]) {
            ReplayState.indicators[name].applyOptions({ visible: newState });
        }
    }
}

async function loadCandles() {
    if (!ReplayState.sessionId) return;

    try {
        const response = await fetch(
            `/api/replay/sessions/${ReplayState.sessionId}/candles?start=0&limit=500`
        );
        const data = await response.json();

        if (data.status === 'success') {
            // Create indicator series during the initial load.
            if (data.data.indicator_names && Object.keys(ReplayState.indicators).length === 0) {
                createIndicatorSeries(data.data.indicator_names);
            }

            updateChart(data.data);
            updateState(data.data);
        }
    } catch (error) {
        console.error('Candle yükleme hatası:', error);
    }
}

function updateChart(data) {
    if (!ReplayState.candleSeries) return;

    // Candles
    if (data.candles && data.candles.length > 0) {
        ReplayState.candleSeries.setData(data.candles);
    }

    // Volume
    if (data.volumes && data.volumes.length > 0) {
        ReplayState.volumeSeries.setData(data.volumes);
    }

    // Overlay Indicators (EMA, SMA, BB)
    if (data.indicators) {
        for (const [name, values] of Object.entries(data.indicators)) {
            // Handle only overlay indicators.
            if (!isOverlayIndicator(name)) continue;

            // If there is no series for this indicator, create it.
            if (!ReplayState.indicators[name] && data.indicator_names) {
                createIndicatorSeries(data.indicator_names);
            }

            // Set the data.
            if (ReplayState.indicators[name] && values.length > 0) {
                ReplayState.indicators[name].setData(values);
            }
        }
    }

    // Markers (trade entry/exit)
    if (data.markers && data.markers.length > 0) {
        ReplayState.candleSeries.setMarkers(data.markers);
    } else {
        ReplayState.candleSeries.setMarkers([]);
    }

    // Trade boxes (SL/TP zones)
    if (data.trades) {
        drawTradeBoxes(data.trades);
        updateTradesPanel(data.trades);
    }
}

function drawTradeBoxes(trades) {
    // Clear previous trade series.
    if (ReplayState.tradeAreas && ReplayState.tradeAreas.length > 0) {
        ReplayState.tradeAreas.forEach(series => {
            try {
                ReplayState.chart.removeSeries(series);
            } catch (e) {}
        });
    }
    ReplayState.tradeAreas = [];
    ReplayState.tradeLines = [];

    // Show the box only for the last 5 trades.
    const recentTrades = trades.slice(-5);

    recentTrades.forEach(trade => {
        if (!trade.entry_price || !trade.entry_time || !trade.exit_time) return;

        const isLong = trade.side === 'LONG';
        const isWin = trade.pnl >= 0;

        // Colors: TP is always green, SL is always red.
        const tpColor = 'rgba(38, 166, 154, ';   // green
        const slColor = 'rgba(239, 83, 80, ';    // red

        // Draw a box (between entry and exit) with the Baseline Series.
        // LONG: TP above (topFill), SL below (bottomFill)
        // SHORT: TP below (bottomFill), SL above (topFill)

        // TP region (green) - between entry and TP
        if (trade.take_profit) {
            const tpBox = ReplayState.chart.addBaselineSeries({
                baseValue: { type: 'price', price: trade.entry_price },
                // LONG: green on top, SHORT: green at the bottom
                topLineColor: isLong ? tpColor + '0)' : 'rgba(0, 0, 0, 0)',
                topFillColor1: isLong ? tpColor + '0.3)' : 'rgba(0, 0, 0, 0)',
                topFillColor2: isLong ? tpColor + '0.1)' : 'rgba(0, 0, 0, 0)',
                bottomLineColor: isLong ? 'rgba(0, 0, 0, 0)' : tpColor + '0)',
                bottomFillColor1: isLong ? 'rgba(0, 0, 0, 0)' : tpColor + '0.3)',
                bottomFillColor2: isLong ? 'rgba(0, 0, 0, 0)' : tpColor + '0.1)',
                lineWidth: 0,
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false
            });

            // TP box data - always use the TP value
            const tpData = [];
            const timeStep = (trade.exit_time - trade.entry_time) / 10;
            for (let t = trade.entry_time; t <= trade.exit_time; t += timeStep) {
                tpData.push({
                    time: Math.floor(t),
                    value: trade.take_profit
                });
            }
            if (tpData.length > 0) {
                tpBox.setData(tpData);
                ReplayState.tradeAreas.push(tpBox);
            }
        }

        // SL region (red) - between the entry and the SL.
        if (trade.stop_loss) {
            const slBox = ReplayState.chart.addBaselineSeries({
                baseValue: { type: 'price', price: trade.entry_price },
                // LONG: red at the bottom, SHORT: red at the top
                topLineColor: isLong ? 'rgba(0, 0, 0, 0)' : slColor + '0)',
                topFillColor1: isLong ? 'rgba(0, 0, 0, 0)' : slColor + '0.3)',
                topFillColor2: isLong ? 'rgba(0, 0, 0, 0)' : slColor + '0.1)',
                bottomLineColor: isLong ? slColor + '0)' : 'rgba(0, 0, 0, 0)',
                bottomFillColor1: isLong ? slColor + '0.3)' : 'rgba(0, 0, 0, 0)',
                bottomFillColor2: isLong ? slColor + '0.1)' : 'rgba(0, 0, 0, 0)',
                lineWidth: 0,
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false
            });

            // SL box data - always use the SL value
            const slData = [];
            const timeStep = (trade.exit_time - trade.entry_time) / 10;
            for (let t = trade.entry_time; t <= trade.exit_time; t += timeStep) {
                slData.push({
                    time: Math.floor(t),
                    value: trade.stop_loss
                });
            }
            if (slData.length > 0) {
                slBox.setData(slData);
                ReplayState.tradeAreas.push(slBox);
            }
        }
    });
}

function updateTradesPanel(trades) {
    const panel = document.getElementById('trades-panel');
    if (!panel) return;

    // Stats hesapla
    const totalTrades = trades.length;
    const wins = trades.filter(t => t.win).length;
    const losses = totalTrades - wins;
    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(1) : 0;
    const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const pnlClass = totalPnl >= 0 ? 'positive' : 'negative';
    const pnlText = totalPnl >= 0 ? `+$${totalPnl.toFixed(2)}` : `$${totalPnl.toFixed(2)}`;

    // Exit reason istatistikleri
    const exitReasons = {};
    trades.forEach(t => {
        const reason = t.exit_reason || 'OTHER';
        exitReasons[reason] = (exitReasons[reason] || 0) + 1;
    });

    // Trade list - last 5 trades
    const recentTrades = trades.slice(-5).reverse();

    let html = `
        <div class="stat-item">
            <span class="stat-label">Trades</span>
            <span class="stat-value">${totalTrades} <small style="color: var(--text-secondary);">(${wins}W/${losses}L)</small></span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Win Rate</span>
            <span class="stat-value">${winRate}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total P&L</span>
            <span class="stat-value ${pnlClass}">${pnlText}</span>
        </div>
    `;

    // Exit reason breakdown
    if (Object.keys(exitReasons).length > 0) {
        html += `<hr style="border-color: var(--border-color); margin: 8px 0;">`;
        html += `<div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;">Exit Reasons:</div>`;

        const exitIcons = { 'TP': '🎯', 'SL': '🛑', 'BE': '⚖️', 'TRAILING': '📈', 'SIGNAL': '📊', 'TIMEOUT': '⏰', 'PE1': '📤', 'PE2': '📤', 'PE3': '📤' };
        for (const [reason, count] of Object.entries(exitReasons)) {
            // PE specific icon (PE1, PE2, PE3)
            const icon = reason.toUpperCase().startsWith('PE') ? '📤' : (exitIcons[reason.toUpperCase()] || '✖️');
            const pct = ((count / totalTrades) * 100).toFixed(0);
            html += `
                <div class="stat-item" style="font-size: 11px;">
                    <span>${icon} ${reason}</span>
                    <span class="stat-value">${count} (${pct}%)</span>
                </div>
            `;
        }
    }

    // Recent trades with details
    if (recentTrades.length > 0) {
        html += `<hr style="border-color: var(--border-color); margin: 8px 0;">`;
        html += `<div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;">Recent Trades:</div>`;

        recentTrades.forEach(trade => {
            const isLong = trade.side === 'LONG';
            const isWin = trade.pnl >= 0;
            const sideClass = isLong ? 'positive' : 'negative';
            const pnlTradeClass = isWin ? 'positive' : 'negative';
            const pnlTradeText = isWin ? `+$${trade.pnl.toFixed(2)}` : `-$${Math.abs(trade.pnl).toFixed(2)}`;

            // Exit reason icon
            const exitIcons = { 'TP': '🎯', 'SL': '🛑', 'BE': '⚖️', 'TRAILING': '📈' };
            const exitReason = trade.exit_reason?.toUpperCase() || '';
            const exitIcon = exitReason.startsWith('PE') ? '📤' : (exitIcons[exitReason] || '');

            // Duration
            const duration = trade.duration_minutes || 0;
            const durationText = duration >= 60 ? `${(duration/60).toFixed(1)}h` : `${duration}m`;

            html += `
                <div class="stat-item" style="font-size: 11px; padding: 4px 0;">
                    <span>
                        <span class="${sideClass}" style="font-weight: 600;">${trade.side}</span>
                        <span style="color: var(--text-secondary); margin-left: 4px;">${exitIcon}${trade.exit_reason || ''}</span>
                    </span>
                    <span class="stat-value ${pnlTradeClass}">${pnlTradeText}</span>
                </div>
                <div style="font-size: 10px; color: var(--text-secondary); margin-top: -2px; margin-bottom: 4px;">
                    Entry: $${trade.entry_price?.toFixed(2) || '-'} → Exit: $${trade.exit_price?.toFixed(2) || '-'} (${durationText})
                </div>
            `;
        });
    }

    panel.innerHTML = html;
}

function updateState(data) {
    ReplayState.currentBar = data.current_position;

    // Timeline update
    document.getElementById('timeline-slider').value = data.current_position;
    document.getElementById('current-bar').textContent = `Bar: ${data.current_position}`;

    // Show date (from the last candle).
    if (data.candles && data.candles.length > 0) {
        const lastCandle = data.candles[data.candles.length - 1];
        const date = new Date(lastCandle.time * 1000);
        document.getElementById('current-date').textContent = date.toLocaleString('tr-TR');
    }
}

// ============================================
// PLAYBACK CONTROLS
// ============================================
function togglePlayPause() {
    if (ReplayState.isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!ReplayState.sessionId) return;

    ReplayState.isPlaying = true;
    updatePlayButton();

    // Backend'e bildir
    fetch(`/api/replay/sessions/${ReplayState.sessionId}/play`, { method: 'POST' });

    // Start interval
    const interval = 1000 / ReplayState.speed;
    ReplayState.playInterval = setInterval(async () => {
        if (ReplayState.currentBar >= ReplayState.totalBars - 1) {
            stopPlayback();
            return;
        }
        await step(1);
    }, interval);
}

function stopPlayback() {
    ReplayState.isPlaying = false;
    updatePlayButton();

    if (ReplayState.playInterval) {
        clearInterval(ReplayState.playInterval);
        ReplayState.playInterval = null;
    }

    // Backend'e bildir
    if (ReplayState.sessionId) {
        fetch(`/api/replay/sessions/${ReplayState.sessionId}/pause`, { method: 'POST' });
    }
}

function updatePlayButton() {
    const btn = document.getElementById('btn-play-pause');
    if (ReplayState.isPlaying) {
        btn.textContent = '⏸️';
        btn.classList.add('playing');
    } else {
        btn.textContent = '▶️';
        btn.classList.remove('playing');
    }
}

async function step(direction) {
    if (!ReplayState.sessionId) return;

    const newPos = Math.max(0, Math.min(
        ReplayState.currentBar + direction,
        ReplayState.totalBars - 1
    ));

    if (newPos === ReplayState.currentBar) return;

    try {
        const response = await fetch(
            `/api/replay/sessions/${ReplayState.sessionId}/step`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ direction: direction })
            }
        );

        const data = await response.json();
        if (data.status === 'success') {
            await loadCandles();
            updateIndicatorValues(data.data.state);
        }
    } catch (error) {
        console.error('Step hatası:', error);
    }
}

// Cache the trade bar indices.
let cachedTradeBarIndices = null;
let cachedTradesSessionId = null;

async function loadTradeBarIndices() {
    /**
     * Backtest parquet'ten trade'leri yükle ve bar index'lerini hesapla
     * Bu fonksiyon session başladığında bir kere çağrılır
     */
    if (cachedTradesSessionId === ReplayState.sessionId && cachedTradeBarIndices) {
        return cachedTradeBarIndices;
    }

    cachedTradeBarIndices = [];
    cachedTradesSessionId = ReplayState.sessionId;

    try {
        // First, get all the candles (for timestamp -> index mapping).
        // Get the symbol and timeframe from the session information.
        const sessionResponse = await fetch(`/api/replay/sessions/${ReplayState.sessionId}`);
        const sessionData = await sessionResponse.json();

        if (sessionData.status !== 'success') {
            console.error('Session bilgisi alınamadı');
            return [];
        }

        const { symbol, timeframe } = sessionData.data.session;
        console.log(`📊 Trade'ler yükleniyor: ${symbol} ${timeframe}`);

        // Get backtest trades directly from parquet.
        const tradesResponse = await fetch(`/api/backtest/trades?symbol=${symbol}&timeframe=${timeframe}`);

        if (!tradesResponse.ok) {
            console.log('Backtest trades endpoint yok, candles endpoint deneniyor...');
            // Fallback: candles endpoint'inden al
            const candlesResponse = await fetch(`/api/replay/sessions/${ReplayState.sessionId}/candles?start=0&limit=50000`);
            const candlesData = await candlesResponse.json();

            if (candlesData.status === 'success' && candlesData.data.trades) {
                const trades = candlesData.data.trades;
                const candles = candlesData.data.candles || [];

                // Timestamp -> index mapping
                const tsToIndex = {};
                candles.forEach((c, idx) => tsToIndex[c.time] = idx);

                // Convert trades to bar index.
                trades.forEach(trade => {
                    const entryTs = trade.entry_time;
                    let closestIdx = 0;
                    let minDiff = Infinity;

                    for (const [ts, idx] of Object.entries(tsToIndex)) {
                        const diff = Math.abs(parseInt(ts) - entryTs);
                        if (diff < minDiff) {
                            minDiff = diff;
                            closestIdx = idx;
                        }
                    }
                    cachedTradeBarIndices.push(closestIdx);
                });
            }
        } else {
            const tradesData = await tradesResponse.json();
            if (tradesData.status === 'success' && tradesData.data.trade_bar_indices) {
                cachedTradeBarIndices = tradesData.data.trade_bar_indices;
            }
        }

        // Sort
        cachedTradeBarIndices.sort((a, b) => a - b);
        console.log(`✅ ${cachedTradeBarIndices.length} trade yüklendi:`, cachedTradeBarIndices.slice(0, 10));

    } catch (error) {
        console.error('Trade bar indices yüklenemedi:', error);
    }

    return cachedTradeBarIndices;
}

async function jumpToTrade(direction) {
    /**
     * Bir sonraki veya önceki trade'e atla
     * direction: 1 = sonraki trade, -1 = önceki trade
     */
    if (!ReplayState.sessionId) return;

    const tradeBarIndices = await loadTradeBarIndices();

    if (!tradeBarIndices || tradeBarIndices.length === 0) {
        showToast('Trade not found', 'warning');
        await step(direction);
        return;
    }

    const currentBar = ReplayState.currentBar;
    let targetBar = null;

    if (direction > 0) {
        // Next trade: the first trade that is greater than the current bar.
        for (const barIdx of tradeBarIndices) {
            if (barIdx > currentBar) {
                targetBar = barIdx;
                break;
            }
        }
    } else {
        // Previous trade: The last trade was smaller than the current bar.
        for (let i = tradeBarIndices.length - 1; i >= 0; i--) {
            if (tradeBarIndices[i] < currentBar) {
                targetBar = tradeBarIndices[i];
                break;
            }
        }
    }

    if (targetBar !== null) {
        await seek(targetBar);
        console.log(`🎯 Trade'e atlandı: bar ${targetBar}`);
    } else {
        showToast(direction > 0 ? 'Next trade not available.' : 'Previous trade does not exist.', 'info');
    }
}

async function seek(position) {
    if (!ReplayState.sessionId) return;

    try {
        const response = await fetch(
            `/api/replay/sessions/${ReplayState.sessionId}/seek`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ position: position })
            }
        );

        const data = await response.json();
        if (data.status === 'success') {
            await loadCandles();
            updateIndicatorValues(data.data.state);
        }
    } catch (error) {
        console.error('Seek hatası:', error);
    }
}

function onTimelineChange(e) {
    stopPlayback();
    seek(parseInt(e.target.value));
}

function onSpeedChange(e) {
    ReplayState.speed = parseFloat(e.target.value);

    // If playing, restart with the new speed.
    if (ReplayState.isPlaying) {
        stopPlayback();
        startPlayback();
    }

    // Backend'e bildir
    if (ReplayState.sessionId) {
        fetch(`/api/replay/sessions/${ReplayState.sessionId}/speed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed: ReplayState.speed })
        });
    }
}

// ============================================
// STATS UPDATE
// ============================================
function updateIndicatorValues(state) {
    if (!state || !state.indicators) return;

    const indicators = state.indicators;

    // Update all indicators dynamically.
    for (const [name, value] of Object.entries(indicators)) {
        const el = document.getElementById(`ind-${name}`);
        if (!el) continue;

        const nameLower = name.toLowerCase();

        // RSI and oscillator specific formatting.
        if (nameLower.includes('rsi') || nameLower.includes('stoch')) {
            el.textContent = value.toFixed(2);
            el.className = 'stat-value';
            if (value >= 70) el.classList.add('negative');  // Overbought
            else if (value <= 30) el.classList.add('positive');  // Oversold
        }
        // ADX for custom formatting
        else if (nameLower.includes('adx')) {
            el.textContent = value.toFixed(2);
            el.className = 'stat-value';
            if (value >= 25) el.classList.add('positive');  // Strong trend
        }
        // Price-based indicators (EMA, SMA, BB, etc.)
        else {
            el.textContent = value.toLocaleString('en-US', { maximumFractionDigits: 2 });
        }
    }
}

// ============================================
// HELPERS
// ============================================
function showLoading(message) {
    // utils.js'den
    if (typeof LoadingSpinner !== 'undefined') {
        LoadingSpinner.show(message);
    }
}

function hideLoading() {
    if (typeof LoadingSpinner !== 'undefined') {
        LoadingSpinner.hide();
    }
}

function showSuccess(message) {
    if (typeof showToast !== 'undefined') {
        showToast(message, 'success');
    } else {
        console.log('✅', message);
    }
}

function showError(message) {
    if (typeof showToast !== 'undefined') {
        showToast(message, 'error');
    } else {
        console.error('❌', message);
    }
}
