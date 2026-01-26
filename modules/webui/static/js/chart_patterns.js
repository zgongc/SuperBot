/**
 * Trendline Breakout Analysis Module
 * Support/Resistance trendline breakout detection
 */

// Chart instance
let chart = null;
let candleSeries = null;
let markers = [];
let pricelines = [];
let trendlineSeriesList = [];

// Analysis data
let analysisData = null;
let candleData = [];

// Visibility settings
let visibilitySettings = {
    swing: true,
    resistance: true,
    support: true,
    breakouts: true,
    targets: true
};

// Data mode: 'bars' or 'date'
let dataMode = 'bars';

/**
 * Toggle data mode (bars ↔ date)
 */
function toggleDataMode() {
    dataMode = dataMode === 'bars' ? 'date' : 'bars';
    document.getElementById('bars-mode-inputs').style.display = dataMode === 'bars' ? 'block' : 'none';
    document.getElementById('date-mode-inputs').style.display = dataMode === 'date' ? 'flex' : 'none';
    document.getElementById('data-mode-text').textContent = dataMode === 'bars' ? 'Bar Count' : 'Date Range';
}

/**
 * Toggle end date input based on "Now" checkbox
 */
function toggleEndDate() {
    const checkbox = document.getElementById('end-now');
    const endDateInput = document.getElementById('cp-end-date');
    endDateInput.disabled = checkbox.checked;
    if (checkbox.checked) endDateInput.value = '';
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
    initChart();
    loadSymbols();
    setupEventListeners();
    initDateInputs();
});

/**
 * Initialize date inputs with default values
 */
function initDateInputs() {
    const startInput = document.getElementById('cp-start-date');
    const endInput = document.getElementById('cp-end-date');

    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];

    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    const weekAgoStr = weekAgo.toISOString().split('T')[0];

    startInput.value = weekAgoStr;
    endInput.value = todayStr;
}

/**
 * Initialize LightweightCharts
 */
function initChart() {
    const container = document.getElementById('cp-chart');

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 500,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(42, 46, 57, 0.5)',
        },
        timeScale: {
            borderColor: 'rgba(42, 46, 57, 0.5)',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });

    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });
}

/**
 * Load available symbols
 */
async function loadSymbols() {
    const select = document.getElementById('cp-symbol');

    try {
        const response = await fetch('/api/data/symbols');
        const data = await response.json();

        if (data.status === 'success' && data.data.symbols && data.data.symbols.length > 0) {
            select.innerHTML = '<option value="">Select symbol...</option>';
            data.data.symbols.forEach(symbol => {
                select.innerHTML += `<option value="${symbol}">${symbol}</option>`;
            });
            if (select.querySelector('option[value="BTCUSDT"]')) {
                select.value = 'BTCUSDT';
            } else {
                select.selectedIndex = 1;
            }
        } else {
            select.innerHTML = `
                <option value="">Select a symbol...</option>
                <option value="BTCUSDT" selected>BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
            `;
        }
    } catch (error) {
        console.error('Failed to load symbols:', error);
        select.innerHTML = `
            <option value="BTCUSDT" selected>BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
        `;
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    document.getElementById('btn-analyze').addEventListener('click', runAnalysis);
    document.getElementById('cp-symbol').addEventListener('change', resetChart);
    document.getElementById('cp-timeframe').addEventListener('change', resetChart);
}

/**
 * Reset chart
 */
function resetChart() {
    clearAllDrawings();
    candleData = [];
    candleSeries.setData([]);
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });
    analysisData = null;

    // Reset stats
    document.getElementById('stat-bullish').textContent = '0';
    document.getElementById('stat-bearish').textContent = '0';
    document.getElementById('stat-trendlines').textContent = '0';
    document.getElementById('stat-swings').textContent = '0';

    // Reset lists
    document.getElementById('trendlines-list').innerHTML = '<p class="empty-message">Waiting for analysis...</p>';
    document.getElementById('patterns-list').innerHTML = '<p class="empty-message">Waiting for analysis...</p>';
}

/**
 * Run Trendline Breakout Analysis
 */
async function runAnalysis() {
    const symbol = document.getElementById('cp-symbol').value;
    const timeframe = document.getElementById('cp-timeframe').value;

    if (!symbol) {
        alert('Please select a symbol');
        return;
    }

    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    let requestBody = { symbol, timeframe };

    if (dataMode === 'bars') {
        requestBody.limit = parseInt(document.getElementById('cp-limit').value);
    } else {
        const startDate = document.getElementById('cp-start-date').value;
        const endNow = document.getElementById('end-now').checked;
        const endDate = document.getElementById('cp-end-date').value;

        if (!startDate) {
            alert('Please select a start date');
            btn.disabled = false;
            btn.textContent = 'Analyze';
            return;
        }

        requestBody.start_date = startDate;
        if (!endNow && endDate) {
            requestBody.end_date = endDate;
        }
    }

    try {
        const response = await fetch('/api/chart-patterns/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (data.status === 'success') {
            analysisData = data.data;
            candleData = analysisData.candles || [];
            updateChart();
            updateStats();
            updateTrendlinesList();
            updateBreakoutsList();
        } else {
            alert('Analysis error: ' + (data.message || data.data?.error));
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error during analysis: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze';
    }
}

/**
 * Update chart with candle data and trendlines
 */
function updateChart() {
    if (candleData.length === 0) {
        console.warn('No candle data to display');
        return;
    }

    chart.priceScale('right').applyOptions({ autoScale: true });
    candleSeries.setData(candleData);
    clearAllDrawings();

    if (analysisData) {
        drawTrendlines();
        drawSwingMarkers();
        drawBreakouts();
    }

    chart.timeScale().fitContent();
}

/**
 * Clear all chart drawings
 */
function clearAllDrawings() {
    markers = [];
    candleSeries.setMarkers([]);

    pricelines.forEach(line => {
        try { candleSeries.removePriceLine(line); } catch(e) {}
    });
    pricelines = [];

    trendlineSeriesList.forEach(series => {
        try { chart.removeSeries(series); } catch(e) {}
    });
    trendlineSeriesList = [];
}

/**
 * Draw trendlines on chart
 */
function drawTrendlines() {
    if (!analysisData || !analysisData.trendlines) return;

    const trendlines = analysisData.trendlines || [];

    trendlines.forEach(tl => {
        // Check visibility
        if (tl.type === 'resistance' && !visibilitySettings.resistance) return;
        if (tl.type === 'support' && !visibilitySettings.support) return;

        const color = tl.type === 'support' ? '#26a69a' : '#ef5350';

        const lineSeries = chart.addLineSeries({
            color: color,
            lineWidth: 2,
            lineStyle: 0,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        // Trendline has start and end points with different values (sloped line)
        lineSeries.setData([
            { time: tl.start_time, value: tl.start_value },
            { time: tl.end_time, value: tl.end_value }
        ]);

        trendlineSeriesList.push(lineSeries);
    });
}

/**
 * Draw swing point markers
 */
function drawSwingMarkers() {
    if (!visibilitySettings.swing) return;
    if (!analysisData || !analysisData.swings) return;

    const swings = analysisData.swings || [];
    let allMarkers = [];

    swings.forEach(swing => {
        allMarkers.push({
            time: swing.time / 1000,
            position: swing.type === 'high' ? 'aboveBar' : 'belowBar',
            color: swing.type === 'high' ? '#ef5350' : '#26a69a',
            shape: swing.type === 'high' ? 'arrowDown' : 'arrowUp',
            text: swing.label || (swing.type === 'high' ? 'SH' : 'SL'),
            size: 1
        });
    });

    if (allMarkers.length > 0) {
        allMarkers.sort((a, b) => a.time - b.time);
        candleSeries.setMarkers(allMarkers);
    }
}

/**
 * Draw breakout signals
 */
function drawBreakouts() {
    if (!visibilitySettings.breakouts) return;
    if (!analysisData || !analysisData.patterns) return;

    const patterns = analysisData.patterns || [];

    patterns.forEach(p => {
        const color = p.type === 'bullish' ? '#26a69a' : '#ef5350';

        // Draw breakout marker
        if (p.breakout_price && p.end_time) {
            // Add a breakout line at the neckline (trendline value)
            if (p.neckline) {
                const breakoutLine = candleSeries.createPriceLine({
                    price: p.neckline,
                    color: color,
                    lineWidth: 2,
                    lineStyle: 2, // Dashed
                    axisLabelVisible: true,
                    title: p.display_name
                });
                pricelines.push(breakoutLine);
            }
        }

        // Draw target line
        if (visibilitySettings.targets && p.target) {
            const targetLine = candleSeries.createPriceLine({
                price: p.target,
                color: color,
                lineWidth: 1,
                lineStyle: 2,
                axisLabelVisible: true,
                title: `Target ${p.target.toFixed(2)}`
            });
            pricelines.push(targetLine);
        }
    });
}

/**
 * Update chart visibility based on checkboxes
 */
function updateChartVisibility() {
    visibilitySettings.swing = document.getElementById('show-swing').checked;
    visibilitySettings.resistance = document.getElementById('show-resistance').checked;
    visibilitySettings.support = document.getElementById('show-support').checked;
    visibilitySettings.breakouts = document.getElementById('show-breakouts').checked;
    visibilitySettings.targets = document.getElementById('show-targets').checked;

    clearAllDrawings();
    if (candleData.length > 0) {
        candleSeries.setData(candleData);
        drawTrendlines();
        drawSwingMarkers();
        drawBreakouts();
    }
}

/**
 * Update stats section
 */
function updateStats() {
    if (!analysisData || !analysisData.stats) return;

    const stats = analysisData.stats;
    const swings = analysisData.swings || [];

    document.getElementById('stat-bullish').textContent = stats.bullish || 0;
    document.getElementById('stat-bearish').textContent = stats.bearish || 0;
    document.getElementById('stat-trendlines').textContent = stats.trendlines || 0;
    document.getElementById('stat-swings').textContent = swings.length || 0;
}

/**
 * Update trendlines list
 */
function updateTrendlinesList() {
    const container = document.getElementById('trendlines-list');

    if (!analysisData || !analysisData.trendlines || analysisData.trendlines.length === 0) {
        container.innerHTML = '<p class="empty-message">No trendlines detected</p>';
        return;
    }

    const trendlines = analysisData.trendlines;
    let html = '';

    trendlines.forEach((tl, idx) => {
        const typeClass = tl.type === 'support' ? 'bullish' : 'bearish';
        const typeName = tl.type === 'support' ? 'Support' : 'Resistance';
        const slopeDir = tl.slope > 0 ? '↗' : (tl.slope < 0 ? '↘' : '→');
        const score = tl.score ? tl.score.toFixed(0) : '0';
        const violations = tl.violations || 0;
        const isViolated = tl.is_violated ? 'VIOLATED' : 'VALID';
        const statusClass = tl.is_violated ? 'bearish' : 'bullish';

        html += `
            <div class="formation-item" data-type="${tl.type}">
                <span class="formation-type ${typeClass}">${typeName} ${slopeDir}</span>
                <div class="formation-info">
                    <span class="formation-price">${tl.start_value.toFixed(2)} → ${tl.end_value.toFixed(2)}</span>
                    <span class="formation-time">Violations: ${violations}</span>
                </div>
                <div>
                    <span class="badge badge-${typeClass}">${typeName}</span>
                    <span class="badge badge-${statusClass}">${isViolated}</span>
                    <span class="badge">Score: ${score}</span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Update breakouts list
 */
function updateBreakoutsList() {
    const container = document.getElementById('patterns-list');

    if (!analysisData || !analysisData.patterns || analysisData.patterns.length === 0) {
        container.innerHTML = '<p class="empty-message">No breakouts detected</p>';
        return;
    }

    const patterns = analysisData.patterns;
    window.allPatterns = patterns;

    let html = '';
    patterns.forEach(p => {
        const endTime = p.end_time ? new Date(p.end_time).toLocaleString() : '-';
        const typeClass = p.type === 'bullish' ? 'bullish' : 'bearish';
        const targetStr = p.target ? p.target.toFixed(2) : '-';

        html += `
            <div class="formation-item" data-type="${p.type}">
                <span class="formation-type ${typeClass}">${p.display_name}</span>
                <div class="formation-info">
                    <span class="formation-price">Breakout: ${p.breakout_price ? p.breakout_price.toFixed(2) : '-'}</span>
                    <span class="formation-time">${endTime}</span>
                </div>
                <div>
                    <span class="badge badge-${typeClass}">${p.type}</span>
                    <span class="badge">${p.confidence.toFixed(0)}%</span>
                    <span class="badge">Target: ${targetStr}</span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}
