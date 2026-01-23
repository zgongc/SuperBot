/**
 * Candlestick Pattern Analysis JavaScript
 */

// Global variables
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let analysisData = null;
let currentFilter = 'all';
let isDateMode = false;
let patternConfig = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    initChart();
    loadSymbols();
});

/**
 * Load pattern config from API
 */
async function loadConfig() {
    try {
        const response = await fetch('/api/patterns/config');
        const data = await response.json();

        if (data.status === 'success' && data.data) {
            patternConfig = data.data;

            // Build filter checkboxes
            buildFilterCheckboxes(patternConfig);
        }
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

/**
 * Build filter checkboxes from config
 */
function buildFilterCheckboxes(config) {
    const container = document.getElementById('pattern-filters');
    if (!container) return;

    // Group patterns by type
    const bullish = [];
    const bearish = [];
    const neutral = [];

    for (const [code, info] of Object.entries(config)) {
        const item = { code, ...info };
        if (info.type === 'bullish') bullish.push(item);
        else if (info.type === 'bearish') bearish.push(item);
        else neutral.push(item);
    }

    let html = '';

    // Bullish patterns
    html += '<div class="filter-group"><div class="filter-group-title bullish">Bullish</div><div class="filter-items">';
    bullish.forEach(p => {
        html += `
            <label class="filter-checkbox">
                <input type="checkbox" id="filter-${p.code}" ${p.enabled ? 'checked' : ''} disabled>
                <span class="filter-item filter-bullish" title="${p.description}">${p.name}</span>
            </label>
        `;
    });
    html += '</div></div>';

    // Bearish patterns
    html += '<div class="filter-group"><div class="filter-group-title bearish">Bearish</div><div class="filter-items">';
    bearish.forEach(p => {
        html += `
            <label class="filter-checkbox">
                <input type="checkbox" id="filter-${p.code}" ${p.enabled ? 'checked' : ''} disabled>
                <span class="filter-item filter-bearish" title="${p.description}">${p.name}</span>
            </label>
        `;
    });
    html += '</div></div>';

    // Neutral patterns
    html += '<div class="filter-group"><div class="filter-group-title neutral">Neutral</div><div class="filter-items">';
    neutral.forEach(p => {
        html += `
            <label class="filter-checkbox">
                <input type="checkbox" id="filter-${p.code}" ${p.enabled ? 'checked' : ''} disabled>
                <span class="filter-item filter-neutral" title="${p.description}">${p.name}</span>
            </label>
        `;
    });
    html += '</div></div>';

    container.innerHTML = html;
}

/**
 * Initialize LightweightCharts
 */
function initChart() {
    const container = document.getElementById('pattern-chart');
    if (!container) return;

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 500,
        layout: {
            background: { color: 'transparent' },
            textColor: '#9B9B9B',
        },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.3)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.3)' },
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

    // Candlestick series
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });

    // Volume series
    volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.85,
            bottom: 0,
        },
    });

    // Resize handler
    window.addEventListener('resize', () => {
        if (chart) {
            chart.applyOptions({ width: container.clientWidth });
        }
    });
}

/**
 * Load available symbols from data/parquets folder
 */
async function loadSymbols() {
    const select = document.getElementById('pattern-symbol');

    try {
        // Get symbols from data/parquets folder (downloaded data)
        const response = await fetch('/api/data/symbols');
        const data = await response.json();

        if (data.status === 'success' && data.data.symbols && data.data.symbols.length > 0) {
            select.innerHTML = '<option value="">Select symbol...</option>';
            data.data.symbols.forEach(symbol => {
                select.innerHTML += `<option value="${symbol}">${symbol}</option>`;
            });
            // Pre-select BTCUSDT if available
            if (select.querySelector('option[value="BTCUSDT"]')) {
                select.value = 'BTCUSDT';
            } else {
                // Otherwise select first symbol
                select.selectedIndex = 1;
            }
        } else {
            // Fallback to common symbols if no data downloaded
            select.innerHTML = `
                <option value="">Select a symbol...</option>
                <option value="BTCUSDT" selected>BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
                <option value="BNBUSDT">BNBUSDT</option>
                <option value="SOLUSDT">SOLUSDT</option>
            `;
        }
    } catch (error) {
        console.error('Failed to load symbols:', error);
        // Fallback
        select.innerHTML = `
            <option value="">Select a symbol...</option>
            <option value="BTCUSDT" selected>BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
        `;
    }
}

/**
 * Toggle between bar count and date mode
 */
function toggleDataMode() {
    isDateMode = !isDateMode;

    const barsInputs = document.getElementById('bars-mode-inputs');
    const dateInputs = document.getElementById('date-mode-inputs');
    const modeText = document.getElementById('data-mode-text');

    if (isDateMode) {
        barsInputs.style.display = 'none';
        dateInputs.style.display = 'flex';
        modeText.textContent = 'Date Range';
    } else {
        barsInputs.style.display = 'block';
        dateInputs.style.display = 'none';
        modeText.textContent = 'Bar Count';
    }
}

/**
 * Toggle end date input
 */
function toggleEndDate() {
    const checkbox = document.getElementById('end-now');
    const endDateInput = document.getElementById('pattern-end-date');
    endDateInput.disabled = checkbox.checked;

    if (checkbox.checked) {
        endDateInput.value = '';
    }
}

/**
 * Run pattern analysis
 */
async function analyzePatterns() {
    const symbol = document.getElementById('pattern-symbol').value;
    if (!symbol) {
        showToast('Please select a symbol', 'warning');
        return;
    }

    const timeframe = document.getElementById('pattern-timeframe').value;
    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    try {
        const body = {
            symbol: symbol,
            timeframe: timeframe
        };

        if (isDateMode) {
            const startDate = document.getElementById('pattern-start-date').value;
            const useNow = document.getElementById('end-now').checked;

            if (!startDate) {
                showToast('Please select a start date', 'warning');
                btn.disabled = false;
                btn.textContent = 'Analyze';
                return;
            }

            body.start_date = startDate;
            body.end_date = useNow ? new Date().toISOString().split('T')[0] : document.getElementById('pattern-end-date').value;
        } else {
            body.limit = parseInt(document.getElementById('pattern-limit').value) || 500;
        }

        const response = await fetch('/api/patterns/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await response.json();

        if (data.status === 'success') {
            analysisData = data.data;
            updateChart(data.data);
            updateStatistics(data.data);
            updatePatternFrequency(data.data);
            updatePatternsList(data.data);
            showToast('Analysis completed', 'success');
        } else {
            showToast(data.message || 'Analysis failed', 'error');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Analysis failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze';
    }
}

/**
 * Update chart with analysis data
 */
function updateChart(data) {
    if (!chart || !candleSeries) return;

    // Set candle data
    if (data.candles && data.candles.length > 0) {
        candleSeries.setData(data.candles);

        // Volume data
        const volumeData = data.candles.map(c => ({
            time: c.time,
            value: c.volume,
            color: c.close >= c.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
        }));
        volumeSeries.setData(volumeData);
    }

    // Set markers for patterns
    if (data.annotations && data.annotations.markers) {
        candleSeries.setMarkers(data.annotations.markers);
    }

    // Fit content
    chart.timeScale().fitContent();
}

/**
 * Update statistics display
 */
function updateStatistics(data) {
    const summary = data.summary || {};

    document.getElementById('stat-total').textContent = summary.total_patterns || 0;
    document.getElementById('stat-bullish').textContent = summary.bullish || 0;
    document.getElementById('stat-bearish').textContent = summary.bearish || 0;
    document.getElementById('stat-neutral').textContent = summary.neutral || 0;

    // Update bias indicator
    const biasIndicator = document.getElementById('bias-indicator');
    const biasText = document.getElementById('bias-text');
    const bias = summary.bias || 'neutral';

    biasIndicator.className = 'bias-indicator ' + bias;
    biasText.textContent = bias.charAt(0).toUpperCase() + bias.slice(1);
}

/**
 * Update pattern frequency display
 */
function updatePatternFrequency(data) {
    const container = document.getElementById('pattern-frequency');
    const counts = data.pattern_counts || {};

    if (Object.keys(counts).length === 0) {
        container.innerHTML = '<p class="empty-message">No patterns detected</p>';
        return;
    }

    // Sort by count
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);

    let html = '';
    sorted.forEach(([name, count]) => {
        // Get pattern type for coloring
        const pattern = data.patterns.find(p => p.code === name);
        const type = pattern ? pattern.type : 'neutral';

        html += `
            <div class="pattern-freq-item">
                <span class="pattern-badge ${type}">${formatPatternName(name)}</span>
                <span class="count">${count}</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Update patterns list
 */
function updatePatternsList(data) {
    const container = document.getElementById('patterns-list');
    let patterns = data.recent_patterns || [];

    // Apply filter
    if (currentFilter !== 'all') {
        patterns = patterns.filter(p => p.type === currentFilter);
    }

    if (patterns.length === 0) {
        container.innerHTML = '<p class="empty-message">No patterns found</p>';
        return;
    }

    // Sort by time (most recent first)
    patterns.sort((a, b) => b.time - a.time);

    let html = '';
    patterns.forEach(p => {
        const time = new Date(p.time).toLocaleString();

        html += `
            <div class="pattern-item" onclick="scrollToPattern(${p.time})">
                <div>
                    <div class="pattern-name">${p.name}</div>
                    <div class="pattern-time">${time}</div>
                </div>
                <div>
                    <span class="pattern-badge ${p.type}">${p.type}</span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Filter patterns by type
 */
function filterPatterns(filter) {
    currentFilter = filter;

    // Update button states
    document.querySelectorAll('.filter-patterns .btn-filter').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.filter === filter) {
            btn.classList.add('active');
        }
    });

    // Re-render patterns list
    if (analysisData) {
        updatePatternsList(analysisData);
    }
}

/**
 * Scroll chart to pattern time
 */
function scrollToPattern(timestamp) {
    if (!chart) return;

    const time = timestamp / 1000; // Convert to seconds
    chart.timeScale().scrollToPosition(-5, false);

    // Find the candle and center on it
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (visibleRange) {
        chart.timeScale().setVisibleRange({
            from: time - 50 * 60, // 50 bars before
            to: time + 10 * 60   // 10 bars after
        });
    }
}

/**
 * Format pattern name for display
 */
function formatPatternName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Check if toast container exists
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        padding: 12px 20px;
        margin-bottom: 10px;
        border-radius: 8px;
        color: white;
        font-size: 14px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        background: ${type === 'success' ? '#26a69a' : type === 'error' ? '#ef5350' : type === 'warning' ? '#ff9800' : '#2196f3'};
    `;
    toast.textContent = message;

    container.appendChild(toast);

    // Animate in
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(0)';
    }, 10);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
