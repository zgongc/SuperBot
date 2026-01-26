/**
 * SMC (Smart Money Concepts) Analysis Module
 */

// Chart instance
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let markers = [];
let pricelines = [];
let bosLines = [];      // BOS horizontal lines
let chochLines = [];    // CHoCH horizontal lines
let lineSeries = [];    // Line series for structure lines
let fvgSeries = [];     // FVG zone line series
let gapSeries = [];     // Gap zone line series
let obSeries = [];      // Order Block zone series

// Analysis data
let analysisData = null;
let candleData = [];

// Visibility settings
let visibilitySettings = {
    bos: true,
    choch: true,
    swing: true,
    fvg: true,
    gap: true,
    ob: false,
    liquidity: false,
    qml: false,
    ftr: true,
    levels: true
};

// Data mode: 'bars' or 'date'
let dataMode = 'bars';

/**
 * Toggle data mode (bars ↔ date)
 */
function toggleDataMode() {
    dataMode = dataMode === 'bars' ? 'date' : 'bars';

    // Update UI
    document.getElementById('bars-mode-inputs').style.display = dataMode === 'bars' ? 'block' : 'none';
    document.getElementById('date-mode-inputs').style.display = dataMode === 'date' ? 'flex' : 'none';
    document.getElementById('data-mode-text').textContent = dataMode === 'bars' ? 'Bar Sayisi' : 'Tarih Araligi';
}

/**
 * Set data mode (bars or date) - for programmatic use
 */
function setDataMode(mode) {
    dataMode = mode;

    // Update UI
    document.getElementById('bars-mode-inputs').style.display = mode === 'bars' ? 'block' : 'none';
    document.getElementById('date-mode-inputs').style.display = mode === 'date' ? 'flex' : 'none';
    document.getElementById('data-mode-text').textContent = mode === 'bars' ? 'Bar Sayisi' : 'Tarih Araligi';
}

/**
 * Toggle end date input based on "Now" checkbox
 */
function toggleEndDate() {
    const checkbox = document.getElementById('end-now');
    const endDateInput = document.getElementById('smc-end-date');

    if (checkbox.checked) {
        endDateInput.disabled = true;
        endDateInput.value = '';
    } else {
        endDateInput.disabled = false;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();  // Load config first to set checkbox states
    initChart();
    loadSymbols();
    setupEventListeners();
    initDateInputs();
});

/**
 * Load analysis config and set initial checkbox states
 */
async function loadConfig() {
    try {
        const response = await fetch('/api/smc/config');
        const data = await response.json();

        if (data.status === 'success' && data.data) {
            const config = data.data;

            // Map config keys to checkbox IDs
            const mapping = {
                'bos': 'show-bos',
                'choch': 'show-choch',
                'swing': 'show-swing',
                'fvg': 'show-fvg',
                'gap': 'show-gap',
                'orderblocks': 'show-ob',
                'liquidity': 'show-liquidity',
                'qml': 'show-qml',
                'ftr': 'show-ftr',
                'levels': 'show-levels'
            };

            // Set checkbox states based on config
            for (const [configKey, checkboxId] of Object.entries(mapping)) {
                const checkbox = document.getElementById(checkboxId);
                if (checkbox && config[configKey]) {
                    checkbox.checked = config[configKey].show;
                    // Also update visibility settings
                    const settingKey = configKey === 'orderblocks' ? 'ob' : configKey;
                    visibilitySettings[settingKey] = config[configKey].show;
                }
            }

            console.log('Config loaded:', config);
        }
    } catch (error) {
        console.error('Failed to load config:', error);
        // Keep default checkbox states on error
    }
}

/**
 * Initialize date inputs with default values
 */
function initDateInputs() {
    const startInput = document.getElementById('smc-start-date');
    const endInput = document.getElementById('smc-end-date');

    // Today's date in YYYY-MM-DD format
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];

    // Yesterday's date
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];

    // Set defaults
    startInput.value = yesterdayStr;
    endInput.value = todayStr;
}

/**
 * Initialize LightweightCharts
 */
function initChart() {
    const container = document.getElementById('smc-chart');

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

    // Candlestick series
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });

    // Handle resize
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });
}

/**
 * Load available symbols from data/parquets directory
 */
async function loadSymbols() {
    const select = document.getElementById('smc-symbol');

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
                <option value="XRPUSDT">XRPUSDT</option>
            `;
        }
    } catch (error) {
        console.error('Failed to load symbols:', error);
        select.innerHTML = `
            <option value="">Select a symbol...</option>
            <option value="BTCUSDT" selected>BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
        `;
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Analyze button
    document.getElementById('btn-analyze').addEventListener('click', runAnalysis);

    // Symbol change - reset chart
    document.getElementById('smc-symbol').addEventListener('change', resetChart);

    // Timeframe change - reset chart
    document.getElementById('smc-timeframe').addEventListener('change', resetChart);

    // Formation filter buttons
    document.querySelectorAll('.btn-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            filterFormations(e.target.dataset.filter);
        });
    });
}

/**
 * Reset chart and clear all data when symbol/timeframe changes
 */
function resetChart() {
    // Clear all drawings
    clearAllDrawings();

    // Clear candle data
    candleData = [];
    candleSeries.setData([]);

    // Reset time scale to clear old range
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });

    // Clear analysis data
    analysisData = null;

    // Reset summary panel
    const trendEl = document.getElementById('summary-trend');
    if (trendEl) {
        trendEl.textContent = '-';
        trendEl.className = 'summary-value';
    }
    const biasEl = document.getElementById('summary-bias');
    if (biasEl) biasEl.textContent = '-';
    const structureEl = document.getElementById('summary-structure');
    if (structureEl) structureEl.textContent = '-';
    const swingHighEl = document.getElementById('summary-swing-high');
    if (swingHighEl) swingHighEl.textContent = '-';
    const swingLowEl = document.getElementById('summary-swing-low');
    if (swingLowEl) swingLowEl.textContent = '-';

    // Reset stats
    const statBos = document.getElementById('stat-bos');
    if (statBos) statBos.textContent = '0';
    const statChoch = document.getElementById('stat-choch');
    if (statChoch) statChoch.textContent = '0';
    const statFvg = document.getElementById('stat-fvg');
    if (statFvg) statFvg.textContent = '0';
    const statActiveFvg = document.getElementById('stat-active-fvg');
    if (statActiveFvg) statActiveFvg.textContent = '0';

    // Clear active FVGs list
    const fvgContainer = document.getElementById('active-fvg-list');
    if (fvgContainer) {
        fvgContainer.innerHTML = '<p class="empty-message">Waiting for analysis...</p>';
    }

    // Clear formations list
    const formationsContainer = document.getElementById('formations-list');
    if (formationsContainer) {
        formationsContainer.innerHTML = '<p class="empty-message">Waiting for analysis...</p>';
    }
}

/**
 * Run SMC Analysis
 */
async function runAnalysis() {
    const symbol = document.getElementById('smc-symbol').value;
    const timeframe = document.getElementById('smc-timeframe').value;

    if (!symbol) {
        alert('Please select a symbol');
        return;
    }

    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    // Build request body based on data mode
    let requestBody = { symbol, timeframe };
    let loadLimit = 500; // Default for candle loading

    if (dataMode === 'bars') {
        const limit = parseInt(document.getElementById('smc-limit').value);
        requestBody.limit = limit;
        loadLimit = limit;
    } else {
        // Date mode
        const startDate = document.getElementById('smc-start-date').value;
        const endNow = document.getElementById('end-now').checked;
        const endDate = document.getElementById('smc-end-date').value;

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
        // For date mode, load more candles to ensure coverage
        loadLimit = 2000;
    }

    try {
        const response = await fetch('/api/smc/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (data.status === 'success') {
            analysisData = data.data;
            await loadCandleData(symbol, timeframe, loadLimit, requestBody.start_date, requestBody.end_date);
            updateChart();
            updateSummary();
            updateStats();
            updateActiveFVGs();
            updateFormations();
            updateLastBar();
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
 * Load candle data for chart
 */
async function loadCandleData(symbol, timeframe, limit, startDate = null, endDate = null) {
    try {
        // Build URL with optional date parameters
        let url = `/api/replay/candles?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`;
        if (startDate) {
            url += `&start_date=${startDate}`;
        }
        if (endDate) {
            url += `&end_date=${endDate}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        console.log('Candle API response:', data);

        if (data.status === 'success' && data.data && data.data.candles) {
            candleData = data.data.candles.map(c => ({
                // Use UTC timestamps directly - chart displays in UTC
                time: c.time || Math.floor(c.timestamp / 1000),
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close
            }));
            console.log('Loaded candles:', candleData.length, 'First:', candleData[0], 'Last:', candleData[candleData.length-1]);
        } else {
            console.error('Candle data not found in response:', data);
            candleData = [];
        }
    } catch (error) {
        console.error('Failed to load candle data:', error);
        candleData = [];
    }
}

/**
 * Update chart with candle data and annotations
 */
function updateChart() {
    if (candleData.length === 0) {
        console.warn('No candle data to display');
        return;
    }

    // Reset scales before setting new data
    chart.priceScale('right').applyOptions({ autoScale: true });

    // Set candle data
    candleSeries.setData(candleData);

    // Clear all previous drawings
    clearAllDrawings();

    // Add annotations from analysis based on visibility settings
    if (analysisData) {
        drawFormations();
    }

    // Fit content to show all data
    chart.timeScale().fitContent();
}

/**
 * Clear all chart drawings
 */
function clearAllDrawings() {
    // Clear markers
    markers = [];
    candleSeries.setMarkers([]);

    // Clear price lines (swing levels)
    pricelines.forEach(line => {
        try { candleSeries.removePriceLine(line); } catch(e) {}
    });
    pricelines = [];

    // Clear line series (BOS/CHoCH lines)
    lineSeries.forEach(series => {
        try { chart.removeSeries(series); } catch(e) {}
    });
    lineSeries = [];

    // Clear FVG series
    fvgSeries.forEach(series => {
        try { chart.removeSeries(series); } catch(e) {}
    });
    fvgSeries = [];

    gapSeries.forEach(series => {
        try { chart.removeSeries(series); } catch(e) {}
    });
    gapSeries = [];

    obSeries.forEach(series => {
        try { chart.removeSeries(series); } catch(e) {}
    });
    obSeries = [];
}

/**
 * Draw all formations based on visibility settings
 */
function drawFormations() {
    if (!analysisData) return;

    const formations = analysisData.formations || {};
    const annotations = analysisData.annotations || {};
    let allMarkers = [];

    // BOS lines (labels are drawn on the line series itself)
    if (visibilitySettings.bos && formations.bos) {
        formations.bos.forEach(bos => {
            drawStructureLine(bos, 'bos');
        });
    }

    // CHoCH lines (labels are drawn on the line series itself)
    if (visibilitySettings.choch && formations.choch) {
        formations.choch.forEach(choch => {
            drawStructureLine(choch, 'choch');
        });
    }

    // Swing markers - use UTC timestamps directly
    // Filter for swing markers: SH, SL, HH, HL, LH, LL, H, L
    if (visibilitySettings.swing && annotations.markers) {
        const swingLabels = ['SH', 'SL', 'HH', 'HL', 'LH', 'LL', 'H', 'L'];
        const swingMarkers = annotations.markers
            .filter(m => swingLabels.includes(m.text));
        allMarkers = allMarkers.concat(swingMarkers);
    }

    // FVG zones - draw limited number based on config (max_box per type, most recent)
    // Filled FVGs extend only to fill point, unfilled extend to last candle
    if (visibilitySettings.fvg && formations.fvg) {
        const maxPerType = analysisData?.fvg_max_box || 10;
        const bullishFvgs = formations.fvg.filter(f => f.type === 'bullish').slice(-maxPerType);
        const bearishFvgs = formations.fvg.filter(f => f.type === 'bearish').slice(-maxPerType);
        const fvgsToShow = [...bullishFvgs, ...bearishFvgs];
        fvgsToShow.forEach(fvg => {
            drawFVGZone(fvg);
        });
    }

    // Gap zones - draw limited number (max 5 bullish + 5 bearish, most recent)
    console.log('Gap visibility:', visibilitySettings.gap, 'formations.gap:', formations.gap?.length);
    if (visibilitySettings.gap && formations.gap) {
        const maxPerType = 5;
        const bullishGaps = formations.gap.filter(g => g.type === 'bullish').slice(-maxPerType);
        const bearishGaps = formations.gap.filter(g => g.type === 'bearish').slice(-maxPerType);
        const gapsToShow = [...bullishGaps, ...bearishGaps];
        console.log('Drawing', gapsToShow.length, 'gaps (max', maxPerType * 2, ')');
        gapsToShow.forEach(gap => {
            drawGapZone(gap);
        });
    }

    // Order Block zones
    console.log('OB visibility:', visibilitySettings.ob, 'formations:', formations.ob);
    if (visibilitySettings.ob && formations.ob) {
        formations.ob.forEach(ob => {
            console.log('OB formation:', ob);
            // Draw both active and broken OBs (broken ones will be shorter)
            drawOBZone(ob);
        });
    }

    // Liquidity level lines
    if (visibilitySettings.liquidity && formations.liquidity) {
        formations.liquidity.forEach(liq => {
            if (!liq.swept) {
                drawLiquidityLine(liq);
            }
        });
    }

    // QML patterns - draw full pattern line with marker
    if (visibilitySettings.qml && formations.qml) {
        console.log('QML formations:', formations.qml);
        formations.qml.forEach(qml => {
            // Draw the full QML pattern (Left Shoulder → Head → Right Shoulder → Break)
            drawQMLPattern(qml);
        });
    }

    // FTR/FTB zones
    if (visibilitySettings.ftr && formations.ftr_zones) {
        console.log('FTR zones:', formations.ftr_zones);
        formations.ftr_zones.forEach(zone => {
            // Only draw active (non-invalidated) zones
            if (!zone.invalidated) {
                drawFTRZone(zone);
            }
        });
    }

    // Set all markers (allMarkers from swing + markers from OB/other drawings)
    const combinedMarkers = [
        ...allMarkers.map(m => ({
            time: m.time,
            position: m.position,
            color: m.color,
            shape: m.shape,
            text: m.text,
            size: m.size || 1
        })),
        ...markers  // OB markers added by drawOBZone
    ];

    if (combinedMarkers.length > 0) {
        // Sort by time
        combinedMarkers.sort((a, b) => a.time - b.time);
        markers = combinedMarkers;  // Update global markers array
        candleSeries.setMarkers(markers);
    }

    // Swing level lines
    if (visibilitySettings.levels && annotations.lines) {
        annotations.lines.forEach(line => {
            const priceLine = candleSeries.createPriceLine({
                price: line.price,
                color: line.color,
                lineWidth: line.lineWidth || 1,
                lineStyle: line.lineStyle || 2,
                axisLabelVisible: true,
                title: line.title || ''
            });
            pricelines.push(priceLine);
        });
    }
}

/**
 * Draw BOS/CHoCH structure line with label in the middle
 * Uses a separate line series for the label text
 */
function drawStructureLine(formation, type) {
    const price = formation.broken_level;
    const breakTime = formation.break_time;
    const swingTime = formation.swing_time;
    const isBullish = formation.type === 'bullish';

    if (!price || !breakTime) return null;

    // Convert milliseconds to seconds (no UTC offset - chart handles timezone display)
    let breakTimeSec = breakTime > 1e12 ? Math.floor(breakTime / 1000) : breakTime;

    // Use swing_time if available, otherwise estimate
    let startTimeSec;
    if (swingTime) {
        startTimeSec = swingTime > 1e12 ? Math.floor(swingTime / 1000) : swingTime;
    } else {
        // Fallback: go back ~30 candles (timeframe-aware would be better but we don't have it here)
        startTimeSec = breakTimeSec - (30 * 300);
    }

    // Line color based on type
    const lineColor = type === 'choch'
        ? '#ffeb3b'  // Yellow for CHoCH
        : (isBullish ? '#26a69a' : '#ef5350');  // Green/Red for BOS

    // Calculate middle point of the line for label
    const middleTimeSec = Math.floor((startTimeSec + breakTimeSec) / 2);

    // Create line series for the structure line
    const structureLine = chart.addLineSeries({
        color: lineColor,
        lineWidth: 1,
        lineStyle: 2, // Dashed
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });

    // Set line data - horizontal line from swing to break point
    structureLine.setData([
        { time: startTimeSec, value: price },
        { time: breakTimeSec, value: price }
    ]);

    lineSeries.push(structureLine);

    // Create a separate invisible line series just for the label marker
    // This allows us to position the label exactly on the structure line
    const labelSeries = chart.addLineSeries({
        color: 'transparent',
        lineWidth: 0,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });

    // Set single point at middle of the line
    labelSeries.setData([
        { time: middleTimeSec, value: price }
    ]);

    // Add marker on this series - it will appear at the exact price level
    labelSeries.setMarkers([{
        time: middleTimeSec,
        position: 'aboveBar',
        color: lineColor,
        shape: 'text',
        text: type === 'choch' ? 'CHoCH' : 'BOS',
        size: -1
    }]);

    lineSeries.push(labelSeries);

    // Don't return marker - we've already placed it on the label series
    return null;
}

/**
 * Draw FVG zone - TradingView ICT style
 * - Solid top/bottom border lines
 * - Light fill between
 * - Filled FVGs end at fill point
 */
function drawFVGZone(fvg) {
    if (!fvg || !fvg.top || !fvg.bottom || !fvg.created_time) return;
    if (!candleData || candleData.length === 0) return;

    const isBullish = fvg.type === 'bullish';
    const topPrice = fvg.top;
    const bottomPrice = fvg.bottom;

    // Convert milliseconds to seconds (UTC)
    let startTimeSec = fvg.created_time > 1e12 ? Math.floor(fvg.created_time / 1000) : fvg.created_time;

    // End time: if filled_time exists, go up to there, otherwise go up to the last frame.
    let endTimeSec = null;
    if (fvg.filled && fvg.filled_time) {
        endTimeSec = fvg.filled_time > 1e12 ? Math.floor(fvg.filled_time / 1000) : fvg.filled_time;
    }

    // Find candles within FVG time range
    let relevantCandles;
    if (endTimeSec) {
        relevantCandles = candleData.filter(c => c.time >= startTimeSec && c.time <= endTimeSec);
    } else {
        relevantCandles = candleData.filter(c => c.time >= startTimeSec);
    }
    if (relevantCandles.length === 0) return;

    // Colors - TradingView ICT style
    const lineColor = isBullish ? 'rgba(0, 188, 212, 0.9)' : 'rgba(239, 83, 80, 0.9)';
    const fillColor = isBullish ? 'rgba(0, 188, 212, 0.12)' : 'rgba(239, 83, 80, 0.12)';

    // Check if we have fill_history for stepped visualization
    const fillHistory = fvg.fill_history || [];

    if (fillHistory.length > 0) {
        // STEPPED VISUALIZATION: FVG progressively narrows as it gets filled
        // Build segments based on fill history

        // Sort fill_history by time and get unique level changes
        const sortedHistory = [...fillHistory]
            .map(([t, level]) => ({
                time: t > 1e12 ? Math.floor(t / 1000) : t,
                level: level
            }))
            .sort((a, b) => a.time - b.time);

        // For bullish FVG: price fills from top down, so "remaining" is from fill level to bottom
        // For bearish FVG: price fills from bottom up, so "remaining" is from fill level to top

        // Build segments: each segment has a start time, end time, and remaining zone
        let segments = [];
        // Start with full FVG (no fill yet)
        // For bullish: fillLevel starts at top (nothing filled from top yet)
        // For bearish: fillLevel starts at bottom (nothing filled from bottom yet)
        let currentFillLevel = isBullish ? topPrice : bottomPrice;
        let segmentStart = startTimeSec;

        // Track the deepest fill level seen so far (same initial value)
        let deepestFill = isBullish ? topPrice : bottomPrice;

        for (let i = 0; i < sortedHistory.length; i++) {
            const entry = sortedHistory[i];

            // Check if this is a deeper fill
            let isDeeper = false;
            if (isBullish) {
                // Bullish: lower level = deeper fill
                if (entry.level < deepestFill) {
                    isDeeper = true;
                    deepestFill = entry.level;
                }
            } else {
                // Bearish: higher level = deeper fill
                if (entry.level > deepestFill) {
                    isDeeper = true;
                    deepestFill = entry.level;
                }
            }

            if (isDeeper && entry.time > segmentStart) {
                // Close previous segment
                segments.push({
                    startTime: segmentStart,
                    endTime: entry.time,
                    fillLevel: currentFillLevel
                });
                segmentStart = entry.time;
                currentFillLevel = deepestFill;
            }
        }

        // Add final segment (from last fill to end)
        const finalEndTime = endTimeSec || relevantCandles[relevantCandles.length - 1].time;
        if (segmentStart < finalEndTime) {
            segments.push({
                startTime: segmentStart,
                endTime: finalEndTime,
                fillLevel: currentFillLevel
            });
        }

        // Draw each segment
        for (const seg of segments) {
            const segCandles = relevantCandles.filter(c => c.time >= seg.startTime && c.time <= seg.endTime);
            if (segCandles.length === 0) continue;

            // Determine zone boundaries for this segment
            // FVG fill logic:
            // - Bullish FVG: price drops INTO the gap from above, so fill starts from TOP
            //   remaining unfilled zone = from current fill level down to bottom
            // - Bearish FVG: price rises INTO the gap from below, so fill starts from BOTTOM
            //   remaining unfilled zone = from top down to current fill level
            let segTop, segBottom;
            if (isBullish) {
                // Bullish: remaining zone is from fill level (where price dropped to) down to bottom
                segTop = seg.fillLevel;
                segBottom = bottomPrice;
            } else {
                // Bearish: remaining zone is from top down to fill level (where price rose to)
                segTop = topPrice;
                segBottom = seg.fillLevel;
            }

            // Skip if zone is too small
            if (segTop <= segBottom) continue;

            // Draw segment fill
            const segFill = chart.addBaselineSeries({
                baseValue: { type: 'price', price: segBottom },
                topLineColor: 'transparent',
                topFillColor1: fillColor,
                topFillColor2: fillColor,
                bottomLineColor: 'transparent',
                bottomFillColor1: 'transparent',
                bottomFillColor2: 'transparent',
                lineWidth: 0,
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });
            segFill.setData(segCandles.map(c => ({ time: c.time, value: segTop })));
            fvgSeries.push(segFill);
        }

        // Draw single line at the "target" edge
        // Bullish: bottom line (price needs to drop to fill)
        // Bearish: top line (price needs to rise to fill)
        const edgeLine = chart.addLineSeries({
            color: lineColor,
            lineWidth: 1,
            lineStyle: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
        });
        const edgePrice = isBullish ? bottomPrice : topPrice;
        edgeLine.setData(relevantCandles.map(c => ({ time: c.time, value: edgePrice })));
        fvgSeries.push(edgeLine);

    } else {
        // SIMPLE MODE: No fill history, draw full zone with single edge line

        // 1. Single edge line (Bullish: bottom, Bearish: top)
        const edgeLine = chart.addLineSeries({
            color: lineColor,
            lineWidth: 1,
            lineStyle: 0, // Solid
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
        });
        const edgePrice = isBullish ? bottomPrice : topPrice;
        edgeLine.setData(relevantCandles.map(c => ({ time: c.time, value: edgePrice })));
        fvgSeries.push(edgeLine);

        // 2. Fill between lines (light)
        const fvgBox = chart.addBaselineSeries({
            baseValue: { type: 'price', price: bottomPrice },
            topLineColor: 'transparent',
            topFillColor1: fillColor,
            topFillColor2: fillColor,
            bottomLineColor: 'transparent',
            bottomFillColor1: 'transparent',
            bottomFillColor2: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
        });

        fvgBox.setData(relevantCandles.map(c => ({ time: c.time, value: topPrice })));
        fvgSeries.push(fvgBox);
    }
}

/**
 * Draw Gap zone on chart (True GAP - 2 mum arası wick boşluğu)
 * - Unfilled: Solid fill + dashed borders, extends to last candle
 * - Filled: Very faint fill, extends only to fill point
 */
function drawGapZone(gap) {
    if (!gap || !gap.top || !gap.bottom || !gap.created_time) return;
    if (!candleData || candleData.length === 0) return;

    const isBullish = gap.type === 'bullish';
    const isFilled = gap.filled === true;
    const topPrice = gap.top;
    const bottomPrice = gap.bottom;

    // Convert milliseconds to seconds (UTC)
    let startTimeSec = gap.created_time > 1e12 ? Math.floor(gap.created_time / 1000) : gap.created_time;

    // End time: if filled_time exists, go up to there, otherwise go up to the last frame.
    let endTimeSec = null;
    if (isFilled && gap.filled_time) {
        endTimeSec = gap.filled_time > 1e12 ? Math.floor(gap.filled_time / 1000) : gap.filled_time;
    }

    // Find candles within Gap time range
    let relevantCandles;
    if (endTimeSec) {
        relevantCandles = candleData.filter(c => c.time >= startTimeSec && c.time <= endTimeSec);
    } else {
        relevantCandles = candleData.filter(c => c.time >= startTimeSec);
    }

    if (relevantCandles.length === 0) return;

    // Colors - Mavi renk (blue)
    // Bullish: light blue, Bearish: dark blue
    const borderColor = isBullish ? 'rgba(33, 150, 243, 0.8)' : 'rgba(63, 81, 181, 0.8)';
    const baseColor = isBullish ? 'rgba(33, 150, 243, ' : 'rgba(63, 81, 181, ';

    // Opacity based on fill status
    let fillOpacity1, fillOpacity2;
    if (isFilled) {
        fillOpacity1 = '0.1)';
        fillOpacity2 = '0.05)';
    } else {
        fillOpacity1 = '0.25)';
        fillOpacity2 = '0.15)';
    }

    // 1. Top border line (dashed)
    const topLine = chart.addLineSeries({
        color: borderColor,
        lineWidth: 1,
        lineStyle: 2, // Dashed
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });
    topLine.setData(relevantCandles.map(c => ({ time: c.time, value: topPrice })));
    gapSeries.push(topLine);

    // 2. Bottom border line (dashed)
    const bottomLine = chart.addLineSeries({
        color: borderColor,
        lineWidth: 1,
        lineStyle: 2, // Dashed
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });
    bottomLine.setData(relevantCandles.map(c => ({ time: c.time, value: bottomPrice })));
    gapSeries.push(bottomLine);

    // 3. Fill between lines
    const gapBox = chart.addBaselineSeries({
        baseValue: { type: 'price', price: bottomPrice },
        topLineColor: 'transparent',
        topFillColor1: baseColor + fillOpacity1,
        topFillColor2: baseColor + fillOpacity2,
        bottomLineColor: 'transparent',
        bottomFillColor1: 'transparent',
        bottomFillColor2: 'transparent',
        lineWidth: 0,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });

    const boxData = relevantCandles.map(c => ({
        time: c.time,
        value: topPrice
    }));

    gapBox.setData(boxData);
    gapSeries.push(gapBox);
}

/**
 * Draw Order Block zone on chart
 * OB extends from creation (index) to structure break (move_index) or until broken
 */
function drawOBZone(ob) {
    if (!ob || !ob.top || !ob.bottom) return;
    if (!candleData || candleData.length === 0) return;

    const isBullish = ob.type === 'bullish';
    const topPrice = ob.top;
    const bottomPrice = ob.bottom;

    // Find the candleData index from the OB timestamp.
    let startIndex = 0;
    if (ob.timestamp) {
        const obTimeSec = Math.floor(ob.timestamp / 1000);
        const firstCandleTime = candleData[0]?.time || 0;
        const lastCandleTime = candleData[candleData.length - 1]?.time || 0;

        // If the OB chart range is outside the specified limits, do not draw it.
        if (obTimeSec < firstCandleTime) {
            console.log('OB before chart range:', ob.type, 'obTime:', obTimeSec, 'firstCandle:', firstCandleTime);
            startIndex = 0;
        } else if (obTimeSec > lastCandleTime) {
            console.log('OB after chart range - skipping:', ob.type);
            return;
        } else {
            startIndex = candleData.findIndex(c => c.time >= obTimeSec);
            if (startIndex === -1) startIndex = 0;
        }
    } else if (ob.index !== undefined) {
        startIndex = Math.max(0, Math.min(ob.index, candleData.length - 1));
    }

    // Determine the OB end point.
    let endIndex = candleData.length - 1;
    if (ob.status === 'broken') {
        // Broken OB: show until the point where it breaks the price zone.
        for (let i = startIndex + 1; i < candleData.length; i++) {
            if (isBullish && candleData[i].close < bottomPrice) {
                endIndex = i;
                break;
            } else if (!isBullish && candleData[i].close > topPrice) {
                endIndex = i;
                break;
            }
        }
    }

    // OB zone candles get
    const obCandles = candleData.slice(startIndex, endIndex + 1);
    if (obCandles.length === 0) return;

    console.log('Drawing OB:', ob.type, 'startIndex:', startIndex, 'endIndex:', endIndex, 'status:', ob.status, 'top:', topPrice, 'bottom:', bottomPrice);

    // Colors based on OB type - more distinct colors
    const fillColor = isBullish ? 'rgba(38, 166, 154, ' : 'rgba(239, 83, 80, ';
    const opacity = ob.status === 'active' ? '0.35)' : '0.15)';

    // Draw the OB zone with BaselineSeries.
    const obZone = chart.addBaselineSeries({
        baseValue: { type: 'price', price: bottomPrice },
        topLineColor: fillColor + '0)',  // Top line transparent
        topFillColor1: fillColor + opacity,
        topFillColor2: fillColor + (ob.status === 'active' ? '0.2)' : '0.1)'),
        bottomLineColor: 'transparent',
        bottomFillColor1: 'transparent',
        bottomFillColor2: 'transparent',
        lineWidth: 0,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });

    const obData = obCandles.map(c => ({
        time: c.time,
        value: topPrice
    }));

    obZone.setData(obData);
    obSeries.push(obZone);

    // OB label marker - write "OB" at the starting point.
    const borderColor = isBullish ? '#26a69a' : '#ef5350';
    const startCandle = obCandles[0];

    const obMarker = {
        time: startCandle.time,
        position: isBullish ? 'belowBar' : 'aboveBar',
        color: borderColor,
        shape: isBullish ? 'arrowUp' : 'arrowDown',
        text: 'OB',
        size: 1
    };
    markers.push(obMarker);
}

/**
 * Draw Liquidity level line on chart
 */
function drawLiquidityLine(liq) {
    if (!liq || !liq.level) return;

    const isBuySide = liq.type && liq.type.includes('buy_side');
    const color = '#ff9800';  // Orange for liquidity

    const priceLine = candleSeries.createPriceLine({
        price: liq.level,
        color: color,
        lineWidth: 1,
        lineStyle: 1,  // Dashed
        axisLabelVisible: true,
        title: isBuySide ? 'BSL' : 'SSL'
    });
    pricelines.push(priceLine);
}

/**
 * Draw QML (Quasimodo) pattern on chart - TradingView EmreKb Style
 *
 * Bullish QML (5 nokta zigzag): h2 → l1 → h1 → l0 → h0
 *   Koşullar: h2 > h1 (LH), l1 > l0 (LL), h0 > h1, close > l1 (MSB)
 *   Çizgiler: h2-l1, l1-h1, h1-l0, l0-h0, l1 horizontal line
 *
 * Bearish QML (5 nokta zigzag): l2 → h1 → l1 → h0 → l0
 *   Koşullar: l2 < l1 (HL), h1 < h0 (HH), l0 < l1, close < h1 (MSB)
 *   Çizgiler: l2-h1, h1-l1, l1-h0, h0-l0, h1 horizontal line
 */
function drawQMLPattern(qml) {
    if (!qml || !candleData || candleData.length === 0) return;

    const isBullish = qml.type === 'bullish';
    const lineColor = isBullish ? '#22c55e' : '#ef4444';  // Green for bullish, Red for bearish

    // TradingView style: 5 nokta zigzag
    // Bullish: h2, l1, h1, l0, h0
    // Bearish: l2, h1, l1, h0, l0
    let points = [];
    let entryLevel, entryLevelTime;

    if (isBullish) {
        // Bullish QML points: h2 → l1 → h1 → l0 → h0
        if (!qml.h2_time || !qml.l1_time || !qml.h1_time || !qml.l0_time || !qml.h0_time) {
            console.warn('Bullish QML missing timestamps:', qml);
            return;
        }
        points = [
            { time: Math.floor(qml.h2_time / 1000), value: qml.h2 },
            { time: Math.floor(qml.l1_time / 1000), value: qml.l1 },
            { time: Math.floor(qml.h1_time / 1000), value: qml.h1 },
            { time: Math.floor(qml.l0_time / 1000), value: qml.l0 },
            { time: Math.floor(qml.h0_time / 1000), value: qml.h0 },
        ];
        entryLevel = qml.l1;  // Entry at l1 level
        entryLevelTime = Math.floor(qml.l1_time / 1000);
    } else {
        // Bearish QML points: l2 → h1 → l1 → h0 → l0
        if (!qml.l2_time || !qml.h1_time || !qml.l1_time || !qml.h0_time || !qml.l0_time) {
            console.warn('Bearish QML missing timestamps:', qml);
            return;
        }
        points = [
            { time: Math.floor(qml.l2_time / 1000), value: qml.l2 },
            { time: Math.floor(qml.h1_time / 1000), value: qml.h1 },
            { time: Math.floor(qml.l1_time / 1000), value: qml.l1 },
            { time: Math.floor(qml.h0_time / 1000), value: qml.h0 },
            { time: Math.floor(qml.l0_time / 1000), value: qml.l0 },
        ];
        entryLevel = qml.h1;  // Entry at h1 level
        entryLevelTime = Math.floor(qml.h1_time / 1000);
    }

    // Sort points by time
    points.sort((a, b) => a.time - b.time);

    // Check if points are in chart range
    const firstCandleTime = candleData[0]?.time || 0;
    const lastCandleTime = candleData[candleData.length - 1]?.time || 0;

    if (points[0].time < firstCandleTime || points[points.length - 1].time > lastCandleTime) {
        console.log('QML pattern outside chart range');
        return;
    }

    // Draw zigzag line connecting all 5 points
    const qmlLine = chart.addLineSeries({
        color: lineColor,
        lineWidth: 2,
        lineStyle: 0,  // Solid
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });
    qmlLine.setData(points);
    lineSeries.push(qmlLine);

    // Draw horizontal entry level line from entry point to current bar
    const msbTime = Math.floor(qml.timestamp / 1000);
    const entryLine = chart.addLineSeries({
        color: lineColor,
        lineWidth: 2,
        lineStyle: 0,  // Solid
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });
    entryLine.setData([
        { time: entryLevelTime, value: entryLevel },
        { time: msbTime, value: entryLevel }
    ]);
    lineSeries.push(entryLine);

    // Add QM! label at entry level
    const labelSeries = chart.addLineSeries({
        color: 'transparent',
        lineWidth: 0,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });
    labelSeries.setData([{ time: msbTime, value: entryLevel }]);
    labelSeries.setMarkers([{
        time: msbTime,
        position: isBullish ? 'belowBar' : 'aboveBar',
        color: lineColor,
        shape: isBullish ? 'arrowUp' : 'arrowDown',
        text: 'QM!',
        size: 1
    }]);
    lineSeries.push(labelSeries);

    // Draw QML Zone if available
    if (qml.zone_top && qml.zone_bottom) {
        drawQMLZone(qml, entryLevelTime, msbTime, lineColor, isBullish);
    }

    console.log('Drew QML pattern:', qml.type);
    console.log('  Points:', points);
    console.log('  Entry level:', entryLevel, '@', entryLevelTime);
}

/**
 * Draw QML Zone (entry zone box)
 * Zone starts from Head and extends until price enters the zone (mitigated) or max 50 bars
 */
function drawQMLZone(qml, startTime, endTime, lineColor, isBullish) {
    if (!candleData || candleData.length === 0) return;

    const zoneTop = qml.zone_top;
    const zoneBottom = qml.zone_bottom;

    // Find candles starting from startTime (Head time)
    const startIdx = candleData.findIndex(c => c.time >= startTime);
    if (startIdx === -1) return;

    // Find where zone is mitigated (price enters zone) or limit to 50 bars
    let endIdx = Math.min(startIdx + 50, candleData.length - 1);

    for (let i = startIdx; i < candleData.length; i++) {
        const candle = candleData[i];
        // Zone is mitigated when price enters the zone
        if (isBullish) {
            // For bullish QML, zone is below current price - mitigated when price drops into zone
            if (candle.low <= zoneTop && candle.low >= zoneBottom) {
                endIdx = i + 1;  // Include mitigation candle
                break;
            }
        } else {
            // For bearish QML, zone is above current price - mitigated when price rises into zone
            if (candle.high >= zoneBottom && candle.high <= zoneTop) {
                endIdx = i + 1;
                break;
            }
        }

        // Max 50 bars
        if (i - startIdx >= 50) {
            endIdx = i;
            break;
        }
    }

    // Get candles for zone
    const zoneCandles = candleData.slice(startIdx, endIdx + 1);
    if (zoneCandles.length === 0) return;

    // Create BaselineSeries for QML zone (entry area)
    const fillColor = isBullish ? 'rgba(33, 150, 243, ' : 'rgba(244, 67, 54, ';

    const qmlZone = chart.addBaselineSeries({
        baseValue: { type: 'price', price: zoneBottom },
        topLineColor: fillColor + '0)',
        topFillColor1: fillColor + '0.2)',
        topFillColor2: fillColor + '0.1)',
        bottomLineColor: 'transparent',
        bottomFillColor1: 'transparent',
        bottomFillColor2: 'transparent',
        lineWidth: 0,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });

    const zoneData = zoneCandles.map(c => ({
        time: c.time,
        value: zoneTop
    }));

    qmlZone.setData(zoneData);
    fvgSeries.push(qmlZone);  // Use fvgSeries for cleanup
}

/**
 * Draw FTR/FTB Zone on chart
 * FTR = Failed to Return (fresh zone, not yet tested)
 * FTB = First Time Back (zone tested once - optimal entry)
 */
function drawFTRZone(zone) {
    if (!zone || !candleData || candleData.length === 0) return;

    const isBullish = zone.type === 'bullish';

    // Zone start time (milliseconds to seconds, UTC)
    const startTime = Math.floor(zone.created_time / 1000);

    // Find start index in candle data
    const startIdx = candleData.findIndex(c => c.time >= startTime);
    if (startIdx === -1) return;

    // Determine zone end based on status
    let endIdx;
    if (zone.invalidated && zone.invalidated_time) {
        // Zone was invalidated - end at invalidation point
        const invalidTime = Math.floor(zone.invalidated_time / 1000);
        endIdx = candleData.findIndex(c => c.time >= invalidTime);
        if (endIdx === -1) endIdx = candleData.length - 1;
    } else if (zone.ftb_time) {
        // FTB occurred - the zone ends on the initial touch.
        const ftbTime = Math.floor(zone.ftb_time / 1000);
        endIdx = candleData.findIndex(c => c.time >= ftbTime);
        if (endIdx === -1) endIdx = candleData.length - 1;
    } else {
        // Fresh zone - extend to end of chart (not yet tested)
        endIdx = candleData.length - 1;
    }

    // Get candles for zone
    const zoneCandles = candleData.slice(startIdx, endIdx + 1);
    if (zoneCandles.length === 0) return;

    // FTR Zone color - Orange (koyu)
    const fillColor = 'rgba(255, 152, 0, ';  // Orange

    // Opacity based on status
    const opacity = zone.status === 'fresh' ? '0.4)' : (zone.status === 'ftb' ? '0.3)' : '0.2)');

    // Create BaselineSeries for FTR zone
    const ftrZone = chart.addBaselineSeries({
        baseValue: { type: 'price', price: zone.bottom },
        topLineColor: fillColor + '0)',
        topFillColor1: fillColor + opacity,
        topFillColor2: fillColor + (zone.status === 'fresh' ? '0.25)' : '0.15)'),
        bottomLineColor: 'transparent',
        bottomFillColor1: 'transparent',
        bottomFillColor2: 'transparent',
        lineWidth: 0,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
    });

    const zoneData = zoneCandles.map(c => ({
        time: c.time,
        value: zone.top
    }));

    ftrZone.setData(zoneData);
    fvgSeries.push(ftrZone);  // Use fvgSeries for cleanup

    // Add FTR label marker (only the text "FTR", no circle)
    if (zone.ftr_candle_time) {
        const ftrCandleTime = Math.floor(zone.ftr_candle_time / 1000);
        const ftrCandleIdx = candleData.findIndex(c => c.time === ftrCandleTime);

        if (ftrCandleIdx !== -1) {
            const ftrCandle = candleData[ftrCandleIdx];
            const ftrMarkerSeries = chart.addLineSeries({
                color: 'transparent',
                lineWidth: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            });

            // Place marker at the FTR candle
            ftrMarkerSeries.setData([
                { time: ftrCandle.time, value: isBullish ? ftrCandle.low : ftrCandle.high }
            ]);

            // Only the text "FTR", no shape.
            ftrMarkerSeries.setMarkers([
                {
                    time: ftrCandle.time,
                    position: isBullish ? 'belowBar' : 'aboveBar',
                    color: '#ff9800',  // Orange
                    shape: 'text',
                    text: 'FTR',
                    size: 0
                }
            ]);

            lineSeries.push(ftrMarkerSeries);
        }
    }

    console.log('Drew FTR zone:', zone.type, zone.status, '@', zone.top, '-', zone.bottom);
}

/**
 * Update chart visibility based on checkboxes
 */
function updateChartVisibility() {
    visibilitySettings.bos = document.getElementById('show-bos').checked;
    visibilitySettings.choch = document.getElementById('show-choch').checked;
    visibilitySettings.swing = document.getElementById('show-swing').checked;
    visibilitySettings.fvg = document.getElementById('show-fvg').checked;
    visibilitySettings.levels = document.getElementById('show-levels').checked;

    // Gap checkbox
    const gapCheckbox = document.getElementById('show-gap');
    if (gapCheckbox) visibilitySettings.gap = gapCheckbox.checked;

    // New Price Action checkboxes
    const obCheckbox = document.getElementById('show-ob');
    const liqCheckbox = document.getElementById('show-liquidity');
    const qmlCheckbox = document.getElementById('show-qml');
    const ftrCheckbox = document.getElementById('show-ftr');

    if (obCheckbox) visibilitySettings.ob = obCheckbox.checked;
    if (liqCheckbox) visibilitySettings.liquidity = liqCheckbox.checked;
    if (qmlCheckbox) visibilitySettings.qml = qmlCheckbox.checked;
    if (ftrCheckbox) visibilitySettings.ftr = ftrCheckbox.checked;

    // Redraw formations
    clearAllDrawings();
    if (candleData.length > 0) {
        candleSeries.setData(candleData);
        drawFormations();
    }
}

/**
 * Add price lines to chart (legacy - kept for compatibility)
 */
function addPriceLines(lines) {
    if (!lines || lines.length === 0) return;

    lines.forEach(line => {
        const priceLine = candleSeries.createPriceLine({
            price: line.price,
            color: line.color,
            lineWidth: line.lineWidth || 1,
            lineStyle: line.lineStyle || 2,
            axisLabelVisible: true,
            title: line.title || ''
        });
        pricelines.push(priceLine);
    });
}

/**
 * Update summary section
 */
function updateSummary() {
    if (!analysisData) return;

    const summary = analysisData.summary || {};
    const levels = analysisData.levels || {};
    const lastBar = analysisData.last_bar || {};

    // Trend
    const trendEl = document.getElementById('summary-trend');
    const trend = summary.current_trend || lastBar.trend || 'unknown';
    trendEl.textContent = trend;
    trendEl.className = 'summary-value ' + (trend === 'uptrend' ? 'bullish' : trend === 'downtrend' ? 'bearish' : 'neutral');

    // Bias
    const biasEl = document.getElementById('summary-bias');
    const bias = lastBar.market_bias || 'neutral';
    biasEl.textContent = bias;
    biasEl.className = 'summary-value ' + (bias === 'bullish' ? 'bullish' : bias === 'bearish' ? 'bearish' : 'neutral');

    // Structure
    const structureEl = document.getElementById('summary-structure');
    const structure = lastBar.structure || 'ranging';
    structureEl.textContent = structure;

    // Swing levels
    document.getElementById('summary-swing-high').textContent = levels.swing_high ? levels.swing_high.toFixed(2) : '-';
    document.getElementById('summary-swing-low').textContent = levels.swing_low ? levels.swing_low.toFixed(2) : '-';
}

/**
 * Update stats section
 */
function updateStats() {
    if (!analysisData) return;

    const summary = analysisData.summary || {};
    const formations = analysisData.formations || {};

    document.getElementById('stat-bos').textContent = summary.bos_count || 0;
    document.getElementById('stat-choch').textContent = summary.choch_count || 0;
    document.getElementById('stat-fvg').textContent = summary.fvg_count || 0;
    document.getElementById('stat-active-fvg').textContent = summary.active_fvg_count || 0;
    document.getElementById('stat-swing').textContent = summary.swing_count || 0;

    // Extended stats (new detectors)
    const obEl = document.getElementById('stat-ob');
    const liqEl = document.getElementById('stat-liquidity');
    const qmlEl = document.getElementById('stat-qml');
    const ftrEl = document.getElementById('stat-ftr');
    const gapEl = document.getElementById('stat-gap');

    if (obEl) obEl.textContent = summary.ob_count || 0;
    if (liqEl) liqEl.textContent = summary.liquidity_count || 0;
    if (qmlEl) qmlEl.textContent = summary.qml_count || 0;
    if (ftrEl) ftrEl.textContent = summary.ftr_count || (formations.ftr_zones?.length || 0);
    if (gapEl) gapEl.textContent = formations.gap?.length || 0;
}

/**
 * Update active FVGs list
 */
function updateActiveFVGs() {
    const container = document.getElementById('active-fvg-list');

    if (!analysisData || !analysisData.active || !analysisData.active.fvg || analysisData.active.fvg.length === 0) {
        container.innerHTML = '<p class="empty-message">No active FVG found</p>';
        return;
    }

    let html = '';
    analysisData.active.fvg.forEach(fvg => {
        const typeClass = fvg.type === 'bullish' ? 'bullish' : 'bearish';
        html += `
            <div class="fvg-item ${typeClass}">
                <span class="fvg-type">${fvg.type.toUpperCase()}</span>
                <span class="fvg-range">${fvg.bottom.toFixed(2)} - ${fvg.top.toFixed(2)}</span>
                <span class="fvg-fill">${fvg.filled_percent.toFixed(0)}% dolu</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Update formations list
 */
function updateFormations() {
    const container = document.getElementById('formations-list');

    if (!analysisData || !analysisData.formations) {
        container.innerHTML = '<p class="empty-message">No occurrences found</p>';
        return;
    }

    const formations = analysisData.formations;
    let allFormations = [];

    // Collect all formations with type info
    if (formations.bos) {
        formations.bos.forEach(f => allFormations.push({ ...f, formationType: 'bos' }));
    }
    if (formations.choch) {
        formations.choch.forEach(f => allFormations.push({ ...f, formationType: 'choch' }));
    }
    if (formations.fvg) {
        formations.fvg.forEach(f => allFormations.push({ ...f, formationType: 'fvg' }));
    }
    if (formations.swing) {
        formations.swing.forEach(f => allFormations.push({ ...f, formationType: 'swing' }));
    }
    if (formations.ob) {
        formations.ob.forEach(f => allFormations.push({ ...f, formationType: 'ob' }));
    }
    if (formations.liquidity) {
        formations.liquidity.forEach(f => allFormations.push({ ...f, formationType: 'liquidity' }));
    }
    if (formations.qml) {
        formations.qml.forEach(f => allFormations.push({ ...f, formationType: 'qml' }));
    }

    // Sort by time/index descending (most recent first)
    allFormations.sort((a, b) => {
        const timeA = a.break_time || a.created_time || a.time || a.timestamp || a.index || 0;
        const timeB = b.break_time || b.created_time || b.time || b.timestamp || b.index || 0;
        return timeB - timeA;
    });

    // Store for filtering
    window.allFormations = allFormations;

    renderFormations(allFormations);
}

/**
 * Render formations list
 */
function renderFormations(formations) {
    const container = document.getElementById('formations-list');

    if (formations.length === 0) {
        container.innerHTML = '<p class="empty-message">No occurrences found</p>';
        return;
    }

    let html = '';
    formations.slice(0, 50).forEach(f => {
        const time = f.break_time || f.created_time || f.time || f.timestamp;
        const timeStr = time ? new Date(time).toLocaleString() : '-';
        const price = f.break_price || f.broken_level || f.price || f.top || f.level || f.break_level;
        // Handle different type formats
        let dirClass = 'neutral';
        if (f.type === 'bullish' || f.type === 'high' || (f.type && f.type.includes('bullish'))) {
            dirClass = 'bullish';
        } else if (f.type === 'bearish' || f.type === 'low' || (f.type && f.type.includes('bearish'))) {
            dirClass = 'bearish';
        } else if (f.type && (f.type.includes('buy_side') || f.type.includes('sell_side'))) {
            // Liquidity zones
            dirClass = f.type.includes('buy_side') ? 'bullish' : 'bearish';
        }

        html += `
            <div class="formation-item" data-type="${f.formationType}">
                <span class="formation-type ${f.formationType} ${dirClass}">${f.formationType.toUpperCase()}</span>
                <div class="formation-info">
                    <span class="formation-price">${price ? price.toFixed(2) : '-'}</span>
                    <span class="formation-time">${timeStr}</span>
                </div>
                <span class="badge badge-${dirClass}">${f.type}</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Filter formations by type
 */
function filterFormations(type) {
    if (!window.allFormations) return;

    let filtered;
    if (type === 'all') {
        filtered = window.allFormations;
    } else {
        filtered = window.allFormations.filter(f => f.formationType === type);
    }

    renderFormations(filtered);
}

/**
 * Update last bar analysis
 */
function updateLastBar() {
    const container = document.getElementById('last-bar-info');

    if (!analysisData || !analysisData.last_bar) {
        container.innerHTML = '<p class="empty-message">Data not found</p>';
        return;
    }

    const bar = analysisData.last_bar;
    const time = bar.timestamp ? new Date(bar.timestamp).toLocaleString() : '-';

    let html = `
        <div class="last-bar-item">
            <h4>Zaman</h4>
            <span class="value">${time}</span>
        </div>
        <div class="last-bar-item">
            <h4>Bar Index</h4>
            <span class="value">${bar.bar_index || '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Trend</h4>
            <span class="value">${bar.trend || '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Market Bias</h4>
            <span class="value">${bar.market_bias || '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Structure</h4>
            <span class="value">${bar.structure || '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Swing High</h4>
            <span class="value">${bar.swing_high ? bar.swing_high.toFixed(2) : '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Swing Low</h4>
            <span class="value">${bar.swing_low ? bar.swing_low.toFixed(2) : '-'}</span>
        </div>
        <div class="last-bar-item">
            <h4>Active FVG</h4>
            <span class="value">${bar.active_fvgs ? bar.active_fvgs.length : 0}</span>
        </div>
    `;

    // Add new formations if any
    if (bar.new_bos) {
        html += `
            <div class="last-bar-item">
                <h4>New BOS</h4>
                <span class="value badge badge-${bar.new_bos.type}">${bar.new_bos.type} @ ${bar.new_bos.broken_level?.toFixed(2) || '-'}</span>
            </div>
        `;
    }

    if (bar.new_choch) {
        html += `
            <div class="last-bar-item">
                <h4>New CHoCH</h4>
                <span class="value badge badge-${bar.new_choch.type}">${bar.new_choch.type} @ ${bar.new_choch.broken_level?.toFixed(2) || '-'}</span>
            </div>
        `;
    }

    if (bar.new_fvg) {
        html += `
            <div class="last-bar-item">
                <h4>New FVG</h4>
                <span class="value badge badge-${bar.new_fvg.type}">${bar.new_fvg.type} (${bar.new_fvg.bottom?.toFixed(2)} - ${bar.new_fvg.top?.toFixed(2)})</span>
            </div>
        `;
    }

    if (bar.new_swing) {
        html += `
            <div class="last-bar-item">
                <h4>New Swing</h4>
                <span class="value badge badge-${bar.new_swing.type === 'high' ? 'bullish' : 'bearish'}">${bar.new_swing.type} @ ${bar.new_swing.price?.toFixed(2) || '-'}</span>
            </div>
        `;
    }

    container.innerHTML = html;
}
