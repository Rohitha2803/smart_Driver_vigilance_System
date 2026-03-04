/**
 * Driver Safety Monitor — Frontend Application
 * WebSocket client for real-time video streaming and detection status.
 */

// ─── State ──────────────────────────────────────────────────
let ws = null;
let isMonitoring = false;
let frameCount = 0;
let fpsInterval = null;
let lastFpsTime = Date.now();
let alarmAudioCtx = null;
let alarmOscillator = null;
let isAlarmPlaying = false;

// ─── DOM Elements ───────────────────────────────────────────
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');

const videoPlaceholder = document.getElementById('videoPlaceholder');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');

const connectionBadge = document.getElementById('connectionBadge');
const connectionText = document.getElementById('connectionText');
const systemStatus = document.getElementById('systemStatus');
const systemStatusText = document.getElementById('systemStatusText');

const earValue = document.getElementById('earValue');
const marValue = document.getElementById('marValue');
const blinkValue = document.getElementById('blinkValue');
const earBar = document.getElementById('earBar');
const marBar = document.getElementById('marBar');

const drowsyText = document.getElementById('drowsyText');
const drowsyStatus = document.getElementById('drowsyStatus');
const drowsyCard = document.getElementById('drowsyCard');

const yawnText = document.getElementById('yawnText');
const yawnStatus = document.getElementById('yawnStatus');
const yawnCard = document.getElementById('yawnCard');

const phoneText = document.getElementById('phoneText');
const phoneStatus = document.getElementById('phoneStatus');
const phoneCard = document.getElementById('phoneCard');
const phoneConfidence = document.getElementById('phoneConfidence');

const alertBanner = document.getElementById('alertBanner');
const alertBannerText = document.getElementById('alertBannerText');
const alertOverlay = document.getElementById('alertOverlay');

const logEntries = document.getElementById('logEntries');
const fpsCounter = document.getElementById('fpsCounter');

// ─── WebSocket Connection ───────────────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        updateConnectionStatus(true);
        addLog('Connected to server', 'info');
    };

    ws.onclose = () => {
        updateConnectionStatus(false);
        addLog('Disconnected from server', 'warning');
        if (isMonitoring) {
            // Auto-reconnect after 2 seconds
            setTimeout(connectWebSocket, 2000);
        }
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        addLog('Connection error', 'danger');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'frame') {
                renderFrame(data.frame);
                updateStatus(data.status);
                frameCount++;
            } else if (data.type === 'control') {
                addLog(`Server: ${data.message}`, data.status === 'error' ? 'danger' : 'info');
            }
        } catch (e) {
            console.error('Error processing message:', e);
        }
    };
}

// ─── Connection Status ──────────────────────────────────────
function updateConnectionStatus(connected) {
    connectionBadge.className = `badge ${connected ? 'badge-connected' : 'badge-disconnected'}`;
    connectionText.textContent = connected ? 'Connected' : 'Disconnected';
}

// ─── Start / Stop Monitoring ────────────────────────────────
function startMonitoring() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        // Wait for connection then start
        const waitForOpen = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                clearInterval(waitForOpen);
                sendStartCommand();
            }
        }, 200);
        return;
    }
    sendStartCommand();
}

function sendStartCommand() {
    ws.send(JSON.stringify({ action: 'start', camera_index: 0 }));
    isMonitoring = true;

    btnStart.disabled = true;
    btnStop.disabled = false;
    videoPlaceholder.classList.add('hidden');

    systemStatus.className = 'badge badge-monitoring';
    systemStatusText.textContent = 'Monitoring';

    addLog('Monitoring started', 'info');
    startFpsCounter();
}

function stopMonitoring() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
    }

    isMonitoring = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    videoPlaceholder.classList.remove('hidden');

    systemStatus.className = 'badge badge-idle';
    systemStatusText.textContent = 'Idle';

    stopAlarm();
    hideAlert();
    stopFpsCounter();

    addLog('Monitoring stopped', 'info');
}

// ─── Frame Rendering ────────────────────────────────────────
const tempImg = new Image();
tempImg.onload = function () {
    canvas.width = tempImg.width;
    canvas.height = tempImg.height;
    ctx.drawImage(tempImg, 0, 0);
};

function renderFrame(base64Data) {
    tempImg.src = 'data:image/jpeg;base64,' + base64Data;
}

// ─── Status Updates ─────────────────────────────────────────
let lastDrowsyState = false;
let lastYawnState = false;
let lastPhoneState = false;
let lastHeadTurnState = false;

function updateStatus(status) {
    const drowsy = status.drowsiness;
    const phone = status.phone;

    // ── EAR ──
    earValue.textContent = drowsy.ear_avg.toFixed(3);
    const earPct = Math.min(100, (drowsy.ear_avg / 0.4) * 100);
    earBar.style.width = earPct + '%';
    earBar.className = `metric-bar-fill ${drowsy.ear_avg < 0.22 ? 'danger' : ''}`;

    const earCard = document.getElementById('earCard');
    earCard.className = `metric-card glass-card ${drowsy.ear_avg < 0.22 ? 'alert-active' : ''}`;

    // ── MAR ──
    marValue.textContent = drowsy.mar.toFixed(3);
    const marPct = Math.min(100, (drowsy.mar / 1.0) * 100);
    marBar.style.width = marPct + '%';
    marBar.className = `metric-bar-fill mar-bar ${drowsy.yawning ? 'danger' : ''}`;

    const marCardEl = document.getElementById('marCard');
    marCardEl.className = `metric-card glass-card ${drowsy.yawning ? 'warning-active' : ''}`;

    // ── Blinks ──
    blinkValue.textContent = drowsy.blink_count;

    // ── Drowsy Status ──
    if (drowsy.drowsy) {
        drowsyText.textContent = 'DROWSY!';
        drowsyStatus.querySelector('.status-dot').className = 'status-dot status-danger';
        drowsyCard.className = 'metric-card glass-card status-card alert-active';
        if (!lastDrowsyState) {
            addLog('⚠️ DROWSINESS DETECTED — Driver appears drowsy!', 'danger');
            lastDrowsyState = true;
        }
    } else {
        drowsyText.textContent = 'Normal';
        drowsyStatus.querySelector('.status-dot').className = 'status-dot status-safe';
        drowsyCard.className = 'metric-card glass-card status-card';
        if (lastDrowsyState) {
            addLog('✓ Driver alert again', 'info');
            lastDrowsyState = false;
        }
    }

    // ── Yawn Status ──
    if (drowsy.yawning) {
        yawnText.textContent = 'Yawning!';
        yawnStatus.querySelector('.status-dot').className = 'status-dot status-warning';
        yawnCard.className = 'metric-card glass-card status-card warning-active';
        if (!lastYawnState) {
            addLog('🥱 Yawning detected', 'warning');
            lastYawnState = true;
        }
    } else {
        yawnText.textContent = 'Not Detected';
        yawnStatus.querySelector('.status-dot').className = 'status-dot status-safe';
        yawnCard.className = 'metric-card glass-card status-card';
        lastYawnState = false;
    }

    // ── Phone Status ──
    if (phone.alert) {
        phoneText.textContent = 'PHONE IN USE!';
        phoneStatus.querySelector('.status-dot').className = 'status-dot status-danger';
        phoneCard.className = 'metric-card glass-card status-card alert-active';
        phoneConfidence.textContent = `Confidence: ${(phone.confidence * 100).toFixed(0)}%`;
        if (!lastPhoneState) {
            addLog('📱 MOBILE PHONE DETECTED — Driver using phone!', 'danger');
            lastPhoneState = true;
        }
    } else if (phone.phone_detected) {
        phoneText.textContent = 'Detecting...';
        phoneStatus.querySelector('.status-dot').className = 'status-dot status-warning';
        phoneCard.className = 'metric-card glass-card status-card warning-active';
        phoneConfidence.textContent = `Confidence: ${(phone.confidence * 100).toFixed(0)}%`;
    } else {
        phoneText.textContent = 'Not Detected';
        phoneStatus.querySelector('.status-dot').className = 'status-dot status-safe';
        phoneCard.className = 'metric-card glass-card status-card';
        phoneConfidence.textContent = '';
        if (lastPhoneState) {
            addLog('✓ Phone no longer detected', 'info');
            lastPhoneState = false;
        }
    }

    // ── Head Turn Status ──
    const headTurnText = document.getElementById('headTurnText');
    const headTurnStatus = document.getElementById('headTurnStatus');
    const headTurnCard = document.getElementById('headTurnCard');
    const headTurnInfo = document.getElementById('headTurnInfo');

    if (drowsy.head_turn_alert) {
        headTurnText.textContent = `DISTRACTED (${drowsy.head_direction.toUpperCase()})`;
        headTurnStatus.querySelector('.status-dot').className = 'status-dot status-danger';
        headTurnCard.className = 'metric-card glass-card status-card alert-active';
        headTurnInfo.textContent = 'Looking away from road!';
        if (!lastHeadTurnState) {
            addLog(`⚠️ DISTRACTED — Looking ${drowsy.head_direction} from road!`, 'danger');
            lastHeadTurnState = true;
        }
    } else if (drowsy.looking_sideways) {
        headTurnText.textContent = `Looking ${drowsy.head_direction}`;
        headTurnStatus.querySelector('.status-dot').className = 'status-dot status-warning';
        headTurnCard.className = 'metric-card glass-card status-card warning-active';
        headTurnInfo.textContent = 'Keep eyes on road';
    } else {
        headTurnText.textContent = 'Center';
        headTurnStatus.querySelector('.status-dot').className = 'status-dot status-safe';
        headTurnCard.className = 'metric-card glass-card status-card';
        headTurnInfo.textContent = '';
        if (lastHeadTurnState) {
            addLog('✓ Eyes back on road', 'info');
            lastHeadTurnState = false;
        }
    }

    // ── Global Alert State ──
    const hasAlert = drowsy.drowsy || phone.alert || drowsy.head_turn_alert;
    if (hasAlert) {
        let alertMsg = 'ALERT DETECTED!';
        if (drowsy.drowsy && phone.alert) alertMsg = 'DROWSINESS & PHONE DETECTED!';
        else if (drowsy.drowsy) alertMsg = 'DROWSINESS DETECTED!';
        else if (phone.alert) alertMsg = 'MOBILE PHONE DETECTED!';
        else if (drowsy.head_turn_alert) alertMsg = 'DISTRACTED - LOOK AHEAD!';

        showAlert(alertMsg);
        playAlarm();
        systemStatus.className = 'badge badge-alert';
        systemStatusText.textContent = 'ALERT';
    } else {
        hideAlert();
        stopAlarm();
        if (isMonitoring) {
            systemStatus.className = 'badge badge-monitoring';
            systemStatusText.textContent = 'Monitoring';
        }
    }
}

// ─── Alert Banner ───────────────────────────────────────────
function showAlert(message) {
    alertBanner.classList.remove('hidden');
    alertBannerText.textContent = message;
    alertOverlay.classList.add('active');
}

function hideAlert() {
    alertBanner.classList.add('hidden');
    alertOverlay.classList.remove('active');
}

// ─── Alarm Sound (Web Audio API) ────────────────────────────
function playAlarm() {
    if (isAlarmPlaying) return;

    try {
        alarmAudioCtx = new (window.AudioContext || window.webkitAudioContext)();

        // Create an alarm pattern: two alternating tones
        function createTone(freq, startTime, duration) {
            const osc = alarmAudioCtx.createOscillator();
            const gain = alarmAudioCtx.createGain();

            osc.type = 'sine';
            osc.frequency.value = freq;

            gain.gain.setValueAtTime(0, startTime);
            gain.gain.linearRampToValueAtTime(0.3, startTime + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);

            osc.connect(gain);
            gain.connect(alarmAudioCtx.destination);

            osc.start(startTime);
            osc.stop(startTime + duration);
        }

        // Play alternating beeps
        const now = alarmAudioCtx.currentTime;
        for (let i = 0; i < 6; i++) {
            createTone(i % 2 === 0 ? 800 : 1200, now + i * 0.25, 0.2);
        }

        isAlarmPlaying = true;
        // Reset after alarm pattern completes
        setTimeout(() => {
            isAlarmPlaying = false;
        }, 1800);

    } catch (e) {
        console.error('Audio error:', e);
    }
}

function stopAlarm() {
    isAlarmPlaying = false;
    if (alarmAudioCtx && alarmAudioCtx.state === 'running') {
        alarmAudioCtx.close().catch(() => { });
        alarmAudioCtx = null;
    }
}

// ─── FPS Counter ────────────────────────────────────────────
function startFpsCounter() {
    frameCount = 0;
    lastFpsTime = Date.now();
    fpsInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastFpsTime) / 1000;
        const fps = Math.round(frameCount / elapsed);
        fpsCounter.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFpsTime = now;
    }, 1000);
}

function stopFpsCounter() {
    if (fpsInterval) {
        clearInterval(fpsInterval);
        fpsInterval = null;
    }
    fpsCounter.textContent = '0 FPS';
}

// ─── Event Log ──────────────────────────────────────────────
function addLog(message, level = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;

    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-msg">${message}</span>
    `;

    logEntries.prepend(entry);

    // Keep max 50 entries
    while (logEntries.children.length > 50) {
        logEntries.removeChild(logEntries.lastChild);
    }
}

function clearLog() {
    logEntries.innerHTML = '';
    addLog('Log cleared', 'info');
}

// ─── Initialize ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Set canvas size
    canvas.width = 640;
    canvas.height = 480;

    // Draw placeholder on canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Connect WebSocket
    connectWebSocket();
});
