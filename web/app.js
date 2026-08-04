/**
 * Tianjin Mahjong AI — Main Application
 *
 * Pipeline:
 *   Camera → Periodic snapshot → ONNX inference (YOLOv8n)
 *   → Tile detections → Game state update → Strategy recommendation
 *   → Display update + optional voice output
 */

// === Constants ===
const MODEL_URL = 'https://github.com/Z3E0/tianjin-mahjong/releases/latest/download/model.onnx';
const SNAPSHOT_INTERVAL = 3000; // ms between auto-snapshots
const DETECTION_CONFIDENCE = 0.5;
const DETECTION_IOU = 0.45;
const IMG_SIZE = 640;

// === State ===
const state = {
  running: false,
  modelLoaded: false,
  model: null,
  session: null,

  // Game state
  handTiles: [],      // Currently detected hand tiles (IDs 0-33)
  discards: [],       // Discarded tiles
  melds: [],          // Declared melds (碰/杠)
  hunDora: null,      // 混儿 indicator tile ID (null if not set)
  lastRecommendation: null,

  // Detection state
  lastSnapshot: 0,
  detections: [],
  fps: 0,
  frameCount: 0,
  fpsTimer: 0,

  // Zone system
  zones: {
    hand: { x: 0.05, y: 0.15, w: 0.90, h: 0.35 },    // Top area — normalized 0-1
    discard: { x: 0.05, y: 0.55, w: 0.90, h: 0.35 },  // Bottom area
  },
  editingZones: false,
};

// === DOM Elements ===
const els = {
  camera: document.getElementById('camera'),
  overlay: document.getElementById('overlay'),
  statusText: document.getElementById('status-text'),
  fpsCounter: document.getElementById('fps-counter'),
  handTiles: document.getElementById('hand-tiles'),
  handRaw: document.getElementById('hand-raw'),
  tileCount: document.getElementById('tile-count'),
  discardTiles: document.getElementById('discard-tiles'),
  recContent: document.getElementById('rec-content'),
  debugLog: document.getElementById('debug-log'),
  btnToggle: document.getElementById('btn-toggle'),
  btnSnapshot: document.getElementById('btn-snapshot'),
  btnClearDiscards: document.getElementById('btn-clear-discards'),
};

// === Logging ===
function log(msg) {
  const time = new Date().toLocaleTimeString();
  const line = `[${time}] ${msg}`;
  console.log(line);
  if (els.debugLog) {
    els.debugLog.textContent += line + '\n';
    els.debugLog.scrollTop = els.debugLog.scrollHeight;
  }
}

// === Model Loading ===
async function loadModel() {
  log('Loading ONNX model...');
  els.statusText.textContent = 'Loading ONNX model...';

  try {
    // Create ONNX inference session
    const sessionOptions = {
      executionProviders: ['wasm'], // wasm works everywhere; 'webgl' for GPU
      graphOptimizationLevel: 'all',
    };

    state.session = await ort.InferenceSession.create(MODEL_URL, sessionOptions);
    state.modelLoaded = true;
    log(`Model loaded. Input: ${state.session.inputNames[0]}, Output: ${state.session.outputNames[0]}`);
    els.statusText.textContent = 'Model loaded ✓';
  } catch (err) {
    log(`Model load failed: ${err.message}`);
    els.statusText.textContent = 'Model load failed — check console';
  }
}

// === Camera Setup ===
async function setupCamera() {
  log('Requesting camera...');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Back camera
        width: { ideal: IMG_SIZE },
        height: { ideal: IMG_SIZE },
      },
      audio: false,
    });

    els.camera.srcObject = stream;
    await els.camera.play();

    // Set canvas size to match video
    els.camera.addEventListener('loadedmetadata', () => {
      els.overlay.width = els.camera.videoWidth;
      els.overlay.height = els.camera.videoHeight;
      log(`Camera ready: ${els.camera.videoWidth}x${els.camera.videoHeight}`);
    });

    log('Camera started ✓');
  } catch (err) {
    log(`Camera error: ${err.message}`);
    els.statusText.textContent = 'Camera access denied';
  }
}

// === ONNX Inference ===
async function runInference(imageData) {
  if (!state.session) return [];

  try {
    // Prepare input tensor
    // YOLOv8 expects: [1, 3, 640, 640] float32, normalized 0-1
    const input = preprocessImage(imageData);

    const feeds = {};
    feeds[state.session.inputNames[0]] = input;

    // Run inference
    const startTime = performance.now();
    const results = await state.session.run(feeds);
    const elapsed = performance.now() - startTime;

    // Parse YOLO output
    const output = results[state.session.outputNames[0]];
    const detections = parseYoloOutput(output, imageData.width, imageData.height);

    state.frameCount++;
    state.fps = Math.round(1000 / elapsed);

    return detections;
  } catch (err) {
    log(`Inference error: ${err.message}`);
    return [];
  }
}

/**
 * Preprocess image for YOLOv8 ONNX: resize to 640x640, normalize.
 */
function preprocessImage(imageData) {
  // Create offscreen canvas for resizing
  const canvas = document.createElement('canvas');
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext('2d');

  // Draw image to canvas, letterbox to maintain aspect ratio
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = imageData.width;
  srcCanvas.height = imageData.height;
  const srcCtx = srcCanvas.getContext('2d');
  srcCtx.putImageData(imageData, 0, 0);

  // Calculate scale to fit 640x640
  const scale = Math.min(IMG_SIZE / imageData.width, IMG_SIZE / imageData.height);
  const scaledW = imageData.width * scale;
  const scaledH = imageData.height * scale;
  const offsetX = (IMG_SIZE - scaledW) / 2;
  const offsetY = (IMG_SIZE - scaledH) / 2;

  // Fill with grey (letterbox)
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  ctx.drawImage(srcCanvas, offsetX, offsetY, scaledW, scaledH);

  // Get pixel data
  const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const pixels = imgData.data;

  // Convert to float32 tensor [1, 3, 640, 640], normalize to [0, 1]
  const input = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
  const stride = IMG_SIZE * IMG_SIZE;

  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const px = i * 4;
    input[i] = pixels[px] / 255.0;                    // R
    input[stride + i] = pixels[px + 1] / 255.0;       // G
    input[2 * stride + i] = pixels[px + 2] / 255.0;   // B
  }

  return new ort.Tensor('float32', input, [1, 3, IMG_SIZE, IMG_SIZE]);
}

/**
 * Parse YOLOv8 ONNX output.
 * Output shape: [1, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
 * For our 34-class model: [1, 38, 8400] where 38 = 4 + 34
 */
function parseYoloOutput(output, imgW, imgH) {
  const data = output.cpuData; // Float32Array
  const dims = output.dims;    // [1, num_channels, num_anchors]
  const numChannels = dims[1]; // 4 + num_classes
  const numAnchors = dims[2];
  const numClasses = numChannels - 4;

  const detections = [];
  const scaleX = imgW / IMG_SIZE;
  const scaleY = imgH / IMG_SIZE;

  for (let i = 0; i < numAnchors; i++) {
    // Find max class confidence
    let maxConf = 0;
    let maxClass = -1;
    const classOffset = 4 * numAnchors;

    for (let c = 0; c < numClasses; c++) {
      const conf = data[classOffset + c * numAnchors + i];
      if (conf > maxConf) {
        maxConf = conf;
        maxClass = c;
      }
    }

    if (maxConf < DETECTION_CONFIDENCE || maxClass >= TIANJIN_ENGINE.NUM_TILES) continue;

    // Bounding box (cx, cy, w, h normalized)
    const cx = data[i];
    const cy = data[numAnchors + i];
    const w = data[2 * numAnchors + i];
    const h = data[3 * numAnchors + i];

    // Convert to pixel coordinates
    const x1 = (cx - w / 2) * IMG_SIZE * scaleX;
    const y1 = (cy - h / 2) * IMG_SIZE * scaleY;
    const x2 = (cx + w / 2) * IMG_SIZE * scaleX;
    const y2 = (cy + h / 2) * IMG_SIZE * scaleY;

    detections.push({
      classId: maxClass,
      className: TIANJIN_ENGINE.TILE_NAMES[maxClass],
      confidence: maxConf,
      bbox: { x1, y1, x2, y2 },
    });
  }

  // NMS (Non-Maximum Suppression) - simple version
  return nonMaxSuppression(detections, DETECTION_IOU);
}

function nonMaxSuppression(detections, iouThreshold) {
  if (detections.length === 0) return [];

  // Sort by confidence descending
  detections.sort((a, b) => b.confidence - a.confidence);

  const keep = [];
  const suppressed = new Set();

  for (let i = 0; i < detections.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(detections[i]);

    for (let j = i + 1; j < detections.length; j++) {
      if (suppressed.has(j)) continue;
      if (detections[i].classId !== detections[j].classId) continue;

      const iou = computeBoxIOU(detections[i].bbox, detections[j].bbox);
      if (iou > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return keep;
}

function computeBoxIOU(a, b) {
  const xi1 = Math.max(a.x1, b.x1);
  const yi1 = Math.max(a.y1, b.y1);
  const xi2 = Math.min(a.x2, b.x2);
  const yi2 = Math.min(a.y2, b.y2);

  if (xi2 <= xi1 || yi2 <= yi1) return 0;

  const interArea = (xi2 - xi1) * (yi2 - yi1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  return interArea / (areaA + areaB - interArea);
}

// === Snapshot & Detection ===
async function captureAndDetect() {
  if (!state.modelLoaded) {
    log('Model not loaded yet');
    return;
  }

  const now = Date.now();
  if (now - state.lastSnapshot < 500) return; // Throttle
  state.lastSnapshot = now;

  // Draw current video frame to canvas, get ImageData
  const canvas = document.createElement('canvas');
  canvas.width = els.camera.videoWidth;
  canvas.height = els.camera.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(els.camera, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Run inference
  const detections = await runInference(imageData);
  state.detections = detections;

  // Update game state from detections
  updateGameState(detections);

  // Draw overlay
  drawOverlay(detections);

  // Update UI
  updateDisplay();

  // Run strategy
  runStrategy();

  els.statusText.textContent = `Detected: ${detections.length} tiles | ${state.fps} FPS`;
}
// === Zone System ===
// Zones are stored as normalized coordinates (0-1) relative to video dimensions
function loadZones() {
  try {
    const saved = localStorage.getItem('mj_zones');
    if (saved) state.zones = JSON.parse(saved);
  } catch(e) {}
}

function saveZones() {
  localStorage.setItem('mj_zones', JSON.stringify(state.zones));
}

function zoneToPixel(zone) {
  const vw = els.camera.videoWidth || 640;
  const vh = els.camera.videoHeight || 480;
  return {
    x: zone.x * vw, y: zone.y * vh,
    w: zone.w * vw, h: zone.h * vh
  };
}

function isInZone(det, zoneName) {
  const zone = state.zones[zoneName];
  if (!zone) return false;
  const cx = (det.bbox.x1 + det.bbox.x2) / 2;
  const cy = (det.bbox.y1 + det.bbox.y2) / 2;
  const vw = els.camera.videoWidth || 640;
  const vh = els.camera.videoHeight || 480;
  return cx >= zone.x * vw && cx <= (zone.x + zone.w) * vw &&
         cy >= zone.y * vh && cy <= (zone.y + zone.h) * vh;
}

// === Zone Editing (touch-based drag/resize) ===
let dragState = null; // { zone, handle, startX, startY, origZone }

function initZoneEditor() {
  const canvas = els.overlay;
  canvas.style.pointerEvents = 'auto'; // re-enable for zone editing

  function getHandle(e) {
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const px = touch.clientX - rect.left;
    const py = touch.clientY - rect.top;
    const scaleX = (els.camera.videoWidth || 640) / rect.width;
    const scaleY = (els.camera.videoHeight || 480) / rect.height;
    const vx = px * scaleX;
    const vy = py * scaleY;
    const HANDLE_R = 25; // touch radius

    for (const [name, zone] of Object.entries(state.zones)) {
      const pz = zoneToPixel(zone);
      // Corner handles
      const corners = [
        { h: 'tl', x: pz.x, y: pz.y },
        { h: 'tr', x: pz.x + pz.w, y: pz.y },
        { h: 'bl', x: pz.x, y: pz.y + pz.h },
        { h: 'br', x: pz.x + pz.w, y: pz.y + pz.h },
        { h: 'body', x: pz.x + pz.w/2, y: pz.y + pz.h/2 },
      ];
      for (const c of corners) {
        if (Math.abs(vx - c.x) < HANDLE_R && Math.abs(vy - c.y) < HANDLE_R) {
          return { zoneName: name, handle: c.h };
        }
      }
    }
    return null;
  }

  canvas.addEventListener('touchstart', (e) => {
    if (!state.editingZones) return;
    const hit = getHandle(e);
    if (!hit) return;
    e.preventDefault();
    e.stopPropagation();
    dragState = {
      zoneName: hit.zoneName,
      handle: hit.handle,
      startX: e.touches[0].clientX,
      startY: e.touches[0].clientY,
      origZone: { ...state.zones[hit.zoneName] },
    };
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    if (!dragState || !state.editingZones) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const dx = (e.touches[0].clientX - dragState.startX) / rect.width;
    const dy = (e.touches[0].clientY - dragState.startY) / rect.height;
    const z = dragState.origZone;
    const zone = state.zones[dragState.zoneName];

    switch (dragState.handle) {
      case 'body':
        zone.x = Math.max(0, Math.min(1 - z.w, z.x + dx));
        zone.y = Math.max(0, Math.min(1 - z.h, z.y + dy));
        break;
      case 'tl':
        zone.x = Math.max(0, z.x + dx);
        zone.y = Math.max(0, z.y + dy);
        zone.w = Math.max(0.05, z.w - dx);
        zone.h = Math.max(0.05, z.h - dy);
        break;
      case 'br':
        zone.w = Math.max(0.05, z.w + dx);
        zone.h = Math.max(0.05, z.h + dy);
        break;
      case 'tr':
        zone.y = Math.max(0, z.y + dy);
        zone.w = Math.max(0.05, z.w + dx);
        zone.h = Math.max(0.05, z.h - dy);
        break;
      case 'bl':
        zone.x = Math.max(0, z.x + dx);
        zone.w = Math.max(0.05, z.w - dx);
        zone.h = Math.max(0.05, z.h + dy);
        break;
    }
    drawZoneOverlay();
  }, { passive: false });

  canvas.addEventListener('touchend', () => {
    if (dragState) {
      saveZones();
      drawZoneOverlay();
      dragState = null;
    }
  });
}

function drawZoneOverlay() {
  const canvas = els.overlay;
  const ctx = canvas.getContext('2d');
  // Don't clear — detection overlay draws on top. We draw zones separate.
  if (!state.editingZones) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const [name, zone] of Object.entries(state.zones)) {
    const pz = zoneToPixel(zone);
    const color = name === 'hand' ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)';
    const border = name === 'hand' ? '#4caf50' : '#f44336';
    const label = name === 'hand' ? '🀄 Hand' : '🗑️ Discard';

    // Zone fill
    ctx.fillStyle = color;
    ctx.fillRect(pz.x, pz.y, pz.w, pz.h);

    // Border
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(pz.x, pz.y, pz.w, pz.h);
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = border;
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(label, pz.x + 6, pz.y + 22);

    // Corner handles
    const corners = [
      [pz.x, pz.y], [pz.x + pz.w, pz.y],
      [pz.x, pz.y + pz.h], [pz.x + pz.w, pz.y + pz.h],
    ];
    for (const [cx, cy] of corners) {
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = border;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

// === Game State Update ===
function updateGameState(detections) {
  const engine = TIANJIN_ENGINE;
  const handIds = [];
  const discardIds = [];

  for (const det of detections) {
    if (det.classId >= engine.NUM_TILES) continue;
    if (isInZone(det, 'hand')) {
      handIds.push(det.classId);
    } else if (isInZone(det, 'discard')) {
      discardIds.push(det.classId);
    }
    // Tiles not in any zone are ignored
  }

  state.handTiles = handIds.slice(0, 14);
  // Accumulate discards (don't replace each frame — user clears manually)
  for (const id of discardIds) {
    state.discards.push(id);
  }
}

function drawOverlay(detections) {
  const canvas = els.overlay;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw zones (semi-transparent, always visible)
  for (const [name, zone] of Object.entries(state.zones)) {
    const pz = zoneToPixel(zone);
    const fill = name === 'hand' ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)';
    const border = name === 'hand' ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)';
    ctx.fillStyle = fill;
    ctx.fillRect(pz.x, pz.y, pz.w, pz.h);
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(pz.x, pz.y, pz.w, pz.h);
    ctx.setLineDash([]);
  }

  // Draw detection boxes, colored by zone
  for (const det of detections) {
    const { x1, y1, x2, y2 } = det.bbox;
    const w = x2 - x1;
    const h = y2 - y1;

    let color = '#888'; // unzoned
    if (isInZone(det, 'hand')) color = '#4caf50';
    else if (isInZone(det, 'discard')) color = '#f44336';

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    ctx.fillStyle = color;
    const label = `${det.className} ${Math.round(det.confidence * 100)}%`;
    ctx.font = '11px sans-serif';
    const tw = ctx.measureText(label).width + 6;
    ctx.fillRect(x1, y1 - 18, tw, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 3, y1 - 5);
  }
}

// === UI Updates (no change from original, except editing toggle) ===
function updateDisplay() {
  const tileCounts = TIANJIN_ENGINE.countTiles(state.handTiles);
  els.tileCount.textContent = `${state.handTiles.length} tiles`;

  let handHtml = '';
  for (let i = 0; i < TIANJIN_ENGINE.NUM_TILES; i++) {
    if (tileCounts[i] > 0) {
      handHtml += `<span class="tile-badge">${TIANJIN_ENGINE.TILE_NAMES[i]}×${tileCounts[i]}</span>`;
    }
  }
  if (!handHtml) handHtml = '<span class="rec-waiting">No tiles in hand zone</span>';
  els.handTiles.innerHTML = handHtml;

  const rawNames = state.handTiles.map(id => TIANJIN_ENGINE.ID_TO_TILE[id]).join(', ');
  els.handRaw.textContent = rawNames ? `[${rawNames}]` : '';

  const discardCounts = TIANJIN_ENGINE.countTiles(state.discards);
  let discardHtml = '';
  for (let i = 0; i < TIANJIN_ENGINE.NUM_TILES; i++) {
    if (discardCounts[i] > 0) {
      discardHtml += `<span class="tile-badge">${TIANJIN_ENGINE.TILE_NAMES[i]}×${discardCounts[i]}</span>`;
    }
  }
  if (!discardHtml) discardHtml = '<span class="rec-waiting">None yet</span>';
  els.discardTiles.innerHTML = discardHtml;

  // Hun indicator
  const hunEl = document.getElementById('hun-display');
  if (hunEl) {
    hunEl.textContent = state.hunDora != null
      ? `混儿: ${TIANJIN_ENGINE.TILE_NAMES[state.hunDora]}`
      : '混儿: not set';
  }
}

// === Strategy (unchanged logic, uses zones now) ===
function runStrategy() {
  if (state.handTiles.length < 13) {
    els.recContent.innerHTML = '<p class="rec-waiting">Need at least 13 tiles in hand zone</p>';
    return;
  }

  const engine = TIANJIN_ENGINE;
  const hunDoraId = state.hunDora;

  if (state.handTiles.length === 14) {
    const results = engine.analyzeDiscard(
      state.handTiles, state.discards, state.melds, hunDoraId
    );
    if (results.length === 0) {
      els.recContent.innerHTML = '<p class="rec-waiting">No valid discards found</p>';
      return;
    }

    const best = results[0];
    state.lastRecommendation = best;

    let html = `<div class="rec-result">`;
    html += `<div class="rec-top">打 ${best.tileName}</div>`;

    if (best.shanten === -1) {
      html += `<div class="rec-detail">🎉 胡了！</div>`;
    } else if (best.shanten === 0 && best.waitCount > 0) {
      html += `<div class="rec-detail">听 ${best.waits.length} 种 ${best.waitCount} 张</div>`;
      const waitNames = best.waits.map(w =>
        `${engine.TILE_NAMES[w.tileId]}(${w.remaining})`
      ).join(' ');
      html += `<div class="rec-detail">等: ${waitNames}</div>`;
    } else {
      html += `<div class="rec-detail">距听牌 ${best.shanten} 步</div>`;
    }

    html += `<div class="rec-detail">安全度: ${engine.safetyEmoji(best.safety)} ${best.safety}</div>`;

    if (results.length > 1) {
      html += `<div class="rec-options">`;
      for (let i = 1; i < Math.min(results.length, 4); i++) {
        const r = results[i];
        const safetyClass = r.safety >= 70 ? 'safety-safe' : r.safety >= 40 ? 'safety-caution' : 'safety-danger';
        html += `<div class="rec-option ${safetyClass}">`;
        html += `<span class="tile-name">${r.tileName}</span>`;
        html += r.shanten === 0
          ? `<span class="tile-info">听 ${r.waitCount} 张</span>`
          : `<span class="tile-info">${r.shanten} 向听 | ${engine.safetyEmoji(r.safety)} ${r.safety}</span>`;
        html += `</div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
    els.recContent.innerHTML = html;
    speakRecommendation(best);

  } else if (state.handTiles.length === 13) {
    const remaining = engine.getRemaining(state.handTiles, state.discards, state.melds);
    const shanten = engine.calcShanten(state.handTiles, hunDoraId, remaining);

    if (shanten === 0) {
      const waits = engine.findWaits(state.handTiles, hunDoraId, remaining);
      const totalWait = waits.reduce((s, w) => s + w.remaining, 0);
      const waitNames = waits.map(w => `${engine.TILE_NAMES[w.tileId]}(${w.remaining})`).join(' ');
      els.recContent.innerHTML = `
        <div class="rec-result">
          <div class="rec-top">🎯 听牌！</div>
          <div class="rec-detail">等 ${waits.length} 种 ${totalWait} 张</div>
          <div class="rec-detail">${waitNames}</div>
        </div>`;
    } else {
      els.recContent.innerHTML = `
        <div class="rec-result">
          <div class="rec-top">📊 ${shanten} 向听</div>
          <div class="rec-detail">距听牌还差 ${shanten} 步</div>
        </div>`;
    }
  }
}

// === Voice Output ===
function speakRecommendation(recommendation) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const text = buildVoiceText(recommendation);
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'zh-CN';
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

function buildVoiceText(rec) {
  const parts = [`打${rec.tileName}`];
  if (rec.shanten === -1) parts.push('恭喜，胡了！');
  else if (rec.shanten === 0) parts.push(`听${rec.waitCount}张`);
  else parts.push(`距听牌还差${rec.shanten}步`);
  return parts.join('，');
}

// === Manual Controls ===
function toggleRunning() {
  state.running = !state.running;
  els.btnToggle.textContent = state.running ? '⏸ Pause' : '▶ Start';
  if (state.running) { log('Auto-detection started'); autoSnapshotLoop(); }
  else { log('Auto-detection paused'); }
}

async function manualSnapshot() {
  log('Manual snapshot');
  await captureAndDetect();
}

function clearDiscards() {
  state.discards = [];
  updateDisplay();
  log('Discards cleared');
}

function toggleZoneEdit() {
  state.editingZones = !state.editingZones;
  const btn = document.getElementById('btn-zone-edit');
  btn.textContent = state.editingZones ? '✅ Done' : '✏️ Zones';
  btn.style.background = state.editingZones ? 'var(--safe)' : '';
  if (state.editingZones) {
    drawZoneOverlay();
    log('Zone editing: drag corners to resize, drag center to move');
  } else {
    saveZones();
    els.overlay.getContext('2d').clearRect(0, 0, els.overlay.width, els.overlay.height);
  }
}

// === Auto-snapshot Loop ===
let snapshotTimer = null;
function autoSnapshotLoop() {
  if (!state.running) return;
  captureAndDetect().finally(() => {
    if (state.running) snapshotTimer = setTimeout(autoSnapshotLoop, SNAPSHOT_INTERVAL);
  });
}

// === Initialization ===
async function init() {
  log('Tianjin Mahjong AI — Initializing...');
  els.statusText.textContent = 'Initializing...';

  loadZones();
  await setupCamera();
  await loadModel();

  // Build wild card picker
  buildHunPicker();

  // Event listeners
  els.btnToggle.addEventListener('click', toggleRunning);
  els.btnSnapshot.addEventListener('click', manualSnapshot);
  els.btnClearDiscards.addEventListener('click', clearDiscards);

  const zoneBtn = document.getElementById('btn-zone-edit');
  if (zoneBtn) zoneBtn.addEventListener('click', toggleZoneEdit);

  initZoneEditor();

  // Auto-start
  state.running = true;
  els.btnToggle.textContent = '⏸ Pause';
  els.statusText.textContent = 'Ready — position zones, set 混儿, then detect';
  log('Ready ✓');
  autoSnapshotLoop();
}

// === Wild Card Picker ===
function buildHunPicker() {
  const container = document.getElementById('hun-picker');
  if (!container) return;

  const engine = TIANJIN_ENGINE;
  let html = '<div class="hun-label">混儿 (wild card): </div>';
  html += '<div class="hun-tiles">';

  for (let i = 0; i < engine.NUM_TILES; i++) {
    const selected = state.hunDora === i ? ' selected' : '';
    html += `<span class="hun-tile${selected}" data-tile="${i}">${engine.TILE_NAMES[i]}</span>`;
  }
  html += '</div>';
  html += '<button id="btn-clear-hun" class="small-btn">Clear</button>';
  container.innerHTML = html;

  // Click handlers
  container.querySelectorAll('.hun-tile').forEach(el => {
    el.addEventListener('click', () => {
      const id = parseInt(el.dataset.tile);
      state.hunDora = (state.hunDora === id) ? null : id;
      buildHunPicker(); // rebuild to update selection
      updateDisplay();
      log(`混儿 set to: ${state.hunDora != null ? engine.TILE_NAMES[state.hunDora] : 'none'}`);
    });
  });

  document.getElementById('btn-clear-hun').addEventListener('click', () => {
    state.hunDora = null;
    buildHunPicker();
    updateDisplay();
    log('混儿 cleared');
  });
}

// === Start ===
init().catch(err => {
  log(`Init failed: ${err.message}`);
  els.statusText.textContent = 'Initialization failed';
});
