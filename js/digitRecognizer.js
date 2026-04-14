/**
 * digitRecognizer.js
 *
 * v1.3.0: セル単位OCR復帰 + 前処理強化
 *   - セルごとにグレースケール画像をTesseract (PSM 10: 単一文字) で認識
 *   - 各セルで複数前処理バリエーションを試し多数決
 *   - 空セル検出 (stddev) で不要な認識をスキップ
 *   - パディング追加で数字輪郭がフレームに接しないように
 *   - 数独制約チェックで重複除去
 */

const DigitRecognizer = (() => {
  let _ocrWorker = null;
  let _ocrReady = false;

  async function _initOCR() {
    if (_ocrReady && _ocrWorker) return _ocrWorker;
    try {
      _ocrWorker = await Tesseract.createWorker('eng', 1, {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            // 静かにする
          }
        }
      });
      await _ocrWorker.setParameters({
        tessedit_char_whitelist: '123456789',
        tessedit_pageseg_mode: '10', // PSM_SINGLE_CHAR
      });
      _ocrReady = true;
      console.log('Tesseract.js OCR worker ready (PSM 10)');
      return _ocrWorker;
    } catch (err) {
      console.warn('Tesseract.js init failed:', err);
      _ocrWorker = null;
      _ocrReady = false;
      return null;
    }
  }

  async function loadModel() { await _initOCR(); }

  // ── メイン認識 ──
  async function recognize(cells, cellsGray, warpedCanvas) {
    await loadModel();
    if (!_ocrWorker || !_ocrReady) {
      console.error('OCR worker not available');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const source = cellsGray || cells;
    if (!source) {
      console.error('No cell data');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    let recognized = 0;
    let skipped = 0;

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cell = source[r][c];
        if (!cell) { skipped++; continue; }

        // 空セル判定
        if (_isEmpty(cell)) { skipped++; continue; }

        // 複数バリエーションで認識
        const digit = await _recognizeCell(cell);
        if (digit > 0) {
          grid[r][c] = digit;
          recognized++;
        }
      }
    }

    console.log('Recognized: ' + recognized + ', skipped: ' + skipped);

    // 数独制約チェック
    _resolveConflicts(grid);

    const final = grid.flat().filter(v => v !== 0).length;
    console.log('After conflict resolution: ' + final + ' digits');
    return grid;
  }

  // ────────────────────────────────────────
  // 空セル判定: 中央領域のピクセル標準偏差
  // ────────────────────────────────────────
  function _isEmpty(cellImageData) {
    const d = cellImageData.data;
    const w = cellImageData.width;
    const h = cellImageData.height;
    const margin = Math.floor(Math.min(w, h) * 0.2);

    let sum = 0, sumSq = 0, count = 0;
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        const idx = (y * w + x) * 4;
        const v = d[idx];
        sum += v;
        sumSq += v * v;
        count++;
      }
    }
    if (count === 0) return true;
    const mean = sum / count;
    const stddev = Math.sqrt(Math.max(0, sumSq / count - mean * mean));

    // 黒ピクセル率も考慮
    let darkCount = 0;
    const threshold = mean * 0.6;
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        const idx = (y * w + x) * 4;
        if (d[idx] < threshold) darkCount++;
      }
    }
    const darkRatio = darkCount / count;

    // stddev低い = 均一 = 空, darkRatio低すぎ = 数字なし
    return stddev < 15 || darkRatio < 0.02;
  }

  // ────────────────────────────────────────
  // セル単位で複数前処理バリエーションを試して多数決
  // ────────────────────────────────────────
  async function _recognizeCell(cellImageData) {
    const variants = _createVariants(cellImageData);
    const votes = {};
    const confs = {};

    for (const canvas of variants) {
      try {
        const result = await _ocrWorker.recognize(canvas);
        if (result.data && result.data.text) {
          const text = result.data.text.trim();
          // 単一文字で1-9のみ受け付ける
          if (/^[1-9]$/.test(text)) {
            const d = parseInt(text, 10);
            const conf = result.data.confidence || 0;
            votes[d] = (votes[d] || 0) + 1;
            if (!confs[d] || conf > confs[d]) confs[d] = conf;
          }
        }
      } catch (e) {
        // skip
      }
    }

    const digits = Object.keys(votes).map(Number);
    if (digits.length === 0) return 0;

    digits.sort((a, b) => {
      const vd = votes[b] - votes[a];
      if (vd !== 0) return vd;
      return (confs[b] || 0) - (confs[a] || 0);
    });

    const best = digits[0];
    const bestVotes = votes[best];
    const bestConf = confs[best] || 0;

    // 最低でも2票、または1票でも信頼度80以上
    if (bestVotes >= 2 || bestConf >= 80) {
      return best;
    }

    return 0;
  }

  // ────────────────────────────────────────
  // 前処理バリエーション: 各種変換したcanvasの配列
  // ────────────────────────────────────────
  function _createVariants(cellImageData) {
    const results = [];
    const w = cellImageData.width;
    const h = cellImageData.height;

    // V1: グレースケールそのまま + パディング + 3倍拡大
    results.push(_prepareCell(cellImageData, false, 3));

    // V2: グレースケール + パディング + 4倍拡大
    results.push(_prepareCell(cellImageData, false, 4));

    // V3: コントラスト強調 + パディング + 3倍
    results.push(_prepareCell(cellImageData, true, 3));

    // V4: Otsu二値化 + パディング + 3倍
    results.push(_prepareCellBinarized(cellImageData, 3));

    // V5: コントラスト強調 + 4倍
    results.push(_prepareCell(cellImageData, true, 4));

    return results;
  }

  // ────────────────────────────────────────
  // セル画像の前処理: パディング + 拡大
  // ────────────────────────────────────────
  function _prepareCell(cellImageData, enhanceContrast, scale) {
    const w = cellImageData.width;
    const h = cellImageData.height;

    // パディング: 数字の周囲に白い余白を確保 (Tesseractは余白があると認識精度が上がる)
    const pad = Math.floor(Math.min(w, h) * 0.3);
    const pw = w + pad * 2;
    const ph = h + pad * 2;

    const canvas = document.createElement('canvas');
    canvas.width = pw * scale;
    canvas.height = ph * scale;
    const ctx = canvas.getContext('2d');

    // 白背景
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 一時canvasにImageDataを描画
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    const tmpCtx = tmp.getContext('2d');

    if (enhanceContrast) {
      const imgData = new ImageData(new Uint8ClampedArray(cellImageData.data), w, h);
      _enhanceImageData(imgData);
      tmpCtx.putImageData(imgData, 0, 0);
    } else {
      tmpCtx.putImageData(cellImageData, 0, 0);
    }

    // 拡大描画 (パディング込み)
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, w, h, pad * scale, pad * scale, w * scale, h * scale);

    return canvas;
  }

  // ────────────────────────────────────────
  // Otsu二値化バリエーション
  // ────────────────────────────────────────
  function _prepareCellBinarized(cellImageData, scale) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const d = cellImageData.data;

    // Otsu閾値計算
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < d.length; i += 4) {
      histogram[d[i]]++;
    }
    const total = w * h;
    let sumB = 0, wB = 0, max = 0, sum = 0, thresh = 128;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];
    for (let i = 0; i < 256; i++) {
      wB += histogram[i];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += i * histogram[i];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > max) { max = between; thresh = i; }
    }

    // 二値化 (黒文字 on 白背景)
    const imgData = new ImageData(new Uint8ClampedArray(d.length), w, h);
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i] < thresh ? 0 : 255;
      imgData.data[i] = imgData.data[i + 1] = imgData.data[i + 2] = v;
      imgData.data[i + 3] = 255;
    }

    // パディング + 拡大
    const pad = Math.floor(Math.min(w, h) * 0.3);
    const pw = w + pad * 2;
    const ph = h + pad * 2;
    const canvas = document.createElement('canvas');
    canvas.width = pw * scale;
    canvas.height = ph * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d').putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false; // 二値化画像はスムージングしない
    ctx.drawImage(tmp, 0, 0, w, h, pad * scale, pad * scale, w * scale, h * scale);

    return canvas;
  }

  // ────────────────────────────────────────
  // ImageData のコントラスト強調 (min-max stretch)
  // ────────────────────────────────────────
  function _enhanceImageData(imgData) {
    const d = imgData.data;
    let minV = 255, maxV = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] < minV) minV = d[i];
      if (d[i] > maxV) maxV = d[i];
    }
    const range = Math.max(1, maxV - minV);
    for (let i = 0; i < d.length; i += 4) {
      const s = Math.round(((d[i] - minV) / range) * 255);
      d[i] = d[i + 1] = d[i + 2] = s;
    }
  }

  // ────────────────────────────────────────
  // 数独制約チェック
  // ────────────────────────────────────────
  function _resolveConflicts(grid) {
    let changed = true;
    while (changed) {
      changed = false;
      for (let row = 0; row < 9; row++) {
        const seen = {};
        for (let col = 0; col < 9; col++) {
          const v = grid[row][col];
          if (v === 0) continue;
          if (seen[v] !== undefined) { grid[row][col] = 0; changed = true; }
          else seen[v] = col;
        }
      }
      for (let col = 0; col < 9; col++) {
        const seen = {};
        for (let row = 0; row < 9; row++) {
          const v = grid[row][col];
          if (v === 0) continue;
          if (seen[v] !== undefined) { grid[row][col] = 0; changed = true; }
          else seen[v] = row;
        }
      }
      for (let br = 0; br < 9; br += 3) {
        for (let bc = 0; bc < 9; bc += 3) {
          const seen = {};
          for (let r = br; r < br + 3; r++) {
            for (let c = bc; c < bc + 3; c++) {
              const v = grid[r][c];
              if (v === 0) continue;
              if (seen[v]) { grid[r][c] = 0; changed = true; }
              else seen[v] = true;
            }
          }
        }
      }
    }
  }

  async function terminate() {
    if (_ocrWorker) {
      await _ocrWorker.terminate();
      _ocrWorker = null;
      _ocrReady = false;
    }
  }

  return { loadModel, recognize, terminate };
})();
