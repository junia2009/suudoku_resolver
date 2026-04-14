/**
 * digitRecognizer.js
 *
 * v1.2.0: ハイブリッドアプローチ
 *   - OpenCVでグリッド線を形態学的に除去 → 全体OCR (PSM 11 sparse text)
 *   - 元画像でも全体OCR (PSM 6 single block) → バリエーション追加
 *   - セルのピクセル標準偏差で空セルを検出し誤検出をフィルタ
 *   - 5バリエーションの多数決マージ
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
            console.log('OCR progress: ' + Math.round(m.progress * 100) + '%');
          }
        }
      });
      await _ocrWorker.setParameters({
        tessedit_char_whitelist: '123456789',
        tessedit_pageseg_mode: '6',
      });
      _ocrReady = true;
      console.log('Tesseract.js OCR worker ready');
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
    if (!_ocrWorker || !_ocrReady || !warpedCanvas) {
      console.error('OCR worker or warped canvas not available');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const gridW = warpedCanvas.width;
    const gridH = warpedCanvas.height;
    const cellW = gridW / 9;
    const cellH = gridH / 9;
    console.log('Grid: ' + gridW + 'x' + gridH + ', cell: ' + cellW.toFixed(1) + 'x' + cellH.toFixed(1));

    // 空セル検出 (グレースケールセルの標準偏差)
    const emptyCells = _detectEmptyCells(cellsGray || cells);
    console.log('Non-empty cells: ' + emptyCells.flat().filter(v => !v).length);

    const allResults = [];

    // ── グリッド線除去を試行 ──
    let cleaned = null;
    if (typeof cv !== 'undefined' && cv.Mat) {
      try {
        cleaned = _removeGridLines(warpedCanvas);
        console.log('Grid lines removed');
      } catch (e) {
        console.warn('Grid line removal failed:', e);
      }
    }

    // ── 線除去画像で OCR (PSM 11: sparse text) ──
    if (cleaned) {
      try {
        await _ocrWorker.setParameters({ tessedit_pageseg_mode: '11' });
      } catch (e) {
        console.warn('PSM 11 not supported, using PSM 6');
        await _ocrWorker.setParameters({ tessedit_pageseg_mode: '6' });
      }

      // V1: cleaned 2x
      const c2 = _scaleCanvas(cleaned, 2);
      const r1 = await _recognizeWholeGrid(c2, cellW, cellH, 2);
      allResults.push(...r1);
      console.log('V1 (cleaned sparse 2x): ' + r1.length);

      // V2: cleaned 3x
      const c3 = _scaleCanvas(cleaned, 3);
      const r2 = await _recognizeWholeGrid(c3, cellW, cellH, 3);
      allResults.push(...r2);
      console.log('V2 (cleaned sparse 3x): ' + r2.length);

      // V3: cleaned + contrast enhanced 2x
      const ec = _enhanceContrast(cleaned);
      const ec2 = _scaleCanvas(ec, 2);
      const r3 = await _recognizeWholeGrid(ec2, cellW, cellH, 2);
      allResults.push(...r3);
      console.log('V3 (cleaned enhanced 2x): ' + r3.length);
    }

    // ── 元画像(線あり)で OCR (PSM 6: single block) ──
    await _ocrWorker.setParameters({ tessedit_pageseg_mode: '6' });

    // V4: original 2x
    const s2 = _scaleCanvas(warpedCanvas, 2);
    const r4 = await _recognizeWholeGrid(s2, cellW, cellH, 2);
    allResults.push(...r4);
    console.log('V4 (original block 2x): ' + r4.length);

    // V5: original 3x
    const s3 = _scaleCanvas(warpedCanvas, 3);
    const r5 = await _recognizeWholeGrid(s3, cellW, cellH, 3);
    allResults.push(...r5);
    console.log('V5 (original block 3x): ' + r5.length);

    // ── 空セルフィルタ ──
    const filtered = allResults.filter(d => !emptyCells[d.row][d.col]);
    console.log('After empty filter: ' + filtered.length + '/' + allResults.length);

    // ── 多数決マージ ──
    const grid = _mergeResults(filtered);

    // ── 数独制約チェック ──
    _resolveConflicts(grid);

    console.log('Final: ' + grid.flat().filter(v => v !== 0).length + ' digits');
    return grid;
  }

  // ────────────────────────────────────────
  // 空セル検出: セル内ピクセルの標準偏差が低い = 空
  // ────────────────────────────────────────
  function _detectEmptyCells(cellSource) {
    const result = Array.from({ length: 9 }, () => new Array(9).fill(false));
    if (!cellSource) return result;

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cell = cellSource[r][c];
        if (!cell) { result[r][c] = true; continue; }

        const d = cell.data;
        const w = cell.width;
        const h = cell.height;
        const margin = Math.floor(Math.min(w, h) * 0.15);

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

        if (count === 0) { result[r][c] = true; continue; }
        const mean = sum / count;
        const stddev = Math.sqrt(Math.max(0, sumSq / count - mean * mean));
        result[r][c] = stddev < 18;
      }
    }
    return result;
  }

  // ────────────────────────────────────────
  // OpenCV で グリッド線を形態学的に除去
  // ────────────────────────────────────────
  function _removeGridLines(canvas) {
    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // 適応的二値化 (反転: 文字・線が白)
    const binary = new cv.Mat();
    cv.adaptiveThreshold(gray, binary, 255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 10);

    const cols = gray.cols;
    const rows = gray.rows;

    // 水平線検出: 長い水平構造のみ残す
    const hLen = Math.floor(cols / 12);
    const hKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(hLen, 1));
    const hLines = new cv.Mat();
    cv.morphologyEx(binary, hLines, cv.MORPH_OPEN, hKernel);

    // 垂直線検出: 長い垂直構造のみ残す
    const vLen = Math.floor(rows / 12);
    const vKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, vLen));
    const vLines = new cv.Mat();
    cv.morphologyEx(binary, vLines, cv.MORPH_OPEN, vKernel);

    // 線を合成
    const lines = new cv.Mat();
    cv.add(hLines, vLines, lines);

    // 線を少し膨張して確実に除去
    const dKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const linesDil = new cv.Mat();
    cv.dilate(lines, linesDil, dKernel);

    // 二値画像から線を除去 → 文字だけ残る
    const cleaned = new cv.Mat();
    cv.subtract(binary, linesDil, cleaned);

    // 小さなノイズ除去
    const morphK = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    const denoised = new cv.Mat();
    cv.morphologyEx(cleaned, denoised, cv.MORPH_OPEN, morphK);

    // 反転: 黒文字 on 白背景 (Tesseract向き)
    const result = new cv.Mat();
    cv.bitwise_not(denoised, result);

    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    cv.imshow(out, result);

    // メモリ解放
    [src, gray, binary, hLines, vLines, lines, hKernel, vKernel,
     dKernel, linesDil, cleaned, morphK, denoised, result].forEach(m => m.delete());

    return out;
  }

  // ────────────────────────────────────────
  // グリッド全体画像を OCR
  // ────────────────────────────────────────
  async function _recognizeWholeGrid(canvas, cellW, cellH, scale) {
    try {
      const result = await _ocrWorker.recognize(canvas);
      const detections = [];

      // words → symbols ルートで文字座標を取得
      if (result.data && result.data.words) {
        for (const word of result.data.words) {
          if (word.symbols && word.symbols.length > 0) {
            for (const sym of word.symbols) {
              _processSymbol(sym, cellW, cellH, scale, detections);
            }
          } else {
            // symbols がない場合は word の text + bbox で推定
            const text = word.text.trim();
            for (let i = 0; i < text.length; i++) {
              if (!/^[1-9]$/.test(text[i])) continue;
              const bbox = word.bbox;
              const charWidth = (bbox.x1 - bbox.x0) / Math.max(text.length, 1);
              const cx = (bbox.x0 + charWidth * (i + 0.5)) / scale;
              const cy = (bbox.y0 + bbox.y1) / 2 / scale;
              const col = Math.floor(cx / cellW);
              const row = Math.floor(cy / cellH);
              if (row >= 0 && row < 9 && col >= 0 && col < 9) {
                detections.push({
                  digit: parseInt(text[i], 10), row, col,
                  confidence: word.confidence || 0
                });
              }
            }
          }
        }
      }

      return detections;
    } catch (err) {
      console.warn('Grid OCR error:', err);
      return [];
    }
  }

  function _processSymbol(sym, cellW, cellH, scale, detections) {
    const ch = sym.text.trim();
    if (!/^[1-9]$/.test(ch)) return;
    const bbox = sym.bbox;
    const cx = (bbox.x0 + bbox.x1) / 2 / scale;
    const cy = (bbox.y0 + bbox.y1) / 2 / scale;
    const col = Math.floor(cx / cellW);
    const row = Math.floor(cy / cellH);
    if (row >= 0 && row < 9 && col >= 0 && col < 9) {
      detections.push({
        digit: parseInt(ch, 10), row, col,
        confidence: sym.confidence || 0
      });
    }
  }

  // ────────────────────────────────────────
  // 多数決マージ
  // ────────────────────────────────────────
  function _mergeResults(allDetections) {
    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    const cellVotes = Array.from({ length: 9 }, () =>
      Array.from({ length: 9 }, () => ({}))
    );
    const cellBestConf = Array.from({ length: 9 }, () =>
      Array.from({ length: 9 }, () => ({}))
    );

    for (const det of allDetections) {
      const { row, col, digit, confidence } = det;
      cellVotes[row][col][digit] = (cellVotes[row][col][digit] || 0) + 1;
      if (!cellBestConf[row][col][digit] || confidence > cellBestConf[row][col][digit]) {
        cellBestConf[row][col][digit] = confidence;
      }
    }

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const votes = cellVotes[r][c];
        const digits = Object.keys(votes).map(Number);
        if (digits.length === 0) continue;

        digits.sort((a, b) => {
          const d = votes[b] - votes[a];
          if (d !== 0) return d;
          return (cellBestConf[r][c][b] || 0) - (cellBestConf[r][c][a] || 0);
        });

        const best = digits[0];
        const bestVotes = votes[best];
        const conf = cellBestConf[r][c][best] || 0;

        // 2票以上、または1票でも信頼度85以上なら確定
        if (bestVotes >= 2 || conf >= 85) {
          grid[r][c] = best;
        }
      }
    }

    return grid;
  }

  // ────────────────────────────────────────
  // Canvas 拡大
  // ────────────────────────────────────────
  function _scaleCanvas(canvas, scale) {
    const out = document.createElement('canvas');
    out.width = canvas.width * scale;
    out.height = canvas.height * scale;
    const ctx = out.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(canvas, 0, 0, out.width, out.height);
    return out;
  }

  // ────────────────────────────────────────
  // コントラスト強調 (min-max stretch)
  // ────────────────────────────────────────
  function _enhanceContrast(canvas) {
    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext('2d');
    ctx.drawImage(canvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, out.width, out.height);
    const d = imgData.data;
    let minV = 255, maxV = 0;
    for (let i = 0; i < d.length; i += 4) {
      const v = (d[i] + d[i + 1] + d[i + 2]) / 3;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const range = Math.max(1, maxV - minV);
    for (let i = 0; i < d.length; i += 4) {
      const v = (d[i] + d[i + 1] + d[i + 2]) / 3;
      const s = Math.round(((v - minV) / range) * 255);
      d[i] = d[i + 1] = d[i + 2] = s;
    }
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  // ────────────────────────────────────────
  // 数独制約チェック: 行・列・ブロックの重複を除去
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
