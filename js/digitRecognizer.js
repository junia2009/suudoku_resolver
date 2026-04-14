/**
 * digitRecognizer.js
 *
 * 新アプローチ: グリッド全体の一括OCR
 *
 * 認識戦略:
 *   1. 補正済みグリッド画像全体をTesseract.jsに渡す
 *   2. 検出された文字の座標からセル位置(row, col)を逆算
 *   3. 複数の前処理バリエーションで試行し、結果をマージ
 *   4. 数独制約チェックで重複を除去
 *
 * なぜこのアプローチか:
 *   - セル単位の切り出し+二値化で数字が壊れる問題を回避
 *   - Tesseractは大きな画像のほうが文字認識精度が高い
 *   - 文字の位置情報(bounding box)がセル特定に使える
 */

const DigitRecognizer = (() => {
  let _ocrWorker = null;
  let _ocrReady = false;

  // ─────────────────────────────────────────────
  // Tesseract.js OCR ワーカー初期化
  // ─────────────────────────────────────────────
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
        tessedit_pageseg_mode: '6', // PSM_SINGLE_BLOCK
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

  async function loadModel() {
    await _initOCR();
  }

  // ─────────────────────────────────────────────
  // メイン認識: グリッド全体画像から数字を一括認識
  // ─────────────────────────────────────────────
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

    console.log('Grid OCR: ' + gridW + 'x' + gridH + ', cell: ' + cellW.toFixed(1) + 'x' + cellH.toFixed(1));

    const allResults = [];

    // バリエーション1: 2倍拡大
    const s2 = _scaleCanvas(warpedCanvas, 2);
    const r1 = await _recognizeWholeGrid(s2, cellW, cellH, 2);
    allResults.push(...r1);
    console.log('Variant 1 (2x): ' + r1.length + ' chars');

    // バリエーション2: 3倍拡大
    const s3 = _scaleCanvas(warpedCanvas, 3);
    const r2 = await _recognizeWholeGrid(s3, cellW, cellH, 3);
    allResults.push(...r2);
    console.log('Variant 2 (3x): ' + r2.length + ' chars');

    // バリエーション3: コントラスト強調 + 2倍
    const enh = _enhanceContrast(warpedCanvas);
    const es2 = _scaleCanvas(enh, 2);
    const r3 = await _recognizeWholeGrid(es2, cellW, cellH, 2);
    allResults.push(...r3);
    console.log('Variant 3 (enhanced 2x): ' + r3.length + ' chars');

    // バリエーション4: シャープ化 + 3倍
    const shp = _sharpen(warpedCanvas);
    const ss3 = _scaleCanvas(shp, 3);
    const r4 = await _recognizeWholeGrid(ss3, cellW, cellH, 3);
    allResults.push(...r4);
    console.log('Variant 4 (sharp 3x): ' + r4.length + ' chars');

    // 全結果をセル単位で多数決マージ
    const grid = _mergeResults(allResults);

    // 数独制約チェック
    _resolveConflicts(grid);

    console.log('Final: ' + grid.flat().filter(v => v !== 0).length + ' digits');
    return grid;
  }

  // ─────────────────────────────────────────────
  // グリッド全体画像をOCRして文字位置を取得
  // ─────────────────────────────────────────────
  async function _recognizeWholeGrid(canvas, cellW, cellH, scale) {
    try {
      const result = await _ocrWorker.recognize(canvas);
      const detections = [];

      // 全シンボル(文字)をチェック
      if (result.data.symbols) {
        for (const sym of result.data.symbols) {
          _processSymbol(sym, cellW, cellH, scale, detections);
        }
      }

      // symbols がない場合は words → symbols
      if (detections.length === 0 && result.data.words) {
        for (const word of result.data.words) {
          if (word.symbols) {
            for (const sym of word.symbols) {
              _processSymbol(sym, cellW, cellH, scale, detections);
            }
          } else {
            // symbols もない場合は word の text と bbox で推定
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
                  digit: parseInt(text[i], 10),
                  row, col,
                  confidence: word.confidence || 0,
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
        digit: parseInt(ch, 10),
        row, col,
        confidence: sym.confidence || 0,
      });
    }
  }

  // ─────────────────────────────────────────────
  // 全バリエーションの結果をセル単位で多数決マージ
  // ─────────────────────────────────────────────
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

        // 最多投票の数字を選択
        digits.sort((a, b) => {
          const voteDiff = votes[b] - votes[a];
          if (voteDiff !== 0) return voteDiff;
          return (cellBestConf[r][c][b] || 0) - (cellBestConf[r][c][a] || 0);
        });

        const bestDigit = digits[0];
        const bestVotes = votes[bestDigit];
        const conf = cellBestConf[r][c][bestDigit] || 0;

        // 2票以上、または1票でも高信頼(>=85)なら確定
        if (bestVotes >= 2 || conf >= 85) {
          grid[r][c] = bestDigit;
        }
      }
    }

    return grid;
  }

  // ─────────────────────────────────────────────
  // Canvas を均等拡大
  // ─────────────────────────────────────────────
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

  // ─────────────────────────────────────────────
  // コントラスト強調 (min-max stretch)
  // ─────────────────────────────────────────────
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
      const stretched = Math.round(((v - minV) / range) * 255);
      d[i] = stretched;
      d[i + 1] = stretched;
      d[i + 2] = stretched;
    }

    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  // ─────────────────────────────────────────────
  // シャープ化 (アンシャープマスク)
  // ─────────────────────────────────────────────
  function _sharpen(canvas) {
    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext('2d');

    const blurCanvas = document.createElement('canvas');
    blurCanvas.width = canvas.width;
    blurCanvas.height = canvas.height;
    const blurCtx = blurCanvas.getContext('2d');
    blurCtx.filter = 'blur(1px)';
    blurCtx.drawImage(canvas, 0, 0);

    const origCanvas = document.createElement('canvas');
    origCanvas.width = canvas.width;
    origCanvas.height = canvas.height;
    origCanvas.getContext('2d').drawImage(canvas, 0, 0);
    const orig = origCanvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    const blur = blurCtx.getImageData(0, 0, canvas.width, canvas.height);

    const result = ctx.createImageData(canvas.width, canvas.height);
    const amount = 1.5;

    for (let i = 0; i < orig.data.length; i += 4) {
      for (let ch = 0; ch < 3; ch++) {
        const val = orig.data[i + ch] + (orig.data[i + ch] - blur.data[i + ch]) * amount;
        result.data[i + ch] = Math.max(0, Math.min(255, Math.round(val)));
      }
      result.data[i + 3] = 255;
    }

    ctx.putImageData(result, 0, 0);
    return out;
  }

  // ─────────────────────────────────────────────
  // 数独制約チェック: 重複を削除
  // ─────────────────────────────────────────────
  function _resolveConflicts(grid) {
    let changed = true;
    while (changed) {
      changed = false;

      for (let row = 0; row < 9; row++) {
        const seen = {};
        for (let col = 0; col < 9; col++) {
          const val = grid[row][col];
          if (val === 0) continue;
          if (seen[val] !== undefined) {
            grid[row][col] = 0;
            changed = true;
          } else {
            seen[val] = col;
          }
        }
      }

      for (let col = 0; col < 9; col++) {
        const seen = {};
        for (let row = 0; row < 9; row++) {
          const val = grid[row][col];
          if (val === 0) continue;
          if (seen[val] !== undefined) {
            grid[row][col] = 0;
            changed = true;
          } else {
            seen[val] = row;
          }
        }
      }

      for (let br = 0; br < 9; br += 3) {
        for (let bc = 0; bc < 9; bc += 3) {
          const seen = {};
          for (let r = br; r < br + 3; r++) {
            for (let c = bc; c < bc + 3; c++) {
              const val = grid[r][c];
              if (val === 0) continue;
              if (seen[val]) {
                grid[r][c] = 0;
                changed = true;
              } else {
                seen[val] = true;
              }
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
