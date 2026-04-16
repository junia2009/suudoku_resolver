/**
 * digitRecognizer.js  v1.8.0
 *
 * トリプルPSMアプローチ + Otsu二値化 + 連結成分ベース切り出し:
 *   - PSM 7 (単一テキスト行), PSM 10 (単一文字), PSM 13 (Raw line) の3ワーカー
 *   - 適応的閾値 + Otsu二値化の2系統 × 複数前処理
 *   - 連結成分解析による数字領域の精密切り出し
 *   - 信頼度を考慮した衝突解決
 */

const DigitRecognizer = (() => {
  let _worker7  = null; // PSM 7: single text line
  let _worker10 = null; // PSM 10: single char
  let _worker13 = null; // PSM 13: raw line
  let _ready7  = false;
  let _ready10 = false;
  let _ready13 = false;

  async function _initWorker(psm) {
    const worker = await Tesseract.createWorker('eng', 1, {
      logger: () => {}
    });
    await worker.setParameters({
      tessedit_char_whitelist: '123456789',
      tessedit_pageseg_mode: String(psm),
    });
    return worker;
  }

  async function _initOCR() {
    const tasks = [];
    if (!_ready7 || !_worker7) {
      tasks.push(
        _initWorker(7).then(w => { _worker7 = w; _ready7 = true; console.log('PSM 7 ready'); })
          .catch(e => { console.warn('PSM 7 init failed:', e); })
      );
    }
    if (!_ready10 || !_worker10) {
      tasks.push(
        _initWorker(10).then(w => { _worker10 = w; _ready10 = true; console.log('PSM 10 ready'); })
          .catch(e => { console.warn('PSM 10 init failed:', e); })
      );
    }
    if (!_ready13 || !_worker13) {
      tasks.push(
        _initWorker(13).then(w => { _worker13 = w; _ready13 = true; console.log('PSM 13 ready'); })
          .catch(e => { console.warn('PSM 13 init failed:', e); })
      );
    }
    await Promise.all(tasks);
  }

  async function loadModel() { await _initOCR(); }

  // ── メイン認識 ──
  async function recognize(cells, cellsGray, warpedCanvas, cellsOtsu) {
    await loadModel();
    if (!_worker7 && !_worker10 && !_worker13) {
      console.error('No OCR workers available');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const gray = cellsGray || cells;
    const binary = cells;
    const otsu = cellsOtsu || null;
    if (!gray) {
      console.error('No cell data');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    const confGrid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    let recognized = 0;
    let skipped = 0;

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cellGray = gray[r][c];
        const cellBin = binary ? binary[r][c] : null;
        const cellOtsu = otsu ? otsu[r][c] : null;
        if (!cellGray) { skipped++; continue; }
        if (_isEmpty(cellGray)) { skipped++; continue; }

        const result = await _recognizeCell(cellGray, cellBin, cellOtsu);
        if (result.digit > 0) {
          grid[r][c] = result.digit;
          confGrid[r][c] = result.conf;
          recognized++;
        }
      }
    }

    console.log('Recognized: ' + recognized + ', skipped: ' + skipped);
    _resolveConflicts(grid, confGrid);
    const final = grid.flat().filter(v => v !== 0).length;
    console.log('After conflict resolution: ' + final + ' digits');
    return grid;
  }

  // ── 空セル判定 (改善版) ──
  function _isEmpty(cellImageData) {
    const d = cellImageData.data;
    const w = cellImageData.width;
    const h = cellImageData.height;
    // 中央60%の領域で判定（マージン20%）
    const margin = Math.floor(Math.min(w, h) * 0.2);
    let sum = 0, sumSq = 0, count = 0;
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        const v = d[(y * w + x) * 4];
        sum += v; sumSq += v * v; count++;
      }
    }
    if (count === 0) return true;
    const mean = sum / count;
    const stddev = Math.sqrt(Math.max(0, sumSq / count - mean * mean));

    // 暗いピクセルの割合を計算
    let darkCount = 0;
    const threshold = mean * 0.6;
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        if (d[(y * w + x) * 4] < threshold) darkCount++;
      }
    }
    const darkRatio = darkCount / count;

    // stddev < 10 → ほぼ均一（空白）
    // darkRatio < 0.008 → 暗いピクセルがほぼ無い
    return stddev < 10 || darkRatio < 0.008;
  }

  // ── セル認識: トリプルPSM × 複数前処理 + Otsu ──
  async function _recognizeCell(cellGray, cellBin, cellOtsu) {
    // 前処理バリエーション生成
    const variants = [];
    // グレースケール系 (3x, 4x, コントラスト強調)
    variants.push(_prepareCell(cellGray, false, 3));
    variants.push(_prepareCell(cellGray, false, 4));
    variants.push(_prepareCell(cellGray, true, 3));
    variants.push(_prepareCell(cellGray, true, 4));

    // 連結成分ベースの数字切り出し
    const ccCentered = _centerDigitCC(cellGray);
    if (ccCentered) {
      variants.push(_prepareCell(ccCentered, false, 3));
      variants.push(_prepareCell(ccCentered, true, 3));
    } else {
      // フォールバック: 従来のバウンディングボックス切り出し
      const centered = _centerDigit(cellGray);
      if (centered) {
        variants.push(_prepareCell(centered, false, 3));
        variants.push(_prepareCell(centered, true, 3));
      }
    }

    // 二値化セル (適応的閾値)
    if (cellBin) {
      variants.push(_prepareBinaryCell(cellBin, 3));
      variants.push(_prepareBinaryCell(cellBin, 4));
    }
    // Otsu 二値化セル
    if (cellOtsu) {
      variants.push(_prepareBinaryCell(cellOtsu, 3));
      variants.push(_prepareBinaryCell(cellOtsu, 4));
    }

    const votes = {};
    const confs = {};
    const maxConfs = {};

    // PSM 10 で全バリエーション認識
    if (_worker10 && _ready10) {
      for (const canvas of variants) {
        const r = await _ocrOne(_worker10, canvas);
        if (r) {
          votes[r.digit] = (votes[r.digit] || 0) + 1;
          confs[r.digit] = (confs[r.digit] || 0) + r.conf;
          if (!maxConfs[r.digit] || r.conf > maxConfs[r.digit]) maxConfs[r.digit] = r.conf;
        }
      }
    }

    // PSM 7 で主要バリエーション認識
    if (_worker7 && _ready7) {
      const psm7variants = [
        _prepareCell(cellGray, false, 3),
        _prepareCell(cellGray, true, 3),
        _prepareCell(cellGray, false, 4),
      ];
      if (ccCentered) {
        psm7variants.push(_prepareCell(ccCentered, false, 3));
      }
      if (cellOtsu) {
        psm7variants.push(_prepareBinaryCell(cellOtsu, 3));
      }
      for (const canvas of psm7variants) {
        const r = await _ocrOne(_worker7, canvas);
        if (r) {
          votes[r.digit] = (votes[r.digit] || 0) + 1;
          confs[r.digit] = (confs[r.digit] || 0) + r.conf;
          if (!maxConfs[r.digit] || r.conf > maxConfs[r.digit]) maxConfs[r.digit] = r.conf;
        }
      }
    }

    // PSM 13 で主要バリエーション認識
    if (_worker13 && _ready13) {
      const psm13variants = [
        _prepareCell(cellGray, false, 3),
        _prepareCell(cellGray, false, 4),
        _prepareCell(cellGray, true, 3),
      ];
      if (ccCentered) {
        psm13variants.push(_prepareCell(ccCentered, false, 3));
      }
      for (const canvas of psm13variants) {
        const r = await _ocrOne(_worker13, canvas);
        if (r) {
          votes[r.digit] = (votes[r.digit] || 0) + 1;
          confs[r.digit] = (confs[r.digit] || 0) + r.conf;
          if (!maxConfs[r.digit] || r.conf > maxConfs[r.digit]) maxConfs[r.digit] = r.conf;
        }
      }
    }

    const digits = Object.keys(votes).map(Number);
    if (digits.length === 0) return { digit: 0, conf: 0 };

    // スコアリング: 投票数 × 平均信頼度 でランク付け
    digits.sort((a, b) => {
      const vd = votes[b] - votes[a];
      if (vd !== 0) return vd;
      // 同票なら平均信頼度で比較
      const avgA = confs[a] / votes[a];
      const avgB = confs[b] / votes[b];
      return avgB - avgA;
    });

    let best = digits[0];
    let bestVotes = votes[best];
    let bestMaxConf = maxConfs[best] || 0;

    // ── 混同ペア構造解析 (3/8, 5/9) ──
    // 両候補に票がある場合、閉領域(穴)の数で構造的に判別
    if (digits.length >= 2) {
      const target = ccCentered || cellGray;

      // 3 vs 8: 8は2穴, 3は0穴
      if ((best === 3 || best === 8) && votes[3] && votes[8]) {
        const holes = _countHoles(target);
        const structural = holes >= 2 ? 8 : 3;
        if (structural !== best) {
          console.log(`Structure 3/8: ${best}→${structural} (holes=${holes})`);
          best = structural;
          bestVotes = votes[best] || bestVotes;
          bestMaxConf = maxConfs[best] || bestMaxConf;
        }
      }

      // 5 vs 9: 9は上部にループ(穴あり), 5はオープン
      if ((best === 5 || best === 9) && votes[5] && votes[9]) {
        const topHoles = _countHoles(target, 0, 0.55);
        const structural = topHoles >= 1 ? 9 : 5;
        if (structural !== best) {
          console.log(`Structure 5/9: ${best}→${structural} (topHoles=${topHoles})`);
          best = structural;
          bestVotes = votes[best] || bestVotes;
          bestMaxConf = maxConfs[best] || bestMaxConf;
        }
      }
    }

    // 2票以上、または1票でも最大信頼度60以上
    if (bestVotes >= 2 || bestMaxConf >= 60) {
      return { digit: best, conf: bestMaxConf };
    }
    return { digit: 0, conf: 0 };
  }

  // ── 個別OCR実行 ──
  async function _ocrOne(worker, canvas) {
    try {
      const result = await worker.recognize(canvas);
      if (result.data && result.data.text) {
        const text = result.data.text.trim();
        if (/^[1-9]$/.test(text)) {
          return { digit: parseInt(text, 10), conf: result.data.confidence || 0 };
        }
      }
    } catch (e) { /* skip */ }
    return null;
  }

  // ── セル画像の前処理: パディング + 拡大 ──
  function _prepareCell(cellImageData, enhanceContrast, scale) {
    const w = cellImageData.width;
    const h = cellImageData.height;
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
    const tmpCtx = tmp.getContext('2d');
    if (enhanceContrast) {
      const imgData = new ImageData(new Uint8ClampedArray(cellImageData.data), w, h);
      _enhanceImageData(imgData);
      tmpCtx.putImageData(imgData, 0, 0);
    } else {
      tmpCtx.putImageData(cellImageData, 0, 0);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, w, h, pad * scale, pad * scale, w * scale, h * scale);
    return canvas;
  }

  // ── 二値化済みセルの前処理 ──
  function _prepareBinaryCell(cellImageData, scale) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const d = cellImageData.data;
    const imgData = new ImageData(new Uint8ClampedArray(d.length), w, h);
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i] > 128 ? 0 : 255;
      imgData.data[i] = imgData.data[i + 1] = imgData.data[i + 2] = v;
      imgData.data[i + 3] = 255;
    }
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
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w, h, pad * scale, pad * scale, w * scale, h * scale);
    return canvas;
  }

  // ── 連結成分ベースの数字中心切り出し (改善版) ──
  function _centerDigitCC(cellImageData) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const d = cellImageData.data;

    // Otsu的な二値化: ヒストグラムから自動閾値決定
    const hist = new Array(256).fill(0);
    for (let i = 0; i < d.length; i += 4) hist[d[i]]++;
    const total = w * h;
    let sumAll = 0;
    for (let i = 0; i < 256; i++) sumAll += i * hist[i];

    let sumB = 0, wB = 0, maxVar = 0, bestThresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sumAll - sumB) / wF;
      const variance = wB * wF * (mB - mF) * (mB - mF);
      if (variance > maxVar) { maxVar = variance; bestThresh = t; }
    }

    // 二値化 → 連結成分のラベリング
    const binary = new Uint8Array(w * h);
    const margin = Math.floor(Math.min(w, h) * 0.08);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // マージン内はスキップ（グリッド線除去）
        if (y < margin || y >= h - margin || x < margin || x >= w - margin) continue;
        binary[y * w + x] = d[(y * w + x) * 4] < bestThresh ? 1 : 0;
      }
    }

    // 連結成分検出（4近傍）
    const labels = new Int32Array(w * h);
    let nextLabel = 1;
    const components = []; // [{minX, maxX, minY, maxY, area}]

    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        if (binary[y * w + x] === 1 && labels[y * w + x] === 0) {
          // BFS
          const comp = { minX: x, maxX: x, minY: y, maxY: y, area: 0 };
          const queue = [[x, y]];
          labels[y * w + x] = nextLabel;
          while (queue.length > 0) {
            const [cx, cy] = queue.shift();
            comp.area++;
            if (cx < comp.minX) comp.minX = cx;
            if (cx > comp.maxX) comp.maxX = cx;
            if (cy < comp.minY) comp.minY = cy;
            if (cy > comp.maxY) comp.maxY = cy;
            const neighbors = [[cx-1,cy],[cx+1,cy],[cx,cy-1],[cx,cy+1]];
            for (const [nx, ny] of neighbors) {
              if (nx >= margin && nx < w - margin && ny >= margin && ny < h - margin
                  && binary[ny * w + nx] === 1 && labels[ny * w + nx] === 0) {
                labels[ny * w + nx] = nextLabel;
                queue.push([nx, ny]);
              }
            }
          }
          components.push(comp);
          nextLabel++;
        }
      }
    }

    if (components.length === 0) return null;

    // ノイズ除去: 小さすぎる成分を除外（セル面積の2%未満）
    const minArea = w * h * 0.02;
    const filtered = components.filter(c => c.area >= minArea);
    if (filtered.length === 0) return null;

    // 最大の連結成分を数字として選択（中央に近いものを優先）
    const centerX = w / 2;
    const centerY = h / 2;
    filtered.sort((a, b) => {
      // まず面積でソート（大きい順）
      const areaDiff = b.area - a.area;
      if (Math.abs(areaDiff) > minArea) return areaDiff;
      // 同程度なら中央に近い方を優先
      const distA = Math.abs((a.minX + a.maxX) / 2 - centerX) + Math.abs((a.minY + a.maxY) / 2 - centerY);
      const distB = Math.abs((b.minX + b.maxX) / 2 - centerX) + Math.abs((b.minY + b.maxY) / 2 - centerY);
      return distA - distB;
    });

    // 近接した成分をマージ（数字が分離している場合: 例えば '4' の上部と下部）
    let bbox = { ...filtered[0] };
    for (let i = 1; i < filtered.length; i++) {
      const c = filtered[i];
      const gap = Math.min(
        Math.abs(c.minX - bbox.maxX), Math.abs(bbox.minX - c.maxX),
        Math.abs(c.minY - bbox.maxY), Math.abs(bbox.minY - c.maxY)
      );
      // セルサイズの 20% 以内ならマージ
      if (gap < Math.min(w, h) * 0.2) {
        bbox.minX = Math.min(bbox.minX, c.minX);
        bbox.maxX = Math.max(bbox.maxX, c.maxX);
        bbox.minY = Math.min(bbox.minY, c.minY);
        bbox.maxY = Math.max(bbox.maxY, c.maxY);
      }
    }

    const dw = bbox.maxX - bbox.minX + 1;
    const dh = bbox.maxY - bbox.minY + 1;
    if (dw < 4 || dh < 4) return null;

    // 正方形に切り出し、パディング付き
    const side = Math.max(dw, dh);
    const outSize = Math.round(side * 1.4);
    const ox = Math.round((outSize - dw) / 2);
    const oy = Math.round((outSize - dh) / 2);
    const out = new ImageData(outSize, outSize);
    for (let i = 0; i < out.data.length; i += 4) {
      out.data[i] = out.data[i + 1] = out.data[i + 2] = 255;
      out.data[i + 3] = 255;
    }
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const srcIdx = ((bbox.minY + y) * w + (bbox.minX + x)) * 4;
        const dstIdx = ((oy + y) * outSize + (ox + x)) * 4;
        out.data[dstIdx] = out.data[dstIdx + 1] = out.data[dstIdx + 2] = d[srcIdx];
        out.data[dstIdx + 3] = 255;
      }
    }
    return out;
  }

  // ── 数字の中心切り出し (従来版 フォールバック) ──
  function _centerDigit(cellImageData) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const d = cellImageData.data;
    let sum = 0, count = 0;
    for (let i = 0; i < d.length; i += 4) { sum += d[i]; count++; }
    const mean = sum / count;
    const thresh = mean * 0.7;
    let minX = w, maxX = 0, minY = h, maxY = 0;
    const margin = Math.floor(Math.min(w, h) * 0.1);
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        if (d[(y * w + x) * 4] < thresh) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (minX >= maxX || minY >= maxY) return null;
    const dw = maxX - minX + 1;
    const dh = maxY - minY + 1;
    if (dw < 4 || dh < 4) return null;
    const side = Math.max(dw, dh);
    const outSize = Math.round(side * 1.4);
    const ox = Math.round((outSize - dw) / 2);
    const oy = Math.round((outSize - dh) / 2);
    const out = new ImageData(outSize, outSize);
    for (let i = 0; i < out.data.length; i += 4) {
      out.data[i] = out.data[i + 1] = out.data[i + 2] = 255;
      out.data[i + 3] = 255;
    }
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const srcIdx = ((minY + y) * w + (minX + x)) * 4;
        const dstIdx = ((oy + y) * outSize + (ox + x)) * 4;
        out.data[dstIdx] = out.data[dstIdx + 1] = out.data[dstIdx + 2] = d[srcIdx];
        out.data[dstIdx + 3] = 255;
      }
    }
    return out;
  }

  // ── コントラスト強調 ──
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

  // ── 構造解析: 閉領域 (穴) の数を数える ──
  // 8 → 2穴, 0/6/9/4 → 1穴, 1/2/3/5/7 → 0穴
  // startYRatio/endYRatio で垂直方向の解析範囲を指定可能
  function _countHoles(imageData, startYRatio, endYRatio) {
    if (!startYRatio) startYRatio = 0;
    if (!endYRatio) endYRatio = 1;
    const w = imageData.width;
    const h = imageData.height;
    const d = imageData.data;
    const yStart = Math.floor(h * startYRatio);
    const yEnd = Math.floor(h * endYRatio);
    const rh = yEnd - yStart;
    if (rh < 4 || w < 4) return 0;

    // Otsu二値化
    const hist = new Array(256).fill(0);
    for (let y = yStart; y < yEnd; y++) {
      for (let x = 0; x < w; x++) hist[d[(y * w + x) * 4]]++;
    }
    const total = w * rh;
    let sumAll = 0;
    for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let sumB = 0, wB = 0, maxVar = 0, thresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sumAll - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > maxVar) { maxVar = v; thresh = t; }
    }

    // 1 = 前景(暗), 0 = 背景(明)
    const bin = new Uint8Array(w * rh);
    for (let y = 0; y < rh; y++) {
      for (let x = 0; x < w; x++) {
        bin[y * w + x] = d[((yStart + y) * w + x) * 4] < thresh ? 1 : 0;
      }
    }

    // 外周背景をフラッドフィルで除去
    const visited = new Uint8Array(w * rh);
    const queue = [];
    for (let x = 0; x < w; x++) {
      if (!bin[x] && !visited[x]) { visited[x] = 1; queue.push(x); }
      const idx = (rh - 1) * w + x;
      if (!bin[idx] && !visited[idx]) { visited[idx] = 1; queue.push(idx); }
    }
    for (let y = 0; y < rh; y++) {
      const l = y * w;
      if (!bin[l] && !visited[l]) { visited[l] = 1; queue.push(l); }
      const r = y * w + w - 1;
      if (!bin[r] && !visited[r]) { visited[r] = 1; queue.push(r); }
    }
    while (queue.length > 0) {
      const idx = queue.shift();
      const x = idx % w, y = Math.floor(idx / w);
      for (const [dx, dy] of [[0,-1],[0,1],[-1,0],[1,0]]) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < rh) {
          const ni = ny * w + nx;
          if (!bin[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); }
        }
      }
    }

    // 残った未訪問の背景ピクセル群 = 穴
    // ノイズ除去: 極小領域 (全体の0.5%未満) は無視
    const minHoleArea = Math.max(3, Math.floor(total * 0.005));
    let holes = 0;
    for (let i = 0; i < w * rh; i++) {
      if (!bin[i] && !visited[i]) {
        let area = 0;
        const hq = [i];
        visited[i] = 1;
        while (hq.length > 0) {
          const hi = hq.shift();
          area++;
          const hx = hi % w, hy = Math.floor(hi / w);
          for (const [dx, dy] of [[0,-1],[0,1],[-1,0],[1,0]]) {
            const nx = hx + dx, ny = hy + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < rh) {
              const ni = ny * w + nx;
              if (!bin[ni] && !visited[ni]) { visited[ni] = 1; hq.push(ni); }
            }
          }
        }
        if (area >= minHoleArea) holes++;
      }
    }
    return holes;
  }

  // ── 信頼度を考慮した数独制約チェック ──
  function _resolveConflicts(grid, confGrid) {
    let changed = true;
    while (changed) {
      changed = false;

      // 行チェック
      for (let row = 0; row < 9; row++) {
        const seen = {}; // digit → {col, conf}
        for (let col = 0; col < 9; col++) {
          const v = grid[row][col];
          if (v === 0) continue;
          if (seen[v] !== undefined) {
            // 信頼度が低い方を削除
            const prevCol = seen[v].col;
            const prevConf = seen[v].conf;
            const curConf = confGrid[row][col];
            if (curConf < prevConf) {
              grid[row][col] = 0; confGrid[row][col] = 0;
            } else {
              grid[row][prevCol] = 0; confGrid[row][prevCol] = 0;
              seen[v] = { col, conf: curConf };
            }
            changed = true;
          } else {
            seen[v] = { col, conf: confGrid[row][col] };
          }
        }
      }

      // 列チェック
      for (let col = 0; col < 9; col++) {
        const seen = {};
        for (let row = 0; row < 9; row++) {
          const v = grid[row][col];
          if (v === 0) continue;
          if (seen[v] !== undefined) {
            const prevRow = seen[v].row;
            const prevConf = seen[v].conf;
            const curConf = confGrid[row][col];
            if (curConf < prevConf) {
              grid[row][col] = 0; confGrid[row][col] = 0;
            } else {
              grid[prevRow][col] = 0; confGrid[prevRow][col] = 0;
              seen[v] = { row, conf: curConf };
            }
            changed = true;
          } else {
            seen[v] = { row, conf: confGrid[row][col] };
          }
        }
      }

      // ボックスチェック
      for (let br = 0; br < 9; br += 3) {
        for (let bc = 0; bc < 9; bc += 3) {
          const seen = {};
          for (let r = br; r < br + 3; r++) {
            for (let c = bc; c < bc + 3; c++) {
              const v = grid[r][c];
              if (v === 0) continue;
              if (seen[v]) {
                const pr = seen[v].r;
                const pc = seen[v].c;
                const prevConf = seen[v].conf;
                const curConf = confGrid[r][c];
                if (curConf < prevConf) {
                  grid[r][c] = 0; confGrid[r][c] = 0;
                } else {
                  grid[pr][pc] = 0; confGrid[pr][pc] = 0;
                  seen[v] = { r, c, conf: curConf };
                }
                changed = true;
              } else {
                seen[v] = { r, c, conf: confGrid[r][c] };
              }
            }
          }
        }
      }
    }
  }

  async function terminate() {
    if (_worker7)  { await _worker7.terminate();  _worker7  = null; _ready7  = false; }
    if (_worker10) { await _worker10.terminate(); _worker10 = null; _ready10 = false; }
    if (_worker13) { await _worker13.terminate(); _worker13 = null; _ready13 = false; }
  }

  return { loadModel, recognize, terminate };
})();
