/**
 * digitRecognizer.js  v1.6.0
 *
 * デュアルPSMアプローチ + 高精度前処理:
 *   - PSM 10 (単一文字) と PSM 13 (Raw line) の2つのOCRワーカーで独立認識
 *   - 異なるセグメンテーションエンジンの結果を統合して精度向上
 *   - 各セルで複数前処理バリエーション × 2 PSMモード = 多数決
 *
 * v1.5.0 精度向上:
 *   - Otsu's 二値化バリアント追加 (Otsu 1979, IEEE Trans. SMC 9(1))
 *   - アンシャープマスクによる輪郭強調バリアント追加
 *   - 連結成分解析を用いた空セル判定強化でフォールスポジティブ削減
 *   - 信頼度スコアに基づく競合解決で誤認識数字の優先選択精度向上
 */

const DigitRecognizer = (() => {
  let _worker10 = null; // PSM 10: single char
  let _worker13 = null; // PSM 13: raw line
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
  async function recognize(cells, cellsGray, warpedCanvas) {
    await loadModel();
    if (!_worker10 && !_worker13) {
      console.error('No OCR workers available');
      return Array.from({ length: 9 }, () => new Array(9).fill(0));
    }

    const gray = cellsGray || cells;
    const binary = cells;
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
        if (!cellGray) { skipped++; continue; }
        if (_isEmpty(cellGray)) { skipped++; continue; }

        const result = await _recognizeCell(cellGray, cellBin);
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

  // ── 空セル判定 (連結成分解析強化版) ──
  function _isEmpty(cellImageData) {
    const d = cellImageData.data;
    const w = cellImageData.width;
    const h = cellImageData.height;
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
    let darkCount = 0;
    const threshold = mean * 0.6;
    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        if (d[(y * w + x) * 4] < threshold) darkCount++;
      }
    }
    const darkRatio = darkCount / count;

    // 均一画像 or 暗ピクセルが極少 → 空
    if (stddev < 12 || darkRatio < 0.01) return true;

    // 暗ピクセルが多い場合は明らかに数字あり
    if (darkRatio > 0.4) return false;

    // 連結成分解析: 最大の暗ピクセル塊が数字として十分か検証
    // (散在したノイズ点がある空セルとの判別)
    const binary = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        binary[y * w + x] = d[(y * w + x) * 4] < threshold ? 1 : 0;
      }
    }
    const maxComp = _largestComponentSize(binary, w, h);
    // 最大連結成分がセル面積の2%未満なら空とみなす
    return maxComp < w * h * 0.02;
  }

  // ── 最大連結成分の画素数を返す (4近傍 BFS) ──
  function _largestComponentSize(binary, w, h) {
    const visited = new Uint8Array(w * h);
    let maxSize = 0;
    const queue = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!binary[idx] || visited[idx]) continue;
        // BFS
        let size = 0;
        queue.length = 0;
        queue.push(idx);
        visited[idx] = 1;
        let head = 0;
        while (head < queue.length) {
          const cur = queue[head++];
          size++;
          const cy = Math.floor(cur / w);
          const cx = cur % w;
          if (cy > 0)     { const ni = cur - w; if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); } }
          if (cy < h - 1) { const ni = cur + w; if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); } }
          if (cx > 0)     { const ni = cur - 1; if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); } }
          if (cx < w - 1) { const ni = cur + 1; if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); } }
        }
        if (size > maxSize) maxSize = size;
      }
    }
    return maxSize;
  }

  // ── セル認識: デュアルPSM × 複数前処理 ──
  async function _recognizeCell(cellGray, cellBin) {
    // 前処理バリエーション生成
    const variants = [];
    // グレースケール系
    variants.push(_prepareCell(cellGray, false, 3));
    variants.push(_prepareCell(cellGray, false, 4));
    variants.push(_prepareCell(cellGray, true, 3));
    variants.push(_prepareCell(cellGray, true, 4));
    // 数字中心切り出し
    const centered = _centerDigit(cellGray);
    if (centered) {
      variants.push(_prepareCell(centered, false, 3));
      variants.push(_prepareCell(centered, true, 3));
    }
    // 二値化セル
    if (cellBin) {
      variants.push(_prepareBinaryCell(cellBin, 3));
      variants.push(_prepareBinaryCell(cellBin, 4));
    }
    // Otsu 二値化バリアント (Otsu 1979 — 文書画像の大域的二値化に最適)
    variants.push(_prepareWithOtsu(cellGray, 3));
    variants.push(_prepareWithOtsu(cellGray, 4));
    // アンシャープマスク + グレースケール/Otsu バリアント (輪郭強調で認識率向上)
    const sharpened = _sharpenedCopy(cellGray);
    variants.push(_prepareCell(sharpened, false, 3));
    variants.push(_prepareWithOtsu(sharpened, 3));

    const votes = {};
    const confs = {};

    // PSM 10 で全バリエーション認識
    if (_worker10 && _ready10) {
      for (const canvas of variants) {
        const r = await _ocrOne(_worker10, canvas);
        if (r) { votes[r.digit] = (votes[r.digit] || 0) + 1; if (!confs[r.digit] || r.conf > confs[r.digit]) confs[r.digit] = r.conf; }
      }
    }

    // PSM 13 で主要バリエーション認識 (全部はやらず速度考慮)
    if (_worker13 && _ready13) {
      // グレー3x, グレー4x, コントラスト3x, centered3x, Otsu3x → 最大5つ
      const psm13variants = [
        _prepareCell(cellGray, false, 3),
        _prepareCell(cellGray, false, 4),
        _prepareCell(cellGray, true, 3),
        _prepareWithOtsu(cellGray, 3),
      ];
      if (centered) {
        psm13variants.push(_prepareCell(centered, false, 3));
      }
      for (const canvas of psm13variants) {
        const r = await _ocrOne(_worker13, canvas);
        if (r) { votes[r.digit] = (votes[r.digit] || 0) + 1; if (!confs[r.digit] || r.conf > confs[r.digit]) confs[r.digit] = r.conf; }
      }
    }

    const digits = Object.keys(votes).map(Number);
    if (digits.length === 0) return { digit: 0, conf: 0 };

    digits.sort((a, b) => {
      const vd = votes[b] - votes[a];
      if (vd !== 0) return vd;
      return (confs[b] || 0) - (confs[a] || 0);
    });

    const best = digits[0];
    const bestVotes = votes[best];
    const bestConf = confs[best] || 0;

    // 2票以上、または1票でも信頼度65以上
    if (bestVotes >= 2 || bestConf >= 65) {
      return { digit: best, conf: bestConf };
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

  // ── Otsu's 大域的二値化しきい値計算 (Otsu 1979) ──
  // クラス間分散を最大化するしきい値を求める古典的手法
  function _computeOtsuThreshold(d, w, h) {
    const hist = new Int32Array(256);
    for (let i = 0; i < d.length; i += 4) hist[d[i]]++;
    const total = w * h;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, maxVar = 0, threshold = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = (wB / total) * (wF / total) * (mB - mF) * (mB - mF);
      if (varBetween > maxVar) { maxVar = varBetween; threshold = t; }
    }
    return threshold;
  }

  // ── Otsu 二値化前処理: 白地に黒文字の二値画像を生成 ──
  function _prepareWithOtsu(cellImageData, scale) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const d = cellImageData.data;
    const thresh = _computeOtsuThreshold(d, w, h);
    const bin = new ImageData(new Uint8ClampedArray(d.length), w, h);
    for (let i = 0; i < d.length; i += 4) {
      // しきい値以下 (暗ピクセル = 数字ストローク) → 黒(0), 背景 → 白(255)
      const v = d[i] <= thresh ? 0 : 255;
      bin.data[i] = bin.data[i + 1] = bin.data[i + 2] = v;
      bin.data[i + 3] = 255;
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
    tmp.getContext('2d').putImageData(bin, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w, h, pad * scale, pad * scale, w * scale, h * scale);
    return canvas;
  }

  // ── アンシャープマスク: ラプラシアンカーネルで輪郭強調 ──
  // 参考: Gonzalez & Woods "Digital Image Processing" §3.6
  // カーネル: [0,-1,0; -1,5,-1; 0,-1,0]
  // 中心係数 5 = 1 (元画像) + 4 (ラプラシアン強調量), 隣接係数 -1 は差分に相当
  function _applyUnsharpMask(imgData) {
    const w = imgData.width;
    const h = imgData.height;
    const d = imgData.data;
    const src = new Uint8ClampedArray(d);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        const v = Math.max(0, Math.min(255,
          5 * src[i]
          - src[((y - 1) * w + x) * 4]
          - src[((y + 1) * w + x) * 4]
          - src[(y * w + x - 1) * 4]
          - src[(y * w + x + 1) * 4]
        ));
        d[i] = d[i + 1] = d[i + 2] = v;
      }
    }
  }

  // ── アンシャープマスクを適用したコピーを返す ──
  function _sharpenedCopy(cellImageData) {
    const w = cellImageData.width;
    const h = cellImageData.height;
    const copy = new ImageData(new Uint8ClampedArray(cellImageData.data), w, h);
    _applyUnsharpMask(copy);
    return copy;
  }

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

  // ── 数独制約チェック: 信頼度スコアを用いて低信頼の競合数字を除去 ──
  function _resolveConflicts(grid, confGrid) {
    // 指定セルを 0 にし、confGrid が渡されていれば信頼度もクリア
    function clearCell(r, c) {
      grid[r][c] = 0;
      if (confGrid) confGrid[r][c] = 0;
    }

    let changed = true;
    while (changed) {
      changed = false;
      // 行チェック
      for (let row = 0; row < 9; row++) {
        const seen = {};
        for (let col = 0; col < 9; col++) {
          const v = grid[row][col];
          if (v === 0) continue;
          const conf = confGrid ? confGrid[row][col] : 0;
          if (seen[v] !== undefined) {
            // 信頼度が低い方を削除 (同値なら後者を削除)
            const prev = seen[v];
            if (conf > prev.conf) {
              clearCell(row, prev.col);
              seen[v] = { col, conf };
            } else {
              clearCell(row, col);
            }
            changed = true;
          } else {
            seen[v] = { col, conf };
          }
        }
      }
      // 列チェック
      for (let col = 0; col < 9; col++) {
        const seen = {};
        for (let row = 0; row < 9; row++) {
          const v = grid[row][col];
          if (v === 0) continue;
          const conf = confGrid ? confGrid[row][col] : 0;
          if (seen[v] !== undefined) {
            const prev = seen[v];
            if (conf > prev.conf) {
              clearCell(prev.row, col);
              seen[v] = { row, conf };
            } else {
              clearCell(row, col);
            }
            changed = true;
          } else {
            seen[v] = { row, conf };
          }
        }
      }
      // 3×3 ブロックチェック
      for (let br = 0; br < 9; br += 3) {
        for (let bc = 0; bc < 9; bc += 3) {
          const seen = {};
          for (let r = br; r < br + 3; r++) {
            for (let c = bc; c < bc + 3; c++) {
              const v = grid[r][c];
              if (v === 0) continue;
              const conf = confGrid ? confGrid[r][c] : 0;
              if (seen[v] !== undefined) {
                const prev = seen[v];
                if (conf > prev.conf) {
                  clearCell(prev.r, prev.c);
                  seen[v] = { r, c, conf };
                } else {
                  clearCell(r, c);
                }
                changed = true;
              } else {
                seen[v] = { r, c, conf };
              }
            }
          }
        }
      }
    }
  }

  async function terminate() {
    if (_worker10) { await _worker10.terminate(); _worker10 = null; _ready10 = false; }
    if (_worker13) { await _worker13.terminate(); _worker13 = null; _ready13 = false; }
  }

  return { loadModel, recognize, terminate };
})();
