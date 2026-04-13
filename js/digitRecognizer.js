/**
 * digitRecognizer.js
 * Tesseract.js (OCR) をメイン認識エンジンとして使用し、
 * TensorFlow.js (MNIST CNN) をフォールバックとして統合する。
 *
 * 認識戦略 (精度向上版):
 *   1. ピクセル輝度 + 連結成分分析で空白判定
 *   2. OCR 用に複数の前処理バリエーションを生成
 *   3. Tesseract.js OCR で印刷数字を認識（メイン）
 *   4. MNIST CNN で推論（サブ）
 *   5. クロス検証: 両方一致 → 高信頼 / 不一致 → 高信頼の方を採用
 *   6. 数独制約チェック: 行・列・ブロック内の重複を検出→候補を調整
 */

const DigitRecognizer = (() => {
  let _model = null;
  let _modelLoading = false;
  let _ocrWorker = null;
  let _ocrReady = false;

  // MNIST モデルの URL
  const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

  // 空白判定閾値 (白ピクセル率 — BINARY_INV後、数字部分が白)
  const EMPTY_THRESHOLD = 0.03;

  // 信頼度閾値
  const OCR_HIGH_CONFIDENCE  = 70;   // OCR 高信頼 (0-100)
  const OCR_LOW_CONFIDENCE   = 30;   // OCR 最低限の信頼度
  const CNN_HIGH_CONFIDENCE  = 0.7;  // CNN 高信頼 (0-1)
  const CNN_LOW_CONFIDENCE   = 0.1;  // CNN 最低限の信頼度

  // ─────────────────────────────────────────────
  // Tesseract.js OCR ワーカー初期化
  // ─────────────────────────────────────────────
  async function _initOCR() {
    if (_ocrReady && _ocrWorker) return _ocrWorker;

    try {
      _ocrWorker = await Tesseract.createWorker('eng', 1, {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            console.log(`OCR 認識中: ${Math.round(m.progress * 100)}%`);
          }
        }
      });
      // 数字のみ認識するよう制限
      await _ocrWorker.setParameters({
        tessedit_char_whitelist: '123456789',
        tessedit_pageseg_mode: '10', // PSM_SINGLE_CHAR: 単一文字認識
      });
      _ocrReady = true;
      console.log('Tesseract.js OCR ワーカー初期化完了');
      return _ocrWorker;
    } catch (err) {
      console.warn('Tesseract.js 初期化失敗:', err);
      _ocrWorker = null;
      _ocrReady = false;
      return null;
    }
  }

  // ─────────────────────────────────────────────
  // MNIST モデルのロード (非同期)
  // ─────────────────────────────────────────────
  async function loadModel() {
    // OCR ワーカーとモデルを並列でロード
    const ocrPromise = _initOCR();

    if (!_model && !_modelLoading) {
      _modelLoading = true;
      try {
        _model = await tf.loadLayersModel('indexeddb://sudoku-mnist-model');
        console.log('MNIST モデルをキャッシュからロードしました');
      } catch (_cacheErr) {
        try {
          _model = await tf.loadLayersModel(MODEL_URL);
          await _model.save('indexeddb://sudoku-mnist-model');
          console.log('MNIST モデルをリモートからロードし、キャッシュしました');
        } catch (err) {
          console.warn('MNIST モデルのロード失敗:', err);
          _model = null;
        }
      } finally {
        _modelLoading = false;
      }
    }

    await ocrPromise;
    return _model;
  }

  // ─────────────────────────────────────────────
  // 9×9 セル ImageData[][] → 数字配列 number[][]
  // クロス検証 + 数独制約チェック付き
  // ─────────────────────────────────────────────
  async function recognize(cells) {
    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    const confidenceMap = Array.from({ length: 9 }, () =>
      new Array(9).fill(null)
    );

    await loadModel();

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        const imageData = cells[row][col];
        if (_isEmpty(imageData)) {
          grid[row][col] = 0;
          confidenceMap[row][col] = { digit: 0, conf: 1.0, source: 'empty' };
          continue;
        }

        // 両エンジンで認識してクロス検証
        const result = await _crossValidate(imageData);
        grid[row][col] = result.digit;
        confidenceMap[row][col] = result;
      }
    }

    // 数独制約チェック: 重複検出→低信頼セルを空白に戻す
    _resolveConflicts(grid, confidenceMap);

    return grid;
  }

  // ─────────────────────────────────────────────
  // クロス検証: OCR と CNN の両方で認識して最適な結果を返す
  // ─────────────────────────────────────────────
  async function _crossValidate(imageData) {
    // OCR 認識（複数バリエーション試行）
    const ocrResult = await _predictWithOCRMulti(imageData);
    // CNN 認識
    const cnnResult = _model
      ? await _predictWithModel(imageData)
      : { digit: 0, confidence: 0 };

    // ケース1: 両方一致 → 高信頼
    if (ocrResult.digit !== 0 && ocrResult.digit === cnnResult.digit) {
      return {
        digit: ocrResult.digit,
        conf: Math.max(ocrResult.confidence / 100, cnnResult.confidence),
        source: 'both'
      };
    }

    // ケース2: OCR 高信頼
    if (ocrResult.digit !== 0 && ocrResult.confidence >= OCR_HIGH_CONFIDENCE) {
      return {
        digit: ocrResult.digit,
        conf: ocrResult.confidence / 100,
        source: 'ocr'
      };
    }

    // ケース3: CNN 高信頼
    if (cnnResult.digit !== 0 && cnnResult.confidence >= CNN_HIGH_CONFIDENCE) {
      return {
        digit: cnnResult.digit,
        conf: cnnResult.confidence,
        source: 'cnn'
      };
    }

    // ケース4: 両方低信頼 → 信頼度が高い方を採用
    if (ocrResult.digit !== 0 && cnnResult.digit !== 0) {
      const ocrNorm = ocrResult.confidence / 100;
      if (ocrNorm >= cnnResult.confidence) {
        return { digit: ocrResult.digit, conf: ocrNorm, source: 'ocr-low' };
      } else {
        return { digit: cnnResult.digit, conf: cnnResult.confidence, source: 'cnn-low' };
      }
    }

    // ケース5: 片方のみ結果あり
    if (ocrResult.digit !== 0 && ocrResult.confidence >= OCR_LOW_CONFIDENCE) {
      return { digit: ocrResult.digit, conf: ocrResult.confidence / 100, source: 'ocr-only' };
    }
    if (cnnResult.digit !== 0 && cnnResult.confidence >= CNN_LOW_CONFIDENCE) {
      return { digit: cnnResult.digit, conf: cnnResult.confidence, source: 'cnn-only' };
    }

    // ケース6: 認識不能
    return { digit: 0, conf: 0, source: 'none' };
  }

  // ─────────────────────────────────────────────
  // 空白判定: 白ピクセル率 + バウンディングボックスチェック
  // ─────────────────────────────────────────────
  function _isEmpty(imageData) {
    const data = imageData.data;
    const w = imageData.width;
    const h = imageData.height;
    const margin = Math.floor(Math.min(w, h) * 0.15);
    let white = 0;
    let innerTotal = 0;

    for (let y = margin; y < h - margin; y++) {
      for (let x = margin; x < w - margin; x++) {
        const idx = (y * w + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness > 128) white++;
        innerTotal++;
      }
    }

    if (innerTotal === 0) return true;
    const whiteRatio = white / innerTotal;

    if (whiteRatio < EMPTY_THRESHOLD) return true;

    // 白ピクセルが少量でも小さな点の場合はノイズ → 空白判定
    if (whiteRatio < 0.08) {
      let minX = w, maxX = 0, minY = h, maxY = 0;
      for (let y = margin; y < h - margin; y++) {
        for (let x = margin; x < w - margin; x++) {
          const idx = (y * w + x) * 4;
          const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          if (brightness > 128) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }
      const bbW = maxX - minX;
      const bbH = maxY - minY;
      if (bbW < (w * 0.2) || bbH < (h * 0.2)) return true;
    }

    return false;
  }

  // ─────────────────────────────────────────────
  // 複数バリエーション OCR: 異なる前処理で試行して最良結果を返す
  // ─────────────────────────────────────────────
  async function _predictWithOCRMulti(imageData) {
    if (!_ocrWorker || !_ocrReady) {
      return { digit: 0, confidence: 0 };
    }

    // バリエーション1: 標準 (3x拡大 + 二値反転)
    const enhanced1 = _enhanceForOCR(imageData, 3, 128);
    const result1 = await _predictWithOCR(enhanced1);

    // 高信頼ならそのまま返す
    if (result1.digit !== 0 && result1.confidence >= OCR_HIGH_CONFIDENCE) {
      return result1;
    }

    // バリエーション2: 高倍率 (4x拡大 + パディング追加)
    const enhanced2 = _enhanceForOCR(imageData, 4, 128);
    const result2 = await _predictWithOCR(enhanced2);

    // バリエーション3: 閾値変更 (明るめの閾値)
    const enhanced3 = _enhanceForOCR(imageData, 3, 100);
    const result3 = await _predictWithOCR(enhanced3);

    // 多数決 + 信頼度で最良を決定
    const results = [result1, result2, result3].filter(r => r.digit !== 0);
    if (results.length === 0) return { digit: 0, confidence: 0 };

    // 同じ数字が2つ以上 → そちらを採用
    const votes = {};
    for (const r of results) {
      votes[r.digit] = (votes[r.digit] || 0) + 1;
    }
    const maxVotes = Math.max(...Object.values(votes));
    const majorityDigits = Object.entries(votes)
      .filter(([_, v]) => v === maxVotes)
      .map(([d]) => parseInt(d, 10));

    const best = results
      .filter(r => majorityDigits.includes(r.digit))
      .sort((a, b) => b.confidence - a.confidence)[0];

    return best || results.sort((a, b) => b.confidence - a.confidence)[0];
  }

  // ─────────────────────────────────────────────
  // OCR 用前処理: 白背景・黒文字に変換 + パディング + 膨張
  // @param scale 拡大倍率
  // @param threshold 二値化閾値
  // ─────────────────────────────────────────────
  function _enhanceForOCR(imageData, scale = 3, threshold = 128) {
    const { width, height } = imageData;
    const pad = Math.round(width * scale * 0.15);
    const outW = width * scale + pad * 2;
    const outH = height * scale + pad * 2;

    const canvas = document.createElement('canvas');
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext('2d');

    // 白背景
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, outW, outH);

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = width;
    tmpCanvas.height = height;
    tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);

    ctx.drawImage(tmpCanvas, pad, pad, width * scale, height * scale);

    // 二値化 & 反転（BINARY_INV→白文字 を 黒文字に）
    const enlarged = ctx.getImageData(0, 0, outW, outH);
    const d = enlarged.data;
    for (let i = 0; i < d.length; i += 4) {
      const brightness = (d[i] + d[i + 1] + d[i + 2]) / 3;
      const val = brightness > threshold ? 0 : 255;
      d[i] = val;
      d[i + 1] = val;
      d[i + 2] = val;
      d[i + 3] = 255;
    }

    // モルフォロジー的膨張で細いストロークを太くする
    _dilateInPlace(d, outW, outH, 1);

    ctx.putImageData(enlarged, 0, 0);

    return canvas;
  }

  /**
   * ピクセルデータ上で簡易膨張処理 (黒=文字を1ピクセル太くする)
   */
  function _dilateInPlace(data, w, h, radius) {
    const copy = new Uint8Array(data.length);
    copy.set(data);

    for (let y = radius; y < h - radius; y++) {
      for (let x = radius; x < w - radius; x++) {
        const idx = (y * w + x) * 4;
        if (copy[idx] === 0) continue; // 白ピクセルはスキップ

        let hasBlack = false;
        for (let dy = -radius; dy <= radius && !hasBlack; dy++) {
          for (let dx = -radius; dx <= radius && !hasBlack; dx++) {
            const ni = ((y + dy) * w + (x + dx)) * 4;
            if (copy[ni] === 0) hasBlack = true;
          }
        }
        if (hasBlack) {
          data[idx] = 0;
          data[idx + 1] = 0;
          data[idx + 2] = 0;
        }
      }
    }
  }

  // ─────────────────────────────────────────────
  // Tesseract.js OCR で単一セルを認識
  // ─────────────────────────────────────────────
  async function _predictWithOCR(canvas) {
    if (!_ocrWorker || !_ocrReady) {
      return { digit: 0, confidence: 0 };
    }

    try {
      const result = await _ocrWorker.recognize(canvas);
      const text = result.data.text.trim();

      // 数字 (1-9) のみ抽出
      const match = text.match(/[1-9]/);
      if (!match) return { digit: 0, confidence: 0 };

      const digit = parseInt(match[0], 10);
      const confidence = result.data.confidence || 0;

      return { digit, confidence };
    } catch (err) {
      console.warn('OCR 認識エラー:', err);
      return { digit: 0, confidence: 0 };
    }
  }

  // ─────────────────────────────────────────────
  // セル画像の前処理: 数字を検出し28×28に中央配置
  // MNIST形式に合わせる
  // ─────────────────────────────────────────────
  function _centerDigit(imageData) {
    const { data, width, height } = imageData;

    let minX = width, maxX = 0, minY = height, maxY = 0;
    let hasWhite = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness > 128) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          hasWhite = true;
        }
      }
    }

    if (!hasWhite) return imageData;

    const digitW = maxX - minX + 1;
    const digitH = maxY - minY + 1;

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

    const outCanvas = document.createElement('canvas');
    outCanvas.width = 28;
    outCanvas.height = 28;
    const ctx = outCanvas.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 28, 28);

    const scale = Math.min(20 / digitW, 20 / digitH);
    const scaledW = Math.round(digitW * scale);
    const scaledH = Math.round(digitH * scale);
    const offsetX = Math.round((28 - scaledW) / 2);
    const offsetY = Math.round((28 - scaledH) / 2);

    ctx.drawImage(srcCanvas, minX, minY, digitW, digitH, offsetX, offsetY, scaledW, scaledH);

    return ctx.getImageData(0, 0, 28, 28);
  }

  // ─────────────────────────────────────────────
  // TF.js MNIST モデルで推論 (結果に信頼度を含む)
  // ─────────────────────────────────────────────
  async function _predictWithModel(imageData) {
    const centered = _centerDigit(imageData);

    const tensor = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(centered, 1);
      const normalized = imgTensor.div(255.0);
      return normalized.expandDims(0);
    });

    try {
      const prediction = _model.predict(tensor);
      const values     = await prediction.data();
      prediction.dispose();

      let maxIdx = 1;
      let maxVal = values.length > 1 ? values[1] : 0;
      const upperBound = Math.min(values.length - 1, 9);
      for (let i = 2; i <= upperBound; i++) {
        if (values[i] > maxVal) { maxVal = values[i]; maxIdx = i; }
      }

      if (maxVal < CNN_LOW_CONFIDENCE) return { digit: 0, confidence: 0 };
      return { digit: maxIdx, confidence: maxVal };
    } finally {
      tensor.dispose();
    }
  }

  // ─────────────────────────────────────────────
  // 数独制約チェック: 行・列・ブロック内の重複を検出し、
  // 低信頼のセルを空白(0)に戻して手動修正を促す
  // ─────────────────────────────────────────────
  function _resolveConflicts(grid, confidenceMap) {
    const conflicts = [];

    // 行チェック
    for (let row = 0; row < 9; row++) {
      const seen = {};
      for (let col = 0; col < 9; col++) {
        const val = grid[row][col];
        if (val === 0) continue;
        if (seen[val] !== undefined) {
          conflicts.push([row, col, row, seen[val]]);
        }
        seen[val] = col;
      }
    }

    // 列チェック
    for (let col = 0; col < 9; col++) {
      const seen = {};
      for (let row = 0; row < 9; row++) {
        const val = grid[row][col];
        if (val === 0) continue;
        if (seen[val] !== undefined) {
          conflicts.push([row, col, seen[val], col]);
        }
        seen[val] = row;
      }
    }

    // 3×3 ブロックチェック
    for (let br = 0; br < 9; br += 3) {
      for (let bc = 0; bc < 9; bc += 3) {
        const seen = {};
        for (let r = br; r < br + 3; r++) {
          for (let c = bc; c < bc + 3; c++) {
            const val = grid[r][c];
            if (val === 0) continue;
            const key = `${val}`;
            if (seen[key]) {
              conflicts.push([r, c, seen[key][0], seen[key][1]]);
            }
            seen[key] = [r, c];
          }
        }
      }
    }

    // 重複があるセルのうち、信頼度が低い方を空白に戻す
    for (const [r1, c1, r2, c2] of conflicts) {
      const conf1 = confidenceMap[r1][c1]?.conf || 0;
      const conf2 = confidenceMap[r2][c2]?.conf || 0;

      if (conf1 < conf2) {
        console.log(`制約違反 → [${r1},${c1}]=${grid[r1][c1]}(conf:${conf1.toFixed(2)}) を空白に`);
        grid[r1][c1] = 0;
      } else {
        console.log(`制約違反 → [${r2},${c2}]=${grid[r2][c2]}(conf:${conf2.toFixed(2)}) を空白に`);
        grid[r2][c2] = 0;
      }
    }

    if (conflicts.length > 0) {
      console.log(`${conflicts.length}件の数独制約違反を検出し、低信頼セルを空白にしました`);
    }
  }

  // ─────────────────────────────────────────────
  // OCR ワーカー終了 (アプリ終了時呼び出し)
  // ─────────────────────────────────────────────
  async function terminate() {
    if (_ocrWorker) {
      await _ocrWorker.terminate();
      _ocrWorker = null;
      _ocrReady = false;
    }
  }

  return { loadModel, recognize, terminate };
})();
