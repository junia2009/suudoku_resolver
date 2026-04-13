/**
 * digitRecognizer.js
 * TensorFlow.js を使って各セルの ImageData から数字 (0-9) を認識する。
 *
 * 手法:
 *   - ピクセル輝度で空白判定（白ピクセルが少ない → 空白 = 0）
 *   - 非空白セルは MNIST 学習済みモデル (IndexedDB キャッシュ) で 1-9 を推論
 *   - モデルが利用できない場合は CNN 内蔵フォールバック（軽量 JS ヒューリスティック）
 *
 * NOTE: 外部モデルとして MNIST の TensorFlow.js 変換済みモデルを
 *   /model/ ディレクトリに配置することを推奨。
 *   ただし GitHub Pages 環境ではモデルファイルを同梱する必要があるため、
 *   本実装ではオンライン CDN モデルを使用し、IndexedDB にキャッシュする。
 */

const DigitRecognizer = (() => {
  let _model = null;
  let _modelLoading = false;

  // MNIST モデルの URL (TensorFlow Hub 互換形式)
  const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

  // 空白判定閾値 (白ピクセル率 — BINARY_INV後、数字部分が白)
  const EMPTY_THRESHOLD = 0.03;

  // ─────────────────────────────────────────────
  // モデルのロード (非同期)
  // ─────────────────────────────────────────────
  async function loadModel() {
    if (_model) return _model;
    if (_modelLoading) {
      // 既にロード中なら完了を待つ
      return new Promise((resolve) => {
        const check = setInterval(() => {
          if (_model || !_modelLoading) {
            clearInterval(check);
            resolve(_model);
          }
        }, 200);
      });
    }

    _modelLoading = true;
    try {
      // IndexedDB キャッシュを優先して高速ロード
      _model = await tf.loadLayersModel('indexeddb://sudoku-mnist-model');
      console.log('モデルをキャッシュからロードしました');
    } catch (_cacheErr) {
      try {
        _model = await tf.loadLayersModel(MODEL_URL);
        // キャッシュに保存
        await _model.save('indexeddb://sudoku-mnist-model');
        console.log('モデルをリモートからロードし、キャッシュしました');
      } catch (err) {
        console.warn('モデルのロードに失敗しました。ヒューリスティックを使用します:', err);
        _model = null;
      }
    } finally {
      _modelLoading = false;
    }
    return _model;
  }

  // ─────────────────────────────────────────────
  // 9×9 セル ImageData[][] → 数字配列 number[][]
  // ─────────────────────────────────────────────
  async function recognize(cells) {
    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));

    // モデルロードを試みる
    await loadModel();

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        const imageData = cells[row][col];
        if (_isEmpty(imageData)) {
          grid[row][col] = 0;
          continue;
        }

        if (_model) {
          grid[row][col] = await _predictWithModel(imageData);
        } else {
          grid[row][col] = _predictHeuristic(imageData);
        }
      }
    }

    return grid;
  }

  // ─────────────────────────────────────────────
  // 空白判定: BINARY_INV後の白ピクセル（数字部分）の割合で判断
  // マージン領域を除いた内側のみをチェックする
  // ─────────────────────────────────────────────
  function _isEmpty(imageData) {
    const data = imageData.data;
    const w = imageData.width;
    const h = imageData.height;
    // マージン領域（黒い余白）を除いた内側のみをチェック
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
    return (white / innerTotal) < EMPTY_THRESHOLD;
  }

  // ─────────────────────────────────────────────
  // セル画像の前処理: 数字を検出し28×28に中央配置
  // MNIST形式（20×20の数字を28×28の中央に配置）に合わせる
  // ─────────────────────────────────────────────
  function _centerDigit(imageData) {
    const { data, width, height } = imageData;

    // 白ピクセル（数字部分）のバウンディングボックスを検出
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

    // 元のImageDataをcanvasに描画
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

    // 28×28 出力canvas（MNIST規格: 数字を20×20領域に収め中央配置）
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
  // TF.js モデルで推論
  // ─────────────────────────────────────────────
  async function _predictWithModel(imageData) {
    // 数字を中央配置して28×28にリサイズ（MNIST形式に合わせる）
    const centered = _centerDigit(imageData);

    const tensor = tf.tidy(() => {
      // グレースケール化 & 正規化
      const imgTensor = tf.browser.fromPixels(centered, 1); // [28, 28, 1]
      const normalized = imgTensor.div(255.0);
      return normalized.expandDims(0); // [1, 28, 28, 1]
    });

    try {
      const prediction = _model.predict(tensor);
      const values     = await prediction.data();
      prediction.dispose();

      // 数独は1-9のみ使用するため、クラス0を除外して最大確率を探す
      let maxIdx = 1;
      let maxVal = values[1];
      for (let i = 2; i <= 9; i++) {
        if (values[i] > maxVal) { maxVal = values[i]; maxIdx = i; }
      }

      // 信頼度が低すぎる場合は空白として扱う
      if (maxVal < 0.1) return 0;

      return maxIdx;
    } finally {
      tensor.dispose();
    }
  }

  // ─────────────────────────────────────────────
  // フォールバック: 簡易ヒューリスティック
  // (モデルが使えない場合。精度は低い)
  // ─────────────────────────────────────────────
  function _predictHeuristic(imageData) {
    // 28×28 に縮小したピクセルパターンで垂直/水平の黒ピクセル分布を
    // 特徴量として利用し、最も近いテンプレートを選ぶ
    // 実用上はモデルを使うことを強く推奨する

    const { data, width, height } = imageData;
    const GRID = 7;
    const cellW = width  / GRID;
    const cellH = height / GRID;

    // GRID×GRID のヒートマップ作成
    const heatmap = new Float32Array(GRID * GRID);
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        let dark = 0, cnt = 0;
        for (let py = Math.round(gy * cellH); py < Math.round((gy + 1) * cellH); py++) {
          for (let px = Math.round(gx * cellW); px < Math.round((gx + 1) * cellW); px++) {
            const idx = (py * width + px) * 4;
            const br  = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            if (br < 128) dark++;
            cnt++;
          }
        }
        heatmap[gy * GRID + gx] = cnt > 0 ? dark / cnt : 0;
      }
    }

    // 黒ピクセルの垂直・水平分布から数字を推定
    const rowSums = new Float32Array(GRID);
    const colSums = new Float32Array(GRID);
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        rowSums[r] += heatmap[r * GRID + c];
        colSums[c] += heatmap[r * GRID + c];
      }
    }

    const totalDark = heatmap.reduce((a, b) => a + b, 0);
    const topHalf    = rowSums.slice(0, 3).reduce((a, b) => a + b, 0);
    const bottomHalf = rowSums.slice(4).reduce((a, b) => a + b, 0);
    const leftHalf   = colSums.slice(0, 3).reduce((a, b) => a + b, 0);
    const rightHalf  = colSums.slice(4).reduce((a, b) => a + b, 0);
    const center     = heatmap[3 * GRID + 3];

    // 非常に簡易な判定 (実装の参考程度)
    if (totalDark < 1.0) return 1;
    if (topHalf > bottomHalf * 1.5 && leftHalf > rightHalf) return 7;
    if (topHalf > bottomHalf * 1.5) return 1;
    if (center > 0.5) return 8;
    if (leftHalf > rightHalf * 1.3) return 4;
    if (totalDark > 5.0) return 8;
    return 5; // デフォルト
  }

  return { loadModel, recognize };
})();
