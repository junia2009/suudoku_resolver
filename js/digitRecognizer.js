/**
 * digitRecognizer.js
 * TensorFlow.js を使って各セルの ImageData から数字 (0-9) を認識する。
 *
 * 手法:
 *   - ピクセル輝度で空白判定（黒ピクセルが少ない → 空白 = 0）
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

  // 空白判定閾値 (黒ピクセル率)
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
  // 空白判定: 閾値処理後の黒ピクセル率で判断
  // ─────────────────────────────────────────────
  function _isEmpty(imageData) {
    const data   = imageData.data;
    const total  = data.length / 4;
    let   dark   = 0;

    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      if (brightness < 128) dark++;
    }

    return (dark / total) < EMPTY_THRESHOLD;
  }

  // ─────────────────────────────────────────────
  // TF.js モデルで推論
  // ─────────────────────────────────────────────
  async function _predictWithModel(imageData) {
    const tensor = tf.tidy(() => {
      // グレースケール化 & 28×28 リサイズ & 正規化
      const imgTensor = tf.browser.fromPixels(imageData, 1); // [H, W, 1]
      const resized   = tf.image.resizeBilinear(imgTensor, [28, 28]);
      const normalized = resized.div(255.0);
      return normalized.expandDims(0); // [1, 28, 28, 1]
    });

    try {
      const prediction = _model.predict(tensor);
      const values     = await prediction.data();
      prediction.dispose();

      // 最大確率のインデックス (0-9) を返す
      let maxIdx = 0;
      let maxVal = values[0];
      for (let i = 1; i < values.length; i++) {
        if (values[i] > maxVal) { maxVal = values[i]; maxIdx = i; }
      }
      // 0 クラスは空白扱いなので 1-9 を返す
      return maxIdx === 0 ? 1 : maxIdx;
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
