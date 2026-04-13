/**
 * imageProcessor.js
 * OpenCV.js を利用して数独グリッドを検出・射影変換（歪み補正）し、
 * 9×9 セル画像を切り出す。
 *
 * 処理フロー:
 *  1. グレースケール変換
 *  2. ガウシアンブラー
 *  3. 適応的閾値処理 (adaptiveThreshold)
 *  4. 輪郭検出 → 最大の四辺形を数独グリッドとして選択
 *  5. 射影変換 (getPerspectiveTransform + warpPerspective)
 *  6. 9×9 に分割してセル画像配列を返す
 *
 * 失敗した場合は null を返し、UI 側で手動コーナー選択に切り替える。
 */

const ImageProcessor = (() => {
  const WARP_SIZE = 540; // 歪み補正後の正方形サイズ (px) — 9×60で高解像度
  const CELL_SIZE = WARP_SIZE / 9;

  let _cvReady = false;
  let _corners = null;       // 手動選択コーナー [{x,y} × 4]
  let _manualMode = false;
  let _srcCanvas = null;     // 検出対象 canvas

  function onCvReady() { _cvReady = true; }

  // ─────────────────────────────────────────────
  // メイン: 自動検出 → 歪み補正
  // ─────────────────────────────────────────────
  /**
   * @param {HTMLCanvasElement} srcCanvas  入力画像 canvas
   * @returns {{ detected: HTMLCanvasElement, warped: HTMLCanvasElement, cells: ImageData[][] } | null}
   */
  function process(srcCanvas) {
    if (!_cvReady) {
      alert('OpenCV.js がまだ読み込まれていません。少し待ってください。');
      return null;
    }

    _srcCanvas = srcCanvas;
    _manualMode = false;
    _corners = null;

    let src, gray, blurred, thresh, result;
    try {
      src     = cv.imread(srcCanvas);
      gray    = new cv.Mat();
      blurred = new cv.Mat();
      thresh  = new cv.Mat();

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      // コントラスト正規化: CLAHE が使えれば使い、なければ通常のequalizeHist
      let equalized;
      try {
        const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
        equalized = new cv.Mat();
        clahe.apply(gray, equalized);
        clahe.delete();
      } catch (_claheErr) {
        equalized = new cv.Mat();
        cv.equalizeHist(gray, equalized);
      }

      cv.GaussianBlur(equalized, blurred, new cv.Size(9, 9), 0);

      // マルチ閾値戦略: 異なるパラメータで試行して最良のグリッド検出を行う
      const threshConfigs = [
        { blockSize: 11, C: 2 },
        { blockSize: 15, C: 3 },
        { blockSize: 21, C: 5 },
        { blockSize: 7,  C: 2 },
      ];

      let corners = null;
      for (const cfg of threshConfigs) {
        const t = new cv.Mat();
        cv.adaptiveThreshold(
          blurred, t, 255,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY_INV,
          cfg.blockSize, cfg.C
        );

        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(t, t, cv.MORPH_CLOSE, kernel);
        kernel.delete();

        corners = _findGridCorners(t, src.cols, src.rows);
        if (!thresh.rows) { t.copyTo(thresh); } // 最初の閾値結果を保持
        t.delete();
        if (corners) break;
      }

      equalized.delete();

      if (!corners) {
        // 自動検出失敗 → 手動モードへ
        _enterManualMode(srcCanvas);
        return null;
      }

      const detected = _drawDetectedBox(srcCanvas, corners);
      const warped   = _warpPerspective(src, corners);
      const cells    = _extractCells(warped);

      result = { detected, warped: _matToCanvas(warped, WARP_SIZE, WARP_SIZE), cells };
      warped.delete();
      return result;
    } catch (e) {
      console.error('ImageProcessor.process error:', e);
      return null;
    } finally {
      if (src)     src.delete();
      if (gray)    gray.delete();
      if (blurred) blurred.delete();
      if (thresh)  thresh.delete();
    }
  }

  // ─────────────────────────────────────────────
  // 最大四辺形の検出
  // ─────────────────────────────────────────────
  function _findGridCorners(threshMat, imgW, imgH) {
    const contours  = new cv.MatVector();
    const hierarchy = new cv.Mat();

    try {
      cv.findContours(threshMat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      let bestArea = 0;
      let bestCorners = null;
      const minArea = imgW * imgH * 0.1; // 画像面積の10%以上

      for (let i = 0; i < contours.size(); i++) {
        const cnt  = contours.get(i);
        const area = cv.contourArea(cnt);
        if (area < minArea) { cnt.delete(); continue; }

        const peri   = cv.arcLength(cnt, true);
        const approx = new cv.Mat();
        cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

        if (approx.rows === 4 && area > bestArea) {
          bestArea = area;
          bestCorners = _sortCorners(approx);
        }

        approx.delete();
        cnt.delete();
      }

      return bestCorners;
    } finally {
      contours.delete();
      hierarchy.delete();
    }
  }

  /**
   * 4点コーナーを [左上, 右上, 右下, 左下] の順に並べ替える
   * sum(x+y) と diff(x-y) を使う標準的な手法
   */
  function _sortCorners(approxMat) {
    const pts = [];
    for (let i = 0; i < 4; i++) {
      pts.push({ x: approxMat.data32S[i * 2], y: approxMat.data32S[i * 2 + 1] });
    }

    const sums  = pts.map(p => p.x + p.y);
    const diffs = pts.map(p => p.x - p.y);

    // sum が最小 → 左上, 最大 → 右下
    // diff が最大 → 右上 (x大, y小), 最小 → 左下 (x小, y大)
    const tl = pts[sums.indexOf(Math.min(...sums))];
    const br = pts[sums.indexOf(Math.max(...sums))];
    const tr = pts[diffs.indexOf(Math.max(...diffs))];
    const bl = pts[diffs.indexOf(Math.min(...diffs))];

    return [tl, tr, br, bl];
  }

  // ─────────────────────────────────────────────
  // 検出枠を元画像に描画した canvas を返す
  // ─────────────────────────────────────────────
  function _drawDetectedBox(srcCanvas, corners) {
    const out = document.getElementById('canvas-detected');
    out.width  = srcCanvas.width;
    out.height = srcCanvas.height;
    const ctx = out.getContext('2d');
    ctx.drawImage(srcCanvas, 0, 0);

    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth   = 3;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();

    corners.forEach((pt, idx) => {
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#000';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText(['TL','TR','BR','BL'][idx], pt.x + 8, pt.y - 4);
    });

    return out;
  }

  // ─────────────────────────────────────────────
  // 射影変換 → 450×450 の cv.Mat を返す
  // ─────────────────────────────────────────────
  function _warpPerspective(srcMat, corners) {
    const S = WARP_SIZE;
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0].x, corners[0].y,
      corners[1].x, corners[1].y,
      corners[2].x, corners[2].y,
      corners[3].x, corners[3].y,
    ]);
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, S, 0, S, S, 0, S,
    ]);

    const M      = cv.getPerspectiveTransform(srcPts, dstPts);
    const warped = new cv.Mat();
    const dsize  = new cv.Size(S, S);
    cv.warpPerspective(srcMat, warped, M, dsize);

    srcPts.delete();
    dstPts.delete();
    M.delete();
    return warped;
  }

  // ─────────────────────────────────────────────
  // 9×9 セル ImageData[][] を抽出する
  // Hough線検出 → グリッド交点からセル境界を精密決定
  // フォールバック: 均等分割
  // ─────────────────────────────────────────────
  function _extractCells(warpedMat) {
    const gray = new cv.Mat();
    cv.cvtColor(warpedMat, gray, cv.COLOR_RGBA2GRAY);

    // コントラスト正規化: CLAHE→equalizeHistフォールバック
    let grayEq;
    try {
      const clahe = new cv.CLAHE(2.0, new cv.Size(4, 4));
      grayEq = new cv.Mat();
      clahe.apply(gray, grayEq);
      clahe.delete();
    } catch (_claheErr) {
      grayEq = new cv.Mat();
      cv.equalizeHist(gray, grayEq);
    }

    // Hough線検出でグリッド線の正確な位置を特定
    const gridLines = _detectGridLines(grayEq);

    const CS = Math.floor(CELL_SIZE);
    const cells = [];
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width  = CS;
    tmpCanvas.height = CS;
    const tmpCtx = tmpCanvas.getContext('2d');

    for (let row = 0; row < 9; row++) {
      cells[row] = [];
      for (let col = 0; col < 9; col++) {
        let x, y, w, h;
        if (gridLines) {
          // Hough線検出結果からセル座標を取得
          x = gridLines.vLines[col];
          y = gridLines.hLines[row];
          w = gridLines.vLines[col + 1] - x;
          h = gridLines.hLines[row + 1] - y;
        } else {
          // フォールバック: 均等分割
          x = Math.round(col * CELL_SIZE);
          y = Math.round(row * CELL_SIZE);
          w = CS;
          h = CS;
        }

        // 範囲チェック
        x = Math.max(0, Math.min(x, grayEq.cols - 1));
        y = Math.max(0, Math.min(y, grayEq.rows - 1));
        w = Math.min(w, grayEq.cols - x);
        h = Math.min(h, grayEq.rows - y);
        if (w < 5 || h < 5) { w = CS; h = CS; x = col * CS; y = row * CS; }

        // セルのグレースケール ROI を取得
        const cellGray = grayEq.roi(new cv.Rect(x, y, w, h));

        // ガウシアンブラーで微小ノイズを軽減
        const blurred = new cv.Mat();
        cv.GaussianBlur(cellGray, blurred, new cv.Size(3, 3), 0);

        // セル単位で適応的閾値処理
        const cellThresh = new cv.Mat();
        let blockSize = Math.max(5, Math.round(Math.min(w, h) * 0.3));
        if (blockSize % 2 === 0) blockSize++;
        cv.adaptiveThreshold(
          blurred, cellThresh, 255,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY_INV,
          blockSize, 7
        );

        // グリッド線除去: 境界ピクセルをゼロ化
        const borderW = Math.max(3, Math.floor(Math.min(w, h) * 0.08));
        _clearCellBorder(cellThresh, borderW);

        // ノイズ除去: 小さな連結成分を除去
        _removeSmallComponents(cellThresh, Math.min(w, h));

        // 統一サイズ (CS×CS) にリサイズして ImageData に変換
        const resized = new cv.Mat();
        cv.resize(cellThresh, resized, new cv.Size(CS, CS));

        const roiCanvas = _matToCanvas(resized, CS, CS);
        tmpCtx.clearRect(0, 0, CS, CS);
        tmpCtx.drawImage(roiCanvas, 0, 0);
        cells[row][col] = tmpCtx.getImageData(0, 0, CS, CS);

        cellGray.delete();
        blurred.delete();
        cellThresh.delete();
        resized.delete();
      }
    }

    gray.delete();
    grayEq.delete();
    return cells;
  }

  // ─────────────────────────────────────────────
  // Hough線検出でグリッド線の位置を検出する
  // 水平線10本 + 垂直線10本（9×9グリッドの境界）を特定
  // ─────────────────────────────────────────────
  function _detectGridLines(grayMat) {
    try {
      const edges = new cv.Mat();
      cv.Canny(grayMat, edges, 50, 150);

      // モルフォロジーで線を強調
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      cv.dilate(edges, edges, kernel);
      kernel.delete();

      const lines = new cv.Mat();
      cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 80,
        Math.round(WARP_SIZE * 0.4), Math.round(WARP_SIZE * 0.05));

      const hLines = []; // 水平線のy座標
      const vLines = []; // 垂直線のx座標

      for (let i = 0; i < lines.rows; i++) {
        const x1 = lines.data32S[i * 4];
        const y1 = lines.data32S[i * 4 + 1];
        const x2 = lines.data32S[i * 4 + 2];
        const y2 = lines.data32S[i * 4 + 3];

        const angle = Math.abs(Math.atan2(y2 - y1, x2 - x1));
        if (angle < 0.15) {
          // ほぼ水平
          hLines.push(Math.round((y1 + y2) / 2));
        } else if (angle > Math.PI / 2 - 0.15) {
          // ほぼ垂直
          vLines.push(Math.round((x1 + x2) / 2));
        }
      }

      edges.delete();
      lines.delete();

      // クラスタリング: 近い線をグループ化して10本ずつに絞る
      const hClusters = _clusterLines(hLines, WARP_SIZE, 10);
      const vClusters = _clusterLines(vLines, WARP_SIZE, 10);

      if (hClusters.length !== 10 || vClusters.length !== 10) {
        console.log(`Hough線検出: 水平${hClusters.length}本, 垂直${vClusters.length}本 → 均等分割にフォールバック`);
        return null;
      }

      hClusters.sort((a, b) => a - b);
      vClusters.sort((a, b) => a - b);

      console.log('Hough線検出成功:', { hLines: hClusters, vLines: vClusters });
      return { hLines: hClusters, vLines: vClusters };
    } catch (e) {
      console.warn('Hough線検出失敗:', e);
      return null;
    }
  }

  /**
   * 近接する線座標をクラスタリングして expectedCount 本に絞る
   */
  function _clusterLines(lineCoords, totalSize, expectedCount) {
    if (lineCoords.length === 0) return [];

    lineCoords.sort((a, b) => a - b);
    const minGap = totalSize / (expectedCount * 3); // クラスタ間最小距離
    const clusters = [];
    let currentCluster = [lineCoords[0]];

    for (let i = 1; i < lineCoords.length; i++) {
      if (lineCoords[i] - lineCoords[i - 1] < minGap) {
        currentCluster.push(lineCoords[i]);
      } else {
        clusters.push(currentCluster);
        currentCluster = [lineCoords[i]];
      }
    }
    clusters.push(currentCluster);

    // 各クラスタの中央値を取る
    return clusters.map(c => {
      c.sort((a, b) => a - b);
      return c[Math.floor(c.length / 2)];
    });
  }

  /**
   * セル画像の境界領域をゼロ化してグリッド線と数字の接続を切断する
   */
  function _clearCellBorder(mat, borderWidth) {
    const h = mat.rows;
    const w = mat.cols;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (y < borderWidth || y >= h - borderWidth ||
            x < borderWidth || x >= w - borderWidth) {
          mat.ucharPtr(y, x)[0] = 0;
        }
      }
    }
  }

  /**
   * 小さな連結成分をノイズとして除去する
   */
  function _removeSmallComponents(mat, cellSize) {
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    const src = mat.clone();
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    src.delete();

    // セル面積の1%未満の成分はノイズとみなす（数字は通常セル面積の5%以上を占める）
    const minArea = cellSize * cellSize * 0.01;
    const result = cv.Mat.zeros(mat.rows, mat.cols, cv.CV_8UC1);

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area >= minArea) {
        cv.drawContours(result, contours, i, new cv.Scalar(255), cv.FILLED);
      }
      cnt.delete();
    }

    result.copyTo(mat);
    result.delete();
    contours.delete();
    hierarchy.delete();
  }

  // ─────────────────────────────────────────────
  // cv.Mat → HTMLCanvasElement
  // ─────────────────────────────────────────────
  function _matToCanvas(mat, w, h) {
    const c = document.createElement('canvas');
    c.width  = w || mat.cols;
    c.height = h || mat.rows;
    cv.imshow(c, mat);
    return c;
  }

  // ─────────────────────────────────────────────
  // 手動コーナー選択モード
  // ─────────────────────────────────────────────
  function _enterManualMode(srcCanvas) {
    _manualMode = true;
    _corners    = [];

    document.getElementById('detection-error').classList.remove('hidden');

    const detectedCanvas = document.getElementById('canvas-detected');
    detectedCanvas.width  = srcCanvas.width;
    detectedCanvas.height = srcCanvas.height;
    const ctx = detectedCanvas.getContext('2d');
    ctx.drawImage(srcCanvas, 0, 0);

    const labels = ['左上', '右上', '右下', '左下'];

    detectedCanvas.addEventListener('click', _manualClickHandler);
    detectedCanvas.style.cursor = 'crosshair';

    window._manualCtx    = ctx;
    window._manualLabels = labels;
    window._manualSrcCanvas = srcCanvas;
  }

  function _manualClickHandler(e) {
    const rect  = e.target.getBoundingClientRect();
    const scaleX = e.target.width  / rect.width;
    const scaleY = e.target.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top)  * scaleY;

    _corners.push({ x, y });

    const ctx    = window._manualCtx;
    const labels = window._manualLabels;

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(labels[_corners.length - 1], x + 8, y - 4);

    if (_corners.length === 4) {
      e.target.removeEventListener('click', _manualClickHandler);
      e.target.style.cursor = 'default';
      _finalizeManual();
    }
  }

  function _finalizeManual() {
    if (!_cvReady || !_srcCanvas) return;

    let srcMat;
    try {
      srcMat = cv.imread(_srcCanvas);
      const warped = _warpPerspective(srcMat, _corners);
      const cells  = _extractCells(warped);
      const warpedCanvas = _matToCanvas(warped, WARP_SIZE, WARP_SIZE);
      warped.delete();

      // イベントを発火して UI に通知
      const event = new CustomEvent('manualCornersComplete', {
        detail: { cells, warpedCanvas }
      });
      document.dispatchEvent(event);
    } finally {
      if (srcMat) srcMat.delete();
    }
  }

  /**
   * 手動コーナー再選択リセット
   */
  function resetManual() {
    _corners    = [];
    _manualMode = false;
    const canvas = document.getElementById('canvas-detected');
    canvas.removeEventListener('click', _manualClickHandler);
    canvas.style.cursor = 'default';
  }

  return { process, onCvReady, resetManual, get isManualMode() { return _manualMode; } };
})();
