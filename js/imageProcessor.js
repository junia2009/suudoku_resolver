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
  const WARP_SIZE = 450; // 歪み補正後の正方形サイズ (px)
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
      cv.GaussianBlur(gray, blurred, new cv.Size(9, 9), 0);
      cv.adaptiveThreshold(
        blurred, thresh, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11, 2
      );

      // モルフォロジー演算でノイズ除去
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, kernel);
      kernel.delete();

      const corners = _findGridCorners(thresh, src.cols, src.rows);

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
   */
  function _sortCorners(approxMat) {
    const pts = [];
    for (let i = 0; i < 4; i++) {
      pts.push({ x: approxMat.data32S[i * 2], y: approxMat.data32S[i * 2 + 1] });
    }

    // sum が最小 → 左上, 最大 → 右下
    // diff が最小 → 右上, 最大 → 左下
    pts.sort((a, b) => (a.x + a.y) - (b.x + b.y));
    const tl = pts[0];
    const br = pts[3];
    const rest = [pts[1], pts[2]];
    rest.sort((a, b) => a.x - b.x);
    const bl = rest[0];
    const tr = rest[1];
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
  // ─────────────────────────────────────────────
  function _extractCells(warpedMat) {
    // グレースケール + 閾値処理
    const gray   = new cv.Mat();
    const thresh = new cv.Mat();
    cv.cvtColor(warpedMat, gray, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(
      gray, thresh, 255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY_INV,
      11, 2
    );

    const CS = Math.floor(CELL_SIZE);
    const cells = [];
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width  = CS;
    tmpCanvas.height = CS;
    const tmpCtx = tmpCanvas.getContext('2d');

    for (let row = 0; row < 9; row++) {
      cells[row] = [];
      for (let col = 0; col < 9; col++) {
        const x = col * CELL_SIZE;
        const y = row * CELL_SIZE;

        // ROI (margin: 10%)
        const margin = Math.floor(CS * 0.1);
        const roi = thresh.roi(new cv.Rect(
          Math.round(x) + margin,
          Math.round(y) + margin,
          CS - margin * 2,
          CS - margin * 2
        ));

        const roiCanvas = _matToCanvas(roi, CS - margin * 2, CS - margin * 2);
        tmpCtx.clearRect(0, 0, CS, CS);
        tmpCtx.drawImage(roiCanvas, margin, margin);
        cells[row][col] = tmpCtx.getImageData(0, 0, CS, CS);

        roi.delete();
      }
    }

    gray.delete();
    thresh.delete();
    return cells;
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
