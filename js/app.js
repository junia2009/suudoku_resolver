/**
 * app.js
 * アプリケーションのエントリポイント。
 * 各モジュールを統合し、ステップ間のデータフローを管理する。
 */

// ─────────────────────────────────────────────
// グローバル状態
// ─────────────────────────────────────────────
const AppState = {
  previewCanvas:  null,   // ステップ1で取得した canvas
  cells:          null,   // 9×9 ImageData[][] (補正後の二値化セル)
  cellsGray:      null,   // 9×9 ImageData[][] (グレースケールセル, OCR用)
  warpedCanvas:   null,   // 補正後プレビュー canvas
  recognizedGrid: null,   // 認識結果 number[][]
  editedGrid:     null,   // ユーザー編集後 number[][]
  solvedGrid:     null,   // 解答 number[][]
};

// ─────────────────────────────────────────────
// OpenCV.js ロード完了コールバック
// ─────────────────────────────────────────────
function onOpenCvReady() {
  console.log('OpenCV.js が読み込まれました');
  ImageProcessor.onCvReady();
}

function onOpenCvError() {
  console.error('OpenCV.js の読み込みに失敗しました');
  alert('OpenCV.js の読み込みに失敗しました。ネットワーク接続を確認してください。\n\n補正機能が利用できませんが、手動コーナー選択で続行できます。');
}

// ─────────────────────────────────────────────
// 初期化
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  Camera.init();
  _bindStepButtons();
  _bindManualCornersEvent();
  UI.showStep(1);

  // TF.js モデルをバックグラウンドでプリロード
  DigitRecognizer.loadModel().catch(err => {
    console.warn('モデルプリロード失敗 (後ほど再試行します):', err);
  });
});

// ─────────────────────────────────────────────
// ボタンバインド
// ─────────────────────────────────────────────
function _bindStepButtons() {

  // ── ステップ1 → ステップ2 ──
  document.getElementById('btn-next-1').addEventListener('click', async () => {
    const previewCanvas = Camera.getPreviewCanvas();
    if (!previewCanvas || previewCanvas.width === 0) {
      alert('画像を選択してください。');
      return;
    }

    UI.showLoading('グリッドを検出中...');
    // 非同期処理を次フレームに遅延してローディングを表示させる
    await _nextFrame();

    try {
      const result = ImageProcessor.process(previewCanvas);
      if (result) {
        // 自動検出成功
        AppState.cells        = result.cells;
        AppState.cellsGray    = result.cellsGray;
        AppState.warpedCanvas = result.warped;
        // warped canvas を step-2 に表示
        _showWarpedCanvas(result.warped);
        document.getElementById('detection-error').classList.add('hidden');
        UI.hideLoading();
        UI.showStep(2);
      } else if (ImageProcessor.isManualMode) {
        // 手動コーナー選択中 → ロード非表示でステップ2へ
        UI.hideLoading();
        UI.showStep(2);
      } else {
        UI.hideLoading();
        alert('グリッドの検出に失敗しました。別の画像を試してください。');
      }
    } catch (e) {
      UI.hideLoading();
      console.error(e);
      alert('処理中にエラーが発生しました: ' + e.message);
    }
  });

  // ── ステップ2 → ステップ1 ──
  document.getElementById('btn-back-2').addEventListener('click', () => {
    ImageProcessor.resetManual();
    document.getElementById('detection-error').classList.add('hidden');
    UI.showStep(1);
  });

  // ── ステップ2 → ステップ3 (数字認識) ──
  document.getElementById('btn-next-2').addEventListener('click', async () => {
    if (!AppState.cells) {
      alert('補正画像がありません。グリッドを検出してください。');
      return;
    }

    UI.showLoading('数字を認識中...');
    await _nextFrame();

    try {
      const grid = await DigitRecognizer.recognize(AppState.cells, AppState.cellsGray, AppState.warpedCanvas);
      AppState.recognizedGrid = grid;
      AppState.editedGrid     = grid.map(r => [...r]);

      // 認識結果のサニティチェック
      const givenCount = grid.flat().filter(v => v !== 0).length;
      if (givenCount < 10) {
        console.warn(`認識されたヒント数が少なすぎます: ${givenCount}個`);
      } else if (givenCount > 40) {
        console.warn(`認識されたヒント数が多すぎます: ${givenCount}個`);
      }

      // 認識結果の given マスク (0以外 = given)
      UI.renderGrid('sudoku-input-grid', AppState.editedGrid, true);
      UI.hideLoading();

      // ヒント数警告
      if (givenCount < 10) {
        alert(`認識されたヒント数が${givenCount}個と少なめです。\n画像が鮮明か確認し、必要に応じて手動で修正してください。`);
      }

      UI.showStep(3);
    } catch (e) {
      UI.hideLoading();
      console.error(e);
      alert('数字認識中にエラーが発生しました: ' + e.message);
    }
  });

  // ── ステップ3 → ステップ2 ──
  document.getElementById('btn-back-3').addEventListener('click', () => {
    UI.showStep(2);
  });

  // ── ステップ3 → ステップ4 (解く) ──
  document.getElementById('btn-solve').addEventListener('click', async () => {
    // DOM から最新の編集済みグリッドを読み取る
    const currentGrid = UI.readGridFromDom('sudoku-input-grid');
    AppState.editedGrid = currentGrid;

    if (!SudokuSolver.validate(currentGrid)) {
      alert('入力に矛盾があります。重複している数字がないか確認してください。');
      return;
    }

    UI.showLoading('解いています...');
    await _nextFrame();

    try {
      const solved = SudokuSolver.solve(currentGrid);

      if (solved) {
        AppState.solvedGrid = solved;
        UI.renderSolution('sudoku-solution-grid', currentGrid, solved);
        UI.showSolveResult(true, '✓ 解答が見つかりました！');
      } else {
        AppState.solvedGrid = null;
        UI.showSolveResult(false, '× 解答が見つかりませんでした。入力値を確認してください。');
      }

      UI.hideLoading();
      UI.showStep(4);
    } catch (e) {
      UI.hideLoading();
      console.error(e);
      alert('解析中にエラーが発生しました: ' + e.message);
    }
  });

  // ── ステップ4 → ステップ3 ──
  document.getElementById('btn-back-4').addEventListener('click', () => {
    UI.showStep(3);
  });

  // ── 最初からやり直す ──
  document.getElementById('btn-restart').addEventListener('click', () => {
    Object.keys(AppState).forEach(k => { AppState[k] = null; });
    // カメラ停止 & 入力要素リセット
    Camera.stopCamera();
    document.getElementById('input-gallery').value = '';
    document.getElementById('camera-preview-container').classList.add('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('detection-error').classList.add('hidden');
    ImageProcessor.resetManual();
    UI.showStep(1);
  });
}

// ─────────────────────────────────────────────
// 手動コーナー選択完了イベント
// ─────────────────────────────────────────────
function _bindManualCornersEvent() {
  document.addEventListener('manualCornersComplete', (e) => {
    const { cells, cellsGray, warpedCanvas } = e.detail;
    AppState.cells        = cells;
    AppState.cellsGray    = cellsGray;
    AppState.warpedCanvas = warpedCanvas;
    _showWarpedCanvas(warpedCanvas);
    document.getElementById('detection-error').classList.add('hidden');
    alert('コーナー選択完了！「次へ」ボタンで数字認識に進んでください。');
  });
}

// ─────────────────────────────────────────────
// 補正後画像を step-2 の canvas に表示
// ─────────────────────────────────────────────
function _showWarpedCanvas(warpedCanvas) {
  const target = document.getElementById('canvas-warped');
  target.width  = warpedCanvas.width;
  target.height = warpedCanvas.height;
  target.getContext('2d').drawImage(warpedCanvas, 0, 0);
}

// ─────────────────────────────────────────────
// 次フレームまで待機 (UI 更新を確実にするための非同期遅延)
// ─────────────────────────────────────────────
function _nextFrame() {
  return new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
}
