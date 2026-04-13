/**
 * ui.js
 * ステップ間のナビゲーション、グリッド描画、ローディング制御を管理する。
 */

const UI = (() => {
  let _currentStep = 1;

  // ─────────────────────────────────────────────
  // ステップ表示制御
  // ─────────────────────────────────────────────
  function showStep(n) {
    document.querySelectorAll('.step-section').forEach(el => {
      el.classList.remove('active');
      el.classList.add('hidden');
    });
    document.querySelectorAll('#step-indicator .step').forEach((el, idx) => {
      el.classList.remove('active', 'done');
      if (idx + 1 < n)  el.classList.add('done');
      if (idx + 1 === n) el.classList.add('active');
    });

    const target = document.getElementById(`step-${n}`);
    if (target) {
      target.classList.remove('hidden');
      target.classList.add('active');
    }
    _currentStep = n;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ─────────────────────────────────────────────
  // ローディング
  // ─────────────────────────────────────────────
  function showLoading(msg = '処理中...') {
    document.getElementById('loading-msg').textContent = msg;
    document.getElementById('loading-overlay').classList.remove('hidden');
  }

  function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
  }

  // ─────────────────────────────────────────────
  // 数独グリッド描画 (入力確認用)
  // ─────────────────────────────────────────────
  /**
   * @param {string}   containerId  描画先コンテナの ID
   * @param {number[][]} grid       9×9 数字配列
   * @param {boolean}  editable     セルを編集可能にするか
   * @param {number[][]} givenMask  元々の数字 (1=given, 0=solved) ※任意
   */
  function renderGrid(containerId, grid, editable = false, givenMask = null) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.classList.add('sudoku-grid');

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        const cell = document.createElement('div');
        cell.classList.add('sudoku-cell');
        cell.dataset.row = row;
        cell.dataset.col = col;

        const val = grid[row][col];
        const isGiven = givenMask ? givenMask[row][col] === 1 : val !== 0;

        if (isGiven) {
          cell.classList.add('given');
          cell.textContent = val || '';
        } else if (val !== 0) {
          cell.classList.add('solved');
          cell.textContent = val;
        } else {
          cell.classList.add('empty');
          cell.textContent = '';
        }

        if (editable) {
          _makeEditable(cell, grid, row, col);
        }

        container.appendChild(cell);
      }
    }
  }

  /**
   * セルを編集可能にする（タップ→数字選択 or キーボード入力）
   */
  function _makeEditable(cellEl, grid, row, col) {
    cellEl.addEventListener('click', () => {
      // 既存の選択を外す
      document.querySelectorAll('.sudoku-cell.selected').forEach(el => el.classList.remove('selected'));
      cellEl.classList.add('selected');

      // input を表示
      const current = grid[row][col];
      const input = document.createElement('input');
      input.type    = 'text';
      input.inputMode = 'numeric';
      input.pattern = '[0-9]';
      input.maxLength = 1;
      input.value = current !== 0 ? String(current) : '';
      cellEl.textContent = '';
      cellEl.appendChild(input);
      input.focus();
      input.select();

      const commit = () => {
        const v = parseInt(input.value, 10);
        const newVal = (isNaN(v) || v < 0 || v > 9) ? 0 : v;
        grid[row][col] = newVal;
        cellEl.removeChild(input);
        cellEl.classList.remove('given', 'solved', 'empty', 'selected');

        if (newVal !== 0) {
          cellEl.classList.add('given');
          cellEl.textContent = String(newVal);
        } else {
          cellEl.classList.add('empty');
          cellEl.textContent = '';
        }
      };

      input.addEventListener('blur',    commit);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === 'Tab') {
          e.preventDefault();
          commit();
        }
        if (e.key === 'Escape') {
          input.value = current !== 0 ? String(current) : '';
          commit();
        }
      });
      // モバイルでの数字以外の入力を除去
      input.addEventListener('input', () => {
        input.value = input.value.replace(/[^0-9]/g, '').slice(0, 1);
      });
    });
  }

  // ─────────────────────────────────────────────
  // 現在のグリッド値を DOM から読み取る
  // ─────────────────────────────────────────────
  function readGridFromDom(containerId) {
    const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
    const cells = document.querySelectorAll(`#${containerId} .sudoku-cell`);
    cells.forEach(cell => {
      const row = parseInt(cell.dataset.row, 10);
      const col = parseInt(cell.dataset.col, 10);
      const text = cell.textContent.trim();
      const v    = parseInt(text, 10);
      grid[row][col] = isNaN(v) ? 0 : v;
    });
    return grid;
  }

  // ─────────────────────────────────────────────
  // 解答表示用グリッド (given/solved を色分け)
  // ─────────────────────────────────────────────
  function renderSolution(containerId, originalGrid, solvedGrid) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.classList.add('sudoku-grid');

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        const cell = document.createElement('div');
        cell.classList.add('sudoku-cell');
        cell.dataset.row = row;
        cell.dataset.col = col;

        const orig   = originalGrid[row][col];
        const solved = solvedGrid[row][col];

        if (orig !== 0) {
          cell.classList.add('given');
        } else {
          cell.classList.add('solved');
        }
        cell.textContent = solved !== 0 ? String(solved) : '';
        container.appendChild(cell);
      }
    }
  }

  // ─────────────────────────────────────────────
  // 解答結果メッセージ
  // ─────────────────────────────────────────────
  function showSolveResult(success, msg) {
    const el = document.getElementById('solve-result');
    el.textContent = msg;
    el.className   = success ? 'success' : 'error';
  }

  return { showStep, showLoading, hideLoading, renderGrid, readGridFromDom, renderSolution, showSolveResult };
})();
