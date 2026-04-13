/**
 * sudokuSolver.js
 * バックトラッキングアルゴリズムによる数独ソルバー。
 *
 * 入力: number[][] (9×9, 0 = 空白)
 * 出力: number[][] (9×9, 解答済み) または null (解なし)
 */

const SudokuSolver = (() => {

  // ─────────────────────────────────────────────
  // メインsolve関数 (入力配列を直接変更しないようコピーする)
  // ─────────────────────────────────────────────
  function solve(inputGrid) {
    if (!validate(inputGrid)) return null;

    // ディープコピー
    const grid = inputGrid.map(row => [...row]);
    const result = _backtrack(grid);
    return result ? grid : null;
  }

  // ─────────────────────────────────────────────
  // バックトラッキング
  // ─────────────────────────────────────────────
  function _backtrack(grid) {
    const cell = _findEmptyCell(grid);
    if (!cell) return true; // すべて埋まった

    const [row, col] = cell;

    for (let num = 1; num <= 9; num++) {
      if (_isValid(grid, row, col, num)) {
        grid[row][col] = num;
        if (_backtrack(grid)) return true;
        grid[row][col] = 0;
      }
    }

    return false; // バックトラック
  }

  // ─────────────────────────────────────────────
  // MRV ヒューリスティック: 候補数が最小の空白セルを選ぶ
  // (最小残余値 = Minimum Remaining Values)
  // ─────────────────────────────────────────────
  function _findEmptyCell(grid) {
    let minCandidates = 10;
    let bestCell = null;

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (grid[row][col] !== 0) continue;

        let count = 0;
        for (let num = 1; num <= 9; num++) {
          if (_isValid(grid, row, col, num)) count++;
        }

        if (count === 0) return null; // 矛盾 (解なし)
        if (count < minCandidates) {
          minCandidates = count;
          bestCell = [row, col];
          if (count === 1) return bestCell; // 即決定
        }
      }
    }

    return bestCell;
  }

  // ─────────────────────────────────────────────
  // 配置検証: 行・列・3×3ブロックで重複チェック
  // ─────────────────────────────────────────────
  function _isValid(grid, row, col, num) {
    // 行チェック
    if (grid[row].includes(num)) return false;

    // 列チェック
    for (let r = 0; r < 9; r++) {
      if (grid[r][col] === num) return false;
    }

    // 3×3 ブロックチェック
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    for (let r = boxRow; r < boxRow + 3; r++) {
      for (let c = boxCol; c < boxCol + 3; c++) {
        if (grid[r][c] === num) return false;
      }
    }

    return true;
  }

  // ─────────────────────────────────────────────
  // 入力グリッドのバリデーション
  // ─────────────────────────────────────────────
  function validate(grid) {
    if (!Array.isArray(grid) || grid.length !== 9) return false;
    for (let row = 0; row < 9; row++) {
      if (!Array.isArray(grid[row]) || grid[row].length !== 9) return false;
      for (let col = 0; col < 9; col++) {
        const val = grid[row][col];
        if (!Number.isInteger(val) || val < 0 || val > 9) return false;
      }
    }

    // 既存の数字の矛盾チェック
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        const val = grid[row][col];
        if (val === 0) continue;

        // 一時的に空白にして isValid チェック
        grid[row][col] = 0;
        const ok = _isValid(grid, row, col, val);
        grid[row][col] = val;
        if (!ok) return false;
      }
    }

    return true;
  }

  // ─────────────────────────────────────────────
  // 解の一意性チェック (最大2解まで探索)
  // ─────────────────────────────────────────────
  function countSolutions(inputGrid, max = 2) {
    const grid = inputGrid.map(row => [...row]);
    let count = 0;

    function bt() {
      if (count >= max) return;
      const cell = _findEmptyCell(grid);
      if (!cell) { count++; return; }

      const [row, col] = cell;
      for (let num = 1; num <= 9; num++) {
        if (_isValid(grid, row, col, num)) {
          grid[row][col] = num;
          bt();
          grid[row][col] = 0;
        }
      }
    }

    bt();
    return count;
  }

  return { solve, validate, countSolutions };
})();
