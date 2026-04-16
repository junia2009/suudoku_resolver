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
  // 制約伝播 → バックトラッキングの2段階で解く
  // ─────────────────────────────────────────────
  function solve(inputGrid) {
    if (!validate(inputGrid)) return null;

    // ディープコピー
    const grid = inputGrid.map(row => [...row]);

    // 制約伝播で確定できるセルを先に埋める
    if (!_propagate(grid)) return null;

    const result = _backtrack(grid);
    return result ? grid : null;
  }

  // ─────────────────────────────────────────────
  // 解法ステップ記録付きsolve
  // 戻り値: { solved: number[][], steps: Step[] } | null
  // Step: { row, col, num, reason, detail }
  // ─────────────────────────────────────────────
  function solveWithSteps(inputGrid) {
    if (!validate(inputGrid)) return null;

    const grid = inputGrid.map(row => [...row]);
    const steps = [];

    // 制約伝播 (ステップ記録付き)
    if (!_propagateWithSteps(grid, steps)) return null;

    // バックトラッキング (ステップ記録付き)
    const btSteps = [];
    const result = _backtrackWithSteps(grid, btSteps);
    if (!result) return null;

    // バックトラッキングで確定した手順を追加
    btSteps.forEach(s => steps.push(s));

    return { solved: grid, steps };
  }

  // ─────────────────────────────────────────────
  // 制約伝播 (ステップ記録付き)
  // ─────────────────────────────────────────────
  function _propagateWithSteps(grid, steps) {
    let changed = true;
    while (changed) {
      changed = false;

      // Naked Singles
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
          if (grid[row][col] !== 0) continue;

          const candidates = [];
          for (let num = 1; num <= 9; num++) {
            if (_isValid(grid, row, col, num)) candidates.push(num);
          }

          if (candidates.length === 0) return false;
          if (candidates.length === 1) {
            grid[row][col] = candidates[0];
            steps.push({
              row, col, num: candidates[0],
              reason: 'naked_single',
              detail: `R${row+1}C${col+1}: この位置に入れられる数字は ${candidates[0]} のみ`
            });
            changed = true;
          }
        }
      }

      // Hidden Singles
      let hiddenFound = false;
      for (let num = 1; num <= 9 && !hiddenFound; num++) {
        // 行
        for (let row = 0; row < 9 && !hiddenFound; row++) {
          if (grid[row].includes(num)) continue;
          let onlyCol = -1, count = 0;
          for (let col = 0; col < 9; col++) {
            if (grid[row][col] === 0 && _isValid(grid, row, col, num)) {
              onlyCol = col; count++;
              if (count > 1) break;
            }
          }
          if (count === 1) {
            grid[row][onlyCol] = num;
            steps.push({
              row, col: onlyCol, num,
              reason: 'hidden_single_row',
              detail: `R${row+1}C${onlyCol+1}: ${row+1}行目で ${num} を置ける場所はここだけ`
            });
            changed = true;
            hiddenFound = true;
          }
        }

        // 列
        for (let col = 0; col < 9 && !hiddenFound; col++) {
          let hasNum = false;
          for (let r = 0; r < 9; r++) { if (grid[r][col] === num) { hasNum = true; break; } }
          if (hasNum) continue;
          let onlyRow = -1, count = 0;
          for (let row = 0; row < 9; row++) {
            if (grid[row][col] === 0 && _isValid(grid, row, col, num)) {
              onlyRow = row; count++;
              if (count > 1) break;
            }
          }
          if (count === 1) {
            grid[onlyRow][col] = num;
            steps.push({
              row: onlyRow, col, num,
              reason: 'hidden_single_col',
              detail: `R${onlyRow+1}C${col+1}: ${col+1}列目で ${num} を置ける場所はここだけ`
            });
            changed = true;
            hiddenFound = true;
          }
        }

        // 3×3 ブロック
        for (let br = 0; br < 9 && !hiddenFound; br += 3) {
          for (let bc = 0; bc < 9 && !hiddenFound; bc += 3) {
            let hasNum = false;
            for (let r = br; r < br + 3 && !hasNum; r++) {
              for (let c = bc; c < bc + 3 && !hasNum; c++) {
                if (grid[r][c] === num) hasNum = true;
              }
            }
            if (hasNum) continue;
            let onlyR = -1, onlyC = -1, count = 0;
            for (let r = br; r < br + 3; r++) {
              for (let c = bc; c < bc + 3; c++) {
                if (grid[r][c] === 0 && _isValid(grid, r, c, num)) {
                  onlyR = r; onlyC = c; count++;
                  if (count > 1) break;
                }
              }
              if (count > 1) break;
            }
            if (count === 1) {
              grid[onlyR][onlyC] = num;
              const blockNum = Math.floor(br / 3) * 3 + Math.floor(bc / 3) + 1;
              steps.push({
                row: onlyR, col: onlyC, num,
                reason: 'hidden_single_box',
                detail: `R${onlyR+1}C${onlyC+1}: ブロック${blockNum}で ${num} を置ける場所はここだけ`
              });
              changed = true;
              hiddenFound = true;
            }
          }
        }
      }
    }
    return true;
  }

  // ─────────────────────────────────────────────
  // バックトラッキング (ステップ記録付き)
  // ─────────────────────────────────────────────
  function _backtrackWithSteps(grid, steps) {
    const cell = _findEmptyCell(grid);
    if (cell === null) return true;
    if (cell === false) return false;

    const [row, col] = cell;

    for (let num = 1; num <= 9; num++) {
      if (_isValid(grid, row, col, num)) {
        grid[row][col] = num;

        // バックトラッキング前に制約伝播で埋められるか試す
        const snapshot = grid.map(r => [...r]);
        const propSteps = [];
        const propOk = _propagateWithSteps(grid, propSteps);

        if (propOk && _backtrackWithSteps(grid, [])) {
          // 成功 → 仮定と制約伝播ステップを記録
          steps.push({
            row, col, num,
            reason: 'backtrack',
            detail: `R${row+1}C${col+1}: 候補から ${num} を仮定`
          });
          propSteps.forEach(s => steps.push(s));
          return true;
        }

        // 失敗 → 復元
        for (let r = 0; r < 9; r++) {
          for (let c = 0; c < 9; c++) {
            grid[r][c] = snapshot[r][c];
          }
        }
      }
    }

    return false;
  }

  // ─────────────────────────────────────────────
  // 制約伝播: Naked Singles + Hidden Singles を繰り返し適用
  // バックトラッキング前に確定セルを埋めて探索空間を縮小する
  // ─────────────────────────────────────────────
  function _propagate(grid) {
    let changed = true;
    while (changed) {
      changed = false;

      // Naked Singles: 候補が1つしかないセルを確定
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
          if (grid[row][col] !== 0) continue;

          const candidates = [];
          for (let num = 1; num <= 9; num++) {
            if (_isValid(grid, row, col, num)) candidates.push(num);
          }

          if (candidates.length === 0) return false; // 矛盾
          if (candidates.length === 1) {
            grid[row][col] = candidates[0];
            changed = true;
          }
        }
      }

      // Hidden Singles: 行・列・ブロック内で特定の数字が
      // 1箇所にしか入れないセルを確定
      // 1つ確定したら即座にwhile loopを再開して整合性を保つ
      let hiddenFound = false;
      for (let num = 1; num <= 9 && !hiddenFound; num++) {
        // 行
        for (let row = 0; row < 9 && !hiddenFound; row++) {
          if (grid[row].includes(num)) continue;
          let onlyCol = -1;
          let count = 0;
          for (let col = 0; col < 9; col++) {
            if (grid[row][col] === 0 && _isValid(grid, row, col, num)) {
              onlyCol = col;
              count++;
              if (count > 1) break;
            }
          }
          if (count === 1) {
            grid[row][onlyCol] = num;
            changed = true;
            hiddenFound = true;
          }
          // count === 0 は矛盾だが、Naked Singlesで検出済みなのでスキップ
        }

        // 列
        for (let col = 0; col < 9 && !hiddenFound; col++) {
          let hasNum = false;
          for (let r = 0; r < 9; r++) { if (grid[r][col] === num) { hasNum = true; break; } }
          if (hasNum) continue;
          let onlyRow = -1;
          let count = 0;
          for (let row = 0; row < 9; row++) {
            if (grid[row][col] === 0 && _isValid(grid, row, col, num)) {
              onlyRow = row;
              count++;
              if (count > 1) break;
            }
          }
          if (count === 1) {
            grid[onlyRow][col] = num;
            changed = true;
            hiddenFound = true;
          }
        }

        // 3×3 ブロック
        for (let br = 0; br < 9 && !hiddenFound; br += 3) {
          for (let bc = 0; bc < 9 && !hiddenFound; bc += 3) {
            let hasNum = false;
            for (let r = br; r < br + 3 && !hasNum; r++) {
              for (let c = bc; c < bc + 3 && !hasNum; c++) {
                if (grid[r][c] === num) hasNum = true;
              }
            }
            if (hasNum) continue;
            let onlyR = -1, onlyC = -1;
            let count = 0;
            for (let r = br; r < br + 3; r++) {
              for (let c = bc; c < bc + 3; c++) {
                if (grid[r][c] === 0 && _isValid(grid, r, c, num)) {
                  onlyR = r;
                  onlyC = c;
                  count++;
                  if (count > 1) break;
                }
              }
              if (count > 1) break;
            }
            if (count === 1) {
              grid[onlyR][onlyC] = num;
              changed = true;
              hiddenFound = true;
            }
          }
        }
      }
    }
    return true;
  }

  // ─────────────────────────────────────────────
  // バックトラッキング
  // ─────────────────────────────────────────────
  function _backtrack(grid) {
    const cell = _findEmptyCell(grid);
    if (cell === null) return true; // すべて埋まった (成功)
    if (cell === false) return false; // 矛盾 (候補0のセルあり)

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
  // 戻り値: [row, col] | null (全セル確定) | false (矛盾)
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

        if (count === 0) return false; // 矛盾 (解なし)
        if (count < minCandidates) {
          minCandidates = count;
          bestCell = [row, col];
          if (count === 1) return bestCell; // 即決定
        }
      }
    }

    return bestCell; // null = 全セル埋まった
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
      if (cell === null) { count++; return; }
      if (cell === false) return; // 矛盾

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

  return { solve, solveWithSteps, validate, countSolutions };
})();
