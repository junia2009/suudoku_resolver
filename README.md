# 数独解読アプリ — Sudoku Solver

カメラ撮影・ギャラリー読み込みから数独を自動解読するウェブアプリです。  
GitHub Pages で動作し、インストール不要でスマートフォン・PC から利用できます。

## 🔗 公開URL
`https://<あなたのGitHubユーザー名>.github.io/<リポジトリ名>/`

---

## 機能

| ステップ | 処理内容 |
|----------|----------|
| 1. 画像入力 | カメラ撮影 または ギャラリーから画像選択 |
| 2. 歪み補正 | OpenCV.js で数独グリッドを自動検出・射影変換 |
| 3. 数字認識 | TensorFlow.js (MNIST モデル) で 1〜9 を認識。誤認識はタップして修正可能 |
| 4. 解答表示 | バックトラッキングアルゴリズムで解を求めて表示 |

---

## 技術スタック

- **フロントエンド**: HTML / CSS / Vanilla JavaScript (モジュール構成)
- **画像処理**: [OpenCV.js](https://docs.opencv.org/4.8.0/opencv.js) — グレースケール・適応的閾値・輪郭検出・射影変換
- **数字認識**: [TensorFlow.js](https://www.tensorflow.org/js) + MNIST 転移学習モデル
- **ソルバー**: バックトラッキング + MRV ヒューリスティック
- **ホスティング**: GitHub Pages (静的サイト)

---

## ローカルで動かす

```bash
# Python がある場合
python -m http.server 8080
# または Node.js がある場合
npx serve .
```

ブラウザで `http://localhost:8080` を開いてください。  
**※ `file://` プロトコルでは OpenCV.js / TF.js が動作しません。必ずローカルサーバー経由で開いてください。**

---

## GitHub Pages へのデプロイ

1. このリポジトリを GitHub にプッシュします。
2. GitHub のリポジトリページ → **Settings** → **Pages**
3. **Source** を `Deploy from a branch` に設定し、ブランチ `main` / `/ (root)` を選択して **Save**。
4. 数分後に公開 URL が表示されます。

---

## ファイル構成

```
.
├── index.html              # メインページ
├── css/
│   └── style.css           # スタイルシート
├── js/
│   ├── app.js              # エントリポイント・状態管理
│   ├── camera.js           # カメラ/ギャラリー入力
│   ├── imageProcessor.js   # 歪み補正 (OpenCV.js)
│   ├── digitRecognizer.js  # 数字認識 (TF.js)
│   ├── sudokuSolver.js     # バックトラッキングソルバー
│   └── ui.js               # UI制御・グリッド描画
└── README.md
```

---

## 注意事項

- **HTTPS 必須**: カメラ API (`getUserMedia`) は HTTPS 環境のみで使用可能です。GitHub Pages は自動的に HTTPS で提供されます。
- **モデルキャッシュ**: 初回実行時に TF.js モデルをダウンロードし IndexedDB にキャッシュします。2回目以降はオフラインでも動作します。
- **グリッド検出精度**: 明るい場所での撮影、グリッドが画面の大部分を占めるように撮影すると精度が上がります。
- **ブラウザ対応**: Chrome / Safari (iOS 14.5+) / Firefox 最新版を推奨します。

---

## ライセンス

MIT License
