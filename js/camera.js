/**
 * camera.js
 * getUserMedia API でブラウザ内カメラプレビュー + 撮影、
 * およびギャラリー選択から画像を読み込む。
 */

const Camera = (() => {
  let _originalImage = null;
  let _stream = null;

  function init() {
    const btnCamera  = document.getElementById('btn-camera');
    const btnGallery = document.getElementById('btn-gallery');
    const inputGallery = document.getElementById('input-gallery');
    const btnCapture = document.getElementById('btn-capture');
    const btnClose   = document.getElementById('btn-camera-close');

    btnCamera.addEventListener('click', _startCamera);
    btnGallery.addEventListener('click', () => inputGallery.click());
    inputGallery.addEventListener('change', (e) => _handleFile(e.target.files[0]));
    btnCapture.addEventListener('click', _capturePhoto);
    btnClose.addEventListener('click', _stopCamera);
  }

  // ── カメラ起動 ──
  async function _startCamera() {
    try {
      // 背面カメラ優先
      const constraints = {
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
        audio: false,
      };

      _stream = await navigator.mediaDevices.getUserMedia(constraints);
      const video = document.getElementById('camera-video');
      video.srcObject = _stream;

      // メタデータ読み込み後に表示
      video.onloadedmetadata = () => {
        video.play();
        document.getElementById('camera-preview-container').classList.remove('hidden');
        document.getElementById('preview-container').classList.add('hidden');
      };
    } catch (err) {
      console.error('Camera access failed:', err);
      if (err.name === 'NotAllowedError') {
        alert('カメラへのアクセスが拒否されました。\nブラウザの設定でカメラの使用を許可してください。');
      } else if (err.name === 'NotFoundError') {
        alert('カメラが見つかりませんでした。\nカメラが接続されているか確認してください。');
      } else {
        alert('カメラの起動に失敗しました: ' + err.message);
      }
    }
  }

  // ── 撮影 ──
  function _capturePhoto() {
    const video = document.getElementById('camera-video');
    if (!video.videoWidth || !video.videoHeight) {
      alert('カメラの映像がまだ準備できていません。');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // 撮影画像をImageに変換
    const img = new Image();
    img.onload = () => {
      _originalImage = img;
      _drawPreview(img);
      _stopCamera();
      document.getElementById('preview-container').classList.remove('hidden');
    };
    img.src = canvas.toDataURL('image/jpeg', 0.92);
  }

  // ── カメラ停止 ──
  function _stopCamera() {
    if (_stream) {
      _stream.getTracks().forEach(track => track.stop());
      _stream = null;
    }
    const video = document.getElementById('camera-video');
    video.srcObject = null;
    document.getElementById('camera-preview-container').classList.add('hidden');
  }

  // ── ファイル読み込み ──
  function _handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('画像ファイルを選択してください。');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      alert('ファイルサイズが大きすぎます（上限 20MB）');
      return;
    }

    UI.showLoading('画像を読み込み中...');
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        _originalImage = img;
        _drawPreview(img);
        UI.hideLoading();
        document.getElementById('preview-container').classList.remove('hidden');
      };
      img.onerror = () => {
        UI.hideLoading();
        alert('画像の読み込みに失敗しました。');
      };
      img.src = event.target.result;
    };
    reader.onerror = () => {
      UI.hideLoading();
      alert('ファイルの読み込みに失敗しました。');
    };
    reader.readAsDataURL(file);
  }

  // ── プレビュー描画 ──
  function _drawPreview(img) {
    const canvas = document.getElementById('canvas-preview');
    const MAX = 600;
    let w = img.naturalWidth;
    let h = img.naturalHeight;
    if (w > MAX) { h = Math.round(h * MAX / w); w = MAX; }
    if (h > MAX) { w = Math.round(w * MAX / h); h = MAX; }
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(img, 0, 0, w, h);
  }

  function getOriginalImage() { return _originalImage; }
  function getPreviewCanvas() { return document.getElementById('canvas-preview'); }
  function stopCamera() { _stopCamera(); }

  return { init, getOriginalImage, getPreviewCanvas, stopCamera };
})();
