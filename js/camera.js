/**
 * camera.js
 * カメラ撮影・ギャラリー選択から画像を読み込み、
 * canvas-preview に描画して次のステップに渡す。
 */

const Camera = (() => {
  let _originalImage = null; // HTMLImageElement

  function init() {
    const btnCamera  = document.getElementById('btn-camera');
    const btnGallery = document.getElementById('btn-gallery');
    const inputCamera  = document.getElementById('input-camera');
    const inputGallery = document.getElementById('input-gallery');

    btnCamera.addEventListener('click', () => inputCamera.click());
    btnGallery.addEventListener('click', () => inputGallery.click());

    inputCamera.addEventListener('change',  (e) => _handleFile(e.target.files[0]));
    inputGallery.addEventListener('change', (e) => _handleFile(e.target.files[0]));
  }

  /**
   * File オブジェクトを受け取り、プレビューに表示する
   */
  function _handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('画像ファイルを選択してください。');
      return;
    }

    // ファイルサイズ上限 20MB
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

  /**
   * canvas-preview に画像を描画（最大幅 600px にリサイズ）
   */
  function _drawPreview(img) {
    const canvas = document.getElementById('canvas-preview');
    const MAX = 600;
    let w = img.naturalWidth;
    let h = img.naturalHeight;

    if (w > MAX) {
      h = Math.round(h * MAX / w);
      w = MAX;
    }
    if (h > MAX) {
      w = Math.round(w * MAX / h);
      h = MAX;
    }

    canvas.width  = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
  }

  /**
   * 外部から取得できるよう元画像を返す
   */
  function getOriginalImage() {
    return _originalImage;
  }

  /**
   * プレビュー canvas を ImageData として返す
   */
  function getPreviewCanvas() {
    return document.getElementById('canvas-preview');
  }

  return { init, getOriginalImage, getPreviewCanvas };
})();
