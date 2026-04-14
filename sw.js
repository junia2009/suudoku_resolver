/**
 * Service Worker — ナンプレ絶対解くマン
 * Cache-first + network fallback 戦略
 */

const CACHE_NAME = 'numplace-solver-v1';

const PRECACHE_URLS = [
  './',
  './index.html',
  './css/style.css',
  './js/app.js',
  './js/camera.js',
  './js/digitRecognizer.js',
  './js/imageProcessor.js',
  './js/sudokuSolver.js',
  './js/ui.js',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './manifest.json',
];

// CDN リソース（大きいのでネットワーク優先、キャッシュにフォールバック）
const CDN_URLS = [
  'https://docs.opencv.org/4.8.0/opencv.js',
  'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js',
];

// install: ローカルファイルをプリキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// activate: 古いキャッシュを削除
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// fetch: ローカルはキャッシュ優先、CDNはネットワーク優先
self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // CDN: network-first
  if (CDN_URLS.some((cdn) => url.startsWith(cdn))) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // ローカル: cache-first
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        // 成功したレスポンスをキャッシュに追加
        if (response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
