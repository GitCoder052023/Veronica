// Serve MediaPipe's wasm runtime from our own origin (no CDN needed at runtime).
import { cpSync, existsSync, mkdirSync } from 'node:fs'
const src = 'node_modules/@mediapipe/tasks-vision/wasm'
if (existsSync(src)) {
  mkdirSync('public/mediapipe', { recursive: true })
  cpSync(src, 'public/mediapipe/wasm', { recursive: true })
  console.log('copied MediaPipe wasm → public/mediapipe/wasm')
}
