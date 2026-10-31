import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision'
import type { RawHand } from './GestureEngine'

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const WASM_PATH = '/mediapipe/wasm'

export interface TrackerStats {
  fps: number
  inferenceMs: number
  delegate: 'GPU' | 'CPU'
  resolution: string
}

/** Webcam + MediaPipe HandLandmarker. Runs detection once per *new* video frame
 *  (requestVideoFrameCallback), so work is paced by the camera, not the display. */
export class HandTracker {
  readonly video: HTMLVideoElement
  private landmarker: HandLandmarker | null = null
  private stream: MediaStream | null = null
  private running = false
  private lastVideoTime = -1
  private fpsEma = 0
  private infEma = 0
  private lastFrame = 0
  delegate: 'GPU' | 'CPU' = 'GPU'
  onResult: ((hands: RawHand[], tMs: number) => void) | null = null
  onStats: ((s: TrackerStats) => void) | null = null
  lastResult: HandLandmarkerResult | null = null

  constructor() {
    this.video = document.createElement('video')
    this.video.playsInline = true
    this.video.muted = true
  }

  static async devices(): Promise<MediaDeviceInfo[]> {
    return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
  }

  private async createLandmarker(delegate: 'GPU' | 'CPU') {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH)
    return HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.5,
    })
  }

  async start(deviceId?: string) {
    if (this.running) return
    if (!this.landmarker) {
      try {
        this.landmarker = await this.createLandmarker('GPU')
        this.delegate = 'GPU'
      } catch (e) {
        console.warn('GPU delegate unavailable, falling back to CPU', e)
        this.landmarker = await this.createLandmarker('CPU')
        this.delegate = 'CPU'
      }
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 60 } },
      audio: false,
    })
    this.video.srcObject = this.stream
    await this.video.play()
    this.running = true
    this.schedule()
  }

  private schedule() {
    if (!this.running) return
    const v = this.video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => this.step())
    else requestAnimationFrame(() => this.step())
  }

  private step() {
    if (!this.running || !this.landmarker) return
    if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = this.video.currentTime
      const now = performance.now()
      const t0 = performance.now()
      const res = this.landmarker.detectForVideo(this.video, now)
      const inf = performance.now() - t0
      this.lastResult = res
      this.infEma = this.infEma ? this.infEma * 0.9 + inf * 0.1 : inf
      if (this.lastFrame) this.fpsEma = this.fpsEma ? this.fpsEma * 0.9 + (1000 / (now - this.lastFrame)) * 0.1 : 1000 / (now - this.lastFrame)
      this.lastFrame = now
      const hands: RawHand[] = res.landmarks.map((lm, i) => ({
        handedness: res.handedness[i]?.[0]?.categoryName ?? `Hand${i}`,
        score: res.handedness[i]?.[0]?.score ?? 0,
        image: lm,
        world: res.worldLandmarks[i] ?? lm,
      }))
      this.onResult?.(hands, now)
      this.onStats?.({ fps: Math.round(this.fpsEma), inferenceMs: +this.infEma.toFixed(1), delegate: this.delegate, resolution: `${this.video.videoWidth}×${this.video.videoHeight}` })
    }
    this.schedule()
  }

  stop() {
    this.running = false
    this.stream?.getTracks().forEach((t) => t.stop())
    this.stream = null
    this.video.srcObject = null
  }

  dispose() {
    this.stop()
    this.landmarker?.close()
    this.landmarker = null
  }

  get isRunning() {
    return this.running
  }
}
