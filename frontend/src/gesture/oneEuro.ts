/** One Euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cutoff
 *  rises with speed — heavy smoothing when the hand is still (kills jitter), little
 *  smoothing when it moves fast (kills lag). Timestamps are seconds. */
export class OneEuro {
  private x: number | null = null
  private dx = 0
  private t: number | null = null
  constructor(public minCutoff = 1.5, public beta = 3.0, public dCutoff = 1.0) {}

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff)
    return 1 / (1 + tau / dt)
  }

  filter(value: number, t: number): number {
    if (this.x === null || this.t === null) {
      this.x = value
      this.t = t
      return value
    }
    const dt = Math.max(1e-4, t - this.t)
    this.t = t
    const dxRaw = (value - this.x) / dt
    const ad = OneEuro.alpha(this.dCutoff, dt)
    this.dx = ad * dxRaw + (1 - ad) * this.dx
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx)
    const a = OneEuro.alpha(cutoff, dt)
    this.x = a * value + (1 - a) * this.x
    return this.x
  }

  reset() {
    this.x = null
    this.t = null
    this.dx = 0
  }
}

export class OneEuro2 {
  readonly x: OneEuro
  readonly y: OneEuro
  constructor(minCutoff = 1.5, beta = 3.0) {
    this.x = new OneEuro(minCutoff, beta)
    this.y = new OneEuro(minCutoff, beta)
  }
  filter(x: number, y: number, t: number): [number, number] {
    return [this.x.filter(x, t), this.y.filter(y, t)]
  }
  set(minCutoff: number, beta: number) {
    this.x.minCutoff = this.y.minCutoff = minCutoff
    this.x.beta = this.y.beta = beta
  }
  reset() {
    this.x.reset()
    this.y.reset()
  }
}
