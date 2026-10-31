export class Emitter<E extends Record<string, unknown>> {
  private handlers = new Map<keyof E, Set<(p: never) => void>>()
  on<K extends keyof E>(k: K, fn: (p: E[K]) => void): () => void {
    if (!this.handlers.has(k)) this.handlers.set(k, new Set())
    this.handlers.get(k)!.add(fn as (p: never) => void)
    return () => this.handlers.get(k)!.delete(fn as (p: never) => void)
  }
  emit<K extends keyof E>(k: K, p: E[K]) {
    this.handlers.get(k)?.forEach((fn) => (fn as (p: E[K]) => void)(p))
  }
}
