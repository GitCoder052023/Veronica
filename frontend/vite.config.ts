import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const API = process.env.VERONICA_API ?? 'http://127.0.0.1:8787'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: API, changeOrigin: true } } },
  preview: { proxy: { '/api': { target: API, changeOrigin: true } } },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
} as never)
