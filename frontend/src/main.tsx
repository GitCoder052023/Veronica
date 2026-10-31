import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// Settings saved before the VERONICA rename (onboarding, calibration, camera, last scene)
try {
  for (const k of Object.keys(localStorage)) {
    if (k.startsWith('stark.') && localStorage.getItem('veronica.' + k.slice(6)) === null) {
      localStorage.setItem('veronica.' + k.slice(6), localStorage.getItem(k)!)
    }
  }
} catch {
  /* storage unavailable */
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

if (import.meta.env.DEV) {
  // handy for debugging from the console: __veronica.engine.setView('iso')
  import('./store/studio').then(({ runtime }) => ((window as unknown as { __veronica: unknown }).__veronica = runtime))
}
