import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Route agent synthesis requests to Qwen on port 8082
      '/agent': {
        target: 'http://localhost:8082',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/agent/, '')
      },
      // Route evolution core requests to the AutoResearcher on port 8081
      '/autoresearch': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/autoresearch/, '')
      }
    }
  }
})