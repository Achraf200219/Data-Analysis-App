import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration for the Vanna React front‑end.
//
// We enable a proxy so that API requests to `/api` during development are
// forwarded to the FastAPI server running on port 8000.  This avoids CORS
// issues and allows us to call the backend as if it were served from the same
// origin.  Adjust the `target` if you run the backend on a different port.

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
      },
    },
  },
});