import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

const proxy = { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, timeout: 0, proxyTimeout: 0 } };

export default defineConfig({
  plugins: [react()],
  server: { proxy, port: 5173, strictPort: true },
  preview: { proxy },
  test: { environment: 'jsdom', setupFiles: './src/test/setup.ts', globals: true, clearMocks: true },
});
