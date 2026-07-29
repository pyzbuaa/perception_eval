import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 1300,
    rollupOptions: {
      output: {
        manualChunks: {
          'ui-vendor': ['antd', '@ant-design/icons'],
          charts: ['echarts', 'echarts-for-react'],
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:18080',
      '/artifacts': 'http://127.0.0.1:18080',
    },
  },
})
