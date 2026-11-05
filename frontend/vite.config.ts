import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// WEB_UI_PLAN.md section 6.
//
// The bundle is built *into the Python package* and committed, so `clipper ui`
// needs Python and nothing else. Two consequences are encoded below.
//
// `rollupOptions.output` names every chunk without a content hash. Normally a
// hash is what makes a bundle cacheable; here the bundle is served by Flask
// with `max_age=0` and read from a git working tree, so the hash buys nothing
// and costs a renamed pair of files in every commit that touches the frontend.
// Stable names mean `git diff` shows what changed rather than one file
// appearing and another vanishing.
//
// The dev proxy is what keeps CORS out of this project entirely: `npm run dev`
// on :5173 forwards the three server-owned prefixes to the Flask process on
// :5151, so the browser only ever sees one origin and `api.ts` can use
// relative URLs in both modes.
const BACKEND = "http://127.0.0.1:5151"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  build: {
    outDir: path.resolve(import.meta.dirname, "../src/clipper/webui/dist"),
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: false },
      "/media": { target: BACKEND, changeOrigin: false },
      "/healthz": { target: BACKEND, changeOrigin: false },
    },
  },
})
